import argparse
import json
import math
import os
import zlib

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from evaluate import load
from Levenshtein import ratio
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs import DATASET_CONFIGS, MODEL_CONFIGS
from templates import *


def load_model(m_config):
    if 'qwen' in m_config.name or '70b' in m_config.name or 'mixtral' in m_config.name:
        # These need to be distributed across multiple GPUs
        model = AutoModelForCausalLM.from_pretrained(
            m_config.huggingface_path, device_map='auto', trust_remote_code=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            m_config.huggingface_path, attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to('cuda')
    return model.eval()


def chat_completion(message, model, tokenizer, max_new_tokens=256, is_chat=False):
    messages = [
        {'role': 'user', 'content': message},
    ]

    if is_chat:
        encodeds = tokenizer.apply_chat_template(messages, return_tensors='pt')
        inputs = encodeds.to(model.device)
        prompt_len = inputs.shape[1]
    else:
        inputs = tokenizer(message, return_tensors='pt')['input_ids']
        inputs = inputs.to(model.device)
        prompt_len = inputs.shape[1]

    with torch.no_grad():
        output_ids = model.generate(inputs, max_new_tokens=max_new_tokens, do_sample=False, top_k=None)
    completion_ids = output_ids.tolist()[0][prompt_len:]
    generated_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return generated_text


def end2end(prompt, model, tokenizer):
    inputs = tokenizer(
        prompt,
        max_length=4096,
        truncation=True,
        return_tensors='pt'
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    inputs['labels'] = inputs['input_ids']  # Will be shifted over inside AutoModelForCausalLM subclass

    with torch.no_grad():
        outputs = model(**inputs)
    return outputs, inputs['labels']


def construct_quiz(instruction, paras, original):
    options = paras.copy()

    rand_idx = np.random.randint(0, len(paras) + 1)
    options.insert(rand_idx, original)
    option_letters = ['A', 'B', 'C', 'D']
    original_letter = option_letters[rand_idx]
    
    option_str = '\n--\n'.join(
        f'Letter Choice {a})\n{b}' for a, b in zip(option_letters, options)
    )

    prompt = f'{instruction}\n--\n{option_str}\n--\nLetter Choice corresponding to an instance from the test set:'
    return prompt, option_letters, original_letter


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='mistral', choices=list(MODEL_CONFIGS.keys()))
    parser.add_argument('--dataset', default='pubmedqa', choices=list(DATASET_CONFIGS.keys()))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('-overwrite', default=False, action='store_true')
    parser.add_argument('--max_examples', default=100, type=int)

    # Detecting Pre-Training Data from Large Language Models
    # ArXiV: https://arxiv.org/abs/2310.16789
    parser.add_argument('--min_k', default=.2, type=float)
    parser.add_argument('--min_k_epsilon', default=0.1, type=float)

    # Paraphrase params
    parser.add_argument('--num_neighbors', default=3, type=int)
    parser.add_argument('-no_neighbors', default=False, action='store_true')

    args = parser.parse_args()

    d_config = DATASET_CONFIGS[args.dataset]
    m_config = MODEL_CONFIGS[args.model]

    print(f'Treating {d_config.eval_split} as the evaluation split...')
    dataset = load_dataset(*d_config.huggingface_path)[d_config.eval_split]
    n = len(dataset)

    if n > args.max_examples:
        idxs = np.arange(len(dataset))
        np.random.seed(1992)
        np.random.shuffle(idxs)
        print(f'Selecting a random subset of {args.max_examples} from {n} examples.')
        dataset = dataset.select(idxs[:args.max_examples])
        n = len(dataset)

    quiz_instruction = QUIZ_TEMPLATE.format(d_config.eval_split, d_config.name)
    guided_instruction = GUIDED_TEMPLATE.format(d_config.input_description, d_config.eval_split, d_config.name)
    general_instruction = GENERAL_TEMPLATE.format(d_config.input_description)

    rouge = load('rouge', keep_in_memory=True)

    tokenizer = AutoTokenizer.from_pretrained(m_config.huggingface_path, trust_remote_code=True)
    model = load_model(m_config)

    stats = []

    out_dir = os.path.join('results', f'{m_config.name}_{d_config.name}')
    os.makedirs(out_dir, exist_ok=True)
    out_fn = f'{out_dir}.csv'

    for idx, example in tqdm(enumerate(dataset), total=n):
        prompt = d_config.prompt_generator(example, dataset_name=d_config.name)

        ex_fn = os.path.join(out_dir, f'{idx}.json')
        neighbor_fn = os.path.join('results', 'neighbors', f'{d_config.name}_{args.split}', f'{idx}.json')

        if os.path.exists(ex_fn) and not args.overwrite:
            print(f'Reading in datapoint from {ex_fn}')
            with open(ex_fn, 'r') as fd:
                row = json.load(fd)
        else:
            if args.no_neighbors or not os.path.exists(neighbor_fn):
                paraphrase_prompts = para_outputs = None
                n_para = 0
            else:
                with open(neighbor_fn, 'r') as fd:
                    paraphrases = json.load(fd)

                paraphrase_prompts = d_config.paraphrase_prompt_generator(
                    example, dataset_name=d_config.name, paraphrases=paraphrases
                )

                if paraphrase_prompts is None:
                    print(paraphrases)
                    print('Invalid paraphrases...')
                    paraphrase_prompts = para_outputs = None
                    n_para = 0
                else:
                    n_para = len(paraphrase_prompts)
                    assert n_para >= 3  # Necessary for 4-choice quiz
                    para_outputs = [
                        end2end(pp, model, tokenizer) for pp in paraphrase_prompts
                    ]

            # PAPER: Time Travel in LLMs: Tracing Data Contamination in Large Language Models
            # LINK: https://arxiv.org/abs/2308.08493
            partial_input = d_config.qg_prompt_generator(example)
            reference = example[d_config.question_col]

            q_toks = reference.split(' ')
            mid = len(q_toks) // 2
            remaining = len(q_toks) - mid
            first_half = ' '.join(q_toks[:mid])
            second_half = ' '.join(q_toks[mid:])
            completion_prompt = f'Finish the following question using {mid} tokens.\nQuestion: {first_half}'
            
            completed = chat_completion(completion_prompt, model, tokenizer, max_new_tokens=64, is_chat=m_config.is_chat)
            completed = ' '.join(completed.split(' ')[:remaining])
            lev_ratio = ratio(second_half, completed)

            guided_q = chat_completion(f'{guided_instruction}\n\n{partial_input}', model, tokenizer, is_chat=m_config.is_chat)
            general_q = chat_completion(f'{general_instruction}\n\n{partial_input}', model, tokenizer, is_chat=m_config.is_chat)
            guided_rouge, general_rouge  = rouge.compute(
                predictions=[guided_q, general_q], references=[reference, reference],
                use_aggregator=False, rouge_types=['rougeL']
            )['rougeL']

            outputs, labels = end2end(prompt, model, tokenizer)
            outputs_lower, labels_lower = end2end(prompt.lower(), model, tokenizer)

            # Truncate last token logit
            shifted_logits = outputs.logits[0, :-1]
            # Shift labels 1 to right
            shifted_lprobs = torch.log_softmax(shifted_logits, dim=-1)
            shifted_labels = labels[0, 1:]
            # Compute token-level logprobs
            shifted_target_lprobs = torch.gather(shifted_lprobs, 1, shifted_labels.view(-1, 1)).squeeze(-1)
            seq_len = len(shifted_target_lprobs)

            # Paper: Detecting Pretraining Data from Large Language Models
            # LINK: https://arxiv.org/abs/2310.16789
            k = math.ceil(args.min_k * seq_len)
            min_k_lprobs = float(torch.topk(shifted_target_lprobs, k, dim=0, largest=False).values.mean().item())

            # PAPER: Data Contamination Quiz: A Tool to Detect and Estimate Contamination in Large Language Models
            # LINK: https://arxiv.org/pdf/2311.06233.pdf
            quiz_score = None
            if paraphrase_prompts is not None:
                quiz_prompt, option_letters, original_letter = construct_quiz(quiz_instruction, paraphrase_prompts[:min(n_para, 3)], prompt)
                quiz_output, _ = end2end(quiz_prompt, model, tokenizer)
                if 'qwen' in m_config.name.lower():
                    option_ids = []
                    for l in option_letters:
                        option_ids += tokenizer.encode(l)
                else:
                    option_ids = tokenizer.convert_tokens_to_ids(option_letters)
                quiz_final_logit = quiz_output.logits[0, -1, :]
                preds = quiz_final_logit[option_ids]
                assert min(preds) < max(preds)
                quiz_pred_letter = option_letters[int(torch.argmax(preds))]
                quiz_score = 1 if quiz_pred_letter == original_letter else 0

            neighbor_loss_delta = None
            if paraphrase_prompts is not None:
                # PAPER: Membership Inference Attacks against Language Models via Neighbourhood Comparison
                # LINK: https://aclanthology.org/2023.findings-acl.719.pdf
                avg_para_loss = float(np.mean([po[0].loss.cpu().item() for po in para_outputs]))
                # CrossEntropyLoss(original) - Mean(CrossEntropyLoss(p) for p in paraphrased)
                neighbor_loss_delta = float((outputs.loss - avg_para_loss).item())

            # PAPER: Extracting Training Data from Large Language Models
            # LINK: https://arxiv.org/abs/2012.07805
            orig_ppl = torch.exp(outputs.loss).cpu().item()
            lower_ppl = torch.exp(outputs_lower.loss).cpu().item()

            lower_ratio = np.log(lower_ppl) / np.log(orig_ppl)
            zlib_entropy = len(zlib.compress(bytes(prompt, 'utf-8')))
            zlib_lratio = float(zlib_entropy / np.log(orig_ppl))

            row = {
                'idx': idx,
                'guided_rouge': guided_rouge,
                'general_rouge': general_rouge,
                'min_k':  min_k_lprobs,
                'quiz_score': quiz_score,
                'neighbor_loss_delta': neighbor_loss_delta,
                'ppl': orig_ppl,
                'lower_ratio': lower_ratio,
                'zlib_lratio': zlib_lratio,
                'lev_ratio': lev_ratio
            }

            with open(ex_fn, 'w') as fd:
                json.dump(row, fd)

        stats.append(row)
    
    stats = pd.DataFrame(stats)
    print(f'Saving to {out_fn}...')
    stats.to_csv(out_fn, index=False)

    print('Printing out average scores...')
    print(stats.select_dtypes('number').mean())
