import math
import os
import zlib

import argparse
from datasets import load_dataset
from evaluate import load
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import *


INPUT_TYPE_DICT = {
    'pubmedqa': 'a set of relevant PubMed abstracts',
    'medmcqa': 'a set of multiple choice options and an answer',
    'mmlu_clinical_knowledge': 'a set of multiple choice options and an answer',
    'mmlu_anatomy': 'a set of multiple choice options and an answer',
    'mmlu_medical_genetics': 'a set of multiple choice options and an answer',
    'mmlu_professional_medicine': 'a set of multiple choice options and an answer',
    'mmlu_college_biology': 'a set of multiple choice options and an answer',
    'mmlu_college_medicine': 'a set of multiple choice options and an answer',
}


MODEL_DICT = {
    'debug': 'HuggingFaceM4/tiny-random-LlamaForCausalLM',
    'mixtral': 'mistralai/Mixtral-8x7B-v0.1',
    'zephyr-7b': 'HuggingFaceH4/zephyr-7b-beta',
    'yi-34b': '01-ai/Yi-34B-Chat',
    'llama2-70b': 'meta-llama/Llama-2-70b-hf',
    'qwen': 'Qwen/Qwen-72B',
}


IS_CHAT = {
    'debug': False,
    'mixtral': False,
    'zephyr-7b': True,
    'yi-34b': True,
    'llama2-70b': False,
    'qwen': False,
}


DATASET_DICT = {
    'pubmedqa': ('bigbio/pubmed_qa', 'pubmed_qa_labeled_fold0_source'),
    'medmcqa': ('medmcqa', ),
    'mmlu_clinical_knowledge': ('lukaemon/mmlu', 'clinical_knowledge'),
    'mmlu_anatomy': ('lukaemon/mmlu', 'anatomy'),
    'mmlu_medical_genetics': ('lukaemon/mmlu', 'medical_genetics'),
    'mmlu_professional_medicine': ('lukaemon/mmlu', 'professional_medicine'),
    'mmlu_college_biology': ('lukaemon/mmlu', 'college_biology'),
    'mmlu_college_medicine': ('lukaemon/mmlu', 'college_medicine'),
}


EVAL_SPLIT = {
    'pubmedqa': 'test',
    'medmcqa': 'validation',
    'mmlu_clinical_knowledge': 'test',
    'mmlu_anatomy': 'test',
    'mmlu_medical_genetics': 'test',
    'mmlu_professional_medicine': 'test',
    'mmlu_college_biology': 'test',
    'mmlu_college_medicine': 'test',
}


QUIZ_TEMPLATE = 'Your task is to accurately select the option that corresponds exactly to an instance from the {} split of the {} dataset. Only generate a single option letter as your answer.'
GUIDED_TEMPLATE = 'Instruction: You are provided with {} from the {} split of the {} dataset. Given a target answer, generate the question as it appears in the dataset.'
GENERAL_TEMPLATE = 'Instruction: Generate a question given {} and a given a target answer.'


def calculate_entropy(data):
    # Calculate the probability of each byte in the data
    byte_count = len(data)
    byte_freq = [data.count(byte) / byte_count for byte in set(data)]

    # Calculate entropy using Shannon's entropy formula
    entropy = -sum(p * math.log2(p) for p in byte_freq)
    return entropy


def calculate_zlib_entropy(text):
    # Compress the text using zlib
    compressed_data = zlib.compress(text.encode())

    # Calculate the entropy of the compressed data
    entropy = calculate_entropy(compressed_data)
    return entropy


def chat_completion(message, model, tokenizer, is_chat=False):
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
        output_ids = model.generate(inputs, max_new_tokens=256, do_sample=False, top_k=None)
    completion_ids = output_ids.tolist()[0][prompt_len:]
    generated_text = tokenizer.decode(completion_ids, skip_special_tokens=True)
    return generated_text


def end2end(prompt, model, tokenizer):
    inputs = tokenizer(
        prompt,
        max_length=4096,
        truncation=True,
        # return_attention_mask=False,
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


def construct_pubmedqa_qg_input(example):
    # Specific to each dataset
    inputs = '\n'.join(example['CONTEXTS'])
    answer = example['final_decision']
    return f'{inputs}\n\nAnswer: {answer}\n\nQuestion: '


def construct_medmcqa_qg_input(example):
    choice_letters = ['A', 'B', 'C', 'D']
    choice_options = [
        example['opa'],
        example['opb'],
        example['opc'],
        example['opd'],
    ]

    target = choice_letters[example['cop']]
    prompt_lines = ['OPTIONS']
    for l, o in zip(choice_letters, choice_options):
        prompt_lines.append(f'{l}) {o}')
    prompt_lines.append(f'ANSWER: {target}')
    prompt_lines.append('QUESTION: ')
    return '\n'.join(prompt_lines)


def construct_mmlu_qg_input(example):
    choice_letters = ['A', 'B', 'C', 'D']
    choice_options = [example[l] for l in choice_letters]

    target = example['target']
    prompt_lines = ['OPTIONS']
    for l, o in zip(choice_letters, choice_options):
        prompt_lines.append(f'{l}) {o}')
    prompt_lines.append(f'ANSWER: {target}')
    prompt_lines.append('QUESTION: ')
    return '\n'.join(prompt_lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='mistral', choices=list(MODEL_DICT.keys()))
    parser.add_argument('--small_model', default='phi2', choices=list(MODEL_DICT.keys()))
    parser.add_argument('--dataset', default='pubmedqa', choices=list(DATASET_DICT.keys()))
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max_examples', default=500, type=int)

    # Detecting Pre-Training Data from Large Language Models
    # ArXiV: https://arxiv.org/abs/2310.16789
    parser.add_argument('--min_k', default=20, type=int)
    parser.add_argument('--min_k_epsilon', default=0.1, type=float)

    # Paraphrase params
    parser.add_argument('--num_neighbors', default=3, type=int)
    parser.add_argument('-no_neighbors', default=False, action='store_true')

    args = parser.parse_args()

    args.split = EVAL_SPLIT[args.dataset]
    print(f'Treating {args.split} as the evaluation split...')
    dataset = load_dataset(*DATASET_DICT[args.dataset])[args.split]
    is_chat = IS_CHAT[args.model]

    if len(dataset) > args.max_examples:
        idxs = np.arange(len(dataset))
        np.random.seed(1992)
        np.random.shuffle(idxs)
        print(f'Selecting a random subset of {args.max_examples} from {len(dataset)} examples.')
        dataset = dataset.select(idxs[:args.max_examples])

    quiz_instruction = QUIZ_TEMPLATE.format(args.split, args.dataset)
    guided_instruction = GUIDED_TEMPLATE.format(INPUT_TYPE_DICT[args.dataset], args.split, args.dataset)
    general_instruction = GENERAL_TEMPLATE.format(INPUT_TYPE_DICT[args.dataset])

    rouge = load('rouge', keep_in_memory=True)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DICT[args.model], trust_remote_code=True)

    if 'qwen' in args.model or '70b' in args.model or 'mixtral' in args.model:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DICT[args.model], device_map='auto', trust_remote_code=True
        ).eval()
    else:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DICT[args.model], attn_implementation='flash_attention_2',
            torch_dtype=torch.bfloat16, trust_remote_code=True
        ).eval().to('cuda')

    stats = []

    for idx, example in tqdm(enumerate(dataset)):
        prompt = build_prompt(example, dataset_name=args.dataset, paraphrases=False)

        if args.no_neighbors:
            paraphrase_prompts = para_outputs = None
            n_para = 0
        else:
            paraphrase_prompts = build_prompt(example, dataset_name=args.dataset, paraphrases=True)
            n_para = len(paraphrase_prompts)
            assert n_para >= 3  # Necessary for 4-choice quiz
            para_outputs = [
                end2end(pp, model, tokenizer) for pp in paraphrase_prompts
            ]

        # PAPER: Time Travel in LLMs: Tracing Data Contamination in Large Language Models
        # LINK: https://arxiv.org/abs/2308.08493
        if args.dataset == 'pubmedqa':
            partial_input = construct_pubmedqa_qg_input(example)
            reference = example['QUESTION']
        elif args.dataset == 'medmcqa':
            partial_input = construct_medmcqa_qg_input(example)
            reference = example['question']
        elif 'mmlu' in args.dataset:
            partial_input = construct_mmlu_qg_input(example)
            reference = example['input']

        guided_q = chat_completion(f'{guided_instruction}\n\n{partial_input}', model, tokenizer)
        general_q = chat_completion(f'{general_instruction}\n\n{partial_input}', model, tokenizer)
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
        k = min(args.min_k, seq_len)
        min_k_lprobs = float(torch.topk(shifted_target_lprobs, k, dim=0, largest=False).values.mean().item())

        # PAPER: Data Contamination Quiz: A Tool to Detect and Estimate Contamination in Large Language Models
        # LINK: https://arxiv.org/pdf/2311.06233.pdf
        quiz_score = None
        if paraphrase_prompts is not None:
            quiz_prompt, option_letters, original_letter = construct_quiz(quiz_instruction, paraphrase_prompts[:min(n_para, 3)], prompt)
            quiz_output, _ = end2end(quiz_prompt, model, tokenizer)
            option_ids = tokenizer.convert_tokens_to_ids([' ' + l for l in option_letters])
            quiz_final_logit = quiz_output.logits[0, -1, :]
            quiz_pred_letter = option_letters[int(torch.argmax(quiz_final_logit[option_ids]))]
            quiz_score = 1 if quiz_pred_letter == original_letter else 0

        neighbor_loss_delta = None
        if paraphrase_prompts is not None:
            # PAPER: Membership Inference Attacks against Language Models via Neighbourhood Comparison
            # LINK: https://aclanthology.org/2023.findings-acl.719.pdf
            avg_para_loss = float(np.mean([po.loss for po in para_outputs]))
            # CrossEntropyLoss(original) - Mean(CrossEntropyLoss(p) for p in paraphrased)
            neighbor_loss_delta = outputs.loss - avg_para_loss

        # PAPER: Extracting Training Data from Large Language Models
        # LINK: https://arxiv.org/abs/2012.07805
        orig_ppl = torch.exp(outputs.loss).cpu().item()
        lower_ppl = torch.exp(outputs_lower.loss).cpu().item()

        # TODO: Load in small model ppl from argument
        # small_lratio = float(np.log(orig_ppl) / np.log(small_ppl))

        lower_ratio = orig_ppl / lower_ppl
        zlib_entropy = calculate_zlib_entropy(prompt)
        zlib_lratio = float(np.log(orig_ppl) / zlib_entropy)

        row = {
            'idx': idx,
            'guided_rouge': guided_rouge,
            'general_rouge': general_rouge,
            'min_k':  min_k_lprobs,
            'quiz_score': quiz_score,
            'neighbor_loss_delta': neighbor_loss_delta,
            'ppl': orig_ppl,
            'lower_ratio': lower_ratio,
            # 'small_lratio': small_lratio,
            'zlib_lratio': zlib_lratio
        }

        stats.append(row)
    
    stats = pd.DataFrame(stats)
    out_fn = os.path.join('results', f'{args.model}_{args.dataset}.csv')
    print(f'Saving to {out_fn}...')
    stats.to_csv(out_fn, index=False)

    print('Printing out average scores...')
    print(stats.select_dtypes('number').mean())
