import math
import os
import zlib

import argparse
from datasets import load_dataset
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import *


INPUT_TYPE_DICT = {
    'pubmedqa': 'a set of relevant PubMed abstracts'
}


MODEL_DICT = {
    'debug': 'HuggingFaceM4/tiny-random-LlamaForCausalLM',
    'phi2': 'microsoft/phi-2',
}


DATASET_DICT = {
    'pubmedqa': ('bigbio/pubmed_qa', 'pubmed_qa_labeled_fold0_source')
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


def generate(prompts, model, tokenizer):
    inputs = tokenizer(
        prompts,
        max_length=4096,
        padding='longest',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model.generate(**inputs)
    # TODO decode
    return outputs.sequences


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


def construct_qg_prompt(instruction, example):
    # Specific to each dataset
    inputs = '\n'.join(example['CONTEXTS'])
    answer = example['final_decision']
    return f'{instruction}\n\n{inputs}\n\nAnswer: {answer}\n\nQuestion: '


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='phi2', choices=list(MODEL_DICT.keys()))
    parser.add_argument('--small_model', default='phi2', choices=list(MODEL_DICT.keys()))
    parser.add_argument('--dataset', default='pubmedqa', choices=list(DATASET_DICT.keys()))
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--split', default='test')

    # Detecting Pre-Training Data from Large Language Models
    # ArXiV: https://arxiv.org/abs/2310.16789
    parser.add_argument('--min_k', default=20, type=int)
    parser.add_argument('--min_k_epsilon', default=0.1, type=float)

    # Paraphrase params
    parser.add_argument('--num_neighbors', default=3, type=int)

    args = parser.parse_args()

    dataset = load_dataset(*DATASET_DICT[args.dataset])[args.split]
    quiz_instruction = QUIZ_TEMPLATE.format(args.split, args.dataset)
    guided_instruction = GUIDED_TEMPLATE.format(INPUT_TYPE_DICT[args.dataset], args.split, args.dataset)
    general_instruction = GENERAL_TEMPLATE.format(INPUT_TYPE_DICT[args.dataset])

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DICT[args.model], trust_remote_code=True)
    #  flash_attn=True,
    model = AutoModelForCausalLM.from_pretrained(MODEL_DICT[args.model], torch_dtype='auto', trust_remote_code=True).eval().to(args.device)

    small_tokenizer = AutoTokenizer.from_pretrained(MODEL_DICT[args.small_model], trust_remote_code=True)
    #  flash_attn=True,
    small_model = AutoModelForCausalLM.from_pretrained(MODEL_DICT[args.small_model], torch_dtype='auto', trust_remote_code=True).eval().to(args.device)

    stats = []

    for idx, example in tqdm(enumerate(dataset)):
        prompt = build_prompt(example, dataset_name=args.dataset, paraphrases=False)
        paraphrase_prompts = build_prompt(example, dataset_name=args.dataset, paraphrases=True)
        n_para = len(paraphrase_prompts)
        assert n_para >= 3  # Necessary for 4-choice quiz

        outputs, labels = end2end(prompt, model, tokenizer)
        outputs_lower, labels_lower = end2end(prompt.lower(), model, tokenizer)
        outputs_small, labels_small = end2end(prompt, small_model, small_tokenizer)

        para_outputs = [
            end2end(pp, model, tokenizer) for pp in paraphrase_prompts
        ]

        # PAPER: Time Travel in LLMs: Tracing Data Contamination in Large Language Models
        # LINK: https://arxiv.org/abs/2308.08493
        guided_prompt = construct_qg_prompt(guided_instruction)
        general_instruction = construct_qg_prompt(general_instruction)

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
        quiz_prompt, option_letters, original_letter = construct_quiz(quiz_instruction, paraphrase_prompts[:min(n_para, 3)], prompt)
        quiz_output, _ = end2end(quiz_prompt, model, tokenizer)
        option_ids = tokenizer.convert_tokens_to_ids([' ' + l for l in option_letters])
        quiz_final_logit = quiz_output.logits[0, -1, :]
        quiz_pred_letter = option_letters[int(torch.argmax(quiz_final_logit[option_ids]))]
        quiz_score = 1 if quiz_pred_letter == original_letter else 0

        # PAPER: Membership Inference Attacks against Language Models via Neighbourhood Comparison
        # LINK: https://aclanthology.org/2023.findings-acl.719.pdf
        avg_para_loss = float(np.mean([po.loss for po in para_outputs]))
        # CrossEntropyLoss(original) - Mean(CrossEntropyLoss(p) for p in paraphrased)
        neighbor_loss_delta = outputs.loss - avg_para_loss

        # PAPER: Extracting Training Data from Large Language Models
        # LINK: https://arxiv.org/abs/2012.07805
        orig_ppl = torch.exp(outputs.loss)
        lower_ppl = torch.exp(outputs_lower.loss)
        small_ppl = torch.exp(outputs_small.loss)

        lower_ratio = orig_ppl / lower_ppl
        small_lratio = float(np.log(orig_ppl) / np.log(small_ppl))

        zlib_entropy = calculate_zlib_entropy(prompt)
        zlib_lratio = float(np.log(orig_ppl) / zlib_entropy)

        row = {
            'idx': idx,
            'min_k':  min_k_lprobs,
            'quiz_score': quiz_score,
            'neighbor_loss_delta': neighbor_loss_delta,
            'ppl': orig_ppl,
            'lower_ratio': lower_ratio,
            'small_lratio': small_lratio,
            'zlib_lratio': zlib_lratio
        }

        stats.append(row)
    
    stats = pd.DataFrame(stats)
    out_fn = os.path.join('results', f'{args.model}_{args.dataset}.csv')
    print(f'Saving to {out_fn}...')
    stats.to_csv(out_fn, index=False)

    print('Printing out average scores...')
    print(stats.select_dtypes('number').mean())
