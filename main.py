import os

import argparse
from datasets import load_dataset
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils import *


MODEL_DICT = {
    'debug': 'HuggingFaceM4/tiny-random-LlamaForCausalLM',
    'phi2': 'microsoft/phi-2',
}


DATASET_DICT = {
    'pubmedqa': ('bigbio/pubmed_qa', 'pubmed_qa_labeled_fold0_source')
}


QUIZ_TEMPLATE = 'Your task is to accurately select the option that corresponds exactly to an instance from the {} split of the {} dataset. Only generate a single option letter as your answer.'


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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='phi2', choices=list(MODEL_DICT.keys()))
    parser.add_argument('--ref_model', default='phi2', choices=list(MODEL_DICT.keys()))
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

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DICT[args.model], trust_remote_code=True)
    #  flash_attn=True,
    model = AutoModelForCausalLM.from_pretrained(MODEL_DICT[args.model], torch_dtype='auto', trust_remote_code=True).eval().to(args.device)

    ref_tokenizer = AutoTokenizer.from_pretrained(MODEL_DICT[args.ref_model], trust_remote_code=True)
    #  flash_attn=True,
    ref_model = AutoModelForCausalLM.from_pretrained(MODEL_DICT[args.ref_model], torch_dtype='auto', trust_remote_code=True).eval().to(args.device)

    stats = []

    for idx, example in tqdm(enumerate(dataset)):
        prompt = build_prompt(example, dataset_name=args.dataset, paraphrases=False)
        paraphrase_prompts = build_prompt(example, dataset_name=args.dataset, paraphrases=True)
        n_para = len(paraphrase_prompts)
        assert n_para >= 3  # Necessary for 4-choice quiz

        outputs, labels = end2end(prompt, model, tokenizer)
        outputs_lower, labels_lower = end2end(prompt.lower(), model, tokenizer)
        outputs_ref, labels_ref = end2end(prompt, ref_model, ref_tokenizer)

        shifted_logits = outputs.logits[0, :-1]
        shifted_lprobs = torch.log_softmax(shifted_logits, dim=-1)

        shifted_labels = labels[0, 1:]

        shifted_target_lprobs = torch.gather(shifted_lprobs, 1, shifted_labels.view(-1, 1)).squeeze(-1)

        seq_len = len(shifted_target_lprobs)

        k = min(args.min_k, seq_len)
        min_k_lprobs = float(torch.topk(shifted_target_lprobs, k, dim=0, largest=False).values.mean().item())

        ppl = torch.exp(outputs.loss)

        quiz_prompt, option_letters, original_letter = construct_quiz(quiz_instruction, paraphrase_prompts[:min(n_para, 3)], prompt)

        quiz_output, _ = end2end(quiz_prompt, model, tokenizer)
        option_ids = tokenizer.convert_tokens_to_ids([' ' + l for l in option_letters])
        quiz_final_logit = quiz_output.logits[0, -1, :]
        quiz_pred_letter = option_letters[int(torch.argmax(quiz_final_logit[option_ids]))]

        quiz_score = 1 if quiz_pred_letter == original_letter else 0

        row = {
            'idx': idx,
            'min_k':  min_k_lprobs,
            'ppl': ppl,
            'quiz_score': quiz_score,
        }

        stats.append(row)
    
    stats = pd.DataFrame(stats)
    out_fn = os.path.join('results', f'{args.model}_{args.dataset}.csv')
    print(f'Saving to {out_fn}...')
    stats.to_csv(out_fn, index=False)

    print('Printing out average scores...')
    print(stats.select_dtypes('number').mean())
