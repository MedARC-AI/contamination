import json
import os
import regex as re

import argparse
from datasets import load_dataset
import numpy as np
from openai import AzureOpenAI
import pandas as pd
from tqdm import tqdm

from utils import build_prompt
from main import DATASET_DICT, EVAL_SPLIT


assert 'OPENAI_API_KEY' in os.environ
client = AzureOpenAI(
    api_key=os.environ.get('OPENAI_API_KEY'),
    azure_endpoint='https://east-us-2-llm.openai.azure.com/',
    api_version='2023-05-15',
    azure_deployment='misc-gpt4-turbo'
)


# Figure 4 from https://arxiv.org/pdf/2311.06233.pdf
# THREE_PARAPHRASE_INSTRUCTION = ("""
# Instruction: Your task is to create a three-choice quiz by only replacing the words in the provided text with their synonyms. The meaning and sentence structure of the three new options must exactly mirror every detail in the text. You must not include the provided text as an option. You must make sure that:
# (1) You generate three distinct options based on the provided text;
# (2) Options are ordered;
# (3) There is not any extra explanation; and
# (4) You comply with every specific symbol and letter detail in the given text.
# """).strip()


THREE_PARAPHRASE_INSTRUCTION = ("""
Instruction: Your task is to generate 3 distinct paraphrases of the following {} by only replacing the words in the provided text with their synonyms. The meaning and sentence structure of the three paraphrases must exactly mirror every detail in the text. You must make sure that:
(1) You generate three distinct paraphrases;
(2) There is not any extra explanation; and
(3) You comply with every specific symbol and letter detail in the given {}.

Return a JSON array of 3 strings.
""").strip()


def chatgpt(messages, model='gpt-4', temperature=0.1, max_tokens=2048):
    completion = client.with_options(max_retries=5).chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens,
        response_format={'type': 'json_object'}
    )
    obj = json.loads(completion.choices[0].message.content)
    if type(obj) == list:
        return obj

    if type(obj) == list:
        return obj
    elif 'paraphrases' in obj:
        return obj['paraphrases']
    elif 'paraphrase1' in obj:
        return [
            obj['paraphrase1'],
            obj['paraphrase2'],
            obj['paraphrase3'],
        ]

    print('Could not parse json')
    return None 


def gen_mcqa_neighbor_prompt(example, q_col='question'):
    q_instruct = THREE_PARAPHRASE_INSTRUCTION.format('QUESTION', 'QUESTION')
    q = example[q_col]
    q_prompt = f'{q_instruct}\n\nQUESTION:\n{q}\n\n'
    q_messages = [
        {'role': 'system', 'content': 'You are a helpful assistant for text paraphrasing.'},
        {'role': 'user', 'content': q_prompt}
    ]

    q_output = chatgpt(q_messages)
    try:
        assert len(q_output) == 3
    except:
        print(q_prompt)
        return None

    return {'question_para': q_output}


def gen_pubmedqa_neighbor_prompt(example):
    ctxs_output = [[], [], []]
    
    q_instruct = THREE_PARAPHRASE_INSTRUCTION.format('QUESTION', 'QUESTION')
    q = example['QUESTION']
    q_prompt = f'{q_instruct}\n\nQUESTION:\n{q}\n\n'
    q_messages = [
        {'role': 'system', 'content': 'You are a helpful assistant for text paraphrasing.'},
        {'role': 'user', 'content': q_prompt}
    ]

    q_output = chatgpt(q_messages)
    assert len(q_output) == 3

    for ctx in example['CONTEXTS']:
        ctx_instruct = THREE_PARAPHRASE_INSTRUCTION.format('ABSTRACT', 'ABSTRACT')
        ctx_prompt = f'{ctx_instruct}\n\nABSTRACT:\n{ctx}\n\n'
        ctx_messages = [
            {'role': 'system', 'content': 'You are a helpful assistant for text paraphrasing.'},
            {'role': 'user', 'content': ctx_prompt}
        ]
        try:
            ctx_output = chatgpt(ctx_messages)
            assert len(ctx_output) == 3
        except:
            print(ctx_prompt)
            return None

        for i, o in enumerate(ctx_output):
            ctxs_output[i].append(o)

    return {
        'CONTEXTS_para': ctxs_output,
        'QUESTION_para': q_output
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='gpt-4')
    parser.add_argument('--dataset', default='pubmedqa', choices=list(DATASET_DICT.keys()))
    parser.add_argument('--max_examples', default=100, type=int)

    parser.add_argument('-overwrite', default=False, action='store_true')

    # Paraphrase params
    parser.add_argument('--num_neighbors', default=3, type=int)

    args = parser.parse_args()

    args.split = EVAL_SPLIT[args.dataset]
    cache_dir = os.path.join('results', 'neighbors', f'{args.dataset}_{args.split}')
    os.makedirs(cache_dir, exist_ok=True)

    dataset = load_dataset(*DATASET_DICT[args.dataset])[args.split]

    if len(dataset) > args.max_examples:
        idxs = np.arange(len(dataset))
        np.random.seed(1992)
        np.random.shuffle(idxs)
        print(f'Selecting a random subset of {args.max_examples} from {len(dataset)} examples.')
        dataset = dataset.select(idxs[:args.max_examples])


    new_cols = []
    for idx, example in tqdm(enumerate(dataset), total=len(dataset)):
        out_fn = os.path.join(cache_dir, f'{idx}.json')
        if os.path.exists(out_fn):
            print(f'Loading from cache --> {out_fn}')
            with open(out_fn, 'r') as fd:
                obj = json.load(fd)
        else:
            if args.dataset == 'pubmedqa':
                obj = gen_pubmedqa_neighbor_prompt(example)
            elif args.dataset == 'medmcqa':
                obj = gen_mcqa_neighbor_prompt(example, q_col='question')
            elif 'mmlu' in args.dataset:
                obj = gen_mcqa_neighbor_prompt(example, q_col='input')
            if obj is None:
                print(f'Unable to process {idx}...Moving on for now')
            else:
                print(f'Saving parsed to {out_fn}')
                with open(out_fn, 'w') as fd:
                    json.dump(obj, fd)
        
        new_cols.append(obj)

    # dataset_w_new_cols = dataset.map(lambda _, idx: new_cols[idx], with_indices=True)

    # out_dir = f'results/neighbors_{args.dataset}'
    # print(f'Saving to {out_dir}')
    # dataset_w_new_cols.save_to_disk(out_dir)
