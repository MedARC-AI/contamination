import os

import argparse
import backoff
from datasets import load_dataset
import numpy as np
from openai import OpenAI
import pandas as pd

from utils import build_prompt
from main import DATASET_DICT


assert 'OPENAI_KEY' in os.environ
client = OpenAI(
    # This is the default and can be omitted
    api_key=os.environ.get('OPENAI_KEY'),
)


# Figure 4 from https://arxiv.org/pdf/2311.06233.pdf
PARAPHRASE_INSTRUCTION = """
Instruction: Your task is to create a three-choice quiz by only replacing the words in the provided text with their synonyms.
The meaning and sentence structure of the three new options must exactly mirror every detail in the text. You must not
include the provided text as an option. You must make sure that:
(1) You generate three distinct options based on the provided text;
(2) Options are ordered;
(3) There is not any extra explanation; and
(4) You comply with every specific symbol and letter detail in the given text.
"""


def chatgpt(messages, model='gpt-4', temperature=0.1, max_tokens=2048):
    completion = client.with_options(max_retries=5).chat.completions.create(
        model=model, messages=messages, temperature=temperature, max_tokens=max_tokens
    )
    return completion.choices[0].message.content


def gen_prompt_and_neighbor_prompts(example, dataset_name='pubmedqa'):
    original_prompt = build_prompt(example, dataset_name=dataset_name, paraphrases=False)
    para_prompt = f'{PARAPHRASE_INSTRUCTION}\n--\n{original_prompt}\n--\n'

    messages = [
        {'role': 'system', 'content': 'You are a helpful assistant for text paraphrasing.'},
        {'role': 'user', 'content': para_prompt}
    ]

    gpt_output = chatgpt(messages)

    # Parse output
    print('here')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='gpt-3.5-turbo', choices=[x.id for x in client.models.list()])
    parser.add_argument('--dataset', default='pubmedqa', choices=list(DATASET_DICT.keys()))
    parser.add_argument('--split', default='test')

    # Paraphrase params
    parser.add_argument('--num_neighbors', default=3, type=int)

    args = parser.parse_args()

    dataset = load_dataset(*DATASET_DICT[args.dataset])[args.split]

    dataset = dataset.map(
        lambda example: gen_prompt_and_neighbor_prompts(example, dataset_name=args.dataset)
    )
