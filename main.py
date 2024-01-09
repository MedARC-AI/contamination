import argparse
from datasets import load_dataset
import pandas as pd
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='phi2', choices=list(MODEL_DICT.keys()))
    parser.add_argument('--dataset', default='pubmedqa', choices=list(DATASET_DICT.keys()))
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--split', default='test')

    args = parser.parse_args()

    dataset = load_dataset(*DATASET_DICT[args.dataset])[args.split]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DICT[args.model], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DICT[args.model], trust_remote_code=True).eval().to(args.device)

    stats = []

    for example in tqdm(dataset):
        prompt = build_prompt(example, dataset_name=args.dataset)

        inputs = tokenizer(
            prompt,
            max_length=4096,
            truncation=True,
        )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        outputs = model(**inputs)

        print('here')

