import argparse
from datasets import load_dataset
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', default='phi2', choices=list(MODEL_DICT.keys()))
    parser.add_argument('--dataset', default='pubmedqa', choices=list(DATASET_DICT.keys()))
    parser.add_argument('--device', default=0, type=int)
    parser.add_argument('--split', default='test')

    parser.add_argument('--min_k', default=20, type=int)
    parser.add_argument('--min_k_epsilon', default=0.1, type=float)

    args = parser.parse_args()

    dataset = load_dataset(*DATASET_DICT[args.dataset])[args.split]

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DICT[args.model], trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_DICT[args.model], trust_remote_code=True).eval().to(args.device)

    stats = []

    for idx, example in tqdm(enumerate(dataset)):
        prompt = build_prompt(example, dataset_name=args.dataset)

        inputs = tokenizer(
            prompt,
            max_length=4096,
            truncation=True,
            # return_attention_mask=False,
            return_tensors='pt'
        )

        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        labels = inputs['input_ids'][0, 1:]

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits[0, :-1]
        lprobs = torch.log_softmax(logits, dim=-1)

        target_lprobs = torch.gather(lprobs, 1, labels.view(-1, 1)).squeeze(-1)

        seq_len = len(target_lprobs)

        k = min(args.min_k, seq_len)
        min_k_lprobs = float(torch.topk(target_lprobs, k, dim=0, largest=False).values.mean().item())

        row = {
            'idx': idx,
            'min_k':  min_k_lprobs,
        }

