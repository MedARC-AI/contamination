def build_prompt(example, dataset_name, paraphrases=False):
    if dataset_name == 'pubmedqa':
        return build_pubmedqa_prompt(example, paraphrases=paraphrases)
    else:
        raise Exception(f'Unrecognized dataset -> {dataset_name}')


def _build_pubmedqa_prompt(ctxs, question, answer):
    return '\n'.join(ctxs) + '\n\n' + question + '\n\n' + 'Answer: ' + answer


def build_pubmedqa_prompt(example, paraphrases=False):
    q = example['QUESTION']
    a = example['final_decision']
    ctxs = example['CONTEXTS']
    # TODO: CONTEXTS_para doesn't exist now so let's just use CONTEXTS 3x as a placeholder
    # ctxs_para = [example['CONTEXTS'] for _ in range(3)]
    ctxs_para = [['fake'] for _ in range(3)]
    # ctxs_para = example['CONTEXTS_para']

    if paraphrases:
        return [_build_pubmedqa_prompt(para_ctxs, q, a) for para_ctxs in ctxs_para]
    else:
        return _build_pubmedqa_prompt(ctxs, q, a)
