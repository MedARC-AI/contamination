def build_prompt(example, dataset_name):
    if dataset_name == 'pubmedqa':
        return build_pubmedqa_prompt(example)
    elif dataset_name == 'medmcqa':
        return build_medmcqa_prompt(example)
    elif 'mmlu' in dataset_name:
        return build_mmmlu_prompt(example)
    else:
        raise Exception(f'Unrecognized dataset -> {dataset_name}')
    

def build_paraphrase_prompts(example, dataset_name, paraphrases):
    if dataset_name == 'pubmedqa':
        return build_pubmedqa_paraphrase_prompts(example, paraphrases=paraphrases)
    elif dataset_name == 'medmcqa':
        return build_medmcqa_paraphrase_prompts(example, paraphrases=paraphrases)
    elif 'mmlu' in dataset_name:
        return build_mmmlu_paraphrase_prompts(example, paraphrases=paraphrases)
    else:
        raise Exception(f'Unrecognized dataset -> {dataset_name}')


def _build_pubmedqa_prompt(ctxs, question, answer):
    return '\n'.join(ctxs) + '\n\n' + question + '\n\n' + 'Answer: ' + answer


def build_pubmedqa_paraphrase_prompts(example, paraphrases):
    a = example['final_decision']
    return [_build_pubmedqa_prompt(para_ctxs, para_q, a) for para_q, para_ctxs in zip(paraphrases['QUESTION_para'], paraphrases['CONTEXTS_para'])]


def build_pubmedqa_prompt(example):
    q = example['QUESTION']
    a = example['final_decision']
    ctxs = example['CONTEXTS']
    return _build_pubmedqa_prompt(ctxs, q, a)


def build_medmcqa_paraphrase_prompts(example, paraphrases):
    para_qs = paraphrases['question_para']
    choice_letters = ['A', 'B', 'C', 'D']
    choice_options = [
        example['opa'],
        example['opb'],
        example['opc'],
        example['opd'],
    ]

    target = choice_letters[example['cop']]
    outputs = []
    for pq in para_qs:
        prompt_lines = [f'QUESTION: {pq}', 'CHOICES']
        for l, o in zip(choice_letters, choice_options):
            prompt_lines.append(f'{l}) {o}')
        prompt_lines.append(f'ANSWER: {target}')
        outputs.append('\n'.join(prompt_lines))
    return outputs


def build_medmcqa_prompt(example):
    q = example['question']
    # para_qs = ['fake' for x in example['question_para']]
    para_qs = ['fake' for _ in range(3)]
    choice_letters = ['A', 'B', 'C', 'D']
    choice_options = [
        example['opa'],
        example['opb'],
        example['opc'],
        example['opd'],
    ]

    target = choice_letters[example['cop']]
    prompt_lines = [f'QUESTION: {q}', 'CHOICES']
    for l, o in zip(choice_letters, choice_options):
        prompt_lines.append(f'{l}) {o}')
    
    prompt_lines.append(f'ANSWER: {target}')

    return '\n'.join(prompt_lines)


def build_mmmlu_prompt(example):
    q = example['input']
    choice_letters = ['A', 'B', 'C', 'D']
    choice_options = [example[l] for l in choice_letters]
    target = example['target']

    prompt_lines = [f'QUESTION: {q}', 'CHOICES']
    for l, o in zip(choice_letters, choice_options):
        prompt_lines.append(f'{l}) {o}')
    
    prompt_lines.append(f'ANSWER: {target}')

    return '\n'.join(prompt_lines)


def build_mmmlu_paraphrase_prompts(example, paraphrases):
    para_qs = example['input_para']
    choice_letters = ['A', 'B', 'C', 'D']
    choice_options = [example[l] for l in choice_letters]

    target = example['target']
    outputs = []
    for pq in para_qs:
        prompt_lines = [f'QUESTION: {pq}', 'CHOICES']
        for l, o in zip(choice_letters, choice_options):
            prompt_lines.append(f'{l}) {o}')
        prompt_lines.append(f'ANSWER: {target}')
        outputs.append('\n'.join(prompt_lines))
    return outputs
