def build_prompt(example, dataset_name, paraphrases=False):
    if dataset_name == 'pubmedqa':
        return build_pubmedqa_prompt(example, paraphrases=paraphrases)
    elif dataset_name == 'medmcqa':
        return build_medmcqa_prompt(example, paraphrases=paraphrases)
    elif 'mmlu' in dataset_name:
        return build_mmmlu_prompt(example, paraphrases=paraphrases)
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


def build_medmcqa_prompt(example, paraphrases=False):
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

    if paraphrases:
        outputs = []
        for pq in para_qs:
            prompt_lines = [f'QUESTION: {pq}', 'CHOICES']
            for l, o in zip(choice_letters, choice_options):
                prompt_lines.append(f'{l}) {o}')
            prompt_lines.append(f'ANSWER: {target}')
            outputs.append('\n'.join(prompt_lines))
        return outputs
    else:
        prompt_lines = [f'QUESTION: {q}', 'CHOICES']
        for l, o in zip(choice_letters, choice_options):
            prompt_lines.append(f'{l}) {o}')
        
        prompt_lines.append(f'ANSWER: {target}')

        return '\n'.join(prompt_lines)


def build_mmmlu_prompt(example, paraphrases=False):
    q = example['input']
    para_qs = ['fake' for _ in range(3)]
    choice_letters = ['A', 'B', 'C', 'D']
    choice_options = [
        example[l] for l in choice_letters
    ]

    target = example['target']

    if paraphrases:
        outputs = []
        for pq in para_qs:
            prompt_lines = [f'QUESTION: {pq}', 'CHOICES']
            for l, o in zip(choice_letters, choice_options):
                prompt_lines.append(f'{l}) {o}')
            prompt_lines.append(f'ANSWER: {target}')
            outputs.append('\n'.join(prompt_lines))
        return outputs
    else:
        prompt_lines = [f'QUESTION: {q}', 'CHOICES']
        for l, o in zip(choice_letters, choice_options):
            prompt_lines.append(f'{l}) {o}')
        
        prompt_lines.append(f'ANSWER: {target}')

        return '\n'.join(prompt_lines)