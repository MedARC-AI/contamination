def build_prompt(example, dataset_name):
    if dataset_name == 'pubmedqa':
        return build_pubmedqa_prompt(example)
    else:
        raise Exception(f'Unrecognized dataset -> {dataset_name}')


def build_pubmedqa_prompt(example):
    return '\n'.join(example['CONTEXTS']) + '\n\n' + example['QUESTION'] + '\n\n' + 'Answer: ' + example['final_decision']