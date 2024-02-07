from dataclasses import dataclass
from typing import Callable


@dataclass
class DatasetConfig:
    name: str
    huggingface_path: tuple
    eval_split: str = 'test'
    input_description: str = 'a set of multiple choice options and an answer'
    question_col: str = 'question'
    prompt_generator: Callable
    paraphrase_prompt_generator: Callable
    qg_prompt_generator: Callable


@dataclass
class ModelConfig:
    name: str
    huggingface_path: str
    is_chat: bool



def construct_pubmedqa_qg_input(example):
    # QG: Question Generation conditioned on the inputs
    inputs = '\n'.join(example['CONTEXTS'])
    answer = example['final_decision']
    return f'{inputs}\n\nAnswer: {answer}\n\nQUESTION: '


def construct_medqa_qg_input(example):
    choice_letters = ['A', 'B', 'C', 'D']
    choice_options = [
        example['ending0'],
        example['ending1'],
        example['ending2'],
        example['ending3'],
    ]

    target = choice_letters[example['label']]
    prompt_lines = ['OPTIONS']
    for l, o in zip(choice_letters, choice_options):
        prompt_lines.append(f'{l}) {o}')
    prompt_lines.append(f'ANSWER: {target}')
    prompt_lines.append('QUESTION: ')
    return '\n'.join(prompt_lines)


def construct_medmcqa_qg_input(example):
    choice_letters = ['A', 'B', 'C', 'D']
    choice_options = [
        example['opa'],
        example['opb'],
        example['opc'],
        example['opd'],
    ]

    target = choice_letters[example['cop']]
    prompt_lines = ['OPTIONS']
    for l, o in zip(choice_letters, choice_options):
        prompt_lines.append(f'{l}) {o}')
    prompt_lines.append(f'ANSWER: {target}')
    prompt_lines.append('QUESTION: ')
    return '\n'.join(prompt_lines)


def construct_mmlu_qg_input(example):
    choice_letters = ['A', 'B', 'C', 'D']
    choice_options = [example[l] for l in choice_letters]

    target = example['target']
    prompt_lines = ['OPTIONS']
    for l, o in zip(choice_letters, choice_options):
        prompt_lines.append(f'{l}) {o}')
    prompt_lines.append(f'ANSWER: {target}')
    prompt_lines.append('QUESTION: ')
    return '\n'.join(prompt_lines)


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


def build_medqa_prompt(example):
    q = example['sent1']
    choice_letters = ['A', 'B', 'C', 'D']
    choice_options = [
        example['ending0'],
        example['ending1'],
        example['ending2'],
        example['ending3'],
    ]

    target = choice_letters[example['label']]
    prompt_lines = [f'QUESTION: {q}', 'CHOICES']
    for l, o in zip(choice_letters, choice_options):
        prompt_lines.append(f'{l}) {o}')
    
    prompt_lines.append(f'ANSWER: {target}')

    return '\n'.join(prompt_lines)


def build_medqa_paraphrase_prompts(example, paraphrases):
    para_qs = paraphrases['sent1_para']
    choice_letters = ['A', 'B', 'C', 'D']
    choice_options = [
        example['ending0'],
        example['ending1'],
        example['ending2'],
        example['ending3'],
    ]

    target = choice_letters[example['label']]
    outputs = []
    for pq in para_qs:
        prompt_lines = [f'QUESTION: {pq}', 'CHOICES']
        for l, o in zip(choice_letters, choice_options):
            prompt_lines.append(f'{l}) {o}')
        prompt_lines.append(f'ANSWER: {target}')
        outputs.append('\n'.join(prompt_lines))
    return outputs


def build_mmlu_prompt(example):
    q = example['input']
    choice_letters = ['A', 'B', 'C', 'D']
    choice_options = [example[l] for l in choice_letters]
    target = example['target']

    prompt_lines = [f'QUESTION: {q}', 'CHOICES']
    for l, o in zip(choice_letters, choice_options):
        prompt_lines.append(f'{l}) {o}')
    
    prompt_lines.append(f'ANSWER: {target}')

    return '\n'.join(prompt_lines)


def build_mmlu_paraphrase_prompts(example, paraphrases):
    para_qs = paraphrases['question_para']
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


DATASET_CONFIGS = {
    'pubmedqa': DatasetConfig(
        name='pubmedqa',
        huggingface_path=('bigbio/pubmed_qa', 'pubmed_qa_labeled_fold0_source'),
        input_description='a PubMed abstract',
        question_col='QUESTION',
        prompt_generator=build_pubmedqa_prompt,
        paraphrase_prompt_generator=build_pubmedqa_paraphrase_prompts,
        qg_prompt_generator=construct_pubmedqa_qg_input,
    ),
    'medmcqa': DatasetConfig(
        name='medmcqa',
        huggingface_path=('medmcqa', ),
        eval_split='validation',
        question_col='question',
        prompt_generator=build_medmcqa_prompt,
        paraphrase_prompt_generator=build_medmcqa_paraphrase_prompts,
        qg_prompt_generator=construct_medmcqa_qg_input,
    ),
    'medqa': DatasetConfig(
        name='medqa',
        huggingface_path=('GBaker/MedQA-USMLE-4-options-hf', ),
        question_col='sent1',
        prompt_generator=build_medqa_prompt,
        paraphrase_prompt_generator=build_medqa_paraphrase_prompts,
        qg_prompt_generator=construct_medqa_qg_input,
    ),
    'mmlu_anatomy': DatasetConfig(
        name='mmlu_anatomy',
        huggingface_path=('lukaemon/mmlu', 'anatomy'),
        question_col='input',
        prompt_generator=build_mmlu_prompt,
        paraphrase_prompt_generator=build_mmlu_paraphrase_prompts,
        qg_prompt_generator=construct_mmlu_qg_input,
    ),
    'mmlu_clinical_knowledge': DatasetConfig(
        name='mmlu_clinical_knowledge',
        huggingface_path=('lukaemon/mmlu', 'clinical_knowledge'),
        question_col='input',
        prompt_generator=build_mmlu_prompt,
        paraphrase_prompt_generator=build_mmlu_paraphrase_prompts,
        qg_prompt_generator=construct_mmlu_qg_input,
    ),
    'mmlu_medical_genetics': DatasetConfig(
        name='mmlu_medical_genetics',
        huggingface_path=('lukaemon/mmlu', 'medical_genetics'),
        question_col='input',
        prompt_generator=build_mmlu_prompt,
        paraphrase_prompt_generator=build_mmlu_paraphrase_prompts,
        qg_prompt_generator=construct_mmlu_qg_input,
    ),
    'mmlu_professional_medicine': DatasetConfig(
        name='mmlu_professional_medicine',
        huggingface_path=('lukaemon/mmlu', 'professional_medicine'),
        question_col='input',
        prompt_generator=build_mmlu_prompt,
        paraphrase_prompt_generator=build_mmlu_paraphrase_prompts,
        qg_prompt_generator=construct_mmlu_qg_input,
    ),
    'mmlu_college_biology': DatasetConfig(
        name='mmlu_college_biology',
        huggingface_path=('lukaemon/mmlu', 'college_biology'),
        question_col='input',
        prompt_generator=build_mmlu_prompt,
        paraphrase_prompt_generator=build_mmlu_paraphrase_prompts,
        qg_prompt_generator=construct_mmlu_qg_input,
    ),
    'mmlu_college_medicine': DatasetConfig(
        name='mmlu_college_medicine',
        huggingface_path=('lukaemon/mmlu', 'college_medicine'),
        question_col='input',
        prompt_generator=build_mmlu_prompt,
        paraphrase_prompt_generator=build_mmlu_paraphrase_prompts,
        qg_prompt_generator=construct_mmlu_qg_input,
    ),
}


MODEL_CONFIGS = {
    'debug': ModelConfig(
        name='debug',
        huggingface_path='HuggingFaceM4/tiny-random-LlamaForCausalLM',
        is_chat=False
    ),
    'mixtral': ModelConfig(
        name='mixtral',
        huggingface_path='mistralai/Mixtral-8x7B-v0.1',
        is_chat=False
    ),
    'zephyr-7b': ModelConfig(
        name='zephyr-7b',
        huggingface_path='HuggingFaceH4/zephyr-7b-beta',
        is_chat=True
    ),
    'yi-34b': ModelConfig(
        name='yi-34b',
        huggingface_path='01-ai/Yi-34B-Chat',
        is_chat=True
    ),
    'qwen-72b': ModelConfig(
        name='qwen-72b',
        huggingface_path='Qwen/Qwen-72B',
        is_chat=False
    )
}
