# MultiMEDQA Contamination Detection

## Description

We provide which implements Membership Inference Attacks (MIA) from 6 papers to determine whether a provided HuggingFace Model was trained on any of the tests sets in the [MultiMedQA benchmark](https://www.nature.com/articles/s41586-023-06291-2). The MultiMedQA benchmark consists of the following QA datasets:

* PubMedQA
* MedMCQA
* MedQA
* MMLU - Anatomy, Clinical Knowledge, College Biology, College Medicine, Medical Genetics, Professional Medicine

*The script is easily extensible to other QA datasets*

The methods implemented are from the following papers:

* [Capabilities of GPT-4 on Medical Challenge Problems](https://arxiv.org/abs/2303.13375)
* [Detecting Pretraining Data from Large Language Models](https://arxiv.org/abs/2310.16789)
* [Data Contamination Quiz: A Tool to Detect and Estimate Contamination in Large Language Models](https://arxiv.org/abs/2311.06233)
* [Membership Inference Attacks against Language Models via Neighbourhood Comparison](https://aclanthology.org/2023.findings-acl.719/)
* [Time Travel in LLMs: Tracing Data Contamination in Large Language Models](https://arxiv.org/abs/2308.08493)
* [Extracting Training Data from Large Language Models](https://arxiv.org/abs/2012.07805)

## Setup

`pip install -r requirements.txt`

## Running

If testing on a dataset not in MultiMedQA, first add it to the `DATASET_CONFIGS` dictionary in `configs.py`.

If testing on a HuggingFace model not in `MODEL_CONFIGS`, add it as well.

### Generating Paraphrases (Neighbors)

2 of the contamination methods from:

* [Data Contamination Quiz: A Tool to Detect and Estimate Contamination in Large Language Models](https://arxiv.org/abs/2311.06233)
* [Membership Inference Attacks against Language Models via Neighbourhood Comparison](https://aclanthology.org/2023.findings-acl.719/)

involve pre-computing paraphrased versions of test set instances. If you want to run these tests, you must first run `python paraphrase.py` which will generate `--num_neighbors` paraphrases of each test set instance and save inside `./results/neighbors`.

The main script can be run with `-no_neighbors` if you want to disable these paraphrase-based scores.

### Running Contamination Metrics

`python main.py --model qwen-72b --dataset pubmedqa --max_examples 100`

will test `qwen-72b` on the full suite of metrics on a random sample of `100` examples from the  `pubmedqa` test set.

The results will print to the console as well as be saved to a `.csv` file under  `./results`.

#### Statistical Significance

`python significance_test.py` will print out if the ROUGE-L for “Guided” prompt is statistically significantly higher (p < 0.05) then the "General", which is explained in the below paper:

* [Data Contamination Quiz: A Tool to Detect and Estimate Contamination in Large Language Models](https://arxiv.org/abs/2311.06233)
