#!/bin/bash
set -e

DATASETS=(
    mmlu_anatomy mmlu_clinical_knowledge mmlu_college_medicine mmlu_medical_genetics mmlu_professional_medicine mmlu_college_biology pubmedqa medmcqa
)

for dataset in "${DATASETS[@]}"
do
    echo "Starting $dataset..."
    python3 generate_neighbors.py --dataset $dataset
done
