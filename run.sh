#!/bin/bash
set -e

MODEL=$1
DATASETS=(
    pubmedqa medmcqa mmlu_anatomy mmlu_clinical_knowledge mmlu_college_medicine mmlu_medical_genetics mmlu_professional_medicine mmlu_college_biology
)

for dataset in "${DATASETS[@]}"
do
    echo "Starting $dataset for $MODEL..."
    python3 main.py --dataset $dataset --model $MODEL -no_neighbors -overwrite
done
