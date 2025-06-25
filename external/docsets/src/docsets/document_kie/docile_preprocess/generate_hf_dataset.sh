#!/bin/bash
set -euo pipefail
DATASET_PATH=/home/$USER/.cache/atria/datasets/docile
python generate_hf_dataset.py --docile_path ${DATASET_PATH} \
    --preprocessed_dataset_path ${DATASET_PATH}/preprocessed_dataset \
    --arrow_format_path ${DATASET_PATH}/preprocessed_dataset/arrow/
