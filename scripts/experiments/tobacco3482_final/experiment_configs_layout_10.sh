#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
CONFIGS=(
    # "${SCRIPT_DIR}/dp_per_label/unaugmented_dataset_layout_cond/train_sbv1.4.sh 0 10"
    # "${SCRIPT_DIR}/dp_per_label/unaugmented_dataset_layout_cond/train_sbv1.4.sh 1 10"
    # "${SCRIPT_DIR}/dp_per_label/unaugmented_dataset_layout_cond/train_sbv1.4.sh 2 10"
    # "${SCRIPT_DIR}/dp_per_label/unaugmented_dataset_layout_cond/train_sbv1.4.sh 3 10"
    # "${SCRIPT_DIR}/dp_per_label/unaugmented_dataset_layout_cond/train_sbv1.4.sh 4 10"
    # "${SCRIPT_DIR}/dp_per_label/unaugmented_dataset_layout_cond/train_sbv1.4.sh 5 10"
    # "${SCRIPT_DIR}/dp_per_label/unaugmented_dataset_layout_cond/train_sbv1.4.sh 6 10"
    # "${SCRIPT_DIR}/dp_per_label/unaugmented_dataset_layout_cond/train_sbv1.4.sh 7 10"
    # "${SCRIPT_DIR}/dp_per_label/unaugmented_dataset_layout_cond/train_sbv1.4.sh 8 10"
    # "${SCRIPT_DIR}/dp_per_label/unaugmented_dataset_layout_cond/train_sbv1.4.sh 9 10"

    "${SCRIPT_DIR}/dp_per_label/unaugmented_dataset_layout_cond/train_kl_f4.sh 0 10"
    "${SCRIPT_DIR}/dp_per_label/unaugmented_dataset_layout_cond/train_kl_f4.sh 1 10"
    "${SCRIPT_DIR}/dp_per_label/unaugmented_dataset_layout_cond/train_kl_f4.sh 2 10"
    "${SCRIPT_DIR}/dp_per_label/unaugmented_dataset_layout_cond/train_kl_f4.sh 3 10"
    "${SCRIPT_DIR}/dp_per_label/unaugmented_dataset_layout_cond/train_kl_f4.sh 4 10"
    # "${SCRIPT_DIR}/dp_per_label/unaugmented_dataset_layout_cond/train_kl_f4.sh 5 10"
    # "${SCRIPT_DIR}/dp_per_label/unaugmented_dataset_layout_cond/train_kl_f4.sh 6 10"
    # "${SCRIPT_DIR}/dp_per_label/unaugmented_dataset_layout_cond/train_kl_f4.sh 7 10"
    # "${SCRIPT_DIR}/dp_per_label/unaugmented_dataset_layout_cond/train_kl_f4.sh 8 10"
    # "${SCRIPT_DIR}/dp_per_label/unaugmented_dataset_layout_cond/train_kl_f4.sh 9 10"
)

if [ -z "$1" ]; then
    echo "Please provide a config ID as an argument."
    exit 1
fi

CONFIG_ID=$1

if [ "$CONFIG_ID" -lt 0 ] || [ "$CONFIG_ID" -ge "${#CONFIGS[@]}" ]; then
    echo "Invalid config ID. Please provide a valid ID between 0 and $((${#CONFIGS[@]} - 1))."
    exit 1
fi

SCRIPT_PATH="${CONFIGS[$CONFIG_ID]}"
echo "Running script: $SCRIPT_PATH"
RUN=eval $SCRIPT_PATH
$RUN