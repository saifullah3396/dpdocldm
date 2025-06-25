#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
CONFIGS=(
    # cfg training configs
    "${SCRIPT_DIR}/dp/augmented_dataset/train_cfg_kl_f4.sh"
    "${SCRIPT_DIR}/dp/augmented_dataset/train_cfg_sbv1.4.sh"
    "${SCRIPT_DIR}/dp/unaugmented_dataset/train_cfg_kl_f4.sh"
    "${SCRIPT_DIR}/dp/unaugmented_dataset/train_cfg_sbv1.4.sh"
    "${SCRIPT_DIR}/dp_promise/augmented_dataset/train_cfg_kl_f4.sh"
    "${SCRIPT_DIR}/dp_promise/augmented_dataset/train_cfg_sbv1.4.sh"
    "${SCRIPT_DIR}/dp_promise/unaugmented_dataset/train_cfg_kl_f4.sh"
    "${SCRIPT_DIR}/dp_promise/unaugmented_dataset/train_cfg_sbv1.4.sh"

    # per label training configs
    "${SCRIPT_DIR}/dp_per_label/augmented_dataset/train_kl_f4.sh"
    "${SCRIPT_DIR}/dp_per_label/augmented_dataset/train_sbv1.4.sh"
    "${SCRIPT_DIR}/dp_per_label/unaugmented_dataset/train_kl_f4.sh"
    "${SCRIPT_DIR}/dp_per_label/unaugmented_dataset/train_sbv1.4.sh"
    "${SCRIPT_DIR}/dp_promise_per_label/augmented_dataset/train_kl_f4.sh"
    "${SCRIPT_DIR}/dp_promise_per_label/augmented_dataset/train_sbv1.4.sh"
    "${SCRIPT_DIR}/dp_promise_per_label/unaugmented_dataset/train_kl_f4.sh"
    "${SCRIPT_DIR}/dp_promise_per_label/unaugmented_dataset/train_sbv1.4.sh"
)

total_configs=${#CONFIGS[@]}
for ((i = 0; i < total_configs; i++)); do
    SCRIPT_PATH="${CONFIGS[$i]}"
    EXP_NAME=$(basename "$SCRIPT_PATH" .sh)  # Extract filename without extension
    echo "Running script: $SCRIPT_PATH"
    echo "Extracted EXP_NAME: $EXP_NAME"
    bash "$SCRIPT_PATH"
done