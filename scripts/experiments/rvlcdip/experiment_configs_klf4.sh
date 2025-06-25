#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
CONFIGS=(
    # cfg training configs
    # "${SCRIPT_DIR}/dp/augmented_dataset/train_cfg_kl_f4.sh"
    # "${SCRIPT_DIR}/dp/unaugmented_dataset/train_cfg_kl_f4.sh"
    "${SCRIPT_DIR}/dp/unaugmented_dataset_class_cond/train_cfg_kl_f4.sh"
    # "${SCRIPT_DIR}/dp_promise/augmented_dataset/train_cfg_kl_f4.sh"
    # "${SCRIPT_DIR}/dp_promise/unaugmented_dataset/train_cfg_kl_f4.sh"
    "${SCRIPT_DIR}/dp_promise/unaugmented_dataset_class_cond/train_cfg_kl_f4.sh"

    # per label training configs
    # "${SCRIPT_DIR}/dp_per_label/augmented_dataset/train_kl_f4.sh"
    # "${SCRIPT_DIR}/dp_per_label/unaugmented_dataset/train_kl_f4.sh"
    "${SCRIPT_DIR}/dp_per_label/unaugmented_dataset_class_cond/train_kl_f4.sh"
    # "${SCRIPT_DIR}/dp_promise_per_label/augmented_dataset/train_kl_f4.sh"
    # "${SCRIPT_DIR}/dp_promise_per_label/unaugmented_dataset/train_kl_f4.sh"
    "${SCRIPT_DIR}/dp_promise_per_label/unaugmented_dataset_class_cond/train_kl_f4.sh"
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
bash "$SCRIPT_PATH"