#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
BASE_PATH=$SCRIPT_DIR/../../../
PYTHONPATH=$BASE_PATH/src:$BASE_PATH/external/atria/src:$BASE_PATH/external/docsets/src:$PYTHONPATH
OVERRIDE_PATH=$1
EXP_NAME=$2
echo $OVERRIDE_PATH $EXP_NAME
PYTHONPATH=$PYTHONPATH TASK=document_classification DATASET=rvlcdip python $BASE_PATH/external/atria/src/atria/core/task_runners/atria_trainer.py \
    hydra.searchpath=[pkg://atria/conf,pkg://docsets/conf,pkg://dp_diffusion/conf] \
    +trainer_configs=image_classification/train \
    torch_model_builder@task_module.torch_model_builder=timm \
    +task_module.torch_model_builder.model_name=convnext_base \
    +task_module.torch_model_builder.drop_rate=0.5 \
    image_size=256 \
    experiment_name=$EXP_NAME \
    +data_module.train_dataset_override_path=$OVERRIDE_PATH \
    data_module.max_train_samples=50000
