#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
BASE_DIR=$SCRIPT_DIR/../../../../../
PYTHONPATH=$BASE_DIR/src:$BASE_DIR/external/atria/src:$BASE_DIR/external/docsets/src:$BASE_DIR/external/opacus_dpdm:$PYTHONPATH
MODEL_CHECKPOINT=output/atria_trainer/IitCdip/iitcdip_unaug_pretraining_sbv1.4/checkpoints/checkpoint_1279790.pt
if [ -e $MODEL_CHECKPOINT ]; then
    echo "Running training with model checkpoint=$MODEL_CHECKPOINT"
else
    echo "$MODEL_CHECKPOINT does not exist."
    exit 1
fi
PYTHONPATH=$PYTHONPATH TASK=document_classification DATASET=rvlcdip TASK_MODULE=sbv1.4_classifier_guidance \
    python $BASE_DIR/external/atria/src/atria/core/task_runners/atria_trainer.py \
    hydra.searchpath=[pkg://atria/conf,pkg://docsets/conf,pkg://dp_diffusion/conf] \
    +trainer_configs=dp_latent_diffusion/preextracted_features \
    torch_model_builder@task_module.torch_model_builder.reverse_diffusion_model=local \
    +task_module.torch_model_builder.reverse_diffusion_model.model_name=dp_diffusion.models.openai_models.dp_promise_unet.models.UNetModel \
    +model_config@task_module.torch_model_builder.reverse_diffusion_model=dp_promise_openai_unet_model_32x32_v1 \
    task_module.torch_model_builder.reverse_diffusion_model.in_channels=4 \
    task_module.torch_model_builder.reverse_diffusion_model.out_channels=4 \
    model_checkpoint=$MODEL_CHECKPOINT \
    image_size=256 \
    experiment_name=rvlcdip_dp_unaug_sbv1.4_cfg \
    features_key=stable-diffusion-v1-4