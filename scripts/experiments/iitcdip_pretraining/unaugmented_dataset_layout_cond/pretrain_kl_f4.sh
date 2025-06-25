#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
BASE_DIR=$SCRIPT_DIR/../../../../
PYTHONPATH=$BASE_DIR/src:$BASE_DIR/external/atria/src:$BASE_DIR/external/docsets/src:$PYTHONPATH

# run training with openai model used in Dp-Promise repository (see README.md) with scale 0.4 default
PYTHONPATH=$PYTHONPATH TASK=document_classification DATASET=iitcdip_with_ocr TASK_MODULE=klf4_cfg_and_layout_cond python $BASE_DIR/external/atria/src/atria/core/task_runners/atria_trainer.py \
    hydra.searchpath=[pkg://atria/conf,pkg://docsets/conf,pkg://dp_diffusion/conf] \
    +trainer_configs=layout_cond_latent_diffusion/with_feature_extraction \
    torch_model_builder@task_module.torch_model_builder.reverse_diffusion_model=local \
    +task_module.torch_model_builder.reverse_diffusion_model.model_name=dp_diffusion.models.openai_models.dp_promise_unet.models.UNetModel \
    +model_config@task_module.torch_model_builder.reverse_diffusion_model=dp_promise_openai_unet_model_64x64_v1 \
    task_module.torch_model_builder.reverse_diffusion_model.in_channels=6 \
    task_module.torch_model_builder.reverse_diffusion_model.out_channels=3 \
    image_size=256 \
    feature_extraction_batch_size=64 \
    experiment_name=iitcdip_unaug_layout_cond_pretraining_klf4 \
    features_key=klf4-layout \
    do_train=False \
    do_test=False

# run training with openai model used in Dp-Promise repository (see README.md) with scale 0.4 default
PYTHONPATH=$PYTHONPATH TASK=document_classification DATASET=iitcdip_with_ocr TASK_MODULE=klf4_cfg_and_layout_cond python $BASE_DIR/external/atria/src/atria/core/task_runners/atria_trainer.py \
    hydra.searchpath=[pkg://atria/conf,pkg://docsets/conf,pkg://dp_diffusion/conf] \
    +trainer_configs=layout_cond_latent_diffusion/preextracted_features \
    torch_model_builder@task_module.torch_model_builder.reverse_diffusion_model=local \
    +task_module.torch_model_builder.reverse_diffusion_model.model_name=dp_diffusion.models.openai_models.dp_promise_unet.models.UNetModel \
    +model_config@task_module.torch_model_builder.reverse_diffusion_model=dp_promise_openai_unet_model_64x64_v1 \
    task_module.torch_model_builder.reverse_diffusion_model.in_channels=6 \
    task_module.torch_model_builder.reverse_diffusion_model.out_channels=3 \
    image_size=256 \
    experiment_name=iitcdip_unaug_layout_cond_pretraining_klf4 \
    features_key=klf4-layout \
    training_engine.max_epochs=10 \
