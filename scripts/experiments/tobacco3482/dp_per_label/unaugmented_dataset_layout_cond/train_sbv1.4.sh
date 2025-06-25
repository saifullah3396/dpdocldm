#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
BASE_DIR=$SCRIPT_DIR/../../../../../
PYTHONPATH=$BASE_DIR/src:$BASE_DIR/external/atria/src:$BASE_DIR/external/docsets/src:$BASE_DIR/external/opacus_dpdm:$PYTHONPATH
MODEL_CHECKPOINT=output/atria_trainer/IitCdip/iitcdip_unaug_layout_cond_pretraining_sbv1.4/checkpoints/checkpoint_319950.pt
if [ -e $MODEL_CHECKPOINT ]; then
    echo "Running training with model checkpoint=$MODEL_CHECKPOINT"
else
    echo "$MODEL_CHECKPOINT does not exist."
    exit 1
fi
for LABEL in {0..9}; do
    PYTHONPATH=$PYTHONPATH TASK=document_classification DATASET=tobacco3482_with_ocr TASK_MODULE=sbv1.4_layout_cond \
        python $BASE_DIR/external/atria/src/atria/core/task_runners/atria_trainer.py \
        hydra.searchpath=[pkg://atria/conf,pkg://docsets/conf,pkg://dp_diffusion/conf] \
        +trainer_configs=dp_latent_diffusion/preextracted_features \
        torch_model_builder@task_module.torch_model_builder.reverse_diffusion_model=local \
        +task_module.torch_model_builder.reverse_diffusion_model.model_name=dp_diffusion.models.openai_models.dp_promise_unet.models.UNetModel \
        +model_config@task_module.torch_model_builder.reverse_diffusion_model=dp_promise_openai_unet_model_32x32_v1 \
        task_module.torch_model_builder.reverse_diffusion_model.in_channels=8 \
        task_module.torch_model_builder.reverse_diffusion_model.out_channels=4 \
        model_checkpoint=$MODEL_CHECKPOINT \
        image_size=256 \
        experiment_name=dp_unaug_layout_cond_sbv1.4_cfg_per_label \
        features_key=stable-diffusion-v1-4-layout \
        data_module.max_test_samples=1000 \
        visualization_engine.visualize_every_n_epochs=25 \
        train_batch_size=64 \
        max_physical_batch_size=16 \
        noise_multiplicity=32 \
        test_engine.test_guidance_scales=[1] \
        training_engine.max_epochs=250 \
        training_engine.target_delta=null \
        training_engine.optimizers.lr=1e-4 \
        data_module.subset_label=$LABEL
done