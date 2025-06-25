#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
BASE_DIR=$SCRIPT_DIR/../../../../../
PYTHONPATH=$BASE_DIR/src:$BASE_DIR/external/atria/src:$BASE_DIR/external/docsets/src:$BASE_DIR/external/opacus_dpdm:$PYTHONPATH
MODEL_CHECKPOINT=output/atria_trainer/IitCdip/iitcdip_unaug_pretraining_klf4/checkpoints/checkpoint_1279790.pt
if [ -e $MODEL_CHECKPOINT ]; then
    echo "Running training with model checkpoint=$MODEL_CHECKPOINT"
else
    echo "$MODEL_CHECKPOINT does not exist."
    exit 1
fi
for LABEL in {0..15}; do
    PYTHONPATH=$PYTHONPATH TASK=document_classification DATASET=rvlcdip TASK_MODULE=klf4_uncond \
        python $BASE_DIR/src/dp_diffusion/task_runners/dp_promise_trainer.py \
        hydra.searchpath=[pkg://atria/conf,pkg://docsets/conf,pkg://dp_diffusion/conf] \
        +trainer_configs=dp_promise_latent_diffusion/preextracted_features \
        torch_model_builder@task_module.torch_model_builder.reverse_diffusion_model=local \
        +task_module.torch_model_builder.reverse_diffusion_model.model_name=dp_diffusion.models.openai_models.dp_promise_unet.models.UNetModel \
        +model_config@task_module.torch_model_builder.reverse_diffusion_model=dp_promise_openai_unet_model_64x64_v1 \
        task_module.torch_model_builder.reverse_diffusion_model.in_channels=3 \
        task_module.torch_model_builder.reverse_diffusion_model.out_channels=3 \
        model_checkpoint=$MODEL_CHECKPOINT \
        image_size=256 \
        experiment_name=rvlcdip_dp_promise_unaug_klf4_cfg_per_label \
        features_key=klf4_pretrained_iitcidp_133000 \
        data_module.subset_label=$LABEL \
        data_module.max_test_samples=10000 \
        vae_checkpoint=pretrained_models/klf4_pretrained_iitcidp_133000.pt \
        max_physical_batch_size=192 \
        private_training_engine.cutoff_ratio=0.95 \
        private_training_engine.noise_multiplier=1.5 \
        visualization_engine.visualize_every_n_epochs=50
done