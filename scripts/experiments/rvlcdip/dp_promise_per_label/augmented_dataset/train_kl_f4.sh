#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
BASE_DIR=$SCRIPT_DIR/../../../../../
PYTHONPATH=$BASE_DIR/src:$BASE_DIR/external/atria/src:$BASE_DIR/external/docsets/src:$BASE_DIR/external/opacus_dpdm:$PYTHONPATH
MODEL_CHECKPOINT=output/atria_trainer/IitCdip/iitcdip_aug_pretraining_klf4/checkpoints/checkpoint_1279790.pt
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
        experiment_name=rvlcdip_dp_promise_aug_klf4_cfg_per_label \
        features_key=klf4_pretrained_iitcidp_133000 \
        data_module.subset_label=$LABEL \
        data_module.max_test_samples=10000  \
        vae_checkpoint=pretrained_models/klf4_pretrained_iitcidp_133000.pt \
        max_physical_batch_size=192 \
        private_training_engine.cutoff_ratio=0.95 \
        private_training_engine.noise_multiplier=1.5 \
        visualization_engine.visualize_every_n_epochs=50
done
# 2025-02-20 14:01:58 serv-3318 dp_diffusion.engines.dp_promise_training[4116316] INFO Computing epsilon with parameters:
# 2025-02-20 14:01:58 serv-3318 dp_diffusion.engines.dp_promise_training[4116316] INFO prob1: 0.012658227848101266
# 2025-02-20 14:01:58 serv-3318 dp_diffusion.engines.dp_promise_training[4116316] INFO prob2: 0.05263157894736842
# 2025-02-20 14:01:58 serv-3318 dp_diffusion.engines.dp_promise_training[4116316] INFO niter1: 79
# 2025-02-20 14:01:58 serv-3318 dp_diffusion.engines.dp_promise_training[4116316] INFO niter2: 950
# 2025-02-20 14:01:58 serv-3318 dp_diffusion.engines.dp_promise_training[4116316] INFO sigma: 2.19
# 2025-02-20 14:01:58 serv-3318 dp_diffusion.engines.dp_promise_training[4116316] INFO alpha_cumprod_S: 0.0002752058790065348
# 2025-02-20 14:01:58 serv-3318 dp_diffusion.engines.dp_promise_training[4116316] INFO input_dim: 12288
# 2025-02-20 14:01:58 serv-3318 dp_diffusion.engines.dp_promise_training[4116316] INFO delta: 3.125e-06