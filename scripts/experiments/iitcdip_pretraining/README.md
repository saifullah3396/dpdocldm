# Experiment scripts
1. [Latent Diffusion Pretraining on IIT-CDIP Augmented Images with KL-F4 64x64x3 Latent Representations](train_unconditional_image_ldm_kl_f4_256_augmented.sh)
2. [Latent Diffusion Pretraining on IIT-CDIP Augmented Images with Stable Diffusion V1-4 32x32x4 Latent Representations](train_unconditional_image_ldm_kl_f8_256_augmented.sh)

Currently these scripts are only used for following:
- Generating preprocessed IIT-CDIP images to (1000x1000) or (256x256) sizes
- Generating and storing latent features
- Plan is to use this for pretraining models as done with DP-Promise based code but at the moment
  features are generated here and then used for pretraining diffusion model on latent features using DP-Promise code
- Once features are generated, we will try to pretrain a diffusion model for 64x64x3 KL-F4 representation using this code
  as well as DP-Promise code to see whether they bring similar results. A single model for starters