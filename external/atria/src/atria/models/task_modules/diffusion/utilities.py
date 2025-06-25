from functools import partial
import io

import numpy as np
import torch
from atria.core.utilities.logging import get_logger

logger = get_logger(__name__)


def _extract_into_tensor(arr, timesteps, broadcast_shape):
    """
    Extract values from a 1-D numpy array for a batch of indices.

    :param arr: the 1-D numpy array.
    :param timesteps: a tensor of indices into the array to extract.
    :param broadcast_shape: a larger shape of K dimensions with the batch
                            dimension equal to the length of timesteps.
    :return: a tensor of shape [batch_size, 1, ...] where the shape has K dims.
    """
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(arr)
    res = arr[timesteps].float().to(timesteps.device)
    while len(res.shape) < len(broadcast_shape):
        res = res[..., None]
    return res.expand(broadcast_shape)


def _compute_snr(noise_scheduler, timesteps):
    """
    Computes SNR as per https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L847-L849
    """
    alphas_cumprod = noise_scheduler.alphas_cumprod
    sqrt_alphas_cumprod = alphas_cumprod**0.5
    sqrt_one_minus_alphas_cumprod = (1.0 - alphas_cumprod) ** 0.5

    # Expand the tensors.
    # Adapted from https://github.com/TiankaiHang/Min-SNR-Diffusion-Training/blob/521b624bd70c67cee4bdf49225915f5945a872e3/guided_diffusion/gaussian_diffusion.py#L1026
    sqrt_alphas_cumprod = sqrt_alphas_cumprod.to(device=timesteps.device)[
        timesteps
    ].float()
    while len(sqrt_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_alphas_cumprod = sqrt_alphas_cumprod[..., None]
    alpha = sqrt_alphas_cumprod.expand(timesteps.shape)

    sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.to(
        device=timesteps.device
    )[timesteps].float()
    while len(sqrt_one_minus_alphas_cumprod.shape) < len(timesteps.shape):
        sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod[..., None]
    sigma = sqrt_one_minus_alphas_cumprod.expand(timesteps.shape)

    # Compute SNR.
    snr = (alpha / sigma) ** 2
    return snr


def _guidance_wrapper(
    model: torch.nn.Module, guidance_scale: float, use_cfg: bool = False
):
    def forward_with_guidance(*args, rescaled_phi=0.0, **kwargs):
        assert "class_labels" in kwargs, "class_labels must be provided"
        assert hasattr(model, "class_embedding"), "model must have class_embedding"

        model_output = model(*args, **kwargs)
        if hasattr(model_output, "sample"):
            logits = model_output.sample
        else:
            logits = model_output
        if not use_cfg or guidance_scale <= 1:
            return logits

        # print_once(logger, f"Using cfg with guidance scale: {guidance_scale}")
        null_kwargs = {}
        null_kwargs["class_labels"] = (
            model.class_embedding.num_embeddings - 1
        ) * torch.ones_like(
            kwargs["class_labels"], device=kwargs["class_labels"].device
        )
        if "channel_wise_condition" in kwargs:
            null_kwargs["channel_wise_condition"] = kwargs["channel_wise_condition"]
        model_output = model(*args, **null_kwargs)
        if hasattr(model_output, "sample"):
            null_logits = model_output.sample
        else:
            null_logits = model_output
        scaled_logits = null_logits + (logits - null_logits) * guidance_scale

        if rescaled_phi == 0.0:
            return scaled_logits

        std_fn = partial(
            torch.std, dim=tuple(range(1, scaled_logits.ndim)), keepdim=True
        )
        rescaled_logits = scaled_logits * (std_fn(logits) / std_fn(scaled_logits))

        return rescaled_logits * rescaled_phi + scaled_logits * (1.0 - rescaled_phi)

    return forward_with_guidance


def _unnormalize(
    input: torch.Tensor,
    mean: torch.Tensor = torch.tensor([0.5]),
    std: torch.Tensor = torch.tensor([0.5]),
) -> torch.Tensor:
    from torchvision.transforms import Normalize

    return Normalize((-mean / std).tolist(), (1.0 / std).tolist())(input).clamp(0, 1)


def _dropout_label_for_cfg_training(
    class_labels: torch.Tensor,
    num_classes: torch.Tensor,
    probability: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    if class_labels is not None:
        if num_classes is None:
            raise ValueError
        else:
            with torch.no_grad():
                boolean_ = torch.bernoulli(
                    probability * torch.ones_like(class_labels, device=device)
                ).bool()
                no_class_label = num_classes * torch.ones_like(
                    class_labels, device=device
                )
                class_labels = torch.where(boolean_, no_class_label, class_labels)
                return class_labels
    else:
        return None


def _set_scaling_factor(vae, scaling_factor):
    from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

    if isinstance(vae, AutoencoderKL):
        vae.register_to_config(scaling_factor=scaling_factor)
    else:
        vae.scaling_factor = scaling_factor


def _get_scaling_factor(vae):
    from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL

    if isinstance(vae, AutoencoderKL):
        return vae.config.scaling_factor
    else:
        return vae.scaling_factor


def _tensor_image_to_bytes(image: torch.Tensor):
    from PIL import Image

    img_pil = Image.fromarray(
        (image.detach().cpu().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    )
    img_byte_arr = io.BytesIO()
    img_pil.save(img_byte_arr, format="PNG")
    img_bytes = img_byte_arr.getvalue()
    return img_bytes
