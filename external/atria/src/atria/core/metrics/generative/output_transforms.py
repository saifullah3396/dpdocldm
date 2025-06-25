from typing import Union

import torch
from atria.core.models.model_outputs import (
    AutoEncoderModelOutput,
    DiffusionModelOutput,
    VarAutoEncoderGANModelOutput,
    VarAutoEncoderModelOutput,
)
from atria.core.utilities.logging import get_logger, warn_once
from atria.models.task_modules.diffusion.utilities import _unnormalize

logger = get_logger(__name__)


def _format_shape_and_scale_for_fid(x: torch.Tensor):
    # convert grayscale to rgb if necessary
    if x.shape[1] == 1:
        x = x.repeat(1, 3, 1, 1)

    # if the image is in range -1 to 1 we convert it to 0 to 1
    if x.min() < 0 or x.max() > 1:
        warn_once(
            logger,
            f"WrapperInceptionV3 for FID computation assumes an input image is in the range 0 to 1. "
            f"Converting it to 0 to 1. Actual range = [{x.min()}, {x.max()}]",
        )
        x = _unnormalize(x)

    # inception model inputs must be images in range 0 to 1
    assert (
        x.min() >= 0.0 and x.max() <= 1.0
    ), f"Input image must be in range 0 to 1. min={x.min()}, min={x.max()}"

    return x


def _fid_output_transform(
    output: Union[
        AutoEncoderModelOutput,
        VarAutoEncoderModelOutput,
        VarAutoEncoderGANModelOutput,
        DiffusionModelOutput,
    ],
    use_reconstruction: bool = False,
):
    output_classes = (
        AutoEncoderModelOutput,
        VarAutoEncoderModelOutput,
        VarAutoEncoderGANModelOutput,
        DiffusionModelOutput,
    )
    assert isinstance(
        output,
        output_classes,
    ), f"Expected one of {output_classes}, got {type(output)}"

    real = output.real
    generated = output.reconstructed if use_reconstruction else output.generated
    logger.debug("Calculating FID with the following image stats:")
    logger.debug(
        f"Real shape: {real.shape}, Real min: {real.min()}, Real max: {real.max()}"
    )
    logger.debug(f"Real mean: {real.mean()}, Real std: {real.std()}")
    logger.debug(f"Real dtype: {real.dtype}")
    logger.debug(
        f"Generated shape: {generated.shape}, Generated min: {generated.min()}, Generated max: {generated.max()}"
    )
    logger.debug(
        f"Generated mean: {generated.mean()}, Generated std: {generated.std()}"
    )
    logger.debug(f"Generated dtype: {generated.dtype}")

    real = _format_shape_and_scale_for_fid(real)
    generated = _format_shape_and_scale_for_fid(generated)
    return generated, real
