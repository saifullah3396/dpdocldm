import math
from typing import Optional

import ignite.distributed as idist
import torch
import torchvision
from atria.core.constants import DataKeys
from atria.core.utilities.logging import get_logger
from atria.core.utilities.typing import BatchDict
from atria.models.task_modules.diffusion.utilities import _unnormalize
from atria.models.task_modules.var_autoencoding.base import VarAutoEncodingModule
from ignite.engine import Engine

logger = get_logger(__name__)


class ImageVarAutoEncodingModule(VarAutoEncodingModule):
    _REQUIRES_BUILDER_DICT = False

    def visualization_step(
        self,
        batch: BatchDict,
        evaluation_engine: Optional[Engine] = None,
        training_engine: Optional[Engine] = None,
        **kwargs,
    ) -> None:
        image = self._prepare_input(batch)
        reconstruction, posterior = self._model_forward(image)
        generated_samples = self.decode(torch.randn_like(posterior.sample()))
        global_step = (
            training_engine.state.iteration if training_engine is not None else 1
        )
        # step
        global_step = (
            training_engine.state.iteration if training_engine is not None else 1
        )
        if idist.get_rank() == 0:
            # this only saves first batch always if you want you can shuffle validation set and save random batches
            logger.info(
                f"Saving image batch {evaluation_engine.state.iteration} to tensorboard"
            )
            if input.min() < 0 and input.max() > 1.0:
                image = _unnormalize(image)
                generated_samples = _unnormalize(generated_samples)
                reconstruction = _unnormalize(reconstruction)

            # save images to tensorboard
            num_samples = batch[self._input_key].shape[0]
            self._tb_logger.writer.add_image(
                f"{self._input_key}",
                torchvision.utils.make_grid(image, nrow=int(math.sqrt(num_samples))),
                global_step,
            )
            self._tb_logger.writer.add_image(
                f"{DataKeys.GEN_SAMPLES}_{self._input_key}",
                torchvision.utils.make_grid(
                    generated_samples,
                    nrow=int(math.sqrt(num_samples)),
                ),
                global_step,
            )
            self._tb_logger.writer.add_image(
                f"/{DataKeys.RECONS}_{self._input_key}",
                torchvision.utils.make_grid(
                    reconstruction,
                    nrow=int(math.sqrt(num_samples)),
                ),
                global_step,  # this is iteration of the training engine1
            )
