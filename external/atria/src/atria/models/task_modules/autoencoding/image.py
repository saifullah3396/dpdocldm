import math
from typing import Optional

import ignite.distributed as idist
import torchvision
from atria.core.utilities.logging import get_logger
from atria.core.utilities.typing import BatchDict
from atria.models.task_modules.autoencoding.base import AutoEncodingModule
from atria.models.task_modules.diffusion.utilities import _unnormalize
from ignite.engine import Engine

logger = get_logger(__name__)


class ImageAutoEncodingModule(AutoEncodingModule):
    _REQUIRES_BUILDER_DICT = False

    def visualization_step(
        self,
        batch: BatchDict,
        evaluation_engine: Optional[Engine] = None,
        training_engine: Optional[Engine] = None,
        **kwargs,
    ) -> None:
        input = self._prepare_input(batch)
        reconstruction = self._model_forward(input)
        global_step = (
            training_engine.state.iteration if training_engine is not None else 1
        )
        if idist.get_rank() == 0:
            # this only saves first batch always if you want you can shuffle validation set and save random batches
            logger.info(
                f"Saving image batch {evaluation_engine.state.iteration} to tensorboard"
            )

            if input.min() < 0 and input.max() > 1.0:
                # assume image is normalized -1 to 1 here
                input = _unnormalize(input)
                reconstruction = _unnormalize(reconstruction)

            # save images to tensorboard
            num_samples = input.shape[0]
            self._tb_logger.writer.add_image(
                f"visualization/input",
                torchvision.utils.make_grid(input, nrow=int(math.sqrt(num_samples))),
                global_step,
            )
            self._tb_logger.writer.add_image(
                f"visualization/reconstruction",
                torchvision.utils.make_grid(
                    reconstruction,
                    nrow=int(math.sqrt(num_samples)),
                ),
                global_step,  # this is iteration of the training engine1
            )
