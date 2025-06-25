from torch import nn

from atria.core.utilities.common import _get_possible_args
from atria.core.utilities.logging import get_logger

logger = get_logger(__name__)


class ImageModel(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image, **kwargs):
        return self.model(image, **kwargs)


def _filter_args_for_class(cls, config):
    # filter out unnecessary parameters
    possible_args = _get_possible_args(cls)
    leftover_args = set(config.keys()) - set(possible_args)
    if len(leftover_args) > 0:
        logger.warning(
            f"Following parameters are not used in the model initialization: {leftover_args}"
        )
    return {k: v for k, v in config.items() if k in possible_args}
