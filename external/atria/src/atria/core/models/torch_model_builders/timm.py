import json
from typing import List, Optional

import hydra_zen
import timm
from atria.core.models.tasks import ModelTasks
from atria.core.models.torch_model_builders.base import TorchModelBuilderBase
from atria.core.models.torch_model_builders.utilities import (
    ImageModel,
)
from atria.core.utilities.logging import get_logger
from torch import nn

logger = get_logger(__name__)


class TimmModelBuilder(TorchModelBuilderBase):
    def __init__(
        self,
        model_name: str = hydra_zen.MISSING,
        model_task: ModelTasks = ModelTasks.image_classification,
        model_cache_dir: Optional[str] = None,
        pretrained: bool = True,
        convert_bn_to_gn: bool = False,
        is_frozen: bool = False,
        required_in_training_step: bool = True,
        pretrained_checkpoint: Optional[str] = None,
        strict: bool = True,
        frozen_keys_patterns: Optional[List[str]] = None,
        unfrozen_keys_patterns: Optional[List[str]] = None,
        **model_config,
    ):
        super().__init__(
            model_name=model_name,
            model_task=model_task,
            model_cache_dir=model_cache_dir,
            pretrained=pretrained,
            convert_bn_to_gn=convert_bn_to_gn,
            is_frozen=is_frozen,
            required_in_training_step=required_in_training_step,
            pretrained_checkpoint=pretrained_checkpoint,
            strict=strict,
            frozen_keys_patterns=frozen_keys_patterns,
            unfrozen_keys_patterns=unfrozen_keys_patterns,
            **model_config,
        )

    def _build(self) -> nn.Module:
        """
        Initialize the timm model with the provided arguments.

        Returns:
            nn.Module: The initialized timm model wrapped in TimmModel.
        """
        assert self._model_task in [
            ModelTasks.image_classification,
        ], f"Task {self._model_task} not supported for {self.__class__.__name__}."

        kwargs = {}
        kwargs["model_name"] = self._model_name
        kwargs["pretrained"] = self._pretrained
        kwargs = {**kwargs, **self._model_config}

        if "num_labels" in kwargs:
            kwargs["num_classes"] = kwargs.pop("num_labels")

        logger.info(
            f"Initializing timm model [{self._model_name}] with the following configuration:\n{json.dumps(kwargs, indent=4)}",
        )
        return ImageModel(
            timm.create_model(
                **kwargs,
            )
        )
