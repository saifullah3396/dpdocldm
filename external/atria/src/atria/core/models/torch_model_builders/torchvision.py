import os
from typing import List, Optional

import hydra_zen
import torch
from atria.core.models.tasks import ModelTasks
from atria.core.models.torch_model_builders.base import TorchModelBuilderBase
from atria.core.models.torch_model_builders.utilities import ImageModel
from atria.core.utilities.logging import get_logger
from atria.core.utilities.nn_modules import _get_last_module, _replace_module_with_name
from torch import nn

logger = get_logger(__name__)


class TorchVisionModelBuilder(TorchModelBuilderBase):
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
        assert self._model_task in [
            ModelTasks.image_classification,
        ], f"Task {self._model_task} not supported for {self.__class__.__name__}."

        # Remove num_labels from model_initialization_args if it exists and save it
        num_labels: Optional[int] = self._model_config.pop("num_labels", None)

        # Set the cache directory for TorchVision models
        os.environ["TORCH_HOME"] = self._model_cache_dir

        # Load the model from TorchVision hub
        model: nn.Module = torch.hub.load(
            "pytorch/vision:v0.10.0",
            self._model_name,
            pretrained=self._pretrained,
            verbose=False,
            **self._model_config,
        )

        # Replace the classification head if num_labels is provided
        if num_labels is not None:
            name, module = _get_last_module(model)
            _replace_module_with_name(
                model, name, nn.Linear(module.in_features, num_labels)
            )
        else:
            logger.warning(
                "No 'num_labels' in 'model_initialization_kwargs' provided. Classification head will not be replaced."
            )

        # Wrap the model to ensure consistent input naming
        return ImageModel(model)
