from __future__ import annotations
from typing import List, Optional

import hydra_zen
import torch
from atria.core.models.tasks import ModelTasks
from atria.core.models.torch_model_builders.base import TorchModelBuilderBase
from atria.core.utilities.common import _get_possible_args, _resolve_module_from_path
from atria.core.utilities.logging import get_logger
from diffusers.models import AutoencoderKL

logger = get_logger(__name__)


class DiffusersModelBuilder(TorchModelBuilderBase):
    """
    A model constructor for Diffusers models, specifically for generation tasks.
    """

    def __init__(
        self,
        model_name: str = hydra_zen.MISSING,
        model_task: ModelTasks = ModelTasks.diffusion,
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

    def _validate_params(self):
        assert self._model_task in [
            ModelTasks.autoencoding,
            ModelTasks.image_generation,
            ModelTasks.diffusion,
        ], f"Task {self._model_task} not supported for {self.__class__.__name__}."

        if self._model_task == ModelTasks.autoencoding:
            possible_choices = [
                "AutoEncoderKL",
            ]
            assert (
                self._model_name in possible_choices
            ), f"Model class {self._model_name} not supported for {self._model_task} in diffusers. Possible choices: {possible_choices}"
        elif self._model_task == ModelTasks.diffusion:
            possible_choices = [
                "UNet2DModel",
                "UNet2DConditionModel",
                "AutoencoderKL",
            ]
            assert (
                self._model_name in possible_choices
            ), f"Model class {self._model_name} not supported for {self._model_task} in diffusers. Possible choices: {possible_choices}"
        else:
            raise ValueError(f"Task {self._model_task} not supported.")

    def _build(self) -> torch.nn.Module:
        """
        Initialize the model based on the specified task and whether it is pretrained.

        Returns:
            torch.nn.Module: The initialized model.

        Raises:
            ValueError: If the model task is not supported.
        """
        self._validate_params()

        # Initialize the model
        model_config_name_or_path = self._model_config.pop(
            "model_config_name_or_path", None
        )

        try:
            model_class = _resolve_module_from_path(
                ".".join(["diffusers", self._model_name])
            )
        except Exception as e:
            logger.exception(f"Error loading model class {self._model_name}: {e}")

        # filter out unnecessary parameters
        possible_args = _get_possible_args(model_class)
        leftover_args = set(self._model_config.keys()) - set(possible_args)
        if len(leftover_args) > 0:
            logger.warning(
                f"Following parameters are not used in the model initialization: {leftover_args}"
            )
        self._model_config = {
            k: v for k, v in self._model_config.items() if k in possible_args
        }

        logger.info(
            f"Initializing model [{model_class}] with the following parameters:"
        )
        logger.info(self._model_config)
        if model_config_name_or_path is not None:
            if self._pretrained:
                additional_kwargs = {}
                if issubclass(model_class, AutoencoderKL):
                    additional_kwargs = dict(
                        subfolder="vae",
                    )
                model = model_class.from_pretrained(
                    model_config_name_or_path,
                    cache_dir=self._model_cache_dir,
                    **self._model_config,
                    **additional_kwargs,
                )
            else:
                config = model_class.load_config(model_config_name_or_path)
                model = model_class.from_config(config)
        else:
            model = model_class(
                **self._model_config,
            )

        return model
