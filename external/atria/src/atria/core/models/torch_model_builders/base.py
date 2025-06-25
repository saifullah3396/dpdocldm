import os
from abc import abstractmethod
from typing import List, Optional

import hydra_zen
import torch
from atria.core.data.data_modules.dataset_cacher.dataset_cacher import ATRIA_CACHE_DIR
from atria.core.models.tasks import ModelTasks
from atria.core.models.utilities.nn_modules import _batch_norm_to_group_norm
from atria.core.utilities.logging import get_logger

logger = get_logger(__name__)


class TorchModelBuilderBase:
    """
    TorchModelLoader is a dataclass that serves as a base class for constructing
    PyTorch models with optional pretrained weights and checkpoints.

    Attributes:
        model_name (str): The name of the model to be constructed.
        model_task (ModelTasks): The task for which the model is intended.
        model_cache_dir (str): The directory where the model weights are stored.
        pretrained (bool): Whether to load pretrained weights.
        convert_bn_to_gn (bool): Whether to convert BatchNorm layers to GroupNorm layers.
        is_frozen (bool): Whether to freeze the model.
        required_in_training_step (bool): Whether the model is required in the training step.
        model_config (Dict[str, Any]): Additional model configuration.
        pretrained_checkpoint (Optional[str]): The path to the model checkpoint.
        strict (bool): Whether to strictly enforce the model checkpoint keys.
        frozen_keys_patterns (Optional[List[str]]): Patterns to freeze model layers.
        unfrozen_keys_patterns (Optional[List[str]]): Patterns to unfreeze model layers.

    """

    def __init__(
        self,
        model_name: str = hydra_zen.MISSING,
        model_task: ModelTasks = hydra_zen.MISSING,
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
        self._model_name = model_name
        self._model_task = model_task
        self._model_config = model_config if model_config is not None else {}
        self._model_cache_dir = model_cache_dir or os.path.join(
            ATRIA_CACHE_DIR, "models"
        )
        self._pretrained = pretrained
        self._convert_bn_to_gn = convert_bn_to_gn
        self._is_frozen = is_frozen
        self._required_in_training_step = required_in_training_step
        self._pretrained_checkpoint = pretrained_checkpoint
        self._strict = strict
        self._frozen_keys_patterns = frozen_keys_patterns
        self._unfrozen_keys_patterns = unfrozen_keys_patterns

    @property
    def model_name(self) -> str:
        return self._model_name

    @property
    def is_frozen(self) -> bool:
        return self._is_frozen

    @property
    def required_in_training_step(self) -> bool:
        return self._required_in_training_step

    def build(
        self,
        key: Optional[str] = None,
    ) -> torch.nn.Module:
        model = self._build()

        # this is internal checkpoiont path directly for any loaded model
        if self._pretrained and self._pretrained_checkpoint is not None:
            logger.info(
                f"Loading pretrained weights from [{self._pretrained_checkpoint}]"
            )
            checkpoint = torch.load(self._pretrained_checkpoint, map_location="cpu")
            if "state_dict" in checkpoint:
                checkpoint = checkpoint["state_dict"]
            if "model" in checkpoint:
                checkpoint = checkpoint["model"]
            checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
            checkpoint = {k.replace("vae.", ""): v for k, v in checkpoint.items()}
            keys = model.load_state_dict(checkpoint, strict=self._strict)
            if not self._strict:
                if keys.missing_keys:
                    logger.warning(
                        f"Found keys that are in the model state dict but not in the checkpoint: {keys.missing_keys}"
                    )
                if keys.unexpected_keys:
                    logger.warning(
                        f"Found keys that are not in the model state dict but in the checkpoint: {keys.unexpected_keys}"
                    )

        # replace batch norm with group norm if required
        if self._convert_bn_to_gn:
            if key is not None:
                logger.warning(
                    f"Converting BatchNorm layers to GroupNorm layers in the model assigned to [{key}]. "
                    "If this is not intended, set convert_bn_to_gn=False."
                )
            else:
                logger.warning(
                    "Converting BatchNorm layers to GroupNorm layers in the model. "
                    "If this is not intended, set convert_bn_to_gn=False."
                )
            _batch_norm_to_group_norm(model)

        # freeze the model if required
        if self._is_frozen:
            if key is not None:
                logger.warning(
                    f"Freezing the model assigned to [{key}]. If this is not intended, set is_frozen=False in its config."
                )
            else:
                logger.warning(
                    "Freezing the model. If this is not intended, set is_frozen=False in its config."
                )
            model.requires_grad_(False)

        if self._frozen_keys_patterns is not None:
            logger.info(
                f"Freezing model layers based on patterns: {self._frozen_keys_patterns}"
            )
            logger.info(f"Model layers: {[k for k, v in model.named_parameters()]}")
            for name, param in model.named_parameters():
                if self._frozen_keys_patterns is not None:
                    for pattern in self._frozen_keys_patterns:
                        if pattern in name:
                            param.requires_grad = False
                            break
                if self._unfrozen_keys_patterns is not None:
                    for pattern in self._unfrozen_keys_patterns:
                        if pattern in name:
                            param.requires_grad = True
                            break

            trainable_params = {
                name: param.requires_grad
                for name, param in model.named_parameters()
                if param.requires_grad
            }
            logger.info(f"Current trainable parameters: {trainable_params}")

        return model

    @abstractmethod
    def _build(self) -> torch.nn.Module:
        """
        Abstract method to initialize the model. Must be implemented by subclasses.

        Returns:
            torch.nn.Module: The initialized PyTorch model.
        """
