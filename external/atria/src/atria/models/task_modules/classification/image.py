from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Dict, List, Optional, Union

import torch
from atria.core.constants import DataKeys
from atria.core.data.datasets.dataset_metadata import DatasetMetadata
from atria.core.models.model_outputs import ClassificationModelOutput, ModelOutput
from atria.core.models.task_modules.atria_task_module import (
    CheckpointConfig,
    TorchModelDict,
)
from atria.core.models.torch_model_builders.base import TorchModelBuilderBase
from atria.core.models.utilities.nn_modules import _get_logits_from_output
from atria.core.training.utilities.constants import TrainingStage
from atria.core.utilities.typing import BatchDict
from atria.models.task_modules.classification.base import ClassificationModule
from ignite.contrib.handlers import TensorboardLogger
from timm.data.mixup import Mixup
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy

import torch


@dataclass
class MixupConfig:
    mixup_alpha: float = 1.0
    cutmix_alpha: float = 0.0
    cutmix_minmax: Optional[float] = None
    prob: float = 1.0
    switch_prob: float = 0.5
    mode: str = "batch"
    mixup_prob: float = 1.0
    correct_lam: bool = True
    label_smoothing: float = 0.1


class ImageClassificationModule(ClassificationModule):
    _REQUIRES_BUILDER_DICT = False
    _SUPPORTED_BUILDERS = [
        "LocalTorchModelBuilder",
        "TorchVisionModelBuilder",
        "TransformersModelBuilder",
        "TimmModelBuilder",
    ]

    def __init__(
        self,
        torch_model_builder: Union[
            partial[TorchModelBuilderBase], Dict[str, partial[TorchModelBuilderBase]]
        ],
        checkpoint_configs: Optional[List[CheckpointConfig]] = None,
        dataset_metadata: Optional[DatasetMetadata] = None,
        tb_logger: Optional[TensorboardLogger] = None,
        mixup_config: Optional[MixupConfig] = None,
    ):
        super().__init__(
            torch_model_builder=torch_model_builder,
            checkpoint_configs=checkpoint_configs,
            dataset_metadata=dataset_metadata,
            tb_logger=tb_logger,
        )
        self._mixup_config = mixup_config

    def required_keys_in_batch(self, stage: TrainingStage) -> List[str]:
        if stage == TrainingStage.train:
            return [DataKeys.IMAGE, DataKeys.LABEL]
        elif stage == TrainingStage.validation:
            return [DataKeys.IMAGE, DataKeys.LABEL]
        elif stage == TrainingStage.test:
            return [DataKeys.IMAGE, DataKeys.LABEL]
        elif stage == TrainingStage.predict:
            return [DataKeys.IMAGE]
        else:
            raise ValueError(f"Stage {stage} not supported")

    def _build_model(self) -> Union[torch.nn.Module, TorchModelDict]:
        model = super()._build_model()

        self._mixup: Mixup = None
        if self._mixup_config is not None:
            self._mixup = Mixup(
                num_classes=len(self._dataset_metadata.labels),
                mixup_alpha=self._mixup_config.mixup_alpha,
                cutmix_alpha=self._mixup_config.cutmix_alpha,
                cutmix_minmax=self._mixup_config.cutmix_minmax,
                label_smoothing=self._mixup_config.label_smoothing,
            )
        if self._mixup is not None:
            self._loss_fn_train = (
                LabelSmoothingCrossEntropy(self._mixup.label_smoothing)
                if self._mixup.label_smoothing > 0.0
                else SoftTargetCrossEntropy()
            )
        else:
            self._loss_fn_train = torch.nn.CrossEntropyLoss()
        self._loss_fn_eval = torch.nn.CrossEntropyLoss()

        return model

    def training_step(self, batch: BatchDict, **kwargs) -> ModelOutput:  # get data
        image, label = batch[DataKeys.IMAGE], batch[DataKeys.LABEL]
        # mixup
        if self._mixup is not None:
            image, label = self._mixup(image, label)
        model_output = self._torch_model(image)
        logits = _get_logits_from_output(model_output)
        loss = self._loss_fn_train(logits, label)
        return ClassificationModelOutput(
            loss=loss, logits=logits, label=batch[DataKeys.LABEL]
        )

    def _model_forward(self, batch: Dict[str, torch.Tensor | torch.Any]) -> torch.Any:
        assert (
            self._model_built
        ), "Model must be built before training. Call build_model() first"
        if isinstance(self._torch_model, dict):
            raise NotImplementedError(
                "Model forward must be implemented in the task module when multiple models are used"
            )
        return self._torch_model(batch[DataKeys.IMAGE])
