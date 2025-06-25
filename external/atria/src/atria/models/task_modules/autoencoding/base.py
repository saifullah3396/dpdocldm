from functools import partial
from typing import Dict, List, Optional, Union

import torch
from atria.core.constants import DataKeys
from atria.core.data.datasets.dataset_metadata import DatasetMetadata
from atria.core.models.model_outputs import AutoEncoderModelOutput, ModelOutput
from atria.core.models.task_modules.atria_task_module import (
    AtriaTaskModule,
    CheckpointConfig,
)
from atria.core.models.torch_model_builders.base import TorchModelBuilderBase
from atria.core.training.utilities.constants import TrainingStage
from atria.core.utilities.logging import get_logger
from atria.core.utilities.typing import BatchDict
from ignite.contrib.handlers import TensorboardLogger

logger = get_logger(__name__)


class AutoEncodingModule(AtriaTaskModule):
    _REQUIRES_BUILDER_DICT = False
    _SUPPORTED_BUILDERS = [
        "LocalTorchModelBuilder",
        "TorchVisionModelBuilder",
        "TransformersModelBuilder",
        "TimmModelBuilder",
        "DiffusersModelBuilder",
    ]

    def __init__(
        self,
        torch_model_builder: Union[
            partial[TorchModelBuilderBase], Dict[str, partial[TorchModelBuilderBase]]
        ],
        checkpoint_configs: Optional[List[CheckpointConfig]] = None,
        dataset_metadata: Optional[DatasetMetadata] = None,
        tb_logger: Optional[TensorboardLogger] = None,
        input_key: str = DataKeys.IMAGE,
        loss_type: str = "l2",
    ):
        super().__init__(
            torch_model_builder=torch_model_builder,
            checkpoint_configs=checkpoint_configs,
            dataset_metadata=dataset_metadata,
            tb_logger=tb_logger,
        )
        self._input_key = input_key
        self._loss_type = loss_type

    def _build_model(
        self,
    ) -> torch.nn.Module:
        encoder_decoder_model: torch.nn.Module = super()._build_model()

        # initialize classification related stuff
        if self._loss_type == "l2":
            self._loss_fn = torch.nn.MSELoss()
        else:
            raise NotImplementedError()

        return encoder_decoder_model

    def _prepare_input(self, batch: BatchDict) -> torch.Tensor:
        return batch[self._input_key]

    def required_keys_in_batch(self, stage: TrainingStage) -> List[str]:
        if stage == TrainingStage.train:
            return [self._input_key]
        elif stage == TrainingStage.validation:
            return [self._input_key]
        elif stage == TrainingStage.test:
            return [self._input_key]
        elif stage == TrainingStage.predict:
            return []
        else:
            raise ValueError(f"Stage {stage} not supported")

    def training_step(self, batch: BatchDict, **kwargs) -> ModelOutput:
        input = self._prepare_input(batch)
        reconstruction = self._model_forward(input)
        loss = self._loss_fn(input=reconstruction, target=input)
        return AutoEncoderModelOutput(
            loss=loss, real=input, reconstructed=reconstruction
        )

    def evaluation_step(self, batch: BatchDict, **kwargs) -> ModelOutput:
        input = self._prepare_input(batch)
        reconstruction = self._model_forward(input)
        loss = self._loss_fn(input=reconstruction, target=input)
        return AutoEncoderModelOutput(
            loss=loss, real=input, reconstructed=reconstruction
        )

    def predict_step(self, batch: BatchDict, **kwargs) -> ModelOutput:
        input = self._prepare_input(batch)
        reconstruction = self._model_forward(input)
        loss = self._loss_fn(input=reconstruction, target=input)
        return AutoEncoderModelOutput(
            loss=loss, real=input, reconstructed=reconstruction
        )
