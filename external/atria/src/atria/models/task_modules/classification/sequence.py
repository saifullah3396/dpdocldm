from __future__ import annotations

from functools import partial
from typing import Any, Dict, List, Optional, Union

from ignite.contrib.handlers import TensorboardLogger

from atria.core.constants import DataKeys
from atria.core.data.datasets.dataset_metadata import DatasetMetadata
from atria.core.models.model_outputs import ClassificationModelOutput
from atria.core.models.torch_model_builders.base import TorchModelBuilderBase
from atria.core.training.utilities.constants import TrainingStage
from atria.core.utilities.typing import BatchDict
from atria.models.task_modules.classification.base import ClassificationModule


class SequenceClassificationModule(ClassificationModule):
    _REQUIRES_BUILDER_DICT = False
    _SUPPORTED_BUILDERS = [
        "LocalTorchModelBuilder",
        "TransformersModelBuilder",
    ]

    def __init__(
        self,
        torch_model_builder: Union[
            partial[TorchModelBuilderBase], Dict[str, partial[TorchModelBuilderBase]]
        ],
        checkpoint: Optional[str] = None,
        load_checkpoint_strict: bool = False,
        dataset_metadata: Optional[DatasetMetadata] = None,
        tb_logger: Optional[TensorboardLogger] = None,
        use_bbox: bool = True,
        use_image: bool = True,
    ):
        super().__init__(
            torch_model_builder=torch_model_builder,
            checkpoint=checkpoint,
            load_checkpoint_strict=load_checkpoint_strict,
            dataset_metadata=dataset_metadata,
            tb_logger=tb_logger,
        )
        self._use_bbox = use_bbox
        self._use_image = use_image

    def required_keys_in_batch(self, stage: TrainingStage) -> List[str]:
        required_keys = [DataKeys.TOKEN_IDS, DataKeys.ATTENTION_MASKS]
        if self._use_bbox:
            required_keys.append(DataKeys.TOKEN_BBOXES)
        if self._use_image:
            required_keys.append(DataKeys.IMAGE)
        if stage == TrainingStage.train:
            return required_keys + [DataKeys.LABEL]
        elif stage == TrainingStage.validation:
            return required_keys + [DataKeys.LABEL]
        elif stage == TrainingStage.test:
            return required_keys + [DataKeys.LABEL]
        elif stage == TrainingStage.predict:
            return required_keys
        else:
            raise ValueError(f"Stage {stage} not supported")

    def _model_forward(self, batch: BatchDict) -> Any:
        if self._use_bbox:
            batch["bbox"] = batch[DataKeys.TOKEN_BBOXES]
        if self._use_image:
            batch["pixel_values"] = batch[DataKeys.IMAGE]
        if DataKeys.LABEL in batch:
            batch["labels"] = batch[DataKeys.LABEL]

        filtered_batch = self._filter_batch_keys_for_model_forward(batch)
        self._validate_batch_keys_for_model_forward(filtered_batch)
        return self._torch_model(**filtered_batch)

    def training_step(self, batch: BatchDict, **kwargs) -> ClassificationModelOutput:
        hf_output = self._model_forward(batch)
        return ClassificationModelOutput(
            loss=hf_output.loss, logits=hf_output.logits, label=batch[DataKeys.LABEL]
        )

    def evaluation_step(self, batch: BatchDict, **kwargs) -> ClassificationModelOutput:
        hf_output = self._model_forward(batch)
        return ClassificationModelOutput(
            loss=hf_output.loss, logits=hf_output.logits, label=batch[DataKeys.LABEL]
        )

    def predict_step(self, batch: BatchDict, **kwargs) -> ClassificationModelOutput:
        hf_output = self._model_forward(batch)
        return ClassificationModelOutput(
            loss=hf_output.loss,
            logits=hf_output.logits,
            prediction=hf_output.logits.argmax(dim=-1),
        )
