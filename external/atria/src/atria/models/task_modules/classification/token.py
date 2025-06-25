from functools import partial
from typing import Any, Dict, List, Optional, Union

from atria.core.constants import DataKeys
from atria.core.data.datasets.dataset_metadata import DatasetMetadata
from atria.core.models.model_outputs import ModelOutput, TokenClassificationModelOutput
from atria.core.models.torch_model_builders.base import TorchModelBuilderBase
from atria.core.models.utilities.nn_modules import _convert_label_tensors_to_tags
from atria.core.training.utilities.constants import TrainingStage
from atria.core.utilities.typing import BatchDict
from atria.models.task_modules.classification.base import ClassificationModule
from ignite.handlers import TensorboardLogger


class TokenClassificationModule(ClassificationModule):  # type: ignore
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
        input_stride: int = 0,
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
        self._input_stride = input_stride

    def required_keys_in_batch(self, stage: TrainingStage) -> List[str]:
        required_keys = [
            DataKeys.TOKEN_IDS,
            DataKeys.ATTENTION_MASKS,
        ]
        if self._use_bbox:
            required_keys.append(DataKeys.TOKEN_BBOXES)
        if self._use_image:
            required_keys.append(DataKeys.IMAGE)
        if stage == TrainingStage.train:
            return required_keys + [DataKeys.LABEL]
        elif stage == TrainingStage.validation:
            return required_keys + [DataKeys.WORD_IDS, DataKeys.LABEL]
        elif stage == TrainingStage.test:
            return required_keys + [DataKeys.WORD_IDS, DataKeys.LABEL]
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

    def training_step(self, batch: BatchDict, **kwargs) -> ModelOutput:
        label = batch[DataKeys.LABEL]
        hf_output = self._model_forward(batch)

        # map the labels from the model output to the evaluation labels
        predicted_labels = hf_output.logits.argmax(-1).detach()
        target_labels = label.detach().clone()
        predicted_labels, target_labels = _convert_label_tensors_to_tags(
            class_labels=self._dataset_metadata.labels,
            predicted_labels=predicted_labels,
            target_labels=target_labels,
            ignore_label=-100,
        )
        return TokenClassificationModelOutput(
            loss=hf_output.loss,
            logits=hf_output.logits,
            predicted_labels=predicted_labels,
            target_labels=target_labels,
        )

    def evaluation_step(self, batch: BatchDict, **kwargs) -> ModelOutput:
        label = batch[DataKeys.LABEL]

        # model does not take word_ids as input, so we need to remove it from the batch
        word_ids = batch.pop(DataKeys.WORD_IDS)

        # get the model output
        hf_output = self._model_forward(batch)

        # here we check if the input is strided or not. With strided input tokenization, the first "number of stride"
        # tokens are to be ignored for evaluation as they will be repeated tokens from the previous part of the documnet
        # first we check if there are overflowing samples in the batch and if so for these tokens first N stride tokens
        # are to be ignored for evaluation
        ignore_label = -100
        target_labels = label.clone().detach()
        if self._input_stride > 0:
            for sample_idx, sample_word_ids in enumerate(word_ids):
                # if the minimum word id is greater than 0, then we have an overflowing sample
                # this means this is a continuation of the previous sample and the first N tokens
                if sample_word_ids[sample_word_ids != ignore_label].min() > 0:
                    target_labels[sample_idx][: self._input_stride] = ignore_label

        # map the labels from the model output to the evaluation labels
        predicted_labels = hf_output.logits.argmax(-1)
        predicted_labels, target_labels = _convert_label_tensors_to_tags(
            class_labels=self._dataset_metadata.labels,
            predicted_labels=predicted_labels,
            target_labels=target_labels,
            ignore_label=ignore_label,
        )
        return TokenClassificationModelOutput(
            loss=hf_output.loss,
            logits=hf_output.logits,
            predicted_labels=predicted_labels,
            target_labels=target_labels,
        )

    def predict_step(self, batch: BatchDict, **kwargs) -> ModelOutput:
        hf_output = self._model_forward(batch)
        return TokenClassificationModelOutput(
            logits=hf_output.logits, predicted_labels=hf_output.logits.argmax(-1)
        )
