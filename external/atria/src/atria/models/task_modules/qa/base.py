from __future__ import annotations

from functools import partial
from typing import Any, Dict, List, Optional, Union

from atria.core.constants import DataKeys
from atria.core.data.datasets.dataset_metadata import DatasetMetadata
from atria.core.models.model_outputs import ModelOutput, SequenceQAModelOutput
from atria.core.models.task_modules.atria_task_module import (
    AtriaTaskModule,
    CheckpointConfig,
)
from atria.core.models.torch_model_builders.base import TorchModelBuilderBase
from atria.core.training.utilities.constants import TrainingStage
from atria.core.utilities.logging import get_logger
from atria.core.utilities.typing import BatchDict
from ignite.handlers import TensorboardLogger

logger = get_logger(__name__)


class QuestionAnsweringModule(AtriaTaskModule):
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
        checkpoint_configs: Optional[List[CheckpointConfig]] = None,
        dataset_metadata: Optional[DatasetMetadata] = None,
        tb_logger: Optional[TensorboardLogger] = None,
        use_bbox: bool = True,
        use_image: bool = True,
        input_stride: int = 0,
    ):
        super().__init__(
            torch_model_builder=torch_model_builder,
            checkpoint_configs=checkpoint_configs,
            dataset_metadata=dataset_metadata,
            tb_logger=tb_logger,
        )
        self._use_bbox = use_bbox
        self._use_image = use_image
        self._input_stride = input_stride

    def required_keys_in_batch(self, stage: TrainingStage) -> List[str]:
        required_keys = [
            DataKeys.WORDS,
            DataKeys.WORD_IDS,
            DataKeys.SEQUENCE_IDS,
            DataKeys.QUESTION_ID,
            DataKeys.TOKEN_IDS,
            DataKeys.ATTENTION_MASKS,
            DataKeys.GOLD_ANSWERS,
        ]
        if self._use_bbox:
            required_keys.append(DataKeys.TOKEN_BBOXES)
        if self._use_image:
            required_keys.append(DataKeys.IMAGE)
        if stage == TrainingStage.train:
            return required_keys + [
                DataKeys.START_TOKEN_IDX,
                DataKeys.END_TOKEN_IDX,
            ]
        elif stage == TrainingStage.validation:
            return required_keys + [
                DataKeys.START_TOKEN_IDX,
                DataKeys.END_TOKEN_IDX,
            ]
        elif stage == TrainingStage.test:
            return required_keys
        elif stage == TrainingStage.predict:
            return required_keys
        else:
            raise ValueError(f"Stage {stage} not supported")

    def _model_forward(self, batch: BatchDict) -> Any:
        batch["start_positions"] = batch[DataKeys.START_TOKEN_IDX]
        batch["end_positions"] = batch[DataKeys.END_TOKEN_IDX]

        if self._use_bbox:
            batch["bbox"] = batch[DataKeys.TOKEN_BBOXES]
        if self._use_image:
            batch["pixel_values"] = batch[DataKeys.IMAGE]

        filtered_batch = self._filter_batch_keys_for_model_forward(batch)
        self._validate_batch_keys_for_model_forward(filtered_batch)
        return self._torch_model(**filtered_batch)

    def training_step(self, batch: BatchDict, **kwargs) -> ModelOutput:
        # draw bounding boxes on image for debugging
        # for bboxes, image in zip(
        #     input["bbox"], input["pixel_values"]
        # ):  # each box is [x1,y1,x2,y2] normalized
        #     import cv2
        #     import numpy as np

        #     image = np.array(image.detach().cpu().permute(1, 2, 0).numpy()).copy()
        #     print("image", image.shape)
        #     h, w, c = image.shape
        #     for box in bboxes:
        #         p1 = (int(box[0] / 1000.0 * w), int(box[1] / 1000.0 * h))
        #         p2 = (int(box[2] / 1000.0 * w), int(box[3] / 1000.0 * h))
        #         cv2.rectangle(image, p1, p2, (255, 0, 0), 1)
        #     import matplotlib.pyplot as plt

        #     plt.imshow(image)
        #     plt.show()

        # compute logits
        hf_outputs = self._model_forward(batch)
        return SequenceQAModelOutput(
            loss=hf_outputs.loss,
            start_logits=hf_outputs.start_logits,
            end_logits=hf_outputs.end_logits,
            words=batch[DataKeys.WORDS],
            word_ids=batch[DataKeys.WORD_IDS],
            sequence_ids=batch[DataKeys.SEQUENCE_IDS],
            question_id=batch[DataKeys.QUESTION_ID],
            gold_answers=batch[DataKeys.GOLD_ANSWERS],
        )

    def evaluation_step(self, batch: BatchDict, **kwargs) -> ModelOutput:
        hf_outputs = self._model_forward(batch)
        return SequenceQAModelOutput(
            loss=hf_outputs.loss,
            start_logits=hf_outputs.start_logits,
            end_logits=hf_outputs.end_logits,
            words=batch[DataKeys.WORDS],
            word_ids=batch[DataKeys.WORD_IDS],
            sequence_ids=batch[DataKeys.SEQUENCE_IDS],
            question_id=batch[DataKeys.QUESTION_ID],
            gold_answers=batch[DataKeys.GOLD_ANSWERS],
        )

    def predict_step(self, batch: BatchDict, **kwargs) -> ModelOutput:
        hf_outputs = self._model_forward(batch)
        # predicted_answers = self._prepare_answers(batch, hf_outputs)
        return SequenceQAModelOutput(
            loss=hf_outputs.loss,
            start_logits=hf_outputs.start_logits,
            end_logits=hf_outputs.end_logits,
            # predicted_answers=predicted_answers,
            words=batch[DataKeys.WORDS],
            word_ids=batch[DataKeys.WORD_IDS],
            sequence_ids=batch[DataKeys.SEQUENCE_IDS],
            question_id=batch[DataKeys.QUESTION_ID],
            gold_answers=batch[DataKeys.GOLD_ANSWERS],
        )

    # def _prepare_answers(
    #     self, batch: BatchDict, model_output: ModelOutput
    # ) -> List[str]:
    #     pred_start_token_index_batch = model_output.start_logits.argmax(dim=-1)
    #     pred_end_token_index_batch = model_output.end_logits.argmax(dim=-1)
    #     pred_answers = []
    #     for input_id, start_idx, end_idx in zip(
    #         batch[DataKeys.TOKEN_IDS],
    #         pred_start_token_index_batch,
    #         pred_end_token_index_batch,
    #     ):
    #         pred_answers.append(
    #             self._tokenizer.tokenizer.decode(input_id[start_idx : end_idx + 1])
    #         )
    #     return pred_answers
