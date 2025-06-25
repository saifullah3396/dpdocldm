from typing import List

import torch

from atria.core.constants import DataKeys
from atria.core.models.model_outputs import ClassificationModelOutput, ModelOutput
from atria.core.models.task_modules.atria_task_module import AtriaTaskModule
from atria.core.models.utilities.nn_modules import _get_logits_from_output
from atria.core.training.utilities.constants import TrainingStage
from atria.core.utilities.logging import get_logger
from atria.core.utilities.typing import BatchDict

logger = get_logger(__name__)


class ClassificationModule(AtriaTaskModule):
    _REQUIRES_BUILDER_DICT = False

    def _build_model(self) -> torch.nn.Module:
        model: torch.nn.Module = super()._build_model()

        # initialize classification related stuff
        self._loss_fn_train = torch.nn.CrossEntropyLoss()
        self._loss_fn_eval = torch.nn.CrossEntropyLoss()
        return model

    def required_keys_in_batch(self, stage: TrainingStage) -> List[str]:
        if stage == TrainingStage.train:
            return [DataKeys.LABEL]
        elif stage == TrainingStage.validation:
            return [DataKeys.LABEL]
        elif stage == TrainingStage.test:
            return [DataKeys.LABEL]
        elif stage == TrainingStage.predict:
            return []
        else:
            raise ValueError(f"Stage {stage} not supported")

    def training_step(self, batch: BatchDict, **kwargs) -> ModelOutput:
        label = batch.pop(DataKeys.LABEL)
        model_output = self._model_forward(batch)
        logits = _get_logits_from_output(model_output)
        loss = self._loss_fn_train(logits, label)
        return ClassificationModelOutput(loss=loss, logits=logits, label=label)

    def evaluation_step(self, batch: BatchDict, **kwargs) -> ModelOutput:
        label = batch.pop(DataKeys.LABEL)
        model_output = self._model_forward(batch)
        logits = _get_logits_from_output(model_output)
        loss = self._loss_fn_eval(logits, label)
        return ClassificationModelOutput(loss=loss, logits=logits, label=label)

    def predict_step(self, batch: BatchDict, **kwargs) -> ModelOutput:
        label = batch.pop(DataKeys.LABEL)
        model_output = self._model_forward(batch)
        logits = _get_logits_from_output(model_output)
        loss = self._loss_fn_eval(logits, label)
        return ClassificationModelOutput(
            loss=loss, logits=logits, prediction=logits.argmax(dim=-1)
        )
