from typing import Any, Sequence, Tuple, Union

import torch
from atria.core.models.utilities.common import _validate_keys_in_batch
from atria.core.training.engines.engine_steps.base import BaseEngineStep
from atria.core.training.utilities.constants import TrainingStage
from atria.core.utilities.logging import get_logger
from ignite.engine import Engine

logger = get_logger(__name__)


class EvaluationStep(BaseEngineStep):
    def __call__(
        self, engine: Engine, batch: Sequence[torch.Tensor], **kwargs
    ) -> Union[Any, Tuple[torch.Tensor]]:

        import torch
        from torch.cuda.amp import autocast

        # validate model is built
        self._task_module.validate_model_built()

        # validate batch keys
        _validate_keys_in_batch(
            keys=self._task_module.required_keys_in_batch(stage=self.stage), batch=batch
        )

        # ready model for evaluation
        self._task_module.eval()
        if self._with_amp:
            self._task_module.half()

        with torch.no_grad():
            with autocast(enabled=self._with_amp):
                self._convert_batch_to_device(batch)

                # forward pass
                return self._model_step(
                    engine=engine,
                    batch=batch,
                    test_run=self._test_run,
                    **kwargs,
                )

    def _model_step(
        self,
        engine: Engine,
        batch: Sequence[torch.Tensor],
        test_run: bool = False,
        **kwargs,
    ) -> Union[Any, Tuple[torch.Tensor]]:
        raise NotImplementedError("Subclasses must implement this method")


class ValidationStep(EvaluationStep):
    @property
    def stage(self) -> TrainingStage:
        return TrainingStage.validation

    def _model_step(
        self, engine: Engine, batch: Sequence[torch.Tensor], test_run: bool = False
    ) -> Union[Any, Tuple[torch.Tensor]]:
        # forward pass
        return self._task_module.evaluation_step(
            evaluation_engine=engine,
            training_engine=self._parent_engine,
            batch=batch,
            stage=self.stage,
            test_run=test_run,
        )


class VisualizationStep(EvaluationStep):
    @property
    def stage(self) -> TrainingStage:
        return TrainingStage.visualization

    def _model_step(
        self, engine: Engine, batch: Sequence[torch.Tensor], test_run: bool = False
    ) -> Union[Any, Tuple[torch.Tensor]]:
        # forward pass
        return self._task_module.visualization_step(
            evaluation_engine=engine,
            training_engine=self._parent_engine,
            batch=batch,
            stage=self.stage,
            test_run=test_run,
        )


class TestStep(EvaluationStep):
    @property
    def stage(self) -> TrainingStage:
        return TrainingStage.test

    def _model_step(
        self, engine: Engine, batch: Sequence[torch.Tensor], test_run: bool = False
    ) -> Union[Any, Tuple[torch.Tensor]]:
        # forward pass
        return self._task_module.evaluation_step(
            evaluation_engine=engine,
            batch=batch,
            stage=self.stage,
            test_run=test_run,
        )


class PredictionStep(EvaluationStep):
    @property
    def stage(self) -> TrainingStage:
        return TrainingStage.predict

    def _model_step(
        self, engine: Engine, batch: Sequence[torch.Tensor], test_run: bool = False
    ) -> Union[Any, Tuple[torch.Tensor]]:
        # forward pass
        return self._task_module.predict_step(
            evaluation_engine=engine,
            batch=batch,
            stage=self.stage,
            test_run=test_run,
        )


class FeatureExtractorStep(EvaluationStep):
    @property
    def stage(self) -> str:
        return "FeatureExtractor"

    def _model_step(
        self, engine: Engine, batch: Sequence[torch.Tensor], test_run: bool = False
    ) -> Union[Any, Tuple[torch.Tensor]]:
        assert hasattr(self._task_module, "feature_extractor_step"), (
            f"Task module [{self._task_module.__class__.__name__}] "
            f"does not have method [feature_extractor_step]."
        )

        return self._task_module.feature_extractor_step(
            engine=engine,
            batch=batch,
            stage=self.stage,
            test_run=test_run,
        )
