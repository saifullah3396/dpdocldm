from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from atria.core.models.task_modules.atria_task_module import AtriaTaskModule
from atria.core.training.configs.gradient_config import GradientConfig
from atria.core.training.engines.engine_steps.training import TrainingStep
from atria.core.training.engines.events import OptimizerEvents
from atria.core.training.engines.utilities import _detach_tensor
from atria.core.utilities.logging import get_logger
from ignite.engine import Engine
from torch.optim import Optimizer

logger = get_logger(__name__)


class DPTrainingStep(TrainingStep):
    def __init__(
        self,
        task_module: AtriaTaskModule,
        device: Union[str, torch.device],
        optimizers: Dict[str, Optimizer],
        gradient_config: GradientConfig,
        grad_scaler: Optional["torch.cuda.amp.GradScaler"] = None,
        output_transform: Callable[[Any], Any] = _detach_tensor,
        non_blocking_tensor_conv: bool = False,
        with_amp: bool = False,
        test_run: bool = False,
    ):
        super().__init__(
            task_module=task_module,
            device=device,
            optimizers=optimizers,
            gradient_config=gradient_config,
            grad_scaler=grad_scaler,
            output_transform=output_transform,
            non_blocking_tensor_conv=non_blocking_tensor_conv,
            with_amp=with_amp,
            test_run=test_run,
        )

    def _setup_amp(self) -> None:
        if self._with_amp:
            raise NotImplementedError("DPTrainingStep does not support AMP yet.")

    def _update_optimizers_with_grad_scaler(
        self,
        engine: Engine,
        loss: torch.Tensor,
        optimizer_key: Optional[str] = None,
    ) -> None:
        raise NotImplementedError("DPTrainingStep does not support AMP yet.")

    def __call__(
        self, engine: Engine, batch: Dict[str, torch.Tensor]
    ) -> Union[Any, Tuple[torch.Tensor]]:
        self._validate_inputs(batch)
        self._validate_gradient_config()
        self._reset_optimizers(engine=engine)
        self._task_module.train()
        self._convert_batch_to_device(batch)
        loss, model_output = self._call_forward(engine=engine, batch=batch)
        self._update_optimizers(engine=engine, loss=loss)

        optimizer_updated = False
        for opt in self._optimizers.values():
            if not opt._is_last_step_skipped:
                optimizer_updated = True

        if optimizer_updated:
            engine.fire_event(OptimizerEvents.optimizer_step)
            engine.state.optimizer_step += 1

        return model_output
