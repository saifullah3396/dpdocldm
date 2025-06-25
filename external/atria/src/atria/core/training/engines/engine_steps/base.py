from abc import ABCMeta, abstractmethod
from typing import Any, Dict, Sequence, Tuple, Union

import torch
from atria.core.models.task_modules.atria_task_module import AtriaTaskModule
from atria.core.training.utilities.constants import TrainingStage
from atria.core.utilities.logging import get_logger
from ignite.engine import Engine

logger = get_logger(__name__)


class BaseEngineStep(object, metaclass=ABCMeta):
    def __init__(
        self,
        task_module: AtriaTaskModule,
        device: Union[str, torch.device],
        non_blocking_tensor_conv: bool = False,
        with_amp: bool = False,
        test_run: bool = False,
    ):
        self._task_module = task_module
        self._device = torch.device(device)
        self._non_blocking_tensor_conv = non_blocking_tensor_conv
        self._with_amp = with_amp
        self._test_run = test_run
        self._parent_engine = None

    def attach_parent_engine(self, engine: Engine) -> None:
        self._parent_engine = engine

    def _convert_batch_to_device(self, batch: Dict[str, torch.Tensor]) -> None:
        from ignite.utils import convert_tensor

        try:
            # put batch to device
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = convert_tensor(
                        value,
                        device=self._device,
                        non_blocking=self._non_blocking_tensor_conv,
                    )
        except Exception as e:
            logger.exception(
                f"Unable to convert batch to device. "
                f"Did you forget to setup a runtime_data_transforms or collate_fn?\nError: {e}"
            )
            exit(1)

    @property
    @abstractmethod
    def stage(self) -> TrainingStage:
        pass

    @abstractmethod
    def __call__(
        self, engine: Engine, batch: Sequence[torch.Tensor]
    ) -> Union[Any, Tuple[torch.Tensor]]:
        pass
