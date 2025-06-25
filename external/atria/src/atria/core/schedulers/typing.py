from typing import Callable, TypeAlias, Union

import ignite
import ignite.handlers
import torch

LRSchedulerType: TypeAlias = Union[
    torch.optim.lr_scheduler.LRScheduler, Callable, ignite.handlers.LRScheduler
]
