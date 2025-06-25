from dataclasses import dataclass


@dataclass
class ModelEmaConfig:
    enabled: bool = False

    # EMA decay
    momentum: float = 0.0001

    # EMA warmup
    momentum_warmup: float = 0.0

    # warmup iteartions
    warmup_iters: int = 0

    # update every n epochs
    update_every: int = 1
