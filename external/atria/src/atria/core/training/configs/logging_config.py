from dataclasses import dataclass
from typing import Optional


@dataclass
class LoggingConfig:
    # N update steps to log on
    logging_steps: int = 100

    # pbar update refresh
    refresh_rate: int = 10

    # Whether to log gpu stats
    log_gpu_stats: Optional[bool] = False

    # Whether to do profiling
    profile_time: Optional[bool] = False

    # Whether to log outputs to tensorboard
    log_to_tb: Optional[bool] = True
