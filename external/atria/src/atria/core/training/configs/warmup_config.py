from dataclasses import dataclass
from typing import Optional


@dataclass
class WarmupConfig:
    # Linear warmup over warmup_ratio fraction of total steps
    warmup_ratio: Optional[float] = None

    # Linear warmup over warmup_steps
    warmup_steps: Optional[int] = None

    def __post_init__(self):
        import logging

        if self.warmup_ratio is not None:
            if self.warmup_ratio < 0 or self.warmup_ratio > 1:
                raise ValueError("warmup_ratio must lie in range [0,1]")
            elif self.warmup_ratio is not None and self.warmup_steps is not None:
                logging.info(
                    "Both warmup_ratio and warmup_steps given, warmup_steps will override"
                    " any effect of warmup_ratio during training"
                )

        if self.warmup_ratio is None:
            self.warmup_ratio = 0
        if self.warmup_steps is None:
            self.warmup_steps = 0
