from dataclasses import dataclass



@dataclass
class GradientConfig:
    # Whether to use gradient clipping or not
    enable_grad_clipping: bool = False

    # Max gradient norm
    max_grad_norm: float = 1.0

    # Number of updates steps to accumulate before performing a backward/update pass
    gradient_accumulation_steps: int = 1
