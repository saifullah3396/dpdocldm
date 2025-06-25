
import torch


def lambda_lr(
    optimizer: torch.optim.Optimizer,
    num_training_steps: int,
    num_warmup_steps: int,
    lambda_fn: str = "linear",
    last_epoch: int = -1,
):
    if lambda_fn == "linear":

        def linear_lambda_lr(current_step: int):
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))
            return max(
                0.0,
                float(num_training_steps - current_step)
                / float(max(1, num_training_steps - num_warmup_steps)),
            )

        lambda_fn = linear_lambda_lr
    else:
        raise ValueError(f"Unknown lambda_fn: {lambda_fn}")
    return torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda_fn, last_epoch=last_epoch
    )
