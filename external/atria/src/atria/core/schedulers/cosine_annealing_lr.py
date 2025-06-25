import torch


def cosine_annealing_lr(
    optimizer: torch.optim.Optimizer,
    total_update_steps: int,
    total_warmup_steps: int,
    steps_per_epoch: int,
    restarts: bool = False,
    eta_min: int = 0,
    last_epoch: int = -1,
):
    if not restarts:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=total_update_steps - total_warmup_steps,
            eta_min=eta_min,
            last_epoch=last_epoch,
        )
    else:
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=steps_per_epoch - total_warmup_steps,
            eta_min=eta_min,
            last_epoch=last_epoch,
        )
