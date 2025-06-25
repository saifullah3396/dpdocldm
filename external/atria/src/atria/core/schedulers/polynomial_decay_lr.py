from torch.optim.lr_scheduler import _LRScheduler


class PolynomialDecayLR(_LRScheduler):
    """Polynomial learning rate decay until step reach to max_decay_step

    Args:
        optimizer (Optimizer): Wrapped optimizer.
        max_decay_steps: after this step, we stop decreasing learning rate
        end_learning_rate: scheduler stoping learning rate decay, value of learning rate must be this value
        power: The power of the polynomial.
    """

    def __init__(self, optimizer, max_decay_steps, end_learning_rate=0.0001, power=1.0):
        if max_decay_steps <= 1.0:
            raise ValueError("max_decay_steps should be greater than 1.")
        self.max_decay_steps = max_decay_steps
        self.end_learning_rate = end_learning_rate
        self.power = power
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch > self.max_decay_steps:
            return [self.end_learning_rate for _ in self.base_lrs]

        return [
            (base_lr - self.end_learning_rate)
            * ((1 - self.last_epoch / self.max_decay_steps) ** (self.power))
            + self.end_learning_rate
            for base_lr in self.base_lrs
        ]

    def _get_closed_form_lr(self):
        return [
            (base_lr - self.end_learning_rate)
            * ((1 - self.last_epoch / self.max_decay_steps) ** (self.power))
            + self.end_learning_rate
            for base_lr in self.base_lrs
        ]


def polynomial_decay_lr(optimizer, num_training_steps: int, max_decay_steps: int = -1):
    max_decay_steps = num_training_steps if max_decay_steps == -1 else max_decay_steps
    return PolynomialDecayLR(optimizer, max_decay_steps=max_decay_steps)
