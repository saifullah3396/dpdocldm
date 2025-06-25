from typing import Callable, Union

import torch
from ignite.metrics import Metric

from atria.core.metrics.layout.precision import LayoutPrecision
from atria.core.metrics.layout.recall import LayoutRecall


def f1_score(output_transform: Callable, device: Union[str, torch.device]) -> Metric:
    precision = LayoutPrecision(
        average=False, output_transform=output_transform, device=device
    )
    recall = LayoutRecall(
        average=False, output_transform=output_transform, device=device
    )
    return (precision * recall * 2 / (precision + recall)).mean()
