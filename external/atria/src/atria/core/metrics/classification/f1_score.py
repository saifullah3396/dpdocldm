from typing import Callable, Union

import torch
from ignite.metrics import Metric


def f1_score(output_transform: Callable, device: Union[str, torch.device]) -> Metric:
    from ignite.metrics import Precision, Recall

    # use ignite arthematics of metrics to compute f1 score
    # unnecessary complication
    precision = Precision(
        average=False, output_transform=output_transform, device=device
    )
    recall = Recall(average=False, output_transform=output_transform, device=device)
    return (precision * recall * 2 / (precision + recall)).mean()
