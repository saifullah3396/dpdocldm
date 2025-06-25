from typing import Callable

from atria.core.utilities.common import _resolve_module_from_path


def seqeval_classification_metric(
    metric_func: str, output_transform: Callable, **kwargs
):
    from functools import partial

    from atria.core.metrics.common.epoch_dict_metric import EpochDictMetric

    return partial(
        EpochDictMetric,
        compute_fn=partial(_resolve_module_from_path(metric_func), **kwargs),
        output_transform=output_transform,
    )
