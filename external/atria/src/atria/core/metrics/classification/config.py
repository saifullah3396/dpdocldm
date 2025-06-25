from atria.core.metrics.classification.output_transforms import (
    _classification_metrics_output_transform,
)
from atria.core.registry.module_registry import AtriaModuleRegistry
from atria.core.utilities.common import _get_parent_module

AtriaModuleRegistry.register_metric(
    module="ignite.metrics",
    registered_class_or_func=["Accuracy"],
    is_multilabel=False,
    output_transform=_classification_metrics_output_transform,
    device="cpu",
    zen_partial=True,
)
AtriaModuleRegistry.register_metric(
    module="ignite.metrics",
    registered_class_or_func=["Precision", "Recall"],
    is_multilabel=False,
    average=True,
    output_transform=_classification_metrics_output_transform,
    device="cpu",
    zen_partial=True,
)
AtriaModuleRegistry.register_metric(
    module=_get_parent_module(__name__) + ".f1_score",
    name="f1",
    registered_class_or_func="f1_score",
    output_transform=_classification_metrics_output_transform,
    device="cpu",
    zen_partial=True,
)
