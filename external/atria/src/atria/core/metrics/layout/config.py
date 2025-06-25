from atria.core.metrics.layout.output_transforms import (
    _layout_token_classification_metrics_output_transform,
)
from atria.core.registry.module_registry import AtriaModuleRegistry
from atria.core.utilities.common import _get_parent_module

AtriaModuleRegistry.register_metric(
    module=_get_parent_module(__name__) + ".precision",
    registered_class_or_func="LayoutPrecision",
    name="layout_precision",
    is_multilabel=False,
    output_transform=_layout_token_classification_metrics_output_transform,
    device="cpu",
    zen_partial=True,
)
AtriaModuleRegistry.register_metric(
    module=_get_parent_module(__name__) + ".recall",
    registered_class_or_func="LayoutRecall",
    name="layout_recall",
    is_multilabel=False,
    output_transform=_layout_token_classification_metrics_output_transform,
    device="cpu",
    zen_partial=True,
)
AtriaModuleRegistry.register_metric(
    module=_get_parent_module(__name__) + ".f1_score",
    registered_class_or_func="f1_score",
    name="layout_f1",
    output_transform=_layout_token_classification_metrics_output_transform,
    device="cpu",
    zen_partial=True,
)
