from atria.core.metrics.token_classification.output_transforms import (
    _token_classification_output_transform,
)
from atria.core.registry.module_registry import AtriaModuleRegistry
from atria.core.utilities.common import _get_parent_module

for metric in [
    "accuracy_score",
    "precision_score",
    "recall_score",
    "f1_score",
    "classification_report",
]:
    kwargs = {}
    if metric in [
        "precision_score",
        "recall_score",
        "f1_score",
        "classification_report",
    ]:
        kwargs["scheme"] = "IOB2"
    AtriaModuleRegistry.register_metric(
        name=f"seqeval_{metric}",
        module=_get_parent_module(__name__) + ".seqeval",
        registered_class_or_func="seqeval_classification_metric",
        metric_func=f"seqeval.metrics.{metric}",
        output_transform=_token_classification_output_transform,
        zen_partial=False,
        **kwargs,
    )
