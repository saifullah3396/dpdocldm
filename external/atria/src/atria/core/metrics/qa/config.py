from atria.core.metrics.qa.output_transforms import (
    anls_output_transform,
    sequence_anls_output_transform,
)
from atria.core.registry.module_registry import AtriaModuleRegistry

AtriaModuleRegistry.register_metric(
    module="atria.core.metrics.qa.anls",
    registered_class_or_func=["ANLS"],
    name=["anls"],
    output_transform=anls_output_transform,
    device="cpu",
    threshold=0.5,
    zen_partial=True,
)
AtriaModuleRegistry.register_metric(
    module="atria.core.metrics.qa.sequence_anls",
    registered_class_or_func=["sequence_anls"],
    output_transform=sequence_anls_output_transform,
    device="cpu",
    threshold=0.5,
    zen_partial=True,
)
