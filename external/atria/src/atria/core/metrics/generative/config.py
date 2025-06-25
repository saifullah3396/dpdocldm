from atria.core.metrics.generative.output_transforms import _fid_output_transform
from atria.core.registry.module_registry import AtriaModuleRegistry
from atria.core.utilities.common import _get_parent_module

AtriaModuleRegistry.register_metric(
    module=_get_parent_module(__name__) + ".fid_score",
    name="fid_score",
    registered_class_or_func="default_fid_score",
    output_transform=_fid_output_transform,
    zen_partial=True,
)
