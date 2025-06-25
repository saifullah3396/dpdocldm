from atria.core.registry.module_registry import AtriaModuleRegistry
from atria.core.utilities.common import _get_parent_module

# register task modules type=[atria]
AtriaModuleRegistry.register_task_module(
    module=_get_parent_module(__name__) + ".task_modules.atria_task_module",
    registered_class_or_func="AtriaTaskModule",
    name="atria_task_module",
    hydra_defaults=[
        "_self_",
        {"/torch_model_builder@torch_model_builder": "timm"},
    ],
)

# register torch model torch_model_builders as child node of task_module
AtriaModuleRegistry.register_torch_model_builder(
    module=_get_parent_module(__name__) + ".torch_model_builders.timm",
    registered_class_or_func="TimmModelBuilder",
    name="timm",
)
AtriaModuleRegistry.register_torch_model_builder(
    module=_get_parent_module(__name__) + ".torch_model_builders.torchvision",
    registered_class_or_func="TorchVisionModelBuilder",
    name="torchvision",
)
AtriaModuleRegistry.register_torch_model_builder(
    module=_get_parent_module(__name__) + ".torch_model_builders.local",
    registered_class_or_func="LocalTorchModelBuilder",
    name="local",
)
AtriaModuleRegistry.register_torch_model_builder(
    module=_get_parent_module(__name__) + ".torch_model_builders.transformers",
    registered_class_or_func="TransformersModelBuilder",
    name="transformers",
)
AtriaModuleRegistry.register_torch_model_builder(
    module=_get_parent_module(__name__) + ".torch_model_builders.diffusers",
    registered_class_or_func="DiffusersModelBuilder",
    name="diffusers",
)
AtriaModuleRegistry.register_torch_model_builder(
    module=_get_parent_module(__name__) + ".torch_model_builders.detectron2",
    registered_class_or_func="Detectron2ModelBuilder",
    name="detectron2",
)
