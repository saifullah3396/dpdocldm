from atria.core.registry.module_registry import AtriaModuleRegistry
from atria.core.utilities.common import _get_parent_module

AtriaModuleRegistry.register_lr_scheduler(
    module="torch.optim.lr_scheduler",
    registered_class_or_func=["StepLR", "MultiStepLR", "ExponentialLR", "CyclicLR"],
    name=["step_lr", "multi_step_lr", "exponential_lr", "cyclic_lr"],
)

AtriaModuleRegistry.register_lr_scheduler(
    module="ignite.handlers",
    registered_class_or_func="ReduceLROnPlateauScheduler",
    name="reduce_lr_on_plateau",
)
AtriaModuleRegistry.register_lr_scheduler(
    module=_get_parent_module(__name__) + ".lambda_lr",
    registered_class_or_func="lambda_lr",
    lambda_fn="linear",
)

AtriaModuleRegistry.register_lr_scheduler(
    module=_get_parent_module(__name__) + ".cosine_annealing_lr",
    registered_class_or_func="cosine_annealing_lr",
    name="cosine_annealing_lr",
)
AtriaModuleRegistry.register_lr_scheduler(
    module=_get_parent_module(__name__) + ".polynomial_decay_lr",
    registered_class_or_func="PolynomialDecayLR",
    name="polynomial_decay_lr",
)
