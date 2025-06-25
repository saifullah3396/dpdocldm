from atria.core.registry.module_registry import AtriaModuleRegistry
from atria.core.utilities.common import _get_parent_module

AtriaModuleRegistry.register_optimizer(
    module="torch.optim",
    registered_class_or_func=[
        "Adadelta",
        "Adam",
        "AdamW",
        "SparseAdam",
        "Adagrad",
        "Adamax",
        "ASGD",
        "LBFGS",
        "RMSprop",
        "Rprop",
        "SGD",
    ],
    name=[
        "adadelta",
        "adam",
        "adamw",
        "sparse_adam",
        "adagrad",
        "adamax",
        "asgd",
        "lbfgs",
        "rmsprop",
        "rprop",
        "sgd",
    ],
)
AtriaModuleRegistry.register_optimizer(
    module=".".join((_get_parent_module(__name__), f"lars")),
    registered_class_or_func="LARS",
    name="lars",
)
