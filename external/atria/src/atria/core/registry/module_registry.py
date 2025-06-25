from __future__ import annotations

from collections.abc import MutableMapping
from typing import List, Optional, Union

from atria.core.utilities.common import _resolve_module_from_path
from atria.core.utilities.logging import get_logger
from attr import dataclass
from hydra.core.config_store import ConfigNode, ConfigStore

logger = get_logger(__name__)


def flatten_dict(
    d: MutableMapping, parent_key: str = "", sep: str = "."
) -> MutableMapping:
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, MutableMapping):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


PROVIDER_NAME = "atria"


class AtriaGroups:
    DATA_MODULE = "data_module"
    DATA_TRANSFORM = "data_transform"
    BATCH_SAMPLER = "batch_sampler"
    DATASET_CACHER = "dataset_cacher"
    DATALOADER_BUILDER = "dataloader_builder"
    TRAIN_VALIDATION_SPLITTER = "train_validation_splitter"
    DATA_COLLATOR = "data_collator"
    TASK_MODULE = "task_module"
    MODEL_CONFIG = "model_config"
    TORCH_MODEL_BUILDER = "torch_model_builder"
    TASK_RUNNER = "task_runner"
    METRIC = "metric"
    LR_SCHEDULER = "lr_scheduler"
    OPTIMIZER = "optimizer"
    ENGINE = "engine"
    ENGINE_STEP = "engine_step"


@dataclass
class ModuleConfig:
    group: str
    name: str
    module: str
    package: str = None
    build_kwargs: dict = None
    registers_target: bool = True

    def __attrs_post_init__(self):
        if self.build_kwargs is None:
            self.build_kwargs = {}


class AtriaModuleRegistry:
    REGISTERED_MODULE_CONFIG: List[ModuleConfig] = []

    @staticmethod
    def build_module_configurations() -> ConfigStore:
        from hydra_zen import builds

        cs = ConfigStore.instance()
        for module_config in AtriaModuleRegistry.REGISTERED_MODULE_CONFIG:
            try:
                cs.store(
                    group=module_config.group,
                    name=module_config.name,
                    node=builds(
                        _resolve_module_from_path(module_config.module),
                        **module_config.build_kwargs,
                    ),
                    provider=PROVIDER_NAME,
                    package=module_config.package,
                )
                if not module_config.registers_target:
                    config_node: ConfigNode = cs.repo[module_config.group][
                        module_config.name + ".yaml"
                    ]
                    del config_node.node._target_
                    del config_node.node._partial_
            except Exception as e:
                logger.exception(
                    f"Failed to register module {module_config.name} from {module_config.module}: {e}"
                )
                exit()

        return cs

    @staticmethod
    def register_module_configuration(
        group: str,
        name: str,
        module: str,
        build_kwargs: dict = None,
        lazy_build: bool = True,
        is_global_package: bool = False,
        registers_target: bool = True,
    ):
        if build_kwargs is None:
            build_kwargs = {}

        if lazy_build:
            AtriaModuleRegistry.REGISTERED_MODULE_CONFIG.append(
                ModuleConfig(
                    group=group,
                    name=name,
                    module=module,
                    build_kwargs=build_kwargs,
                    package="__global__" if is_global_package else None,
                    registers_target=registers_target,
                )
            )
        else:
            from hydra.core.config_store import ConfigStore
            from hydra_zen import builds

            cs = ConfigStore.instance()
            cs.store(
                group=PROVIDER_NAME + "/" + group,
                name=name,
                node=builds(
                    _resolve_module_from_path(module),
                    **build_kwargs,
                ),
                provider=PROVIDER_NAME,
                package="__global__" if is_global_package else None,
            )

    @staticmethod
    def register_class_in_group(
        group: str,
        module: str,
        registered_class_or_func: Union[str, List[str]],
        name: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ):
        from atria.core.utilities.string_utils import _convert_to_snake_case

        if not isinstance(registered_class_or_func, (tuple, list)):
            registered_class_or_func = [registered_class_or_func]
        if name is None:
            name = [None] * len(registered_class_or_func)
        if not isinstance(name, (tuple, list)):
            name = [name]
        assert len(registered_class_or_func) == len(name), (
            f"Length of registered_class_or_func ({len(registered_class_or_func)}) "
            f"and name ({len(name)}) must be the same."
        )
        is_global_package = kwargs.pop("is_global_package", False)
        registers_target = kwargs.pop("registers_target", True)
        for single_name, single_class_or_func in zip(name, registered_class_or_func):
            # register torch model torch_model_builders as child node of task_module
            AtriaModuleRegistry.register_module_configuration(
                group=group,
                name=(
                    single_name
                    if single_name is not None
                    else _convert_to_snake_case(single_class_or_func)
                ),
                module=module + "." + single_class_or_func,
                build_kwargs=dict(populate_full_signature=True, **kwargs),
                is_global_package=is_global_package,
                registers_target=registers_target,
            )

    @staticmethod
    def register_data_module(
        module: str,
        registered_class_or_func: Union[str, List[str]],
        name: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ):
        AtriaModuleRegistry.register_class_in_group(
            group=AtriaGroups.DATA_MODULE,
            module=module,
            registered_class_or_func=registered_class_or_func,
            name=name,
            **kwargs,
        )

    @staticmethod
    def register_dataset_cacher(
        module: str,
        registered_class_or_func: Union[str, List[str]],
        name: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ):
        AtriaModuleRegistry.register_class_in_group(
            group=AtriaGroups.DATASET_CACHER,
            module=module,
            registered_class_or_func=registered_class_or_func,
            name=name,
            **kwargs,
        )

    @staticmethod
    def register_train_validation_splitter(
        module: str,
        registered_class_or_func: Union[str, List[str]],
        name: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ):
        AtriaModuleRegistry.register_class_in_group(
            group=AtriaGroups.TRAIN_VALIDATION_SPLITTER,
            module=module,
            registered_class_or_func=registered_class_or_func,
            name=name,
            **kwargs,
        )

    @staticmethod
    def register_data_transform(
        module: str,
        registered_class_or_func: Union[str, List[str]],
        name: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ):
        AtriaModuleRegistry.register_class_in_group(
            group=AtriaGroups.DATA_TRANSFORM,
            module=module,
            registered_class_or_func=registered_class_or_func,
            name=name,
            **kwargs,
        )

    @staticmethod
    def register_batch_sampler(
        module: str,
        registered_class_or_func: Union[str, List[str]],
        name: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ):
        AtriaModuleRegistry.register_class_in_group(
            group=AtriaGroups.BATCH_SAMPLER,
            module=module,
            registered_class_or_func=registered_class_or_func,
            name=name,
            **kwargs,
        )

    @staticmethod
    def register_dataloader_builder(
        module: str,
        registered_class_or_func: Union[str, List[str]],
        name: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ):
        AtriaModuleRegistry.register_class_in_group(
            group=AtriaGroups.DATALOADER_BUILDER,
            module=module,
            registered_class_or_func=registered_class_or_func,
            name=name,
            **kwargs,
        )

    @staticmethod
    def register_data_collator(
        module: str,
        registered_class_or_func: Union[str, List[str]],
        name: Optional[Union[str, List[str]]] = None,
        **kwargs,
    ):
        AtriaModuleRegistry.register_class_in_group(
            group=AtriaGroups.DATA_COLLATOR,
            module=module,
            registered_class_or_func=registered_class_or_func,
            name=name,
            **kwargs,
        )

    @staticmethod
    def register_task_module(
        module: str, registered_class_or_func: Union[str, List[str]], **kwargs
    ):
        AtriaModuleRegistry.register_class_in_group(
            group=AtriaGroups.TASK_MODULE,
            module=module,
            registered_class_or_func=registered_class_or_func,
            zen_partial=True,
            **kwargs,
        )

    @staticmethod
    def register_torch_model_builder(
        module: str, registered_class_or_func: Union[str, List[str]], **kwargs
    ):
        AtriaModuleRegistry.register_class_in_group(
            group=AtriaGroups.TORCH_MODEL_BUILDER,
            module=module,
            registered_class_or_func=registered_class_or_func,
            zen_partial=True,
            **kwargs,
        )

    @staticmethod
    def register_model_config(
        module: str, registered_class_or_func: Union[str, List[str]], **kwargs
    ):
        AtriaModuleRegistry.register_class_in_group(
            group=AtriaGroups.MODEL_CONFIG,
            module=module,
            registered_class_or_func=registered_class_or_func,
            zen_partial=False,
            registers_target=False,
            **kwargs,
        )

    @staticmethod
    def register_metric(
        module: str, registered_class_or_func: Union[str, List[str]], **kwargs
    ):
        AtriaModuleRegistry.register_class_in_group(
            group=AtriaGroups.METRIC,
            module=module,
            registered_class_or_func=registered_class_or_func,
            **kwargs,
        )

    @staticmethod
    def register_lr_scheduler(
        module: str, registered_class_or_func: Union[str, List[str]], **kwargs
    ):
        AtriaModuleRegistry.register_class_in_group(
            group=AtriaGroups.LR_SCHEDULER,
            module=module,
            registered_class_or_func=registered_class_or_func,
            zen_partial=True,
            **kwargs,
        )

    @staticmethod
    def register_optimizer(
        module: str, registered_class_or_func: Union[str, List[str]], **kwargs
    ):
        AtriaModuleRegistry.register_class_in_group(
            group=AtriaGroups.OPTIMIZER,
            module=module,
            registered_class_or_func=registered_class_or_func,
            zen_partial=True,
            **kwargs,
        )

    @staticmethod
    def register_engine(
        module: str, registered_class_or_func: Union[str, List[str]], **kwargs
    ):
        AtriaModuleRegistry.register_class_in_group(
            group=AtriaGroups.ENGINE,
            module=module,
            registered_class_or_func=registered_class_or_func,
            zen_partial=True,
            **kwargs,
        )

    @staticmethod
    def register_engine_step(
        module: str, registered_class_or_func: Union[str, List[str]], **kwargs
    ):
        AtriaModuleRegistry.register_class_in_group(
            group=AtriaGroups.ENGINE_STEP,
            module=module,
            registered_class_or_func=registered_class_or_func,
            zen_partial=True,
            **kwargs,
        )

    @staticmethod
    def register_task_runner(
        module: str, registered_class_or_func: Union[str, List[str]], **kwargs
    ):
        AtriaModuleRegistry.register_class_in_group(
            group="",  # since task runner is a top level module, it must be placed in the root of the registry, therefore it has no group
            module=module,
            registered_class_or_func=registered_class_or_func,
            is_global_package=True,
            **kwargs,
        )
