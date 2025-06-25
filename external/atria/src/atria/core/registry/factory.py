from __future__ import annotations

import importlib
from dataclasses import is_dataclass

from atria.core.registry.exceptions import (
    ModuleNotFoundError,
    RegistryGroupNotFoundError,
    RegistrySubgroupNotFoundError,
)
from atria.core.registry.module_registry import AtriaModuleRegistry
from atria.core.utilities.dataclasses.dacite_wrapper import from_dict
from atria.core.utilities.logging import get_logger

logger = get_logger(__name__, reset=True)


class ModuleFactory:
    """
    The module factory that initializes the module based on its name and registry group.
    """

    @classmethod
    def create_instance(
        cls,
        module_name: str,
        module_kwargs: dict,
        registry_group: str,
        module_sub_group: str = None,
    ):
        module_class = cls.load_class_by_name(
            module_name=module_name,
            registry_group=registry_group,
            module_sub_group=module_sub_group,
        )
        # if the module is a dataclass, use dacite to create the instance
        if is_dataclass(module_class):
            return from_dict(data_class=module_class, data=module_kwargs)
        else:
            return module_class(**module_kwargs)

    @classmethod
    def load_class_by_name(
        cls,
        module_name: str,
        registry_group: str,
        module_sub_group: str = None,
    ):
        # first we try to see if the module class is directly provided by the user as a string
        try:
            module = getattr(
                importlib.import_module(".".join(module_name.split(".")[:-1])),
                module_name.split(".")[-1],
            )
            return module
        except Exception as e:
            pass

        # after that we try to load the module from the registry
        module_registry = AtriaModuleRegistry.REGISTRY.get(registry_group, None)
        if not module_registry:
            raise RegistryGroupNotFoundError(
                f"Requested module registry [{registry_group}] not found."
            )

        # load the sub-group registry if specified
        if module_sub_group is not None:
            module_registry = module_registry.get(module_sub_group, None)
            if module_registry is None:
                raise RegistrySubgroupNotFoundError(
                    f"Sub-group [{module_sub_group}] not found in registry [{registry_group}]. "
                    f"Supported sub-groups are: {module_registry.keys()}"
                )

        #
        module_lazy_loader = module_registry.get(module_name, None)
        if module_lazy_loader is None:
            raise ModuleNotFoundError(
                f"{registry_group} [{module_name}] is not supported. "
                f"Supported modules are: {module_registry.keys()}"
            )

        # lazy load the module by calling the function
        return module_lazy_loader()
