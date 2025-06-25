# my_app.py
from omegaconf import DictConfig, ListConfig, OmegaConf


def resolve_dir_name(input: str) -> str:
    if "." in input:
        input = input.split(".")[-1]
    ret = input.replace("{", "")
    ret = ret.replace("}", "")
    ret = ret.replace("[", "")
    ret = ret.replace("]", "")
    ret = ret.replace(",", "_")
    ret = ret.replace("/", "_")
    ret = ret.replace("=", "-")
    return ret


def dir_name_from_overrides(overrides: ListConfig, filter: DictConfig) -> str:
    task_overrides: ListConfig = overrides.task
    overrides_filtered = []
    for override in task_overrides:
        output_key_for_override = None
        for key, target_key in filter.items():
            if key in override:
                output_key_for_override = target_key
                break
        if output_key_for_override is not None:
            key, value = override.split("=")
            overrides_filtered.append(f"{output_key_for_override}={value}")
    ret: str = "_".join(overrides_filtered)
    ret = ret.replace("{", "")
    ret = ret.replace("}", "")
    ret = ret.replace("[", "")
    ret = ret.replace("]", "")
    ret = ret.replace(",", "_")
    ret = ret.replace("/", "_")
    ret = ret.replace("=", "-")
    if ret == "":
        return "default"
    return ret


def resolve_tuple(*args):
    return tuple(args)


if not OmegaConf.has_resolver("resolve_dir_name"):
    OmegaConf.register_new_resolver("resolve_dir_name", resolve_dir_name)
if not OmegaConf.has_resolver("dir_name_from_overrides"):
    OmegaConf.register_new_resolver("dir_name_from_overrides", dir_name_from_overrides)
if not OmegaConf.has_resolver("as_tuple"):
    OmegaConf.register_new_resolver("as_tuple", resolve_tuple)
