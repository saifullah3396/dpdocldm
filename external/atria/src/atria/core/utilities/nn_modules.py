from typing import List

from torch import nn

from atria.core.utilities.logging import get_logger

logger = get_logger(__name__)


def _set_module_with_name(model: nn.Module, module_name: str, module: nn.Module):
    return setattr(model, module_name, module)


def _replace_module_with_name(module, target_name, new_module):
    target_name = target_name.split(".")
    if len(target_name) > 1:
        _replace_module_with_name(
            getattr(module, target_name[0]), ".".join(target_name[1:]), new_module
        )
    setattr(module, target_name[-1], new_module)


def _get_all_nn_modules_in_object(object: nn.Module):
    return {k: v for k, v in object.__dict__.items() if isinstance(v, nn.Module)}


def _get_last_module(model: nn.Module):
    return list((module_name, module) for module_name, module in model.named_modules())[
        -1
    ]


def _find_layer_in_model(model: nn.Module, layer_name: str):
    # get encoder
    layer = [x for x, m in model.named_modules() if x == layer_name]
    if len(layer) == 0:
        raise ValueError(f"Encoder layer {layer_name} not found in the model.")
    return layer[0]


def _freeze_layers_by_name(model: nn.Module, layer_names: List[str]):
    for layer_name in layer_names:
        layer = _find_layer_in_model(model, layer_name)
        for p in layer.parameters():
            p.requires_grad = False


def _freeze_layers(layers):
    for layer in layers:
        for p in layer.parameters():
            p.requires_grad = False
