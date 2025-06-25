import functools
from dataclasses import is_dataclass
from typing import Any, Dict, List

import torch
from atria.core.utilities.logging import get_logger

logger = get_logger(__name__)


def _rsetattr(obj: Any, attr: str, val: Any) -> None:
    pre, _, post = attr.rpartition(".")
    return setattr(_rgetattr(obj, pre) if pre else obj, post, val)


def _rgetattr(obj: Any, attr: str, *args: Any) -> Any:
    def _getattr(obj: Any, attr: str) -> Any:
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def _set_module_with_name(
    model: torch.nn.Module, module_name: str, module: torch.nn.Module
) -> None:
    return setattr(model, module_name, module)


def _replace_module_with_name(
    module: torch.nn.Module, target_name: str, new_module: torch.nn.Module
) -> None:
    target_name = target_name.split(".")
    if len(target_name) > 1:
        _replace_module_with_name(
            getattr(module, target_name[0]), ".".join(target_name[1:]), new_module
        )
    setattr(module, target_name[-1], new_module)


def _get_all_nn_modules_in_object(
    object: torch.nn.Module,
) -> Dict[str, torch.nn.Module]:
    return {k: v for k, v in object.__dict__.items() if isinstance(v, torch.nn.Module)}


def _get_last_module(model: torch.nn.Module) -> Any:
    return list((module_name, module) for module_name, module in model.named_modules())[
        -1
    ]


def _find_layer_in_model(model: torch.nn.Module, layer_name: str) -> str:
    layer = [x for x, m in model.named_modules() if x == layer_name]
    if len(layer) == 0:
        raise ValueError(f"Encoder layer {layer_name} not found in the model.")
    return layer[0]


def _freeze_layers_by_name(model: torch.nn.Module, layer_names: List[str]) -> None:
    for layer_name in layer_names:
        layer = _find_layer_in_model(model, layer_name)
        for p in layer.parameters():
            p.requires_grad = False


def _freeze_layers(layers: List[torch.nn.Module]) -> None:
    for layer in layers:
        for p in layer.parameters():
            p.requires_grad = False


def _summarize_model(model: object) -> None:
    from atria.core.models.task_modules.atria_task_module import TorchModelDict
    from torchinfo import summary

    if isinstance(model.torch_model, TorchModelDict):
        logger.info(f"Trainable models:")
        logger.info(summary(model.torch_model.trainable_models, verbose=0, depth=2))
        logger.info(f"Non-trainable models:")
        logger.info(summary(model.torch_model.non_trainable_models, verbose=0, depth=2))

    for k, v in _get_all_nn_modules_in_object(model).items():
        logger.info(f"Model component [{k}]:")
        logger.info(summary(v, verbose=0, depth=2))


def _batch_norm_to_group_norm(model: torch.nn.Module) -> None:
    import torch

    for name, module in model.named_modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            num_channels = module.num_features

            def get_groups(num_channels: int, groups: int) -> int:
                if num_channels % groups != 0:
                    groups = groups // 2
                    groups = get_groups(num_channels, groups)
                return groups

            groups = get_groups(num_channels, 32)
            bn = _rgetattr(model, name)
            gn = torch.nn.GroupNorm(groups, num_channels)
            _rsetattr(model, name, gn)


def _remove_lora_layers(model: torch.nn.Module) -> None:
    from diffusers.models.lora import LoRACompatibleConv, LoRACompatibleLinear
    from torch import nn

    for name, module in model.named_modules():
        if isinstance(module, LoRACompatibleLinear):
            new_module = nn.Linear(module.in_features, module.out_features)
            new_module.__dict__.update(module.__dict__)
            _rsetattr(model, name, new_module)
        if isinstance(module, LoRACompatibleConv):
            new_module = nn.Conv2d(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                module.stride,
                module.padding,
                module.dilation,
                module.groups,
            )
            _rsetattr(model, name, new_module)


def _get_logits_from_output(model_output):
    import torch

    # we assume first element is logits
    if isinstance(model_output, torch.Tensor):  # usually timm returns a tensor directly
        return model_output
    elif isinstance(model_output, tuple):  # usually timm returns a tensor directly
        return model_output[0]
    elif is_dataclass(model_output):  # usually huggingface returns a dataclass
        return getattr(model_output, "logits")
    elif isinstance(model_output, dict):
        if "logits" in model_output:
            return model_output["logits"]
    else:
        raise ValueError(f"Could not extract logits from model output: {model_output}")


def _convert_label_tensors_to_tags(
    class_labels: List[str],
    predicted_labels: "torch.Tensor",
    target_labels: "torch.Tensor",
    ignore_label: int = -100,
):
    # convert tensor to list
    predicted_labels = predicted_labels.detach().cpu().tolist()
    target_labels = target_labels.detach().cpu().tolist()
    predicted_labels = [
        [
            class_labels[pred_label]
            for (pred_label, target_label) in zip(prediction, target)
            if target_label != ignore_label
        ]
        for prediction, target in zip(predicted_labels, target_labels)
    ]
    target_labels = [
        [
            class_labels[target_label]
            for target_label in target
            if target_label != ignore_label
        ]
        for target in target_labels
    ]
    return predicted_labels, target_labels
