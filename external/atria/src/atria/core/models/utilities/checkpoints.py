from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from atria.core.utilities.logging import get_logger
from fsspec.core import url_to_fs
from fsspec.implementations.local import AbstractFileSystem
from torch import nn

DEFAULT_STATE_DICT_KEY = "state_dict"

logger = get_logger(__name__)


def _get_filesystem(path: Path, **kwargs: Any) -> AbstractFileSystem:
    fs, _ = url_to_fs(str(path), **kwargs)
    return fs


def _filter_keys(checkpoint, keys: List[str]):
    checkpoint_filtered = {}
    for state in checkpoint:
        updated_state = state
        for key in keys:
            if key in updated_state:
                updated_state = updated_state.replace(key, "")
        checkpoint_filtered[updated_state] = checkpoint[state]
    return checkpoint_filtered


def _prepend_keys(checkpoint, keys: List[str]):
    checkpoint_prepended = {}
    for state in checkpoint:
        updated_state = state
        for key in keys:
            if key not in updated_state:
                updated_state = key + updated_state

        checkpoint_prepended[updated_state] = checkpoint[state]
    return checkpoint_prepended


def _replace_keys(checkpoint, key: str, replacement: str):
    checkpoint_filtered = {}
    for state in checkpoint:
        updated_state = state
        if key in updated_state:
            updated_state = updated_state.replace(key, replacement)
        checkpoint_filtered[updated_state] = checkpoint[state]
    return checkpoint_filtered


def _filter_with_prefix(checkpoint, prefix_key: str):
    checkpoint_filtered = {}
    for state in checkpoint:
        if state.startswith(prefix_key):
            checkpoint_filtered[state[len(prefix_key) + 1 :]] = checkpoint[state]
    return checkpoint_filtered


def _load_checkpoint(
    path_or_url: Union[str, Path],
) -> Any:
    """Loads a checkpoint.

    Args:
        path_or_url: Path or URL of the checkpoint.
        map_location: a function, ``torch.device``, string or a dict specifying how to remap storage locations.
    """
    import ignite.distributed as idist
    import torch

    map_location = idist.device()
    if idist.get_world_size() > 1:
        map_location = "cpu"

    if not isinstance(path_or_url, (str)):
        # any sort of BytesIO or similar
        return torch.load(path_or_url, map_location=map_location)
    if str(path_or_url).startswith("http"):
        return torch.hub.load_state_dict_from_url(
            str(path_or_url), map_location=map_location
        )
    fs = _get_filesystem(path_or_url)
    try:
        with fs.open(path_or_url, "rb") as f:
            return torch.load(f, map_location=map_location)
    except Exception as e:
        logger.error(f"Error loading the checkpoint: {e}")
        raise e


def _load_checkpoint_into_model(
    model: nn.Module,
    checkpoint: Dict[str, Any],
    model_state_dict_path: Optional[str],
    strict: bool = True,
):
    from atria.core.models.task_modules.atria_task_module import TorchModelDict

    def get_available_keys(model: Union[nn.Module, Any]):
        if isinstance(model, nn.Module):
            return list(model.state_dict().keys())
        else:
            return list(model.__dict__.keys())

    if model_state_dict_path is not None:
        for path in model_state_dict_path.split("."):
            assert hasattr(
                model, path
            ), f"Target key {path} not found in the model. Available keys = {get_available_keys(model)}"
            model = getattr(model, path)

    if isinstance(model, TorchModelDict):
        raise RuntimeError(
            "Cannot load a checkpoint into a TorchModelDict. Please load the checkpoint into the model directly."
            "In TorchModelDict the checkpoint can be loaded into the following keys:"
            "- trainable_models"
            "- non_trainable_models"
        )

    state_dict = model.state_dict()
    for checkpoint_key in list(checkpoint.keys()):
        checkpoint_key_shape = checkpoint[checkpoint_key].shape
        if checkpoint_key not in state_dict:
            logger.warning(
                f"Key {checkpoint_key} not found in the model state dict. Skipping."
            )
            continue
        model_key_shape = state_dict[checkpoint_key].shape
        if checkpoint_key_shape != model_key_shape:
            logger.warning(
                f"Shape mismatch for key {checkpoint_key}: checkpoint shape = {checkpoint_key_shape}, model shape = {model_key_shape}"
            )
            checkpoint.pop(checkpoint_key)

    keys = model.load_state_dict(checkpoint, strict=strict)
    if not strict:
        if keys.missing_keys:
            logger.warning(
                f"Found keys that are in the model state dict but not in the checkpoint: {keys.missing_keys}"
            )
        if keys.unexpected_keys:
            logger.warning(
                f"Found keys that are not in the model state dict but in the checkpoint: {keys.unexpected_keys}"
            )
