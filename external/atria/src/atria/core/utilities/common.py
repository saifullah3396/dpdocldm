from __future__ import annotations

import inspect
import json
from functools import partial
from importlib import import_module
from typing import Any

from atria.core.utilities.logging import get_logger
from dacite import Any, Type

logger = get_logger(__name__)


def _resolve_module_from_path(module_path: str):
    path = module_path.rsplit(".", 1)
    if len(path) == 1:
        raise ValueError(
            f"Invalid module path: {module_path}. It should be in the form 'module_name.class_name'."
        )
    module_name, class_name = path
    module = import_module(module_name)
    return getattr(module, class_name)


def _get_parent_module(module_name: str):
    # Check if the module name has a parent (e.g., 'parent.child' -> 'parent')
    parent_module_name = (
        module_name.rsplit(".", 1)[0] if "." in module_name else module_name
    )

    return parent_module_name


def _validate_keys_in_output(keys, output, metric_name=None):
    if not isinstance(keys, list):
        keys = keys[0]

    for key in keys:
        assert (
            key in output
        ), f"Key '{key}' required by metric '{metric_name}' not found in output: {output.keys()}"


def _unwrap_partial(partial_object: partial) -> Any:
    assert isinstance(partial_object, partial)
    object = partial_object.func
    while hasattr(object, "__wrapped__"):
        object = object.__wrapped__
    return object


def _validate_partial_class(object: Any, target_class: Type[Any], object_name: str):
    assert isinstance(
        object, partial
    ), f"{object_name} must be a partial object of class {target_class} for late initialization"
    unwrapped = _unwrap_partial(object)
    if not callable(unwrapped):
        assert issubclass(
            unwrapped,
            target_class,
        ), f"{object_name} partial class must be a subclass of {target_class} or a callable function"


def _msg_with_separator(msg: str, separator: str = "=") -> str:
    separator = separator * (len(msg) + 8)
    return f"{separator} {msg} {separator}"


def _pretty_print_dict(x):
    return json.dumps(x, default=lambda o: getattr(o, "__dict__", str(o)), indent=4)


def _get_possible_args(func):
    return inspect.signature(func).parameters


def _get_required_args(func):
    # Get the signature of the function
    sig = inspect.signature(func)
    # Iterate over parameters and filter out those without default values
    required_args = [
        param.name
        for param in sig.parameters.values()
        if param.default == inspect.Parameter.empty
        and param.kind
        in [inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY]
    ]
    return required_args
