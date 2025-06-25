# Copyright (c) 2024 Massachusetts Institute of Technology
# SPDX-License-Identifier: MIT
# pyright: reportUnnecessaryTypeIgnoreComment=false
import inspect
from typing import Callable, cast

from hydra_zen.third_party.pydantic import (
    _T,
    _constructor_as_fn,
    _default_parser,
    _get_signature,
)


def atria_pydantic_parser(
    target, *, parser: Callable[[_T], _T] = _default_parser
) -> _T:
    try:
        if inspect.isbuiltin(target):
            return target

        if not (_get_signature(target)):
            return target

        if inspect.isclass(target):
            return cast(_T, parser(_constructor_as_fn(target)))
        return parser(target)
    except Exception as e:
        raise ValueError(f"Failed to parse target={target} with pydantic: {e}")
