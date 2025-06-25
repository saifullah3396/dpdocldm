from __future__ import annotations

import re
from typing import Any, Dict


def _indent_string(s: str, ind: str) -> str:
    """
    Indents a given string by a specified prefix.

    Args:
        s (str): The input string.
        ind (str): The prefix to use for indentation.

    Returns:
        str: The indented string.
    """
    import textwrap

    return textwrap.indent(s, ind)


def _convert_to_snake_case(s: str) -> str:
    """
    Converts a camel case string to snake case.

    Args:
        s (str): The camel case string.

    Returns:
        str: The underscored lower case string.
    """
    return re.sub(r"([A-Z])", r"_\1", s).lower().lstrip("_")


def _format_recursive_dict(d: Dict[str, Any], depth: int = 0) -> str:
    """
    Recursively prints the arguments in a nested dictionary.

    Args:
        d (Dict[str, Any]): The input dictionary.
        depth (int): The current depth of recursion.

    Returns:
        str: The formatted string of arguments.
    """
    msg = ""
    for k, v in d.items():
        tabs = "  " * depth
        msg += f"{tabs}{k}: "
        if isinstance(v, dict):
            msg += "\n"
            msg += _format_recursive_dict(v, depth + 1)
        else:
            msg += f"{v}\n"
    return msg
