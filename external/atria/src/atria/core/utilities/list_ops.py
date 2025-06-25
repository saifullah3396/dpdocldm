from __future__ import annotations

from typing import Any, Dict, Generator, List, Union

import torch


def _format_to_list(obj: Any) -> List[Any]:
    """
    Ensures the input object is a list. If not, it converts it to a list.

    Args:
        obj (Any): The input object.

    Returns:
        List[Any]: The input object as a list.
    """
    if not isinstance(obj, list):
        obj = [obj]
    return obj


def _list_of_dict_to_dict_of_list(
    list_of_dict: List[Dict[str, torch.Tensor]]
) -> Dict[str, torch.Tensor]:
    """
    Concatenates a list of dictionaries into a single dictionary with concatenated tensors.

    Args:
        list_dict (List[Dict[str, torch.Tensor]]): The list of dictionaries.

    Returns:
        Dict[str, torch.Tensor]: The concatenated dictionary.
    """
    return {k: [dic[k] for dic in list_of_dict] for k in list_of_dict[0]}


def _drange(
    min_val: Union[int, float], max_val: Union[int, float], step_val: Union[int, float]
) -> Generator[Union[int, float], None, None]:
    """
    Generates a range of numbers with a specified step value.

    Args:
        min_val (Union[int, float]): The minimum value.
        max_val (Union[int, float]): The maximum value.
        step_val (Union[int, float]): The step value.

    Yields:
        Generator[Union[int, float], None, None]: The range of numbers.
    """
    curr = min_val
    while curr < max_val:
        yield curr
        curr += step_val
