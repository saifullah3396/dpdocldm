import numbers
from typing import Any, Dict, List, Mapping, Optional, Union

import numpy as np
import torch
from atria.core.utilities.logging import get_logger, warn_once
from atria.core.utilities.typing import BatchDict

logger = get_logger(__name__)


def _flatten_list(nested_list):
    flattened = []
    for element in nested_list:
        if isinstance(element, list):
            flattened.extend(_flatten_list(element))
        else:
            flattened.append(element)
    return flattened


def _filter_batch_with_keys(
    batch: BatchDict, batch_filter_key_map: Mapping[str, str]
) -> BatchDict:
    if batch_filter_key_map is None:
        return batch

    filtered_batch = {}
    for target_key_in_batch, mapped_key_in_batch in batch_filter_key_map.items():
        if mapped_key_in_batch not in batch:
            warn_once(
                logger,
                f"Mapping '{target_key_in_batch}' (target_key_in_batch) -> '{mapped_key_in_batch}' (mapped_key_in_batch) in 'batch_filter_key_map' is invalid. "
                f"Make sure the key '{mapped_key_in_batch}' (mapped_key_in_batch) is present in the batch: {list(batch.keys())}. Ignoring... ",
            )
            continue
        filtered_batch[target_key_in_batch] = batch[mapped_key_in_batch]
    return filtered_batch


def _list_to_tensor(list_of_items, dtype=None):
    import numpy as np
    import torch

    # # replace None with -100 for word ids if they are present
    # list_of_items[list_of_items == None] = -100
    if len(list_of_items) == 0:
        return list_of_items
    if isinstance(list_of_items, torch.Tensor):  # if it is already a tensor
        output = list_of_items
    elif isinstance(list_of_items[0], numbers.Number):  # if it is a list of number type
        output = torch.tensor(list_of_items)
    elif isinstance(list_of_items[0], torch.Tensor):  # if it is a list of torch tensors
        output = torch.stack(list_of_items)
    elif isinstance(list_of_items[0], np.ndarray):  # if it is a list of numpy arrays
        output = torch.from_numpy(np.array(list_of_items))
    elif isinstance(list_of_items[0], str):  # if it is a list of strings, leave it
        output = list_of_items
    elif isinstance(list_of_items, list):  # if it is a list of list
        list_of_items = [_list_to_tensor(l) for l in list_of_items]
        if isinstance(list_of_items[0], torch.Tensor):
            output = torch.stack([_list_to_tensor(l) for l in list_of_items])
        else:
            output = list_of_items
    else:
        output = torch.tensor(list_of_items)
    if isinstance(output, torch.Tensor):
        output = output.to(dtype)
    return output


def _format_tensor_list_to_tensor(array):
    tensor_shapes = [t.shape for t in array]
    if tensor_shapes[0] == torch.Size([]):
        return array
    elif tensor_shapes.count(tensor_shapes[0]) == len(
        tensor_shapes
    ):  # all tensors have equal shape, we make a batch tensor
        return torch.stack(array)
    else:  # otherwise we return a list of tensors, this is not tested!
        return array


def _format_array_list_to_tensor(array):
    array_shapes = [a.shape for a in array]
    if array_shapes.count(array_shapes[0]) == len(
        array_shapes
    ):  # all tensors have equal shape, we make a batch tensor
        return torch.from_numpy(np.array(array))
    else:  # otherwise we return a list of tensors, this is not tested!
        return [torch.from_numpy(a) for a in array]


def _object_list_to_maybe_tensor(
    features: List[Union[int, float, str, torch.Tensor, np.ndarray, Any]]
) -> torch.Tensor:
    try:
        if isinstance(features, list) and len(features) == 0:
            return features
        if isinstance(features, torch.Tensor):
            return features
        elif isinstance(features[0], numbers.Number):
            return torch.tensor(features)
        elif isinstance(features[0], torch.Tensor):
            return _format_tensor_list_to_tensor(features)
        elif isinstance(features[0], np.ndarray):
            return _format_array_list_to_tensor(features)
        elif isinstance(features[0], list):  # if its a list
            outputs = [_object_list_to_maybe_tensor(f) for f in features]
            if isinstance(outputs[0], torch.Tensor):
                # check if all outputs have equal size to be stacked
                if all(
                    [len(outputs[0]) == len(outputs[i]) for i in range(1, len(outputs))]
                ):
                    return torch.stack(outputs)
                else:
                    return outputs
            return outputs
        else:
            return features
    except Exception as e:
        logger.exception(
            f"Exception raised while trying to convert object list to tensor: {features}. Error: {e}"
        )
        raise e


class SimpleBatchDataCollator:
    """
    Data collator for converting data in the batch to a dictionary of pytorch tensors.
    """

    def __call__(self, features):
        batch = {}
        for k in features[0].keys():
            batch[k] = [sample[k] for sample in features]
        return batch


class BatchToTensorDataCollator:
    """
    Data collator for converting data in the batch to a dictionary of pytorch tensors.
    """

    def __init__(
        self,
        batch_filter_key_map: Optional[Dict[str, str]] = None,
    ):
        self.batch_filter_key_map = batch_filter_key_map

    def __call__(self, features: Union[list, Dict]):
        if isinstance(features, list):
            features = _flatten_list(features)

            # convert list of dictionaries to a batch of key-wise lists
            collated_batch = {
                k: [sample[k] for sample in features] for k in features[0].keys()
            }
        elif isinstance(features, dict):
            collated_batch = features
        else:
            raise ValueError(
                "The input features must be either a list of dictionaries or a dictionary."
            )

        # filter batch with keys and re-map accordingly
        collated_batch = _filter_batch_with_keys(
            collated_batch,  # key is used as default
            batch_filter_key_map={
                "__index__": "__index__",
                "__key__": "__key__",
                **(
                    self.batch_filter_key_map
                    if self.batch_filter_key_map is not None
                    else {}
                ),
            },
        )

        # convert all objects in batch to torch tensors if possible
        for k, v in collated_batch.items():
            if k in ["__index__", "__key__"]:
                collated_batch[k] = v
                continue

            try:
                collated_batch[k] = _object_list_to_maybe_tensor(v)
            except Exception as e:
                logger.exception(
                    f"Exception raised while trying to collate feature into batch of tensors: {k}."
                    f" Did you forget to set transforms on the input data? "
                    f"Error: {e}"
                )
                exit()

        if len(collated_batch) == 0:
            raise RuntimeError(
                "No data keys were found in the batch after filtering. Please check the batch_filter_key_map."
            )

        return collated_batch
