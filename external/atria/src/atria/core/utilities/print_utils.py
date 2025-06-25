import dataclasses

import numpy as np
import torch
from atria.core.models.model_outputs import ModelOutput
from atria.core.utilities.logging import get_logger

logger = get_logger(__name__)


def _np_array_info_str(array: np.ndarray):
    return f"shape={array.shape}, type={array.dtype}, device={array.device}\nExample: {array[0] if len(array.shape) > 0 else array}"


def _tensor_info_str(tensor: torch.Tensor):
    return f"shape={tensor.shape}, type={tensor.dtype}, device={tensor.device}\nExample: {tensor[0] if len(tensor.shape) > 0 else tensor}"


def _list_element_info_str(l: list):
    return f"shape={len(l)}, type={type(l[0])}\nExample: {l[0]}"


def _print_batch_info(batch):
    logger.debug("Printing batch info:")
    logger.debug("Keys in batch: {}".format(batch.keys()))
    if isinstance(batch, dict):
        for key, value in batch.items():
            msg_str = f"Batch element={key}, "
            if isinstance(value, np.ndarray):
                logger.debug(msg_str + _np_array_info_str(value))
            elif isinstance(value, torch.Tensor):
                logger.debug(msg_str + _tensor_info_str(value))
            elif isinstance(value, list):
                if isinstance(value[0], np.ndarray):
                    logger.debug(msg_str + _np_array_info_str(value[0]))
                elif isinstance(value[0], torch.Tensor):
                    logger.debug(msg_str + _tensor_info_str(value[0]))
                else:
                    logger.debug(msg_str + _list_element_info_str(value))
            else:
                logger.debug(msg_str + f"type={type(value)}\nExample: {value}")
    elif isinstance(batch, list):
        sample = batch[0]
        for key, value in sample.items():
            msg_str = f"Batch element={key}, "
            if isinstance(value, np.ndarray):
                logger.debug(msg_str + _np_array_info_str(value))
            elif isinstance(value, torch.Tensor):
                logger.debug(msg_str + _tensor_info_str(value))
            elif isinstance(value, list):
                if isinstance(value[0], np.ndarray):
                    logger.debug(msg_str + _np_array_info_str(value[0]))
                elif isinstance(value[0], torch.Tensor):
                    logger.debug(msg_str + _tensor_info_str(value[0]))
                else:
                    logger.debug(msg_str + _list_element_info_str(value))
            else:
                logger.debug(msg_str + f"type={type(value)}\nExample: {value}")


def _print_output_info(output: ModelOutput):
    import numpy as np
    import torch

    for key, value in dataclasses.asdict(output).items():
        msg_str = f"Output element={key}, "
        if isinstance(value, np.ndarray):
            logger.debug(msg_str + _np_array_info_str(value))
        elif isinstance(value, torch.Tensor):
            logger.debug(msg_str + _tensor_info_str(value))
        elif isinstance(value, list):
            if isinstance(value[0], np.ndarray):
                logger.debug(msg_str + _np_array_info_str(value[0]))
            elif isinstance(value[0], torch.Tensor):
                logger.debug(msg_str + _tensor_info_str(value[0]))
            else:
                logger.debug(msg_str + _list_element_info_str(value))
        else:
            logger.debug(msg_str + f"type={type(value)}\nExample: {value}")
