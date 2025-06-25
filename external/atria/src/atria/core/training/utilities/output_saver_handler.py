from __future__ import annotations

from pathlib import Path
from typing import List, Union

import h5py
import numpy as np
import torch
from ignite.engine import Engine

from atria.core.constants import DataKeys
from atria.core.utilities.logging import get_logger

# by default save these keys
DEFAULT_KEYS_TO_SAVE = [
    DataKeys.LABEL,
    DataKeys.PRED,
    DataKeys.LOGITS,
]


def update_dataset_at_indices(
    hf: h5py.File,
    key: str,
    indices: np.array,
    data,
    maxshape=(None,),
    overwrite: bool = False,
):
    if key not in hf:
        hf.create_dataset(
            key, data=data, compression="gzip", chunks=True, maxshape=maxshape
        )
    else:
        if maxshape[1:] != hf[key].shape[1:]:
            logger = get_logger(__name__)
            if overwrite:
                logger.info(
                    f"Reinitializing data due to shape mismatch for key={key} since overwrite is set to True."
                )
                del hf[key]
                hf.create_dataset(
                    key, data=data, compression="gzip", chunks=True, maxshape=maxshape
                )
            else:
                logger.error(
                    f"Data overwrite is set to False but there is mismatch between data shapes for key = {key}"
                )
                exit()

        max_len = indices.max() + 1
        if len(hf[key]) < max_len:
            hf[key].resize((indices.max() + 1), axis=0)
            hf[key][indices] = data
        elif overwrite:
            hf[key][indices] = data


class ModelForwardDiskSaver:
    def __init__(
        self,
        output_dir: Union[Path, str],
        checkpoint_file_path: str,
        extra_keys_to_save: List[str] = [],
    ):
        self._output_file = (
            Path(output_dir) / f"model_outputs-{Path(checkpoint_file_path).name}.h5"
        )
        self._extra_keys_to_save = extra_keys_to_save

    def add_key_from_output(
        self, engine: Engine, hf: h5py.File, key: str, indices: np.array
    ):
        # add keys in output if available
        data = None
        if key in engine.state.output:
            data = engine.state.output[key]
        elif key in engine.state.batch:
            data = engine.state.batch[key]

        if data is not None:
            if isinstance(data, torch.Tensor):
                data = data.detach().cpu().numpy()
            max_shape = (None,)
            if isinstance(data, np.ndarray):
                max_shape = (None, *data.shape[1:])
            elif isinstance(data, list):
                max_shape = (None,)
            update_dataset_at_indices(
                hf,
                key=key,
                indices=indices,
                data=data,
                maxshape=max_shape,
                overwrite=True,
            )

    def __call__(self, engine: Engine) -> None:
        hf = h5py.File(self._output_file, "a")

        # get data indices
        indices = engine.state.batch[DataKeys.INDEX]
        indices = np.array(indices)

        # create index dataset
        update_dataset_at_indices(hf, key="index", indices=indices, data=indices)

        # save keys
        for key in DEFAULT_KEYS_TO_SAVE + self._extra_keys_to_save:
            # add labels
            self.add_key_from_output(engine, hf, key, indices)

        hf.close()
