from typing import Callable, List, Optional

from atria.core.data.utilities import _unwrap_dataset
from torch.utils.data import Dataset


class TransformedDataset(Dataset):
    def __init__(
        self,
        dataset,
        transforms: Optional[List[Callable]] = None,
    ):
        self._dataset = dataset
        self._transforms = transforms
        self._transforms_enabled = True

    @property
    def dataset(self):
        return self._dataset

    def disable_transforms(self):
        self._transforms_enabled = False
        self._dataset._apply_postprocessing = False

    def enable_transforms(self):
        self._transforms_enabled = True
        self._dataset._apply_postprocessing = True

    def __getitem__(self, index):
        sample = self._dataset[index]

        if self._transforms_enabled and self._transforms is not None:
            sample = self._transforms(sample)

        return sample

    def __len__(self):
        return self._dataset.__len__()

    def __repr__(self):
        return self._dataset.__repr__()

    def close(self):
        unwrapped_dataset = _unwrap_dataset(self._dataset)
        if hasattr(unwrapped_dataset, "_close"):
            unwrapped_dataset._close()
        if hasattr(unwrapped_dataset, "close"):
            unwrapped_dataset.close()
