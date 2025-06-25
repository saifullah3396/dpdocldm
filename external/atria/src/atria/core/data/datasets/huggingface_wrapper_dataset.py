import dataclasses

from atria.core.data.datasets.torch_dataset import (
    AtriaTorchDataset,
    AtriaTorchDatasetConfig,
)
from atria.core.data.utilities import _get_default_download_config

import datasets
from datasets import load_dataset

logger = datasets.logging.get_logger(__name__)


@dataclasses.dataclass
class AtriaHuggingfaceWrapperDatasetConfig(AtriaTorchDatasetConfig):
    hf_repo: str = None


class AtriaHuggingfaceWrapperDataset(AtriaTorchDataset):
    def _load_split(self, split: datasets.Split, cache_dir: str):
        self._dataset = load_dataset(
            self.config.hf_repo,
            split=split,
            cache_dir=cache_dir,
            download_config=_get_default_download_config(),
        )

    def __getitem__(self, index):
        return self._dataset[index]

    def __len__(self):
        return len(self._dataset)
