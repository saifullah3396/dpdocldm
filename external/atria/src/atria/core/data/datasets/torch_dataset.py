import dataclasses
from abc import abstractmethod
from typing import List

from atria.core.data.data_modules.dataset_info import AtriaDatasetInfo
from torch.utils.data import Dataset

import datasets

logger = datasets.logging.get_logger(__name__)


@dataclasses.dataclass
class AtriaTorchDatasetConfig:
    """BuilderConfig for FusionDataset"""

    name: str = "default"
    version: datasets.Version = datasets.Version("1.0.0")

    description: str = None
    homepage: str = None
    citation: str = None
    license: str = None

    data_dir: str = None
    data_url: str = None


class AtriaTorchDataset(Dataset):
    def __init__(
        self,
        config: AtriaTorchDatasetConfig,
    ):
        self._config = config

    @property
    def info(self) -> AtriaDatasetInfo:
        return self._info()

    @property
    def available_splits(self) -> List[datasets.Split]:
        return self._available_splits()

    @property
    def config(self) -> AtriaTorchDatasetConfig:
        return self._config

    def _info(self):
        return AtriaDatasetInfo(
            description=self.config.description,
            citation=self.config.citation,
            homepage=self.config.homepage,
            license=self.config.license,
            features=self._dataset_features(),
            dataset_name=self.__class__.__name__,
            config_name=self.config.name,
            version=self.config.version,
        )

    @abstractmethod
    def _dataset_features(self):
        raise NotImplementedError("Subclasses must implement _dataset_features()")

    @abstractmethod
    def _load_split(self, split: datasets.Split):
        raise NotImplementedError("Subclasses must implement _dataset_features()")

    @abstractmethod
    def _available_splits(self) -> List[datasets.Split]:
        raise NotImplementedError("Subclasses must implement available_splits()")
