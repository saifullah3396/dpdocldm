import glob
import os
from pathlib import Path
from typing import List, Optional, Tuple, Union

import datasets
from atria.core.data.data_modules.dataset_cacher.shard_list_datasets import (
    MsgpackShardListDataset,
)
from atria.core.data.data_modules.dataset_info import AtriaDatasetInfo
from atria.core.data.data_transforms import DataTransformsDict
from atria.core.utilities.logging import get_logger
from torch.utils.data import Dataset

logger = get_logger(__name__)

ATRIA_CACHE_DIR = os.environ.get("ATRIA_CACHE_DIR", Path.home() / ".cache/atria/")


class FileUrlProvider:
    def __init__(
        self,
        cache_dir: str,
        cache_file_name: Optional[str] = None,
        file_format: str = "msgpack",
    ) -> None:
        """
        Initialize the DatasetReader.

        Args:
            cache_dir (str): Directory where cached dataset files are stored.
            file_format (str): Format of the cached dataset files (default is "msgpack").
        """
        self._cache_dir: Path = Path(cache_dir)
        self._cache_file_name: Optional[str] = cache_file_name
        self._file_format: str = file_format

    def get_shard_file_urls(self, split: str) -> List[str]:
        """
        Prepare file URLs for a given dataset split.

        Args:
            split (str): The dataset split (e.g., "train", "test").

        Returns:
            List[str]: List of file URLs matching the split pattern.
        """
        file_pattern = (
            f"{split}-*.{self._file_format}"
            if self._cache_file_name is None
            else f"{self._cache_file_name}-{split}-*.{self._file_format}"
        )
        return glob.glob(str(self._cache_dir / file_pattern))

    def get_dataset_info_path(self) -> List[str]:
        return str(
            self._cache_dir / f"{self._cache_file_name}_dataset_info.json"
            if self._cache_file_name
            else self._cache_dir / "atria_dataset_info.json"
        )


class DatasetCacher:
    """
    Dataset cacher class to cache and read datasets.

    Attributes:
        cache_type (Optional[str]): Type of the dataset. Defaults to msgpack.
        cache_dir (str): Directory to cache the dataset.
        cache_file_name (str): Name of the cache file for msgpack-based caching.
        preprocessing_batch_size (int): Batch size for preprocessing.
        num_proc (int): Number of processes to use for processing data for cache.
        max_shard_size (int): Maximum shard size for the dataset.
        preprocess_data_transforms_config (DataTransformsConfig): Arguments for preprocessing data transforms.
    """

    def __init__(
        self,
        cache_type: Optional[str] = "msgpack",
        cache_dir: Optional[str] = None,
        cache_file_name: str = "default",
        preprocessing_batch_size: int = 1,
        num_processes: int = 4,
        max_shard_size: int = 100000,
        preprocess_data_transforms: Optional[DataTransformsDict] = None,
        attach_features_with_key: Optional[str] = None,
    ):
        self._cache_type = cache_type
        self._cache_dir = cache_dir or os.path.join(ATRIA_CACHE_DIR, "datasets")
        self._cache_file_name = cache_file_name
        self._preprocessing_batch_size = preprocessing_batch_size
        self._num_processes = num_processes
        self._max_shard_size = max_shard_size
        self._preprocess_data_transforms = (
            preprocess_data_transforms or DataTransformsDict()
        )
        self._attach_features_with_key = attach_features_with_key

        # Post-initialization processing to validate cache_type
        assert self._cache_type in [
            "msgpack",
            "webdataset",
        ], f"Invalid dataset type {self._cache_type}. Possible choices are ['msgpack', 'webdataset']"

        # update self._cache_type
        self._cache_type = (
            "tar" if self._cache_type == "webdataset" else self._cache_type
        )

        # update cache dir
        self._cache_dir = Path(self._cache_dir) / self._cache_type

        # create file url provider
        self._file_url_provider = FileUrlProvider(
            cache_dir=self._cache_dir,
            cache_file_name=self._cache_file_name,
            file_format=self._cache_type,
        )

    @property
    def cache_dir(self):
        return self._cache_dir

    @cache_dir.setter
    def cache_dir(self, cache_dir: str):
        self._cache_dir = Path(cache_dir)
        self._file_url_provider._cache_dir = self._cache_dir

    def load_cache_dir_from_builder(self, builder: datasets.GeneratorBasedBuilder):
        self._cache_dir = Path(builder.cache_dir)
        self._file_url_provider._cache_dir = self._cache_dir

    def read_dataset_from_cache(
        self,
        split: datasets.Split,
        streaming_mode: bool = False,
        dataset_key_filter: Optional[List[str]] = None,
        only_load_features: bool = False,
    ) -> Optional[Tuple[datasets.Dataset, AtriaDatasetInfo]]:
        """
        Reads a dataset split from the cache.

        Args:
            split (datasets.Split): The dataset split to read.
            cache_dir (Optional[str]): Directory to read the cache from.

        Returns:
            Optional[datasets.Dataset]: The dataset split if available, else None.
        """

        from atria.core.data.data_modules.dataset_cacher.dataset_reader import (
            DatasetReader,
        )

        dataset_reader = DatasetReader(
            cache_dir=self._cache_dir,
            file_url_provider=self._file_url_provider,
            file_format=self._cache_type,
            dataset_key_filter=dataset_key_filter,
            only_load_features=only_load_features,
        )

        if dataset_reader.is_available(split):
            dataset, dataset_info = dataset_reader.read_dataset_split(
                split, streaming_mode=streaming_mode
            )

            if self._attach_features_with_key is not None and isinstance(
                dataset, MsgpackShardListDataset
            ):
                logger.info(
                    "Attaching features to dataset with key = {}".format(
                        self._attach_features_with_key
                    )
                )
                dataset.load_features(self._attach_features_with_key)

            return dataset, dataset_info

    def write_dataset_to_cache(
        self,
        dataset: Union[datasets.GeneratorBasedBuilder, Dataset],
        split: datasets.Split,
        dataset_info: AtriaDatasetInfo,
    ) -> Tuple[datasets.Dataset, AtriaDatasetInfo]:
        """
        Writes a dataset split to the cache.

        Args:
            dataset (Union[datasets.GeneratorBasedBuilder, Dataset]): The dataset to write.
            split (datasets.Split): The dataset split to write.
            dataset_info (AtriaDatasetInfo): Information about the dataset.
            cache_dir (Optional[str]): Directory to write the cache to.

        Returns:
            Tuple[datasets.Dataset, AtriaDatasetInfo]: The written dataset split and its info.
        """

        from atria.core.data.data_modules.dataset_cacher.dataset_writer import (
            DatasetWriter,
        )

        dataset_writer = DatasetWriter(
            cache_dir=self._cache_dir,
            file_url_provider=self._file_url_provider,
            dataset_info=dataset_info,
            cache_file_name=self._cache_file_name,
            preprocess_data_transforms=(
                self._preprocess_data_transforms.train
                if split == datasets.Split.TRAIN
                else self._preprocess_data_transforms.evaluation
            ),
            file_format=self._cache_type,
            maxcount=self._max_shard_size,
            num_processes=self._num_processes,
            is_preprocessing_batched=self._preprocessing_batch_size > 1,
            preprocessing_batch_size=self._preprocessing_batch_size,
        )
        dataset_writer.write_dataset_split(split=split, dataset=dataset)
