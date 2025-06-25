import glob
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import datasets
import pandas as pd
import torch
from atria.core.data.data_modules.dataset_info import AtriaDatasetInfo
from atria.core.utilities.logging import get_logger
from datadings.reader import MsgpackReader as MsgpackFileReader
from torch.utils.data import Dataset
from wids import ShardListDataset

logger = get_logger(__name__)


class MsgpackShardListDataset(Dataset):
    """
    A mix of torch dataset and huggingface dataset info backed by a msgpack file or a list of msgpack files.
    This dataset is loaded by the MsgpackBuilder.
    """

    def __init__(
        self,
        files: List[str],
        info: AtriaDatasetInfo,
        split: datasets.Split,
        transformations: List[Callable[[Dict[str, Any]], Dict[str, Any]]],
        dataset_key_filter: Optional[List[str]] = None,
        only_load_features: bool = False,
    ) -> None:
        """
        Args:
            files (List[str]): List of msgpack file paths.
            info (AtriaDatasetInfo): Dataset information.
            split (datasets.Split): Dataset split.
            transformations (List[Callable[[Dict[str, Any]], Dict[str, Any]]]): List of transformations to apply.
        """
        import numpy as np
        from datadings.reader import MsgpackReader as MsgpackFileReader

        self._data_files = []
        for file in sorted(files):
            if "features" in file:
                continue
            self._data_files.append(file)
        self._data = [MsgpackFileReader(f) for f in self._data_files]
        self._info = info
        self._split = split
        self._transformations = transformations
        self._dataset_key_filter = dataset_key_filter
        self._only_load_features = only_load_features

        # map indices from multiple readers
        self._cumulative_sizes: List[int] = []
        self._total_size: int = 0

        for data in self._data:
            self._total_size += len(data)
            self._cumulative_sizes.append(self._total_size)
            data._close()
        self._cumulative_sizes = np.array(self._cumulative_sizes)
        self._features_metadata = None
        self._apply_postprocessing = True

    @property
    def info(self) -> AtriaDatasetInfo:
        """Returns the dataset information."""
        return self._info

    @property
    def split(self) -> datasets.Split:
        """Returns the dataset split."""
        return self._split

    def load_features(self, features_key: str) -> None:
        """
        Attach features to the dataset if available.
        """
        dir_path = Path(self._data_files[0]).parent
        file_search_name = Path(self._data_files[0]).name.split("-")[0]
        features_metadata_file_path = f"{str(dir_path)}/{file_search_name}-*{self._split}-*-{features_key}-metadata.csv"
        features_file_paths = f"{str(dir_path)}/{file_search_name}-*{self._split}-*-{features_key}.msgpack"

        try:
            self._features_data = {
                f: MsgpackFileReader(f)
                for f in glob.glob(str(dir_path / features_file_paths))
            }
            features_metadata_file_path = glob.glob(
                str(dir_path / features_metadata_file_path)
            )
            if len(features_metadata_file_path) == 0:
                return
            assert (
                len(features_metadata_file_path) == 1
            ), f"Multiple features metadata files found: {features_metadata_file_path}"

            logger.info(
                f"Loading features metadata from {features_metadata_file_path[0]}"
            )
            self._features_metadata = pd.read_csv(features_metadata_file_path[0])
            self._features_metadata = self._features_metadata.loc[
                self._features_metadata["index"]
            ]
            self._features_metadata["features_path"] = self._features_metadata[
                "features_path"
            ].apply(lambda x: str(dir_path / Path(x).name))
        except Exception as e:
            logger.warning(f"Failed to load features: {e}")

    def add_transform(
        self, transform: Callable[[Dict[str, Any]], Dict[str, Any]]
    ) -> "MsgpackShardListDataset":
        """
        Add a transformation to the dataset.

        Args:
            transform (Callable[[Dict[str, Any]], Dict[str, Any]]): Transformation function to add.

        Returns:
            MsgpackShardListDataset: The dataset with the added transformation.
        """
        self._transformations.append(transform)
        return self

    def get_shard(self, index: int) -> Union[MsgpackFileReader, int, str]:
        """
        Get the shard and the corresponding element index.

        Args:
            index (int): Index of the element.

        Returns:
            Tuple[MsgpackFileReader, int, str]: The shard, inner index, and URL.
        """
        import numpy as np

        shard_index = np.searchsorted(self._cumulative_sizes, index, side="right")

        if shard_index == 0:
            inner_index = index
        else:
            inner_index = index - self._cumulative_sizes[shard_index - 1]

        shard = self._data[shard_index]
        url = self._data_files[shard_index]
        return shard, inner_index, url

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get the element at the specified index.

        Args:
            index (int): Index of the element.

        Returns:
            Dict[str, Any]: The element at the specified index.
        """
        if self._only_load_features:
            if isinstance(index, torch.Tensor):
                index = index.item()
            features_row = self._features_metadata.iloc[index]
            features_path, feature_index = (
                features_row["features_path"],
                features_row["feature_index"],
            )
            features = self._features_data[features_path][feature_index]
            sample = {"__index__": index, **features}
            return features

        shard, inner_idx, url = self.get_shard(index)
        sample = shard[inner_idx]

        sample["__index__"] = index
        sample["__shard__"] = url
        sample["__shardindex__"] = inner_idx

        # if dataset key filter is provided, filter out the keys that are not in the filter
        if self._dataset_key_filter is not None:
            self._dataset_key_filter = [
                f"{key}.mp".format(key=key) if not key.endswith(".mp") else key
                for key in self._dataset_key_filter
            ]
            if "__index__.mp" not in self._dataset_key_filter:
                self._dataset_key_filter.append("__index__.mp")
            if "__key__" not in self._dataset_key_filter:
                self._dataset_key_filter.append("__key__")

            sample = {k: v for k, v in sample.items() if k in self._dataset_key_filter}

        if not self._apply_postprocessing:
            return sample

        # load features if available
        if self._features_metadata is not None and len(self._features_metadata) > 0:
            # these are post-processed features that are stored in a separate file
            # we load these additional features here and make sure they are attached to the correct sample
            # we also make sure that the index and key are consistent
            if isinstance(index, torch.Tensor):
                index = index.item()
            features_row = self._features_metadata.iloc[index]
            features_path, feature_index = (
                features_row["features_path"],
                features_row["feature_index"],
            )
            features = self._features_data[features_path][feature_index]

            assert features["__index__"] == index, f"{features['__index__']} != {index}"
            assert (
                features["__key__"] == sample["__key__"]
            ), f"{features['__key__']} != {sample['__key__']}"
            sample.update(features)

        for transform in self._transformations:
            sample = transform(sample)

        return sample

    def __len__(self) -> int:
        """
        Get the total number of elements in the dataset.

        Returns:
            int: Total number of elements.
        """
        return self._total_size

    def __repr__(self) -> str:
        """
        Get the string representation of the dataset.

        Returns:
            str: String representation of the dataset.
        """
        return f"Dataset({{\n    features: {list(self._info.features.keys())},\n    num_rows: {self._total_size}\n}})"

    def _close(self) -> None:
        """Close all data readers."""
        for d in self._data:
            d._close()
        if hasattr(self, "_features_data"):
            for d in self._features_data.values():
                d._close()

    def close(self) -> None:
        """Close all data readers."""
        self._close()


class MsgpackListDataset(MsgpackShardListDataset):
    def __init__(
        self,
        files: List[str],
        info: AtriaDatasetInfo,
        split: datasets.Split,
        transformations: List[Callable[[Dict[str, Any]], Dict[str, Any]]],
    ) -> None:
        """
        Args:
            files (List[str]): List of msgpack file paths.
            info (AtriaDatasetInfo): Dataset information.
            split (datasets.Split): Dataset split.
            transformations (List[Callable[[Dict[str, Any]], Dict[str, Any]]]): List of transformations to apply.
        """
        import numpy as np
        from datadings.reader import MsgpackReader as MsgpackFileReader

        self._data_files = []
        for file in sorted(files):
            if "features" in file:
                continue
            self._data_files.append(file)
        self._data = [MsgpackFileReader(f) for f in self._data_files]
        self._info = info
        self._split = split
        self._transformations = transformations

        # map indices from multiple readers
        self._cumulative_sizes: List[int] = []
        self._total_size: int = 0

        for data in self._data:
            self._total_size += len(data)
            self._cumulative_sizes.append(self._total_size)
            data._close()
        self._cumulative_sizes = np.array(self._cumulative_sizes)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """
        Get the element at the specified index.

        Args:
            index (int): Index of the element.

        Returns:
            Dict[str, Any]: The element at the specified index.
        """
        shard, inner_idx, url = self.get_shard(index)
        sample = shard[inner_idx]

        sample["__index__"] = index
        sample["__shard__"] = url
        sample["__shardindex__"] = inner_idx

        for transform in self._transformations:
            sample = transform(sample)

        return sample

    def _close(self) -> None:
        """Close all data readers."""
        for d in self._data:
            d._close()


class TarShardListDataset(ShardListDataset):
    """
    A mix of torch dataset and huggingface dataset info backed by a msgpack file or a list of msgpack files.
    This dataset is loaded by the MsgpackBuilder.
    """

    def __init__(
        self,
        shards: List[str],
        info: Optional[AtriaDatasetInfo] = None,
        split: Optional[datasets.Split] = None,
        cache_size: int = int(1e12),
        cache_dir: Optional[str] = None,
        lru_size: int = 10,
        dataset_name: Optional[str] = None,
        localname: Optional[str] = None,
        transformations: Union[
            str, List[Callable[[Dict[str, Any]], Dict[str, Any]]]
        ] = "PIL",
        keep: bool = False,
        base: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Args:
            shards (List[str]): List of shard file paths.
            info (Optional[AtriaDatasetInfo]): Dataset information.
            split (Optional[datasets.Split]): Dataset split.
            cache_size (int): Cache size.
            cache_dir (Optional[str]): Cache directory.
            lru_size (int): LRU cache size.
            dataset_name (Optional[str]): Dataset name.
            localname (Optional[str]): Local name.
            transformations (Union[str, List[Callable[[Dict[str, Any]], Dict[str, Any]]]]): Transformations to apply.
            keep (bool): Whether to keep the dataset.
            base (Optional[str]): Base directory.
            options (Optional[Dict[str, Any]]): Additional options.
        """
        super().__init__(
            shards,
            cache_size=cache_size,
            cache_dir=cache_dir,
            lru_size=lru_size,
            dataset_name=dataset_name,
            localname=localname,
            transformations=transformations,
            keep=keep,
            base=base,
            options=options,
        )
        self._info = info
        self._split = split

    @property
    def info(self) -> AtriaDatasetInfo:
        """Returns the dataset information."""
        return self._info

    @property
    def split(self) -> datasets.Split:
        """Returns the dataset split."""
        return self._split

    def __repr__(self) -> str:
        """
        Get the string representation of the dataset.

        Returns:
            str: String representation of the dataset.
        """
        return f"Dataset({{\n    features: {list(self.info.features.keys())},\n    num_rows: {self.total_length}\n}})"
