import shutil
from pathlib import Path
from typing import List, Optional, Tuple, Union

import webdataset as wds

from atria.core.data.data_modules.dataset_cacher.dataset_cacher import FileUrlProvider
from atria.core.data.data_modules.dataset_cacher.postprocessor import (
    DatasetPostprocessor,
)
from atria.core.data.data_modules.dataset_cacher.shard_list_datasets import (
    MsgpackShardListDataset,
    TarShardListDataset,
)
from atria.core.data.data_modules.dataset_info import AtriaDatasetInfo
from atria.core.utilities.logging import get_logger

logger = get_logger(__name__)

SUFFIX = "-%06d"
DEFAULT_ENCODE_DECODE_FORMAT = "mp"


class DatasetReader:
    """
    A class to read dataset splits from cached files.
    """

    def __init__(
        self,
        cache_dir: str,
        file_url_provider: FileUrlProvider,
        file_format: str = "msgpack",
        dataset_key_filter: Optional[List[str]] = None,
        only_load_features: bool = False,
    ) -> None:
        """
        Initialize the DatasetReader.

        Args:
            cache_dir (str): Directory where cached dataset files are stored.
            file_format (str): Format of the cached dataset files (default is "msgpack").
        """
        self._cache_dir: Path = Path(cache_dir)
        self._file_url_provider = file_url_provider
        self._file_format: str = file_format
        self._dataset_info: Optional[AtriaDatasetInfo] = None
        self._dataset_key_filter: Optional[List[str]] = dataset_key_filter
        self._only_load_features: bool = only_load_features

    @property
    def dataset_info(self) -> Optional[AtriaDatasetInfo]:
        """
        Returns the dataset information.

        Returns:
            Optional[AtriaDatasetInfo]: The dataset information if loaded, otherwise None.
        """
        return self._dataset_info

    def _sample_post_processor(self) -> DatasetPostprocessor:
        """
        Create a postprocessor for the dataset samples.

        Returns:
            DatasetPostprocessor: The postprocessor for dataset samples.

        Raises:
            RuntimeError: If dataset info is not loaded.
        """
        if self._dataset_info is None:
            raise RuntimeError("Dataset info is not loaded.")
        return DatasetPostprocessor(self._dataset_info.features)

    def is_available(self, split: str) -> bool:
        """
        Check if the dataset split is available.

        Args:
            split (str): The dataset split (e.g., "train", "test").

        Returns:
            bool: True if the split is available, False otherwise.
        """
        try:
            self._dataset_info = AtriaDatasetInfo.from_file(
                self._file_url_provider.get_dataset_info_path()
            )
        except FileNotFoundError:
            return False

        shard_file_urls: List[str] = self._file_url_provider.get_shard_file_urls(split)
        return len(shard_file_urls) > 0

    def read_dataset_split(
        self, split: str, streaming_mode: bool = False
    ) -> Tuple[Union[MsgpackShardListDataset, TarShardListDataset], AtriaDatasetInfo]:
        """
        Read a dataset split.

        Args:
            split (str): The dataset split (e.g., "train", "test").

        Returns:
            wds.WebDataset: The dataset split as a WebDataset.

        Raises:
            RuntimeError: If no files are found for the specified split.
            ValueError: If the file format is unsupported.
        """
        shard_file_urls: List[str] = self._file_url_provider.get_shard_file_urls(split)
        if not shard_file_urls:
            raise RuntimeError(f"No files found for split [{split}]")

        logger.debug(f"Reading dataset split [{split}] from [{self._cache_dir}]")

        # Load dataset info
        self._dataset_info = AtriaDatasetInfo.from_file(
            self._file_url_provider.get_dataset_info_path()
        )
        if self._file_format == "tar":
            if streaming_mode:
                dataset = wds.WebDataset(
                    shard_file_urls,
                    resampled=True,
                    shardshuffle=True,
                    cache_dir=str(self._cache_dir),
                    nodesplitter=wds.split_by_node,
                )
                dataset = dataset.decode().map(self._sample_post_processor())
                dataset.split = split
                dataset.info = self._dataset_info
                return dataset, self._dataset_info
            else:
                import wids

                shards = [
                    {"url": file, "nsamples": wids.wids.compute_num_samples(file)}
                    for file in shard_file_urls
                ]
                shards = [shard for shard in shards if shard["nsamples"] > 0]
                dataset = TarShardListDataset(
                    shards,
                    info=self._dataset_info,
                    split=split,
                )

                # always clean the cache dir on startup this is /tmp/wids
                if Path(dataset.cache_dir).exists():
                    shutil.rmtree(Path(dataset.cache_dir))
                Path(dataset.cache_dir).mkdir(parents=True, exist_ok=True)

                dataset.add_transform(self._sample_post_processor())
                return dataset, self._dataset_info
        elif self._file_format == "msgpack":
            if streaming_mode:
                raise NotImplementedError(
                    "Streaming mode is not supported for msgpack based datasets. "
                    "If you need streaming, use tar based webdatasets."
                )
            else:
                dataset = MsgpackShardListDataset(
                    shard_file_urls,
                    info=self._dataset_info,
                    split=split,
                    transformations=[self._sample_post_processor()],
                    dataset_key_filter=self._dataset_key_filter,
                    only_load_features=self._only_load_features,
                )
                return dataset, self._dataset_info
        else:
            raise ValueError(f"Unsupported file format: {self._file_format}")
