import glob
import itertools
import os
import shutil
import sys
from pathlib import Path
from typing import Callable, List, Optional, OrderedDict, Union

import datasets
import more_itertools
import numpy as np
import ray
import tqdm
from datasets import IterableDataset
from ray.data.block import Block
from ray.data.datasource import FilenameProvider
from torch.utils.data import Dataset

from atria.core.data.data_modules.dataset_cacher.dataset_cacher import FileUrlProvider
from atria.core.data.data_modules.dataset_cacher.preprocessor import (
    DatasetPreprocessor,
)
from atria.core.data.data_modules.dataset_cacher.shard_writer_actor import (
    ShardWriterActor,
)
from atria.core.data.data_modules.dataset_info import AtriaDatasetInfo
from atria.core.utilities.logging import get_logger

logger = get_logger(__name__)

SUFFIX = "-%06d"
DEFAULT_ENCODE_DECODE_FORMAT = "mp"


def convert_to_python_type(sample: dict) -> dict:
    for k, v in sample.items():
        if isinstance(v, (np.ndarray, np.generic)):
            sample[k] = v.item()
        elif isinstance(v, dict):
            sample[k] = convert_to_python_type(v)
    return sample


class ShuffleDatasetFilenameProvider(FilenameProvider):
    """Provides file names for shuffled dataset blocks."""

    def __init__(
        self, split: str, cache_file_name: str = None, file_format: str = "tar"
    ):
        """
        Args:
            split (str): The dataset split (e.g., 'train', 'test').
            file_format (str): The file format (default is 'tar').
        """
        self.split = split
        self.file_format = file_format
        self.cache_file_name = cache_file_name

    def get_filename_for_block(
        self, block: Block, task_index: int, block_index: int
    ) -> str:
        """
        Args:
            block (Block): The data block.
            task_index (int): The task index.
            block_index (int): The block index.

        Returns:
            str: The file name for the block.
        """

        file_name = f"{self.cache_file_name}-" if self.cache_file_name else ""
        file_name += (
            f"{self.split}-{task_index:06d}-{block_index:06}.{self.file_format}"
        )
        return file_name


class DatasetWriter:
    """Handles writing and caching of datasets."""

    def __init__(
        self,
        cache_dir: str,
        file_url_provider: FileUrlProvider,
        dataset_info: AtriaDatasetInfo,
        cache_file_name: Optional[str] = None,
        preprocess_data_transforms: Optional[
            Union[Callable, OrderedDict[str, Callable]]
        ] = None,
        file_format: str = "msgpack",
        maxcount: int = 10000,
        num_processes: int = 4,
        is_preprocessing_batched: bool = False,
        preprocessing_batch_size: int = 32,
        overwrite_cache: bool = False,
        max_memory_per_actor=500 * 1024 * 1024,
    ):
        """
        Args:
            cache_dir (str): Directory to cache the dataset.
            dataset_info (AtriaDatasetInfo): Information about the dataset.
            cache_file_name (Optional[str]): Cache file name.
            preprocess_data_transforms (Optional[Callable]): Data transformation function.
            file_format (str): File format for caching (default is 'msgpack').
            maxcount (int): Maximum count of samples per shard.
            num_processes (int): Number of processes for parallel processing.
            is_preprocessing_batched (bool): Whether preprocessing is batched.
            preprocessing_batch_size (int): Batch size for preprocessing.
            overwrite_cache (bool): Whether to overwrite existing cache.
            max_memory_per_actor (int): Maximum memory per actor.
        """
        self._cache_dir = Path(cache_dir)
        self._file_url_provider = file_url_provider
        self._dataset_info = dataset_info
        self._cache_file_name = cache_file_name
        self._preprocess_data_transforms = preprocess_data_transforms
        self._file_format = file_format
        self._maxcount = maxcount
        self._num_processes = num_processes
        self._is_preprocessing_batched = is_preprocessing_batched
        self._preprocessing_batch_size = preprocessing_batch_size
        self._overwrite_cache = overwrite_cache
        self._max_memory_per_actor = max_memory_per_actor

    def _prepare_cache_dir(self, split: str) -> None:
        """Prepares the cache directory by creating it and cleaning previous runs."""
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Clean the output directory from previous runs
        self._clean_cache_dir(split)

    def _clean_cache_dir(self, split: str) -> None:
        for f in self._file_url_provider.get_shard_file_urls(split):
            Path(f).unlink()

    def _prepare_output_file_pattern(self, split: str, proc: int = 0) -> str:
        """
        Prepares the output file pattern.

        Args:
            split (str): The dataset split.
            proc (int): The process index.

        Returns:
            str: The output file pattern.
        """
        file_name = f"{self._cache_file_name}-" if self._cache_file_name else ""
        file_name += f"{split}-"
        file_name += f"{proc:06d}-"
        file_name += "%06d"
        file_name += f".{self._file_format}"
        return str(self._cache_dir / file_name)

    def _write_split(self, split: str, ds: IterableDataset) -> None:
        """
        Writes a dataset split to the cache.

        Args:
            split (str): The dataset split.
            ds (IterableDataset): The dataset to write.
        """
        ray.init(num_cpus=self._num_processes)

        try:
            preprocessor = DatasetPreprocessor(
                self._dataset_info.features, self._preprocess_data_transforms
            )

            shard_writer_actors = [
                ShardWriterActor.options(memory=self._max_memory_per_actor).remote(
                    self._prepare_output_file_pattern(split, proc),
                    maxcount=self._maxcount,
                    preprocessor=preprocessor,
                )
                for proc in range(self._num_processes)
            ]

            for actor in shard_writer_actors:
                ray.get(actor.load.remote())

            shard_writer_actors_cycled = itertools.cycle(shard_writer_actors)
            data_iterator = (
                more_itertools.chunked(
                    tqdm.tqdm(
                        ds,
                        f"Writing dataset {self._dataset_info.dataset_name} to {self._file_format}",
                    ),
                    self._preprocessing_batch_size,
                )
                if self._is_preprocessing_batched
                else tqdm.tqdm(
                    ds,
                    f"Writing dataset {self._dataset_info.dataset_name} to {self._file_format}",
                )
            )

            remaining_processed = []
            for sample in data_iterator:
                if len(remaining_processed) >= self._num_processes:
                    ready_processed, remaining_processed = ray.wait(
                        remaining_processed, num_returns=1
                    )

                    try:
                        [ray.get(result) for result in ready_processed]
                    except Exception as e:
                        raise e

                actor = next(iter(shard_writer_actors_cycled))
                remaining_processed.append(actor.write.remote(sample))

            write_info = []
            for actor in shard_writer_actors:
                write_info += ray.get(actor.close.remote())

            ray.shutdown()
        except Exception as e:
            ray.shutdown()
            logger.exception(f"Error writing dataset split {split}: {e}")
            self._clean_cache_dir(split)
            sys.exit(0)
        except KeyboardInterrupt:
            ray.shutdown()
            logger.info("Interrupted. Cleaning up cache directory.")
            self._clean_cache_dir(split)
            sys.exit(0)

        return write_info

    def _shuffle_and_rewrite_webdataset(self, split: str) -> None:
        """
        Shuffles and rewrites the webdataset.

        Args:
            split (str): The dataset split.
        """
        shard_file_urls = sorted(
            self._file_url_provider.get_shard_file_urls(split),
        )

        webdataset = ray.data.read_webdataset(
            shard_file_urls, parallelism=self._num_processes
        )

        webdataset = webdataset.map(convert_to_python_type)
        shuffled_webdataset = webdataset.random_shuffle()

        shuffled_dir_path = self._cache_dir / "shuffled"
        if shuffled_dir_path.exists():
            shutil.rmtree(shuffled_dir_path)

        shuffled_webdataset.write_webdataset(
            shuffled_dir_path,
            filename_provider=ShuffleDatasetFilenameProvider(
                split, self._cache_file_name, self._file_format
            ),
        )

        for shard in shard_file_urls:
            os.remove(shard)

        shuffled_shards = sorted(
            glob.glob(str(shuffled_dir_path / f"*.{self._file_format}"))
        )
        for shard in shuffled_shards:
            shutil.move(shard, str(self._cache_dir / Path(shard).name))
        shuffled_dir_path.rmdir()

        ray.shutdown()

    def _write_dataset_info(self, split: str, write_info: List[dict]) -> None:
        from datasets import SplitDict, SplitInfo

        write_info = [x for x in write_info if x.count > 0]

        # load the dataset info from file if already available to update
        try:
            self._dataset_info = AtriaDatasetInfo.from_file(
                self._file_url_provider.get_dataset_info_path()
            )
        except FileNotFoundError:
            pass

        # generate split info dict
        split_info = SplitInfo(
            name=str(split),
            num_bytes=sum([info.size for info in write_info]),
            num_examples=sum([info.count for info in write_info]),
            shard_lengths=[info.count for info in write_info],
        )
        if self._dataset_info.splits is None:
            self._dataset_info.splits = SplitDict(
                dataset_name=self._dataset_info.dataset_name
            )
        self._dataset_info.splits.add(split_info)
        self._dataset_info.to_file(self._file_url_provider.get_dataset_info_path())

    def write_dataset_split(
        self, split: datasets.Split, dataset: Union[IterableDataset, Dataset]
    ) -> None:
        """
        Writes a dataset split to the cache.

        Args:
            split (datasets.Split): The dataset split.
            dataset (Union[IterableDataset, Dataset]): The dataset to write.
        """
        try:
            logger.info(
                f"Preparing dataset split {split} to cache dir {self._cache_dir}."
            )
            self._prepare_cache_dir(split=split)
            write_info = self._write_split(split, dataset)
            self._write_dataset_info(split, write_info)

            if split == datasets.Split.TRAIN and self._file_format == "tar":
                self._shuffle_and_rewrite_webdataset(split)
        except Exception as e:
            logger.exception(f"Error writing dataset split {split}: {e}")
            self._clean_cache_dir(split)
            sys.exit(0)
