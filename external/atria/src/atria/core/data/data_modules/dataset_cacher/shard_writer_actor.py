from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import ray

from atria.core.data.data_modules.dataset_cacher.preprocessor import (
    DatasetPreprocessor,
)
from atria.core.utilities.logging import get_logger

logger = get_logger(__name__)
SUFFIX = "-%06d"
DEFAULT_ENCODE_DECODE_FORMAT = "mp"


@dataclass
class WriteInfo:
    shard: int = 1
    total: int = 0
    count: int = 0
    size: int = 0


@ray.remote
class ShardWriterActor:
    """
    Actor class for writing dataset shards with preprocessing.
    """

    def __init__(
        self, file_path: str, maxcount: int, preprocessor: DatasetPreprocessor
    ) -> None:
        """
        Initialize the ShardWriterActor.

        Args:
            file_path (str): The path where the shards will be written.
            maxcount (int): The maximum number of samples per shard.
            preprocessor (DatasetPreprocessor): The preprocessor to apply to each sample.
        """
        import webdataset as wds

        from atria.core.data.data_modules.dataset_cacher.msgpack_shard_writer import (
            MsgpackShardWriter,
        )
        from atria.core.data.data_modules.dataset_cacher.preprocessor import (
            DatasetPreprocessor,
        )

        self._file_path: str = file_path
        self._maxcount: int = maxcount
        self._preprocessor: DatasetPreprocessor = preprocessor
        self._writer: Optional[Union[wds.ShardWriter, MsgpackShardWriter]] = None
        self._write_info = []

    def load(self) -> None:
        """
        Load the shard writer and preprocessor.

        Raises:
            ValueError: If the file format is unsupported.
        """
        import webdataset as wds

        from atria.core.data.data_modules.dataset_cacher.msgpack_shard_writer import (
            MsgpackShardWriter,
        )

        if Path(self._file_path).suffix == ".msgpack":
            self._writer = MsgpackShardWriter(self._file_path, maxcount=self._maxcount)
        elif Path(self._file_path).suffix == ".tar":
            self._writer = wds.ShardWriter(self._file_path, maxcount=self._maxcount)
        else:
            raise ValueError(
                f"Unsupported file format: {Path(self._file_path).suffix}. Supported formats are '.msgpack' and '.tar'."
            )
        self._preprocessor.load()
        self._write_info.append(WriteInfo())

    def write(self, sample: Dict[str, Any]) -> None:
        """
        Write a sample to the shard after preprocessing.

        Args:
            sample (Dict[str, Any]): The sample to be written.

        Raises:
            RuntimeError: If the shard writer is not loaded.
        """
        if self._writer is None:
            raise RuntimeError(
                "ShardWriterActor is not loaded. Call 'load' before writing samples."
            )
        for s in self._preprocessor(sample):
            if s is None or not isinstance(s, dict):
                logger.warning(
                    "Sample is not a dictionary. Skipping sample in dataset."
                )
                continue

            self._writer.write(s)

            # Update write info
            if self._writer.shard != self._write_info[-1].shard:
                self._write_info.append(
                    WriteInfo(
                        shard=self._writer.shard,
                        total=self._writer.total,
                        count=self._writer.count,
                        size=self._writer.size,
                    )
                )
            self._write_info[-1].shard = self._writer.shard
            self._write_info[-1].total = self._writer.total
            self._write_info[-1].count = self._writer.count
            self._write_info[-1].size = self._writer.size

    def close(self) -> None:
        """
        Close the shard writer.
        """
        if self._writer is not None:
            self._writer.close()
            self._writer = None

        return self._write_info
