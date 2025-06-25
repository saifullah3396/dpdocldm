import json
from dataclasses import dataclass
from pathlib import Path

from datasets import DatasetInfo


@dataclass
class AtriaDatasetInfo(DatasetInfo):
    @classmethod
    def from_file(cls, file_path: str):
        if Path(file_path).exists():
            with open(file_path, "r", encoding="utf-8") as f:
                dataset_info_dict = json.load(f)
            return cls.from_dict(dataset_info_dict)
        else:
            raise FileNotFoundError(f"Dataset info file not found at {file_path}")

    def to_file(self, file_path: str):
        with open(file_path, "wb") as f:
            self._dump_info(f, pretty_print=True)
