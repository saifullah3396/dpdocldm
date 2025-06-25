import copy
import typing
from typing import TypeAlias

from datasets import Dataset as HFDataset
from torch.utils.data import Dataset as TorchDataset

from atria.core.utilities.logging import get_logger

Dataset: TypeAlias = typing.Union[HFDataset, TorchDataset]

logger = get_logger(__name__)


class DefaultTrainValidationSplitter:
    def __init__(self, seed: int = 42, split_ratio: float = 0.8, shuffle: bool = True):
        # The default seed used for random sampling
        self._seed = seed

        # The train/validation dataset split ratio
        self._split_ratio = split_ratio

        # Whether to shuffle the dataset before splitting
        self._shuffle = shuffle

    def create_hf_random_split(
        self, train_dataset: HFDataset
    ) -> typing.Tuple[HFDataset, HFDataset]:
        output = train_dataset.train_test_split(
            test_size=round(1.0 - self._split_ratio, 2),
            shuffle=self._shuffle,
            seed=self._seed,
            load_from_cache_file=False,
        )
        return output["train"], output["test"]

    def create_torch_sequential_split(
        self, train_dataset: TorchDataset
    ) -> typing.Tuple[TorchDataset, TorchDataset]:
        from torch.utils.data import Subset

        train_dataset_size = len(train_dataset)
        train_split_indices = list(
            range(0, int(train_dataset_size * round(self._shuffle, 2)))
        )
        validation_split_indices = list(
            range(len(train_split_indices), train_dataset_size)
        )

        train_split = Subset(copy.deepcopy(train_dataset), train_split_indices)
        validation_split = Subset(
            copy.deepcopy(train_dataset), validation_split_indices
        )

        # attach the dataset info to the splits
        train_split.info = train_split.dataset.info
        validation_split.info = validation_split.dataset.info

        return train_split, validation_split

    def create_torch_random_split(
        self, train_dataset: TorchDataset
    ) -> typing.Tuple[TorchDataset, TorchDataset]:
        import torch
        from torch.utils.data.dataset import random_split

        train_dataset_size = len(train_dataset)
        validation_dataset_size = int(
            train_dataset_size * round(1.0 - self._split_ratio, 2)
        )
        train_split, validation_split = random_split(
            train_dataset,
            [train_dataset_size - validation_dataset_size, validation_dataset_size],
            generator=torch.Generator().manual_seed(self._seed),
        )

        train_split.info = train_dataset.info
        validation_split.info = train_dataset.info
        return train_split, validation_split

    def __call__(self, train_dataset: Dataset) -> typing.Tuple[Dataset, Dataset]:
        from datasets import Dataset as HFDataset
        from torch.utils.data import Dataset as TorchDataset

        from atria.core.data.data_modules.dataset_cacher.shard_list_datasets import (
            TarShardListDataset,
        )

        if isinstance(train_dataset, HFDataset):
            return self.create_hf_random_split(train_dataset)
        elif isinstance(train_dataset, TorchDataset):
            if isinstance(train_dataset, TarShardListDataset):
                # for the webdataset-based dataset, we need to manually split the dataset by taking the first
                # `_split_ratio` fraction of the dataset as the training set and the rest as the validation set
                # this is because the webdataset does not support randomizing the dataset but the training set
                # is always shuffled during creation
                if self._shuffle:
                    logger.warning(
                        "Shuffling is not supported for TarShardListDataset. The dataset will be split without shuffling."
                    )
                    self._shuffle = False

            if self._shuffle:
                return self.create_torch_random_split(train_dataset)
            else:
                return self.create_torch_sequential_split(train_dataset)
        else:
            raise ValueError(
                f"Train/Validation split cannot be created for dataset of type {train_dataset} is not supported."
            )
