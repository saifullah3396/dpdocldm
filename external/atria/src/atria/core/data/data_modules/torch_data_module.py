import os
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional

import datasets
import hydra_zen
from atria.core.data.batch_samplers import BatchSamplersDict
from atria.core.data.data_modules.atria_data_module import AtriaDataModule
from atria.core.data.data_modules.dataset_cacher.dataset_cacher import (
    ATRIA_CACHE_DIR,
    DatasetCacher,
)
from atria.core.data.data_transforms import DataTransformsDict
from atria.core.data.datasets.huggingface_wrapper_dataset import (
    AtriaHuggingfaceWrapperDataset,
)
from atria.core.data.datasets.torch_dataset import AtriaTorchDataset
from atria.core.data.train_validation_splitter import DefaultTrainValidationSplitter
from atria.core.utilities.logging import get_logger

logger = get_logger(__name__)


class TorchDataModule(AtriaDataModule):
    def __init__(
        self,
        dataset_name: str = hydra_zen.MISSING,
        dataset_config_name: Optional[str] = "default",
        dataset_kwargs: Optional[Dict] = None,
        dataset_dir: Optional[str] = None,
        dataset_key_filter: Optional[List[str]] = None,
        only_load_features: bool = False,
        dataset_output_key_map: Optional[Dict] = None,
        tar_chunk_size: Optional[int] = 1000,
        dataset_cacher: Optional[DatasetCacher] = None,
        caching_enabled: bool = True,
        runtime_data_transforms: DataTransformsDict = DataTransformsDict(),
        batch_samplers: BatchSamplersDict = BatchSamplersDict(),
        train_validation_splitter: Optional[DefaultTrainValidationSplitter] = None,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
        max_test_samples: Optional[int] = None,
        train_dataloader_builder: partial = None,
        evaluation_dataloader_builder: partial = None,
        streaming_mode: bool = False,
        use_validation_set_for_test: bool = False,
        use_train_set_for_test: bool = False,
        use_stratified_sampling: bool = False,
        subset_label: Optional[int] = None,
        train_dataset_override_path: Optional[str] = None,
    ):
        super().__init__(
            dataset_name=dataset_name,
            dataset_config_name=dataset_config_name,
            dataset_kwargs=dataset_kwargs,
            dataset_dir=dataset_dir,
            dataset_key_filter=dataset_key_filter,
            only_load_features=only_load_features,
            dataset_output_key_map=dataset_output_key_map,
            tar_chunk_size=tar_chunk_size,
            dataset_cacher=dataset_cacher,
            caching_enabled=caching_enabled,
            runtime_data_transforms=runtime_data_transforms,
            batch_samplers=batch_samplers,
            train_validation_splitter=train_validation_splitter,
            max_train_samples=max_train_samples,
            max_val_samples=max_val_samples,
            max_test_samples=max_test_samples,
            train_dataloader_builder=train_dataloader_builder,
            evaluation_dataloader_builder=evaluation_dataloader_builder,
            streaming_mode=streaming_mode,
            use_validation_set_for_test=use_validation_set_for_test,
            use_train_set_for_test=use_train_set_for_test,
            use_stratified_sampling=use_stratified_sampling,
            subset_label=subset_label,
            train_dataset_override_path=train_dataset_override_path,
        )

        self._dataset_cacher.cache_dir = (
            Path(self._dataset_cacher.cache_dir)
            / self._dataset_name.split(".")[-1]
            / self._dataset_config_name
        )

    def _get_dataset_class(self) -> AtriaTorchDataset:
        from atria.core.utilities.common import _resolve_module_from_path

        dataset_class = _resolve_module_from_path(self._dataset_name)
        assert issubclass(
            dataset_class, (AtriaTorchDataset, AtriaHuggingfaceWrapperDataset)
        ), (
            "If it is a torch dataset class, you must wrap it within a AtriaTorchDataset."
            "This is necessary to provide a common interface for all torch-based datasets and to allow "
            "additional data caching and processing functionality"
        )
        assert hasattr(dataset_class, "BUILDER_CONFIGS"), (
            "Torch datasets must provide a BUILDER_CONFIGS attribute that returns a list of "
            "AtriaTorchDatasetConfig objects."
        )
        return dataset_class

    def _build_dataset(
        self, split: datasets.Split, return_iterator: bool = False
    ) -> AtriaTorchDataset:
        """
        Builds the dataset for the specified split.

        Args:
            split (datasets.Split): The dataset split (e.g., train, test, validation).
            return_iterator (bool, optional): Whether to return the dataset as an iterator. Defaults to False.

        Returns:
            Tuple[Union[Dataset, Iterator[Dataset]], AtriaDatasetInfo]: The built dataset and its info.
        """
        from atria.core.data.datasets.torch_dataset import AtriaTorchDatasetConfig

        dataset_class = self._get_dataset_class()

        self._dataset_dir = self._dataset_dir or os.path.join(
            ATRIA_CACHE_DIR, "datasets", dataset_class.__name__
        )
        logger.info(f"Setting data_dir to {self._dataset_dir}")

        dataset_config: AtriaTorchDatasetConfig = next(
            (
                config
                for config in dataset_class.BUILDER_CONFIGS
                if config.name == self._dataset_config_name
            ),
            None,
        )
        if dataset_config is None:
            raise ValueError(
                f"Dataset config {self._dataset_config_name} not found in the "
                f"BUILDER_CONFIGS of {dataset_class}. Available configs: {dataset_class.BUILDER_CONFIGS}"
            )
        dataset_config.data_dir = self._dataset_dir
        for key, value in self._dataset_kwargs.items():
            setattr(dataset_config, key, value)

        dataset: AtriaTorchDataset = dataset_class(
            config=dataset_config,
        )
        if split not in dataset.available_splits:
            raise ValueError(
                f"Split {split} not found in available_splits: {dataset.available_splits}"
            )

        dataset._load_split(split=split, cache_dir=self._dataset_cacher.cache_dir)
        assert hasattr(dataset, "info"), (
            "Dataset must provide an info attribute that returns an "
            "object of the following types: (AtriaDatasetInfo)"
        )
        return dataset, dataset.info
