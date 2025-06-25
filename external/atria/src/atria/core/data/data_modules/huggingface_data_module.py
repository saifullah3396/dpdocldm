from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import datasets
import webdataset
from atria.core.data.batch_samplers import BatchSamplersDict
from atria.core.data.data_modules.atria_data_module import AtriaDataModule
from atria.core.data.data_modules.dataset_cacher.dataset_cacher import DatasetCacher
from atria.core.data.data_modules.dataset_info import AtriaDatasetInfo
from atria.core.data.data_modules.utilities import auto_dataloader
from atria.core.data.data_transforms import DataTransformsDict
from atria.core.data.train_validation_splitter import DefaultTrainValidationSplitter
from atria.core.data.utilities import _get_default_download_config
from atria.core.utilities.common import _pretty_print_dict
from atria.core.utilities.logging import get_logger
from datasets import DownloadManager, IterableDataset, IterableDatasetDict
from datasets.download import DownloadConfig
from datasets.utils.py_utils import map_nested
from torch.utils.data import Dataset

logger = get_logger(__name__)


def hf_as_iterable(
    hf_dataset_builder: datasets.GeneratorBasedBuilder,
    split: Optional[str] = None,
    base_path: Optional[str] = None,
    download_config: DownloadConfig = None,
) -> Union[Dict[str, IterableDataset], IterableDataset]:
    logger.debug(
        f"Downloading dataset {hf_dataset_builder.dataset_name} with download config: \n{_pretty_print_dict(download_config)}."
    )
    dl_manager = DownloadManager(
        base_path=base_path or hf_dataset_builder.base_path,
        download_config=download_config,
        dataset_name=hf_dataset_builder.dataset_name,
        data_dir=hf_dataset_builder.config.data_dir,
    )
    hf_dataset_builder._check_manual_download(dl_manager)
    splits_generators = {
        sg.name: sg for sg in hf_dataset_builder._split_generators(dl_manager)
    }
    # By default, return all splits
    if split is None:
        splits_generator = splits_generators
    elif split in splits_generators:
        splits_generator = splits_generators[split]
    else:
        raise ValueError(
            f"Bad split: {split}. Available splits: {list(splits_generators)}"
        )

    # Create a dataset for each of the given splits
    datasets = map_nested(
        hf_dataset_builder._as_streaming_dataset_single,
        splits_generator,
        map_tuple=True,
    )
    if isinstance(datasets, dict):
        datasets = IterableDatasetDict(datasets)
    return datasets


class HuggingfaceDataModule(AtriaDataModule):
    def __init__(
        self,
        dataset_name: str,
        dataset_config_name: Optional[str] = "default",
        dataset_kwargs: Optional[Dict] = None,
        dataset_dir: Optional[str] = None,
        dataset_key_filter: Optional[List[str]] = None,
        only_load_features: bool = False,
        dataset_output_key_map: Optional[Dict] = None,
        tar_chunk_size: Optional[int] = 1000,
        dataset_cacher: Optional[DatasetCacher] = None,
        caching_enabled: bool = True,
        runtime_data_transforms: Optional[DataTransformsDict] = None,
        batch_samplers: Optional[BatchSamplersDict] = None,
        train_validation_splitter: Optional[DefaultTrainValidationSplitter] = None,
        max_train_samples: Optional[int] = None,
        max_val_samples: Optional[int] = None,
        max_test_samples: Optional[int] = None,
        train_dataloader_builder: partial[
            Union[auto_dataloader, webdataset.WebLoader]
        ] = None,
        evaluation_dataloader_builder: partial[
            Union[auto_dataloader, webdataset.WebLoader]
        ] = None,
        streaming_mode: bool = False,
        use_validation_set_for_test: bool = False,
        use_train_set_for_test: bool = False,
        token: bool = False,
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
        self._token = token
        self._builder = None

    def _get_download_config(self) -> DownloadConfig:
        """
        Creates a DownloadConfig object with the necessary configurations.

        Returns:
            DownloadConfig: The download configuration object.
        """

        download_config = _get_default_download_config()
        download_config.token = self._token
        return download_config

    def _get_builder_kwargs(self) -> dict:
        """
        Prepares the keyword arguments required for the dataset builder.

        Returns:
            dict: The keyword arguments for the dataset builder.
        """

        # get dataset info from builder class of original dataset
        builder_kwargs = dict(
            name=self._dataset_config_name,
            data_dir=self._dataset_dir,
            cache_dir=self._dataset_cacher._cache_dir,
            **(self._dataset_kwargs if self._dataset_kwargs is not None else {}),
        )

        return builder_kwargs

    def _get_builder(self) -> datasets.GeneratorBasedBuilder:
        """
        Retrieves the dataset builder for the specified dataset.

        Returns:
            datasets.GeneratorBasedBuilder: The dataset builder instance.
        """

        if self._builder is None:
            import inspect

            from atria.core.utilities.common import _resolve_module_from_path
            from datasets import load_dataset_builder

            builder_kwargs = self._get_builder_kwargs()
            logger.info(
                f"Preparing dataset builder for dataset {self._dataset_name} with config: \n{_pretty_print_dict(builder_kwargs)}."
            )
            # check if the dataset name is a module path
            dataset_class = None
            try:
                dataset_class = _resolve_module_from_path(self._dataset_name)
            except ValueError as e:
                pass

            if dataset_class is not None:
                dataset_module = inspect.getfile(dataset_class)
                assert issubclass(
                    dataset_class, datasets.GeneratorBasedBuilder
                ), f"Dataset [{dataset_class}] must be a subclaas of GeneratorBasedBuilder to load it with {self.__class__.__name__}."
                self._builder = load_dataset_builder(
                    dataset_module,
                    **builder_kwargs,
                )
            else:
                self._builder = load_dataset_builder(
                    self._dataset_name,
                    **builder_kwargs,
                )
            self._builder.info.splits = datasets.SplitDict(
                dataset_name=self._builder.dataset_name
            )
        return self._builder

    def _build_dataset(
        self, split: datasets.Split, return_iterator: bool = False
    ) -> Tuple[
        Union[datasets.Dataset, datasets.IterableDataset],
        AtriaDatasetInfo,
    ]:
        """
        Builds the dataset for the specified split.

        Args:
            split (datasets.Split): The dataset split (e.g., train, test, validation).
            return_iterator (bool, optional): Whether to return the dataset as an iterator. Defaults to False.

        Returns:
            Tuple[Union[Dataset, Iterator[Dataset]], AtriaDatasetInfo]: The built dataset and its info.
        """
        from datasets import disable_caching

        disable_caching()

        builder = self._get_builder()

        # create dataset_info
        dataset_info = AtriaDatasetInfo()
        dataset_info.update(builder.info)

        if return_iterator:
            return (
                hf_as_iterable(
                    builder, split=split, download_config=self._get_download_config()
                ),
                dataset_info,
            )
        else:
            builder.download_and_prepare(download_config=self._get_download_config())
            return builder.as_dataset(split=split), dataset_info

    def _read_dataset_from_cache(
        self, split: datasets.Split, disable_subsampling: bool = False
    ) -> Tuple[Dataset, AtriaDatasetInfo]:
        builder = self._get_builder()
        assert (
            self._dataset_cacher is not None
        ), "Dataset cacher must be provided to cache the dataset if caching is enabled."

        # huggingface makes its own cache directory within the self.dataset_cache_dir
        self._dataset_cacher.load_cache_dir_from_builder(builder)

        output = self._dataset_cacher.read_dataset_from_cache(
            split,
            streaming_mode=self._streaming_mode,
            dataset_key_filter=(
                self._dataset_key_filter
                if split == datasets.Split.TRAIN and not disable_subsampling
                else None
            ),
            only_load_features=self._only_load_features,
        )
        if output is not None:
            return output
