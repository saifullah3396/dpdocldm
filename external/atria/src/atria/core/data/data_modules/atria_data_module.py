import glob
import pickle
from abc import ABC, abstractmethod
from functools import partial
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import datasets
import hydra_zen
import tqdm
from atria.core.constants import DataKeys
from atria.core.data.batch_samplers import BatchSamplersDict
from atria.core.data.data_modules.dataset_cacher.dataset_cacher import DatasetCacher
from atria.core.data.data_modules.dataset_cacher.shard_list_datasets import (
    MsgpackListDataset,
    TarShardListDataset,
)
from atria.core.data.data_modules.dataset_info import AtriaDatasetInfo
from atria.core.data.data_transforms import DataTransformsDict, SampleKeyRemap
from atria.core.data.datasets.dataset_metadata import DatasetMetadata
from atria.core.data.datasets.dataset_wrappers import TransformedDataset
from atria.core.data.train_validation_splitter import DefaultTrainValidationSplitter
from atria.core.training.utilities.constants import TrainingStage
from atria.core.utilities.logging import get_logger
from torch.utils.data import DataLoader, Dataset

logger = get_logger(__name__)


class AtriaDataModule(ABC):
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
        from atria.core.data.data_collators import BatchToTensorDataCollator
        from atria.core.data.data_modules.utilities import auto_dataloader

        if dataset_kwargs is None:
            dataset_kwargs = {}
        if dataset_cacher is None:
            dataset_cacher = DatasetCacher()
        if train_dataloader_builder is None:
            train_dataloader_builder = partial(
                auto_dataloader,
                batch_size=64,
                num_workers=4,
                pin_memory=True,
                drop_last=False,
                collate_fn=BatchToTensorDataCollator(),
            )
        if evaluation_dataloader_builder is None:
            evaluation_dataloader_builder = partial(
                auto_dataloader,
                batch_size=64,
                num_workers=4,
                pin_memory=True,
                drop_last=False,
                collate_fn=BatchToTensorDataCollator(),
            )
        self._dataset_name = dataset_name
        self._dataset_config_name = dataset_config_name
        self._dataset_kwargs = dataset_kwargs
        self._dataset_dir = dataset_dir
        self._dataset_key_filter = dataset_key_filter
        self._only_load_features = only_load_features
        self._dataset_output_key_map = dataset_output_key_map
        self._tar_chunk_size = tar_chunk_size
        self._dataset_cacher = dataset_cacher
        self._caching_enabled = caching_enabled
        self._runtime_data_transforms = runtime_data_transforms
        self._batch_samplers = batch_samplers
        self._train_validation_splitter = train_validation_splitter
        self._max_train_samples = max_train_samples
        self._max_val_samples = max_val_samples
        self._max_test_samples = max_test_samples
        self._train_dataloader_builder = train_dataloader_builder
        self._evaluation_dataloader_builder = evaluation_dataloader_builder
        self._streaming_mode = streaming_mode
        self._use_validation_set_for_test = use_validation_set_for_test
        self._use_train_set_for_test = use_train_set_for_test
        self._use_stratified_sampling = use_stratified_sampling
        self._subset_label = subset_label
        self._train_dataset_override_path = train_dataset_override_path

        # initialize class fields
        self._train_dataset = None
        self._validation_dataset = None
        self._test_dataset = None
        self._dataset_metadata = None

        if self._streaming_mode:
            logger.warning(
                "Streamable dataset is only supported with cache type 'webdataset'."
                f"Forcing 'webdataset' for caching instead of '{self._dataset_cacher._cache_type}'."
            )
            self._dataset_cacher._cache_type = "tar"
            assert self._train_dataloader_builder.func.__qualname__ == "WebLoader", (
                "Streaming mode is only supported for webdataset based datasets which require a webdataset.WebLoader. "
                "You can override the dataloader builder as atria/data_module/dataloader_builder@atria.data_module.train_dataloader_builder=webdataset"
            )
            assert (
                self._evaluation_dataloader_builder.func.__qualname__ == "WebLoader"
            ), (
                "Streaming mode is only supported for webdataset based datasets which require a webdataset.WebLoader. "
                "You can override the dataloader builder as atria/data_module/dataloader_builder@atria.data_module.evaluation_dataloader_builder=webdataset"
            )

        assert not (
            self._use_validation_set_for_test and self._use_train_set_for_test
        ), "Only one of use_validation_set_for_test or use_train_set_for_test can be set to True."

    @property
    def train_dataset(self):
        return self._train_dataset

    @property
    def validation_dataset(self):
        return self._validation_dataset

    @property
    def test_dataset(self):
        return self._test_dataset

    @property
    def dataset_metadata(self):
        return self._dataset_metadata

    def _get_transforms(self, split: datasets.Split) -> Optional[DataTransformsDict]:
        """
        Retrieves the data transforms for the specified split.

        Args:
            split (datasets.Split): The dataset split (e.g., train, test, validation).

        Returns:
            Optional[DataTransformsDict]: The data transforms for the specified split.
        """
        if split in [datasets.Split.TRAIN]:
            return (
                self._runtime_data_transforms.train
                if self._runtime_data_transforms is not None
                else None
            )
        elif split in [datasets.Split.TEST, datasets.Split.VALIDATION]:
            return (
                self._runtime_data_transforms.evaluation
                if self._runtime_data_transforms is not None
                else None
            )
        else:
            raise ValueError(
                f"'split' must be one of the following: "
                "'datasets.Split.TRAIN', 'datasets.Split.VALIDATION', 'datasets.Split.TEST'"
            )

    @abstractmethod
    def _build_dataset(
        self, split: datasets.Split, return_iterator: bool = False
    ) -> Tuple[
        Union[Dataset, Iterator[Dataset], datasets.Dataset, datasets.IterableDataset],
        AtriaDatasetInfo,
    ]:
        pass

    def _read_dataset_from_cache(
        self, split: datasets.Split, disable_subsampling: bool = False
    ) -> Tuple[Dataset, AtriaDatasetInfo]:
        assert (
            self._dataset_cacher is not None
        ), "Dataset cacher must be provided to cache the dataset if caching is enabled."
        return self._dataset_cacher.read_dataset_from_cache(
            split,
            streaming_mode=self._streaming_mode,
            dataset_key_filter=(
                self._dataset_key_filter
                if split == datasets.Split.TRAIN and not disable_subsampling
                else None
            ),
            only_load_features=self._only_load_features,
        )

    def _write_dataset_to_cache(
        self,
        split: datasets.Split,
        dataset: Dataset,
        dataset_info: AtriaDatasetInfo,
    ):
        assert (
            self._dataset_cacher is not None
        ), "Dataset cacher must be provided to cache the dataset if caching is enabled."
        self._dataset_cacher.write_dataset_to_cache(
            dataset=dataset, split=split, dataset_info=dataset_info
        )
        return self._dataset_cacher.read_dataset_from_cache(
            split, streaming_mode=self._streaming_mode
        )

    def _load_dataset(
        self,
        split: datasets.Split,
        disable_subsampling: bool = False,
    ) -> Tuple[Dataset, AtriaDatasetInfo]:
        logger.info(f"Loading dataset split [{split}]...")

        # now wrap the dataset with FusionWebDataset
        if self._caching_enabled:
            dataset_output = self._read_dataset_from_cache(
                split, disable_subsampling=disable_subsampling
            )
            if dataset_output is not None:
                dataset, dataset_info = dataset_output
                dataset = self._prepare_labelled_subset(
                    dataset, split=split, disable_subsampling=disable_subsampling
                )
                return dataset, dataset_info

        # build the dataset from class, if caching is enabled, we need the iterator so dataset are loaded in
        # streaming mode
        try:
            dataset, dataset_info = self._build_dataset(
                split, return_iterator=self._caching_enabled
            )
        except Exception as e:
            logger.error(
                f"Failed to load dataset for split [{split}] with error: {e}. Retrying..."
            )
            return None, None

        if self._caching_enabled:
            return self._write_dataset_to_cache(
                dataset=dataset, split=split, dataset_info=dataset_info
            )
        else:
            return dataset, dataset_info

    def _load_train_val_dataset(self, disable_subsampling: bool = False):
        from atria.core.utilities.common import _pretty_print_dict

        train_dataset, _ = self._load_dataset(
            split=datasets.Split.TRAIN, disable_subsampling=disable_subsampling
        )

        if self._train_dataset_override_path is not None:
            train_dataset_files = glob.glob(self._train_dataset_override_path)
            train_dataset = MsgpackListDataset(
                train_dataset_files,
                info=train_dataset.info,
                split=datasets.Split.TRAIN,
                transformations=train_dataset._transformations,
            )

        val_dataset = None
        try:
            val_dataset, _ = self._load_dataset(
                split=datasets.Split.VALIDATION, disable_subsampling=disable_subsampling
            )
        except ValueError as e:
            logger.warning(f"Unable to load validation split from the dataset: {e}")

        if self._train_validation_splitter is not None and not disable_subsampling:
            logger.info(
                f"Using train/validation sampler [{self._train_validation_splitter}] for splitting the "
                f"dataset with following arguments: {_pretty_print_dict(self._train_validation_splitter)}"
            )
            train_dataset, val_dataset = self._train_validation_splitter(train_dataset)

        # prepare transforms
        train_dataset = self._prepare_data_transforms(
            train_dataset,
            datasets.Split.TRAIN,
            max_samples=self._max_train_samples,
            disable_subsampling=disable_subsampling,
        )
        val_dataset = self._prepare_data_transforms(
            val_dataset,
            datasets.Split.VALIDATION,
            max_samples=self._max_val_samples,
            disable_subsampling=disable_subsampling,
        )

        return train_dataset, val_dataset

    def _load_test_dataset(self, disable_subsampling: bool = False):
        split = datasets.Split.TEST
        if self._use_validation_set_for_test and not disable_subsampling:
            logger.warning(
                "Using validation set for test set since use_validation_set_for_test=True."
            )
            split = datasets.Split.VALIDATION
        elif self._use_train_set_for_test and not disable_subsampling:
            logger.warning(
                "Using train set for test set since use_train_set_for_test=True."
            )
            split = datasets.Split.TRAIN
        test_dataset, _ = self._load_dataset(
            split=split, disable_subsampling=disable_subsampling
        )
        test_dataset = self._prepare_data_transforms(
            test_dataset,
            datasets.Split.TEST,
            max_samples=self._max_test_samples,
            disable_subsampling=disable_subsampling,
        )
        return test_dataset

    def _prepare_labelled_subset(
        self, dataset: Dataset, split: datasets.Split, disable_subsampling: bool = False
    ):
        if self._subset_label is not None and not disable_subsampling:
            import torch
            from torch.utils.data import Subset

            label_indices_cache_path = (
                Path(self._dataset_cacher.cache_dir)
                / f"{self._dataset_cacher._cache_file_name}-{split}-label-{self._subset_label}-indices.pth"
            )

            if not label_indices_cache_path.exists():
                logger.info(
                    f"Generating label indices for subset sampling for split = {split}."
                )
                dataset._apply_postprocessing = False
                label_indices = []
                for idx, sample in tqdm.tqdm(
                    enumerate(dataset), desc="Loading subset indices"
                ):
                    suffix = ".mp"
                    assert (
                        DataKeys.LABEL + suffix in sample
                    ), f"Subset sampling requires labels to be provided in the dataset. Current keys in sample = {sample.keys()} "
                    if sample[DataKeys.LABEL + suffix] == self._subset_label:
                        label_indices.append(
                            idx.item() if isinstance(idx, torch.Tensor) else idx
                        )
                torch.save(label_indices, label_indices_cache_path)
                dataset._apply_postprocessing = True
                if hasattr(dataset, "close"):
                    dataset.close()
            else:
                label_indices = torch.load(label_indices_cache_path)

            logger.info(
                f"Loaded label indices from cache file {label_indices_cache_path} for label = {self._subset_label} with size = {len(label_indices)}"
            )
            dataset = Subset(
                dataset,
                label_indices,
            )
        return dataset

    def _prepare_data_transforms(
        self,
        dataset: Dataset,
        split: datasets.Split,
        max_samples: Optional[int] = None,
        disable_subsampling: bool = False,
    ):
        import webdataset as wds
        from atria.core.data.utilities import _create_random_subset
        from torchvision.transforms import Compose

        if dataset is not None:
            runtime_data_transforms = []
            if self._dataset_output_key_map is not None:
                runtime_data_transforms.append(
                    SampleKeyRemap(self._dataset_output_key_map)
                )

            # wrap the dataset in a TransformedDataset
            cfg_runtime_data_transforms = self._get_transforms(split)
            if cfg_runtime_data_transforms is not None:
                if isinstance(cfg_runtime_data_transforms, dict):
                    cfg_runtime_data_transforms = list(
                        cfg_runtime_data_transforms.values()
                    )
                if not isinstance(cfg_runtime_data_transforms, list):
                    cfg_runtime_data_transforms = [cfg_runtime_data_transforms]
                for transform in cfg_runtime_data_transforms:
                    runtime_data_transforms.append(transform)

            if len(runtime_data_transforms) > 0:
                runtime_data_transforms = Compose(runtime_data_transforms)
            else:
                runtime_data_transforms = None

            if isinstance(dataset, wds.WebDataset):
                if runtime_data_transforms is not None:
                    dataset = dataset.map(runtime_data_transforms)
            else:
                # if max_samples is set get the given number of examples
                if not disable_subsampling:
                    dataset = (
                        _create_random_subset(dataset, max_samples)
                        if max_samples is not None
                        else dataset
                    )

                dataset = TransformedDataset(
                    dataset,
                    transforms=runtime_data_transforms,
                )

            logger.info(f"Loaded dataset split [{split}]:\n{dataset}")
            if runtime_data_transforms is not None:
                logger.info(
                    "Attached data transforms: \n"
                    f"{runtime_data_transforms.transforms}"
                )
        return dataset

    def _prepare_datasets_for_stage(
        self,
        stage: TrainingStage = TrainingStage.train,
        disable_subsampling: bool = False,
    ) -> None:
        if stage is not None:
            logger.info(f"Loading data for stage == {stage}")
        else:
            logger.info(f"Loading data for stage == train|validation|test")

        train_dataset = None
        val_dataset = None
        test_dataset = None

        # Assign train/val datasets for use in dataloaders using the train/val sampler
        # lightning calls training stage 'fit'
        if (
            stage == TrainingStage.train
            or stage is None
            or self._use_validation_set_for_test
            or self._use_train_set_for_test
        ):
            train_dataset, val_dataset = self._load_train_val_dataset(
                disable_subsampling=disable_subsampling,
            )

        # Assign test dataset for use in dataloader(s)
        if stage == TrainingStage.test or stage is None:
            test_dataset = self._load_test_dataset(
                disable_subsampling=disable_subsampling
            )

        return train_dataset, val_dataset, test_dataset

    def _setup_dataset_metadata(self) -> DatasetMetadata:
        from atria.core.data.utilities import _unwrap_dataset

        if self.train_dataset is not None:
            dataset = self.train_dataset
        elif self.test_dataset is not None:
            dataset = self.test_dataset
        else:
            raise ValueError("No dataset found in datamodule.")

        # unwrap the dataset
        dataset = _unwrap_dataset(dataset)

        if not hasattr(dataset, "info"):
            raise ValueError(
                "Dataset must provide an info attribute that returns an "
                "object of the following types: (AtriaDatasetInfo)"
            )

        if isinstance(dataset.info, AtriaDatasetInfo):
            return DatasetMetadata.from_info(dataset.info)
        else:
            raise ValueError(
                "Dataset info should be of type AtriaDatasetInfo, or DatasetMetadata"
            )

    def setup(
        self,
        stage: Optional[TrainingStage] = None,
        disable_subsampling: bool = False,
    ) -> None:
        import ignite.distributed as idist

        # only download dataset on rank 0, all other ranks wait here for rank 0 to load the datasets
        if idist.get_rank() > 0:
            idist.barrier()

        # we manually prepare data and call setup here so dataset related properties can be initalized.
        self._train_dataset, self._validation_dataset, self._test_dataset = (
            self._prepare_datasets_for_stage(
                stage=stage,
                disable_subsampling=disable_subsampling,
            )
        )
        if self._test_dataset is None:
            logger.warning(
                "No test dataset found in the datamodule. Using train dataset"
            )
            self._test_dataset = self._train_dataset
        self._dataset_metadata = self._setup_dataset_metadata()

        if idist.get_rank() == 0:
            idist.barrier()

        # print dataset lengths
        if self.train_dataset is not None:
            logger.info(f"Train dataset length = {len(self.train_dataset)}")
        if self.validation_dataset is not None:
            logger.info(f"Validation dataset length = {len(self.validation_dataset)}")
        if self.test_dataset is not None:
            logger.info(f"Test dataset length = {len(self.test_dataset)}")

        data_labels = self.dataset_metadata.labels
        if data_labels is not None:
            logger.info(f"Labels found in the dataset = {data_labels[:100]}")
            if isinstance(data_labels, list):
                logger.info(f"Number of labels = {len(data_labels)}")
        else:
            logger.warning("No labels found in the dataset.")

    def _build_train_dataloader(self, shuffle: bool = True, **kwargs) -> DataLoader:
        import ignite.distributed as idist
        from torch.utils.data import (
            RandomSampler,
            SequentialSampler,
            WeightedRandomSampler,
        )
        from wids import ChunkedSampler

        # setup sampler
        if shuffle:
            if isinstance(self.train_dataset, TarShardListDataset):
                # tar shard list dataset is a webdataset based dataset which requires chunked sampler
                sampler = ChunkedSampler(
                    self.train_dataset,
                    shuffle=True,
                    shufflefirst=True,
                    chunksize=self.tar_chunk_size,
                )
            else:
                if self._use_stratified_sampling:

                    def _load_sample_weights():
                        sample_weights_cache_path = (
                            Path(self._dataset_cacher.cache_dir)
                            / f"{self._dataset_cacher._cache_file_name}-train-sample_weights.pickle"
                        )
                        if sample_weights_cache_path.exists():
                            with open(sample_weights_cache_path, "rb") as f:
                                logger.info(
                                    f"Loading sample weights from cache: {sample_weights_cache_path}"
                                )
                                return pickle.load(f)
                        else:
                            sample_weights = []
                            self.train_dataset.disable_transforms()
                            for sample in tqdm.tqdm(
                                self.train_dataset, desc="Loading sample weights"
                            ):
                                suffix = ".mp"
                                assert DataKeys.SAMPLE_WEIGHT + suffix in sample, (
                                    "Stratified sampling requires sample weights to be provided in the dataset. You can set this "
                                    "by attaching sample_weights to info object during data loading."
                                )
                                sample_weights.append(
                                    sample[DataKeys.SAMPLE_WEIGHT + suffix]
                                )
                            self.train_dataset.enable_transforms()
                            if hasattr(self.train_dataset, "close"):
                                self.train_dataset.close()
                            logger.info(
                                f"Saving sample weights to cache: {sample_weights_cache_path}"
                            )
                            with open(sample_weights_cache_path, "wb") as f:
                                pickle.dump(sample_weights, f)
                            return sample_weights

                    sample_weights = _load_sample_weights()
                    assert len(sample_weights) == len(
                        self.train_dataset
                    ), "Sample weights length must be equal to the dataset length."
                    sampler = WeightedRandomSampler(
                        sample_weights,
                        len(sample_weights),
                        replacement=False,
                    )
                else:
                    sampler = RandomSampler(self.train_dataset)
        else:
            sampler = SequentialSampler(self.train_dataset)

        # we override kwargs here
        kwargs["sampler"] = sampler
        if "batch_size" in kwargs:
            kwargs["batch_size"] = kwargs["batch_size"] * idist.get_world_size()
        elif "batch_size" in self._train_dataloader_builder.keywords:
            kwargs["batch_size"] = (
                self._train_dataloader_builder.keywords["batch_size"]
                * idist.get_world_size()
            )
        if self._batch_samplers.train is not None:
            assert isinstance(self._batch_samplers.train, partial), (
                "batch_sampler must be a partial initializer which takes the torch.utils.data.Sampler "
                "as input and returns torch.utils.data.BatchSampler."
            )
            kwargs["batch_sampler"] = self._batch_samplers.train(sampler)
        if idist.get_world_size() > 1:
            kwargs["drop_last"] = True
            logger.info("Overriding drop last to True for distributed training.")
        return self._train_dataloader_builder(dataset=self.train_dataset, **kwargs)

    def _build_evaluation_dataloader(
        self,
        dataset: Dataset,
        **kwargs,
    ) -> DataLoader:
        import ignite.distributed as idist
        from torch.utils.data import SequentialSampler

        # setup sampler
        sampler = SequentialSampler(dataset)

        # we override kwargs here
        kwargs["sampler"] = sampler

        # get the overrides
        if "batch_size" in kwargs:
            kwargs["batch_size"] = kwargs["batch_size"] * idist.get_world_size()
        elif "batch_size" in self._evaluation_dataloader_builder.keywords:
            kwargs["batch_size"] = (
                self._evaluation_dataloader_builder.keywords["batch_size"]
                * idist.get_world_size()
            )
        if idist.get_world_size() > 1:
            if len(dataset) % idist.get_world_size() != 0:
                logger.warning(
                    "Enabling distributed evaluation with an eval dataset not divisible by process number. "
                    "This will slightly alter validation results as extra duplicate entries are added to achieve "
                    "equal num of samples per-process."
                )
        # properly setup the kwargs
        if self._batch_samplers.evaluation is not None:
            assert isinstance(self._batch_samplers.evaluation, partial), (
                "batch_sampler must be a partial initializer which takes the torch.utils.data.Sampler "
                "as input and returns torch.utils.data.BatchSampler."
            )
            kwargs["batch_sampler"] = self._batch_samplers.evaluation(sampler)
        return self._evaluation_dataloader_builder(
            dataset,
            shuffle=False,
            drop_last=False,
            **kwargs,
        )

    def _build_streaming_train_dataloader(
        self, shuffle: bool = True, **kwargs
    ) -> DataLoader:
        import webdataset as wds
        from webdataset import pipelinefilter

        assert isinstance(
            self._train_dataset, wds.WebDataset
        ), "Streaming mode is only supported for webdataset based datasets."

        # get the overrides
        if "batch_size" in kwargs:
            batch_size = kwargs["batch_size"]
        elif "batch_size" in self._train_dataloader_builder.keywords:
            batch_size = self._train_dataloader_builder.keywords["batch_size"]

        if "collate_fn" in kwargs:
            collate_fn = kwargs["collate_fn"]
        elif "collate_fn" in self._train_dataloader_builder.keywords:
            collate_fn = self._train_dataloader_builder.keywords["collate_fn"]

        if shuffle:
            self._train_dataset = self._train_dataset.shuffle(
                self._tar_chunk_size
            ).batched(batch_size, collation_fn=collate_fn)
        else:
            self._train_dataset = self._train_dataset.batched(
                batch_size, collation_fn=collate_fn
            )

        kwargs["batch_size"] = None
        train_dataloader: wds.WebLoader = self._train_dataloader_builder(
            dataset=self._train_dataset,
            **kwargs,
        )

        def _unbatched_dict(data):
            for batch in data:
                batch = [dict(zip(batch, t)) for t in zip(*batch.values())]
                assert isinstance(batch, (tuple, list)), batch
                assert len(batch) > 0
                for sample in batch:
                    yield sample

        if shuffle:
            train_dataloader = train_dataloader.compose(
                pipelinefilter(_unbatched_dict)()
            )
            train_dataloader = train_dataloader.shuffle(self._tar_chunk_size).batched(
                batch_size, collation_fn=collate_fn
            )
        return train_dataloader

    def _build_streaming_evaluation_dataloader(
        self,
        dataset: Dataset,
        **kwargs,
    ) -> DataLoader:
        import webdataset as wds

        assert isinstance(
            dataset, wds.WebDataset
        ), "Streaming mode is only supported for webdataset based datasets."

        # get the overrides
        if "batch_size" in kwargs:
            batch_size = kwargs["batch_size"]
        elif "batch_size" in self._train_dataloader_builder.keywords:
            batch_size = self._train_dataloader_builder.keywords["batch_size"]

        if "collate_fn" in kwargs:
            collate_fn = kwargs["collate_fn"]
        elif "collate_fn" in self._train_dataloader_builder.keywords:
            collate_fn = self._train_dataloader_builder.keywords["collate_fn"]

        dataset = dataset.batched(batch_size, collation_fn=collate_fn)
        kwargs["batch_size"] = None
        evaluation_dataloader: wds.WebLoader = self._evaluation_dataloader_builder(
            dataset=dataset,
            **kwargs,
        )
        return evaluation_dataloader

    def train_dataloader(self, shuffle: bool = True, **kwargs) -> DataLoader:
        if self._streaming_mode:
            return self._build_streaming_train_dataloader(shuffle=shuffle, **kwargs)
        else:
            return self._build_train_dataloader(shuffle=shuffle, **kwargs)

    def validation_dataloader(self, **kwargs) -> DataLoader:
        if self._validation_dataset is None:
            raise RuntimeError("Validation dataset is not available in the datamodule.")
        if self._streaming_mode:
            return self._build_streaming_evaluation_dataloader(
                self.validation_dataset, shuffle=False, **kwargs
            )
        else:
            return self._build_evaluation_dataloader(self.validation_dataset, **kwargs)

    def test_dataloader(self, **kwargs) -> DataLoader:
        if self._streaming_mode:
            return self._build_streaming_evaluation_dataloader(
                self.test_dataset, shuffle=False, **kwargs
            )
        else:
            return self._build_evaluation_dataloader(self.test_dataset, **kwargs)
