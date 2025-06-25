from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional, Union

import datasets
from atria.core.constants import DataKeys
from atria.core.data.data_modules.dataset_cacher.dataset_cacher import ATRIA_CACHE_DIR
from atria.core.utilities.logging import get_logger

if TYPE_CHECKING:
    import matplotlib as mpl
    from datasets import DatasetBuilder
    from datasets.features import Features
    from torch.utils.data import Dataset


logger = get_logger(__name__)


def _get_labels_color_map(n: int, name: str = "tab20") -> mpl.colors.ListedColormap:
    import matplotlib.pyplot as plt

    """Get a color map with `n` colors from the specified colormap name."""
    return plt.get_cmap(name, n)


def _create_random_subset(dataset: Dataset, max_samples: int):
    import torch
    from torch.utils.data import Subset

    max_samples = min(max_samples, len(dataset))
    dataset = Subset(
        dataset,
        torch.randperm(len(dataset))[:max_samples],
    )
    return dataset


def _get_available_splits(dataset: Union[Dataset, DatasetBuilder]):
    from atria.core.data.data_modules.dataset_info import AtriaDatasetInfo
    from atria.core.data.datasets.dataset_metadata import DatasetMetadata

    assert hasattr(dataset, "info"), (
        "Dataset must provide an info attribute that returns an "
        "object of the following types: (dict, AtriaDatasetInfo, DatasetMetadata)"
    )
    if isinstance(dataset.info, dict):
        if "splits" not in dataset.info:
            raise ValueError(
                "Dataset info should have a 'splits' key with available split names provided as a list."
            )
        return dataset.info["splits"]
    elif isinstance(dataset.info, (AtriaDatasetInfo, DatasetMetadata)):
        return dataset.info.splits.keys()
    else:
        raise ValueError("Dataset info should be of type dict or AtriaDatasetInfo")


def _unwrap_dataset(dataset):
    from atria.core.data.datasets.dataset_wrappers import TransformedDataset
    from torch.utils.data import Subset

    if isinstance(dataset, (Subset, TransformedDataset)):
        return _unwrap_dataset(dataset.dataset)
    return dataset


def _get_labels_from_features(features) -> Optional[Union[List[str], Features]]:
    """
    Get the labels from the dataset features.

    Returns:
        Optional[Union[List[str], Features]]: The labels or features of the dataset.
    """

    from datasets.features import ClassLabel, Sequence

    labels = None
    for key, value in features.items():
        if isinstance(value, ClassLabel):
            labels = value.names
        elif isinstance(value, Sequence) and isinstance(value.feature, ClassLabel):
            labels = value.feature.names
        elif isinstance(value, list):
            if key in [DataKeys.LABEL, DataKeys.WORD_LABELS, DataKeys.NER_TAGS]:
                labels = value
            elif key == DataKeys.OBJECTS:
                category_id = value[0]["category_id"]
                if isinstance(category_id, ClassLabel):
                    labels = category_id.names
                else:
                    raise ValueError(
                        f"Unsupported category_id type: {type(category_id)}"
                    )
    if labels is None:
        logger.warning("No labels found in the dataset features.")
    return labels


def _get_default_download_config() -> datasets.DownloadConfig:
    """
    Creates a DownloadConfig object with the necessary configurations.

    Returns:
        DownloadConfig: The download configuration object.
    """

    from datasets.download import DownloadConfig

    return DownloadConfig(
        cache_dir=os.path.join(ATRIA_CACHE_DIR, "datasets/downloads"),
        force_download=False,
        force_extract=False,
        use_etag=False,
        delete_extracted=True,
    )
