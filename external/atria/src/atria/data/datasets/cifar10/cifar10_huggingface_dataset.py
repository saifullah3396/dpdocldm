import dataclasses
import os
import pickle
import uuid
from pathlib import Path

import datasets
import numpy as np

from atria.core.constants import DataKeys
from atria.core.data.datasets.huggingface_dataset import (
    AtriaHuggingfaceDataset,
    AtriaHuggingfaceDatasetConfig,
)
from atria.core.data.datasets.mixins.image_classification import (
    ImageClassificationConfigMixin,
    ImageClassificationMixin,
)

logger = datasets.logging.get_logger(__name__)


_CITATION = """\
@TECHREPORT{Krizhevsky09learningmultiple,
    author = {Alex Krizhevsky},
    title = {Learning multiple layers of features from tiny images},
    institution = {},
    year = {2009}
}
"""

_DESCRIPTION = """\
The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images
per class. There are 50000 training images and 10000 test images.
"""

_HOMEPAGE = "https://www.cs.toronto.edu/~kriz/cifar.html"

_DATA_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"

_NAMES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


@dataclasses.dataclass
class Cifar10HFDatasetConfig(
    ImageClassificationConfigMixin, AtriaHuggingfaceDatasetConfig
):
    pass


class Cifar10HFDataset(ImageClassificationMixin, AtriaHuggingfaceDataset):
    BUILDER_CONFIGS = [
        Cifar10HFDatasetConfig(
            name="default",
            version=datasets.Version("1.0.0"),
            description="Default dataset config",
            citation=_CITATION,
            homepage=_HOMEPAGE,
            data_url=_DATA_URL,
            labels=_NAMES,
        ),
    ]

    def _generate_examples(self, data_dir, split):
        """This function returns the examples in the raw (text) form."""

        base_path = "cifar-10-batches-py"
        if split == "train":
            batches = [
                "data_batch_1",
                "data_batch_2",
                "data_batch_3",
                "data_batch_4",
                "data_batch_5",
            ]

        if split == "test":
            batches = ["test_batch"]

        def iterate_over_one_batch(f):
            dict = pickle.load(f, encoding="bytes")

            labels = dict[b"labels"]
            images = dict[b"data"]

            for idx, _ in enumerate(images):
                img_reshaped = np.transpose(
                    np.reshape(images[idx], (3, 32, 32)), (1, 2, 0)
                )
                yield f"{uuid.uuid4()}", {
                    DataKeys.IMAGE: img_reshaped,
                    DataKeys.LABEL: labels[idx],
                }

        for path in os.listdir(Path(data_dir) / base_path):
            if path in batches:
                with open(Path(data_dir) / base_path / path, "rb") as f:
                    yield from iterate_over_one_batch(f)
