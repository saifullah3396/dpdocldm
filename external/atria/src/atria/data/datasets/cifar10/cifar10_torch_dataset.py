import dataclasses

import datasets
from torchvision.datasets import CIFAR10

from atria.core.constants import DataKeys
from atria.core.data.datasets.mixins.image_classification import (
    ImageClassificationConfigMixin,
    ImageClassificationMixin,
)
from atria.core.data.datasets.torch_dataset import (
    AtriaTorchDataset,
    AtriaTorchDatasetConfig,
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
class Cifar10TorchDatasetConfig(
    ImageClassificationConfigMixin, AtriaTorchDatasetConfig
):
    pass


class Cifar10TorchDataset(ImageClassificationMixin, AtriaTorchDataset):
    BUILDER_CONFIGS = [
        Cifar10TorchDatasetConfig(
            name="default",
            version=datasets.Version("1.0.0"),
            description="Default dataset config",
            citation=_CITATION,
            homepage=_HOMEPAGE,
            data_url=_DATA_URL,
            labels=_NAMES,
        ),
    ]

    def _available_splits(self):
        return [datasets.Split.TRAIN, datasets.Split.TEST]

    def _load_split(self, split: datasets.Split, cache_dir: str):
        assert (
            self.config.data_dir is not None
        ), f"data_dir must be provided for {self.__class__.__name__} dataset."
        self._dataset = CIFAR10(
            self.config.data_dir,
            train=split == datasets.Split.TRAIN,
            download=True,
        )

    def __getitem__(self, index):
        output = self._dataset[index]
        return {DataKeys.IMAGE: output[0], DataKeys.LABEL: output[1]}

    def __len__(self):
        return len(self._dataset)
