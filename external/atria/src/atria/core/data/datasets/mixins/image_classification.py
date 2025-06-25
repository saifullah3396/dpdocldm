import dataclasses
from typing import List

import datasets
from datasets.features import Image

from atria.core.constants import DataKeys

logger = datasets.logging.get_logger(__name__)


@dataclasses.dataclass
class ImageClassificationConfigMixin:
    """BuilderConfig for AtriaImageDataset"""

    labels: List[str] = None


class ImageClassificationMixin:
    def _dataset_features(self):
        return datasets.Features(
            {
                DataKeys.IMAGE: Image(decode=True),
                DataKeys.LABEL: datasets.features.ClassLabel(names=self.config.labels),
                DataKeys.IMAGE_WIDTH: datasets.features.Value(dtype="int32"),
                DataKeys.IMAGE_HEIGHT: datasets.features.Value(dtype="int32"),
            }
        )
