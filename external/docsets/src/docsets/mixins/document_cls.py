import dataclasses
from typing import List

import datasets
from atria.core.constants import DataKeys
from datasets.features import Image

logger = datasets.logging.get_logger(__name__)


@dataclasses.dataclass
class DocumentClassificationConfigMixin:
    labels: List[str] = None


class DocumentClassificationMixin:
    def _dataset_features(self, decode_image: bool = True):
        return datasets.Features(
            {
                # image
                DataKeys.INDEX: datasets.Value(dtype="int32"),
                DataKeys.IMAGE: Image(decode=False),
                DataKeys.IMAGE_FILE_PATH: datasets.Value(dtype="string"),
                DataKeys.IMAGE_WIDTH: datasets.features.Value(dtype="int32"),
                DataKeys.IMAGE_HEIGHT: datasets.features.Value(dtype="int32"),
                # text
                DataKeys.WORDS: datasets.Sequence(datasets.Value(dtype="string")),
                DataKeys.WORD_BBOXES: datasets.Sequence(
                    datasets.Sequence(datasets.Value(dtype="float"), length=4)
                ),
                # labels
                DataKeys.LABEL: datasets.features.ClassLabel(names=self.config.labels),
            }
        )
