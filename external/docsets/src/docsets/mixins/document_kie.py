import dataclasses
from typing import List

import datasets
from atria.core.constants import DataKeys
from atria.core.data.datasets.features.class_label import ClassLabel
from datasets.features import Image

logger = datasets.logging.get_logger(__name__)


@dataclasses.dataclass
class DocumentKIEConfigMixin:
    word_labels: List[str] = None


class DocumentKIEMixin:
    def _dataset_features(self):
        return datasets.Features(
            {
                # image
                DataKeys.INDEX: datasets.Value(dtype="int32"),
                DataKeys.IMAGE: Image(decode=True),
                DataKeys.IMAGE_FILE_PATH: datasets.Value(dtype="string"),
                DataKeys.IMAGE_WIDTH: datasets.features.Value(dtype="int32"),
                DataKeys.IMAGE_HEIGHT: datasets.features.Value(dtype="int32"),
                # text
                DataKeys.WORDS: datasets.Sequence(datasets.Value(dtype="string")),
                DataKeys.WORD_BBOXES: datasets.Sequence(
                    datasets.Sequence(datasets.Value(dtype="float"), length=4)
                ),
                # labels
                DataKeys.WORD_LABELS: datasets.Sequence(
                    ClassLabel(
                        names=self.config.word_labels,
                        num_classes=len(self.config.word_labels),
                    )
                ),
            }
        )
