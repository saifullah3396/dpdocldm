import dataclasses
from typing import List

import datasets
from atria.core.constants import DataKeys
from datasets.features import Image

logger = datasets.logging.get_logger(__name__)


@dataclasses.dataclass
class DocumentVQAConfigMixin:
    word_labels: List[str] = None


class DocumentVQAMixin:
    def _dataset_features(self):
        return datasets.Features(
            {
                # image
                DataKeys.INDEX: datasets.Value(dtype="int32"),
                DataKeys.IMAGE: Image(decode=True),
                DataKeys.IMAGE_WIDTH: datasets.features.Value(dtype="int32"),
                DataKeys.IMAGE_HEIGHT: datasets.features.Value(dtype="int32"),
                # text
                DataKeys.WORDS: datasets.Sequence(datasets.Value(dtype="string")),
                DataKeys.WORD_BBOXES: datasets.Sequence(
                    datasets.Sequence(datasets.Value(dtype="float"), length=4)
                ),
                # question/answer
                DataKeys.QUESTION_ID: datasets.Value(dtype="int32"),
                DataKeys.QUESTIONS: datasets.Value(dtype="string"),
                DataKeys.GOLD_ANSWERS: datasets.Sequence(
                    datasets.Value(dtype="string")
                ),
                DataKeys.ANSWER_START_INDICES: datasets.Sequence(
                    datasets.Value(dtype="int32")
                ),
                DataKeys.ANSWER_END_INDICES: datasets.Sequence(
                    datasets.Value(dtype="int32")
                ),
            }
        )
