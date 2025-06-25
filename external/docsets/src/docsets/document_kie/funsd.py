import dataclasses
import json
import os
import uuid
from typing import Any, Dict, Generator, List, Tuple

import datasets
from atria.core.constants import DataKeys
from atria.core.data.datasets.huggingface_dataset import (
    AtriaHuggingfaceDataset,
    AtriaHuggingfaceDatasetConfig,
)
from atria.core.utilities.logging import get_logger
from docsets.mixins.document_kie import DocumentKIEConfigMixin, DocumentKIEMixin
from docsets.utilities import (
    _get_line_bboxes,
    _normalize_bbox,
    _sorted_indices_in_reading_order,
)
from PIL import Image

logger = get_logger(__name__)

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """"""

# You can copy an official description
_DESCRIPTION = """FUNSD Dataset"""

_HOMEPAGE = "https://guillaumejaume.github.io/FUNSD/"
_DATA_URL = "https://guillaumejaume.github.io/FUNSD/dataset.zip"

_LICENSE = "Apache-2.0 license"

_WORD_LABELS = [
    "O",
    "B-HEADER",
    "I-HEADER",
    "B-QUESTION",
    "I-QUESTION",
    "B-ANSWER",
    "I-ANSWER",
]


@dataclasses.dataclass
class FUNSDConfig(DocumentKIEConfigMixin, AtriaHuggingfaceDatasetConfig):
    apply_reading_order_correction: bool = False


class FUNSD(DocumentKIEMixin, AtriaHuggingfaceDataset):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        FUNSDConfig(
            name="default",
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            word_labels=_WORD_LABELS,
            data_url=_DATA_URL,
        ),
        FUNSDConfig(
            name="corrected_reading_order",
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            word_labels=_WORD_LABELS,
            data_url=_DATA_URL,
            apply_reading_order_correction=True,
        ),
    ]

    def _dataset_features(self):
        dataset_features = super()._dataset_features()
        dataset_features[DataKeys.WORD_BBOXES_SEGMENT_LEVEL] = datasets.Sequence(
            datasets.Sequence(datasets.Value(dtype="float"), length=4)
        )
        dataset_features["index"] = datasets.Value(dtype="int32")
        return dataset_features

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:
        _ = self._prepare_data_dir(dl_manager)
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": f"{self.config.data_dir}/dataset/dataset/training_data"
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": f"{self.config.data_dir}/dataset/dataset/testing_data"
                },
            ),
        ]

    def _generate_examples(
        self,
        filepath: str,
    ) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        ann_dir = os.path.join(filepath, "annotations")
        image_dir = os.path.join(filepath, "images")

        for idx, fname in enumerate(sorted(os.listdir(image_dir))):
            name, ext = os.path.splitext(fname)
            ann_path = os.path.join(ann_dir, name + ".json")
            image_path = os.path.join(image_dir, fname)
            with open(ann_path, "r", encoding="utf8") as f:
                annotation = json.load(f)
            image = Image.open(image_path)

            words_list = []
            bboxes = []
            ssl_bboxes = []
            labels = []
            # get annotations
            for item in annotation["form"]:
                cur_line_bboxes = []
                words, label = item["words"], item["label"]
                words = [w for w in words if w["text"].strip() != ""]
                if len(words) == 0:
                    continue

                if label == "other":
                    for w in words:
                        words_list.append(w["text"])
                        labels.append("O")
                        cur_line_bboxes.append(_normalize_bbox(w["box"], image.size))
                else:
                    words_list.append(words[0]["text"])
                    labels.append("B-" + label.upper())
                    cur_line_bboxes.append(_normalize_bbox(words[0]["box"], image.size))
                    for w in words[1:]:
                        words_list.append(w["text"])
                        labels.append("I-" + label.upper())
                        cur_line_bboxes.append(_normalize_bbox(w["box"], image.size))

                # add per word box
                bboxes.extend(cur_line_bboxes)

                # add segment level box
                cur_line_bboxes = _get_line_bboxes(cur_line_bboxes)
                ssl_bboxes.extend(cur_line_bboxes)

            # sort the word reading order
            if self.config.apply_reading_order_correction:
                sorted_indces = _sorted_indices_in_reading_order(bboxes)
                words_list = [words_list[i] for i in sorted_indces]
                labels = [labels[i] for i in sorted_indces]
                bboxes = [bboxes[i] for i in sorted_indces]
                ssl_bboxes = [ssl_bboxes[i] for i in sorted_indces]

            yield str(uuid.uuid4()), {
                DataKeys.INDEX: idx,
                DataKeys.WORDS: words_list,
                DataKeys.WORD_BBOXES: bboxes,
                DataKeys.WORD_BBOXES_SEGMENT_LEVEL: ssl_bboxes,
                DataKeys.WORD_LABELS: [
                    self.config.word_labels.index(l) for l in labels
                ],
                DataKeys.IMAGE_FILE_PATH: image_path,
                DataKeys.IMAGE: image,
            }
