# Copyright 2022 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""SROIE dataset"""


import dataclasses
import json
import os
import uuid
from pathlib import Path

import datasets
import numpy as np
from atria.core.constants import DataKeys
from atria.core.data.datasets.huggingface_dataset import (
    AtriaHuggingfaceDataset,
    AtriaHuggingfaceDatasetConfig,
)
from PIL import Image

from docsets.mixins.document_kie import DocumentKIEConfigMixin, DocumentKIEMixin
from docsets.utilities import _normalize_bbox

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """"""

# You can copy an official description
_DESCRIPTION = """SROIE Receipts Dataset"""

_HOMEPAGE = "https://rrc.cvc.uab.es/?ch=13"

_LICENSE = "Apache-2.0 license"

_WORD_LABELS = [
    "O",
    "B-COMPANY",
    "I-COMPANY",
    "B-DATE",
    "I-DATE",
    "B-ADDRESS",
    "I-ADDRESS",
    "B-TOTAL",
    "I-TOTAL",
]


def convert_to_list(row):
    for k, v in row.items():
        if isinstance(v, np.ndarray):
            row[k] = v.tolist()
            if isinstance(v[0], np.ndarray):
                row[k] = [x.tolist() for x in v]
    return row


@dataclasses.dataclass
class SROIEConfig(DocumentKIEConfigMixin, AtriaHuggingfaceDatasetConfig):
    pass


class SROIE(DocumentKIEMixin, AtriaHuggingfaceDataset):
    """SROIE dataset."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        SROIEConfig(
            name="default",
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            word_labels=_WORD_LABELS,
        ),
    ]

    def _split_generators(self, dl_manager):
        if self.config.data_dir is None:
            raise ValueError(
                f"dataset_dir is required for {self.__class__.__name__} dataset"
            )

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": Path(self.config.data_dir) / "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": Path(self.config.data_dir) / "test"},
            ),
        ]

    def _generate_examples(
        self,
        filepath,
    ):
        ann_dir = os.path.join(filepath, "tagged")
        image_dir = os.path.join(filepath, "images")

        for idx, fname in enumerate(sorted(os.listdir(image_dir))):
            name, ext = os.path.splitext(fname)
            file_path = os.path.join(ann_dir, name + ".json")
            with open(file_path, "r", encoding="utf8") as f:
                sample = json.load(f)
            image_path = os.path.join(image_dir, fname)
            image = Image.open(image_path)

            filtered_words = []
            filtered_boxes = []
            filtered_labels = []
            for word, box, label in zip(
                sample["words"], sample["bbox"], sample["labels"]
            ):
                word = word.strip()
                if len(word) == 0:
                    continue
                filtered_words.append(word)
                filtered_boxes.append(_normalize_bbox(box, image.size))
                filtered_labels.append(label)

            yield str(uuid.uuid4()), {
                DataKeys.INDEX: idx,
                DataKeys.WORDS: filtered_words,
                DataKeys.WORD_BBOXES: filtered_boxes,
                DataKeys.WORD_LABELS: [
                    self.config.word_labels.index(l) for l in filtered_labels
                ],
                DataKeys.IMAGE_FILE_PATH: image_path,
                DataKeys.IMAGE: image,
            }
