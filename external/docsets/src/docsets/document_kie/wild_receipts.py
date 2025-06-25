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

"""WildReceipts dataset"""

import dataclasses
import json
import uuid
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
from atria.core.constants import DataKeys
from atria.core.data.datasets.huggingface_dataset import (
    AtriaHuggingfaceDataset,
    AtriaHuggingfaceDatasetConfig,
)
from PIL import Image

from docsets.mixins.document_kie import DocumentKIEConfigMixin, DocumentKIEMixin
from docsets.utilities import _normalize_bbox, _sorted_indices_in_reading_order

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """"""

# You can copy an official description
_DESCRIPTION = """WildReceipts Dataset"""

_HOMEPAGE = ""

_LICENSE = "Apache-2.0 license"

_DATA_URLS = "https://download.openmmlab.com/mmocr/data/wildreceipt.tar"

_WORD_LABELS = [
    "B-Store_name_value",
    "B-Store_name_key",
    "B-Store_addr_value",
    "B-Store_addr_key",
    "B-Tel_value",
    "B-Tel_key",
    "B-Date_value",
    "B-Date_key",
    "B-Time_value",
    "B-Time_key",
    "B-Prod_item_value",
    "B-Prod_item_key",
    "B-Prod_quantity_value",
    "B-Prod_quantity_key",
    "B-Prod_price_value",
    "B-Prod_price_key",
    "B-Subtotal_value",
    "B-Subtotal_key",
    "B-Tax_value",
    "B-Tax_key",
    "B-Tips_value",
    "B-Tips_key",
    "B-Total_value",
    "B-Total_key",
    "O",
]


def convert_to_list(row):
    for k, v in row.items():
        if isinstance(v, np.ndarray):
            row[k] = v.tolist()
            if isinstance(v[0], np.ndarray):
                row[k] = [x.tolist() for x in v]
    return row


@dataclasses.dataclass
class WildReceiptsConfig(DocumentKIEConfigMixin, AtriaHuggingfaceDatasetConfig):
    apply_reading_order_correction: bool = False


class WildReceipts(DocumentKIEMixin, AtriaHuggingfaceDataset):
    """WildReceipts dataset."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        WildReceiptsConfig(
            name="default",
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            word_labels=_WORD_LABELS,
            data_url=_DATA_URLS,
        ),
    ]
    BUILDER_CONFIGS = [
        WildReceiptsConfig(
            name="corrected_reading_order",
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            word_labels=_WORD_LABELS,
            data_url=_DATA_URLS,
            apply_reading_order_correction=True,
        ),
    ]

    def _split_generators(self, dl_manager):
        downloaded_files = self._prepare_data_dir(dl_manager)
        base_path = Path(self.config.data_dir) / "wildreceipt"
        for file in ["train.txt", "test.txt"]:
            assert (
                base_path / file
            ).exists(), f"Data directory {base_path}/{file} does not exist."
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"filepath": base_path / "train.txt"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"filepath": base_path / "test.txt"},
            ),
        ]

    def _generate_examples(
        self,
        filepath,
    ):
        item_list = []
        with open(filepath, "r") as f:
            for line in f:
                item_list.append(line.rstrip("\n\r"))

        class_list = pd.read_csv(
            filepath.parent / "class_list.txt", delimiter="\s", header=None
        )
        id2labels = dict(zip(class_list[0].tolist(), class_list[1].tolist()))
        for idx, fname in enumerate(item_list):
            ann = json.loads(fname)
            image_path = filepath.parent / ann["file_name"]
            image = Image.open(image_path).convert("RGB")

            words = []
            labels = []
            bboxes = []
            for i in ann["annotations"]:
                label = id2labels[i["label"]]
                if (
                    label == "Ignore" or i["text"] == ""
                ):  # label 0 is attached to ignore so we skip it
                    continue
                if label in ["Others"]:
                    label = "O"
                else:
                    label = "B-" + label
                labels.append(label)
                words.append(i["text"])
                bboxes.append(
                    _normalize_bbox(
                        [i["box"][6], i["box"][7], i["box"][2], i["box"][3]], image.size
                    )
                )

            # sort the word reading order
            if self.config.apply_reading_order_correction:
                sorted_indces = _sorted_indices_in_reading_order(bboxes)
                words = [words[i] for i in sorted_indces]
                labels = [labels[i] for i in sorted_indces]
                bboxes = [bboxes[i] for i in sorted_indces]

            yield str(uuid.uuid4()), {
                DataKeys.INDEX: idx,
                DataKeys.WORDS: words,
                DataKeys.WORD_BBOXES: bboxes,
                DataKeys.WORD_LABELS: [
                    self.config.word_labels.index(l) for l in labels
                ],
                DataKeys.IMAGE_FILE_PATH: image_path,
                DataKeys.IMAGE: image,
            }
