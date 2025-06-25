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

"""DockBank dataset"""


import dataclasses
import json
import os
import uuid
from pathlib import Path

import datasets
from atria.core.constants import DataKeys
from atria.core.data.datasets.huggingface_dataset import (
    AtriaHuggingfaceDataset,
    AtriaHuggingfaceDatasetConfig,
)
from atria.core.utilities.logging import get_logger
from PIL import Image

from docsets.mixins.document_ler import DocumentLERConfigMixin, DocumentLERMixin

logger = get_logger(__name__)

# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@misc{li2020docbank,
    title={DocBank: A Benchmark Dataset for Document Layout Analysis},
    author={Minghao Li and Yiheng Xu and Lei Cui and Shaohan Huang and Furu Wei and Zhoujun Li and Ming Zhou},
    year={2020},
    eprint={2006.01038},
    archivePrefix={arXiv},
    primaryClass={cs.CL}
}
"""

# You can copy an official description
_DESCRIPTION = """\
DocBank is a new large-scale dataset that is constructed using a weak supervision approach.
It enables models to integrate both the textual and layout information for downstream tasks.
The current DocBank dataset totally includes 500K document pages, where 400K for training, 50K for validation and 50K for testing.
"""

_HOMEPAGE = "https://doc-analysis.github.io/docbank-page/index.html"

_LICENSE = "Apache-2.0 license"

_CLASSES = [
    "abstract",
    "author",
    "caption",
    "equation",
    "figure",
    "footer",
    "list",
    "paragraph",
    "reference",
    "section",
    "table",
    "title",
    "date",
]

MAX_WORDS_PER_SAMPLE = 4000


@dataclasses.dataclass
class DocBankLERConfig(DocumentLERConfigMixin, AtriaHuggingfaceDatasetConfig):
    pass


class DocBankLER(DocumentLERMixin, AtriaHuggingfaceDataset):
    BUILDER_CONFIGS = [
        DocBankLERConfig(
            name="default",
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            word_labels=_CLASSES,
        ),
    ]

    def _split_generators(self, dl_manager):
        assert (
            self.config.data_dir is not None
        ), f"data_dir must be provided for {self.__class__.__name__} dataset."

        split_base_dir = Path(self.config.data_dir)
        split_generators = []
        for split, filepath in zip(
            [datasets.Split.TRAIN, datasets.Split.VALIDATION, datasets.Split.TEST],
            ["500K_train.json", "500K_valid.json", "500K_test.json"],
        ):
            split_filepath = split_base_dir / filepath
            assert (split_filepath).exists(), f"File {split_filepath} does not exist."
            split_generators.append(
                datasets.SplitGenerator(
                    name=split,
                    gen_kwargs={"split_filepath": split_base_dir / filepath},
                ),
            )
        return split_generators

    def _create_sample_from_ann(self, text_file, image_file_path):
        words = []
        word_bboxes = []
        labels = []

        with open(text_file, "r", encoding="utf8") as fp:
            for line in fp.readlines():
                tts = line.split("\t")
                if not len(tts) == 10:
                    logger.warning("Incomplete line in file {}".format(text_file))
                    continue

                word = tts[0]
                bbox = list(map(int, tts[1:5]))
                structure = tts[9]

                if len(word) == 0:
                    continue
                x1, y1, x2, y2 = bbox
                area = (x2 - x1) * (y2 - y1)
                if (
                    word == "##LTLine##" or word == "##LTFigure##" and area < 10
                ):  # remove ltline and ltfigures with very small noisy features
                    continue

                words.append(word)
                word_bboxes.append(bbox)  # boxes are already normalized 0 to 1000
                labels.append(structure.strip())
        image = Image.open(image_file_path)
        return {
            DataKeys.IMAGE: image,
            DataKeys.IMAGE_WIDTH: image.size[0],
            DataKeys.IMAGE_HEIGHT: image.size[1],
            DataKeys.WORDS: words,
            DataKeys.WORD_BBOXES: word_bboxes,
            DataKeys.WORD_LABELS: [_CLASSES.index(l) for l in labels],
        }

    def _generate_examples(
        self,
        split_filepath: str,
    ):
        image_base_dir = Path(self.config.data_dir) / "DocBank_500K_ori_img/"
        annotation_base_dir = Path(self.config.data_dir) / "DocBank_500K_txt/"
        with open(split_filepath, "r") as f:
            split_data = json.load(f)
            for idx in range(len(split_data["images"])):
                image_file_path = split_data["images"][idx]["file_name"]
                text_file = os.path.join(
                    annotation_base_dir,
                    image_file_path.replace("_ori.jpg", "") + ".txt",
                )
                sample = self._create_sample_from_ann(
                    text_file, image_base_dir / image_file_path
                )
                if (
                    len(sample[DataKeys.WORDS]) > 0
                    and len(sample[DataKeys.WORDS]) < MAX_WORDS_PER_SAMPLE
                ):
                    yield str(uuid.uuid4()), {
                        DataKeys.INDEX: idx,
                        DataKeys.IMAGE_FILE_PATH: "DocBank_500K_ori_img/"
                        + image_file_path,
                        **sample,
                    }
