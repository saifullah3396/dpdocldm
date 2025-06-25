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

"""Tobacco3482 dataset"""


import gzip
import os
import uuid
from dataclasses import dataclass
from pathlib import Path

import datasets
from atria.core.constants import DataKeys
from atria.core.data.datasets.huggingface_dataset import (
    AtriaHuggingfaceDataset,
    AtriaHuggingfaceDatasetConfig,
)
from docsets.mixins.document_cls import (
    DocumentClassificationConfigMixin,
    DocumentClassificationMixin,
)

_CITATION = """\
@article{Kumar2014StructuralSF,
    title={Structural similarity for document image classification and retrieval},
    author={Jayant Kumar and Peng Ye and David S. Doermann},
    journal={Pattern Recognit. Lett.},
    year={2014},
    volume={43},
    pages={119-126}
}
"""


_DESCRIPTION = """\
The Tobacco3482 dataset consists of 3842 grayscale images in 10 classes. In this version, the dataset is plit into 2782 training images, and 700 test images.
"""


_HOMEPAGE = "https://www.kaggle.com/datasets/patrickaudriaz/tobacco3482jpg"


_LICENSE = "https://www.industrydocuments.ucsf.edu/help/copyright/"

_IMAGE_DATA_NAME = "tobacco3482"
_OCR_DATA_NAME = "tobacco3482_ocr"

_DATA_URLS = [
    f"https://huggingface.co/datasets/sasa3396/tobacco3482/resolve/main/data/{_IMAGE_DATA_NAME}.tar.gz",
    f"https://huggingface.co/datasets/sasa3396/tobacco3482/resolve/main/data/{_OCR_DATA_NAME}.tar.gz",
    f"https://huggingface.co/datasets/sasa3396/tobacco3482/resolve/main/data/train.txt",
    f"https://huggingface.co/datasets/sasa3396/tobacco3482/resolve/main/data/test.txt",
]


_CLASSES = [
    "Letter",
    "Resume",
    "Scientific",
    "ADVE",
    "Email",
    "Report",
    "News",
    "Memo",
    "Form",
    "Note",
]


def extract_archive(path):
    import tarfile

    root_path = path.parent
    folder_name = path.name.replace(".tar.gz", "")

    def extract_nonexisting(archive):
        for member in archive.members:
            name = member.name
            if not (root_path / folder_name / name).exists():
                archive.extract(name, path=root_path / folder_name)

    # print(f"Extracting {path.name} into {root_path / folder_name}...")
    with tarfile.open(path) as archive:
        extract_nonexisting(archive)


def folder_iterator(folder):
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            yield os.path.join(subdir, file)


def _generate_config(name, description, **kwargs):
    return Tobacco3482Config(
        name=name,
        description=description,
        homepage=_HOMEPAGE,
        citation=_CITATION,
        license=_LICENSE,
        labels=_CLASSES,
        data_url=_DATA_URLS,
        **kwargs,
    )


@dataclass
class Tobacco3482Config(
    DocumentClassificationConfigMixin, AtriaHuggingfaceDatasetConfig
):
    load_text: bool = False
    load_ocr: bool = False
    load_images: bool = True
    ocr_conf_threshold: float = 0.99


class Tobacco3482(DocumentClassificationMixin, AtriaHuggingfaceDataset):
    """Ryerson Vision Lab Complex Document Information Processing dataset."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        _generate_config(
            name="images",
            description=_DESCRIPTION + " This configuration contains only images.",
            load_images=True,
            load_text=False,
        ),
        _generate_config(
            name="text",
            description=_DESCRIPTION
            + " This configuration contains only text information.",
            load_images=False,
            load_text=True,
        ),
        _generate_config(
            name="images_with_text",
            description=_DESCRIPTION
            + " This configuration contains both images and text information.",
            load_images=True,
            load_text=True,
        ),
        _generate_config(
            name="images_with_ocr",
            description=_DESCRIPTION
            + " This configuration contains both images and text information.",
            load_images=True,
            load_ocr=True,
        ),
    ]

    def _dataset_features(self):
        features = super()._dataset_features()
        if not self.config.load_text:
            features.pop(DataKeys.WORD_BBOXES)
            features.pop(DataKeys.WORDS)
        if not self.config.load_images:
            features.pop(DataKeys.IMAGE)
        if self.config.load_ocr:
            features[DataKeys.HOCR] = datasets.Value(dtype="string")
        return features

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        downloaded_files = self._prepare_data_dir(dl_manager)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "image_data_dir": os.path.join(
                        self.config.data_dir, _IMAGE_DATA_NAME
                    ),
                    "ocr_data_dir": os.path.join(self.config.data_dir, _OCR_DATA_NAME),
                    "split_file_paths": os.path.join(self.config.data_dir, "train.txt"),
                    "split": "train",
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "image_data_dir": os.path.join(
                        self.config.data_dir, _IMAGE_DATA_NAME
                    ),
                    "ocr_data_dir": os.path.join(self.config.data_dir, _OCR_DATA_NAME),
                    "split_file_paths": os.path.join(self.config.data_dir, "test.txt"),
                    "split": "test",
                },
            ),
        ]

    def _image_file_path_to_class_id(self, image_file_path):
        return int(_CLASSES.index(Path(image_file_path).parent.name))

    def _load_sample_from_file_path(
        self, image_data_dir: str, ocr_data_dir: str, image_file_path: str
    ):
        sample = {}
        if self.config.load_images:
            from PIL import Image

            image_path = os.path.join(image_data_dir, image_file_path)
            image = Image.open(image_path)
            sample[DataKeys.IMAGE] = image
            sample[DataKeys.IMAGE_FILE_PATH] = os.path.relpath(
                image_path, self.config.data_dir
            )
            sample[DataKeys.IMAGE_WIDTH], sample[DataKeys.IMAGE_HEIGHT] = image.size
        if self.config.load_text:
            from atria.core.utilities.text_utilities import TesseractOCRReader

            (words, word_bboxes, _, _) = TesseractOCRReader(
                os.path.join(ocr_data_dir, image_file_path.replace(".jpg", ".hocr")),
                conf_threshold=self.config.ocr_conf_threshold,
            ).parse()
            sample[DataKeys.WORDS] = words
            sample[DataKeys.WORD_BBOXES] = word_bboxes
        if self.config.load_ocr:
            ocr_file_path = Path(ocr_data_dir) / image_file_path.replace(
                ".jpg", ".hocr"
            )
            assert ocr_file_path.exists(), f"OCR file not found: {ocr_file_path}"
            with open(ocr_file_path, "rb") as f:
                sample[DataKeys.HOCR] = gzip.compress(f.read())
        sample[DataKeys.LABEL] = self._image_file_path_to_class_id(image_file_path)
        return sample

    def _generate_examples(self, image_data_dir, ocr_data_dir, split_file_paths, split):
        with open(split_file_paths) as f:
            split_file_paths = f.read().splitlines()

        for idx, image_file_path in enumerate(split_file_paths):
            sample = self._load_sample_from_file_path(
                image_data_dir, ocr_data_dir, image_file_path
            )
            yield str(uuid.uuid4()), {DataKeys.INDEX: idx, **sample}
