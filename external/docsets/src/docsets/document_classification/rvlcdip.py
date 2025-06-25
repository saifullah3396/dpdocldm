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

"""RVL-CDIP (Ryerson Vision Lab Complex Document Information Processing) dataset"""


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
@inproceedings{harley2015icdar,
    title = {Evaluation of Deep Convolutional Nets for Document Image Classification and Retrieval},
    author = {Adam W Harley and Alex Ufkes and Konstantinos G Derpanis},
    booktitle = {International Conference on Document Analysis and Recognition ({ICDAR})}},
    year = {2015}
}
"""


_DESCRIPTION = """\
The RVL-CDIP (Ryerson Vision Lab Complex Document Information Processing) dataset consists of 400,000 grayscale images in 16 classes, with 25,000 images per class. There are 320,000 training images, 40,000 validation images, and 40,000 test images.
"""


_HOMEPAGE = "https://www.cs.cmu.edu/~aharley/rvl-cdip/"


_LICENSE = "https://www.industrydocuments.ucsf.edu/help/copyright/"

_IMAGE_DATA_NAME = "rvl-cdip"
_OCR_DATA_NAME = "rvl-cdip-ocr"

_URLS = {
    _IMAGE_DATA_NAME: f"https://huggingface.co/datasets/rvl_cdip/resolve/main/data/{_IMAGE_DATA_NAME}.tar.gz",
    _OCR_DATA_NAME: f"https://huggingface.co/datasets/sasa3396/rvlcdip/resolve/main/data/{_OCR_DATA_NAME}.tar.gz",
}

_METADATA_URLS = {  # for default let us always have tobacco3482 overlap removed from the dataset
    "default": {
        "labels/default/train.txt": "https://huggingface.co/datasets/sasa3396/rvlcdip/resolve/main/data/tobacco3482_excluded/train.txt",
        "labels/default/test.txt": "https://huggingface.co/datasets/sasa3396/rvlcdip/resolve/main/data/tobacco3482_excluded/test.txt",
        "labels/default/val.txt": "https://huggingface.co/datasets/sasa3396/rvlcdip/resolve/main/data/tobacco3482_excluded/val.txt",
    },
    "tobacco3482_included": {
        "labels/tobacco3482_included/train.txt": "https://huggingface.co/datasets/sasa3396/rvlcdip/resolve/main/data/default/train.txt",
        "labels/tobacco3482_included/test.txt": "https://huggingface.co/datasets/sasa3396/rvlcdip/resolve/main/data/default/test.txt",
        "labels/tobacco3482_included/val.txt": "https://huggingface.co/datasets/sasa3396/rvlcdip/resolve/main/data/default/val.txt",
    },
}

_CLASSES = [
    "letter",  # 0
    "form",  # 1
    "email",  # 2
    "handwritten",  # 3
    "advertisement",  # 4
    "scientific report",  # 5
    "scientific publication",  # 6
    "specification",  # 7
    "file folder",  # 8
    "news article",  # 9
    "budget",  # 10
    "invoice",  # 11
    "presentation",  # 12
    "questionnaire",  # 13
    "resume",  # 14
    "memo",  # 15
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
    return RvlCdipConfig(
        name=name,
        description=description,
        homepage=_HOMEPAGE,
        citation=_CITATION,
        license=_LICENSE,
        labels=_CLASSES,
        data_url={**_URLS, **_METADATA_URLS["default"]},
        **kwargs,
    )


@dataclass
class RvlCdipConfig(DocumentClassificationConfigMixin, AtriaHuggingfaceDatasetConfig):
    load_ocr: bool = False
    load_text: bool = False
    load_images: bool = True
    ocr_conf_threshold: float = 0.99


class RvlCdip(DocumentClassificationMixin, AtriaHuggingfaceDataset):
    """Ryerson Vision Lab Complex Document Information Processing dataset."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        _generate_config(
            name="images",
            description=_DESCRIPTION,
            load_images=True,
            load_text=False,
        ),
        _generate_config(
            name="text",
            description=_DESCRIPTION,
            load_images=False,
            load_text=True,
        ),
        _generate_config(
            name="images_with_text",
            description=_DESCRIPTION,
            load_images=True,
            load_text=True,
        ),
        _generate_config(
            name="images_with_ocr",
            description=_DESCRIPTION,
            load_images=True,
            load_ocr=True,
        ),
    ]

    def _dataset_features(self, decode_image: bool = True):
        features = datasets.Features(
            {
                # image
                DataKeys.INDEX: datasets.Value(dtype="int32"),
                DataKeys.IMAGE: datasets.Image(decode=True),
                DataKeys.IMAGE_FILE_PATH: datasets.Value(dtype="string"),
                DataKeys.IMAGE_WIDTH: datasets.features.Value(dtype="int32"),
                DataKeys.IMAGE_HEIGHT: datasets.features.Value(dtype="int32"),
                # labels
                DataKeys.LABEL: datasets.features.ClassLabel(names=self.config.labels),
            }
        )
        if self.config.load_text:
            features[DataKeys.WORDS] = datasets.Sequence(datasets.Value(dtype="string"))
            features[DataKeys.WORD_BBOXES] = datasets.Sequence(
                datasets.Sequence(datasets.Value(dtype="float"), length=4)
            )
        if self.config.load_ocr:
            features[DataKeys.HOCR] = datasets.Value(dtype="string")
        return features

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        _ = self._prepare_data_dir(dl_manager)
        image_data_dir = os.path.join(self.config.data_dir, _IMAGE_DATA_NAME, "images")
        ocr_data_dir = os.path.join(self.config.data_dir, _OCR_DATA_NAME, "images")
        train_split_file_paths = os.path.join(
            self.config.data_dir,
            "labels/default/train.txt",
        )
        test_split_file_paths = os.path.join(
            self.config.data_dir,
            "labels/default/test.txt",
        )
        val_split_file_paths = os.path.join(
            self.config.data_dir,
            "labels/default/val.txt",
        )

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "image_data_dir": image_data_dir,
                    "ocr_data_dir": ocr_data_dir,
                    "split_file_paths": train_split_file_paths,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "image_data_dir": image_data_dir,
                    "ocr_data_dir": ocr_data_dir,
                    "split_file_paths": test_split_file_paths,
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "image_data_dir": image_data_dir,
                    "ocr_data_dir": ocr_data_dir,
                    "split_file_paths": val_split_file_paths,
                },
            ),
        ]

    def _image_file_path_to_class_id(self, image_file_path):
        return int(_CLASSES.index(Path(image_file_path).parent.name))

    def _load_sample_from_file_path(
        self, image_data_dir: str, ocr_data_dir: str, image_file_path_with_label: str
    ):
        image_file_path, label = image_file_path_with_label.split(" ")

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
                os.path.join(
                    ocr_data_dir, image_file_path.replace(".tif", ".hocr.lstm")
                ),
                conf_threshold=self.config.ocr_conf_threshold,
            ).parse()
            sample[DataKeys.WORDS] = words
            sample[DataKeys.WORD_BBOXES] = word_bboxes
        if self.config.load_ocr:
            ocr_file_path = Path(ocr_data_dir) / image_file_path.replace(
                ".tif", ".hocr.lstm"
            )
            assert ocr_file_path.exists(), f"OCR file not found: {ocr_file_path}"
            with open(ocr_file_path, "rb") as f:
                sample[DataKeys.HOCR] = gzip.compress(f.read())

        sample[DataKeys.LABEL] = label
        return sample

    def _generate_examples(self, image_data_dir, ocr_data_dir, split_file_paths):
        with open(split_file_paths) as f:
            split_file_paths = f.read().splitlines()

        for idx, image_file_path in enumerate(split_file_paths):
            sample = self._load_sample_from_file_path(
                image_data_dir, ocr_data_dir, image_file_path
            )
            yield str(uuid.uuid4()), {DataKeys.INDEX: idx, **sample}
