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

"""CCPDF dataset"""


from __future__ import annotations

import os
import pickle

# count occurences of unique labels
from dataclasses import dataclass
from pathlib import Path
import uuid

import datasets
import tqdm
from atria.core.constants import DataKeys
from atria.core.data.datasets.huggingface_dataset import (
    AtriaHuggingfaceDataset,
    AtriaHuggingfaceDatasetConfig,
)
from atria.core.utilities.logging import get_logger
from docsets.mixins.document_cls import (
    DocumentClassificationConfigMixin,
    DocumentClassificationMixin,
)

logger = get_logger(__name__)

_CITATION = """
"""


_DESCRIPTION = """
"""


_HOMEPAGE = ""


_LICENSE = ""


def extract_archive(path: Path):
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


def folder_iterator(folder: str):
    for subdir, dirs, files in os.walk(folder):
        for file in files:
            yield os.path.join(subdir, file)


def _generate_config(
    name: str,
    description: str,
    load_images: bool,
    load_ocr: bool,
):
    return CCPDFConfig(
        name=name,
        description=description,
        homepage=_HOMEPAGE,
        citation=_CITATION,
        license=_LICENSE,
        load_ocr=load_ocr,
        load_images=load_images,
    )


@dataclass
class CCPDFConfig(DocumentClassificationConfigMixin, AtriaHuggingfaceDatasetConfig):
    load_ocr: bool = False
    load_images: bool = True
    ocr_conf_threshold: float = 0.99
    max_samples: int = 1000


class CCPDF(DocumentClassificationMixin, AtriaHuggingfaceDataset):
    """Industry Tobacco dataset."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        _generate_config(
            name="images_with_ocr",
            description=_DESCRIPTION,
            load_images=True,
            load_ocr=True,
        ),
    ]

    def _dataset_features(self):
        return datasets.Features(
            {
                # image
                DataKeys.INDEX: datasets.Value(dtype="int32"),
                DataKeys.IMAGE: datasets.Image(decode=False),
                DataKeys.IMAGE_FILE_PATH: datasets.Value(dtype="string"),
                DataKeys.IMAGE_WIDTH: datasets.features.Value(dtype="int32"),
                DataKeys.IMAGE_HEIGHT: datasets.features.Value(dtype="int32"),
                DataKeys.PAGE_NUM: datasets.Value(dtype="int32"),
                DataKeys.PDF_FILE_PATH: datasets.Value(dtype="string"),
                # hocr
                DataKeys.HOCR: datasets.Value(dtype="string"),
            }
        )

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        assert (
            self.config.data_dir is not None
        ), f"data_dir must be provided for {self.__class__.__name__} dataset."

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
            ),
        ]

    def _generate_examples(
        self,
    ):
        data_dir = Path(self.config.data_dir)
        pdfs_dir = data_dir / "downloaded"
        ocr_dir = data_dir / "tesseract_ocr"

        if self.config.load_images and self.config.load_ocr:
            assert (
                pdfs_dir.exists() and ocr_dir.exists()
            ), f"PDFs and OCR directories do not exist in {pdfs_dir} and {ocr_dir}."

            sample_idx = 0
            for ocr_file in tqdm.tqdm(os.listdir(ocr_dir)):
                # if sample_idx >= self.config.max_samples:
                #     break
                with open(ocr_dir / ocr_file, "rb") as f:
                    ocr_data = pickle.load(f)
                    pdf_file = pdfs_dir / (Path(ocr_file).stem + ".pdf")
                    if not pdf_file.exists():
                        logger.warning(f"PDF file {pdf_file} does not exist.")
                        continue
                    for page_num in range(len(ocr_data)):
                        sample = {}
                        sample[DataKeys.HOCR] = ocr_data[page_num]
                        sample[DataKeys.PDF_FILE_PATH] = pdf_file
                        sample[DataKeys.PAGE_NUM] = page_num
                        sample[DataKeys.INDEX] = sample_idx
                        sample_idx += 1
                        yield str(uuid.uuid4()), sample
