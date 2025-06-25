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


from __future__ import annotations

import gzip
import os
import pickle
import uuid

# count occurences of unique labels
from dataclasses import dataclass
from pathlib import Path
from typing import List

import datasets
import numpy as np
from atria.core.constants import DataKeys
from atria.core.data.datasets.huggingface_dataset import (
    AtriaHuggingfaceDataset,
    AtriaHuggingfaceDatasetConfig,
)
from atria.core.utilities.logging import get_logger
from datadings.reader import MsgpackReader
from docsets.document_classification.iitcdip_labels import LABEL_WEIGHTS, LABELS
from docsets.mixins.document_cls import (
    DocumentClassificationConfigMixin,
    DocumentClassificationMixin,
)

logger = get_logger(__name__)

_CITATION = """\
@inproceedings{harley2015icdar,
    title = {Evaluation of Deep Convolutional Nets for Document Image Classification and Retrieval},
    author = {Adam W Harley and Alex Ufkes and Konstantinos G Derpanis},
    booktitle = {International Conference on Document Analysis and Recognition ({ICDAR})}},
    year = {2015}
}
"""


_DESCRIPTION = """\
The IIT-CDIP (Ryerson Vision Lab Complex Document Information Processing) dataset consists of 400,000 grayscale images in 16 classes, with 25,000 images per class. There are 320,000 training images, 40,000 validation images, and 40,000 test images.
"""


_HOMEPAGE = ""


_LICENSE = "https://www.industrydocuments.ucsf.edu/help/copyright/"

IMAGE_FILES = [
    f"images/data_{i}_{i+250000}.msgpack" for i in range(0, 10000000, 250000)
]
METADATA_FILES = [
    f"metadata/data_{i}_{i+250000}.msgpack" for i in range(0, 10000000, 250000)
]
TRAIN_TEST_SPLIT_RATIO = 0.995


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
    min_label_occurences: int = 10000,
):
    return IitCdipConfig(
        name=name,
        description=description,
        homepage=_HOMEPAGE,
        citation=_CITATION,
        license=_LICENSE,
        load_ocr=load_ocr,
        load_images=load_images,
        labels=LABELS[min_label_occurences],
        label_weights=LABEL_WEIGHTS[min_label_occurences],
    )


@dataclass
class IitCdipConfig(DocumentClassificationConfigMixin, AtriaHuggingfaceDatasetConfig):
    load_ocr: bool = False
    load_images: bool = True
    ocr_conf_threshold: float = 0.99
    min_label_occurences: int = 10000
    label_weights: List[float] = None


class IitCdip(DocumentClassificationMixin, AtriaHuggingfaceDataset):
    """Industry Tobacco dataset."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        _generate_config(
            name="images_labelled",
            description=_DESCRIPTION,
            load_images=True,
            load_ocr=False,
        ),
        _generate_config(
            name="images_labelled_with_ocr",
            description=_DESCRIPTION,
            load_images=True,
            load_ocr=True,
        ),
        _generate_config(
            name="images_unlabelled",
            description=_DESCRIPTION,
            load_images=True,
            load_ocr=False,
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
                # hocr
                DataKeys.HOCR: datasets.Value(dtype="string"),
                # labels
                DataKeys.LABEL: datasets.features.ClassLabel(names=self.config.labels),
                DataKeys.SAMPLE_WEIGHT: datasets.Value(dtype="float"),
            }
        )

    def _load_preprocessed_msgpacks(self):
        image_msgpacks = []
        metadata_msgpacks = []
        base_path = Path(f"{self.config.data_dir}/preprocessing/")
        if self.config.load_images:
            for image_file in IMAGE_FILES:
                try:
                    image_msgpack = MsgpackReader(str(base_path / image_file))
                except Exception as e:
                    raise RuntimeError(
                        f"Unable to load msgpack file {str(base_path / image_file)}: {e}"
                    )
                image_msgpacks.append(image_msgpack)

        if self.config.load_ocr:
            for metadata_file in METADATA_FILES:
                try:
                    metadata_msgpack = MsgpackReader(str(base_path / metadata_file))
                except Exception as e:
                    raise RuntimeError(
                        f"Unable to load msgpack file {str(base_path / metadata_file)}: {e}"
                    )
                metadata_msgpacks.append(metadata_msgpack)

        return image_msgpacks, metadata_msgpacks

    def _load_dataset_metadata(
        self,
        image_shard_msgpacks: List[MsgpackReader],
        metadata_shard_msgpacks: List[MsgpackReader],
    ):
        if len(metadata_shard_msgpacks) > 0:
            assert len(metadata_shard_msgpacks) == len(
                image_shard_msgpacks
            ), f"Number of metadata and image msgpacks should be the same"

        shard_start_indices = []
        total_size = 0
        sample_index_to_shard_index = []
        for idx, image_msgpack in enumerate(image_shard_msgpacks):
            sample_index_to_shard_index += [idx] * len(image_msgpack)
            shard_start_indices.append(total_size)
            total_size += len(image_msgpack)
            if len(metadata_shard_msgpacks) > 0:
                assert len(metadata_shard_msgpacks[idx]) == len(
                    image_msgpack
                ), f"Number of metadata and image samples should be the same"

        return shard_start_indices, total_size, sample_index_to_shard_index

    def _prepare_label_metadata(self):
        base_path = Path(f"{self.config.data_dir}/preprocessing/")
        with open(base_path / "curated_labels" / "labels.pickle", "rb") as f:
            sample_labels = pickle.load(f)

        # find unique class labels and their counts to calculate weights
        splitted_labels = [
            label.split(";") for label in sample_labels if label != "NO_LABEL"
        ]
        splitted_labels = [label for sublist in splitted_labels for label in sublist]

        # generate finalized sample indices, labels, and weights
        split_indices = []
        split_weights = []
        split_labels = []
        for sample_idx, label in enumerate(sample_labels):
            all_labels_in_sample = [
                x for x in label.split(";") if x in self.config.labels
            ]
            if len(all_labels_in_sample) == 0:
                continue

            # take the lowest frequency label in the sample
            all_labels_in_sample = sorted(
                all_labels_in_sample,
                key=lambda x: self.config.label_weights[self.config.labels.index(x)],
            )

            # this contains labels for all samples whether they are in subset or not
            filtered_sample_label = all_labels_in_sample[
                -1
            ]  # take highest weight label
            split_labels.append(self.config.labels.index(filtered_sample_label))

            # this is equal to the total size of the dataset
            split_weights.append(
                self.config.label_weights[
                    self.config.labels.index(filtered_sample_label)
                ]
            )
            split_indices.append(sample_idx)
        split_indices = np.array(split_indices)
        assert len(split_indices) == len(split_weights) == len(split_labels), (
            "Sample indices, labels, and weights should have the same length"
            f" {len(split_indices)}, {len(split_weights)}, {len(split_labels)}"
        )
        return split_indices, split_labels, split_weights

    def _split_generators(self, dl_manager: datasets.DownloadManager):
        assert (
            self.config.data_dir is not None
        ), f"data_dir must be provided for {self.__class__.__name__} dataset."

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"split": datasets.Split.TRAIN},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"split": datasets.Split.TEST},
            ),
        ]

    def _generate_examples(
        self,
        split: datasets.Split,
    ):
        # first we load the preprocessed msgpacks for this dataset
        self._image_shard_msgpacks, self._metadata_shard_msgpacks = (
            self._load_preprocessed_msgpacks()
        )
        (
            self._shard_start_indices,
            self._total_size,
            self._sample_index_to_shard_index_map,
        ) = self._load_dataset_metadata(
            self._image_shard_msgpacks, self._metadata_shard_msgpacks
        )
        split_indices = np.arange(self._total_size)

        logger.info("Loading IIT-CDIP dataset...")
        logger.info(f"Shard start indices: {self._shard_start_indices}")
        logger.info(f"Total size: {self._total_size}")

        # if load_labelled_subset is True, we only take the indices that have labels
        split_labels = None
        split_weights = None
        if self.config.name in ["images_labelled", "images_labelled_with_ocr"]:
            (
                split_indices,
                split_labels,
                split_weights,
            ) = self._prepare_label_metadata()

        # then we split the indices based on the split ratio
        train_test_split_idx = int(TRAIN_TEST_SPLIT_RATIO * len(split_indices))
        if split == datasets.Split.TRAIN:
            split_indices = split_indices[:train_test_split_idx]
            split_labels = split_labels[:train_test_split_idx]
            split_weights = split_weights[:train_test_split_idx]
        elif split == datasets.Split.TEST:
            split_indices = split_indices[train_test_split_idx:]
            split_labels = split_labels[train_test_split_idx:]
            split_weights = split_weights[train_test_split_idx:]

        logger.info(f"Total size in: {split_indices}")
        for sample_idx, global_sample_index in enumerate(split_indices):
            # get the shard index for this sample and load the corresponding image and metadata
            shard_index = self._sample_index_to_shard_index_map[global_sample_index]
            image_reader = self._image_shard_msgpacks[shard_index]
            metadata_reader = None
            if len(self._metadata_shard_msgpacks) > 0:
                metadata_reader = self._metadata_shard_msgpacks[shard_index]

            # load image
            image_data = image_reader[
                global_sample_index - self._shard_start_indices[shard_index]
            ]

            # load metadata
            if metadata_reader is not None:
                metadata = metadata_reader[
                    global_sample_index - self._shard_start_indices[shard_index]
                ]

                # load the metadata as a dictionary
                assert image_data["key"] == metadata["key"]
                metadata = pickle.loads(metadata["data"])

            # make sure the unique sample keys match between image and metadata
            image_file_path = image_data["key"]
            image = image_data["image"]

            # load the image as a PIL image
            sample = {}
            if self.config.load_images:
                sample[DataKeys.IMAGE] = image
                sample[DataKeys.IMAGE_FILE_PATH] = image_file_path

            if self.config.load_ocr:
                sample[DataKeys.HOCR] = gzip.compress(metadata["hocr"])

            yield str(uuid.uuid4()), {
                **sample,
                DataKeys.INDEX: sample_idx,
                DataKeys.LABEL: split_labels[sample_idx] if split_labels else None,
                DataKeys.SAMPLE_WEIGHT: (
                    split_weights[sample_idx] if split_weights else None
                ),
            }
