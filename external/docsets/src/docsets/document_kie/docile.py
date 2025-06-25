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

"""Docile dataset"""


import base64
import dataclasses
import io
import uuid
from pathlib import Path

import datasets
import numpy as np
import PIL
import pyarrow
import pyarrow_hotfix
from atria.core.constants import DataKeys
from atria.core.data.datasets.huggingface_dataset import (
    AtriaHuggingfaceDataset,
    AtriaHuggingfaceDatasetConfig,
)
from atria.core.utilities.logging import get_logger
from docile.dataset import KILE_FIELDTYPES, LIR_FIELDTYPES, Dataset
from docsets.document_kie.docile_preprocess.preprocessor import (
    generate_unique_entities,
    load_docile_dataset,
    prepare_docile_dataset,
)
from docsets.mixins.document_kie import DocumentKIEConfigMixin, DocumentKIEMixin

pyarrow_hotfix.uninstall()
pyarrow.PyExtensionType.set_auto_load(True)

logger = get_logger(__name__)


# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """"""

# You can copy an official description
_DESCRIPTION = """Docile Dataset"""

_HOMEPAGE = "https://github.com/rossumai/docile"

_LICENSE = "Apache-2.0 license"


_ALL_LABELS = generate_unique_entities()
_KILE_LABELS = (
    [
        "O",
    ]
    + [f"B-{x}" for x in KILE_FIELDTYPES]
    + [f"I-{x}" for x in KILE_FIELDTYPES]
)


_LIR_LABELS = (
    [
        "O",
    ]
    + [f"B-{x}" for x in LIR_FIELDTYPES]
    + [f"I-{x}" for x in LIR_FIELDTYPES]
)


def convert_to_list(row):
    for k, v in row.items():
        if isinstance(v, np.ndarray):
            row[k] = v.tolist()
            if isinstance(v[0], np.ndarray):
                row[k] = [x.tolist() for x in v]
    return row


def filter_empty_words(row):
    filtered_indices = [idx for idx, x in enumerate(row[DataKeys.WORDS]) if x != ""]
    for k in [
        DataKeys.WORDS,
        DataKeys.WORD_BBOXES,
        DataKeys.WORD_BBOXES_SEGMENT_LEVEL,
        DataKeys.WORD_LABELS,
    ]:
        if k not in row.keys():
            continue

        if k == DataKeys.WORDS:
            row[k] = [row[k][idx].strip() for idx in filtered_indices]
        else:
            row[k] = [row[k][idx] for idx in filtered_indices]
    return row


def remap_labels_to_task_labels(labels, config):
    import numpy as np

    all_labels = np.array(_ALL_LABELS)
    if config == "kile":
        label_map = _KILE_LABELS
    elif config == "lir":
        label_map = _LIR_LABELS

    labels_to_idx = dict(zip(label_map, range(len(label_map))))
    remapped_labels = []
    for (
        label
    ) in (
        labels
    ):  # each label is a boolean map to multiple unique entities in _ALL_LABELS
        # here we only take those labels that are present in the label_map (KILE OR LIR or other)
        sample_label = [x for x in all_labels[label] if x in label_map]
        if len(sample_label) > 0:  # now we take the label inde from the label_map
            remapped_labels.append(labels_to_idx[sample_label[0]])
        else:
            remapped_labels.append(labels_to_idx[label_map[0]])
    return remapped_labels


def process(example, config):
    import numpy as np

    all_labels = np.array(_ALL_LABELS)
    if config == "kile":
        label_map = _KILE_LABELS
    elif config == "lir":
        label_map = _LIR_LABELS

    labels_to_idx = dict(zip(label_map, range(len(label_map))))
    labels = example["ner_tags"]
    updated_labels = []
    for idx, label in enumerate(labels):
        # get label id in original map
        indices = np.nonzero(label)[0]

        # get_kile_label
        sample_label = [x for x in all_labels[indices] if x in label_map]
        if len(sample_label) > 0:
            updated_labels.append(labels_to_idx[sample_label[0]])
        else:
            updated_labels.append(labels_to_idx[label_map[0]])

    updated = {
        DataKeys.WORDS: example["tokens"],
        DataKeys.WORD_BBOXES: example["bboxes"],
        DataKeys.WORD_BBOXES_SEGMENT_LEVEL: example[
            "bboxes"
        ],  # we repeat this here as we don't have segment level boxes in this dataset
        DataKeys.WORD_LABELS: updated_labels,
        DataKeys.IMAGE: example["img"],
    }
    return updated


@dataclasses.dataclass
class DocileConfig(DocumentKIEConfigMixin, AtriaHuggingfaceDatasetConfig):
    synthetic: bool = False
    overlap_thr: float = 0.5
    image_shape: tuple = (1024, 1024)


class Docile(DocumentKIEMixin, AtriaHuggingfaceDataset):
    """Docile dataset."""

    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        DocileConfig(
            name="kile",
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            word_labels=_KILE_LABELS,
            synthetic=False,
        ),
        DocileConfig(
            name="lir",
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            word_labels=_LIR_LABELS,
            synthetic=False,
        ),
    ]

    def _initialize_config(self):
        pass

    @property
    def word_labels(self):
        return self.config.word_labels

    def _split_generators(self, dl_manager):
        if not self.config.synthetic:
            for dir in ["preprocessed_dataset"]:
                assert (
                    Path(self.config.data_dir) / dir
                ).exists(), (
                    f"Data directory {self.config.data_dir}/{dir} does not exist."
                )

        for split in ["val", "train"]:
            docile_dataset = Dataset(
                split, self.config.data_dir, load_annotations=False, load_ocr=False
            )
            prepare_docile_dataset(
                docile_dataset,
                self.config.overlap_thr,
                Path(self.config.data_dir),
                image_shape=self.config.image_shape,
            )

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "split": "val"
                },  # We use validation set as test set as we don't have a test set with annotations
            ),
        ]

    def _load_dataset_to_pandas(self, split):
        # here we load the dataset just like how it is loaded in docile
        docile_dataset = Dataset(
            split, self.config.data_dir, load_annotations=False, load_ocr=False
        )

        data = load_docile_dataset(
            docile_dataset,
            self.config.overlap_thr,
            Path(self.config.data_dir) / "preprocessed_dataset",
            image_shape=self.config.image_shape,
        ).as_pandas_dataset()

        # remap token labels to the required task labels for example for KILE we only deal with a subset of all labels
        data["ner_tags"] = data["ner_tags"].apply(
            remap_labels_to_task_labels, args=(self.config.name,)
        )

        # convert tokens to list, in docile tokens are actually words
        data["tokens"] = data["tokens"].apply(lambda x: list(x))

        # rename columns
        data.rename(
            columns={
                "tokens": DataKeys.WORDS,
                "bboxes": DataKeys.WORD_BBOXES,
                "img": DataKeys.IMAGE,
                "ner_tags": DataKeys.WORD_LABELS,
            },
            inplace=True,
        )

        # remove id column
        data.drop(columns=["id"], inplace=True)

        # filter out empty tokens
        data = data.apply(filter_empty_words, axis=1)

        return data

    def _generate_examples(
        self,
        split,
    ):
        data = self._load_dataset_to_pandas(split)
        for idx, sample in data.iterrows():
            if DataKeys.IMAGE in sample:
                pil_image = PIL.Image.open(
                    io.BytesIO(base64.b64decode(sample["image"]))
                )
                image_path = (
                    Path(self.config.data_dir)
                    / f"images_{self.config.image_shape}"
                    / f"{idx}.png"
                )
                if not image_path.parent.exists():
                    image_path.parent.mkdir(parents=True, exist_ok=True)
                pil_image.save(image_path)
                sample[DataKeys.IMAGE_FILE_PATH] = image_path
                sample[DataKeys.IMAGE] = np.array(pil_image)
            sample[DataKeys.INDEX] = idx
            yield str(uuid.uuid4()), sample
