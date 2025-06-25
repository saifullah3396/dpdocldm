import dataclasses
import io
import json
import uuid
from pathlib import Path
from typing import Any, Dict, Generator, List, Tuple

import datasets
import pandas as pd
from atria.core.constants import DataKeys
from atria.core.data.datasets.huggingface_dataset import (
    AtriaHuggingfaceDataset,
    AtriaHuggingfaceDatasetConfig,
)
from atria.core.utilities.logging import get_logger
from docsets.mixins.document_kie import DocumentKIEConfigMixin, DocumentKIEMixin
from docsets.utilities import _get_line_bboxes, _normalize_bbox
from PIL import Image

logger = get_logger(__name__)

_CITATION = """"""
_DESCRIPTION = """CORD Dataset"""
_HOMEPAGE = "https://github.com/clovaai/cord"
_LICENSE = "Apache-2.0 license"

_WORD_LABELS = [
    "O",
    "B-MENU.NM",
    "B-MENU.NUM",
    "B-MENU.UNITPRICE",
    "B-MENU.CNT",
    "B-MENU.DISCOUNTPRICE",
    "B-MENU.PRICE",
    "B-MENU.ITEMSUBTOTAL",
    "B-MENU.VATYN",
    "B-MENU.ETC",
    "B-MENU.SUB.NM",
    "B-MENU.SUB.UNITPRICE",
    "B-MENU.SUB.CNT",
    "B-MENU.SUB.PRICE",
    "B-MENU.SUB.ETC",
    "B-VOID_MENU.NM",
    "B-VOID_MENU.PRICE",
    "B-SUB_TOTAL.SUBTOTAL_PRICE",
    "B-SUB_TOTAL.DISCOUNT_PRICE",
    "B-SUB_TOTAL.SERVICE_PRICE",
    "B-SUB_TOTAL.OTHERSVC_PRICE",
    "B-SUB_TOTAL.TAX_PRICE",
    "B-SUB_TOTAL.ETC",
    "B-TOTAL.TOTAL_PRICE",
    "B-TOTAL.TOTAL_ETC",
    "B-TOTAL.CASHPRICE",
    "B-TOTAL.CHANGEPRICE",
    "B-TOTAL.CREDITCARDPRICE",
    "B-TOTAL.EMONEYPRICE",
    "B-TOTAL.MENUTYPE_CNT",
    "B-TOTAL.MENUQTY_CNT",
    "I-MENU.NM",
    "I-MENU.NUM",
    "I-MENU.UNITPRICE",
    "I-MENU.CNT",
    "I-MENU.DISCOUNTPRICE",
    "I-MENU.PRICE",
    "I-MENU.ITEMSUBTOTAL",
    "I-MENU.VATYN",
    "I-MENU.ETC",
    "I-MENU.SUB.NM",
    "I-MENU.SUB.UNITPRICE",
    "I-MENU.SUB.CNT",
    "I-MENU.SUB.PRICE",
    "I-MENU.SUB.ETC",
    "I-VOID_MENU.NM",
    "I-VOID_MENU.PRICE",
    "I-SUB_TOTAL.SUBTOTAL_PRICE",
    "I-SUB_TOTAL.DISCOUNT_PRICE",
    "I-SUB_TOTAL.SERVICE_PRICE",
    "I-SUB_TOTAL.OTHERSVC_PRICE",
    "I-SUB_TOTAL.TAX_PRICE",
    "I-SUB_TOTAL.ETC",
    "I-TOTAL.TOTAL_PRICE",
    "I-TOTAL.TOTAL_ETC",
    "I-TOTAL.CASHPRICE",
    "I-TOTAL.CHANGEPRICE",
    "I-TOTAL.CREDITCARDPRICE",
    "I-TOTAL.EMONEYPRICE",
    "I-TOTAL.MENUTYPE_CNT",
    "I-TOTAL.MENUQTY_CNT",
]

_BASE_HF_REPO = "https://huggingface.co/datasets/naver-clova-ix/cord-v2"
_DATA_URLS = [
    f"{_BASE_HF_REPO}/resolve/main/data/train-00000-of-00004-b4aaeceff1d90ecb.parquet",
    f"{_BASE_HF_REPO}/resolve/main/data/train-00001-of-00004-7dbbe248962764c5.parquet",
    f"{_BASE_HF_REPO}/resolve/main/data/train-00002-of-00004-688fe1305a55e5cc.parquet",
    f"{_BASE_HF_REPO}/resolve/main/data/train-00003-of-00004-2d0cd200555ed7fd.parquet",
    f"{_BASE_HF_REPO}/resolve/main/data/validation-00000-of-00001-cc3c5779fe22e8ca.parquet",
    f"{_BASE_HF_REPO}/resolve/main/data/test-00000-of-00001-9c204eb3f4e11791.parquet",
]


@dataclasses.dataclass
class CORDConfig(DocumentKIEConfigMixin, AtriaHuggingfaceDatasetConfig):
    pass


class CORD(DocumentKIEMixin, AtriaHuggingfaceDataset):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        CORDConfig(
            name="cordv2",
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
            word_labels=_WORD_LABELS,
            data_url=_DATA_URLS,
        ),
    ]

    def _dataset_features(self):
        dataset_features = super()._dataset_features()
        dataset_features[DataKeys.WORD_BBOXES_SEGMENT_LEVEL] = datasets.Sequence(
            datasets.Sequence(datasets.Value(dtype="float"), length=4)
        )
        return dataset_features

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> List[datasets.SplitGenerator]:
        downloaded_files = self._prepare_data_dir(dl_manager)

        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "data_files": [
                        Path(self.config.data_dir) / k
                        for k in downloaded_files.keys()
                        if "train" in k
                    ]
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "data_files": [
                        Path(self.config.data_dir) / k
                        for k in downloaded_files.keys()
                        if "validation" in k
                    ]
                },
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "data_files": [
                        Path(self.config.data_dir) / k
                        for k in downloaded_files.keys()
                        if "test" in k
                    ]
                },
            ),
        ]

    def _quad_to_box(self, quad: Dict[str, int]) -> Tuple[int, int, int, int]:
        box = (max(0, quad["x1"]), max(0, quad["y1"]), quad["x3"], quad["y3"])
        if box[3] < box[1]:
            bbox = list(box)
            tmp = bbox[3]
            bbox[3] = bbox[1]
            bbox[1] = tmp
            box = tuple(bbox)
        if box[2] < box[0]:
            bbox = list(box)
            tmp = bbox[2]
            bbox[2] = bbox[0]
            bbox[0] = tmp
            box = tuple(bbox)
        return box

    def _create_sample_from_row(self, idx: int, row: Dict[str, Any]) -> Dict[str, Any]:
        words = []
        bboxes = []
        ssl_bboxes = []
        word_labels = []
        image = Image.open(io.BytesIO(row["image"]["bytes"]))
        image_path = Path(self.config.data_dir) / "images" / f"{idx}.png"
        if not image_path.parent.exists():
            image_path.parent.mkdir(parents=True, exist_ok=True)
        if not image_path.exists():
            image.save(image_path)
        annotation = json.loads(row["ground_truth"])
        for item in annotation["valid_line"]:
            cur_line_bboxes = []
            line_words, word_label = item["words"], item["category"]
            line_words = [w for w in line_words if w["text"].strip() != ""]
            if len(line_words) == 0:
                continue
            if word_label == "other":
                for w in line_words:
                    words.append(w["text"])
                    word_labels.append("O")
                    cur_line_bboxes.append(
                        _normalize_bbox(self._quad_to_box(w["quad"]), image.size)
                    )
            else:
                words.append(line_words[0]["text"])
                word_label = word_label.upper().replace("MENU.SUB_", "MENU.SUB.")
                word_labels.append("B-" + word_label)
                cur_line_bboxes.append(
                    _normalize_bbox(
                        self._quad_to_box(line_words[0]["quad"]), image.size
                    )
                )
                for w in line_words[1:]:
                    words.append(w["text"])
                    word_label = word_label.upper().replace("MENU.SUB_", "MENU.SUB.")
                    word_labels.append("I-" + word_label)
                    cur_line_bboxes.append(
                        _normalize_bbox(self._quad_to_box(w["quad"]), image.size)
                    )

            bboxes.extend(cur_line_bboxes)
            cur_line_bboxes = _get_line_bboxes(cur_line_bboxes)
            ssl_bboxes.extend(cur_line_bboxes)

        return {
            DataKeys.IMAGE: image,
            DataKeys.IMAGE_FILE_PATH: image_path,
            DataKeys.WORDS: words,
            DataKeys.WORD_BBOXES: bboxes,
            DataKeys.WORD_BBOXES_SEGMENT_LEVEL: ssl_bboxes,
            DataKeys.WORD_LABELS: [
                self.config.word_labels.index(l) for l in word_labels
            ],
        }

    def _generate_examples(
        self,
        data_files: pd.DataFrame,
    ) -> Generator[Tuple[str, Dict[str, Any]], None, None]:
        data = pd.concat([pd.read_parquet(f) for f in data_files], ignore_index=True)
        for idx, row in data.iterrows():
            yield str(uuid.uuid4()), {
                DataKeys.INDEX: idx,
                **self._create_sample_from_row(idx, row),
            }
