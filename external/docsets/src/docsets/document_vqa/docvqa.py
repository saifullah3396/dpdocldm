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

"""DocVQA dataset"""
import dataclasses
import json
import uuid
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import tqdm
from atria.core.constants import DataKeys
from atria.core.data.datasets.huggingface_dataset import (
    AtriaHuggingfaceDataset,
    AtriaHuggingfaceDatasetConfig,
)
from atria.core.utilities.logging import get_logger
from datasets import load_from_disk
from PIL import Image

from docsets.document_vqa.utilities import (
    anls_metric_str,
    extract_start_end_index_v1,
    extract_start_end_index_v2,
    extract_start_end_index_v3,
)
from docsets.mixins.document_vqa import DocumentVQAConfigMixin, DocumentVQAMixin
from docsets.utilities import _normalize_bbox

logger = get_logger(__name__)
# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """"""

# You can copy an official description
_DESCRIPTION = """DocVQA Receipts Dataset"""

_HOMEPAGE = "https://rrc.cvc.uab.es/?ch=13"

_LICENSE = "Apache-2.0 license"


def convert_to_list(row):
    for k, v in row.items():
        if isinstance(v, np.ndarray):
            row[k] = v.tolist()
            if isinstance(v[0], np.ndarray):
                row[k] = [x.tolist() for x in v]
    return row


def get_subword_start_end(word_start, word_end, subword_idx2word_idx, sequence_ids):
    ## find the separator between the questions and the text
    start_of_context = -1
    for i in range(len(sequence_ids)):
        if sequence_ids[i] == 1:
            start_of_context = i
            break
    num_question_tokens = start_of_context
    assert start_of_context != -1, "Could not find the start of the context"
    subword_start = -1
    subword_end = -1
    for i in range(start_of_context, len(subword_idx2word_idx)):
        if word_start == subword_idx2word_idx[i] and subword_start == -1:
            subword_start = i
        if word_end == subword_idx2word_idx[i]:
            subword_end = i
    return subword_start, subword_end, num_question_tokens


def find_answers_in_words(words, answers, extraction_method="v1"):
    if extraction_method == "v1":
        return extract_start_end_index_v1(answers, words)
    elif extraction_method == "v2":
        return extract_start_end_index_v2(answers, words)
    elif extraction_method == "v1_v2":
        processed_answers, all_not_found = extract_start_end_index_v1(answers, words)
        if all_not_found:
            processed_answers, _ = extract_start_end_index_v2(answers, words)
        return processed_answers, all_not_found
    elif extraction_method == "v2_v1":
        processed_answers, all_not_found = extract_start_end_index_v2(answers, words)
        if all_not_found:
            processed_answers, _ = extract_start_end_index_v1(answers, words)
        return processed_answers, all_not_found
    elif extraction_method == "v3":
        processed_answers, all_not_found = extract_start_end_index_v3(answers, words)
        return processed_answers, all_not_found
    else:
        raise ValueError(f"Extraction method {extraction_method} not supported")


@dataclasses.dataclass
class DocVQAConfig(DocumentVQAConfigMixin, AtriaHuggingfaceDatasetConfig):
    answers_extraction_method: str = "v1"
    use_msr_ocr: bool = True
    store_single_answer_per_example: bool = True


class DocVQA(DocumentVQAMixin, AtriaHuggingfaceDataset):
    VERSION = datasets.Version("1.0.0")

    BUILDER_CONFIGS = [
        DocVQAConfig(
            name="default",
            description=_DESCRIPTION,
            homepage=_HOMEPAGE,
            citation=_CITATION,
            license=_LICENSE,
        ),
    ]

    def _dataset_features(self):
        features = super()._dataset_features()
        features[DataKeys.WORD_BBOXES_SEGMENT_LEVEL] = datasets.Sequence(
            datasets.Sequence(datasets.Value(dtype="float"), length=4)
        )
        return features

    def _split_generators(
        self, dl_manager: datasets.DownloadManager
    ) -> list[datasets.SplitGenerator]:
        if self.config.data_dir is None:
            raise ValueError(
                f"dataset_dir is required for {self.__class__.__name__} dataset"
            )

        return [
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "split": "val",
                    "filepath": Path(self.config.data_dir)
                    / "spdocvqa_qas"
                    / "val_v1.0_withQT.json",
                },  # DocVQA does not provide answers in test set so we use validation for test
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "split": "test",
                    "filepath": Path(self.config.data_dir)
                    / "spdocvqa_qas"
                    / "test_v1.0.json",
                },  # DocVQA does not provide answers in test set so we use validation for test
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "split": "train",
                    "filepath": Path(self.config.data_dir)
                    / "spdocvqa_qas"
                    / "train_v1.0_withQT.json",
                },
            ),
        ]

    def _load_and_preprocess_dataset(self, filepath: str, split: str) -> list[dict]:
        image_dir = Path(self.config.data_dir)
        ocr_dir = Path(self.config.data_dir) / "spdocvqa_ocr"
        with open(filepath) as f:
            dataset = pd.DataFrame(json.load(f)["data"])

        msr_data = None
        if self.config.use_msr_ocr:
            if split in ["train", "val"]:
                msr_file = (
                    Path(self.config.data_dir) / "msr" / f"docvqa_proc_{split}_t3_ocr"
                )
                msr_data = load_from_disk(str(msr_file))
            else:
                msr_file = Path(self.config.data_dir) / "msr" / "test_v1.0_msr.json"
                with open(msr_file, "r", encoding="utf-8") as read_file:
                    msr_data = json.load(read_file)

        processed_dataset = []
        all_gold = []
        all_extracted_not_clean = []
        num_answer_span_found = 0
        for idx, sample in tqdm.tqdm(
            dataset.iterrows(), desc=f"Generating dataset {split}"
        ):
            with open(
                ocr_dir
                / f'{sample["ucsf_document_id"]}_{sample["ucsf_document_page_no"]}.json'
            ) as f:
                ocr = json.load(f)

            if len(ocr["recognitionResults"]) > 1:
                raise ValueError(
                    "More than one recognition result found in OCR file. This is not supported."
                )

            recognition_result = ocr["recognitionResults"][0]
            image_width = recognition_result["width"]
            image_height = recognition_result["height"]
            image_size = (image_width, image_height)

            if msr_data is not None:
                assert sample["questionId"] == msr_data[idx]["questionId"]

            if not self.config.use_msr_ocr:
                filtered_words = []
                filtered_boxes = []
                filtered_ssl_boxes = []
                for line in recognition_result["lines"]:
                    cur_line_bboxes = []
                    for word_and_box in line["words"]:
                        word = word_and_box["text"].strip()
                        if word.startswith("http") or word == "":
                            continue

                        # bbox in this dataset is bbox = [x1,y1,x2,y2,x3,y3,x4,y4] for four corner points
                        x1, y1, x2, y2, x3, y3, x4, y4 = word_and_box["boundingBox"]
                        bbox_x1 = min([x1, x2, x3, x4])
                        bbox_x2 = max([x1, x2, x3, x4])
                        bbox_y1 = min([y1, y2, y3, y4])
                        bbox_y2 = max([y1, y2, y3, y4])
                        filtered_words.append(word.lower())
                        cur_line_bboxes.append(
                            _normalize_bbox(
                                [bbox_x1, bbox_y1, bbox_x2, bbox_y2], image_size
                            )
                        )

                    if len(cur_line_bboxes) > 0:
                        # add word box
                        filtered_boxes.extend(cur_line_bboxes)

                        # add line box
                        cur_line_bboxes = self.get_line_bbox(cur_line_bboxes)
                        filtered_ssl_boxes.extend(cur_line_bboxes)
            else:
                msr_sample = msr_data[idx]
                filtered_boxes = (
                    msr_sample["boxes"]
                    if "boxes" in msr_sample
                    else msr_sample["layout"]
                )
                filtered_words = [word.lower() for word in msr_sample["words"]]
                filtered_ssl_boxes = filtered_boxes

            assert len(filtered_ssl_boxes) == len(filtered_boxes) == len(filtered_words)
            if split in ["train", "val"]:
                answers = (
                    list(set([x.lower() for x in sample["answers"]]))
                    if "answers" in sample
                    else []
                )
                processed_answers, _ = find_answers_in_words(
                    filtered_words, answers, self.config.answers_extraction_method
                )
                answer_start_indices = [
                    ans["answer_start_index"] for ans in processed_answers
                ]
                answer_end_indices = [
                    ans["answer_end_index"] for ans in processed_answers
                ]
                gold_answers = [ans["gold_answer"] for ans in processed_answers]
                # extracted_answers = [ans["extracted_answer"] for ans in processed_answers] # not needed

                if split == "train":
                    if self.config.store_single_answer_per_example:
                        answer_start_indices = answer_start_indices[
                            :1
                        ]  # we only need one sampler per answer for training
                        answer_end_indices = answer_end_indices[:1]
                else:
                    answer_start_indices = answer_start_indices[
                        :1
                    ]  # we only need one sample per answer for eval
                    answer_end_indices = answer_end_indices[
                        :1
                    ]  # we only need one sample per answer for eval

                current_extracted_not_clean = []
                for start_word_id in answer_start_indices:
                    if start_word_id != -1:
                        num_answer_span_found += 1
                        break

                for start_word_id, end_word_id in zip(
                    answer_start_indices, answer_end_indices
                ):
                    if start_word_id != -1:
                        current_extracted_not_clean.append(
                            " ".join(filtered_words[start_word_id : end_word_id + 1])
                        )
                        break

                if len(current_extracted_not_clean) > 0:
                    all_extracted_not_clean.append(current_extracted_not_clean)
                    all_gold.append(gold_answers)
            else:
                answer_start_indices = [-1]
                answer_end_indices = [-1]
                gold_answers = [""]

            sample = {
                # image
                DataKeys.IMAGE_FILE_PATH: image_dir
                / sample["image"].replace("documents", "spdocvqa_images"),
                DataKeys.IMAGE_WIDTH: image_size[0],
                DataKeys.IMAGE_HEIGHT: image_size[1],
                # text
                DataKeys.WORDS: filtered_words,
                DataKeys.WORD_BBOXES: filtered_boxes,
                DataKeys.WORD_BBOXES_SEGMENT_LEVEL: filtered_ssl_boxes,
                # question/answer
                DataKeys.QUESTION_ID: sample["questionId"],
                DataKeys.QUESTIONS: sample["question"].lower(),
                DataKeys.GOLD_ANSWERS: gold_answers,
                DataKeys.ANSWER_START_INDICES: answer_start_indices,
                DataKeys.ANSWER_END_INDICES: answer_end_indices,
            }
            processed_dataset.append(sample)

        if split in ["train", "val"]:
            _, anls = anls_metric_str(
                predictions=all_extracted_not_clean, gold_labels=all_gold
            )
            total_questions_in_dataset = len(processed_dataset)
            logger.info(f"Preprocessed {filepath} dataset statistics:")
            logger.info(f"Extracted answers: {all_extracted_not_clean[:100]}")
            logger.info(f"Extracted gold answers: {all_gold[:100]}")
            logger.info(f"Ground truth ANLS: {anls}")
            logger.info(f"Total questions in dataset: {total_questions_in_dataset}")
            logger.info(f"Total answers found: {num_answer_span_found}")
            logger.info(
                f"Total answers not found: {total_questions_in_dataset - num_answer_span_found}"
            )
        return processed_dataset

    def _generate_examples(
        self,
        split: str,
        filepath: str,
    ):
        image_dir = Path(self.config.data_dir)
        ocr_dir = Path(self.config.data_dir) / "spdocvqa_ocr"
        with open(filepath) as f:
            dataset = pd.DataFrame(json.load(f)["data"])

        msr_data = None
        if self.config.use_msr_ocr:
            if split in ["train", "val"]:
                msr_file = (
                    Path(self.config.data_dir) / "msr" / f"docvqa_proc_{split}_t3_ocr"
                )
                msr_data = load_from_disk(str(msr_file))
            else:
                msr_file = Path(self.config.data_dir) / "msr" / "test_v1.0_msr.json"
                with open(msr_file, "r", encoding="utf-8") as read_file:
                    msr_data = json.load(read_file)

        all_gold = []
        all_extracted_not_clean = []
        num_answer_span_found = 0
        for idx, sample in dataset.iterrows():
            with open(
                ocr_dir
                / f'{sample["ucsf_document_id"]}_{sample["ucsf_document_page_no"]}.json'
            ) as f:
                ocr = json.load(f)

            if len(ocr["recognitionResults"]) > 1:
                raise ValueError(
                    "More than one recognition result found in OCR file. This is not supported."
                )

            recognition_result = ocr["recognitionResults"][0]
            image_width = recognition_result["width"]
            image_height = recognition_result["height"]
            image_size = (image_width, image_height)

            if msr_data is not None:
                assert sample["questionId"] == msr_data[idx]["questionId"]

            if not self.config.use_msr_ocr:
                filtered_words = []
                filtered_boxes = []
                filtered_ssl_boxes = []
                for line in recognition_result["lines"]:
                    cur_line_bboxes = []
                    for word_and_box in line["words"]:
                        word = word_and_box["text"].strip()
                        if word.startswith("http") or word == "":
                            continue

                        # bbox in this dataset is bbox = [x1,y1,x2,y2,x3,y3,x4,y4] for four corner points
                        x1, y1, x2, y2, x3, y3, x4, y4 = word_and_box["boundingBox"]
                        bbox_x1 = min([x1, x2, x3, x4])
                        bbox_x2 = max([x1, x2, x3, x4])
                        bbox_y1 = min([y1, y2, y3, y4])
                        bbox_y2 = max([y1, y2, y3, y4])
                        filtered_words.append(word.lower())
                        cur_line_bboxes.append(
                            _normalize_bbox(
                                [bbox_x1, bbox_y1, bbox_x2, bbox_y2], image_size
                            )
                        )

                    if len(cur_line_bboxes) > 0:
                        # add word box
                        filtered_boxes.extend(cur_line_bboxes)

                        # add line box
                        cur_line_bboxes = self.get_line_bbox(cur_line_bboxes)
                        filtered_ssl_boxes.extend(cur_line_bboxes)
            else:
                msr_sample = msr_data[idx]
                filtered_boxes = (
                    msr_sample["boxes"]
                    if "boxes" in msr_sample
                    else msr_sample["layout"]
                )
                filtered_words = [word.lower() for word in msr_sample["words"]]
                filtered_ssl_boxes = filtered_boxes

            assert len(filtered_ssl_boxes) == len(filtered_boxes) == len(filtered_words)
            if split in ["train", "val"]:
                answers = (
                    list(set([x.lower() for x in sample["answers"]]))
                    if "answers" in sample
                    else []
                )
                processed_answers, _ = find_answers_in_words(
                    filtered_words, answers, self.config.answers_extraction_method
                )
                answer_start_indices = [
                    ans["answer_start_index"] for ans in processed_answers
                ]
                answer_end_indices = [
                    ans["answer_end_index"] for ans in processed_answers
                ]
                gold_answers = [ans["gold_answer"] for ans in processed_answers]
                # extracted_answers = [ans["extracted_answer"] for ans in processed_answers] # not needed

                if split == "train":
                    if self.config.store_single_answer_per_example:
                        answer_start_indices = answer_start_indices[
                            :1
                        ]  # we only need one sampler per answer for training
                        answer_end_indices = answer_end_indices[:1]
                else:
                    answer_start_indices = answer_start_indices[
                        :1
                    ]  # we only need one sample per answer for eval
                    answer_end_indices = answer_end_indices[
                        :1
                    ]  # we only need one sample per answer for eval

                current_extracted_not_clean = []
                for start_word_id in answer_start_indices:
                    if start_word_id != -1:
                        num_answer_span_found += 1
                        break

                for start_word_id, end_word_id in zip(
                    answer_start_indices, answer_end_indices
                ):
                    if start_word_id != -1:
                        current_extracted_not_clean.append(
                            " ".join(filtered_words[start_word_id : end_word_id + 1])
                        )
                        break

                if len(current_extracted_not_clean) > 0:
                    all_extracted_not_clean.append(current_extracted_not_clean)
                    all_gold.append(gold_answers)
            else:
                answer_start_indices = [-1]
                answer_end_indices = [-1]
                gold_answers = [""]
            image = Image.open(
                image_dir / sample["image"].replace("documents", "spdocvqa_images")
            )
            assert image.size == (
                image_size[0],
                image_size[1],
            )
            yield str(uuid.uuid4()), {
                DataKeys.INDEX: idx,
                # image
                DataKeys.IMAGE: Image.open(
                    image_dir / sample["image"].replace("documents", "spdocvqa_images")
                ),
                DataKeys.IMAGE_WIDTH: image_size[0],
                DataKeys.IMAGE_HEIGHT: image_size[1],
                # text
                DataKeys.WORDS: filtered_words,
                DataKeys.WORD_BBOXES: filtered_boxes,
                DataKeys.WORD_BBOXES_SEGMENT_LEVEL: filtered_ssl_boxes,
                # question/answer
                DataKeys.QUESTION_ID: sample["questionId"],
                DataKeys.QUESTIONS: sample["question"].lower(),
                DataKeys.GOLD_ANSWERS: gold_answers,
                DataKeys.ANSWER_START_INDICES: answer_start_indices,
                DataKeys.ANSWER_END_INDICES: answer_end_indices,
            }
