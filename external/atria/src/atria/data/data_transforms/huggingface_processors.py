import inspect
import json
import os
from typing import Any, List, Mapping, Optional, Union

import torch
from atria.core.constants import DataKeys
from atria.core.data.data_modules.dataset_cacher.dataset_cacher import ATRIA_CACHE_DIR
from atria.core.data.data_transforms import DataTransform
from atria.core.utilities.common import _get_required_args
from atria.core.utilities.logging import get_logger
from transformers.utils.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

logger = get_logger(__name__)


class SequenceSanitizer(DataTransform):
    """
    This class is used to sanitize the tokenized sequences to match the original sequence elements.
    For example, if you have labels for words, you can map them to the tokens.
    [0, 1, 2] -> ['hello', 'world', '!']
    This is mapped to tokens like so
    [<padding_element>, 0, <padding_element>, 1,  <padding_element>, 2, <padding_element>, ...] ->
    [<cls-token>, '#hel', '#lo', '#wor', '#ld' '!', <pad-token>, ...]
    """

    def __init__(
        self,
        input_seq_key: str = DataKeys.WORD_LABELS,
        output_seq_key: str = "labels",
        apply_value_to_first_subword: bool = True,
        padding_value: int = -100,
    ):
        super().__init__(key=None)
        self._input_seq_key = input_seq_key
        self._output_seq_key = output_seq_key
        self._apply_value_to_first_subword = apply_value_to_first_subword
        self._padding_value = padding_value

    def _tokenize_sequence(self, samples_batch, sequence_batch):
        pass

        assert (
            DataKeys.WORD_IDS in samples_batch
        ), f"The data_key '{DataKeys.WORD_IDS}' is not present in the input: {list(samples_batch.keys())} "
        word_ids_batch = samples_batch[DataKeys.WORD_IDS]
        tokenized_sequence = [
            [self._padding_value for _ in word_ids_batch[batch_idx]]
            for batch_idx in range(len(word_ids_batch))
        ]
        for batch_idx in range(len(tokenized_sequence)):
            word_ids, sequence = word_ids_batch[batch_idx], sequence_batch[batch_idx]
            last_word_id = None
            for idx, word_id in enumerate(word_ids):
                if word_id == -100:
                    continue
                if (
                    last_word_id is not None
                    and last_word_id == word_id
                    and self._apply_value_to_first_subword
                ):
                    continue
                tokenized_sequence[batch_idx][idx] = sequence[word_id]
                last_word_id = word_id
        return torch.tensor(tokenized_sequence)

    def _apply_transform(
        self, samples: Union[Mapping[str, Any], List[Mapping[str, Any]]]
    ) -> Union[Mapping[str, Any], List[Mapping[str, Any]]]:
        # convert a single sample into samples list if required
        is_single_sample = False
        if not isinstance(samples, list):
            samples = [samples]
            is_single_sample = True

        # convert list of dicts to dict of lists
        samples_batch = {k: [s[k] for s in samples] for k in samples[0].keys()}

        # validate the required keys
        assert (
            self._input_seq_key in samples_batch
        ), f"The data_key '{self._input_seq_key}' is not present in the input: {list(samples_batch.keys())} "

        # now we attach the labels to the tokenized samples
        # check if labels is a sequence and if so we need to attach labels to each subword
        if isinstance(samples_batch[self._input_seq_key][0], (list, torch.Tensor)):
            samples_batch[self._output_seq_key] = self._tokenize_sequence(
                samples_batch, samples_batch[self._input_seq_key]
            )
        else:
            logger.warning(
                f"Sequence key '{self._input_seq_key}' is not a sequence. No need to sanitize the sequence."
            )

        # convert dict of lists to list of dicts
        tokenized_samples_list = [
            dict(zip(samples_batch, t)) for t in zip(*samples_batch.values())
        ]
        for k, v in samples_batch.items():
            assert len(tokenized_samples_list) == len(
                v
            ), f"Not all keys have the same length to create a list of sample dicts. {len(tokenized_samples_list)} =/= {len(v)}"

        if is_single_sample:
            return tokenized_samples_list[0]
        else:
            return tokenized_samples_list


class HuggingfaceProcessor(DataTransform):
    def __init__(
        self,
        model_name: str,
        init_kwargs: Optional[dict] = None,
        call_kwargs: Optional[dict] = None,
        overflow_sampling: str = "return_all",
        max_overflow_samples: int = 10,
        processor_input_key_map: Optional[dict] = None,
    ):
        super().__init__(key=None)

        self._model_name = model_name
        self._init_kwargs = init_kwargs if init_kwargs is not None else {}
        self._call_kwargs = call_kwargs if call_kwargs is not None else {}
        self._overflow_sampling = overflow_sampling
        self._max_overflow_samples = max_overflow_samples
        self._processor_input_key_map = (
            processor_input_key_map
            if processor_input_key_map is not None
            else {"text": "text"}
        )

        # how to return overflowing samples?
        assert self._overflow_sampling in [
            "return_all",
            "return_random_n",
            "no_overflow",
            "return_first_n",
        ], f"Overflow sampling strategy {self._overflow_sampling} is not supported."

        # initialize the tokenizer
        default_init_kwargs = {
            "cache_dir": os.path.join(ATRIA_CACHE_DIR, ".huggingface"),
            "local_files_only": False,
            "apply_ocr": False,
            "image_mean": IMAGENET_DEFAULT_MEAN,
            "image_std": IMAGENET_DEFAULT_STD,
            "add_prefix_space": True,
            "do_lower_case": True,
            "do_normalize": False,
            "do_resize": False,
            "do_rescale": False,
        }
        # setup the default call kwargs
        default_call_kwargs = dict(
            add_special_tokens=True,
            padding="max_length",
            truncation=True,
            max_length=512,
            stride=0,
            pad_to_multiple_of=8,
            is_split_into_words=True,
            return_overflowing_tokens=self._overflow_sampling
            != "no_overflow",  # set some arguments that we need to stay fixed for our case
            return_token_type_ids=None,
            return_attention_mask=None,
            return_special_tokens_mask=False,
            return_offsets_mapping=False,
            return_length=False,
            return_tensors="pt",
            verbose=True,
        )
        self._init_kwargs = {**default_init_kwargs, **self._init_kwargs}
        self._call_kwargs = {**default_call_kwargs, **self._call_kwargs}

        # initialize the default input mapping
        if len(self._processor_input_key_map) == 0:
            logger.info(
                "Default input mapping created: {}".format(
                    self._processor_input_key_map
                )
            )
        self._processor = None

    def prepare_tokenizer_inputs(self, samples: Mapping[str, Any]) -> Mapping[str, Any]:
        # validate the processor input key map
        possible_tokenizer_args = inspect.signature(self._processor.__call__).parameters
        required_tokenize_args = _get_required_args(self._processor.__call__)
        tokenizer_inputs = {}
        for (
            target_key_in_tokenizer,
            key_in_sample,
        ) in self._processor_input_key_map.items():
            assert key_in_sample in samples, (
                f"Mapping '{target_key_in_tokenizer}' (tokenizer_input_key) -> '{key_in_sample}' (sample_key) in 'processor_input_key_map' is invalid. "
                f"Make sure the key '{key_in_sample}' (sample_key) is present in the sample: {list(samples.keys())} "
            )
            assert target_key_in_tokenizer in possible_tokenizer_args, (
                f"Mapping '{target_key_in_tokenizer}' (tokenizer_input_key) -> '{key_in_sample}' (sample_key) in 'processor_input_key_map' is invalid. "
                f"Make sure the key '{target_key_in_tokenizer}' (tokenizer_input_key) is accepted by the processor. "
                f"Valid arguments are: {list(possible_tokenizer_args.keys())}"
            )
            tokenizer_inputs[target_key_in_tokenizer] = samples[key_in_sample]

        for required_arg in required_tokenize_args:
            assert required_arg in tokenizer_inputs, (
                f"Required argument '{required_arg}' is missing in the processor input. "
                f"Make sure the key '{required_arg}' is present in the mapping:\n{json.dumps(self._processor_input_key_map, indent=4)} "
            )
        return tokenizer_inputs

    def _tokenize_samples(self, samples_batch: Mapping[str, Any]) -> Mapping[str, Any]:
        possible_args = inspect.signature(self._processor.__call__).parameters
        for key in list(self._call_kwargs.keys()):
            if key not in possible_args:
                logger.warning(
                    f"Invalid keyword argument '{key}' found in call_kwargs for {self.__class__.__name__}. Skipping it."
                )
                self._call_kwargs.pop(key)

        tokenized_samples_batch = self._processor(
            **self.prepare_tokenizer_inputs(samples_batch), **self._call_kwargs
        )

        # add word ids
        tokenized_samples_batch[DataKeys.WORD_IDS] = torch.tensor(
            [
                [-100 if x is None else x for x in tokenized_samples_batch.word_ids(i)]
                for i in range(len(tokenized_samples_batch[DataKeys.TOKEN_IDS]))
            ]
        )
        tokenized_samples_batch[DataKeys.SEQUENCE_IDS] = torch.tensor(
            [
                [
                    -100 if x is None else x
                    for x in tokenized_samples_batch.sequence_ids(i)
                ]
                for i in range(len(tokenized_samples_batch[DataKeys.TOKEN_IDS]))
            ]
        )

        return tokenized_samples_batch

    def _add_extra_keys_to_overflowing_samples(
        self, tokenized_samples_batch: dict, original_samples_batch: dict
    ):
        overflow_keys_added = set(original_samples_batch.keys()).difference(
            set(tokenized_samples_batch.keys())
        )
        if DataKeys.OVERFLOW_MAPPING in tokenized_samples_batch:
            # post process here we repeat all additional ids if they are matched to the same original file
            # for example if we have 2 overflowed samples from the same original sample we need to repeat the image file path
            overflow_indices = tokenized_samples_batch[DataKeys.OVERFLOW_MAPPING]
            overflowed_data = {
                k: [] for k in overflow_keys_added if k in original_samples_batch
            }
            for batch_index in range(len(tokenized_samples_batch[DataKeys.TOKEN_IDS])):
                org_batch_index = overflow_indices[batch_index]
                for k in overflowed_data.keys():
                    if k in original_samples_batch:
                        overflowed_data[k].append(
                            original_samples_batch[k][org_batch_index]
                        )
            # add new overflowed samples to update the batch, this results in a variable batch size
            for k in overflowed_data.keys():
                tokenized_samples_batch[k] = overflowed_data[k]
        else:
            for k in overflow_keys_added:
                if k in original_samples_batch:
                    tokenized_samples_batch[k] = original_samples_batch[k]
        return tokenized_samples_batch

    def _process_sample_overflow(self, tokenized_samples_batch):
        if (
            self._overflow_sampling in ["no_overflow", "return_all"]
            or DataKeys.OVERFLOW_MAPPING not in tokenized_samples_batch
        ):
            return tokenized_samples_batch

        # take the overflow indices
        overflow_indices = tokenized_samples_batch[DataKeys.OVERFLOW_MAPPING]

        filter_indices = []
        if self._overflow_sampling == "return_first_n":
            # in this case, we generate indices for the first n overflow indices for every sample
            for i in range(torch.max(overflow_indices) + 1):
                filter_indices += torch.where(overflow_indices == i)[0][
                    : self._max_overflow_samples
                ].tolist()
        elif self._overflow_sampling == "return_random_n":
            # in this case, we generate indices for the first n overflow indices for every sample
            for i in range(torch.max(overflow_indices) + 1):
                per_sample_indices = torch.where(overflow_indices == i)[0]
                per_sample_indices = per_sample_indices[
                    torch.randperm(per_sample_indices.nelement())
                ]
                filter_indices += per_sample_indices[
                    : self._max_overflow_samples
                ].tolist()
        else:
            raise ValueError(
                f"Overflow sampling strategy {self._overflow_sampling} is not supported."
            )

        # filter the tokenized samples to only keep the first n overflow indices for every sample
        tokenized_samples_batch = {
            k: (
                v[filter_indices]
                if isinstance(v, torch.Tensor)
                else [v[i] for i in filter_indices]
            )
            for k, v in tokenized_samples_batch.items()
        }

        return tokenized_samples_batch

    # def _sanitize_labels(self, tokenized_samples):
    #     for sample in tokenized_samples:
    #         # sanity check for labels and word ids. The total number of labels must be equal to the number of words as
    #         # we by default only assign label to the first word but there is bug in transformers which messes up for some
    #         # text and assigns multiple subwords the labels resulting in a mismatch

    #         def generate_word_ids_to_index_map(word_ids):
    #             # find all indices
    #             return {
    #                 idx: word_ids[idx].item()
    #                 for idx in range(1, len(word_ids))  # (1, sequence_length)
    #                 if word_ids[idx] != -100 and word_ids[idx] != word_ids[idx - 1]
    #             }

    #         word_ids_to_index_map = generate_word_ids_to_index_map(
    #             sample[DataKeys.WORD_IDS]
    #         )
    #         all_word_ids = [x.item() for x in sample[DataKeys.WORD_IDS] if x != -100]
    #         total_words = max(all_word_ids) - min(all_word_ids) + 1
    #         for idx in range(len(sample["labels"])):
    #             if idx in list(word_ids_to_index_map.keys()):
    #                 sample["labels"][idx] = sample[DataKeys.WORD_LABELS][
    #                     word_ids_to_index_map[idx]
    #                 ]
    #             else:
    #                 sample["labels"][idx] = -100
    #         assert len([x for x in sample["labels"] if x != -100]) == total_words

    def _apply_transform(
        self, samples: Union[Mapping[str, Any], List[Mapping[str, Any]]]
    ) -> Union[Mapping[str, Any], List[Mapping[str, Any]]]:
        if self._processor is None:
            from transformers import AutoProcessor

            self._processor = AutoProcessor.from_pretrained(
                self._model_name, **self._init_kwargs
            )

        # convert a single sample into samples list if required
        if not isinstance(samples, list):
            samples = [samples]

        # convert list of dicts to dict of lists
        samples_batch = {k: [s[k] for s in samples] for k in samples[0].keys()}

        # tokenize the samples
        tokenized_samples_batch = self._tokenize_samples(samples_batch)

        # add extra overflowing keys to the samples
        tokenized_samples_batch = self._add_extra_keys_to_overflowing_samples(
            tokenized_samples_batch, samples_batch
        )

        # see which samples to keep from the overflowed samples
        tokenized_samples_batch = self._process_sample_overflow(tokenized_samples_batch)

        # convert dict of lists to list of dicts
        tokenized_samples_list = [
            dict(zip(tokenized_samples_batch, t))
            for t in zip(*tokenized_samples_batch.values())
        ]

        for k, v in tokenized_samples_batch.items():
            assert len(tokenized_samples_list) == len(
                v
            ), f"Not all keys have the same length to create a list of sample dicts. {len(tokenized_samples_list)} =/= {len(v)}"

        last_overflow_sample_id = None
        for sample in tokenized_samples_list:
            if (
                last_overflow_sample_id is None
                or last_overflow_sample_id != sample["overflow_to_sample_mapping"]
            ):
                sample_id = 0
            sample["__key__"] = sample["__key__"] + f"_{sample_id}"
            sample_id += 1
            last_overflow_sample_id = sample["overflow_to_sample_mapping"]

        # tokenized list can be of variable lenght compared to the original samples
        return tokenized_samples_list


class QuestionAnsweringHuggingfaceProcessor(HuggingfaceProcessor):
    def _apply_transform(
        self, samples: Union[Mapping[str, Any], List[Mapping[str, Any]]]
    ) -> Union[Mapping[str, Any], List[Mapping[str, Any]]]:
        # convert a single sample into samples list if required
        if not isinstance(samples, list):
            samples = [samples]

        # convert list of dicts to dict of lists
        samples_batch = {k: [s[k] for s in samples] for k in samples[0].keys()}

        # tokenize the samples
        tokenized_samples_batch = self._tokenize_samples(samples_batch)

        # add extra overflowing keys to the samples
        tokenized_samples_batch = self._add_extra_keys_to_overflowing_samples(
            tokenized_samples_batch, samples_batch
        )

        # see which samples to keep from the overflowed samples
        tokenized_samples_batch = self._process_sample_overflow(tokenized_samples_batch)

        # convert dict of lists to list of dicts
        tokenized_samples_list = [
            dict(zip(tokenized_samples_batch, t))
            for t in zip(*tokenized_samples_batch.values())
        ]

        for k, v in tokenized_samples_batch.items():
            assert len(tokenized_samples_list) == len(
                v
            ), f"Not all keys have the same length to create a list of sample dicts. {len(tokenized_samples_list)} =/= {len(v)}"

        # tokenized list can be of variable lenght compared to the original samples
        return tokenized_samples_list


class AnswerWordIndicesToAnswerTokenIndices(DataTransform):
    def __init__(
        self,
        answer_start_indices_key: str = DataKeys.ANSWER_START_INDICES,
        answer_end_indices_key: str = DataKeys.ANSWER_END_INDICES,
        remove_no_answer_samples: bool = False,
    ):
        super().__init__(key=None)
        self.answer_start_indices_key = answer_start_indices_key
        self.answer_end_indices_key = answer_end_indices_key
        self.remove_no_answer_samples = remove_no_answer_samples

    def _get_subword_start_end(
        self, word_start, word_end, subword_idx2word_idx, sequence_ids
    ):
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

    def _apply_transform(
        self, samples: Union[Mapping[str, Any], List[Mapping[str, Any]]]
    ) -> Union[Mapping[str, Any], List[Mapping[str, Any]]]:
        if not isinstance(samples, list):
            samples = [samples]

        # validate the required keys
        assert (
            self.answer_start_indices_key in samples[0].keys()
        ), f"The data_key '{self.answer_start_indices_key}' is not present in the input: {list(samples[0].keys())} "
        assert (
            self.answer_end_indices_key in samples[0].keys()
        ), f"The data_key '{self.answer_end_indices_key}' is not present in the input: {list(samples[0].keys())} "

        updated_samples = []
        for sample in samples:
            for start_word_id, end_word_id in zip(
                sample[self.answer_start_indices_key],
                sample[self.answer_end_indices_key],
            ):
                if start_word_id == -1:
                    sample[DataKeys.START_TOKEN_IDX] = 0  # CLS token index
                    sample[DataKeys.END_TOKEN_IDX] = 0  # CLS token index
                    if not self.remove_no_answer_samples:
                        updated_samples.append(sample)
                else:
                    (
                        sample[DataKeys.START_TOKEN_IDX],
                        sample[DataKeys.END_TOKEN_IDX],
                        _,
                    ) = self._get_subword_start_end(
                        start_word_id,
                        end_word_id,
                        sample[DataKeys.WORD_IDS],
                        sample[DataKeys.SEQUENCE_IDS],
                    )
                    if sample[DataKeys.START_TOKEN_IDX] == -1:
                        sample[DataKeys.START_TOKEN_IDX] = 0  # CLS token index
                        sample[DataKeys.END_TOKEN_IDX] = 0  # CLS token index
                    if sample[DataKeys.END_TOKEN_IDX] == -1:
                        sample[DataKeys.END_TOKEN_IDX] = (
                            len(sample[DataKeys.TOKEN_IDS]) - 1
                        )
                    if sample[DataKeys.START_TOKEN_IDX] == -1:
                        if not self.remove_no_answer_samples:
                            updated_samples.append(sample)
                    else:
                        updated_samples.append(sample)

                if (
                    DataKeys.END_TOKEN_IDX in updated_samples[-1]
                    and DataKeys.START_TOKEN_IDX in updated_samples[-1]
                ):
                    assert (
                        updated_samples[-1][DataKeys.END_TOKEN_IDX]
                        >= updated_samples[-1][DataKeys.START_TOKEN_IDX]
                    ), (
                        "End token index is less than start token index. "
                        "Something is wrong in the conversion from answer word indices to answer token indices."
                    )
        return updated_samples
