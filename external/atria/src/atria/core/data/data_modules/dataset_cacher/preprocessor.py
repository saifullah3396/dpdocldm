import hashlib
import json
from typing import Any, Callable, Dict, List, Optional, OrderedDict, Union

from datasets import Features

from atria.core.constants import DataKeys

SUFFIX = "-%06d"
DEFAULT_ENCODE_DECODE_FORMAT = "mp"
IGNORED_HASH_KEYS = [
    f"{k}.{DEFAULT_ENCODE_DECODE_FORMAT}" for k in [DataKeys.IMAGE_FILE_PATH]
]


def convert_bytes_in_nested_dicts(obj):
    if isinstance(obj, bytes):
        return obj.decode("utf-8", errors="replace")
    elif isinstance(obj, dict):
        return {k: convert_bytes_in_nested_dicts(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_bytes_in_nested_dicts(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_bytes_in_nested_dicts(item) for item in obj)
    return obj


def generate_unique_identifier(sample):
    normalized_sample = convert_bytes_in_nested_dicts(
        {k: v for k, v in sample.items() if k not in IGNORED_HASH_KEYS}
    )
    json_string = json.dumps(normalized_sample, sort_keys=True)
    unique_identifier = hashlib.sha256(json_string.encode("utf-8")).hexdigest()
    if f"index.{DEFAULT_ENCODE_DECODE_FORMAT}" in sample:
        unique_identifier = "_".join(
            [str(sample[f"index.{DEFAULT_ENCODE_DECODE_FORMAT}"]), unique_identifier]
        )
    return unique_identifier


class DatasetPreprocessor:
    """
    A class to preprocess and encode samples for web datasets.
    """

    def __init__(
        self,
        features: Features,
        preprocess_data_transforms: Optional[
            Union[Callable, OrderedDict[str, Callable]]
        ] = None,
    ) -> None:
        """
        Initializes the DatasetPreprocessor with features and an optional data transform.

        Args:
            features (Features): The features schema for encoding examples.
            preprocess_data_transforms (Optional[Union[Callable[[Union[List[Dict[str, Any]], Dict[str, Any]]], Union[List[Dict[str, Any]], Dict[str, Any]]], List[Callable[[Union[List[Dict[str, Any]], Dict[str, Any]]], Union[List[Dict[str, Any]], Dict[str, Any]]]]]]):
                An optional callable or list of callables for preprocessing data.
        """
        self._features: Features = features
        self._preprocess_data_transforms: Optional[
            Union[Callable, OrderedDict[str, Callable]]
        ] = preprocess_data_transforms

        # Convert transforms into a list of transforms
        if self._preprocess_data_transforms is not None:
            if isinstance(self._preprocess_data_transforms, OrderedDict):
                self._preprocess_data_transforms = list(
                    self._preprocess_data_transforms.values()
                )
            if not isinstance(self._preprocess_data_transforms, list):
                self._preprocess_data_transforms = [self._preprocess_data_transforms]
            assert all(
                callable(tf) for tf in self._preprocess_data_transforms
            ), "All preprocess_data_transforms must be callable."

    def load(self) -> None:
        """
        Loads the preprocess data transform if it is not None.
        """
        if self._preprocess_data_transforms is not None:
            for tf in self._preprocess_data_transforms:
                if hasattr(tf, "load"):
                    tf.load()

    def __call__(
        self, sample: Union[List[Dict[str, Any]], Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Processes and encodes a sample or a list of samples.

        Args:
            sample (Union[List[Dict[str, Any]], Dict[str, Any]]): A sample or a list of samples to be processed.

        Returns:
            List[Dict[str, Any]]: A list of processed and encoded samples.
        """
        if sample is None:
            return []

        if not isinstance(sample, list):
            sample = [sample]

        # Preprocess the data if a transform is provided
        if self._preprocess_data_transforms is not None:
            # Important note: the preprocess_data_transform can return a list of samples or a single sample
            # This means if a list of transforms is provided the next transform will be applied on newly updated list
            # of samples if that is the case
            for tf in self._preprocess_data_transforms:
                sample = tf(sample)

        # Encode the features
        for s in sample:
            encoded_example: Dict[str, Any] = self._features.encode_example(s)
            s.update(encoded_example)
            for k in self._features.keys():
                s[f"{k}.{DEFAULT_ENCODE_DECODE_FORMAT}"] = s.pop(k)

        # Generate a unique key for each sample
        return [{"__key__": generate_unique_identifier(s), **s} for s in sample]
