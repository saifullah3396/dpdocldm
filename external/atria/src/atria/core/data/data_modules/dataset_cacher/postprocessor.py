from typing import Any, Dict

from datasets import Features

SUFFIX = "-%06d"
DEFAULT_ENCODE_DECODE_FORMAT = "mp"


class DatasetPostprocessor:
    """
    A class to post-process samples from a WebDataset.
    """

    def __init__(self, features: Features) -> None:
        """
        Initializes the DatasetPostprocessor with the given features.

        Args:
            features (Features): The features schema to decode examples.
        """
        self._features = features

    def __call__(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processes a sample from the WebDataset.

        Args:
            sample (Dict[str, Any]): The sample to process.

        Returns:
            Dict[str, Any]: The processed sample.
        """
        # filter and decode the sample
        filtered_sample = {
            [k for k in k.split(".") if k != ""][0]: v for k, v in sample.items()
        }
        filtered_sample.update(
            self._features.decode_example(
                {k: v for k, v in filtered_sample.items() if k in self._features}
            )
        )

        return filtered_sample
