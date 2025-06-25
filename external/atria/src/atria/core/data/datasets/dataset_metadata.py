from collections import OrderedDict
from typing import Dict, List, Optional, Union

import datasets
from atria.core.utilities.logging import get_logger

logger = get_logger(__name__)


class DatasetMetadata:
    """
    A class to represent metadata for a dataset.

    Attributes:
        labels (Optional[List[str]]): A list of labels associated with the dataset.
    """

    def __init__(
        self,
        labels: Optional[List[str]] = None,
    ):
        """
        Initialize the DatasetMetadata instance.

        Args:
            labels (Optional[List[str]]): A list of labels associated with the dataset.
        """
        self.labels = labels

    @classmethod
    def from_info(cls, dataset_info: "datasets.DatasetInfo") -> "DatasetMetadata":
        """
        Create a DatasetMetadata instance from DatasetInfo.

        Args:
            cls: The class type.
            dataset_info (DatasetInfo): The dataset information object.

        Returns:
            DatasetMetadata: An instance of DatasetMetadata with labels extracted from dataset_info.
        """

        from atria.core.data.utilities import _get_labels_from_features

        return DatasetMetadata(
            labels=_get_labels_from_features(dataset_info.features),
        )

    def _generate_random_classification_label_colors(self) -> Dict[str, str]:
        """
        Generate a random color map for classification labels.

        Returns:
            Dict[str, str]: A dictionary mapping each label to a random color in hex format.
        """
        import matplotlib as mpl
        from atria.core.data.utilities import _get_labels_color_map

        cmap = _get_labels_color_map(len(self.labels))
        return {
            label: mpl.colors.rgb2hex(cmap(idx), keep_alpha=False)
            for idx, label in enumerate(self.labels)
        }

    def _generate_random_ner_label_colors(self) -> OrderedDict[str, str]:
        """
        Generate a random color map for Named Entity Recognition (NER) labels.

        Returns:
            OrderedDict[str, str]: An ordered dictionary mapping each label to a random color in hex format.
        """
        import matplotlib as mpl
        from atria.core.data.utilities import _get_labels_color_map

        total_entities = {label.split("-")[-1] for label in self.labels}
        total_entities = list(total_entities)

        cmap = _get_labels_color_map(len(total_entities))
        labels_colors = OrderedDict()
        for label in self.labels:
            base_label = label.split("-")[-1]
            color_index = total_entities.index(base_label)
            color = cmap(color_index)

            if label.startswith("B"):
                color = [c * 0.8 for c in color]
            labels_colors[label] = mpl.colors.rgb2hex(color, keep_alpha=False)
        return labels_colors

    def generate_random_label_colors(
        self,
    ) -> Union[Dict[str, str], OrderedDict[str, str]]:
        """
        Generate a random color map for labels, either for classification or NER.

        Returns:
            Union[Dict[str, str], OrderedDict[str, str]]: A dictionary or ordered dictionary mapping each label to a random color in hex format.
        """
        if any(
            label.startswith("B-") or label.startswith("I-") for label in self.labels
        ):
            return self._generate_random_ner_label_colors()
        else:
            return self._generate_random_classification_label_colors()

    def state_dict(self):
        """
        Get the state dictionary for the dataset metadata.

        Returns:
            Dict: A dictionary containing the state of the dataset metadata.
        """
        return {"labels": self.labels}

    def load_state_dict(self, state_dict):
        """
        Load the state dictionary for the dataset metadata.

        Returns:
            None
        """
        self.labels = state_dict["labels"]
