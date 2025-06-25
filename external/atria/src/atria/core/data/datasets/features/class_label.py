from dataclasses import dataclass

from datasets.features import ClassLabel

CLASS_IGNORE_LABELS = {-1, -100}


@dataclass
class ClassLabel(ClassLabel):
    """
    ClassLabel is an extension of the Huggingface ClassLabel to support ignore labels other than -1, such as -100 used by PyTorch tokenizers.
    """

    def _strval2int(self, value: str) -> int:
        """
        Convert a string value to its corresponding integer class label.

        Args:
            value (str): The string representation of the class label.

        Returns:
            int: The integer representation of the class label.

        Raises:
            ValueError: If the string cannot be converted to a valid class label.
        """
        failed_parse = False
        value = str(value)
        # first attempt - raw string value
        int_value = self._str2int.get(value)
        if int_value is None:
            # second attempt - strip whitespace
            int_value = self._str2int.get(value.strip())
            if int_value is None:
                # third attempt - convert str to int
                try:
                    int_value = int(value)
                except ValueError:
                    failed_parse = True
                else:
                    if (
                        int_value not in CLASS_IGNORE_LABELS
                        or int_value >= self.num_classes
                    ):
                        failed_parse = True
        if failed_parse:
            raise ValueError(f"Invalid string class label {value}")
        return int_value

    def encode_example(self, example_data):
        """
        Encode the given example data into its corresponding integer class label.
        Args:
            example_data: The data to be encoded, which can be a string or an integer.
        Returns:
            int: The encoded integer class label.
        Raises:
            ValueError: If the number of classes is not defined, if the example data is an invalid class label, or if the example data exceeds the number of configured classes.
        """
        if self.num_classes is None:
            raise ValueError(
                "Trying to use ClassLabel feature with undefined number of class. "
                "Please set ClassLabel.names or num_classes."
            )

        # If a string is given, convert to associated integer
        if isinstance(example_data, str):
            example_data = self.str2int(example_data)

        # Allowing -1 to mean no label.
        if example_data < 0 and example_data not in CLASS_IGNORE_LABELS:
            raise ValueError(
                f"Class label ignore values can only be from {CLASS_IGNORE_LABELS}"
            )

        if example_data > self.num_classes:
            raise ValueError(
                f"Class label {example_data:d} greater than configured num_classes {self.num_classes}"
            )
        return example_data
