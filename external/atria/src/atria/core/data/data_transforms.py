from abc import abstractmethod
from typing import Any, Callable, List, Mapping, Optional, OrderedDict, Union


class DataTransform:
    """
    Base class for applying transformations on given keys for dictionary outputs.

    Args:
        key (Optional[Union[str, List[str]]]): Data key(s) to apply this transformation to.
    """

    def __init__(self, key: Optional[Union[str, List[str]]] = None):
        self.key = key

    @abstractmethod
    def _apply_transform(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Apply the transformation to the sample.

        Args:
            sample (Mapping[str, Any]): The input sample to transform.

        Returns:
            Mapping[str, Any]: The transformed sample.
        """
        raise NotImplementedError(
            "Child class must implement the _apply_transform method."
        )

    def _apply_transform_preprocess(
        self, sample: Union[Any, Mapping[str, Any]]
    ) -> Mapping[str, Any]:
        """
        Preprocess and apply the transformation to the given key in the sample.

        Args:
            sample (Union[Any, Mapping[str, Any]]): The input sample to preprocess and transform.

        Returns:
            Mapping[str, Any]: The transformed sample.
        """
        if self.key is not None:
            keys = self.key if isinstance(self.key, list) else [self.key]
            for key in keys:
                sample[key] = self._apply_transform(sample[key])
        else:
            sample = self._apply_transform(sample)
        return sample

    def __call__(
        self, sample: Union[Any, Mapping[str, Any], List[Mapping[str, Any]]]
    ) -> Union[Any, Mapping[str, Any], List[Mapping[str, Any]]]:
        """
        Apply the transformation to the sample(s).

        Args:
            sample (Union[Any, Mapping[str, Any], List[Mapping[str, Any]]]): The input sample(s) to transform.

        Returns:
            Union[Any, Mapping[str, Any], List[Mapping[str, Any]]]: The transformed sample(s).
        """
        if isinstance(sample, list):
            return [self._apply_transform_preprocess(s) for s in sample]
        return self._apply_transform_preprocess(sample)

    def __repr__(self) -> str:
        return super().__repr__()


class WrappedDataTransform(DataTransform):
    """
    Wrapper class for applying a callable transformation on given keys for dictionary outputs.

    Args:
        transform (Optional[Callable]): The transformation function to apply.
    """

    def __init__(
        self,
        key: Union[str, List[str]] = None,
        transform: Union[Callable, List[Callable]] = None,
    ):
        super().__init__(key)
        if transform is None:
            raise ValueError("Wrapped transform cannot be None.")

        from torchvision import transforms

        self.transform = transforms.Compose(transform)

    def _apply_transform(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        """
        Apply the wrapped transformation to the sample.

        Args:
            sample (Mapping[str, Any]): The input sample to transform.

        Returns:
            Mapping[str, Any]: The transformed sample.
        """
        return self.transform(sample)

    def __repr__(self) -> str:
        """
        Return a string representation of the wrapped transformation.

        Returns:
            str: The string representation of the transformation.
        """
        if self.key is not None:
            return f"{self.key}: {str(self.transform)}"
        return str(self.transform)


class DataTransformsDict:
    def __init__(
        self,
        train: Optional[Union[DataTransform, OrderedDict[str, DataTransform]]] = None,
        evaluation: Optional[
            Union[DataTransform, OrderedDict[str, DataTransform]]
        ] = None,
    ):
        self.train = train
        self.evaluation = evaluation


class SampleKeyRemap(DataTransform):
    def __init__(self, output_key_map: dict):
        super().__init__()
        self.output_key_map = output_key_map

    def _apply_transform(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        for target_key, data_key in self.output_key_map.items():
            if data_key in sample:
                sample[target_key] = sample.pop(data_key)
        return sample
