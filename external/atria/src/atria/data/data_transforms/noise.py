from typing import List, Optional, Union

from numpy.typing import ArrayLike

from atria.core.data.data_transforms import DataTransform


class TensorGaussianNoiseRgb(DataTransform):
    """
    Applies RGB Gaussian noise to a numpy image.
    """

    def __init__(
        self, magnitude: float = 0.1, key: Optional[Union[str, List[str]]] = None
    ):
        super().__init__(key=key)
        self.magnitude = magnitude

    def _apply_transform(self, image: ArrayLike) -> ArrayLike:
        import numpy as np
        import torch

        return torch.clip(
            image + np.random.normal(size=image.shape, scale=self.magnitude), 0, 1
        )


class NumpyShotNoiseRgb(DataTransform):
    """
    Applies shot noise to a numpy image.
    """

    def __init__(
        self, magnitude: float = 0.1, key: Optional[Union[str, List[str]]] = None
    ):
        super().__init__(key=key)
        self.magnitude = magnitude

    def _apply_transform(self, image: ArrayLike) -> ArrayLike:
        import cv2
        import numpy as np

        if len(image.shape) == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

        return np.clip(
            np.random.poisson(image * self.magnitude) / float(self.magnitude), 0, 1
        )


class NumpyFibrousNoise(DataTransform):
    """
    Applies fibrous noise to a numpy image.
    """

    def __init__(
        self,
        blur: float = 1.0,
        blotches: float = 5e-5,
        key: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(key=key)
        self.blur = blur
        self.blotches = blotches

    def _apply_transform(self, image: ArrayLike) -> ArrayLike:
        import ocrodeg

        return ocrodeg.printlike_fibrous(image, blur=self.blur, blotches=self.blotches)


class NumpyMultiscaleNoise(DataTransform):
    """
    Applies multiscale noise to a numpy image.
    """

    def __init__(
        self,
        blur: float = 1.0,
        blotches: float = 5e-5,
        key: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(key=key)
        self.blur = blur
        self.blotches = blotches

    def _apply_transform(self, image: ArrayLike) -> ArrayLike:
        import ocrodeg

        return ocrodeg.printlike_multiscale(
            image, blur=self.blur, blotches=self.blotches
        )
