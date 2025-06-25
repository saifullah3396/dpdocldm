import random
from typing import List, Optional, Tuple, Union

from numpy.typing import ArrayLike
from PIL import Image as PILImage

from atria.core.data.data_transforms import DataTransform

from .utilities import clipped_zoom, disk


class PilGaussianBlur(DataTransform):
    """
    Applies gaussian blur to a PIL image.
    """

    def __init__(
        self,
        sigma: Tuple[float, float] = (0.1, 2.0),
        key: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(key=key)
        self.sigma = sigma

    def _apply_transform(self, image: PILImage) -> PILImage:
        from PIL import ImageFilter

        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return image.filter(ImageFilter.GaussianBlur(radius=sigma))


class NumpyGaussianBlur(DataTransform):
    """
    Applies gaussian blur to a NumPy image.
    """

    def __init__(self, magnitude: float, key: Optional[Union[str, List[str]]] = None):
        super().__init__(key=key)
        self.magnitude = magnitude

    def _apply_transform(self, image: ArrayLike) -> ArrayLike:
        import scipy.ndimage as ndi

        return ndi.gaussian_filter(image, self.magnitude)


class NumpyBinaryBlur(DataTransform):
    """
    Applies binary blur to a NumPy image.
    """

    def __init__(self, sigma: float, key: Optional[Union[str, List[str]]] = None):
        super().__init__(key=key)
        self.sigma = sigma

    def _apply_transform(self, image: ArrayLike) -> ArrayLike:
        import ocrodeg

        return ocrodeg.binary_blur(image, sigma=self.sigma)


class NumpyNoisyBinaryBlur(DataTransform):
    """
    Applies noisey binary blur to a NumPy image.
    """

    def __init__(
        self,
        sigma: float,
        noise: float,
        key: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(key=key)
        self.sigma = sigma
        self.noise = noise

    def _apply_transform(self, image: ArrayLike) -> ArrayLike:
        import ocrodeg

        return ocrodeg.binary_blur(image, sigma=self.sigma, noise=self.noise)


class NumpyDefocusBlur(DataTransform):
    """
    Applies defocus blur to a NumPy image.
    """

    def __init__(
        self,
        radius: float,
        alias_blur: float = 0.1,
        key: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(key=key)
        self.radius = radius
        self.alias_blur = alias_blur

    def _apply_transform(self, image: ArrayLike) -> ArrayLike:
        import numpy as np

        kernel = disk(radius=self.radius, alias_blur=self.alias_blur)
        return np.clip(cv2.filter2D(image, -1, kernel), 0, 1)


class NumpyMotionBlur(DataTransform):
    """
    Applies motion blur to a NumPy image.
    """

    def __init__(self, size: float, key: Optional[Union[str, List[str]]] = None):
        super().__init__(key=key)
        self.size = size

    def _apply_transform(self, image: ArrayLike) -> ArrayLike:
        import cv2
        import numpy as np

        kernel_motion_blur = np.zeros((self.size, self.size))
        kernel_motion_blur[int((self.size - 1) / 2), :] = np.ones(
            self.size, dtype=np.float32
        )
        kernel_motion_blur = cv2.warpAffine(
            kernel_motion_blur,
            cv2.getRotationMatrix2D(
                (self.size / 2 - 0.5, self.size / 2 - 0.5),
                np.random.uniform(-45, 45),
                1.0,
            ),
            (self.size, self.size),
        )
        kernel_motion_blur = kernel_motion_blur * (1.0 / np.sum(kernel_motion_blur))
        return cv2.filter2D(image, -1, kernel_motion_blur)


class NumpyZoomBlur(DataTransform):
    """
    Applies zoom blur to a NumPy image.
    """

    def __init__(
        self,
        zoom_factor_start: float,
        zoom_factor_end: float,
        zoom_factor_step: float,
        key: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(key=key)
        self.zoom_factor_start = zoom_factor_start
        self.zoom_factor_end = zoom_factor_end
        self.zoom_factor_step = zoom_factor_step

    def _apply_transform(self, image: ArrayLike) -> ArrayLike:
        out = np.zeros_like(image)
        zoom_factor_range = np.arange(
            self.zoom_factor_start, self.zoom_factor_end, self.zoom_factor_step
        )
        for zoom_factor in zoom_factor_range:
            out += clipped_zoom(image, zoom_factor)
        return np.clip((image + out) / (len(zoom_factor_range) + 1), 0, 1)
