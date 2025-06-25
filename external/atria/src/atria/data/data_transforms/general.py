from typing import List, Optional, Union

from numpy.typing import ArrayLike
from PIL.Image import Image

from atria.core.data.data_transforms import DataTransform


class NumpyToTensor(DataTransform):
    """
    Converts input to tensor
    """

    def __init__(self, key: Optional[Union[str, List[str]]] = None):
        super().__init__(key=key)
        self._tf = self._initialize_transform()

    def _initialize_transform(self):
        import torch
        from torchvision import transforms

        # generate transformations list
        tf = []

        # convert images to tensor
        tf.append(transforms.ToTensor())

        # change dtype to float
        tf.append(transforms.ConvertImageDtype(torch.float))

        # generate torch transformation
        return transforms.Compose(tf)

    def _apply_transform(self, sample: ArrayLike) -> "torch.Tensor":
        return self._tf(sample)


class NumpyBrightness(DataTransform):
    """
    Increases/decreases brightness of a numpy image based on the beta parameter.
    """

    def __init__(self, beta: float, key: Optional[Union[str, List[str]]] = None):
        super().__init__(key=key)
        self.beta = beta

    def _apply_transform(self, image: ArrayLike) -> ArrayLike:
        import numpy as np

        return np.clip(image + self.beta, 0, 1)


class NumpyContrast(DataTransform):
    """
    Increases/decreases contrast of a numpy image based on the alpha parameter.
    """

    def __init__(self, alpha: float, key: Optional[Union[str, List[str]]] = None):
        super().__init__(key=key)
        self.alpha = alpha

    def _apply_transform(self, image: ArrayLike) -> ArrayLike:
        import numpy as np

        channel_means = np.mean(image, axis=(0, 1))
        return np.clip((image - channel_means) * self.alpha + channel_means, 0, 1)


class PilGrayToRgb(DataTransform):
    """
    Converts a rgb image to grayscale.
    """

    def _apply_transform(self, image: Image) -> Image:
        return image.convert("RGB")


class PilRgbToGray(DataTransform):
    """
    Converts a rgb image to grayscale.
    """

    def _apply_transform(self, image: Image) -> Image:
        return image.convert("L")


class TensorGrayToRgb(DataTransform):
    """
    Converts a gray-scale torch image to rgb image.
    """

    def _apply_transform(self, image: "torch.Tensor") -> "torch.Tensor":
        if len(image.shape) == 2 or image.shape[0] == 1:
            return image.repeat(3, 1, 1)
        else:
            return image


class TensorRgbToBgr(DataTransform):
    """
    Converts a torch tensor from RGB to BGR
    """

    def _apply_transform(self, image: "torch.Tensor") -> "torch.Tensor":
        return image.permute(2, 1, 0)
