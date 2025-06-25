import typing
from typing import Any, List, Mapping, Optional, Tuple, Union

from atria.core.constants import DataKeys
from atria.core.data.data_transforms import DataTransform
from numpy.typing import ArrayLike


class NumpyTranslation(DataTransform):
    """
    Applies translation to a NumPy image based on the magnitude in x-y.
    """

    def __init__(
        self,
        magnitude: Tuple[float],
        key: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(key=key)
        self.magnitude = magnitude

    def _apply_transform(self, image: ArrayLike) -> ArrayLike:
        import ocrodeg

        return ocrodeg.transform_image(image, translation=self.magnitude)


class NumpyScale(DataTransform):
    """
    Changes scale of a NumPy image based on the scale in x-y.
    """

    def __init__(
        self,
        scale: Tuple[float],
        fill: float = 1.0,
        key: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(key=key)
        self.scale = scale
        self.fill = fill

    def _apply_transform(self, image: ArrayLike) -> ArrayLike:
        import numpy as np
        import torch
        from torchvision.transforms import RandomAffine

        image = torch.tensor(image).unsqueeze(0)
        scale = np.random.choice(self.scale)
        scale = [scale - 0.025, scale + 0.025]
        t = RandomAffine(degrees=0, scale=scale, fill=self.fill)
        image = t(image).squeeze().numpy()
        return image


class NumpyRotation(DataTransform):
    """
    Applies rotation to a NumPy image based on the magnitude in +/-.
    """

    def __init__(self, magnitude: float, key: Optional[Union[str, List[str]]] = None):
        super().__init__(key=key)
        self.magnitude = magnitude

    def _apply_transform(self, image: ArrayLike) -> ArrayLike:
        import scipy.ndimage as ndi

        return ndi.rotate(image, self.magnitude)


class NumpyRandomChoiceAffine(DataTransform):
    """
    Randomly applies affine transformation to a NumPy image based on the magnitudes of
    rotation degrees, translation, and shear around the top or the bottom value of
    the input range randomly.
    """

    def __init__(
        self,
        degrees: Tuple[float, float] = (0, 0),
        translate: Tuple[float, float] = (0, 0),
        shear: Tuple[float, float] = (0, 0),
        fill: float = 1.0,
        key: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(key=key)
        self.degrees = degrees
        self.translate = translate
        self.shear = shear
        self.fill = fill

    def _apply_transform(self, image: ArrayLike) -> ArrayLike:
        import numpy as np
        import torch
        from torchvision.transforms import RandomAffine

        image = torch.tensor(image).unsqueeze(0)
        translate = np.random.choice(self.translate)
        translate = [translate - 0.01, translate + 0.01]
        degrees = np.random.choice(self.degrees)
        degrees = [degrees - 1, degrees + 1]
        shear = np.random.choice(self.shear)
        shear = [shear - 0.05, shear + 0.05]
        t = RandomAffine(
            degrees=degrees, translate=translate, shear=shear, fill=self.fill
        )
        image = t(image).squeeze().numpy()
        return image


class NumpyElastic(DataTransform):
    """
    Applies elastic transformation to a NumPy image.
    """

    def __init__(
        self,
        alpha: float = 70,
        sigma: float = 500,
        alpha_affine: float = 10,
        random_state: Optional[float] = None,
        key: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(key=key)
        self.alpha = alpha
        self.sigma = sigma
        self.alpha_affine = alpha_affine
        self.random_state = random_state

    def _apply_transform(self, image: ArrayLike) -> ArrayLike:
        import cv2
        import numpy as np
        from scipy.ndimage.interpolation import map_coordinates
        from skimage.filters import gaussian

        assert len(image.shape) == 2
        shape = image.shape
        shape_size = shape[:2]

        image = np.array(image, dtype=np.float32) / 255.0
        shape = image.shape
        shape_size = shape[:2]

        # random affine
        center_square = np.float32(shape_size) // 2
        square_size = min(shape_size) // 3
        pts1 = np.float32(
            [
                center_square + square_size,
                [center_square[0] + square_size, center_square[1] - square_size],
                center_square - square_size,
            ]
        )
        pts2 = pts1 + np.random.uniform(
            -self.alpha_affine, self.alpha_affine, size=pts1.shape
        ).astype(np.float32)
        M = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(
            image, M, shape_size[::-1], borderMode=cv2.BORDER_REFLECT_101
        )

        dx = (
            gaussian(
                np.random.uniform(-1, 1, size=shape[:2]),
                self.sigma,
                mode="reflect",
                truncate=3,
            )
            * self.alpha
        ).astype(np.float32)
        dy = (
            gaussian(
                np.random.uniform(-1, 1, size=shape[:2]),
                self.sigma,
                mode="reflect",
                truncate=3,
            )
            * self.alpha
        ).astype(np.float32)

        x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
        indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
        return (
            np.clip(
                map_coordinates(image, indices, order=1, mode="reflect").reshape(shape),
                0,
                1,
            )
            * 255
        )


class TensorResizeOneDim(DataTransform):
    """
    Resizes a torch tensor image based on the dimensions provided.

    Args:
        resize_dim (int): Resize dimension.
        resize_smaller_dim (bool): Whether to resize smaller dimension or
            the larger dimension.
        max_size (Optional[int]): Max size for any side.
        antialias (Optional[bool]): Whether to use antialiasing while resizing image.
    """

    def __init__(
        self,
        resize_dim: int = 224,
        resize_smaller_dim: bool = False,
        interpolation: str = "bicubic",
        antialias: bool = False,
        key: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(key=key)
        self.resize_dim = resize_dim
        self.resize_smaller_dim = resize_smaller_dim
        self.interpolation = interpolation
        self.antialias = antialias

    def _apply_transform(self, image: "torch.Tensor") -> "torch.Tensor":
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import resize

        # shape (C, H, W)
        image_height = image.shape[1]
        image_width = image.shape[2]

        # get smaller dim
        larger_dim_idx = 1 if image_height > image_width else 2
        smaller_dim_idx = 1 if image_height < image_width else 2

        dim_idx = smaller_dim_idx if self.resize_smaller_dim else larger_dim_idx
        other_dim_idx = larger_dim_idx if self.resize_smaller_dim else smaller_dim_idx

        # find the resize ratio
        resize_ratio = self.resize_dim / image.shape[dim_idx]

        # resize the other dim
        other_dim = resize_ratio * image.shape[other_dim_idx]

        resized_shape = list(image.shape)
        resized_shape[dim_idx] = int(self.resize_dim)
        resized_shape[other_dim_idx] = int(other_dim)
        # resize the image according to the output shape
        return resize(
            image,
            size=resized_shape[1:],
            interpolation=InterpolationMode(self.interpolation),
            antialias=self.antialias,
        )


class TensorResizeWithAspectAndPad(DataTransform):
    """
    Resizes a torch tensor image based on the dimensions provided.

    Args:
        resize_dim (int): Resize dimension.
        resize_smaller_dim (bool): Whether to resize smaller dimension or
            the larger dimension.
        max_size (Optional[int]): Max size for any side.
        antialias (Optional[bool]): Whether to use antialiasing while resizing image.
    """

    def __init__(
        self,
        target_dims: Tuple[int, int] = (64, 256),
        interpolation: str = "bicubic",
        antialias: bool = False,
        pad_value: int = 255,
        key: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(key=key)
        self.target_dims = target_dims
        self.interpolation = interpolation
        self.antialias = antialias
        self.pad_value = pad_value

    def _apply_transform(self, image: "torch.Tensor") -> "torch.Tensor":
        import torchvision.transforms.functional as F
        from torchvision.transforms import InterpolationMode
        from torchvision.transforms.functional import resize

        _, h, w = image.shape
        if int(self.target_dims[0] / h * w) <= self.target_dims[1]:
            resized_shape = (self.target_dims[0], int(self.target_dims[0] / h * w))
            padding = int(self.target_dims[1] - resized_shape[1])
            padding = (padding // 2, 0, padding - padding // 2, 0)  # (x, y, x, y)
        elif int(self.target_dims[1] / w * h) <= self.target_dims[0]:
            resized_shape = (int(self.target_dims[1] / w * h), self.target_dims[1])
            padding = int(self.target_dims[0] - resized_shape[0])
            padding = (0, padding // 2, 0, padding - padding // 2)  # (x, y, x, y)

        resized_image = resize(
            image,
            size=resized_shape,
            interpolation=InterpolationMode(self.interpolation),
            antialias=self.antialias,
        )
        return F.pad(resized_image, padding, self.pad_value, "constant")


class TensorRandomResize(DataTransform):
    """
    Randomly resizes a torch tensor image based on the input list of possible
    dimensions.

    Args:
        resize_dims (list): List of possible sizes to choose from for
            shorter dimension.
        max_resize_dim (int): Maximum resize size for the larger dimension.
        max_iters (int): Maximum number of iterations to do for random sampling.
    """

    def __init__(
        self,
        resize_dims: Optional[typing.List[int]] = None,
        max_larger_dim: int = 512,
        max_iters: int = 100,
        key: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(key=key)
        self.resize_dims = (
            resize_dims if resize_dims is not None else [320, 416, 512, 608, 704]
        )
        self.max_larger_dim = max_larger_dim
        self.max_iters = max_iters

    def _apply_transform(self, image: "torch.Tensor") -> "torch.Tensor":
        import random

        from torchvision.transforms.functional import resize

        # randomly resize the image in the batch as done in ViBertGrid
        # shape (C, H, W)
        image_height = image.shape[1]
        image_width = image.shape[2]

        # get larger dim
        larger_dim_idx = 0 if image_height > image_width else 1
        smaller_dim_idx = 0 if image_height < image_width else 1

        resize_dims = [i for i in self.resize_dims]

        # find random resize dim
        resized_shape = None
        for iter in range(self.max_iters):
            if len(resize_dims) > 0:
                # get smaller dim out of possible dims
                idx, smaller_dim = random.choice(list(enumerate(resize_dims)))

                # find the resize ratio
                resize_ratio = smaller_dim / image.shape[smaller_dim_idx]

                # resize larger dim
                larger_dim = resize_ratio * image.shape[larger_dim_idx]

                # check if larger dim is smaller than max large
                if larger_dim > self.max_larger_dim:
                    resize_dims.pop(idx)
                else:
                    resized_shape = list(image.shape)
                    resized_shape[larger_dim_idx] = int(larger_dim)
                    resized_shape[smaller_dim_idx] = int(smaller_dim)
                    break
            else:
                # if no smaller dim is possible resize image according to
                # larger dim
                larger_dim = self.max_larger_dim

                # find the resize ratio
                resize_ratio = larger_dim / image.shape[larger_dim_idx]

                # resize smaller dim
                smaller_dim = resize_ratio * image.shape[smaller_dim_idx]

                resized_shape = list(image.shape)
                resized_shape[larger_dim_idx] = int(larger_dim)
                resized_shape[smaller_dim_idx] = int(smaller_dim)
                break

        if resized_shape is not None:
            # resize the image according to the output shape
            return resize(image, resized_shape[1:])
        else:
            return image


class TensorRandomResizedCrop(DataTransform):
    """
    Applies random resized cropping on a torch tensor image.
    """

    def __init__(
        self,
        size: Optional[List[int]] = None,
        scale: Tuple[float, float] = (0.08, 1),
        ratio: Tuple[float, float] = (3 / 4, 4 / 3),
        interpolation: str = "bicubic",
        key: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(key=key)

        from torchvision import transforms
        from torchvision.transforms import InterpolationMode

        self.size = size if size is not None else [224, 224]
        self.scale = scale
        self.ratio = ratio
        self.interpolation = interpolation
        self._transform = transforms.RandomResizedCrop(
            size=self.size,
            scale=self.scale,
            ratio=self.ratio,
            interpolation=InterpolationMode[self.interpolation.upper()],
            antialias=False,
        )

    def _apply_transform(self, image) -> "torch.Tensor":
        return self._transform(image)

    def __repr__(self) -> str:
        return str(self._transform)


class TensorResizeShortestEdge(DataTransform):
    """
    Applies random cropping on a torch tensor image.
    """

    def __init__(
        self,
        short_edge_length: List[int] = None,
        sample_style: str = "choice",
        key: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(key=key)
        self.short_edge_length = (
            short_edge_length if short_edge_length is not None else [400, 500, 600]
        )
        self.sample_style = sample_style
        from detectron2.data.transforms import ResizeShortestEdge

        self._transform = ResizeShortestEdge(
            short_edge_length=self.short_edge_length, sample_style=self.sample_style
        )

    def _apply_transform(self, image: "torch.Tensor") -> "torch.Tensor":
        return self._transform(image)


class TensorRandomCrop(DataTransform):
    """
    Applies random cropping on a torch tensor image.
    """

    def __init__(
        self,
        size: Optional[List[int]] = None,
        key: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(key=key)
        from torchvision import transforms

        self.size = size if size is not None else [224, 224]
        self._transform = transforms.RandomCrop(size=self.size)

    def _apply_transform(self, image) -> "torch.Tensor":
        return self._transform(image)


class TensorSquarePad(DataTransform):
    """
    Pads the image to right side with given backgroud pixel values
    """

    def __init__(
        self,
        pad_value: int = 255,
        key: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(key=key)
        self.pad_value = pad_value

    def _apply_transform(self, image: "torch.Tensor") -> "torch.Tensor":
        import torchvision.transforms.functional as F

        w, h = image.size
        max_wh = max([w, h])
        hp = int(max_wh - w)
        vp = int(max_wh - h)
        padding = (0, 0, hp, vp)
        return F.pad(image, padding, self.pad_value, "constant")


class NormalizeAndRescaleBoundingBoxes(DataTransform):
    """
    Normalizes the bbox coordinates based on the image size.
    """

    def __init__(
        self,
        image_key: str = DataKeys.IMAGE,
        bbox_key: str = DataKeys.WORD_BBOXES,
        rescale_constant: float = 1000.0,
        key: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(key=key)
        self.image_key = image_key
        self.bbox_key = bbox_key
        self.rescale_constant = rescale_constant

    def _apply_transform(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        import numpy as np
        import torch
        from PIL import Image

        assert (
            self.bbox_key in sample
        ), f"{self.bbox_key} not found in sample. Available keys are {sample.keys()}"
        assert (
            self.image_key in sample
        ), f"{self.image_key} not found in sample. Available keys are {sample.keys()}"

        image = sample[self.image_key]
        if isinstance(image, torch.Tensor):
            _, h, w = image.shape  # torch tensor is (C, H, W)
        elif isinstance(image, np.ndarray):
            h, w, _ = image.shape  # numpy array is (H, W, C)
        elif isinstance(image, Image):
            w, h = image.size  # PIL image is (W, H)
        else:
            raise TypeError("Image type not supported.")

        # get the bboxes
        bboxes = sample[self.bbox_key]

        # assert bbox has 4 values
        assert all(
            len(bbox) == 4 for bbox in bboxes
        ), "All bounding boxes should have 4 values."

        # if the bboxes are with 0 to 1 normalize to 0-1000 format
        input_type = type(bboxes)
        if isinstance(bboxes, input_type):
            bboxes = torch.tensor(bboxes).float()
        elif isinstance(bboxes, input_type):
            bboxes = torch.from_numpy(bboxes).float()
        elif isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.float()
        else:
            raise TypeError(
                "Bbox type not supported. Supported types are list, np.ndarray, and torch.Tensor."
            )

        # test
        bboxes = bboxes * self.rescale_constant / torch.tensor([w, h, w, h]).float()

        # convert back to the original type
        if input_type == list:
            sample[self.bbox_key] = bboxes.long()
        elif input_type == np.ndarray:
            sample[self.bbox_key] = bboxes.long()
        else:
            sample[self.bbox_key] = bboxes.long()

        return sample


class RescaleBoundingBoxes(DataTransform):
    """
    Rescale the bounding bxoes
    """

    def __init__(
        self,
        bbox_key: str = DataKeys.WORD_BBOXES,
        rescale_constant: float = 1000.0,
        key: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(key=key)
        self.bbox_key = bbox_key
        self.rescale_constant = rescale_constant

    def _apply_transform(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        import numpy as np
        import torch

        assert (
            self.bbox_key in sample
        ), f"{self.bbox_key} not found in sample. Available keys are {sample.keys()}"

        # get the bboxes
        bboxes = sample[self.bbox_key]

        # assert bbox has 4 values
        assert all(
            len(bbox) == 4 for bbox in bboxes
        ), "All bounding boxes should have 4 values."

        # if the bboxes are with 0 to 1 normalize to 0-1000 format
        input_type = type(bboxes)
        if isinstance(bboxes, input_type):
            bboxes = torch.tensor(bboxes).float()
        elif isinstance(bboxes, input_type):
            bboxes = torch.from_numpy(bboxes).float()
        elif isinstance(bboxes, torch.Tensor):
            bboxes = bboxes.float()
        else:
            raise TypeError(
                "Bbox type not supported. Supported types are list, np.ndarray, and torch.Tensor."
            )
        bboxes = bboxes * self.rescale_constant

        # convert back to the original type
        if input_type == list:
            sample[self.bbox_key] = bboxes.long()
        elif input_type == np.ndarray:
            sample[self.bbox_key] = bboxes.long()
        else:
            sample[self.bbox_key] = bboxes.long()

        return sample
