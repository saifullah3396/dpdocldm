from typing import List, Optional, Union

from numpy.typing import ArrayLike

from atria.core.data.data_transforms import DataTransform


class PilSolarization(DataTransform):
    """
    Applies solarization to a numpy image.
    """

    def __init__(self, key: Optional[Union[str, List[str]]] = None):
        super().__init__(key=key)

    def _apply_transform(self, image: "PIL.Image.Image") -> "PIL.Image.Image":
        from PIL import ImageOps

        return ImageOps.solarize(image)


class NumpyRandomDistortion(DataTransform):
    """
    Applies random distortion to a numpy image.
    """

    def __init__(
        self,
        sigma: float,
        maxdelta: float,
        key: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(key=key)
        self.sigma = sigma
        self.maxdelta = maxdelta

    def _apply_transform(self, image: ArrayLike) -> ArrayLike:
        import ocrodeg

        noise = ocrodeg.bounded_gaussian_noise(image.shape, self.sigma, self.maxdelta)
        return ocrodeg.distort_with_noise(image, noise)


class NumpyRandomBlotches(DataTransform):
    """
    Applies random blobs to a numpy image.
    """

    def __init__(
        self,
        fgblobs: float,
        bgblobs: float,
        fgscale: float = 10,
        bgscale: float = 10,
        key: Optional[Union[str, List[str]]] = None,
    ):
        super().__init__(key=key)
        self.fgblobs = fgblobs
        self.bgblobs = bgblobs
        self.fgscale = fgscale
        self.bgscale = bgscale

    def _apply_transform(self, image: ArrayLike) -> ArrayLike:
        import ocrodeg

        return ocrodeg.random_blotches(
            image,
            fgblobs=self.fgblobs,
            bgblobs=self.bgblobs,
            fgscale=self.fgscale,
            bgscale=self.bgscale,
        )


class NumpySurfaceDistortion(DataTransform):
    """
    Applies surface distortion to a numpy image.
    """

    def __init__(self, magnitude: float, key: Optional[Union[str, List[str]]] = None):
        super().__init__(key=key)
        self.magnitude = magnitude

    def _apply_transform(self, image: ArrayLike) -> ArrayLike:
        import ocrodeg

        noise = ocrodeg.noise_distort1d(image.shape, magnitude=self.magnitude)
        return ocrodeg.distort_with_noise(image, noise)


class NumpyThreshold(DataTransform):
    """
    Applies threshold distortion on a numpy image.
    """

    def __init__(self, magnitude: float, key: Optional[Union[str, List[str]]] = None):
        super().__init__(key=key)
        self.magnitude = magnitude

    def _apply_transform(self, image: ArrayLike) -> ArrayLike:
        import scipy.ndimage as ndi

        blurred = ndi.gaussian_filter(image, self.magnitude)
        return 1.0 * (blurred > 0.5)


class NumpyPixelate(DataTransform):
    """
    Applies pixelation to a numpy image.
    """

    def __init__(self, magnitude: float, key: Optional[Union[str, List[str]]] = None):
        super().__init__(key=key)
        self.magnitude = magnitude

    def _apply_transform(self, image: ArrayLike) -> ArrayLike:
        import cv2

        h, w = image.shape
        image = cv2.resize(
            image,
            (int(w * self.magnitude), int(h * self.magnitude)),
            interpolation=cv2.INTER_LINEAR,
        )
        return cv2.resize(image, (w, h), interpolation=cv2.INTER_NEAREST)


class NumpyJpegCompression(DataTransform):
    """
    Applies jpeg compression to a numpy image.
    """

    def __init__(self, quality: float, key: Optional[Union[str, List[str]]] = None):
        super().__init__(key=key)
        self.quality = quality

    def _apply_transform(self, image: ArrayLike) -> ArrayLike:
        import cv2

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), self.quality]
        result, encimg = cv2.imencode(".jpg", image * 255, encode_param)
        decimg = cv2.imdecode(encimg, 0) / 255.0
        return decimg
