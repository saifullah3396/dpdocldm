import copy
import io
from typing import Any, List, Mapping, Optional, Tuple, Union

from atria.core.constants import DataKeys
from atria.core.data.data_transforms import DataTransform
from atria.core.utilities.logging import get_logger
from atria.data.data_transforms.blur import PilGaussianBlur
from atria.data.data_transforms.geometric import TensorSquarePad
from numpy.typing import ArrayLike

from .distortions import PilSolarization
from .general import PilRgbToGray, TensorGrayToRgb, TensorRgbToBgr

logger = get_logger(__name__)


class PilNumpyTensorResize(DataTransform):
    def __init__(
        self,
        key: Optional[Union[str, List[str]]] = None,
        rescale_size: Optional[List[int]] = None,
    ):
        super().__init__(key=key)
        self.rescale_size = rescale_size

    def _apply_transform(
        self, image: Union[ArrayLike, "PIL.Image.Image", "torch.Tensor"]
    ) -> ArrayLike:
        import numpy as np
        import torch
        from PIL import Image
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode

        if isinstance(image, Image.Image):
            return image.resize(self.rescale_size)
        elif isinstance(image, np.ndarray):
            return Image.fromarray(image).resize(self.rescale_size)
        elif isinstance(image, torch.Tensor):
            return transforms.Resize(
                self.rescale_size, interpolation=InterpolationMode.BICUBIC
            )(image)
        else:
            raise ValueError(
                "Image must be either PIL Image, Numpy array or torch.Tensor."
            )


class PilEncode(DataTransform):
    def __init__(
        self, key: Optional[Union[str, List[str]]] = None, encode_format: str = "PNG"
    ):
        super().__init__(key=key)
        self.encode_format = encode_format

    def _apply_transform(self, image: Union[ArrayLike, "PIL.Image.Image"]) -> str:
        from io import BytesIO

        import numpy as np
        from PIL import Image

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        buffer = BytesIO()
        image.save(buffer, format=self.encode_format)
        return buffer.getvalue()


class Binarize(DataTransform):
    def __init__(
        self,
        key: Optional[Union[str, List[str]]] = None,
        binarize_threshold: float = 0.5,
    ):
        super().__init__(key=key)
        self.binarize_threshold = binarize_threshold

    def _apply_transform(self, image: ArrayLike) -> ArrayLike:
        return (image > self.binarize_threshold).to(image.dtype)


class To3ChannelGray(DataTransform):
    def __init__(self, key: Optional[Union[str, List[str]]] = None):
        super().__init__(key=key)

    def _apply_transform(self, image: ArrayLike) -> ArrayLike:
        return image.convert("L").convert("RGB")


class ImageSelect(DataTransform):
    def __init__(
        self,
        key: Optional[Union[str, List[str]]] = None,
        index: Union[int, List[int]] = 0,
        random_select: bool = False,
    ):
        super().__init__(key=key)
        self.index = index
        self.random_select = random_select

    def _apply_transform(
        self, image: List["PIL.Image.Image"]
    ) -> List["PIL.Image.Image"]:
        import random

        if self.random_select:
            images = []
            images.append(image[0])

            rand_index = random.randint(1, len(image) - 1)
            images.append(image[rand_index])
            return images
        else:
            if isinstance(self.index, list):
                return [image[idx] for idx in self.index]
            else:
                return image[self.index]


class ImagePreprocess(DataTransform):
    def __init__(
        self,
        key: Optional[Union[str, List[str]]] = DataKeys.IMAGE,
        square_pad: bool = False,
        rescale_size: Optional[Tuple[int, ...]] = None,
        encode_image: bool = False,
        encode_format: str = "PNG",
        decode_image_if_needed: bool = True,
    ):
        super().__init__(key=key)
        self.square_pad = square_pad
        self.rescale_size = rescale_size
        self.encode_image = encode_image
        self.encode_format = encode_format
        self.decode_image_if_needed = decode_image_if_needed
        self._tf = self._initialize_transform()

    def _initialize_transform(self):
        from torchvision import transforms

        tf = []

        if self.square_pad:
            tf.append(TensorSquarePad())

        if self.rescale_size is not None:
            tf.append(PilNumpyTensorResize(rescale_size=self.rescale_size))

        if self.encode_image:
            tf.append(PilEncode(encode_format=self.encode_format))

        return transforms.Compose(tf)

    def _apply_transform(
        self, image: Union["PIL.Image.Image", ArrayLike]
    ) -> Union[str, ArrayLike]:
        from PIL import Image

        if self.decode_image_if_needed:
            if isinstance(image, dict) and "path" in image and "bytes" in image:
                try:
                    if image["bytes"] is not None:
                        image = Image.open(io.BytesIO(image["bytes"]))
                    else:
                        image = Image.open(image["path"])
                    if image.mode == "1":
                        image = image.convert("L")
                except:
                    # if the image is not a valid image, we create a blank image
                    image = Image.new(mode="L", size=(256, 256))
                    logger.warning("Invalid image found in the dataset.")
        return self._tf(image)


class Cifar10Aug(DataTransform):
    def __init__(
        self,
        key: Optional[Union[str, List[str]]] = None,
        mean: Union[float, List[float]] = (0.4914, 0.4822, 0.4465),
        std: Union[float, List[float]] = (0.247, 0.243, 0.261),
        pad_size: int = 4,
        crop_size: int = 32,
        train: bool = False,
    ):
        super().__init__(key=key)
        self.mean = mean
        self.std = std
        self.pad_size = pad_size
        self.crop_size = crop_size
        self.train = train
        self._tf = self._initialize_transform()

    def _initialize_transform(self):
        import numpy as np
        from torchvision import transforms

        if self.train:
            return transforms.Compose(
                [
                    transforms.Pad(self.pad_size, padding_mode="reflect"),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomCrop(self.crop_size),
                    transforms.ToTensor(),
                    transforms.Normalize(np.array(self.mean), np.array(self.std)),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize(np.array(self.mean), np.array(self.std)),
                ]
            )

    def _apply_transform(self, image: "PIL.Image.Image") -> "torch.Tensor":
        return self._tf(image)

    def __repr__(self):
        super().__repr__()
        return str(self._tf)


class ImageLoader(DataTransform):
    def __init__(
        self,
        key: Optional[Union[str, List[str]]] = None,
        image_file_path_key: str = DataKeys.IMAGE_FILE_PATH,
        output_image_key: str = DataKeys.IMAGE,
    ):
        super().__init__(key=key)
        self.image_file_path_key = image_file_path_key
        self.output_image_key = output_image_key

    def _apply_transform(self, sample: Mapping[str, Any]) -> "torch.Tensor":
        from PIL import Image

        sample[self.output_image_key] = Image.open(sample[self.image_file_path_key])
        return sample


class BasicImageAug(DataTransform):
    """
    Defines a basic image transformation for image classification.
    """

    def __init__(
        self,
        key: Optional[Union[str, List[str]]] = None,
        gray_to_rgb: bool = False,
        rgb_to_bgr: bool = False,
        rgb_to_gray: bool = False,
        resize: Optional[dict] = None,
        center_crop: Optional[List[int]] = None,
        random_resized_crop: Optional[dict] = None,
        normalize: bool = True,
        random_hflip: bool = False,
        random_vflip: bool = False,
        mean: List[float] = (0.485, 0.456, 0.406),
        std: List[float] = (0.229, 0.224, 0.225),
        binarize: bool = False,
        binarize_threshold: float = 0.5,
        to_3_channel_gray: bool = False,
        image_select: Optional[List[int]] = None,
        decode_image_if_needed: bool = True,
    ):
        self.gray_to_rgb = gray_to_rgb
        self.rgb_to_bgr = rgb_to_bgr
        self.rgb_to_gray = rgb_to_gray
        self.resize = resize
        self.center_crop = center_crop
        self.random_resized_crop = random_resized_crop
        self.normalize = normalize
        self.random_hflip = random_hflip
        self.random_vflip = random_vflip
        self.mean = mean
        self.std = std
        self.binarize = binarize
        self.binarize_threshold = binarize_threshold
        self.to_3_channel_gray = to_3_channel_gray
        self.image_select = image_select
        self.decode_image_if_needed = decode_image_if_needed
        self._tf = self._initialize_transform()

        key = key if key is not None else DataKeys.IMAGE
        super().__init__(key=key)

    def _initialize_transform(self):
        import torch
        from torchvision import transforms
        from torchvision.transforms import InterpolationMode

        # generate transformations list
        tf = []

        self._selection_transform = None
        if self.image_select is not None:
            if self.image_select == "random":
                self._selection_transform = ImageSelect(index=0, random_select=True)
            else:
                self._selection_transform = ImageSelect(index=self.image_select)

        # apply rgb to bgr if required
        if self.rgb_to_gray:
            tf.append(PilRgbToGray())

        if self.to_3_channel_gray:
            tf.append(To3ChannelGray())

        # convert images to tensor
        tf.append(transforms.ToTensor())

        # apply gray to rgb if required
        if self.gray_to_rgb:
            tf.append(TensorGrayToRgb())

        # apply rgb to bgr if required
        if self.rgb_to_bgr:
            tf.append(TensorRgbToBgr())

        # apply rescaling if required
        if self.resize is not None:
            resize_args = copy.deepcopy(self.resize)
            if "interpolation" in self.resize:
                resize_args["interpolation"] = InterpolationMode(
                    resize_args["interpolation"]
                )
            tf.append(transforms.Resize(**resize_args))

        # apply image binarization if required
        if self.binarize:
            tf.append(Binarize(binarize_threshold=self.binarize_threshold))

        # apply center crop if required
        if self.center_crop is not None:
            tf.append(transforms.CenterCrop(self.center_crop))

        # apply center crop if required
        if self.random_resized_crop is not None:
            tf.append(transforms.RandomResizedCrop(**self.random_resized_crop))

        # apply random horizontal flip if required
        if self.random_hflip:
            tf.append(transforms.RandomHorizontalFlip(0.5))

        # apply random vertical flip if required
        if self.random_vflip:
            tf.append(transforms.RandomVerticalFlip(0.5))

        # change dtype to float
        tf.append(transforms.ConvertImageDtype(torch.float))

        # normalize image if required
        if self.normalize:
            if isinstance(self.mean, float):
                tf.append(transforms.Normalize((self.mean,), (self.std,)))
            else:
                tf.append(transforms.Normalize(self.mean, self.std))

        # generate torch transformation
        return transforms.Compose(tf)

    def _apply_transform(self, image: "PIL.Image.Image") -> "torch.Tensor":
        if self.decode_image_if_needed:
            from PIL import Image

            if isinstance(image, dict) and "path" in image and "bytes" in image:
                try:
                    image = Image.open(io.BytesIO(image["bytes"]))
                    if image.mode == "1":
                        image = image.convert("L")
                except:
                    # if the image is not a valid image, we create a blank image
                    image = Image.new(mode="L", size=(256, 256))
        if self._selection_transform is not None:
            image = self._selection_transform(image)
        return self._tf(image)

    def __repr__(self):
        super().__repr__()
        return self.key + ":\n" + str(self._tf)


class BinarizationAug(DataTransform):
    """
    Defines a image transformation for image to image binarizaiton task
    """

    def __init__(
        self,
        key: Optional[Union[str, List[str]]] = None,
        image_key: str = DataKeys.IMAGE,
        gt_image_key: str = DataKeys.GT_IMAGE,
        gray_to_rgb: bool = False,
        rgb_to_gray: bool = False,
        normalize: bool = True,
        random_hflip: bool = False,
        random_vflip: bool = False,
        mean: Union[float, List[float]] = (0.5, 0.5, 0.5),
        std: Union[float, List[float]] = (0.5, 0.5, 0.5),
        image_size: int = 256,
    ):
        super().__init__(key=key)
        self.image_key = image_key
        self.gt_image_key = gt_image_key
        self.gray_to_rgb = gray_to_rgb
        self.rgb_to_gray = rgb_to_gray
        self.normalize = normalize
        self.random_hflip = random_hflip
        self.random_vflip = random_vflip
        self.mean = mean
        self.std = std
        self.image_size = image_size
        self._tf = self._initialize_transform()
        self._tf_gt = self._initialize_transform_gt()

    def _initialize_transform(self):
        import torch
        from torchvision import transforms

        tf = []
        if self.rgb_to_gray:
            tf.append(PilRgbToGray())
        tf.append(transforms.ToTensor())
        if self.gray_to_rgb:
            tf.append(TensorGrayToRgb())
        tf.append(transforms.ConvertImageDtype(torch.float))
        if self.normalize:
            if isinstance(self.mean, float):
                tf.append(transforms.Normalize((self.mean,), (self.std,)))
            else:
                tf.append(transforms.Normalize(self.mean, self.std))
        return transforms.Compose(tf)

    def _initialize_transform_gt(self):
        import torch
        from torchvision import transforms

        tf = []
        tf.append(PilRgbToGray())
        tf.append(transforms.ToTensor())
        tf.append(Binarize())
        tf.append(transforms.ConvertImageDtype(torch.float))
        if self.normalize:
            if isinstance(self.mean, float):
                tf.append(transforms.Normalize((self.mean,), (self.std,)))
            else:
                tf.append(transforms.Normalize(self.mean, self.std))
        return transforms.Compose(tf)

    def _apply_transform(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        import random

        import torchvision.transforms.functional as TF
        from torchvision import transforms

        image = sample[self.image_key]
        gt_image = sample[self.gt_image_key]

        # random crop
        i, j, h, w = transforms.RandomCrop.get_params(
            gt_image, output_size=(self.image_size, self.image_size)
        )
        i, j, h, w = transforms.RandomCrop.get_params(
            image, output_size=(self.image_size, self.image_size)
        )
        image = TF.crop(image, i, j, h, w)
        gt_image = TF.crop(gt_image, i, j, h, w)

        # random horizontal flipping
        if self.random_hflip and random.random() > 0.5:
            gt_image = TF.hflip(gt_image)
            image = TF.hflip(image)

        # random vertical flipping
        if self.random_vflip and random.random() > 0.5:
            gt_image = TF.vflip(gt_image)
            image = TF.vflip(image)

        image = self._tf(image)
        gt_image = self._tf_gt(gt_image)

        sample[self.image_key] = image
        sample[self.gt_image_key] = gt_image

        return sample

    def __repr__(self):
        super().__repr__()
        return str(self._tf)


class RandAug(DataTransform):
    """
    Applies the ImageNet Random Augmentation to torch tensor or Numpy image as
    defined in timm for image classification with little modification.
    """

    def __init__(
        self,
        key: Optional[Union[str, List[str]]] = None,
        input_size: int = 224,
        is_training: bool = True,
        use_prefetcher: bool = False,
        no_aug: bool = False,
        scale: Optional[float] = None,
        ratio: Optional[float] = None,
        hflip: float = 0.5,
        vflip: float = 0.0,
        color_jitter: Union[float, List[float]] = 0.4,
        auto_augment: Optional[str] = "rand-m9-mstd0.5-inc1",
        interpolation: str = "bicubic",
        mean: Union[float, List[float]] = (0.485, 0.456, 0.406),
        std: Union[float, List[float]] = (0.229, 0.224, 0.225),
        re_prob: float = 0.0,
        re_mode: str = "const",
        re_count: int = 1,
        re_num_splits: int = 0,
        crop_pct: Optional[float] = None,
        tf_preprocessing: bool = False,
        separate: bool = False,
        n_augs: int = 1,
        fixed_resize: bool = False,
    ):
        super().__init__(key=key)
        self.input_size = input_size
        self.is_training = is_training
        self.use_prefetcher = use_prefetcher
        self.no_aug = no_aug
        self.scale = scale
        self.ratio = ratio
        self.hflip = hflip
        self.vflip = vflip
        self.color_jitter = color_jitter
        self.auto_augment = auto_augment
        self.interpolation = interpolation
        self.mean = mean
        self.std = std
        self.re_prob = re_prob
        self.re_mode = re_mode
        self.re_count = re_count
        self.re_num_splits = re_num_splits
        self.crop_pct = crop_pct
        self.tf_preprocessing = tf_preprocessing
        self.separate = separate
        self.n_augs = n_augs
        self.fixed_resize = fixed_resize
        self._tf = self._initialize_transform()

    def _initialize_transform(self):
        from timm.data import create_transform
        from torchvision import transforms

        torch_str_to_interpolation = {
            "nearest": transforms.InterpolationMode.NEAREST,
            "bilinear": transforms.InterpolationMode.BILINEAR,
            "bicubic": transforms.InterpolationMode.BICUBIC,
            "box": transforms.InterpolationMode.BOX,
            "hamming": transforms.InterpolationMode.HAMMING,
            "lanczos": transforms.InterpolationMode.LANCZOS,
        }

        tf = create_transform(
            input_size=self.input_size,
            is_training=self.is_training,
            use_prefetcher=self.use_prefetcher,
            no_aug=self.no_aug,
            scale=self.scale,
            ratio=self.ratio,
            hflip=self.hflip,
            vflip=self.vflip,
            color_jitter=self.color_jitter,
            auto_augment=self.auto_augment,
            interpolation=self.interpolation,
            mean=self.mean,
            std=self.std,
            re_prob=self.re_prob,
            re_mode=self.re_mode,
            re_count=self.re_count,
            re_num_splits=self.re_num_splits,
            crop_pct=self.crop_pct,
            tf_preprocessing=self.tf_preprocessing,
            separate=self.separate,
        ).transforms

        # replace random resized crop with fixed resizing if required
        if self.fixed_resize:
            tf[0] = transforms.Resize(
                self.input_size,
                interpolation=torch_str_to_interpolation[self.interpolation],
            )

            # this makes sure image is always 3-channeled.
            tf.insert(0, TensorGrayToRgb())

            # this makes sure image is always 3-channeled.
            tf.insert(2, transforms.ToPILImage())
        else:
            # this makes sure image is always 3-channeled.
            tf.insert(0, TensorGrayToRgb())

            # this makes sure image is always 3-channeled.
            tf.insert(1, transforms.ToPILImage())

        # generate torch transformation
        return transforms.Compose(tf)

    def _apply_transform(self, image: "torch.Tensor") -> List["torch.Tensor"]:
        if self.n_augs == 1:
            return self._tf(image)
        else:
            tfs = []
            for _ in self.n_tfs:
                tfs.append(self._tf(image))
                tfs.append(self._tf(image))
        return tfs

    def __repr__(self):
        super().__repr__()
        if self.n_augs == 1:
            return str(self._tf)
        else:
            return str([self._tf for _ in self.n_augs])


class Moco(DataTransform):
    """
    Applies the Standard Moco Augmentation to a torch tensor image.
    """

    def __init__(
        self,
        key: Optional[Union[str, List[str]]] = None,
        image_size: Union[int, Tuple[int, int]] = 224,
        gray_to_rgb: bool = False,
        to_pil: bool = False,
        mean: Union[float, List[float]] = (0.485, 0.456, 0.406),
        std: Union[float, List[float]] = (0.229, 0.224, 0.225),
    ):
        super().__init__(key=key)
        self.image_size = image_size
        self.gray_to_rgb = gray_to_rgb
        self.to_pil = to_pil
        self.mean = mean
        self.std = std
        self._tf = self._initialize_transform()

    def _initialize_transform(self):
        from torchvision import transforms
        from torchvision.transforms.functional import InterpolationMode

        base_transform = []
        if self.gray_to_rgb:
            base_transform.append(TensorGrayToRgb())

        if self.to_pil:
            base_transform.append(transforms.ToPILImage())

        return transforms.Compose(
            base_transform
            + [
                transforms.RandomResizedCrop(
                    self.image_size,
                    scale=(0.2, 1.0),
                    interpolation=InterpolationMode.BICUBIC,
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([PilGaussianBlur(sigma=(0.1, 2.0))], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

    def _apply_transform(self, image: "torch.Tensor") -> List["torch.Tensor"]:
        crops = []
        crops.append(self._tf(image))
        crops.append(self._tf(image))
        return crops

    def __repr__(self):
        super().__repr__()
        return str(self._tf)


class BarlowTwins(DataTransform):
    """
    Applies the Standard BarlowTwins Augmentation to a torch tensor image.
    """

    def __init__(
        self,
        key: Optional[Union[str, List[str]]] = None,
        image_size: Union[int, Tuple[int, int]] = 224,
        gray_to_rgb: bool = False,
        to_pil: bool = False,
        mean: Union[float, List[float]] = (0.485, 0.456, 0.406),
        std: Union[float, List[float]] = (0.229, 0.224, 0.225),
    ):
        super().__init__(key=key)
        self.image_size = image_size
        self.gray_to_rgb = gray_to_rgb
        self.to_pil = to_pil
        self.mean = mean
        self.std = std
        self.__tf1, self.__tf2 = self._initialize_transform()

    def _initialize_transform(self):
        from torchvision import transforms
        from torchvision.transforms.functional import InterpolationMode

        base_transform = []
        if self.gray_to_rgb:
            base_transform.append(TensorGrayToRgb())
        if self.to_pil:
            base_transform.append(transforms.ToPILImage())

        tf1 = transforms.Compose(
            base_transform
            + [
                transforms.RandomResizedCrop(
                    self.image_size, interpolation=InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([PilGaussianBlur([0.1, 2.0])], p=1.0),
                transforms.RandomApply([PilSolarization()], p=0.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )
        tf2 = transforms.Compose(
            base_transform
            + [
                transforms.RandomResizedCrop(
                    self.image_size, interpolation=InterpolationMode.BICUBIC
                ),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([PilGaussianBlur([0.1, 2.0])], p=0.1),
                transforms.RandomApply([PilSolarization()], p=0.0),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean, std=self.std),
            ]
        )

        return tf1, tf2

    def _apply_transform(self, image: "torch.Tensor") -> List["torch.Tensor"]:
        crops = []
        crops.append(self.__tf1(image))
        crops.append(self.__tf2(image))
        return crops

    def __repr__(self):
        super().__repr__()
        return str([self.__tf1, self.__tf2])


class MultiCrop(DataTransform):
    """
    Applies the Standard Multicrop Augmentation to a torch tensor image.
    """

    def __init__(
        self,
        key: Optional[Union[str, List[str]]] = None,
        image_size: Union[int, Tuple[int, int]] = 224,
        gray_to_rgb: bool = False,
        to_pil: bool = False,
        mean: Union[float, List[float]] = (0.485, 0.456, 0.406),
        std: Union[float, List[float]] = (0.229, 0.224, 0.225),
        global_crops_scale: Tuple[float, float] = (0.4, 1.0),
        local_crops_scale: Tuple[float, float] = (0.05, 0.4),
        local_crops_number: int = 12,
        local_crop_size: Union[int, Tuple[int, int]] = 96,
    ):
        super().__init__(key=key)
        self.image_size = image_size
        self.gray_to_rgb = gray_to_rgb
        self.to_pil = to_pil
        self.mean = mean
        self.std = std
        self.global_crops_scale = global_crops_scale
        self.local_crops_scale = local_crops_scale
        self.local_crops_number = local_crops_number
        self.local_crop_size = local_crop_size
        self._initialize_transform()

    def _initialize_transform(self):
        from torchvision import transforms
        from torchvision.transforms.functional import InterpolationMode

        base_transform = []
        if self.gray_to_rgb:
            base_transform.append(TensorGrayToRgb())
        if self.to_pil:
            base_transform.append(transforms.ToPILImage())

        flip_and_color_jitter = transforms.Compose(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1
                        )
                    ],
                    p=0.8,
                ),
                transforms.RandomGrayscale(p=0.2),
            ]
        )
        normalize = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

        # first global crop
        self._global_transform_1 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.image_size,
                    scale=self.global_crops_scale,
                    interpolation=InterpolationMode.BICUBIC,
                ),
                flip_and_color_jitter,
                transforms.RandomApply([PilGaussianBlur([0.1, 2.0])], p=1.0),
                normalize,
            ]
        )
        # second global crop
        self._global_transform_2 = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.image_size,
                    scale=self.global_crops_scale,
                    interpolation=InterpolationMode.BICUBIC,
                ),
                flip_and_color_jitter,
                transforms.RandomApply([PilGaussianBlur([0.1, 2.0])], p=0.1),
                transforms.RandomApply([PilSolarization()], p=0.2),
                normalize,
            ]
        )
        # transformation for the local small crops
        self._local_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    self.local_crop_size,
                    scale=self.local_crops_scale,
                    interpolation=InterpolationMode.BICUBIC,
                ),
                flip_and_color_jitter,
                transforms.RandomApply([PilGaussianBlur([0.1, 2.0])], p=0.5),
                normalize,
            ]
        )

    def _apply_transform(self, image: "torch.Tensor") -> List["torch.Tensor"]:
        crops = []
        crops.append(self._global_transform_1(image))
        crops.append(self._global_transform_2(image))
        for _ in range(self.local_crops_number):
            crops.append(self._local_transform(image))
        return crops

    def __repr__(self):
        super().__repr__()

        import json

        return json.dumps(
            {
                "global_1": str(self._global_transform_1),
                "global_2": str(self._global_transform_2),
                "local": str(self._local_transform),
            },
            indent=2,
        )
