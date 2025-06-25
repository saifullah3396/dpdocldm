from typing import Any, List, Mapping, Optional, Union

from atria.core.constants import DataKeys
from atria.core.data.data_transforms import DataTransform
from atria.core.utilities.logging import get_logger
from atria.data.data_transforms.advanced import PilEncode

logger = get_logger(__name__)


def check_image_size(sample, image):
    """
    Raise an error if the image does not match the size specified in the dict.
    """
    if DataKeys.IMAGE_WIDTH in sample or DataKeys.IMAGE_HEIGHT in sample:
        # image has h, w, c as numpy array
        image_wh = (image.shape[1], image.shape[0])
        expected_wh = (
            sample[DataKeys.IMAGE_WIDTH],
            sample[DataKeys.IMAGE_HEIGHT],
        )
        if not image_wh == expected_wh:
            raise ValueError(
                "Mismatched image shape{}, got {}, expect {}.".format(
                    (
                        " for image " + sample["file_name"]
                        if "file_name" in sample
                        else ""
                    ),
                    image_wh,
                    expected_wh,
                )
                + " Please check the width/height in your annotation."
            )

    # To ensure bbox always remap to original image size
    if DataKeys.IMAGE_HEIGHT not in sample or sample[DataKeys.IMAGE_HEIGHT] is None:
        sample[DataKeys.IMAGE_HEIGHT] = image.shape[0]
    if DataKeys.IMAGE_WIDTH not in sample or sample[DataKeys.IMAGE_WIDTH] is None:
        sample[DataKeys.IMAGE_WIDTH] = image.shape[1]


def detectron2_preprocess_transform_image_and_objects(sample, geometric_tf):
    from detectron2.data.detection_utils import transform_instance_annotations
    from detectron2.data.transforms import apply_transform_gens

    # we always read image in RGB format in the dataset, when it reaches here the image is of numpay array with shape (h, w, c)
    # detectron2 needs image of shape (h, w, c) and in this place of transformation.
    # here we once resize the image to max possible sizes needed during training/testing
    image = sample[DataKeys.IMAGE]

    # sample must contain image height and image width as done for coco type datasets
    # here we assume that the sample has those. If not the image width and heights are set
    check_image_size(sample, image)

    # here the image is resized to correct aspect ratio
    # the returned geometric_tf is needed for bbox transformation in detectron2
    image, geometric_tf = apply_transform_gens(geometric_tf, image)

    # store the image shape here
    image_shape = image.shape[:2]  # h, w

    # To ensure bbox always remap to original image size, we reset image shape here as this is only preprocessing
    sample[DataKeys.IMAGE_HEIGHT] = image_shape[0]
    sample[DataKeys.IMAGE_WIDTH] = image_shape[1]

    if "objects" in sample:  # convert the objects to the new image size
        # here objects are transformed from XYWH_ABS to XYXY_ABS
        sample["objects"] = [
            transform_instance_annotations(obj, geometric_tf, image_shape)
            for obj in sample["objects"]
            if obj.get("iscrowd", 0) == 0
        ]

    # update image in place
    sample[DataKeys.IMAGE] = image
    return sample


def detectron2_realtime_transform_image_and_objects(sample, geometric_tf, mask_on):
    import numpy as np
    import torch
    from detectron2.data.detection_utils import (
        annotations_to_instances,
        filter_empty_instances,
        transform_instance_annotations,
    )
    from detectron2.data.transforms import apply_transform_gens
    from detectron2.structures import BoxMode

    # we always read image in RGB format in the dataset, when it reaches here the image is of numpay array with shape (h, w, c)
    # detectron2 needs image of shape (h, w, c) and in this place of transformation.
    # here we once resize the image to max possible sizes needed during training/testing
    image = np.array(sample[DataKeys.IMAGE])

    # sample must contain image height and image width as done for coco type datasets
    # here we assume that the sample has those. If not the image width and heights are set
    check_image_size(sample, image)

    # here the image is resized to correct aspect ratio
    # the returned geometric_tf is needed for bbox transformation in detectron2
    image, geometric_tf = apply_transform_gens(geometric_tf, image)

    # store the image shape here
    image_shape = image.shape[:2]  # h, w

    if "objects" in sample:  # convert the objects to the new image size
        # USER: Modify this if you want to keep them for some reason.
        for obj in sample["objects"]:
            if not mask_on:
                obj.pop("segmentation", None)
            obj.pop("keypoints", None)

            if "bbox_mode" in obj:
                obj["bbox_mode"] = BoxMode(obj["bbox_mode"])

        # here objects are transformed from XYWH_ABS to XYXY_ABS
        sample["objects"] = [
            transform_instance_annotations(obj, geometric_tf, image_shape)
            for obj in sample["objects"]
            if obj.get("iscrowd", 0) == 0
        ]

        instances = annotations_to_instances(sample["objects"], image_shape)
        sample["instances"] = filter_empty_instances(instances)

    # convert image to tensor and update in place
    sample[DataKeys.IMAGE] = (
        torch.as_tensor(  # here image is kept as 0-255 as that is what is required in detectron2
            np.ascontiguousarray(image.astype("float32").transpose(2, 0, 1))
        )
    )
    return sample


class ObjectDetectionImagePreprocess(DataTransform):
    """
    Defines a basic image preprocessing for image object detection based on detectron2.
    """

    def __init__(self, key: Optional[Union[str, List[str]]] = None):
        super().__init__(key=key)
        self.image_key = DataKeys.IMAGE
        self.encode_image = True
        self.encode_format = "PNG"
        self.min_size = 800  # this is the size used in detectron2 default config, we preprocess all image to this size
        self.max_size = 1333  # this is the size used in detectron2 default config, we preprocess all image to this size
        self._initialize_transform()

    def _initialize_transform(self):
        from detectron2.data.transforms import ResizeShortestEdge
        from torchvision import transforms

        # generate transformations list
        self.image_postprocess_tf = transforms.Compose(
            [transforms.ToPILImage(), PilEncode(encode_format=self.encode_format)]
        )

        # this can only be applied to tensors
        self.geometric_tf = [ResizeShortestEdge(self.min_size, self.max_size)]

    def _apply_transform(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        sample = detectron2_preprocess_transform_image_and_objects(
            sample, self.geometric_tf
        )

        if self.encode_image:
            sample[self.image_key] = self.image_postprocess_tf(sample[self.image_key])
        return sample


class ObjectDetectionImageAug(DataTransform):
    """
    Defines a basic image transformation for image object detection based on detectron2.
    """

    def __init__(self, key: Optional[Union[str, List[str]]] = None):
        super().__init__(key=key)
        self.min_size = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]
        self.max_size = 1333  # this is the size used in detectron2 default config, we preprocess all image to this size
        self.mask_on = False
        self.random_flip = False
        self.sampling_style = "choice"
        self.keep_objects = False
        self._initialize_transform()

    def _initialize_transform(self):
        from detectron2.data.transforms import RandomFlip, ResizeShortestEdge

        # this can only be applied to tensors
        self.geometric_tf = []
        if self.random_flip:
            self.geometric_tf.append(RandomFlip())
        self.geometric_tf.append(
            ResizeShortestEdge(
                self.min_size, self.max_size, sample_style=self.sampling_style
            )
        )

    def _apply_transform(self, sample: Mapping[str, Any]) -> Mapping[str, Any]:
        sample = detectron2_realtime_transform_image_and_objects(
            sample, self.geometric_tf, mask_on=self.mask_on
        )

        if not self.keep_objects:
            sample.pop("objects", None)
        return sample
