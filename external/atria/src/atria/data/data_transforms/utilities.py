from typing import List, Sequence, Union

from numpy.typing import ArrayLike
from transformers import PreTrainedTokenizer

from atria.core.constants import DataKeys


def create_default_data_padding_dict(tokenizer: PreTrainedTokenizer) -> dict:
    return {
        DataKeys.TOKEN_IDS: tokenizer.pad_token_id,
        DataKeys.TOKEN_TYPE_IDS: tokenizer.pad_token_type_id,
        DataKeys.POSITION_IDS: tokenizer.pad_token_id,
        DataKeys.ATTENTION_MASKS: 0,
        DataKeys.TOKEN_BBOXES: [0, 0, 0, 0],
        DataKeys.TOKEN_ANGLES: 0,
        DataKeys.LABEL: -100,
        DataKeys.NER_TAGS: -100,
        DataKeys.WORD_LABELS: -100,
    }


def pad_sequence(
    sequence: Sequence,
    padding_side: str = "right",
    max_length: int = 512,
    padding_element: int = 0,
):
    if max_length == len(sequence):
        return sequence
    if isinstance(sequence[0], (tuple, list)):
        assert len(sequence[0]) == len(padding_element)
    if padding_side == "right":
        return sequence + [padding_element] * (max_length - len(sequence))
    else:
        return [padding_element] * (max_length - len(sequence)) + sequence


def pad_sequences(
    sequences: List[Union[Sequence, "torch.Tensor"]],
    padding_side: str = "right",
    max_length: int = 512,
    padding_element: int = 0,
):
    import torch

    return torch.tensor(
        [
            pad_sequence(
                sequence.tolist() if isinstance(sequence, torch.Tensor) else sequence,
                padding_side=padding_side,
                max_length=max_length,
                padding_element=padding_element,
            )
            for sequence in sequences
        ]
    )


def is_not_sequence(sequence: Union[Sequence, "torch.Tensor"]) -> bool:
    import numpy as np
    import torch

    if isinstance(sequence, torch.Tensor):
        return len(sequence.shape) > 0
    if isinstance(sequence, np.ndarray):
        return len(sequence.shape) > 0
    elif isinstance(sequence, Sequence):
        return len(sequence) == 0
    else:
        return True


def pad_samples(
    samples_dict: Union[dict, List[dict]],
    data_padding_dict: dict,
    padding_side: str,
    max_length: int,
):
    for k, v in samples_dict.items():
        if k not in data_padding_dict:
            continue
        if is_not_sequence(samples_dict[k][0]):
            continue

        samples_dict[k] = pad_sequences(
            samples_dict[k],
            padding_side,
            max_length,
            data_padding_dict[k],
        )

    return samples_dict


def subfinder(words_list, answer_list):
    if len(answer_list) == 0:
        return None, 0, 0
    matches = []
    start_indices = []
    end_indices = []
    for idx, i in enumerate(range(len(words_list))):
        if (
            words_list[i] == answer_list[0]
            and words_list[i : i + len(answer_list)] == answer_list
        ):
            matches.append(answer_list)
            start_indices.append(idx)
            end_indices.append(idx + len(answer_list) - 1)
    if matches:
        return matches[0], start_indices[0], end_indices[0]
    else:
        return None, 0, 0


def normalize_bbox_list(word_bboxes_per_batch):
    import numpy as np

    # if the bboxes are with 0 to 1 normalize to 0-1000 format
    updated_word_bboxes = []
    for word_bboxes_per_sample in word_bboxes_per_batch:
        if len(word_bboxes_per_sample) == 0:
            updated_word_bboxes.append([])
            continue
        word_bboxes_per_sample = np.array(word_bboxes_per_sample)
        if word_bboxes_per_sample.min() > 0 and word_bboxes_per_sample.max() < 1:
            word_bboxes_per_sample = (word_bboxes_per_sample * 1000).astype(int)
        updated_word_bboxes.append(word_bboxes_per_sample.tolist())
    return updated_word_bboxes


def disk(radius, alias_blur=0.1, dtype=None):
    """
    Creats the aliased kernel disk over image.
    """

    import cv2
    import numpy as np

    dtype = dtype or np.float32

    if radius <= 8:
        L = np.arange(-8, 8 + 1)
        ksize = (3, 3)
    else:
        L = np.arange(-radius, radius + 1)
        ksize = (5, 5)
    X, Y = np.meshgrid(L, L)
    aliased_disk = np.array((X**2 + Y**2) <= radius**2, dtype=dtype)
    aliased_disk /= np.sum(aliased_disk)

    # supersample disk to antialias
    return cv2.GaussianBlur(aliased_disk, ksize=ksize, sigmaX=alias_blur)


def clipped_zoom(self, image: ArrayLike, zoom_factor: float):
    """
    Applies clipped zoom over image.
    """

    import numpy as np
    from scipy.ndimage import zoom as scizoom

    h = image.shape[0]
    w = image.shape[1]
    # ceil crop height(= crop width)
    ch = int(np.ceil(h / float(zoom_factor)))
    cw = int(np.ceil(w / float(zoom_factor)))
    top = (h - ch) // 2
    left = (w - cw) // 2
    img = scizoom(
        image[top : top + ch, left : left + cw],
        (self.zoom_factor, self.zoom_factor),
        order=1,
    )
    # trim off any extra pixels
    trim_top = (img.shape[0] - h) // 2
    trim_left = (img.shape[1] - w) // 2

    return img[trim_top : trim_top + h, trim_left : trim_left + w]


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
