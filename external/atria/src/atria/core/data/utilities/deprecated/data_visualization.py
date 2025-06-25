from __future__ import annotations

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision.utils import make_grid

from atria.core.constants import DataKeys
from atria.core.utilities.logging import get_logger

logger = get_logger(__name__)


def _equal_sized_batch(images):
    tensor_shapes = [x.shape for x in images]
    if tensor_shapes.count(tensor_shapes[0]) == len(
        tensor_shapes
    ):  # all tensors have equal shape, we make a batch tensor
        return True
    return False


def _visualize_images(batch, nmax=16, concatenate_images=True):
    if concatenate_images:
        image_grid = make_grid((batch[:nmax]), nrow=4)
        _, h, w = image_grid.shape
        fig, ax = plt.subplots(figsize=(10, 10 * h / w))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(image_grid.permute(1, 2, 0))
        plt.show()
        plt.close(fig)
    else:
        for x in batch:
            _, h, w = x.shape
            fig, ax = plt.subplots(figsize=(10, 10 * h / w))
            ax.set_xticks([])
            ax.set_yticks([])
            ax.imshow(x.permute(1, 2, 0))
            plt.show()
            plt.close(fig)
    return image_grid


# def _print_batch_info(batch):
#     logger.info("Batch information: ")
#     # if tokenizer is not None:
#     #     tokenizer = (
#     #         tokenizer.tokenizer
#     #         if isinstance(tokenizer, HuggingfaceTokenizer)
#     #         else tokenizer
#     #     )
#     #     logger.info(f"Tokenizer: {tokenizer}")

#     if isinstance(batch, dict):
#         for key, value in batch.items():
#             # if tokenizer is not None and key in [DataKeys.TOKEN_IDS]:
#             #     logger.info(
#             #         f"Batch element={key}, shape={len(value)}, type={type(value[0])}\nExample: {value[0]}"
#             #     )
#             #     logger.info(f"Converted string={tokenizer.decode(token_ids=value[0])}")

#             if isinstance(value, (torch.Tensor, np.ndarray)):
#                 logger.info(
#                     f"Batch element={key}, shape={value.shape}, type={value.dtype}\nExample: {value[0]}"
#                 )
#             elif isinstance(value, list):
#                 if isinstance(value[0], (torch.Tensor, np.ndarray)):
#                     logger.info(
#                         f"Batch element={key}, shape={value[0].shape}, type={value[0].dtype}\nExample: {value[0]}"
#                     )
#                 else:
#                     logger.info(
#                         f"Batch element={key}, shape={len(value)}, type={type(value[0])}\nExample: {value[0]}"
#                     )
#             else:
#                 logger.info(
#                     f"Batch element={key}, type={type(value)}\nExample: {value}"
#                 )
#     elif isinstance(batch, list):
#         sample = batch[0]
#         for key, value in sample.items():
#             if isinstance(value, (torch.Tensor, np.ndarray)):
#                 logger.info(
#                     f"Batch element={key}, shape={value.shape}, type={value.dtype}\nExample: {value[0]}"
#                 )
#             elif isinstance(value, list):
#                 if isinstance(value[0], (torch.Tensor, np.ndarray)):
#                     logger.info(
#                         f"Batch element={key}, shape={value[0].shape}, type={value[0].dtype}\nExample: {value[0]}"
#                     )
#                 else:
#                     logger.info(
#                         f"Batch element={key}, shape={len(value)}, type={type(value[0])}\nExample: {value[0]}"
#                     )
#             else:
#                 logger.info(
#                     f"Batch element={key}, type={type(value)}\nExample: {value}"
#                 )


def _draw_instances(image, instances, labels):
    from detectron2.utils.visualizer import ColorMode, Visualizer

    segmentations = instances.segmentation if instances.has("segmentation") else None
    gt_boxes = instances.gt_boxes if instances.has("gt_boxes") else None
    gt_classes = instances.gt_classes if instances.has("gt_classes") else None

    # convert labels
    gt_classes = [x.item() for x in gt_classes]

    # assign random colors to each class
    colors = {}
    for label in labels:
        colors[label] = tuple(np.random.randint(0, 256, 3))

    v = Visualizer(
        image,
        scale=1.0,
        instance_mode=ColorMode.SEGMENTATION,
    )
    result = v.overlay_instances(boxes=gt_boxes, masks=segmentations, labels=gt_classes)
    return np.array(result.get_image())


def _visualize_batch(batch, dataset_metadata: "DatasetMetadata"):
    draw_batch = []
    draw_batch_gt = []
    batch = [dict(zip(batch, t)) for t in zip(*batch.values())]

    if len(batch) > 16:
        logger.warning(
            "Showing only first 4 images in the batch as high-resolution images may take too much memory..."
        )
        batch = batch[:16]

    data_labels = dataset_metadata.labels

    for sample in batch:
        image = sample[DataKeys.IMAGE].permute(1, 2, 0).cpu().numpy()
        if DataKeys.GT_IMAGE in sample:
            gt_image = sample[DataKeys.GT_IMAGE].permute(1, 2, 0).cpu().numpy()
            gt_image = np.ascontiguousarray(gt_image)
            draw_batch_gt.append(torch.from_numpy(gt_image).permute(2, 0, 1))
        image = np.ascontiguousarray(image)
        h, w, c = image.shape

        if DataKeys.CAPTION in sample:
            p1 = (w // 4, 20)  # opencv point is (x, y) not (y, x)
            cv2.putText(
                image,
                text=sample[DataKeys.CAPTION],
                org=p1,
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1,
                color=(0, 0, 255),
                thickness=2,
            )
        if DataKeys.QUESTIONS in sample:
            p1 = (w // 4, 20)  # opencv point is (x, y) not (y, x)
            cv2.putText(
                image,
                text=(
                    (
                        sample[DataKeys.QUESTIONS]
                        + ":"
                        + " ".join(sample[DataKeys.GOLD_ANSWERS])
                    )
                    if DataKeys.GOLD_ANSWERS in sample
                    else sample[DataKeys.QUESTIONS]
                ),
                org=p1,
                fontFace=cv2.FONT_HERSHEY_PLAIN,
                fontScale=1,
                color=(0, 0, 255),
                thickness=2,
            )

        try:
            if DataKeys.WORDS in sample and DataKeys.WORD_BBOXES in sample:
                logger.info("Drawing boxes with word boxes on image...")
                last_box = None
                last_bbox_center = None

                for idx, (_, box) in enumerate(
                    zip(sample[DataKeys.WORDS], sample[DataKeys.WORD_BBOXES])
                ):  # each box is [x1,y1,x2,y2] normalized
                    if last_box is not None and all(last_box == box):
                        continue

                    if box[0] <= 0 or box[1] <= 0 or box[2] <= 0 or box[3] <= 0:
                        continue

                    normalize = True
                    if box[0] < 1 or box[1] < 1 or box[2] < 1 or box[3] < 1:
                        normalize = False

                    if normalize:
                        p1 = (int(box[0] / 1000.0 * w), int(box[1] / 1000.0 * h))
                        p2 = (int(box[2] / 1000.0 * w), int(box[3] / 1000.0 * h))
                    else:
                        p1 = (int(box[0] * w), int(box[1] * h))
                        p2 = (int(box[2] * w), int(box[3] * h))

                    color = (255, 0, 0)
                    lw = 1
                    cv2.rectangle(image, p1, p2, color, lw)
                    if last_bbox_center is not None:
                        bbox_center = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
                        cv2.arrowedLine(
                            image,
                            tuple(int(x) for x in last_bbox_center),
                            tuple(int(x) for x in bbox_center),
                            (255, 0, 0),
                            1,
                            tipLength=0.01,
                        )
                    last_bbox_center = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
                    # cv2.putText(
                    #     image,
                    #     text=str(tokens),
                    #     org=p1,
                    #     fontFace=cv2.FONT_HERSHEY_PLAIN,
                    #     fontScale=1,
                    #     color=(0, 0, 255),
                    #     thickness=1,
                    # )
                    last_box = box

            # if prev_bbox_center is not None:
            #     bbox_center = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
            #     print(prev_bbox_center, bbox_center)
            #     cv2.arrowedLine(
            #         image, prev_bbox_center, bbox_center, (255, 0, 0), 2
            #     )
            # prev_bbox_center = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2

            if DataKeys.TOKEN_IDS in sample and DataKeys.TOKEN_BBOXES in sample:
                logger.info("Drawing boxes with only first token element on image...")
                last_box = None
                last_bbox_center = None

                for idx, (_, box) in enumerate(
                    zip(sample[DataKeys.TOKEN_IDS], sample[DataKeys.TOKEN_BBOXES])
                ):  # each box is [x1,y1,x2,y2] normalized
                    if last_box is not None and all(last_box == box):
                        continue

                    if box[0] <= 0 or box[1] <= 0 or box[2] <= 0 or box[3] <= 0:
                        continue

                    normalize = True
                    if box[0] < 1 or box[1] < 1 or box[2] < 1 or box[3] < 1:
                        normalize = False

                    if normalize:
                        p1 = (int(box[0] / 1000.0 * w), int(box[1] / 1000.0 * h))
                        p2 = (int(box[2] / 1000.0 * w), int(box[3] / 1000.0 * h))
                    else:
                        p1 = (int(box[0] * w), int(box[1] * h))
                        p2 = (int(box[2] * w), int(box[3] * h))

                    if (
                        DataKeys.START_TOKEN_IDX in sample
                        and sample[DataKeys.START_TOKEN_IDX] != 0
                        and idx >= sample[DataKeys.START_TOKEN_IDX].item()
                        and idx <= sample[DataKeys.END_TOKEN_IDX].item()
                    ):
                        color = (0, 0, 255)
                        lw = 2
                    else:
                        color = (255, 0, 0)
                        lw = 1
                    cv2.rectangle(image, p1, p2, color, lw)
                    if last_bbox_center is not None:
                        bbox_center = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
                        cv2.arrowedLine(
                            image,
                            tuple(int(x) for x in last_bbox_center),
                            tuple(int(x) for x in bbox_center),
                            (255, 0, 0),
                            1,
                            tipLength=0.01,
                        )
                    last_bbox_center = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
                    # cv2.putText(
                    #     image,
                    #     text=str(tokens),
                    #     org=p1,
                    #     fontFace=cv2.FONT_HERSHEY_PLAIN,
                    #     fontScale=1,
                    #     color=(0, 0, 255),
                    #     thickness=1,
                    # )
                    last_box = box

            if DataKeys.GT_INSTANCES in sample:
                image = _draw_instances(
                    image, sample[DataKeys.GT_INSTANCES], data_labels
                )

            draw_batch.append(torch.from_numpy(image).permute(2, 0, 1))
        except Exception as e:
            logger.warning(f"Exception in drawing boxes. Skipping... {e}")

    # draw images
    if len(draw_batch) > 0:
        _visualize_images(draw_batch, concatenate_images=_equal_sized_batch(draw_batch))
    if len(draw_batch_gt) > 0:
        _visualize_images(
            draw_batch_gt,
            concatenate_images=_equal_sized_batch(draw_batch_gt),
        )
