import warnings
from typing import Sequence, Tuple

import torch
from ignite.metrics import Recall
from ignite.utils import to_onehot


class LayoutRecall(Recall):
    def _check_shape(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y, y_bbox = output

        # y_pred, y, y_bbox will have shapes
        if y_pred.ndimension() == 3 and y.ndimension() == 2:
            y_pred = y_pred.view(-1, y_pred.size(-1))
            y = y.view(-1)
            y_bbox = y_bbox.view(-1, 4)

        if not (
            y.ndimension() == y_pred.ndimension()
            or y.ndimension() + 1 == y_pred.ndimension()
        ):
            raise ValueError(
                "y must have shape of (batch_size, ...) and y_pred must have "
                "shape of (batch_size, num_categories, ...) or (batch_size, ...), "
                f"but given {y.shape} vs {y_pred.shape}."
            )

        y_shape = y.shape
        y_pred_shape: Tuple[int, ...] = y_pred.shape

        if y.ndimension() + 1 == y_pred.ndimension():
            y_pred_shape = (y_pred_shape[0],) + y_pred_shape[2:]

        if not (y_shape == y_pred_shape):
            raise ValueError("y and y_pred must have compatible shapes.")

        if self._is_multilabel and not (
            y.shape == y_pred.shape and y.ndimension() > 1 and y.shape[1] > 1
        ):
            raise ValueError(
                "y and y_pred must have same shape of (batch_size, num_categories, ...) and num_categories > 1."
            )

        if y_bbox.shape[1] != 4:
            raise ValueError(
                f"y_bbox must have shape of (batch_size, 4), but given {y_bbox.shape}."
            )

    def _check_type(self, output: Sequence[torch.Tensor]) -> None:
        y_pred, y, y_bbox = output

        # y_pred, y, y_bbox will have shapes
        if y_pred.ndimension() == 3 and y.ndimension() == 2:
            y_pred = y_pred.view(-1, y_pred.size(-1))
            y = y.view(-1)
            y_bbox = y_bbox.view(-1, 4)

        if y.ndimension() + 1 == y_pred.ndimension():
            num_classes = y_pred.shape[1]
            if num_classes == 1:
                update_type = "binary"
                self._check_binary_multilabel_cases((y_pred, y))
            else:
                update_type = "multiclass"
        elif y.ndimension() == y_pred.ndimension():
            self._check_binary_multilabel_cases((y_pred, y))

            if self._is_multilabel:
                update_type = "multilabel"
                num_classes = y_pred.shape[1]
            else:
                update_type = "binary"
                num_classes = 1
        else:
            raise RuntimeError(
                f"Invalid shapes of y (shape={y.shape}) and y_pred (shape={y_pred.shape}), check documentation."
                " for expected shapes of y and y_pred."
            )
        if self._type is None:
            self._type = update_type
            self._num_classes = num_classes
        else:
            if self._type != update_type:
                raise RuntimeError(
                    f"Input data type has changed from {self._type} to {update_type}."
                )
            if self._num_classes != num_classes:
                raise ValueError(
                    f"Input data number of classes has changed from {self._num_classes} to {num_classes}"
                )

        if self._type in ["binary", "multiclass"] and self._average == "samples":
            raise ValueError(
                "Argument average='samples' is incompatible with binary and multiclass input data."
            )

        if self._type == "multiclass" and y.dtype != torch.long:
            warnings.warn(
                "`y` should be of dtype long when entry type is multiclass",
                RuntimeWarning,
            )
        if (
            self._type == "binary"
            and self._average is not False
            and (y.dtype != torch.long or y_pred.dtype != torch.long)
        ):
            warnings.warn(
                "`y` and `y_pred` should be of dtype long when entry type is binary and average!=False",
                RuntimeWarning,
            )

        if y_bbox.dtype != torch.long and y_bbox.min() < 0 and y_bbox.max() > 1000:
            warnings.warn(
                "`y_bbox` should be of dtype long normalized between 0 to 1000",
                RuntimeWarning,
            )

    def _prepare_output(self, output: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        y_pred, y, y_bbox = output[0].detach(), output[1].detach(), output[2].detach()

        # y_pred, y, y_bbox will have shapes
        if y_pred.ndimension() == 3 and y.ndimension() == 2:
            y_pred = y_pred.view(-1, y_pred.size(-1))
            y = y.view(-1)
            y_bbox = y_bbox.view(-1, 4)

        # remove -100 labels if present in ground truth
        non_ignore_mask = y != -100
        y_pred = y_pred[non_ignore_mask]
        y = y[non_ignore_mask]
        y_bbox = y_bbox[non_ignore_mask]

        if self._type == "binary" or self._type == "multiclass":
            num_classes = 2 if self._type == "binary" else y_pred.size(1)
            if self._type == "multiclass" and y.max() + 1 > num_classes:
                raise ValueError(
                    f"y_pred contains fewer classes than y. Number of classes in the prediction is {num_classes}"
                    f" and an element in y has invalid class = {y.max().item() + 1}."
                )
            y = y.view(-1)
            if self._type == "binary" and self._average is False:
                y_pred = y_pred.view(-1)
            else:
                y = to_onehot(y.long(), num_classes=num_classes)
                indices = (
                    torch.argmax(y_pred, dim=1)
                    if self._type == "multiclass"
                    else y_pred.long()
                )
                y_pred = to_onehot(indices.view(-1), num_classes=num_classes)
        elif self._type == "multilabel":
            # if y, y_pred shape is (N, C, ...) -> (N * ..., C)
            num_labels = y_pred.size(1)
            y_pred = torch.transpose(y_pred, 1, -1).reshape(-1, num_labels)
            y = torch.transpose(y, 1, -1).reshape(-1, num_labels)

        # Convert from int cuda/cpu to double on self._device
        y_pred = y_pred.to(dtype=torch.float64, device=self._device)
        y = y.to(dtype=torch.float64, device=self._device)
        y_bbox = y_bbox.to(dtype=torch.float64, device=self._device)
        correct = y * y_pred

        # area of bboxes
        y_areas = ((y_bbox[:, 2] - y_bbox[:, 0]) * (y_bbox[:, 3] - y_bbox[:, 1]))[
            :, None
        ] / 1000.0

        return y_areas * y_pred, y_areas * y, y_areas * correct
