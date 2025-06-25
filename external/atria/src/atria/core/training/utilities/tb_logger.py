import numbers
import warnings
from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
from ignite.contrib.handlers.tensorboard_logger import OutputHandler, TensorboardLogger
from ignite.engine import Engine, EventEnum

from atria.core.constants import MetricKeys


class CustomOutputHandler(OutputHandler):
    def __init__(
        self,
        tag: str,
        metric_names: Optional[List[str]] = None,
        output_transform: Optional[Callable] = None,
        global_step_transform: Optional[Callable] = None,
        state_attributes: Optional[List[str]] = None,
        class_labels: Optional[List[str]] = None,
    ):
        super().__init__(
            tag, metric_names, output_transform, global_step_transform, state_attributes
        )
        self._class_labels = class_labels

    def _setup_output_metrics_state_attrs(
        self,
        engine: Engine,
        log_text: Optional[bool] = False,
        key_tuple: Optional[bool] = True,
    ) -> Dict[Any, Any]:
        """Helper method to setup metrics and state attributes to log"""
        metrics_state_attrs = OrderedDict()
        if self.metric_names is not None:
            if isinstance(self.metric_names, str) and self.metric_names == "all":
                metrics_state_attrs = OrderedDict(engine.state.metrics)
            else:
                for name in self.metric_names:
                    if name not in engine.state.metrics:
                        warnings.warn(
                            f"Provided metric name '{name}' is missing "
                            f"in engine's state metrics: {list(engine.state.metrics.keys())}"
                        )
                        continue
                    metrics_state_attrs[name] = engine.state.metrics[name]

        if self.output_transform is not None:
            output_dict = self.output_transform(engine.state.output)

            if not isinstance(output_dict, dict):
                output_dict = {"output": output_dict}

            metrics_state_attrs.update(output_dict)

        if self.state_attributes is not None:
            metrics_state_attrs.update(
                {
                    name: getattr(engine.state, name, None)
                    for name in self.state_attributes
                }
            )

        # type: Dict[Any, Union[str, float, numbers.Number]]
        metrics_state_attrs_dict = OrderedDict()

        def key_tuple_tf(tag: str, name: str, *args: str) -> Tuple[str, ...]:
            return (tag, name) + args

        def key_str_tf(tag: str, name: str, *args: str) -> str:
            return "/".join((tag, name) + args)

        key_tf = key_tuple_tf if key_tuple else key_str_tf

        for name, value in metrics_state_attrs.items():
            if isinstance(value, numbers.Number):
                metrics_state_attrs_dict[key_tf(self.tag, name)] = value
            elif isinstance(value, torch.Tensor) and value.ndimension() == 0:
                metrics_state_attrs_dict[key_tf(self.tag, name)] = value.item()
            elif isinstance(value, torch.Tensor) and value.ndimension() == 1:
                for i, v in enumerate(value):
                    metrics_state_attrs_dict[key_tf(self.tag, name, str(i))] = v.item()
            else:
                if isinstance(value, str) and log_text:
                    metrics_state_attrs_dict[key_tf(self.tag, name)] = value
                elif MetricKeys.CONFUSION_MATRIX in name:
                    metrics_state_attrs_dict[key_tf(self.tag, name)] = value
                else:
                    warnings.warn(
                        f"Logger output_handler can not log metrics value type {type(value)}"
                    )
        return metrics_state_attrs_dict

    def __call__(
        self,
        engine: Engine,
        logger: TensorboardLogger,
        event_name: Union[str, EventEnum],
    ) -> None:
        if not isinstance(logger, TensorboardLogger):
            raise RuntimeError(
                "Handler 'OutputHandler' works only with TensorboardLogger"
            )

        metrics = self._setup_output_metrics_state_attrs(engine, key_tuple=False)

        global_step = self.global_step_transform(engine, event_name)  # type: ignore[misc]
        if not isinstance(global_step, int):
            raise TypeError(
                f"global_step must be int, got {type(global_step)}."
                " Please check the output of global_step_transform."
            )

        for key, value in metrics.items():
            # if we're in test loop we also draw high resolution confusion matrix
            # if MetricKeys.CONFUSION_MATRIX in key:
            #     if TrainingStage.test in key:
            #         img = plot_confusion_matrix(value, self._class_labels, accuracy=accuracy, dpi=300)
            #         logger.writer.add_image(key, img.transpose(2, 0, 1))
            #     else:
            #         img = plot_confusion_matrix(value, self._class_labels, accuracy=accuracy)
            #         logger.writer.add_image(key, img.transpose(2, 0, 1), global_step=global_step)
            # else:
            if MetricKeys.CONFUSION_MATRIX not in key:
                logger.writer.add_scalar(key, value, global_step)


class FusionTensorboardLogger(TensorboardLogger):
    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)

    def _create_output_handler(self, *args: Any, **kwargs: Any) -> "OutputHandler":
        return CustomOutputHandler(*args, **kwargs)
