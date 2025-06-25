from __future__ import annotations

import logging
import logging as log
from collections import OrderedDict
from io import StringIO
from typing import Any, Callable, List, Optional, Union

from ignite.engine import Engine, Events
from ignite.engine.events import CallableEventWithFilter, RemovableEventHandle
from ignite.handlers.tqdm_logger import ProgressBar, _OutputHandler


class TqdmLogFormatter(object):
    def __init__(self, logger):
        self._logger = logger

    def __enter__(self):
        self.__original_formatters = list()

        for handler in self._logger.handlers:
            self.__original_formatters.append(handler.formatter)

            handler.terminator = ""
            formatter = log.Formatter("%(message)s")
            handler.setFormatter(formatter)

        return self._logger

    def __exit__(self, exc_type, exc_value, exc_traceback):
        for handler, formatter in zip(
            self._logger.handlers, self.__original_formatters
        ):
            handler.terminator = "\n"
            handler.setFormatter(formatter)


class TqdmToLogger(StringIO):
    def __init__(self, logger):
        super().__init__()

        self._logger = logger

    def write(self, buffer):
        with TqdmLogFormatter(self._logger) as logger:
            logger.info(buffer)

    def flush(self):
        pass


class AtriaProgressBar(ProgressBar):
    def attach(  # type: ignore[override]
        self,
        engine: Engine,
        metric_names: Optional[Union[str, List[str]]] = None,
        output_transform: Optional[Callable] = None,
        event_name: Union[Events, CallableEventWithFilter] = Events.ITERATION_COMPLETED,
        closing_event_name: Union[
            Events, CallableEventWithFilter
        ] = Events.EPOCH_COMPLETED,
        state_attributes: Optional[List[str]] = None,
        optimizers: Optional[dict] = None,
        optimizer_params: Optional[List[str]] = ["lr"],
        close_on_write: bool = False,
    ) -> None:
        desc = self.tqdm_kwargs.get("desc", None)

        if event_name not in engine._allowed_events:
            raise ValueError(
                f"Logging event {event_name.name} is not in allowed events for this engine"
            )

        if isinstance(closing_event_name, CallableEventWithFilter):
            if closing_event_name.filter is not None:
                raise ValueError("Closing Event should not be a filtered event")

        if not close_on_write and not self._compare_lt(event_name, closing_event_name):
            raise ValueError(
                f"Logging event {event_name} should be called before closing event {closing_event_name}"
            )

        log_handler = _FusionOutputHandler(
            desc,
            metric_names,
            output_transform,
            closing_event_name=closing_event_name,
            state_attributes=state_attributes,
            optimizers=optimizers,
            optimizer_params=optimizer_params,
            close_on_write=close_on_write,
        )

        super(ProgressBar, self).attach(engine, log_handler, event_name)
        engine.add_event_handler(closing_event_name, self._close)

    def attach_opt_params_handler(
        self, engine: Engine, event_name: Union[str, Events], *args: Any, **kwargs: Any
    ) -> RemovableEventHandle:
        """Intentionally empty"""

    def _create_output_handler(self, *args: Any, **kwargs: Any) -> "_OutputHandler":
        return _OutputHandler(*args, **kwargs)

    def _create_opt_params_handler(self, *args: Any, **kwargs: Any) -> Callable:
        """Intentionally empty"""


class _FusionOutputHandler(_OutputHandler):
    def __init__(
        self,
        description: str,
        metric_names: Optional[Union[str, List[str]]] = None,
        output_transform: Optional[Callable] = None,
        closing_event_name: Union[
            Events, CallableEventWithFilter
        ] = Events.EPOCH_COMPLETED,
        state_attributes: Optional[List[str]] = None,
        optimizers: Optional[dict] = None,
        optimizer_params: Optional[List[str]] = ["lr"],
        post_fix_max_length: int = 3,
        close_on_write: bool = False,
    ):
        super().__init__(
            description,
            metric_names,
            output_transform,
            closing_event_name,
            state_attributes,
        )
        self.optimizers = optimizers
        self.optimizer_params = optimizer_params
        self.post_fix_max_length = post_fix_max_length
        self.close_on_write = close_on_write

    def __call__(
        self, engine: Engine, logger: ProgressBar, event_name: Union[str, Events]
    ) -> None:
        pbar_total = self.get_max_number_events(event_name, engine)
        if logger.pbar is None:
            logger._reset(pbar_total=pbar_total)

        max_epochs = engine.state.max_epochs
        default_desc = "Iteration" if max_epochs == 1 else "Epoch"

        desc = self.tag or default_desc
        if not self.close_on_write:
            max_num_of_closing_events = self.get_max_number_events(
                self.closing_event_name, engine
            )
            if max_num_of_closing_events and max_num_of_closing_events > 1:
                global_step = engine.state.get_event_attrib_value(
                    self.closing_event_name
                )
                desc += f" [{global_step}/{max_num_of_closing_events}]"
        logger.pbar.set_description(desc)  # type: ignore[attr-defined]

        rendered_metrics = self._setup_output_metrics_state_attrs(engine, log_text=True)
        metrics = OrderedDict()
        for key, value in rendered_metrics.items():
            key = "_".join(key[1:])  # tqdm has tag as description

            metrics[key] = value

        # this is incredibly expensive don't do this
        # if self.optimizers is not None:
        #     for k, opt in self.optimizers.items():
        #         for param_name in self.optimizer_params:
        #             min_param = 10.0
        #             max_param = 0.0
        #             for pg in opt.param_groups:
        #                 min_param = min(min_param, pg[param_name])
        #                 max_param = max(max_param, pg[param_name])
        #             if (max_param - min_param) < 1e-6:
        #                 param = f"opt/{k}/{param_name}"
        #                 metrics[param] = float(max_param)
        #             else:
        #                 min_param_name = f"opt/{k}/min/{param_name}"
        #                 max_param_name = f"opt/{k}/max/{param_name}"
        #                 metrics[min_param_name] = float(min_param)
        #                 metrics[max_param_name] = float(max_param)
        # simply get the metric
        if self.optimizers is not None:
            for k, opt in self.optimizers.items():
                for param_name in self.optimizer_params:
                    metrics[f"opt/{k}/{param_name}"] = float(
                        opt.param_groups[-1][param_name]
                    )

        if hasattr(engine.state, "ema_momentum"):
            metrics["ema/mom"] = engine.state.ema_momentum

        renamed_metrics = {}
        for name, value in metrics.items():
            # make short name
            new_name = "/".join(
                [s[: self.post_fix_max_length] for s in name.split("/")]
            )
            renamed_metrics[new_name] = value

        if renamed_metrics:
            logger.pbar.set_postfix(renamed_metrics)  # type: ignore[attr-defined]

        if not self.close_on_write:
            global_step = engine.state.get_event_attrib_value(event_name)
            if pbar_total is not None:
                global_step = (global_step - 1) % pbar_total + 1
            logger.pbar.update(global_step - logger.pbar.n)  # type: ignore[attr-defined]
