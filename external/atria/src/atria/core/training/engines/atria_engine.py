import inspect
from functools import partial
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Union

import ignite.distributed as idist
import torch
from atria.core.constants import DataKeys
from atria.core.models.task_modules.atria_task_module import AtriaTaskModule
from atria.core.training.configs.logging_config import LoggingConfig
from atria.core.training.engines.engine_steps.base import BaseEngineStep
from atria.core.training.engines.utilities import FixedBatchIterator
from atria.core.training.utilities.progress_bar import TqdmToLogger
from atria.core.utilities.logging import get_logger
from atria.core.utilities.print_utils import _print_batch_info, _print_output_info
from ignite.engine import Engine, Events, State
from ignite.handlers import ProgressBar, TensorboardLogger
from ignite.metrics import Metric

logger = get_logger(__name__)


class AtriaEngine:
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]],
        dataset_cache_dir: Optional[Union[str, Path]],
        task_module: AtriaTaskModule,
        dataloader: Iterable,
        engine_step: partial[BaseEngineStep],
        device: Union[str, torch.device],
        tb_logger: Optional["TensorboardLogger"] = None,
        max_epochs: Optional[int] = None,
        epoch_length: Optional[int] = None,
        outputs_to_running_avg: Optional[List[str]] = None,
        logging: LoggingConfig = LoggingConfig(),
        metrics: Optional[Dict[str, partial[Metric]]] = None,
        metric_logging_prefix: Optional[str] = None,
        sync_batchnorm: bool = False,
        test_run: bool = False,
        use_fixed_batch_iterator: bool = False,
        checkpoints_dir: str = "checkpoints",
    ):
        self._output_dir = output_dir
        self._dataset_cache_dir = dataset_cache_dir
        self._task_module = task_module
        self._dataloader = dataloader
        self._tb_logger = tb_logger
        self._engine_step = engine_step
        self._device = torch.device(device)
        self._max_epochs = max_epochs
        self._epoch_length = epoch_length
        self._outputs_to_running_avg = (
            outputs_to_running_avg
            if outputs_to_running_avg is not None
            else [DataKeys.LOSS]
        )
        self._logging = logging if logging is not None else LoggingConfig()
        self._metrics = metrics
        self._metric_logging_prefix = metric_logging_prefix
        self._sync_batchnorm = sync_batchnorm
        self._test_run = test_run
        self._use_fixed_batch_iterator = use_fixed_batch_iterator
        self._checkpoints_dir = checkpoints_dir
        self._progress_bar = None
        self._event_handlers = []

        self._initialize_components()

    @property
    def batches_per_epoch(self) -> int:
        return len(self._dataloader)

    @property
    def steps_per_epoch(self) -> int:
        return self.batches_per_epoch

    @property
    def total_update_steps(self) -> int:
        return self.steps_per_epoch * self._max_epochs

    def _initialize_components(self):
        # initialize the engine step
        self._engine_step = self._engine_step(
            task_module=self._task_module, device=self._device, test_run=self._test_run
        )

        # initialize the progress bar
        self._progress_bar = ProgressBar(
            desc=f"Stage [{self._engine_step.stage}]",
            persist=True,
            file=TqdmToLogger(
                get_logger(__name__ + ".tqdm")
            ),  # main logger causes problems here
        )

        # attach the progress bar to task module
        self._task_module.attach_progress_bar(self._progress_bar)

    def _initialize_engine(self) -> Engine:
        engine = Engine(self._engine_step)
        engine.logger.propagate = False
        return engine

    def _configure_engine(self, engine: Engine):
        # the order of attached handlers matters, if epoch-based metrics are attached before
        # then they get logged correctly, as compute() is called on these metrics at EPOCH_COMPLETED
        # progress bar should be attached last so this gets logged correctly
        self._configure_test_run(engine=engine)
        self._configure_metrics(engine=engine)
        self._configure_running_avg_logging(engine=engine)
        self._configure_progress_bar(engine=engine)
        self._configure_tb_logger(engine=engine)
        self._attach_event_handlers(engine=engine)

    def _configure_gpu_stats_callback(self, engine: Engine):
        from atria.core.training.engines.utilities import (
            _attach_gpu_stats_callback_to_engine,
        )

        if self._logging.log_gpu_stats:
            _attach_gpu_stats_callback_to_engine(engine, self._logging.logging_steps)

    def _configure_time_profiler(self, engine: Engine):

        from atria.core.training.engines.utilities import (
            _attach_time_profiler_to_engine,
        )

        if self._logging.profile_time:
            _attach_time_profiler_to_engine(engine)

    def _configure_metrics(self, engine: Engine) -> None:

        from atria.core.training.engines.utilities import _attach_metrics_to_engine

        if self._metrics is not None:

            def initialize_metric(metric):
                possible_args = inspect.signature(metric).parameters

                kwargs = {}
                if "dataset_cache_dir" in possible_args:
                    kwargs["dataset_cache_dir"] = self._dataset_cache_dir
                if "stage" in possible_args:
                    kwargs["stage"] = self._engine_step.stage
                if "device" in possible_args:
                    kwargs["device"] = self._device
                return metric(**kwargs)

            logger.info(
                f"Attaching metrics {self._metrics} to engine [{self.__class__.__name__}]"
            )
            _attach_metrics_to_engine(
                engine=engine,
                metrics={
                    key: initialize_metric(metric)
                    for key, metric in self._metrics.items()
                },
                prefix=self._metric_logging_prefix,
                stage=self._engine_step.stage,
            )

    def _configure_running_avg_logging(self, engine: Engine) -> None:

        from atria.core.training.engines.utilities import (
            _attach_output_logging_to_engine,
        )

        _attach_output_logging_to_engine(
            engine=engine,
            stage=self._engine_step.stage,
            outputs_to_running_avg=self._outputs_to_running_avg,
        )

    def _configure_progress_bar(self, engine: Engine) -> None:
        from atria.core.training.engines.utilities import _log_eval_metrics

        if idist.get_rank() == 0:
            self._progress_bar.attach(
                engine,
                event_name=Events.ITERATION_COMPLETED(every=self._logging.refresh_rate),
                metric_names="all",
            )

            @engine.on(Events.EPOCH_COMPLETED)
            def progress_on_epoch_completed(engine: Engine) -> None:
                _log_eval_metrics(
                    logger=logger,
                    epoch=engine.state.epoch,
                    elapsed=engine.state.times["EPOCH_COMPLETED"],
                    tag=self._engine_step.stage,
                    metrics=engine.state.metrics,
                )

    def _configure_tb_logger(self, engine: Engine):
        from ignite.engine import Events

        if (
            idist.get_rank() == 0
            and self._tb_logger is not None
            and self._logging.log_to_tb
        ):
            # attach tb logger to validation engine
            self._tb_logger.attach_output_handler(
                engine,
                event_name=Events.EPOCH_COMPLETED,
                metric_names="all",
                tag="epoch",
            )

    def _configure_test_run(self, engine: Engine):
        from ignite.engine import Events

        if self._test_run:
            logger.warning(
                f"This is a test run of engine [{self.__class__.__name__}]. "
                "Only a single engine step will be executed."
            )

            def terminate_on_iteration_complete(
                engine,
            ):  # this is necessary for fldp to work with correct privacy accounting
                logger.info("Terminating engine as test_run=True")
                engine.terminate()

            def print_iteration_started_info(engine):
                logger.debug(
                    f"Batch input received for engine [{self.__class__.__name__}]:"
                )
                _print_batch_info(engine.state.batch)

            def print_iteration_completed_info(engine):
                logger.debug(f"Output received for engine [{self.__class__.__name__}]:")
                _print_output_info(engine.state.output)

            engine.add_event_handler(
                Events.ITERATION_COMPLETED, terminate_on_iteration_complete
            )
            engine.add_event_handler(
                Events.ITERATION_STARTED, print_iteration_started_info
            )
            engine.add_event_handler(
                Events.ITERATION_COMPLETED, print_iteration_completed_info
            )

    def _attach_event_handlers(self, engine: Engine):
        for event, handler in self._event_handlers:
            engine.add_event_handler(event, handler)

    def add_event_handler(self, event: Events, handler: Callable):
        self._event_handlers.append((event, handler))

    def run(
        self,
    ) -> State:
        # move task module models to device
        self._task_module.to_device(self._device, sync_bn=self._sync_batchnorm)

        # initialize engine
        engine = self._initialize_engine()

        # configure engine
        self._configure_engine(engine)

        # run engine
        if self._output_dir is not None:
            logger.info(
                f"Running {self.__class__.__name__} engine with batch size [{self._dataloader.batch_size}] and output_dir: {self._output_dir}"
            )
        else:
            logger.info(f"Running engine {self.__class__.__name__}.")
        return engine.run(
            (
                FixedBatchIterator(self._dataloader, self._dataloader.batch_size)
                if self._use_fixed_batch_iterator
                else self._dataloader
            ),
            max_epochs=self._max_epochs,
            epoch_length=self._epoch_length,
        )
