import copy
import math
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Union, cast

import ignite.distributed as idist
import torch
import webdataset as wds
from atria.core.constants import DEFAULT_OPTIMIZER_PARAMETERS_KEY
from atria.core.models.task_modules.atria_task_module import (
    AtriaTaskModule,
    TorchModelDict,
)
from atria.core.schedulers.typing import LRSchedulerType
from atria.core.training.configs.early_stopping_config import EarlyStoppingConfig
from atria.core.training.configs.gradient_config import GradientConfig
from atria.core.training.configs.logging_config import LoggingConfig
from atria.core.training.configs.model_checkpoint import ModelCheckpointConfig
from atria.core.training.configs.model_ema_config import ModelEmaConfig
from atria.core.training.configs.warmup_config import WarmupConfig
from atria.core.training.engines.atria_engine import AtriaEngine
from atria.core.training.engines.engine_steps.training import TrainingStep
from atria.core.training.engines.evaluation import ValidationEngine, VisualizationEngine
from atria.core.training.engines.events import OptimizerEvents
from atria.core.training.engines.utilities import (
    RUN_CONFIG_KEY,
    FixedBatchIterator,
    RunConfig,
    _print_optimizers_info,
    _print_schedulers_info,
)
from atria.core.training.handlers.ema_handler import AtriaEMAHandler
from atria.core.training.utilities.ddp_model_proxy import ModuleProxyWrapper
from atria.core.training.utilities.progress_bar import TqdmToLogger
from atria.core.utilities.common import _validate_partial_class
from atria.core.utilities.logging import get_logger
from ignite.engine import Engine, State
from ignite.handlers import ProgressBar, TensorboardLogger
from ignite.metrics import Metric
from torch.utils.data import DataLoader

logger = get_logger(__name__)


class IgniteTrainingEngine(Engine):
    def state_dict(self) -> OrderedDict:
        state_dict = super().state_dict()
        if hasattr(self.state, "optimizer_step"):
            state_dict["optimizer_step"] = self.state.optimizer_step
        return state_dict

    def load_state_dict(self, state_dict: Mapping) -> None:
        super().load_state_dict(state_dict)
        if hasattr(self.state, "optimizer_step"):
            self.state.optimizer_step = state_dict["optimizer_step"]


class TrainingEngine(AtriaEngine):
    def __init__(
        self,
        run_config: RunConfig,
        output_dir: Optional[Union[str, Path]],
        dataset_cache_dir: Optional[Union[str, Path]],
        task_module: AtriaTaskModule,
        dataloader: Union[DataLoader, wds.WebLoader],
        device: Union[str, torch.device],
        optimizers: Union[
            partial[torch.optim.Optimizer], Dict[str, partial[torch.optim.Optimizer]]
        ],
        engine_step: Optional[partial[TrainingStep]] = None,
        tb_logger: Optional["TensorboardLogger"] = None,
        max_epochs: int = 100,
        epoch_length: Optional[int] = None,
        outputs_to_running_avg: Optional[List[str]] = None,
        logging: LoggingConfig = LoggingConfig(),
        metrics: Optional[List[Metric]] = None,
        metric_logging_prefix: Optional[str] = None,
        sync_batchnorm: bool = False,
        test_run: bool = False,
        use_fixed_batch_iterator: bool = False,
        lr_schedulers: Optional[
            Union[partial[LRSchedulerType], Dict[str, partial[LRSchedulerType]]]
        ] = None,
        validation_engine: Optional[ValidationEngine] = None,
        visualization_engine: Optional[VisualizationEngine] = None,
        eval_training: Optional[bool] = False,
        stop_on_nan: bool = True,
        clear_cuda_cache: Optional[bool] = True,
        model_ema_config: Optional[ModelEmaConfig] = ModelEmaConfig(),
        warmup_config: WarmupConfig = WarmupConfig(),
        early_stopping: EarlyStoppingConfig = EarlyStoppingConfig(),
        model_checkpoint_config: ModelCheckpointConfig = ModelCheckpointConfig(),
        gradient_config: GradientConfig = GradientConfig(),
    ):
        _validate_partial_class(engine_step, TrainingStep, "engine_step")

        self._run_config = run_config
        self._optimizers = optimizers
        self._lr_schedulers = lr_schedulers
        self._validation_engine = validation_engine
        self._visualization_engine = visualization_engine
        self._eval_training = eval_training
        self._stop_on_nan = stop_on_nan
        self._clear_cuda_cache = clear_cuda_cache
        self._model_ema_config = model_ema_config
        self._warmup_config = warmup_config
        self._early_stopping = early_stopping
        self._model_checkpoint_config = model_checkpoint_config
        self._gradient_config = gradient_config
        self._ema_handler = None

        super().__init__(
            output_dir=output_dir,
            dataset_cache_dir=dataset_cache_dir,
            task_module=task_module,
            dataloader=dataloader,
            engine_step=engine_step,
            device=device,
            tb_logger=tb_logger,
            max_epochs=max_epochs,
            epoch_length=epoch_length,
            outputs_to_running_avg=outputs_to_running_avg,
            logging=logging,
            metrics=metrics,
            metric_logging_prefix=metric_logging_prefix,
            sync_batchnorm=sync_batchnorm,
            test_run=test_run,
            use_fixed_batch_iterator=use_fixed_batch_iterator,
            checkpoints_dir=model_checkpoint_config.dir,
        )

    @property
    def steps_per_epoch(self) -> int:
        return (
            self.batches_per_epoch // self._gradient_config.gradient_accumulation_steps
        )

    @property
    def total_warmup_steps(self):
        return (
            self._warmup_config.warmup_steps
            if self._warmup_config.warmup_steps > 0
            else math.ceil(self.total_update_steps * self._warmup_config.warmup_ratio)
        )

    def _initialize_engine(self) -> Engine:
        engine = IgniteTrainingEngine(self._engine_step)
        engine.logger.propagate = False
        return engine

    def _initialize_components(self):
        import inspect

        from ignite.handlers import LRScheduler

        # convert optimizers to dict if not already
        if not isinstance(self._optimizers, dict):
            self._optimizers = {DEFAULT_OPTIMIZER_PARAMETERS_KEY: self._optimizers}

        optimized_parameters_dict = self._task_module.optimized_parameters()
        assert len(optimized_parameters_dict) == len(self._optimizers), (
            "Number of optimizers must match the number of model parameter groups defined in the task_module. "
            f"Optimizers: {len(self._optimizers)} != Model parameter groups: {len(optimized_parameters_dict)}"
        )

        for k, opt in self._optimizers.items():
            _validate_partial_class(opt, torch.optim.Optimizer, f"[{k}] optimizer")
            if k not in optimized_parameters_dict.keys():
                raise ValueError(
                    f"Your optimizer configuration does not align the model optimizer "
                    f"parameter groups. {k} =/= {optimized_parameters_dict.keys()}"
                )

            # initialize the optimizers from partial with the model parameters
            self._optimizers[k] = idist.auto_optim(
                opt(params=optimized_parameters_dict[k])
            )

        # print information
        _print_optimizers_info(self._optimizers)

        # initialize lr schedulers partials
        if self._lr_schedulers is not None:
            # convert lr_schedulers to dict if not already
            if not isinstance(self._lr_schedulers, dict):
                self._lr_schedulers = {
                    DEFAULT_OPTIMIZER_PARAMETERS_KEY: self._lr_schedulers
                }

            for k, sch in self._lr_schedulers.items():
                _validate_partial_class(
                    sch,
                    (
                        torch.optim.lr_scheduler.LRScheduler,
                        LRScheduler,
                    ),
                    f"[{k}] lr_scheduler",
                )
                available_parameters_in_signature = inspect.signature(
                    sch.func
                ).parameters
                kwargs = {}
                for kwarg in [
                    "total_update_steps",
                    "total_warmup_steps",
                    "steps_per_epoch",
                ]:
                    if kwarg in available_parameters_in_signature.keys():
                        kwargs[kwarg] = getattr(self, kwarg)
                logger.info(
                    "Initializing lr scheduler: "
                    + sch.func.__name__
                    + "with kwargs: "
                    + str(kwargs)
                )
                self._lr_schedulers[k] = sch(optimizer=self._optimizers[k], **kwargs)

            # print information
            _print_schedulers_info(self._lr_schedulers)

        # initialize the engine step
        self._engine_step = self._engine_step(
            task_module=self._task_module,
            device=self._device,
            optimizers=self._optimizers,
            gradient_config=self._gradient_config,
            test_run=self._test_run,
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

        # initialize the metrics to the required device
        if self._metrics is not None:
            self._metrics = [metric(device=self._device) for metric in self._metrics]

    def _configure_train_sampler(self, engine: Engine):
        import ignite.distributed as idist
        from torch.utils.data.distributed import DistributedSampler

        if idist.get_world_size() > 1:
            from ignite.engine import Engine, Events

            train_sampler = self._dataloader.sampler
            if not isinstance(train_sampler, DistributedSampler):
                raise TypeError(
                    "Train sampler should be torch DistributedSampler and have `set_epoch` method"
                )

            @engine.on(Events.EPOCH_STARTED)
            def distrib_set_epoch(engine: Engine) -> None:
                cast(DistributedSampler, train_sampler).set_epoch(
                    engine.state.epoch - 1
                )

        else:
            # check whether the correct training sample is being used
            if self._dataloader.sampler is not None and isinstance(
                self._dataloader.sampler, DistributedSampler
            ):

                logger.warning(
                    "Argument train_sampler is a distributed sampler,"
                    " but either there is no distributed setting or world size is < 2. "
                    "Train sampler argument will be ignored",
                    UserWarning,
                )

    def _configure_nan_callback(self, engine: Engine):
        from atria.core.training.engines.utilities import _attach_nan_callback_to_engine

        if self._stop_on_nan:
            _attach_nan_callback_to_engine(engine)

    def _configure_cuda_cache_callback(self, engine: Engine):
        from atria.core.training.engines.utilities import (
            _attach_cuda_cache_callback_to_engine,
        )

        if self._clear_cuda_cache:
            _attach_cuda_cache_callback_to_engine(engine)

    def _configure_model_ema_callback(self, engine: Engine) -> None:
        from torchinfo import summary

        if self._model_ema_config.enabled:
            trainable_model = (
                self._task_module.torch_model.trainable_models
                if isinstance(self._task_module._torch_model, TorchModelDict)
                else self._task_module.torch_model
            )
            if isinstance(trainable_model, ModuleProxyWrapper):
                trainable_model = trainable_model.module

            self._ema_handler = AtriaEMAHandler(
                trainable_model,
                momentum=self._model_ema_config.momentum,
                momentum_warmup=self._model_ema_config.momentum_warmup,
                warmup_iters=self._model_ema_config.warmup_iters,
                handle_buffers="update",
            )

            logger.info(
                f"Attaching EMAHandler with following configuration: {self._model_ema_config}"
            )
            logger.info(f"Ema Model:")
            logger.info(summary(self._ema_handler.ema_model, verbose=0, depth=2))
            self._ema_handler.attach(
                engine,
                name="ema_momentum",
                event=OptimizerEvents.optimizer_step(
                    every=self._model_ema_config.update_every
                ),
            )

    def _configure_metrics(self, engine: Engine) -> None:
        if self._eval_training:
            super()._configure_metrics(engine)

    def _configure_schedulers(self, engine: Engine) -> None:
        from ignite.engine import Events
        from ignite.handlers import (
            LRScheduler,
            ParamScheduler,
            ReduceLROnPlateauScheduler,
            create_lr_scheduler_with_warmup,
        )
        from torch.optim.lr_scheduler import ExponentialLR, MultiStepLR, StepLR

        if self._lr_schedulers is None:
            return

        for k, inner_sch in self._lr_schedulers.items():
            if inner_sch is None:
                continue

            warmup_duration = self.total_warmup_steps
            if warmup_duration > 0:
                logger.info(
                    f"Initialized lr scheduler {inner_sch.__class__.__name__} with warmup. "
                )
                logger.info(f"Warmup ratio = {self._warmup_config.warmup_ratio}. ")
                logger.info(
                    f"Number of warmup steps = {warmup_duration}. This corresponds to optimizer updates, "
                    "not total batches in epoch and therefore its scaled by grad "
                    f"acummulation steps = {self._gradient_config.gradient_accumulation_steps}."
                )

                if isinstance(inner_sch, (StepLR, MultiStepLR)):
                    logger.info(
                        "Warmup updates are triggered per optimizer steps whereas the scheduler updates are triggered per epoch."
                    )
                    sch = create_lr_scheduler_with_warmup(
                        inner_sch,
                        warmup_start_value=0.0,
                        warmup_duration=warmup_duration,
                    )

                    # we want warmup on optimizer update steps and step_lr on epochs, so we create two events first for steps
                    # and then for epochs
                    # Trigger scheduler on iteration_started events before reaching warmup_duration
                    combined_events = OptimizerEvents.optimizer_step(
                        event_filter=lambda _, __: engine.state.optimizer_step
                        <= warmup_duration
                    )

                    # Trigger scheduler on epoch_started events after the warm-up. Epochs are 1-based, thus we do 1 +
                    combined_events |= Events.EPOCH_STARTED(
                        event_filter=lambda _, __: engine.state.epoch
                        > 1 + warmup_duration / self.steps_per_epoch
                    )

                    engine.add_event_handler(combined_events, sch)

                    # update scheduler in dict
                    self._lr_schedulers[k] = sch
                elif isinstance(inner_sch, ReduceLROnPlateauScheduler):
                    logger.info(
                        "Warmup updates are triggered per optimizer steps whereas the scheduler updates are triggered per validation step."
                    )
                    # we want warmup on steps and step_lr on epochs, so we create two events first for steps
                    # and then for epochs
                    sch = create_lr_scheduler_with_warmup(
                        inner_sch,
                        warmup_start_value=0.0,
                        warmup_duration=warmup_duration,
                    )
                    engine.add_event_handler(
                        OptimizerEvents.optimizer_step(
                            event_filter=lambda _, __: engine.state.optimizer_step
                            <= warmup_duration
                        ),
                        sch.schedulers[0],
                    )

                    # Trigger scheduler on epoch_started events after the warm-up. Epochs are 1-based, thus we do 1 +
                    combined_events = Events.COMPLETED | Events.COMPLETED(
                        event_filter=lambda _, __: engine.state.epoch
                        > 1 + warmup_duration / self.steps_per_epoch
                    )

                    if self._validation_engine is not None:
                        self._validation_engine.add_event_handler(
                            combined_events, inner_sch
                        )
                    else:
                        logger.warning(
                            "ReduceLROnPlateauScheduler metric is initialized with no validation engine attached. "
                        )
                    self._lr_schedulers[k] = sch
                else:
                    logger.info(
                        "Both warmup updates and the scheduler updates are triggered per optimizer step."
                    )
                    sch = create_lr_scheduler_with_warmup(
                        inner_sch,
                        warmup_start_value=0.0,
                        warmup_duration=warmup_duration,
                    )
                    engine.add_event_handler(OptimizerEvents.optimizer_step, sch)

                    # update scheduler in dict
                    self._lr_schedulers[k] = sch
            else:
                if not isinstance(inner_sch, ParamScheduler):
                    # convert scheduler to ignite scheduler
                    sch = LRScheduler(inner_sch)
                else:
                    sch = inner_sch

                # update scheduler in dict
                if isinstance(inner_sch, (StepLR, MultiStepLR, ExponentialLR)):
                    logger.info(
                        f"Initialized lr scheduler {inner_sch.__class__.__name__}. Scheduler updates are triggered per epoch. "
                    )
                    engine.add_event_handler(Events.EPOCH_STARTED, sch)
                elif isinstance(inner_sch, ReduceLROnPlateauScheduler):
                    logger.info(
                        f"Initialized lr scheduler {inner_sch.__class__.__name__}. Scheduler updates are triggered per validation step. "
                    )
                    # inner_sch.trainer = training_engine
                    engine.add_event_handler(Events.COMPLETED, sch)
                else:
                    logger.info(
                        f"Initialized lr scheduler {inner_sch.__class__.__name__}. Scheduler updates are triggered per optimizer step. "
                    )
                    engine.add_event_handler(OptimizerEvents.optimizer_step, sch)
                self._lr_schedulers[k] = sch

    def _configure_progress_bar(self, engine: Engine) -> None:
        from atria.core.training.engines.utilities import _log_training_metrics
        from atria.core.training.utilities.constants import TrainingStage
        from ignite.engine import Events

        self._progress_bar.attach(
            engine,
            metric_names="all",
            event_name=Events.ITERATION_COMPLETED(
                every=self._logging.refresh_rate,
            ),
            state_attributes=[
                "optimizer_step",
                "ema_momentum",
            ],
        )

        @engine.on(Events.EPOCH_COMPLETED)
        def progress_on_epoch_completed(engine: Engine) -> None:
            metrics = copy.deepcopy(engine.state.metrics)

            if hasattr(engine.state, f"ema_momentum"):
                metrics["ema/mom"] = engine.state.ema_momentum

            _log_training_metrics(
                logger=logger,
                epoch=engine.state.epoch,
                elapsed=engine.state.times["EPOCH_COMPLETED"],
                tag=TrainingStage.train,
                metrics=metrics,
            )

    def _configure_tb_logger(self, engine: Engine):
        if (
            idist.get_rank() == 0
            and self._tb_logger is not None
            and self._logging.log_to_tb
        ):
            from ignite.engine import Events

            # attach handler to plot trainer's loss every 'logging_steps' iterations
            self._tb_logger.attach_output_handler(
                engine,
                event_name=Events.ITERATION_COMPLETED(
                    every=self._logging.logging_steps
                ),
                tag=f"step",
                metric_names="all",
            )

            # attach the logger to the trainer to log optimizer's parameters, e.g. learning rate at every
            # 'logging_steps' iteration
            for param_name in ["lr", "weight_decay"]:
                for k, opt in self._optimizers.items():
                    self._tb_logger.attach_opt_params_handler(
                        engine,
                        event_name=Events.ITERATION_STARTED(
                            every=self._logging.logging_steps
                        ),
                        optimizer=opt,
                        param_name=param_name,
                        tag=f"step/opt/{k}",
                    )

    def _prepare_checkpoint_state_dict(
        self, engine: Engine, save_weights_only: bool = False
    ) -> Dict[str, Any]:
        from atria.core.training.engines.utilities import (
            MODEL_CHECKPOINT_KEY,
            TRAINING_ENGINE_KEY,
        )

        checkpoint_state_dict = {
            RUN_CONFIG_KEY: self._run_config,
            TRAINING_ENGINE_KEY: engine,
        }
        checkpoint_state_dict = {
            **checkpoint_state_dict,
            **{"task_module": self._task_module},
        }

        # add optimizers and lr/wd scheduler states to checkpoint_state_dict
        lr_schedulers_checkpoint_state_dict = (
            {f"lr_sch_{k}": v for k, v in self._lr_schedulers.items()}
            if self._lr_schedulers
            else {}
        )
        checkpoint_state_dict = {
            **checkpoint_state_dict,
            **{f"opt_{k}": v for k, v in self._optimizers.items()},
            **lr_schedulers_checkpoint_state_dict,
        }

        # add ema handler state to checkpoint_state_dict
        if self._ema_handler is not None:
            checkpoint_state_dict["ema_model"] = self._ema_handler.ema_model
            checkpoint_state_dict["ema_momentum_scheduler"] = (
                self._ema_handler.momentum_scheduler
            )

        # if only to save weights, remove all other keys
        if save_weights_only:
            for k in list(checkpoint_state_dict.keys()):
                if k not in [TRAINING_ENGINE_KEY, MODEL_CHECKPOINT_KEY]:
                    checkpoint_state_dict.pop(k)

        return checkpoint_state_dict

    def _configure_model_checkpointer(self, engine: Engine):
        from ignite.engine import Events
        from ignite.handlers import DiskSaver
        from ignite.handlers.checkpoint import BaseSaveHandler, Checkpoint, DiskSaver

        # setup checkpoint saving if required
        if self._model_checkpoint_config:
            checkpoint_state_dict = self._prepare_checkpoint_state_dict(
                engine,
                save_weights_only=self._model_checkpoint_config.save_weights_only,
            )

            checkpoint_dir = Path(self._output_dir) / self._model_checkpoint_config.dir
            save_handler = DiskSaver(
                checkpoint_dir,
                require_empty=False,
            )
            if self._model_checkpoint_config.save_per_epoch:
                checkpoint_handler = Checkpoint(
                    checkpoint_state_dict,
                    cast(Union[Callable, BaseSaveHandler], save_handler),
                    filename_prefix=self._model_checkpoint_config.name_prefix,
                    global_step_transform=lambda *_: engine.state.epoch,
                    n_saved=self._model_checkpoint_config.n_saved,
                    include_self=True,
                )
                engine.add_event_handler(
                    Events.EPOCH_COMPLETED(
                        every=self._model_checkpoint_config.save_every_iters
                    ),
                    checkpoint_handler,
                )
            else:
                checkpoint_handler = Checkpoint(
                    checkpoint_state_dict,
                    cast(Union[Callable, BaseSaveHandler], save_handler),
                    filename_prefix=self._model_checkpoint_config.name_prefix,
                    n_saved=self._model_checkpoint_config.n_saved,
                    include_self=True,
                )
                engine.add_event_handler(
                    Events.ITERATION_COMPLETED(
                        every=self._model_checkpoint_config.save_every_iters
                    )
                    | Events.COMPLETED,
                    checkpoint_handler,
                )

        if (
            self._validation_engine is not None
            and self._model_checkpoint_config.monitored_metric is not None
        ):
            from ignite.contrib.handlers import global_step_from_engine

            checkpoint_state_dict = self._prepare_checkpoint_state_dict(
                engine,
                save_weights_only=self._model_checkpoint_config.save_weights_only,
            )

            checkpoint_dir = Path(self._output_dir) / self._model_checkpoint_config.dir
            save_handler = DiskSaver(
                checkpoint_dir,
                require_empty=False,
            )

            logger.info(
                f"Configuring best model checkpointing with monitored metric: {self._model_checkpoint_config.monitored_metric}"
            )
            best_model_saver = Checkpoint(
                checkpoint_state_dict,
                save_handler=DiskSaver(
                    checkpoint_dir,
                    require_empty=False,
                ),
                filename_prefix="best",
                n_saved=self._model_checkpoint_config.n_best_saved,
                global_step_transform=global_step_from_engine(engine),
                score_name=self._model_checkpoint_config.monitored_metric.replace(
                    "/", "-"
                ),
                score_function=Checkpoint.get_default_score_fn(
                    self._model_checkpoint_config.monitored_metric,
                    -1 if self._model_checkpoint_config.mode == "min" else 1.0,
                ),
                include_self=True,
            )
            self._validation_engine.add_event_handler(
                Events.COMPLETED,
                best_model_saver,
            )

    def _load_training_state_from_checkpoint(self, engine: Engine):
        from atria.core.training.engines.utilities import (
            MODEL_CHECKPOINT_KEY,
        )
        from ignite.handlers.checkpoint import Checkpoint

        if self._model_checkpoint_config.resume_from_checkpoint:
            import torch
            from atria.core.training.utilities.checkpoints import find_resume_checkpoint

            checkpoint_state_dict = self._prepare_checkpoint_state_dict(
                engine,
                save_weights_only=self._model_checkpoint_config.save_weights_only,
            )
            checkpoint_dir = Path(self._output_dir) / self._model_checkpoint_config.dir
            resume_checkpoint_path = find_resume_checkpoint(
                self._model_checkpoint_config.resume_checkpoint_file,
                checkpoint_dir,
                self._model_checkpoint_config.load_best_checkpoint_resume,
            )
            if resume_checkpoint_path is not None:
                resume_checkpoint = torch.load(
                    resume_checkpoint_path, map_location="cpu"
                )

                if RUN_CONFIG_KEY in resume_checkpoint:
                    self._run_config.compare_configs(resume_checkpoint[RUN_CONFIG_KEY])

                for k in list(checkpoint_state_dict.keys()):
                    if k not in list(resume_checkpoint.keys()):
                        logger.warning(
                            f"Object {k} not found in the resume checkpoint_state_dict."
                        )
                        del checkpoint_state_dict[k]

                load_state_dict = {**checkpoint_state_dict}
                if self._model_checkpoint_config.load_weights_only:
                    for k in list(checkpoint_state_dict.keys()):
                        if k not in [MODEL_CHECKPOINT_KEY]:
                            load_state_dict.pop(k)

                Checkpoint.load_objects(
                    to_load=load_state_dict,
                    checkpoint=resume_checkpoint,
                    strict=False,
                )

    def _configure_early_stopping_callback(self, engine: Engine) -> None:
        if self._early_stopping.enabled:
            from ignite.engine import Events
            from ignite.handlers import Checkpoint, EarlyStopping

            if self._validation_engine is None:
                raise ValueError(
                    "Validation engine is not attached to training. Early stopping can not be configured. "
                    "Did you set do_validation=True in the trainer?"
                )

            es_handler = EarlyStopping(
                patience=self._early_stopping.patience,
                score_function=Checkpoint.get_default_score_fn(
                    self._early_stopping.monitored_metric,
                    -1 if self._early_stopping.mode == "min" else 1.0,
                ),
                trainer=engine,
            )
            self._validation_engine.add_event_handler(Events.COMPLETED, es_handler)

    def _configure_validation_engine(self, engine: Engine) -> None:
        if self._validation_engine is not None:
            self._validation_engine.attach_to_engine(
                parent_engine=engine,
                steps_per_epoch=self.steps_per_epoch,
                ema_handler=self._ema_handler,
            )

    def _configure_visualization_engine(self, engine: Engine) -> None:
        if self._visualization_engine is not None:
            self._visualization_engine.attach_to_engine(
                parent_engine=engine,
                steps_per_epoch=self.steps_per_epoch,
                ema_handler=self._ema_handler,
            )

    def _register_events(self, engine: Engine) -> None:
        engine.register_events(
            *OptimizerEvents,
            event_to_attr={
                OptimizerEvents.optimizer_step: OptimizerEvents.optimizer_step.value
            },
        )

    def _configure_engine(self, engine: Engine):
        # register events if needed
        self._register_events(engine=engine)

        # configure the training engine itself
        self._configure_train_sampler(engine=engine)
        self._configure_nan_callback(engine=engine)
        self._configure_cuda_cache_callback(engine=engine)
        self._configure_gpu_stats_callback(engine=engine)
        self._configure_time_profiler(engine=engine)
        self._configure_model_ema_callback(engine=engine)
        self._configure_metrics(engine=engine)
        self._configure_running_avg_logging(engine=engine)
        self._configure_progress_bar(engine=engine)
        self._configure_tb_logger(engine=engine)

        # configure the stuff where training engine and validation engine are connected
        self._configure_schedulers(engine=engine)
        self._configure_early_stopping_callback(engine=engine)
        self._configure_validation_engine(engine=engine)
        self._configure_visualization_engine(engine=engine)

        # configure model checkpointer
        self._configure_model_checkpointer(engine=engine)

        # print engine configuration info
        self._print_configuration_info()

    def _print_configuration_info(self):
        logger.info(f"Configured Training Engine:")
        logger.info(f"\tTotal steps per epoch = {self.batches_per_epoch}")
        logger.info(
            f"\tGradient accumulation per device = {self._gradient_config.gradient_accumulation_steps}"
        )
        logger.info(
            f"\tTotal optimizer update steps over epoch (scaled by grad accumulation steps) = {self.steps_per_epoch}"
        )
        logger.info(
            f"\tTotal optimizer update over complete training cycle (scaled by grad accumulation steps) = {self.total_update_steps}"
        )
        logger.info(f"\tTotal warmup steps = {self.total_warmup_steps}")
        logger.info(f"\tMax epochs = {self._max_epochs}")

    def run(
        self,
    ) -> State:
        # move task module models to device
        self._task_module.to_device(self._device, sync_bn=self._sync_batchnorm)

        # initialize engine
        engine = self._initialize_engine()

        # configure engine
        self._configure_engine(engine)

        # load training state from checkpoint
        self._load_training_state_from_checkpoint(engine)

        resume_epoch = engine.state.epoch
        if (
            engine._is_done(engine.state) and resume_epoch >= self._max_epochs
        ):  # if we are resuming from last checkpoint and training is already finished
            logger.warning(
                "Training has already been finished! Either increase the number of "
                f"epochs (current={self._max_epochs}) >= {resume_epoch} "
                "OR reset the training from start."
            )
            return

        # run engine
        logger.info(
            f"Running training with batch size [{self._dataloader.batch_size}] and output_dir: {self._output_dir}"
        )
        return engine.run(
            (
                FixedBatchIterator(self._dataloader, self._dataloader.batch_size)
                if self._use_fixed_batch_iterator
                else self._dataloader
            ),
            max_epochs=self._max_epochs,
            epoch_length=self._epoch_length,
        )
