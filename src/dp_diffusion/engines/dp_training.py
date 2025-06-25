import math
import numbers
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union, cast

import torch
import webdataset as wds
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
from atria.core.training.engines.engine_steps.training import TrainingStep
from atria.core.training.engines.evaluation import ValidationEngine, VisualizationEngine
from atria.core.training.engines.training import TrainingEngine
from atria.core.training.engines.utilities import RUN_CONFIG_KEY, RunConfig
from atria.core.utilities.common import _validate_partial_class
from atria.core.utilities.logging import get_logger
from ignite.engine import Engine, Events, State
from ignite.handlers import TensorboardLogger
from ignite.handlers.checkpoint import BaseSaveHandler, Checkpoint
from ignite.metrics import Metric
from torch.utils.data import DataLoader

from dp_diffusion.engines.batch_memory_manager import BatchMemoryManager
from dp_diffusion.engines.dp_training_step import DPTrainingStep
from dp_diffusion.engines.privacy_engine import ExtendedPrivacyEngine
from dp_diffusion.engines.privacy_loss import PrivacyLossMetric

logger = get_logger(__name__)


class PrivacyCheckpoint(Checkpoint):
    @staticmethod
    def setup_filename_pattern(
        with_prefix: bool = True,
        with_score: bool = True,
        with_score_name: bool = True,
        with_global_step: bool = True,
    ) -> str:
        """Helper method to get the default filename pattern for a checkpoint.

        Args:
            with_prefix: If True, the ``filename_prefix`` is added to the filename pattern:
                ``{filename_prefix}_{name}...``. Default, True.
            with_score: If True, ``score`` is added to the filename pattern: ``..._{score}.{ext}``.
                Default, True. At least one of ``with_score`` and ``with_global_step`` should be True.
            with_score_name: If True, ``score_name`` is added to the filename pattern:
                ``..._{score_name}={score}.{ext}``. If activated, argument ``with_score`` should be
                also True, otherwise an error is raised. Default, True.
            with_global_step: If True, ``{global_step}`` is added to the
                filename pattern: ``...{name}_{global_step}...``.
                At least one of ``with_score`` and ``with_global_step`` should be True.

        Examples:
            .. code-block:: python

                from ignite.handlers import Checkpoint

                filename_pattern = Checkpoint.setup_filename_pattern()

                print(filename_pattern)
                > "{filename_prefix}_{name}_{global_step}_{score_name}={score}.{ext}"

        .. versionadded:: 0.4.3
        """
        filename_pattern = "{name}"

        if not (with_global_step or with_score):
            raise ValueError(
                "At least one of with_score and with_global_step should be True."
            )

        if with_global_step:
            filename_pattern += "_{global_step}"

        if with_score_name and with_score:
            filename_pattern += "_{score_name}={score}"
        elif with_score:
            filename_pattern += "_{score}"
        elif with_score_name:
            raise ValueError(
                "If with_score_name is True, with_score should be also True"
            )

        if with_prefix:
            filename_pattern = "{filename_prefix}_" + filename_pattern

        filename_pattern += "-eps={eps}"

        filename_pattern += ".{ext}"
        return filename_pattern

    def __call__(self, engine: Engine) -> None:
        global_step = None
        if self.global_step_transform is not None:
            global_step = self.global_step_transform(engine, engine.last_event_name)

        if self.score_function is not None:
            priority = self.score_function(engine)
            if not isinstance(priority, numbers.Number):
                raise ValueError("Output of score_function should be a number")
        else:
            if global_step is None:
                global_step = engine.state.get_event_attrib_value(
                    Events.ITERATION_COMPLETED
                )
            priority = global_step

        print("Trying to saving checkpoint", global_step)
        if self._check_lt_n_saved() or self._compare_fn(priority):

            priority_str = (
                f"{priority}"
                if isinstance(priority, numbers.Integral)
                else f"{priority:.4f}"
            )
            print(priority_str)

            checkpoint = self._setup_checkpoint()

            name = "checkpoint"
            if len(checkpoint) == 1:
                for k in checkpoint:
                    name = k
                checkpoint = checkpoint[name]

            filename_pattern = self._get_filename_pattern(global_step)

            filename_dict = {
                "filename_prefix": self.filename_prefix,
                "ext": self.ext,
                "name": name,
                "score_name": self.score_name,
                "score": priority_str if (self.score_function is not None) else None,
                "global_step": global_step,
            }
            print(filename_dict)
            if "eps" in engine.state.metrics:
                filename_dict["eps"] = engine.state.metrics["eps"]
            else:
                filename_dict["eps"] = 0
            filename = filename_pattern.format(**filename_dict)

            metadata = {
                "basename": f"{self.filename_prefix}{'_' * int(len(self.filename_prefix) > 0)}{name}",
                "score_name": self.score_name,
                "priority": priority,
            }

            try:
                index = list(
                    map(lambda it: it.filename == filename, self._saved)
                ).index(True)
                to_remove = True
            except ValueError:
                index = 0
                to_remove = not self._check_lt_n_saved()

            if to_remove:
                item = self._saved.pop(index)
                if isinstance(self.save_handler, BaseSaveHandler):
                    self.save_handler.remove(item.filename)

            self._saved.append(Checkpoint.Item(priority, filename))
            self._saved.sort(key=lambda it: it[0])

            if self.include_self:
                # Now that we've updated _saved, we can add our own state_dict.
                checkpoint["checkpointer"] = self.state_dict()

            try:
                self.save_handler(checkpoint, filename, metadata)
            except TypeError:
                self.save_handler(checkpoint, filename)


class DPTrainingEngine(TrainingEngine):
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
        privacy_accountant: str = "rdp",
        target_epsilon: float = 10.0,
        target_delta: Optional[float] = None,
        noise_multiplier: Optional[float] = None,
        max_grad_norm: float = 10.0,
        use_bmm: bool = True,
        max_physical_batch_size: int = 1,
        n_splits: Optional[int] = None,
        noise_multiplicity: int = 1,
    ):
        _validate_partial_class(engine_step, DPTrainingStep, "engine_step")
        self._privacy_accountant = privacy_accountant
        self._target_epsilon = target_epsilon
        self._target_delta = target_delta
        self._noise_multiplier = noise_multiplier
        self._max_grad_norm = max_grad_norm
        self._use_bmm = use_bmm
        self._max_physical_batch_size = max_physical_batch_size
        self._n_splits = n_splits
        self._noise_multiplicity = noise_multiplicity
        self._privacy_engine = None

        super().__init__(
            run_config=run_config,
            output_dir=output_dir,
            dataset_cache_dir=dataset_cache_dir,
            task_module=task_module,
            dataloader=dataloader,
            engine_step=engine_step,
            device=device,
            optimizers=optimizers,
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
            lr_schedulers=lr_schedulers,
            validation_engine=validation_engine,
            visualization_engine=visualization_engine,
            gradient_config=gradient_config,
            eval_training=eval_training,
            stop_on_nan=stop_on_nan,
            clear_cuda_cache=clear_cuda_cache,
            model_ema_config=model_ema_config,
            warmup_config=warmup_config,
            early_stopping=early_stopping,
            model_checkpoint_config=model_checkpoint_config,
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

    def _make_private_module(self, module, optimizer, data_loader):
        assert (
            self._privacy_engine is not None
        ), "Privacy engine is not initialized. Please run `_configure_privacy_engine` "

        module.train()
        if self._target_epsilon is not None:
            return self._privacy_engine.make_private_with_epsilon(
                module=module,
                optimizer=optimizer,  # dp only supports one optimizer at this time
                data_loader=data_loader,
                epochs=self._max_epochs,
                target_epsilon=self._target_epsilon,
                target_delta=self._target_delta,
                max_grad_norm=self._max_grad_norm,
                noise_multiplicity=self._noise_multiplicity,
            )
        else:
            assert (
                self._noise_multiplier is not None
            ), "Either target_epsilon or noise_multiplier must be provided"
            return self._privacy_engine.make_private(
                module=module,
                optimizer=optimizer,  # dp only supports one optimizer at this time
                data_loader=data_loader,
                noise_multiplier=self._noise_multiplier,
                max_grad_norm=self._max_grad_norm,
                noise_multiplicity=self._noise_multiplicity,
            )

    def _configure_privacy_engine(self, engine: Engine):
        if self._target_delta is None:
            self._target_delta = 1.0 / len(self._dataloader.dataset)
        logger.info(
            f"Initializing privacy engine with delta = {self._target_delta} and accountant = {self._privacy_accountant}"
        )
        self._privacy_engine = ExtendedPrivacyEngine(
            accountant=self._privacy_accountant,
        )

        PrivacyLossMetric(
            privacy_engine=self._privacy_engine, delta=self._target_delta
        ).attach(
            engine,
            name="p_loss",
        )

    def _configure_model_checkpointer(self, engine: Engine):
        from ignite.engine import Events
        from ignite.handlers import DiskSaver
        from ignite.handlers.checkpoint import BaseSaveHandler, DiskSaver

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
                checkpoint_handler = PrivacyCheckpoint(
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
                checkpoint_handler = PrivacyCheckpoint(
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

    def _prepare_checkpoint_state_dict(
        self, engine: Engine, save_weights_only: bool = False
    ) -> Dict[str, Any]:
        checkpoint_state_dict = super()._prepare_checkpoint_state_dict(
            engine, save_weights_only
        )
        checkpoint_state_dict["privacy_engine"] = self._privacy_engine
        return checkpoint_state_dict

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

                for key in list(resume_checkpoint["task_module"].keys()):
                    if key in ["dataset_metadata"]:
                        continue
                    if not key.startswith("_module."):
                        resume_checkpoint["task_module"]["_module." + key] = (
                            resume_checkpoint["task_module"].pop(key)
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

    def _configure_engine(self, engine: Engine):
        self._configure_privacy_engine(engine)

        super()._configure_engine(engine)

    def run(
        self,
    ) -> State:
        # move task module models to device
        self._task_module.to_device(self._device, sync_bn=self._sync_batchnorm)

        # initialize engine
        engine = self._initialize_engine()

        # configure engine
        self._configure_engine(engine)

        # make private modules
        if isinstance(self._task_module._torch_model, TorchModelDict):
            (
                self._task_module._torch_model.trainable_models,
                self._optimizers["default"],
                self._dataloader,
            ) = self._make_private_module(
                self._task_module._torch_model.trainable_models,
                self._optimizers["default"],
                self._dataloader,
            )
        else:
            (
                self._task_module._torch_model,
                self._optimizers["default"],
                self._dataloader,
            ) = self._make_private_module(
                self._task_module._torch_model,
                self._optimizers["default"],
                self._dataloader,
            )

        # close the dataset as dp setup loads it
        if isinstance(self._dataloader.dataset, torch.utils.data.Subset):
            self._dataloader.dataset.dataset.close()
        else:
            self._dataloader.dataset.close()

        # initialize the engine step
        self._engine_step: DPTrainingStep
        self._engine_step._optimizers = self._optimizers
        self._engine_step._task_module = self._task_module

        logger.info(
            "Updated the models, optimizers and dataloader for privacy training"
        )
        logger.info("Updated model:")
        self._task_module.print_summary()
        logger.info("Updated optimizer:")
        logger.info(self._optimizers["default"])
        logger.info("Updated dataloader:")
        logger.info(self._dataloader)

        # log sigma and C
        for k, opt in self._optimizers.items():
            logger.info(
                f"Using epsilon={self._target_epsilon}, "
                f"sigma[{k}]={opt.noise_multiplier} and C={self._max_grad_norm}"
            )

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

        if self._use_bmm:
            with BatchMemoryManager(
                data_loader=self._dataloader,
                max_physical_batch_size=self._max_physical_batch_size,
                optimizer=self._optimizers,
                n_splits=self._n_splits,
            ) as memory_safe_data_loader:
                results = engine.run(
                    memory_safe_data_loader,
                    max_epochs=self._max_epochs,
                    epoch_length=self._epoch_length,
                )
            return results
        else:
            return engine.run(
                self._dataloader,
                max_epochs=self._max_epochs,
                epoch_length=self._epoch_length,
            )
