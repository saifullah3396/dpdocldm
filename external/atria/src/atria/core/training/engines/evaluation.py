import glob
from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Union

import ignite.distributed as idist
import pandas as pd
import torch
import webdataset as wds
from atria.core.data.data_modules.dataset_cacher.msgpack_shard_writer import (
    MsgpackShardWriter,
)
from atria.core.models.task_modules.atria_task_module import AtriaTaskModule
from atria.core.training.configs.logging_config import LoggingConfig
from atria.core.training.engines.atria_engine import AtriaEngine
from atria.core.training.engines.engine_steps.evaluation import (
    FeatureExtractorStep,
    PredictionStep,
    TestStep,
    ValidationStep,
    VisualizationStep,
)
from atria.core.training.engines.utilities import FixedBatchIterator
from atria.core.training.handlers.ema_handler import AtriaEMAHandler
from atria.core.training.utilities.output_saver_handler import ModelForwardDiskSaver
from atria.core.utilities.common import _validate_partial_class
from atria.core.utilities.logging import get_logger
from ignite.handlers import TensorboardLogger
from ignite.metrics import Metric
from torch.utils.data import DataLoader

logger = get_logger(__name__)

from ignite.engine import Engine, State


class ValidationEngine(AtriaEngine):
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]],
        dataset_cache_dir: Optional[Union[str, Path]],
        task_module: AtriaTaskModule,
        dataloader: Union[DataLoader, wds.WebLoader],
        device: Union[str, torch.device],
        engine_step: partial[ValidationStep],
        tb_logger: Optional[TensorboardLogger] = None,
        epoch_length: Optional[int] = None,
        outputs_to_running_avg: Optional[List[str]] = None,
        logging: Optional[LoggingConfig] = LoggingConfig(),
        metrics: Optional[Dict[str, partial[Metric]]] = None,
        metric_logging_prefix: Optional[str] = None,
        test_run: bool = False,
        use_fixed_batch_iterator: bool = False,
        checkpoints_dir: str = "checkpoints",
        validate_every_n_epochs: Optional[float] = 1,
        validate_on_start: bool = True,
        min_train_epochs_for_best: Optional[int] = 1,
        use_ema_for_val: bool = False,
    ):
        _validate_partial_class(engine_step, ValidationStep, "engine_step")

        super().__init__(
            output_dir=output_dir,
            dataset_cache_dir=dataset_cache_dir,
            task_module=task_module,
            dataloader=dataloader,
            engine_step=engine_step,
            device=device,
            tb_logger=tb_logger,
            max_epochs=1,
            epoch_length=epoch_length,
            outputs_to_running_avg=outputs_to_running_avg,
            logging=logging,
            metrics=metrics,
            metric_logging_prefix=metric_logging_prefix,
            test_run=test_run,
            use_fixed_batch_iterator=use_fixed_batch_iterator,
            checkpoints_dir=checkpoints_dir,
        )
        self._validate_every_n_epochs = validate_every_n_epochs
        self._validate_on_start = validate_on_start
        self._min_train_epochs_for_best = min_train_epochs_for_best
        self._use_ema_for_val = use_ema_for_val
        self._ema_handler = None
        self._parent_engine = None

    def _configure_engine(self, engine: Engine):
        self._configure_test_run(engine=engine)
        self._configure_metrics(engine=engine)
        self._configure_running_avg_logging(engine=engine)
        self._configure_progress_bar(engine=engine)
        self._configure_tb_logger(engine=engine)
        self._attach_event_handlers(engine=engine)

    def _configure_tb_logger(self, engine: Engine):
        import ignite.distributed as idist
        from ignite.contrib.handlers import global_step_from_engine
        from ignite.engine import Events

        if (
            idist.get_rank() == 0
            and self._tb_logger is not None
            and self._logging.log_to_tb
        ):
            assert (
                self._parent_engine is not None
            ), "Training engine is not set. You must call `attach_to_engine` first."
            # attach tb logger to validation engine
            self._tb_logger.attach_output_handler(
                engine,
                event_name=Events.EPOCH_COMPLETED,
                metric_names="all",
                tag="epoch",
                global_step_transform=global_step_from_engine(self._parent_engine),
            )

    def _initialize_engine(self) -> Engine:
        self._engine_step.attach_parent_engine(self._parent_engine)
        return super()._initialize_engine()

    def attach_to_engine(
        self,
        parent_engine: Engine,
        steps_per_epoch: int,
        ema_handler: Optional[AtriaEMAHandler] = None,
    ):
        from ignite.engine import Events

        if self._validate_every_n_epochs >= 1:
            cond = Events.EPOCH_COMPLETED(every=int(self._validate_every_n_epochs))
            cond = cond | Events.COMPLETED
            if self._validate_on_start:
                cond = cond | Events.STARTED
            parent_engine.add_event_handler(cond, self.run)
        else:
            cond = Events.ITERATION_COMPLETED(
                every=int(self._validate_every_n_epochs * steps_per_epoch)
            )
            cond = cond | Events.COMPLETED
            if self._validate_on_start:
                cond = cond | Events.STARTED
            parent_engine.add_event_handler(
                cond,
                self.run,
            )

        self._parent_engine = parent_engine
        self._ema_handler = ema_handler

    def run(
        self,
    ) -> Engine:
        if self._use_ema_for_val:
            if self._ema_handler is None:
                logger.warning(
                    "EMA handler is not set. You must pass an "
                    "EMA handler to `attach_to_engine` to use ema for validation."
                )
            else:
                self._ema_handler.swap_params()
        super().run()
        if self._use_ema_for_val:
            if self._ema_handler is not None:
                self._ema_handler.swap_params()


class TestEngine(AtriaEngine):
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]],
        dataset_cache_dir: Optional[Union[str, Path]],
        task_module: AtriaTaskModule,
        dataloader: Union[DataLoader, wds.WebLoader],
        device: Union[str, torch.device],
        engine_step: partial[TestStep],
        tb_logger: Optional[TensorboardLogger] = None,
        epoch_length: Optional[int] = None,
        outputs_to_running_avg: Optional[List[str]] = None,
        logging: LoggingConfig = LoggingConfig(),
        metrics: Optional[Dict[str, partial[Metric]]] = None,
        metric_logging_prefix: Optional[str] = None,
        test_run: bool = False,
        use_fixed_batch_iterator: bool = False,
        checkpoints_dir: str = "checkpoints",
        test_checkpoint_file: Optional[str] = None,
        save_model_forward_outputs: bool = False,
        checkpoint_types: Optional[List[str]] = None,
    ):
        _validate_partial_class(engine_step, TestStep, "engine_step")

        super().__init__(
            output_dir=output_dir,
            dataset_cache_dir=dataset_cache_dir,
            task_module=task_module,
            dataloader=dataloader,
            engine_step=engine_step,
            device=device,
            tb_logger=tb_logger,
            max_epochs=1,
            epoch_length=epoch_length,
            outputs_to_running_avg=outputs_to_running_avg,
            logging=logging,
            metrics=metrics,
            metric_logging_prefix=metric_logging_prefix,
            test_run=test_run,
            use_fixed_batch_iterator=use_fixed_batch_iterator,
            checkpoints_dir=checkpoints_dir,
        )
        self._test_checkpoint_file = test_checkpoint_file
        self._save_model_forward_outputs = save_model_forward_outputs
        self._checkpoint_types = checkpoint_types or ["last", "best"]
        for key in self._checkpoint_types:
            assert key in [
                "last",
                "best",
            ], f"Checkpoint type {key} is not supported. Possible types are ['last', 'best']"

    def _configure_model_forward_saver(
        self,
        engine: Engine,
        checkpoint_file_path: str,
    ):
        from ignite.engine import Events

        if self._save_model_forward_outputs:
            engine.add_event_handler(
                Events.ITERATION_COMPLETED,
                ModelForwardDiskSaver(
                    output_dir=self._output_dir,
                    checkpoint_file_path=checkpoint_file_path,
                ),
            )

    def _load_checkpoint(self, checkpoint_type: str):
        import torch
        from atria.core.training.utilities.checkpoints import find_test_checkpoint
        from ignite.handlers import Checkpoint

        checkpoint_dir = Path(self._output_dir) / self._checkpoints_dir
        checkpoint_file = find_test_checkpoint(
            self._test_checkpoint_file,
            checkpoint_dir,
            load_best=checkpoint_type == "best",
        )
        if checkpoint_file is not None:
            test_checkpoint = torch.load(checkpoint_file, map_location="cpu")
            Checkpoint.load_objects(
                to_load={"task_module": self._task_module}, checkpoint=test_checkpoint
            )

        return checkpoint_file

    def run(self) -> dict[str, State]:
        # move task module models to device
        self._task_module.to_device(self._device, sync_bn=self._sync_batchnorm)

        output_states = {}
        for checkpoint_type in self._checkpoint_types:
            logger.info(f"Running test for checkpoint_type=[{checkpoint_type}]")

            # set metric logging prefix to checkpoint type
            if self._metric_logging_prefix is not None:
                self._metric_logging_prefix += "/" + checkpoint_type
            else:
                self._metric_logging_prefix = checkpoint_type

            # initialize engine
            engine = self._initialize_engine()

            # configure engine
            self._configure_engine(engine)

            # configure checkpoint
            checkpoint_file_path = self._load_checkpoint(
                checkpoint_type=checkpoint_type,
            )

            if checkpoint_file_path is None:
                logger.warning(
                    f"No checkpoint file for checkpoint type '{checkpoint_type}'."
                )
                if checkpoint_type != "last":
                    continue
                logger.warning(f"Running evaluation on pre-loaded weights...")

            # configure output saver
            self._configure_model_forward_saver(
                engine=engine,
                checkpoint_file_path=checkpoint_file_path,
            )

            # run engine
            logger.info(
                f"Running test for checkpoint_type=[{checkpoint_type}] with batch size [{self._dataloader.batch_size}] and output_dir: {self._output_dir}"
            )
            output_states[checkpoint_type] = engine.run(
                (
                    FixedBatchIterator(self._dataloader, self._dataloader.batch_size)
                    if self._use_fixed_batch_iterator
                    else self._dataloader
                ),
                max_epochs=self._max_epochs,
                epoch_length=self._epoch_length,
            )
        return output_states


class VisualizationEngine(AtriaEngine):
    def __init__(
        self,
        output_dir: Optional[Union[str, Path]],
        dataset_cache_dir: Optional[Union[str, Path]],
        task_module: AtriaTaskModule,
        dataloader: Union[DataLoader, wds.WebLoader],
        device: Union[str, torch.device],
        engine_step: partial[VisualizationStep],
        tb_logger: Optional[TensorboardLogger] = None,
        epoch_length: Optional[int] = 1,
        outputs_to_running_avg: Optional[List[str]] = None,
        logging: LoggingConfig = LoggingConfig(),
        metrics: Optional[Dict[str, partial[Metric]]] = None,
        metric_logging_prefix: Optional[str] = None,
        test_run: bool = False,
        use_fixed_batch_iterator: bool = False,
        checkpoints_dir: str = "checkpoints",
        visualize_every_n_epochs: Optional[float] = 1,
        visualize_on_start: bool = False,
        use_ema_for_visualize: bool = False,
    ):
        _validate_partial_class(engine_step, VisualizationStep, "engine_step")

        super().__init__(
            output_dir=output_dir,
            dataset_cache_dir=dataset_cache_dir,
            task_module=task_module,
            dataloader=dataloader,
            engine_step=engine_step,
            device=device,
            tb_logger=tb_logger,
            max_epochs=1,
            epoch_length=epoch_length,
            outputs_to_running_avg=outputs_to_running_avg,
            logging=logging,
            metrics=metrics,
            metric_logging_prefix=metric_logging_prefix,
            test_run=test_run,
            use_fixed_batch_iterator=use_fixed_batch_iterator,
            checkpoints_dir=checkpoints_dir,
        )
        self._visualize_every_n_epochs = visualize_every_n_epochs
        self._visualize_on_start = visualize_on_start
        self._use_ema_for_visualize = use_ema_for_visualize
        self._ema_handler = None
        self._parent_engine = None

    def _initialize_engine(self) -> Engine:
        self._engine_step.attach_parent_engine(self._parent_engine)
        return super()._initialize_engine()

    def _configure_engine(self, engine: Engine):
        self._configure_test_run(engine=engine)
        self._configure_progress_bar(engine=engine)

    def _configure_progress_bar(self, engine: Engine) -> None:
        import ignite.distributed as idist
        from ignite.engine import Events

        if idist.get_rank() == 0:
            self._progress_bar.attach(
                engine,
                event_name=Events.ITERATION_COMPLETED(every=self._logging.refresh_rate),
            )

    def attach_to_engine(
        self,
        parent_engine: Engine,
        steps_per_epoch: int,
        ema_handler: Optional[AtriaEMAHandler] = None,
    ):
        from ignite.engine import Events

        if self._visualize_every_n_epochs >= 1:
            cond = Events.EPOCH_COMPLETED(every=int(self._visualize_every_n_epochs))
            cond = cond | Events.COMPLETED
            if self._visualize_on_start:
                cond = cond | Events.STARTED
            parent_engine.add_event_handler(cond, self.run)
        else:
            cond = Events.ITERATION_COMPLETED(
                every=int(self._visualize_every_n_epochs * steps_per_epoch)
            )
            cond = cond | Events.COMPLETED
            if self._visualize_on_start:
                cond = cond | Events.STARTED
            parent_engine.add_event_handler(
                cond,
                self.run,
            )

        self._parent_engine = parent_engine
        self._ema_handler = ema_handler

    def run(
        self,
    ) -> Engine:
        if self._use_ema_for_visualize:
            assert self._ema_handler is not None, (
                "EMA handler is not set. You must pass an "
                "EMA handler to `attach_to_engine` to use ema for visualization."
            )
            self._ema_handler.swap_params()
        super().run()
        if self._use_ema_for_visualize:
            assert self._ema_handler is not None, (
                "EMA handler is not set. You must pass an "
                "EMA handler to `attach_to_engine` to use ema for visualization."
            )
            self._ema_handler.swap_params()


class PredictionEngine(AtriaEngine):
    def __init__(
        self,
        task_module: AtriaTaskModule,
        dataset_cache_dir: Optional[Union[str, Path]],
        dataloader: Union[DataLoader, wds.WebLoader],
        device: Union[str, torch.device],
        engine_step: partial[PredictionStep] = None,
        tb_logger: Optional[TensorboardLogger] = None,
        epoch_length: Optional[int] = None,
        outputs_to_running_avg: Optional[List[str]] = None,
        logging: LoggingConfig = LoggingConfig(),
        metrics: Optional[Dict[str, partial[Metric]]] = None,
        metric_logging_prefix: Optional[str] = None,
        test_run: bool = False,
        use_fixed_batch_iterator: bool = False,
        checkpoints_dir: str = "checkpoints",
    ):
        _validate_partial_class(engine_step, PredictionStep, "engine_step")

        super().__init__(
            dataset_cache_dir=dataset_cache_dir,
            task_module=task_module,
            dataloader=dataloader,
            engine_step=engine_step,
            device=device,
            tb_logger=tb_logger,
            max_epochs=1,
            epoch_length=epoch_length,
            outputs_to_running_avg=outputs_to_running_avg,
            logging=logging,
            metrics=metrics,
            metric_logging_prefix=metric_logging_prefix,
            test_run=test_run,
            use_fixed_batch_iterator=use_fixed_batch_iterator,
            checkpoints_dir=checkpoints_dir,
        )


class FeatureExtractorEngine(AtriaEngine):
    def __init__(
        self,
        features_key: str,
        output_dir: Optional[Union[str, Path]],
        cache_file_name: str,
        engine_step: partial[FeatureExtractorStep],
        dataset_cache_dir: Optional[Union[str, Path]],
        task_module: AtriaTaskModule,
        dataloader: Union[DataLoader, wds.WebLoader],
        device: Union[str, torch.device],
        test_run: bool = False,
        max_shard_size: int = 100000,
    ):
        _validate_partial_class(engine_step, FeatureExtractorStep, "engine_step")
        super().__init__(
            output_dir=output_dir,
            dataset_cache_dir=dataset_cache_dir,
            task_module=task_module,
            dataloader=dataloader,
            engine_step=engine_step,
            device=device,
            tb_logger=None,
            max_epochs=1,
            epoch_length=None,
            outputs_to_running_avg=[],
            logging=LoggingConfig(logging_steps=1, refresh_rate=1),
            metrics=None,
            metric_logging_prefix=None,
            test_run=test_run,
        )
        self._features_key = features_key
        self._cache_file_name = cache_file_name
        self._max_shard_size = max_shard_size

    def _prepare_output_file_pattern(self, split: str, proc: int = 0) -> str:
        file_name = f"{self._cache_file_name}-" if self._cache_file_name else ""
        file_name += f"{split}-"
        file_name += f"{proc:06d}-"
        file_name += "%06d"
        file_name += f"-features-{self._features_key}.msgpack"
        return str(self._dataset_cache_dir / file_name)

    def _features_prepared(self) -> bool:
        from datadings.reader import MsgpackReader

        feature_files = sorted(glob.glob(self._features_path.replace("%06d", "*")))
        if len(feature_files) == 0:
            return False

        total_count = 0
        for feature_file in feature_files:
            try:
                with MsgpackReader(feature_file) as reader:
                    total_count += len(reader)
            except Exception as e:
                logger.warning(
                    f"Feature file {feature_file} is corrupted. Re-running feature extraction. {e}"
                )

                for feature_file in feature_files:
                    Path(feature_file).unlink(missing_ok=True)
                self._features_metadata_path.unlink(missing_ok=True)
                return False

        logger.info(f"Total feature samples found: {total_count}")
        if self._features_metadata_path.exists():
            try:
                metadata = pd.read_csv(self._features_metadata_path)
                assert total_count == len(metadata) and total_count == len(
                    self._dataloader.dataset
                ), (
                    "Features and metadata file lengths do not match. "
                    f"Features: {total_count}, Metadata: {len(metadata)}, Dataset size: {len(self._dataloader.dataset)} "
                    "Re-running feature extraction."
                )
            except AssertionError as e:
                logger.warning(e)
                for feature_file in feature_files:
                    Path(feature_file).unlink(missing_ok=True)
                self._features_metadata_path.unlink(missing_ok=True)
                return False
        else:
            return False
        return True

    def _configure_engine(self, engine: Engine, data_split: str):
        from ignite.engine import Events

        self._configure_test_run(engine=engine)
        self._configure_progress_bar(engine=engine)

        self._features_path_msgpack_writer = MsgpackShardWriter(
            self._features_path, maxcount=self._max_shard_size, overwrite=True
        )

        def write_features(engine: Engine):
            features = engine.state.output
            batch = dict(
                __key__=engine.state.batch["__key__"],
                __index__=engine.state.batch["__index__"],
                **features,
            )

            # convert dict of list to list of dict
            batch = [dict(zip(batch, t)) for t in zip(*batch.values())]

            for sample in batch:
                for key, value in sample.items():
                    if isinstance(value, torch.Tensor):
                        sample[key] = value.detach().cpu().numpy()
                self._features_path_msgpack_writer.write(sample)
                with open(self._features_metadata_path, "a") as f:
                    f.write(
                        f"{sample['__key__']},{sample['__index__']},{self._features_path_msgpack_writer.fname},{self._features_path_msgpack_writer.count-1}\n"
                    )

        def cleanup(engine: Engine):
            self._features_path_msgpack_writer.close()

        engine.add_event_handler(Events.ITERATION_COMPLETED, write_features)
        engine.add_event_handler(Events.EPOCH_COMPLETED, cleanup)

    def run(self, split: str):
        self._features_path = self._prepare_output_file_pattern(
            split, proc=idist.get_rank()
        )
        self._features_metadata_path = (
            Path(self._dataset_cache_dir)
            / f"{self._cache_file_name}-{split}-features-{self._features_key}-metadata.csv"
        )

        if self._features_prepared():
            return

        # prepare metadata file
        logger.info(f"Extracting dataset features to: {self._features_path}")
        logger.info(
            f"Saving dataset features metadata to: {self._features_metadata_path}"
        )

        logger.info(f"Preparing features for split: {split}")
        with open(self._features_metadata_path, "w") as f:
            f.write("key,index,features_path,feature_index\n")

        # move task module models to device
        self._task_module.to_device(self._device, sync_bn=self._sync_batchnorm)

        # initialize engine
        engine = self._initialize_engine()

        # configure engine
        self._configure_engine(engine, split)

        # run engine
        if self._output_dir is not None:
            logger.info(
                f"Running {self.__class__.__name__} engine with batch_size={self._dataloader.batch_size} with output_dir: {self._output_dir}"
            )
        else:
            logger.info(f"Running engine {self.__class__.__name__}.")
        return engine.run(
            self._dataloader,
            max_epochs=self._max_epochs,
            epoch_length=self._epoch_length,
        )
