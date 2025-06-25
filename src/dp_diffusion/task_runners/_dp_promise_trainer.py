import logging
from functools import partial
from typing import Optional

import ignite.distributed as idist
import numpy as np
import torch
from atria.core.data.data_modules.atria_data_module import AtriaDataModule
from atria.core.models.task_modules.atria_task_module import AtriaTaskModule
from atria.core.training.engines.evaluation import (
    FeatureExtractorEngine,
    TestEngine,
    ValidationEngine,
    VisualizationEngine,
)
from atria.core.training.engines.training import TrainingEngine
from atria.core.training.utilities.initialization import (
    _initialize_torch,
    _setup_tensorboard,
)
from opacus.data_loader import DPDataLoader
from atria.core.utilities.common import _msg_with_separator
from atria.core.utilities.logging import get_logger
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf  # q
from omegaconf import DictConfig

from atria.models.task_modules.diffusion.diffusion import DiffusionModule
from dp_diffusion.engines.dp_promise_training import (
    DPPromiseTrainingEngine,
    compute_epsilon,
)


class DPPromiseTrainer:
    def __init__(
        self,
        data_module: AtriaDataModule,
        task_module: partial[AtriaTaskModule],
        non_private_training_engine: partial[TrainingEngine],
        private_training_engine: partial[TrainingEngine],
        validation_engine: partial[ValidationEngine],
        visualization_engine: partial[VisualizationEngine],
        test_engine: partial[TestEngine],
        output_dir: str,
        feature_extractor_engine: Optional[partial[FeatureExtractorEngine]] = None,
        seed: int = 42,
        deterministic: bool = False,
        do_train: bool = True,
        do_validation: bool = True,
        do_visualization: bool = False,
        do_test: bool = True,
        feature_extraction_batch_size: int = 64,
        vis_batch_size: int = 64,
        non_private_batch_size: int = 64,
        private_batch_size: int = 64,
    ):
        self._data_module = data_module
        self._task_module = task_module
        self._non_private_training_engine = non_private_training_engine
        self._private_training_engine = private_training_engine
        self._validation_engine = validation_engine
        self._visualization_engine = visualization_engine
        self._test_engine = test_engine
        self._output_dir = output_dir
        self._feature_extractor_engine = feature_extractor_engine
        self._seed = seed
        self._deterministic = deterministic
        self._do_train = do_train
        self._do_validation = do_validation
        self._do_visualization = do_visualization
        self._do_test = do_test
        self._feature_extraction_batch_size = feature_extraction_batch_size
        self._visualization_batch_size = vis_batch_size
        self._non_private_batch_size = non_private_batch_size
        self._private_batch_size = private_batch_size

        if not self._do_validation:
            self._validation_engine = None
        if not self._do_visualization:
            self._visualization_engine = None

    def run(self, hydra_config: HydraConfig, run_config: DictConfig) -> None:
        logger = get_logger(
            hydra_config=hydra_config, distributed_rank=idist.get_rank()
        )

        # print log config
        for handler in logger.handlers:
            if isinstance(handler, logging.FileHandler):
                logger.info(
                    f"Verbose logs can be found at file: {handler.baseFilename}"
                )

        # initialize training
        _initialize_torch(
            seed=self._seed,
            deterministic=self._deterministic,
        )

        # initialize torch device (cpu or gpu)
        device = idist.device()

        # initialize logging directory and tensorboard logger
        output_dir = hydra_config.runtime.output_dir
        tb_logger = _setup_tensorboard(output_dir)

        # build data module
        logger.info(_msg_with_separator("Setting up data module"))
        self._data_module.setup(
            # for feature extractor engine, we disable the validation splitter as the underlying dataset is the same
            disable_subsampling=self._feature_extractor_engine
            is not None,
        )

        # initialize the task module from partial
        logger.info(_msg_with_separator("Setting up task module"))
        self._task_module = self._task_module(
            dataset_metadata=self._data_module.dataset_metadata,
            tb_logger=tb_logger,
        )
        self._task_module.build_model()
        assert isinstance(
            self._task_module, DiffusionModule
        ), "Task module must be a DiffusionModule"

        # initialize the feature extractor engine from partial
        if self._feature_extractor_engine is not None:
            logger.info(_msg_with_separator("Setting up feature extractor engine"))

            # print dataloader information
            non_private_train_dataloader = self._data_module.train_dataloader(
                batch_size=self._feature_extraction_batch_size, shuffle=False
            )
            validation_dataloder = (
                self._data_module.validation_dataloader(
                    batch_size=self._feature_extraction_batch_size,
                )
                if self._data_module.validation_dataset is not None
                else None
            )
            test_dataloader = self._data_module.test_dataloader(
                batch_size=self._feature_extraction_batch_size,
            )

            # initilize the feature extractor engine from partial
            for split, dataloader in zip(
                ["train", "validation", "test"],
                [non_private_train_dataloader, validation_dataloder, test_dataloader],
            ):
                if dataloader is None:
                    continue

                feature_extractor_engine = self._feature_extractor_engine(
                    output_dir=output_dir,
                    cache_file_name=self._data_module._dataset_cacher._cache_file_name,
                    dataset_cache_dir=self._data_module._dataset_cacher.cache_dir,
                    task_module=self._task_module,
                    dataloader=dataloader,
                    device=device,
                )

                # run the feature extractor engine on training data to extract training features
                feature_extractor_engine.run(split=split)

            # re load the data module to include the extracted features
            self._data_module.setup()

        if self._do_train:
            logger.info(_msg_with_separator("Setting up training"))

            if self._validation_engine is not None:
                # check if validation dataset is available
                if self._data_module.validation_dataset is None:
                    logger.warning(
                        "You have set do_validation=True but there is no validation dataset available. "
                        "To create a validation dataset from the training dataset, set train_val_splitter in the config."
                        "Using test dataset for validation."
                    )
                    logger.info(f"Initializing test dataloader for validation.")
                    validation_dataloder = self._data_module.test_dataloader()
                else:
                    logger.info(f"Initializing validation dataloader.")
                    validation_dataloder = self._data_module.validation_dataloader()

                # initilize the validation engine from partial
                self._validation_engine = self._validation_engine(
                    output_dir=output_dir,
                    dataset_cache_dir=self._data_module._dataset_cacher.cache_dir,
                    task_module=self._task_module,
                    dataloader=validation_dataloder,
                    device=device,
                    tb_logger=tb_logger,
                )

            if self._visualization_engine is not None:
                # initilize the validation engine from partial
                # by default, visualization engine uses the train dataloader as it is
                # generally to be used for generative tasks
                logger.info(f"Initializing train dataloader for visualization.")
                visualization_data_loader = self._data_module.train_dataloader(
                    batch_size=self._visualization_batch_size
                )
                self._visualization_engine = self._visualization_engine(
                    output_dir=output_dir,
                    dataset_cache_dir=self._data_module._dataset_cacher.cache_dir,
                    task_module=self._task_module,
                    dataloader=visualization_data_loader,
                    device=device,
                    tb_logger=tb_logger,
                    epoch_length=1,  # this defines how many batches to run the visualize step for
                )

            # initilize the training engine from partial
            logger.info(
                _msg_with_separator(f"Initializing non-private train dataloader.")
            )
            non_private_train_dataloader = self._data_module.train_dataloader(
                batch_size=self._non_private_batch_size
            )
            non_private_dp_train_dataloader = DPDataLoader.from_data_loader(
                non_private_train_dataloader
            )
            if isinstance(
                non_private_dp_train_dataloader.dataset, torch.utils.data.Subset
            ):
                non_private_dp_train_dataloader.dataset.dataset.close()
            else:
                non_private_dp_train_dataloader.dataset.close()

            logger.info(_msg_with_separator(f"Initializing private train dataloader."))
            private_train_dataloader = self._data_module.train_dataloader(
                batch_size=self._private_batch_size
            )
            logger.info(_msg_with_separator("Setting up non-private training engine"))
            self._non_private_training_engine = self._non_private_training_engine(
                run_config=run_config,
                output_dir=output_dir,
                dataset_cache_dir=self._data_module._dataset_cacher.cache_dir,
                task_module=self._task_module,
                dataloader=non_private_dp_train_dataloader,
                device=device,
                tb_logger=tb_logger,
                validation_engine=None,
                visualization_engine=self._visualization_engine,
            )
            logger.info(_msg_with_separator("Setting up private training engine"))
            self._private_training_engine: DPPromiseTrainingEngine = (
                self._private_training_engine(
                    run_config=run_config,
                    output_dir=output_dir,
                    dataset_cache_dir=self._data_module._dataset_cacher.cache_dir,
                    task_module=self._task_module,
                    dataloader=private_train_dataloader,
                    device=device,
                    tb_logger=tb_logger,
                    validation_engine=self._validation_engine,
                    visualization_engine=self._visualization_engine,
                )
            )

            self._target_epsilon = compute_epsilon(
                non_private_train_dataloader,
                private_train_dataloader,
                self._non_private_training_engine._max_epochs,
                self._private_training_engine._max_epochs,
                self._task_module.forward_noise_scheduler,
                self._private_training_engine._target_delta,
                self._private_training_engine._cutoff_ratio,
                self._private_training_engine._noise_multiplier,
                np.prod(self._task_module._input_shape),
            )
            logger.info(
                f"Satisfy ({self._target_epsilon}, {self._private_training_engine._target_delta})-DP"
            )

            # run the test engine
            diffusion_steps_filter_range = (
                int(
                    self._task_module._diffusion_steps
                    * self._private_training_engine._cutoff_ratio
                ),  # only take from (0, t - cutoff_ratio * t)
                self._task_module._diffusion_steps,
            )
            self._task_module._schedule_sampler.step_filter_range = (
                diffusion_steps_filter_range
            )
            noise_multiplicity = self._task_module._noise_multiplicity
            self._task_module._noise_multiplicity = None
            self._non_private_training_engine.run()

            diffusion_steps_filter_range = (
                0,
                int(
                    self._task_module._diffusion_steps
                    * self._private_training_engine._cutoff_ratio
                ),  # only take from (0, t - cutoff_ratio * t)
            )
            self._task_module._schedule_sampler.step_filter_range = (
                diffusion_steps_filter_range
            )
            self._task_module._noise_multiplicity = noise_multiplicity
            self._private_training_engine.run()

        if self._do_test:
            logger.info(_msg_with_separator("Setting up test engine"))
            # initilize the test engine from partial
            self._test_engine = self._test_engine(
                output_dir=output_dir,
                dataset_cache_dir=self._data_module._dataset_cacher.cache_dir,
                task_module=self._task_module,
                # for FID computations we compare with training set as this is for generative modeling
                dataloader=self._data_module.test_dataloader(),
                device=device,
                tb_logger=tb_logger,
            )

            # run the test engine
            self._test_engine.run()
