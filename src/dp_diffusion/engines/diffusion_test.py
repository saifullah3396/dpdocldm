from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from atria.core.training.engines.evaluation import TestEngine
from atria.models.task_modules.diffusion.diffusion import DiffusionModule
import webdataset as wds
from atria.core.models.task_modules.atria_task_module import AtriaTaskModule
from atria.core.training.configs.logging_config import LoggingConfig
from atria.core.training.engines.engine_steps.evaluation import (
    TestStep,
)
from atria.core.utilities.logging import get_logger
from ignite.handlers import TensorboardLogger
from ignite.metrics import Metric
from torch.utils.data import DataLoader


from ignite.engine import State

logger = get_logger(__name__)


class DiffusionTestEngine(TestEngine):
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
        test_guidance_scales: Optional[List[float]] = None,
    ):
        super().__init__(
            output_dir=output_dir,
            dataset_cache_dir=dataset_cache_dir,
            task_module=task_module,
            dataloader=dataloader,
            engine_step=engine_step,
            device=device,
            tb_logger=tb_logger,
            epoch_length=epoch_length,
            outputs_to_running_avg=outputs_to_running_avg,
            logging=logging,
            metrics=metrics,
            metric_logging_prefix=metric_logging_prefix,
            test_run=test_run,
            use_fixed_batch_iterator=use_fixed_batch_iterator,
            checkpoints_dir=checkpoints_dir,
            test_checkpoint_file=test_checkpoint_file,
            save_model_forward_outputs=save_model_forward_outputs,
            checkpoint_types=checkpoint_types,
        )

        self._test_guidance_scales = test_guidance_scales

    def run(self) -> dict[str, State]:
        assert isinstance(
            self._task_module, DiffusionModule
        ), "Task module must be an instance of DiffusionModule"
        cfg_available = (
            self._task_module._enable_class_conditioning and self._task_module._use_cfg
        )
        if self._test_guidance_scales is None or not cfg_available:
            if (
                self._task_module._enable_class_conditioning
                and self._task_module._use_cfg
            ):
                self._metric_logging_prefix = (
                    self._task_module._forward_noise_scheduler.__class__.__name__
                    + "/"
                    + str(self._task_module._guidance_scale)
                )
                logger.info(
                    "Running test with guidance scale: %s",
                    self._task_module.guidance_scale,
                )
            else:
                self._metric_logging_prefix = (
                    self._task_module._forward_noise_scheduler.__class__.__name__
                )
            if self._task_module.check_output_msgpack_file_exists():
                logger.info(f"Output samples already exists. Skipping test run.")
                return {}
            return super().run()
        else:
            output_per_guidance_scale = {}
            for guidance_scale in self._test_guidance_scales:
                logger.info("Running test with guidance scale: %s", guidance_scale)
                self._task_module.guidance_scale = guidance_scale
                self._metric_logging_prefix = (
                    self._task_module._forward_noise_scheduler.__class__.__name__
                    + "/"
                    + str(self._task_module._guidance_scale)
                )

                # check if output dir for samples is already done and if so skip the run
                if self._task_module.check_output_msgpack_file_exists():
                    logger.info(f"Output samples already exists. Skipping test run.")
                    continue

                output_per_guidance_scale[guidance_scale] = super().run()
            return output_per_guidance_scale
