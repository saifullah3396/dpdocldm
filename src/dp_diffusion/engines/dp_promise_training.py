from functools import partial
from pathlib import Path
from typing import Dict, List, Optional, Union

import torch
from atria.core.training.engines.training import TrainingEngine
from dp_diffusion.engines.dp_training import DPTrainingEngine
from dp_diffusion.engines.dp_training_step import DPTrainingStep
from dp_diffusion.engines.privacy_engine import ExtendedPrivacyEngine
import webdataset as wds
from atria.core.models.task_modules.atria_task_module import (
    AtriaTaskModule,
)
from ignite.engine import Engine
from atria.core.schedulers.typing import LRSchedulerType
from atria.core.training.configs.early_stopping_config import EarlyStoppingConfig
from atria.core.training.configs.gradient_config import GradientConfig
from atria.core.training.configs.logging_config import LoggingConfig
from atria.core.training.configs.model_checkpoint import ModelCheckpointConfig
from atria.core.training.configs.model_ema_config import ModelEmaConfig
from atria.core.training.configs.warmup_config import WarmupConfig
from atria.core.training.engines.engine_steps.training import TrainingStep
from atria.core.training.engines.evaluation import ValidationEngine, VisualizationEngine
from atria.core.training.engines.utilities import RunConfig
from atria.core.utilities.common import _validate_partial_class
from atria.core.utilities.logging import get_logger
from ignite.handlers import TensorboardLogger
from ignite.metrics import Metric
from torch.utils.data import DataLoader

from diffusers.schedulers import DDPMScheduler

import torch
from torch.utils.data import DataLoader


from scipy import optimize
from scipy.stats import norm
from math import sqrt
import numpy as np

logger = get_logger(__name__)


# Dual between mu-GDP and (epsilon,delta)-DP
def delta_eps_mu(eps, mu):
    return norm.cdf(-eps / mu + mu / 2) - np.exp(eps) * norm.cdf(-eps / mu - mu / 2)


# inverse Dual
def eps_from_mu(mu, delta):

    def f(x):
        return delta_eps_mu(x, mu) - delta

    return optimize.root_scalar(f, bracket=[0, 500], method="brentq").root


def gdp_mech(
    sample_rate1, sample_rate2, niter1, niter2, sigma, alpha_cumprod_S, input_dim, delta
):
    mu_1 = sample_rate1 * sqrt(
        niter1 * (np.exp(4 * input_dim * alpha_cumprod_S / (1 - alpha_cumprod_S)) - 1)
    )
    mu_2 = sample_rate2 * sqrt(niter2 * (np.exp(1 / (sigma**2)) - 1))
    mu = sqrt(mu_1**2 + mu_2**2)
    epsilon = eps_from_mu(mu, delta)
    return epsilon


def compute_epsilon(
    dataloader_non_private: DataLoader,
    dataloader_private: DataLoader,
    non_private_epochs: int,
    private_epochs: int,
    diffusion_scheduler: DDPMScheduler,
    delta: int,
    cutoff_ratio: float,
    noise_multiplier,
    input_dim,  # 4 * 4 * 32
):
    prob1 = 1 / len(dataloader_non_private)
    prob2 = 1 / len(dataloader_private)
    niter1 = non_private_epochs * len(dataloader_non_private)
    niter2 = private_epochs * len(dataloader_private)

    betas = diffusion_scheduler.betas

    alphas = 1 - betas
    alphas_cumprod = torch.cumprod(alphas, axis=0)
    alpha_cumprod_S = alphas_cumprod[
        int(cutoff_ratio * len(diffusion_scheduler.betas)) - 1
    ].numpy()

    logger.info("Computing epsilon with parameters:")
    logger.info(f"prob1: {prob1}")
    logger.info(f"prob2: {prob2}")
    logger.info(f"niter1: {niter1}")
    logger.info(f"niter2: {niter2}")
    logger.info(f"sigma: {noise_multiplier}")
    logger.info(f"alpha_cumprod_S: {alpha_cumprod_S}")
    logger.info(f"input_dim: {input_dim}")
    logger.info(f"delta: {delta}")

    epsilon = gdp_mech(
        sample_rate1=prob1,
        sample_rate2=prob2,
        niter1=niter1,
        niter2=niter2,
        sigma=noise_multiplier,
        alpha_cumprod_S=alpha_cumprod_S,
        input_dim=input_dim,
        delta=delta,
    )

    return epsilon


class DPPromiseTrainingEngine(DPTrainingEngine):
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
        noise_multiplier: Optional[float] = None,
        max_grad_norm: float = 10.0,
        use_bmm: bool = True,
        max_physical_batch_size: int = 1,
        n_splits: Optional[int] = None,
        noise_multiplicity: int = 1,
        cutoff_ratio: float = 0.9,
        target_delta: float = 1e-6,
    ):
        _validate_partial_class(engine_step, DPTrainingStep, "engine_step")
        assert noise_multiplier is not None, "noise_multiplier must be provided"
        self._cutoff_ratio = cutoff_ratio

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
            privacy_accountant="gdp",
            target_epsilon=None,
            target_delta=target_delta,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            use_bmm=use_bmm,
            max_physical_batch_size=max_physical_batch_size,
            n_splits=n_splits,
            noise_multiplicity=noise_multiplicity,
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

    def _configure_model_checkpointer(self, engine: Engine):
        TrainingEngine._configure_model_checkpointer(self, engine)
