from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Tuple

from atria.core.utilities.logging import get_logger

if TYPE_CHECKING:
    from ignite.contrib.handlers.base_logger import BaseLogger

logger = get_logger(__name__)


def is_dist_avail_and_initialized():
    import torch.distributed as dist

    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def empty_cuda_cache(_) -> None:
    import torch

    torch.cuda.empty_cache()
    import gc

    gc.collect()


def reset_random_seeds(seed):
    import random

    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _initialize_torch(seed: int = 0, deterministic: bool = False):
    import os

    import ignite.distributed as idist
    import torch

    seed = seed + idist.get_rank()
    logger.info(f"Global seed set to {seed}")
    reset_random_seeds(seed)

    # set seedon environment variable
    os.environ["DEFAULT_SEED"] = str(seed)

    # ensure that all operations are deterministic on GPU (if used) for reproducibility
    if deterministic:
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.determinstic = True
        torch.backends.cudnn.benchmark = False
    else:
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def generate_output_dir(
    output_dir: str,
    model_task: str,
    dataset_name: str,
    model_name: str,
    experiment_name: str,
    overwrite_output_dir: bool = False,
    logging_dir_suffix: str = "",
):
    """
    Sets up the output dir for an experiment based on the arguments.
    """
    from pathlib import Path

    import ignite.distributed as idist

    # generate root output dir = output_dir / model_task / model_name
    output_dir = Path(output_dir) / model_task / dataset_name / Path(model_name)

    # create a unique directory for each experiment
    if logging_dir_suffix != "":
        experiment = f"{experiment_name}/{logging_dir_suffix}"
    else:
        experiment = f"{experiment_name}"

    # append experiment name to output dir
    output_dir = output_dir / experiment

    # overwrite the experiment if required
    if overwrite_output_dir and idist.get_rank() == 0:
        import shutil

        logger.info("Overwriting output directory.")
        shutil.rmtree(output_dir, ignore_errors=True)

    # generate directories
    if not output_dir.exists() and idist.get_rank() == 0:
        output_dir.mkdir(parents=True)

    return output_dir


def _setup_tensorboard(
    output_dir: str,
) -> Tuple[str, BaseLogger]:
    import ignite.distributed as idist

    if idist.get_rank() == 0:
        from ignite.handlers import TensorboardLogger

        log_dir = Path(output_dir) / "tensorboard_log_dir"
        log_dir.mkdir(parents=True, exist_ok=True)
        tb_logger = TensorboardLogger(log_dir=log_dir)
    else:
        tb_logger = None
    return tb_logger
