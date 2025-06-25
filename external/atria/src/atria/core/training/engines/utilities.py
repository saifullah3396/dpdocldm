from __future__ import annotations

from dataclasses import is_dataclass
from functools import partial
from typing import Any, Dict, List, Mapping, Sequence, Union

import torch
from atria.core.models.model_outputs import ModelOutput
from atria.core.training.utilities.constants import TrainingStage
from atria.core.training.utilities.ddp_model_proxy import ModuleProxyWrapper
from atria.core.utilities.logging import get_logger
from atria.core.utilities.string_utils import _indent_string
from ignite.engine import Engine
from ignite.metrics import Metric
from omegaconf import DictConfig, OmegaConf
from sqlalchemy import Engine
from torch import Tensor, nn

logger = get_logger(__name__)

TRAINING_ENGINE_KEY = "training_engine"
MODEL_CHECKPOINT_KEY = "model"
RUN_CONFIG_KEY = "run_config"


class RunConfig:
    def __init__(self, data: DictConfig):
        self._data = OmegaConf.to_container(data, resolve=True)

    def state_dict(self) -> Dict[str, Any]:
        return self._data

    def compare_configs(self, target_data: dict) -> bool:
        differences = _find_differences(self._data, target_data)
        if len(differences) > 0:
            logger.warning(
                "You are trying to continue a training run with different configuration from the previous run."
            )
            logger.warning("Differences:")
            for diff in differences:
                logger.warning(
                    f"Key: {diff[0]}, Previous value: {diff[2]}, New value: {diff[1]}"
                )

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        pass


class FixedBatchIterator:
    def __init__(self, dataloader, fixed_batch_size):
        self.dataloader = dataloader
        self.fixed_batch_size = fixed_batch_size

    def __iter__(self):
        total_samples = 0
        current_batch = None
        for batch in self.dataloader:
            total_samples += batch["label"].shape[0]
            if current_batch is None:
                current_batch = {k: v for k, v in batch.items()}
            else:
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        current_batch[k] = torch.cat([current_batch[k], v], dim=0)
                    else:
                        current_batch[k].extend(v)
            while len(current_batch["__key__"]) >= self.fixed_batch_size:
                yielded_batch = {
                    k: v[: self.fixed_batch_size] for k, v in current_batch.items()
                }
                yield yielded_batch
                current_batch = {
                    k: v[self.fixed_batch_size :] for k, v in current_batch.items()
                }
        if current_batch:
            yield current_batch


def _empty_cuda_cache(_) -> None:
    import torch

    torch.cuda.empty_cache()
    import gc

    gc.collect()


def _extract_output(x: Any, index: int, key: str) -> Any:
    import numbers

    import torch

    if isinstance(x, Mapping):
        return x[key]
    elif isinstance(x, Sequence):
        return x[index]
    elif isinstance(x, (torch.Tensor, numbers.Number)):
        return x
    elif is_dataclass(x):
        return getattr(x, key)
    else:
        raise TypeError(
            "Unhandled type of update_function's output. "
            f"It should either mapping or sequence, but given {type(x)}"
        )


def _detach_tensor(
    tensor: Union[Dict[str, Tensor], ModelOutput]
) -> Union[Dict[str, Tensor], ModelOutput]:
    from ignite.utils import apply_to_tensor

    # detach all the outputs from the graph
    return apply_to_tensor(tensor, lambda tensor: tensor.detach())


def _convert_tensor_to_half(
    model_output: Union[Dict[str, Tensor], ModelOutput]
) -> Union[Dict[str, Tensor], ModelOutput]:
    from ignite.utils import apply_to_tensor

    # detach all the outputs from the graph
    return apply_to_tensor(model_output, lambda tensor: tensor.half())


def _log_training_metrics(logger, epoch, elapsed, tag, metrics):
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(
        f"\nEpoch {epoch} - Training time (seconds): {elapsed:.2f} - {tag} metrics:\n {metrics_output}"
    )


def _log_eval_metrics(logger, epoch, elapsed, tag, metrics):
    metrics_output = "\n".join([f"\t{k}: {v}" for k, v in metrics.items()])
    logger.info(
        f"\nEpoch {epoch} - Evaluation time (seconds): {elapsed:.2f} - {tag} metrics:\n {metrics_output}"
    )


def _attach_metrics_to_engine(
    engine: Engine,
    metrics: Dict[str, Metric],
    stage: TrainingStage,
    prefix: str = None,
):
    from ignite.metrics.metric import EpochWise

    for metric_name, metric in metrics.items():
        logger.debug(f"Attaching metric {metric_name} to engine")
        metric.attach(
            engine,
            (
                f"{stage}/{metric_name}"
                if prefix is None
                else f"{prefix}/{stage}/{metric_name}"
            ),
            usage=EpochWise(),
        )


def _attach_nan_callback_to_engine(engine: Engine):
    from ignite.engine import Events
    from ignite.handlers import TerminateOnNan

    engine.add_event_handler(Events.ITERATION_COMPLETED, TerminateOnNan())


def _attach_cuda_cache_callback_to_engine(engine: Engine):
    import torch
    from ignite.engine import Events

    if torch.cuda.is_available():
        engine.add_event_handler(Events.EPOCH_COMPLETED, _empty_cuda_cache)


def _attach_gpu_stats_callback_to_engine(engine: Engine, logging_steps: int):
    import ignite.distributed as idist
    import torch

    if idist.device() != torch.device("cpu"):
        from ignite.contrib.metrics import GpuInfo
        from ignite.engine import Events

        GpuInfo().attach(
            engine,
            name="gpu",
            event_name=Events.ITERATION_COMPLETED(every=logging_steps),
        )


def _attach_time_profiler_to_engine(engine: Engine):
    from ignite.engine import Events
    from ignite.handlers import BasicTimeProfiler, HandlersTimeProfiler

    handlers_profiler = HandlersTimeProfiler()
    basic_printer = BasicTimeProfiler()
    basic_printer.attach(engine)
    handlers_profiler.attach(engine)

    @engine.on(Events.EPOCH_COMPLETED)
    def log_intermediate_results():
        basic_printer.print_results(basic_printer.get_results())
        handlers_profiler.print_results(handlers_profiler.get_results())


def _attach_output_logging_to_engine(
    engine: Engine, stage: TrainingStage, outputs_to_running_avg: List[str], alpha=0.95
):
    from ignite.metrics import RunningAverage

    for index, key in enumerate(outputs_to_running_avg):
        RunningAverage(
            alpha=alpha,
            output_transform=partial(_extract_output, index=index, key=key),
            epoch_bound=True,
        ).attach(engine, f"{stage}/running_avg_{key}")


def _module_to_device(
    module: nn.Module,
    device: Union[str, torch.device],
    sync_bn: bool = False,
    prepare_for_distributed: bool = False,
) -> nn.Module:
    import ignite.distributed as idist
    import torch

    if prepare_for_distributed:
        module = idist.auto_model(
            module,
            sync_bn=(False if device == torch.device("cpu") else sync_bn),
        )
    else:
        module.to(device)

    if isinstance(module, (nn.parallel.DistributedDataParallel, nn.DataParallel)):
        module = ModuleProxyWrapper(module)

    return module


def _print_optimizers_info(optimizers: Dict[str, torch.optim.Optimizer]) -> None:
    import ignite.distributed as idist

    if idist.get_rank() == 0:
        # print information
        msg = f"Configured optimizers:\n"
        for k, v in optimizers.items():
            opt_str = _indent_string(str(v), " " * 4)
            msg += f"{k}:\n"
            msg += f"{opt_str}\n"
        logger.info(msg)


def _print_schedulers_info(
    lr_schedulers: Dict[str, torch.optim.lr_scheduler.LRScheduler]
) -> None:
    import ignite.distributed as idist

    if idist.get_rank() == 0:
        if lr_schedulers is not None:
            msg = f"Configured learning rate schedulers: \n"
            for k, v in lr_schedulers.items():
                msg += f"{k}:"
                msg += " " * 4 + v.__class__.__name__ + "\n"
            logger.info(msg)


def _find_differences(
    dict1: Union[dict, DictConfig], dict2: Union[dict, DictConfig], path=""
):
    differences = []
    keys1 = set(dict1.keys())
    keys2 = set(dict2.keys())
    for key in keys1 - keys2:
        differences.append((f"{path}.{key}".strip("."), dict1[key], None))
    for key in keys2 - keys1:
        differences.append((f"{path}.{key}".strip("."), None, dict2[key]))
    for key in keys1 & keys2:
        value1 = dict1[key]
        value2 = dict2[key]
        current_path = f"{path}.{key}".strip(".")

        if isinstance(value1, (dict, DictConfig)) and isinstance(
            value2, (dict, DictConfig)
        ):
            differences.extend(_find_differences(value1, value2, current_path))
        elif value1 != value2:
            differences.append((current_path, value1, value2))

    return differences
