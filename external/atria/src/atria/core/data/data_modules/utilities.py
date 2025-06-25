import warnings
from typing import Any, Optional, Union

from torch.utils.data import DataLoader, Dataset

from atria.core.utilities.logging import get_logger

logger = get_logger(__name__)


def auto_dataloader(dataset: Dataset, **kwargs: Any) -> DataLoader:
    """
    Helper method to create a data loader with distributed configuration.
    Taken from ignite.distributed.auto.auto_dataloader and updated for ChunkedSampler.
    """
    from ignite.distributed import DistributedProxySampler
    from ignite.distributed import utils as idist
    from ignite.distributed.comp_models import xla as idist_xla
    from torch.utils.data import DataLoader, IterableDataset
    from torch.utils.data.distributed import DistributedSampler
    from torch.utils.data.sampler import Sampler
    from wids import ChunkedSampler, DistributedChunkedSampler

    rank = idist.get_rank()
    world_size = idist.get_world_size()

    if world_size > 1:
        if "batch_size" in kwargs and kwargs["batch_size"] >= world_size:
            kwargs["batch_size"] //= world_size

        nproc = idist.get_nproc_per_node()
        if "num_workers" in kwargs and kwargs["num_workers"] >= nproc:
            kwargs["num_workers"] = (kwargs["num_workers"] + nproc - 1) // nproc

        if "batch_sampler" not in kwargs:
            if isinstance(dataset, IterableDataset):
                logger.info(
                    "Found iterable dataset, dataloader will be created without any distributed sampling. "
                    "Please, make sure that the dataset itself produces different data on different ranks."
                )
            else:
                sampler: Optional[
                    Union[DistributedProxySampler, DistributedSampler, Sampler]
                ]
                sampler = kwargs.get("sampler", None)
                if isinstance(sampler, DistributedSampler):
                    if sampler.rank != rank:
                        warnings.warn(
                            f"Found distributed sampler with rank={sampler.rank}, but process rank is {rank}"
                        )
                    if sampler.num_replicas != world_size:
                        warnings.warn(
                            f"Found distributed sampler with num_replicas={sampler.num_replicas}, "
                            f"but world size is {world_size}"
                        )
                elif isinstance(sampler, ChunkedSampler):
                    sampler = DistributedChunkedSampler(
                        dataset,
                        num_replicas=world_size,
                        rank=rank,
                        shuffle=sampler.shuffle,
                        shufflefirst=sampler.shufflefirst,
                        seed=sampler.seed,
                        drop_last=sampler.drop_last,
                        chunk_size=sampler.chunk_size,
                    )
                elif sampler is None:
                    # removes "shuffle" from kwargs if sampler is used
                    shuffle = kwargs.pop("shuffle", True)
                    sampler = DistributedSampler(
                        dataset, num_replicas=world_size, rank=rank, shuffle=shuffle
                    )
                else:
                    sampler = DistributedProxySampler(
                        sampler, num_replicas=world_size, rank=rank
                    )
                kwargs["sampler"] = sampler
        else:
            warnings.warn(
                "Found batch_sampler in provided kwargs. Please, make sure that it is compatible "
                "with distributed configuration"
            )

    if (
        idist.has_xla_support
        and idist.backend() == idist_xla.XLA_TPU
        and kwargs.get("pin_memory", False)
    ):
        # TODO: How about XLA GPU ?
        warnings.warn(
            "Found incompatible options: xla support and pin_memory args equal True. "
            "Argument `pin_memory=False` will be used to construct data loader."
        )
        kwargs["pin_memory"] = False
    else:
        kwargs["pin_memory"] = kwargs.get("pin_memory", "cuda" in idist.device().type)

    log_msg = f"Use data loader kwargs for dataset:\n{{\n"
    for k, v in kwargs.items():
        log_msg += f"\t{k}: {v}\n"
    log_msg += "}"
    logger.info(log_msg)
    dataloader = DataLoader(dataset, **kwargs)

    if (
        idist.has_xla_support
        and idist.backend() == idist_xla.XLA_TPU
        and world_size > 1
    ):
        logger.info("DataLoader is wrapped by `MpDeviceLoader` on XLA")

        mp_device_loader_cls = _MpDeviceLoader
        try:
            from torch_xla.distributed.parallel_loader import MpDeviceLoader

            mp_device_loader_cls = MpDeviceLoader
        except ImportError:
            pass

        mp_dataloader = mp_device_loader_cls(dataloader, idist.device())
        mp_dataloader.sampler = dataloader.sampler  # type: ignore[attr-defined]
        return mp_dataloader

    return dataloader
