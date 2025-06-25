import os

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf  # q
from omegaconf import DictConfig
from atria.core.utilities.hydra import yaml_resolvers  # noqa


def initialize_and_run(
    local_rank: int, cfg: DictConfig, hydra_config: DictConfig
) -> None:
    from hydra_zen import instantiate
    from atria.core.task_runners._atria_trainer import AtriaTrainer

    from atria.core.training.engines.utilities import RunConfig
    from atria.core.utilities.pydantic_parser import atria_pydantic_parser
    from omegaconf import OmegaConf  # q

    atria_trainer: AtriaTrainer = instantiate(
        cfg, _convert_="object", _target_wrapper_=atria_pydantic_parser
    )
    return atria_trainer.run(
        hydra_config=hydra_config,
        run_config=RunConfig(data=cfg),
    )


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="atria_trainer",
)
def app(cfg: DictConfig) -> None:
    import ignite.distributed as idist
    from atria.core.utilities.logging import get_logger

    hydra_config = HydraConfig.get()
    logger = get_logger(hydra_config=hydra_config)
    if cfg.n_devices > 1:
        # we run the torch distributed environment with spawn if we have all the gpus on the same script
        # such as when we set --gpus-per-task=N
        ntasks = int(os.environ["SLURM_NTASKS"]) if "SLURM_JOBID" in os.environ else 1
        if ntasks == 1:
            job_id = os.environ["SLURM_JOB_ID"] if "SLURM_JOB_ID" in os.environ else 0
            port = (int(job_id) + 10007) % 16384 + 49152
            logger.info(f"Starting distributed training on port: [{port}]")
            with idist.Parallel(
                backend=cfg.backend,
                nproc_per_node=cfg.n_devices,
                master_port=port,
            ) as parallel:

                return parallel.run(initialize_and_run, cfg, hydra_config)
        elif ntasks == int(cfg.n_devices):
            num_local_gpus = int(os.getenv("SLURM_GPUS_ON_NODE", 1))
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                map(str, range(num_local_gpus))
            )
            with idist.Parallel(backend=cfg.backend) as parallel:
                return parallel.run(initialize_and_run, cfg, hydra_config)
        else:
            raise ValueError(
                f"Your slurm tasks do not match the number of required devices [{ntasks}!={cfg.n_devices}]."
            )
    else:
        try:
            initialize_and_run(0, cfg, hydra_config)
        except Exception as e:
            logger.exception(e)


if __name__ == "__main__":
    app()
