import logging
from functools import partial
from typing import Optional

import atria  # noqa
import hydra
import ignite.distributed as idist
from atria.core.data.data_modules.atria_data_module import AtriaDataModule
from atria.core.models.task_modules.atria_task_module import AtriaTaskModule
from atria.core.training.engines.evaluation import TestEngine
from atria.core.training.engines.utilities import RunConfig
from atria.core.training.utilities.constants import TrainingStage
from atria.core.training.utilities.initialization import (
    _initialize_torch,
    _setup_tensorboard,
)
from atria.core.utilities.common import _msg_with_separator
from atria.core.utilities.logging import get_logger
from atria.core.utilities.pydantic_parser import atria_pydantic_parser
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf  # q
from omegaconf import DictConfig


class AtriaModelEvaluator:
    def __init__(
        self,
        data_module: AtriaDataModule,
        task_module: partial[AtriaTaskModule],
        test_engine: partial[TestEngine],
        output_dir: str,
        seed: int = 42,
        deterministic: bool = False,
        backend: Optional[str] = "nccl",
        n_devices: int = 1,
    ):
        self._output_dir = output_dir
        self._seed = seed
        self._deterministic = deterministic
        self._backend = backend
        self._n_devices = n_devices
        self._data_module = data_module
        self._task_module = task_module
        self._test_engine = test_engine

    @property
    def output_dir(self):
        return self._output_dir

    @property
    def data_module(self):
        return self._data_module

    @property
    def task_module(self):
        return self._task_module

    @property
    def test_engine(self):
        return self._test_engine

    def init(self, hydra_config: HydraConfig, run_config: DictConfig) -> None:
        logger = get_logger(hydra_config=hydra_config)

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
        if hasattr(hydra_config, "runtime"):
            output_dir = hydra_config.runtime.output_dir
        else:
            output_dir = self._output_dir
        tb_logger = _setup_tensorboard(output_dir)

        # build data module
        logger.info(_msg_with_separator("Setting up data module"))
        self._data_module.setup(stage=TrainingStage.test)

        # initialize the task module from partial
        logger.info(_msg_with_separator("Setting up task module"))
        self._task_module = self._task_module(
            dataset_metadata=self._data_module.dataset_metadata,
            tb_logger=tb_logger,
        )
        self._task_module.build_model()

        # initilize the test engine from partial
        logger.info(_msg_with_separator("Setting up test engine"))
        self._test_engine = self._test_engine(
            output_dir=output_dir,
            dataset_cache_dir=self._data_module._dataset_cacher.cache_dir,
            task_module=self._task_module,
            dataloader=self._data_module.test_dataloader(),
            device=device,
            tb_logger=tb_logger,
        )

    def run(self, hydra_config: HydraConfig, run_config: DictConfig) -> None:
        # initialize the model evaluator
        self.init(hydra_config=hydra_config, run_config=run_config)

        # run the test engine
        self._test_engine.run()


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="atria_model_evaluator",
)
def app(cfg: AtriaModelEvaluator) -> None:
    from hydra_zen import instantiate

    atria_model_evaluator: AtriaModelEvaluator = instantiate(
        cfg, _convert_="object", _target_wrapper_=atria_pydantic_parser
    )
    hydra_config = HydraConfig.get()
    logger = get_logger(hydra_config=hydra_config)
    try:
        return atria_model_evaluator.run(
            hydra_config=hydra_config,
            run_config=RunConfig(data=cfg),
        )
    except Exception as e:
        logger.exception(e)
    finally:
        return None


if __name__ == "__main__":
    app()
