from typing import Optional

import atria  # noqa
import hydra
from atria.core.data.data_modules.atria_data_module import AtriaDataModule
from atria.core.training.utilities.constants import TrainingStage
from atria.core.training.utilities.initialization import _initialize_torch
from atria.core.utilities.logging import get_logger
from atria.core.utilities.print_utils import _print_batch_info
from atria.core.utilities.pydantic_parser import atria_pydantic_parser
from hydra.core.hydra_config import HydraConfig
from omegaconf import OmegaConf  # q
from omegaconf import DictConfig


class AtriaDataProcessor:
    def __init__(
        self,
        data_module: AtriaDataModule,
        seed: int = 42,
        deterministic: bool = False,
        backend: Optional[str] = "nccl",
        n_devices: int = 1,
        stage: Optional[str] = None,
    ):
        self._seed = seed
        self._deterministic = deterministic
        self._backend = backend
        self._n_devices = n_devices
        self._data_module = data_module
        self._stage = stage

        # initialize data loaders
        self._train_dataloader = None
        self._val_dataloader = None
        self._test_dataloader = None

    def init(self, hydra_config: HydraConfig) -> None:
        # initialize training
        _initialize_torch(
            seed=self._seed,
            deterministic=self._deterministic,
        )

        # build data module
        stage = TrainingStage.get(self._stage) if self._stage is not None else None
        self._data_module.setup(stage=stage)

        if self._data_module.train_dataset is not None:
            self._train_dataloader = self._data_module.train_dataloader(shuffle=False)
        if self._data_module.validation_dataset is not None:
            self._val_dataloader = self._data_module.validation_dataloader()
        if self._data_module.test_dataset is not None:
            self._test_dataloader = self._data_module.test_dataloader()

    def visualize_batch(self, hydra_config: Optional[HydraConfig] = None) -> None:
        logger = get_logger(hydra_config=hydra_config)
        # show training batch
        if self._train_dataloader is not None:
            logger.info("Showing training batch...")
            for batch in self._train_dataloader:
                _print_batch_info(batch)
                break

        if self._val_dataloader is not None:
            # show validation batch
            logger.info("Showing validation batch...")
            for batch in self._val_dataloader:
                _print_batch_info(batch)
                break

        # show testing batch
        if self._test_dataloader is not None:
            logger.info("Showing testing batch...")
            for batch in self._test_dataloader:
                _print_batch_info(batch)
                break

    def run(self, hydra_config: HydraConfig) -> None:
        self.init(hydra_config=hydra_config)
        self.visualize_batch(hydra_config=hydra_config)


@hydra.main(
    version_base=None,
    config_path="../../conf",
    config_name="atria_data_processor",
)
def app(cfg: AtriaDataProcessor) -> None:
    from hydra_zen import instantiate

    atria_data_processor: AtriaDataProcessor = instantiate(
        cfg, _convert_="object", _target_wrapper_=atria_pydantic_parser
    )
    hydra_config = HydraConfig.get()
    logger = get_logger(hydra_config=hydra_config)
    try:
        return atria_data_processor.run(hydra_config=hydra_config)
    except Exception as e:
        logger.exception(e)
    finally:
        return None


if __name__ == "__main__":
    app()
