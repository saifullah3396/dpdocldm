from typing import List, Optional
import hydra_zen
from atria.core.models.tasks import ModelTasks
from atria.core.models.torch_model_builders.base import TorchModelBuilderBase
from atria.core.models.torch_model_builders.utilities import _filter_args_for_class
from atria.core.utilities.common import _resolve_module_from_path
from atria.core.utilities.logging import get_logger
from torch import nn

logger = get_logger(__name__)


class LocalTorchModelBuilder(TorchModelBuilderBase):
    def __init__(
        self,
        model_name: str = hydra_zen.MISSING,
        model_task: ModelTasks = ModelTasks.image_classification,
        model_cache_dir: Optional[str] = None,
        pretrained: bool = True,
        convert_bn_to_gn: bool = False,
        is_frozen: bool = False,
        required_in_training_step: bool = True,
        pretrained_checkpoint: Optional[str] = None,
        strict: bool = True,
        frozen_keys_patterns: Optional[List[str]] = None,
        unfrozen_keys_patterns: Optional[List[str]] = None,
        **model_config,
    ):
        super().__init__(
            model_name=model_name,
            model_task=model_task,
            model_cache_dir=model_cache_dir,
            pretrained=pretrained,
            convert_bn_to_gn=convert_bn_to_gn,
            is_frozen=is_frozen,
            required_in_training_step=required_in_training_step,
            pretrained_checkpoint=pretrained_checkpoint,
            strict=strict,
            frozen_keys_patterns=frozen_keys_patterns,
            unfrozen_keys_patterns=unfrozen_keys_patterns,
            **model_config,
        )

    def _build(self) -> nn.Module:
        supported_tasks = [v.value for v in ModelTasks]
        assert (
            self._model_task in supported_tasks
        ), f"Task {self._model_task} not supported for {self.__class__.__name__}."

        model_class = _resolve_module_from_path(self._model_name)
        self._model_config = _filter_args_for_class(model_class, self._model_config)
        logger.info("Initializing model with the following parameters:")
        logger.info(self._model_config)
        return model_class(**self._model_config)
