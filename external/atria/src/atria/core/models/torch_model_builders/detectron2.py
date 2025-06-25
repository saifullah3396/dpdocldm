from typing import List, Optional
import hydra_zen
import torch
from atria.core.models.tasks import ModelTasks
from atria.core.models.torch_model_builders.base import TorchModelBuilderBase
from atria.core.utilities.logging import get_logger
from detectron2.config import CfgNode as CN
from detectron2.config import get_cfg
from detectron2.modeling import build_model

logger = get_logger(__name__)


def add_vit_config(cfg: CN) -> None:
    """
    Add Vision Transformer (ViT) specific configuration to the Detectron2 config.

    Args:
        cfg (CN): The Detectron2 configuration node to be modified.
    """
    _C = cfg

    _C.MODEL.VIT = CN()
    _C.MODEL.VIT.NAME = ""
    _C.MODEL.VIT.OUT_FEATURES = ["layer3", "layer5", "layer7", "layer11"]
    _C.MODEL.VIT.IMG_SIZE = [224, 224]
    _C.MODEL.VIT.POS_TYPE = "shared_rel"
    _C.MODEL.VIT.DROP_PATH = 0.0
    _C.MODEL.VIT.MODEL_KWARGS = "{}"


class Detectron2ModelBuilder(TorchModelBuilderBase):
    def __init__(
        self,
        model_name: str = hydra_zen.MISSING,
        model_task: ModelTasks = ModelTasks.object_detection,
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

    def _build(self) -> torch.nn.Module:
        """
        Initialize the Detectron2 model based on the provided configuration.

        Returns:
            torch.nn.Module: The initialized Detectron2 model.
        """
        assert (
            "cfg_path" in self._model_config
        ), "cfg_path must be provided for detectron2 model initialization."

        assert self._model_task in [
            ModelTasks.object_detection,
        ], f"Task {self._model_task} not supported for {self.__class__.__name__}."

        # Instantiate Detectron2 config
        cfg = get_cfg()

        # Add ViT config if specified
        if self._model_config.get("add_vit_config", False):
            add_vit_config(cfg)

        # Merge configuration from file
        cfg.merge_from_file(self._model_config["cfg_path"])

        # Set visualization period if specified
        if "vis_period" in self._model_config:
            cfg.VIS_PERIOD = self._model_config["vis_period"]

        logger.info("Building model with the following config: {}".format(cfg))

        # Instantiate model
        return build_model(cfg)
