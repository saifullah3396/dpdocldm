from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Mapping, Optional, Union

import ignite.distributed as idist
import torch
from atria.core.constants import DEFAULT_OPTIMIZER_PARAMETERS_KEY
from atria.core.data.datasets.dataset_metadata import DatasetMetadata
from atria.core.models.model_outputs import ModelOutput
from atria.core.models.torch_model_builders.base import TorchModelBuilderBase
from atria.core.models.utilities.checkpoints import (
    _filter_with_prefix,
    _load_checkpoint,
    _load_checkpoint_into_model,
)
from atria.core.models.utilities.nn_modules import _summarize_model
from atria.core.training.engines.utilities import _module_to_device
from atria.core.training.utilities.constants import TrainingStage
from atria.core.utilities.common import _get_possible_args, _get_required_args
from atria.core.utilities.logging import get_logger
from atria.core.utilities.typing import BatchDict, TorchNNModule
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Engine
from ignite.handlers import ProgressBar
from torch import nn

logger = get_logger(__name__)


@dataclass
class TorchModelDict:
    trainable_models: nn.ModuleDict
    non_trainable_models: nn.ModuleDict

    def __getitem__(self, key: str) -> nn.Module:
        if key == "trainable_models":
            return self.trainable_models
        elif key == "non_trainable_models":
            return self.non_trainable
        else:
            raise KeyError(f"Key {key} not found in the model dict")


@dataclass
class CheckpointConfig:
    checkpoint_path: str
    checkpoint_state_dict_path: Optional[str] = None
    model_state_dict_path: Optional[str] = None
    load_checkpoint_strict: bool = False


class AtriaTaskModule:
    _REQUIRES_BUILDER_DICT = False
    _SUPPORTED_BUILDERS = ["LocalTorchModelBuilder"]

    def __init__(
        self,
        torch_model_builder: Union[
            partial[TorchModelBuilderBase], Dict[str, partial[TorchModelBuilderBase]]
        ],
        checkpoint_configs: Optional[List[CheckpointConfig]] = None,
        dataset_metadata: Optional[DatasetMetadata] = None,
        tb_logger: Optional[TensorboardLogger] = None,
    ):
        self._torch_model_builder = torch_model_builder
        self._checkpoint_configs = checkpoint_configs
        self._dataset_metadata = dataset_metadata
        self._tb_logger = tb_logger
        self._progress_bar = None
        self._torch_model: Union[TorchNNModule, TorchModelDict] = None
        self._model_built: bool = False

        if (
            self._dataset_metadata is not None
            and self._dataset_metadata.labels is not None
        ):
            logger.info(
                f"Dataset metadata is provided to the model. Labels in metadata = {len(self._dataset_metadata.labels)}"
            )

        if self._REQUIRES_BUILDER_DICT:
            assert isinstance(self._torch_model_builder, dict), (
                f"Model builder must be provided as a dictionary of "
                f"{partial[TorchModelBuilderBase]} when _REQUIRES_BUILDER_DICT is True"
            )
            for builder in self._torch_model_builder.values():
                assert (
                    builder.func.__qualname__ in self._SUPPORTED_BUILDERS
                ), f"Builder {builder.func.__qualname__} not supported for this task module"
        else:
            assert not isinstance(self._torch_model_builder, dict), (
                f"Model builder must be provided as a single instance of "
                f"{partial[TorchModelBuilderBase]} when _REQUIRES_BUILDER_DICT is False"
            )
            assert (
                self._torch_model_builder.func.__qualname__ in self._SUPPORTED_BUILDERS
            ), f"Builder {self._torch_model_builder.func.__qualname__} not supported for this task module. Supported builders are {self._SUPPORTED_BUILDERS}"

    @property
    def model_name(self) -> str:
        if isinstance(self._torch_model_builder, dict):
            return ",".join(
                [builder().model_name for builder in self._torch_model_builder.values()]
            )
        else:
            return self._torch_model_builder().model_name

    @property
    def task_module_name(self) -> str:
        return self.__class__.__name__

    @property
    def torch_model(self) -> Union[TorchNNModule, TorchModelDict]:
        self.validate_model_built()
        return self._torch_model

    def attach_progress_bar(self, progress_bar: ProgressBar):
        self._progress_bar = progress_bar

    def optimized_parameters(self) -> Mapping[str, List[nn.Parameter]]:
        self.validate_model_built()
        if isinstance(self._torch_model, TorchModelDict):
            # by default a single optimizer is applied to all trainable models
            return {
                DEFAULT_OPTIMIZER_PARAMETERS_KEY: list(
                    self._torch_model.trainable_models.parameters()
                )
            }
        elif isinstance(self._torch_model, torch.nn.Module):
            return {
                DEFAULT_OPTIMIZER_PARAMETERS_KEY: list(self._torch_model.parameters())
            }
        else:
            raise ValueError(
                f"Model must be a torch nn.Module or a dictionary of torch nn.Modules. Got {type(self._torch_model)}"
            )

    def ema_modules(self) -> Union["torch.nn.Module", Dict[str, "torch.nn.Module"]]:
        if isinstance(self._torch_model, TorchModelDict):
            return (
                self._torch_model.trainable_models
            )  # ema is only applied to trainable models
        else:
            return self._torch_model

    def _setup_model_config(self) -> Dict[str, Dict[str, Any]]:
        if isinstance(self._torch_model_builder, dict):
            if self._dataset_metadata.labels is not None:
                return {
                    key: dict(num_labels=len(self._dataset_metadata.labels))
                    for key in self._torch_model_builder
                }
            else:
                return {key: {} for key in self._torch_model_builder}
        else:
            if self._dataset_metadata.labels is not None:
                return dict(num_labels=len(self._dataset_metadata.labels))
            else:
                return {}

    def _init_weights(self) -> None:
        pass

    def build_model(
        self,
    ) -> None:
        pass

        # models sometimes download pretrained checkpoints when initializing. Only download it on rank 0
        if idist.get_rank() > 0:  # stop all ranks > 0
            idist.barrier()

        # load the checkpoint if available
        checkpoint_mapping = self._setup_checkpoint_mapping()

        # build the underlying nn model[s]
        self._torch_model = self._build_model()

        # load the checkpoint if available
        if checkpoint_mapping is not None:
            self._load_checkpoints(
                model=self._torch_model, checkpoint_mapping=checkpoint_mapping
            )

        self._validate_nn_structure()

        # initialize weights of the model if required. Note that this happens after the pretrained checkpoint
        self._init_weights()

        # wait for rank 0 to download checkpoints
        if idist.get_rank() == 0:
            idist.barrier()

        # set model built flag
        self._model_built = True

        # batch validation flag
        self._batch_validated = False

        # summarize the model
        self.print_summary()

    def _setup_checkpoint_mapping(self) -> None:
        # if there is a checkpoint, we load dataset metadata from the checkpoint
        if self._checkpoint_configs is None:
            return

        if not isinstance(self._checkpoint_configs, list):
            self._checkpoint_configs = [self._checkpoint_configs]

        checkpoint_mapping = {}
        for checkpoint_config in self._checkpoint_configs:
            logger.info(
                f"Loading checkpoint from checkpoint path: {checkpoint_config.checkpoint_path}"
            )
            if checkpoint_config.checkpoint_path == "":
                logger.info("No checkpoint path provided. Skipping checkpoint loading")
                continue
            checkpoint = _load_checkpoint(checkpoint_config.checkpoint_path)
            if "dataset_metadata" in checkpoint:
                logger.debug(
                    f"Loading dataset metadata from checkpoint: {checkpoint['dataset_metadata']}"
                )
                if self._dataset_metadata is None:
                    self._dataset_metadata = DatasetMetadata()

                self._dataset_metadata.load_state_dict(checkpoint["dataset_metadata"])

            if checkpoint_config.checkpoint_state_dict_path is not None:
                for path in checkpoint_config.checkpoint_state_dict_path.split("."):
                    if path in checkpoint:
                        checkpoint = checkpoint[path]
                    elif len([k for k in checkpoint.keys() if path in k]) > 0:
                        checkpoint = _filter_with_prefix(
                            checkpoint,
                            path,
                        )
                    else:
                        raise KeyError(
                            f"Target key {path} not found in the checkpoint. Available keys = {checkpoint.keys()}"
                        )
            checkpoint_mapping[checkpoint_config.model_state_dict_path] = {
                "checkpoint": checkpoint,
                "checkpoint_state_dict_path": checkpoint_config.checkpoint_state_dict_path,
                "strict": checkpoint_config.load_checkpoint_strict,
            }
        return checkpoint_mapping

    def _load_checkpoints(
        self, model: torch.nn.Module, checkpoint_mapping: Dict[str, Any]
    ) -> None:
        for model_state_dict_path, mapping in checkpoint_mapping.items():
            checkpoint = mapping["checkpoint"]
            strict = mapping["strict"]
            checkpoint_state_dict_path = mapping["checkpoint_state_dict_path"]
            logger.info(
                f"Loading the checkpoint from path [{checkpoint_state_dict_path}] into the model at path "
                f"[{model_state_dict_path}] with strict={strict}"
            )
            _load_checkpoint_into_model(
                model=model,
                checkpoint=checkpoint,
                model_state_dict_path=model_state_dict_path,
                strict=strict,
            )

    def _build_model(
        self,
    ) -> Union[torch.nn.Module, TorchModelDict]:
        if isinstance(self._torch_model_builder, dict):
            model_config = self._setup_model_config()
            assert sorted(model_config.keys()) == sorted(
                self._torch_model_builder.keys()
            ), "Model kwargs must be a dictionary with the same keys as the model builders"

            trainable_models = {}
            non_trainable_models = {}
            for key, builder in self._torch_model_builder.items():
                builder = builder(**model_config[key])
                model = builder.build(
                    key=key,
                )
                if not builder.is_frozen:
                    trainable_models[key] = model
                else:
                    non_trainable_models[key] = model

            # when multiple models are available, we merge some models together under a single wrapped model if they
            # are to be used for training. This is necessary for distributed training where we need to have a single model
            # wrapped under DistributedDataParallel for the optimizer to work correctly
            # therefore we first check if the model gradients are required during training, if so we merge the models
            trainable_models = nn.ModuleDict(trainable_models)
            non_trainable_models = nn.ModuleDict(non_trainable_models)

            return TorchModelDict(
                trainable_models=trainable_models,
                non_trainable_models=non_trainable_models,
            )
        else:
            model_config = self._setup_model_config()
            return self._torch_model_builder(**model_config).build()

    def _validate_nn_structure(self) -> None:
        if isinstance(self._torch_model, TorchModelDict):
            for key, model in self._torch_model.trainable_models.items():
                assert isinstance(
                    model, nn.Module
                ), f"Model must be a torch nn.Module. Got {type(model)} for key {key}"
            for key, model in self._torch_model.non_trainable_models.items():
                assert isinstance(
                    model, nn.Module
                ), f"Model must be a torch nn.Module. Got {type(model)} for key {key}"
        else:
            assert isinstance(
                self._torch_model, nn.Module
            ), f"Model must be a torch nn.Module. Got {type(self._torch_model)}"

    def validate_model_built(self) -> None:
        assert (
            self._model_built
        ), "Model must be built before training. Call build_model() first"

    def to_device(self, device: torch.device, sync_bn: bool = False) -> None:
        # setup model to device
        if isinstance(self._torch_model, TorchModelDict):
            # first we put all trainable models to the device with distributed sync
            logger.info(f"Moving trainable_models to {device}")
            self._torch_model.trainable_models = _module_to_device(
                self._torch_model.trainable_models,
                device=device,
                sync_bn=sync_bn,
                prepare_for_distributed=True,
            )

            # then we put all non-trainable models to the device without distributed sync
            logger.info(f"Moving non_trainable_models to {device}")
            self._torch_model.non_trainable_models = _module_to_device(
                self._torch_model.non_trainable_models,
                device=device,
                sync_bn=sync_bn,
                prepare_for_distributed=False,
            )
        else:
            # for a single model, we put it to the device with distributed sync as it is assumed to be trainable
            logger.info(f"Moving model to {device}")
            self._torch_model = _module_to_device(
                self._torch_model,
                device=device,
                sync_bn=sync_bn,
                prepare_for_distributed=True,
            )

    def train(self):
        if isinstance(self._torch_model, TorchModelDict):
            self._torch_model.trainable_models.train()
        else:
            self._torch_model.train()

    def eval(self):
        if isinstance(self._torch_model, TorchModelDict):
            self._torch_model.trainable_models.eval()
            self._torch_model.non_trainable_models.eval()
        else:
            self._torch_model.eval()

    def half(self):
        if isinstance(self._torch_model, TorchModelDict):
            self._torch_model.trainable_models.half()
            self._torch_model.non_trainable_models.half()
        else:
            self._torch_model.half()

    def required_keys_in_batch(self, stage: TrainingStage) -> List[str]:
        return []

    def training_step(
        self,
        batch: BatchDict,
        training_engine: Optional[Engine] = None,
        **kwargs,
    ) -> ModelOutput:
        return self._model_forward(batch)

    def evaluation_step(
        self,
        batch: BatchDict,
        evaluation_engine: Optional[Engine] = None,
        training_engine: Optional[Engine] = None,
        stage: TrainingStage = TrainingStage.test,
        **kwargs,
    ) -> ModelOutput:
        return self._model_forward(batch)

    def predict_step(
        self,
        batch: BatchDict,
        evaluation_engine: Optional[Engine] = None,
        **kwargs,
    ) -> ModelOutput:
        return self._model_forward(batch)

    def visualization_step(
        self,
        batch: BatchDict,
        evaluation_engine: Optional[Engine] = None,
        training_engine: Optional[Engine] = None,
        **kwargs,
    ) -> None:
        pass

    def _model_forward(self, batch: BatchDict) -> Any:
        assert (
            self._model_built
        ), "Model must be built before training. Call build_model() first"
        if isinstance(self._torch_model, dict):
            raise NotImplementedError(
                "Model forward must be implemented in the task module when multiple models are used"
            )
        self._validate_batch_keys_for_model_forward(batch)
        return self._torch_model(**batch)

    def _filter_batch_keys_for_model_forward(self, batch):
        valid_params = set(_get_possible_args(self._torch_model.forward).keys())
        filtered_batch = {k: v for k, v in batch.items() if k in valid_params}
        return filtered_batch

    def _validate_batch_keys_for_model_forward(self, batch: BatchDict):
        if self._batch_validated:
            return
        possible_model_args = _get_possible_args(self._torch_model.forward)
        required_model_args = _get_required_args(self._torch_model.forward)
        for required_arg in required_model_args:
            assert (
                required_arg not in batch
            ), f"Required argument '{required_arg}' is missing in the batch, batch keys: {list(batch.keys())}"
        for key in batch.keys():
            assert key in possible_model_args, (
                f"Key '{key}' is not a valid argument for the model = {self._torch_model.__class__.__name__}, "
                f"possible arguments are: {list(possible_model_args.keys())}"
            )
        self._batch_validated = True

    def print_summary(self) -> None:
        logger.info("Model loaded with following components:")
        _summarize_model(self)

    def state_dict(self):
        # initialize checkpoint dict
        state_dict = {}

        # add dataset metadata to checkpoint
        if self._dataset_metadata is not None:
            state_dict["dataset_metadata"] = self._dataset_metadata.state_dict()

        # add model to checkpoint
        if isinstance(self.torch_model, TorchModelDict):
            state_dict["trainable_models"] = (
                self.torch_model.trainable_models.state_dict()
            )
            state_dict["non_trainable_models"] = (
                self.torch_model.non_trainable_models.state_dict()
            )
        else:
            state_dict["model"] = self.torch_model.state_dict()
        return state_dict

    def load_state_dict(self, state_dict):
        self._dataset_metadata.load_state_dict(state_dict["dataset_metadata"])

        if "trainable_models" in state_dict:
            assert (
                "non_trainable_models" in state_dict
            ), "Both trainable and non trainable models must be present in the state dict"
            self._torch_model.trainable_models.load_state_dict(
                state_dict["trainable_models"]
            )
            self._torch_model.non_trainable_models.load_state_dict(
                state_dict["non_trainable_models"]
            )
        elif "model" in state_dict:
            self._torch_model.load_state_dict(state_dict["model"])
