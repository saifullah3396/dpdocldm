# pytorch_diffusion + derived encoder decoder
from functools import partial
from typing import Dict, List, Optional, Tuple, Union

import torch
from atria.core.constants import DataKeys
from atria.core.data.datasets.dataset_metadata import DatasetMetadata
from atria.core.models.model_outputs import ModelOutput, VarAutoEncoderModelOutput
from atria.core.models.task_modules.atria_task_module import (
    AtriaTaskModule,
    CheckpointConfig,
)
from atria.core.models.torch_model_builders.base import TorchModelBuilderBase
from atria.core.training.utilities.constants import TrainingStage
from atria.core.utilities.logging import get_logger
from atria.core.utilities.typing import BatchDict
from atria.models.autoencoding.compvis_vae import CompvisAutoencoderKL
from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from ignite.contrib.handlers import TensorboardLogger

logger = get_logger(__name__)


class VarAutoEncodingModule(AtriaTaskModule):
    _REQUIRES_BUILDER_DICT = False
    _SUPPORTED_BUILDERS = [
        "LocalTorchModelBuilder",
        "TorchVisionModelBuilder",
        "TransformersModelBuilder",
        "TimmModelBuilder",
        "DiffusersModelBuilder",
    ]

    def __init__(
        self,
        torch_model_builder: Union[
            partial[TorchModelBuilderBase], Dict[str, partial[TorchModelBuilderBase]]
        ],
        checkpoint_configs: Optional[List[CheckpointConfig]] = None,
        dataset_metadata: Optional[DatasetMetadata] = None,
        tb_logger: Optional[TensorboardLogger] = None,
        input_key: str = DataKeys.IMAGE,
        loss_type: str = "l2",
        rec_loss_weight: float = 1.0,
        kl_loss_weight: float = 1.0,
    ):
        super().__init__(
            torch_model_builder=torch_model_builder,
            checkpoint_configs=checkpoint_configs,
            dataset_metadata=dataset_metadata,
            tb_logger=tb_logger,
        )
        self._input_key = input_key
        self._loss_type = loss_type
        self._rec_loss_weight = rec_loss_weight
        self._kl_loss_weight = kl_loss_weight

    def _build_model(
        self,
    ) -> torch.nn.Module:
        encoder_decoder_model: torch.nn.Module = super()._build_model()

        # initialize classification related stuff
        if self._loss_type == "l2":
            self._loss_fn = torch.nn.MSELoss()
        else:
            raise NotImplementedError()

        return encoder_decoder_model

    def _prepare_input(self, batch: BatchDict) -> torch.Tensor:
        return batch[self._input_key]

    def _compute_loss(
        self,
        input: torch.FloatTensor,
        reconstruction: torch.FloatTensor,
        posterior: DiagonalGaussianDistribution,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        rec_loss = self._loss_fn(input=reconstruction, target=input)  # rec loss
        kl_loss = posterior.kl()  # kl loss
        kl_loss = torch.sum(kl_loss) / kl_loss.shape[0]
        loss = self._rec_loss_weight * rec_loss + self._kl_loss_weight * kl_loss
        return loss, rec_loss, kl_loss

    def _model_forward(
        self,
        input: torch.FloatTensor,
        sample_posterior: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.FloatTensor, DiagonalGaussianDistribution]:
        assert isinstance(self.torch_model, (AutoencoderKL, CompvisAutoencoderKL)), (
            "This method is only supported for models that are instances of "
            "AutoencoderKL or CompvisAutoencoderKL"
        )
        posterior = self.torch_model.encode(input).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.decode(z).sample
        return dec, posterior

    def required_keys_in_batch(self, stage: TrainingStage) -> List[str]:
        if stage == TrainingStage.train:
            return [self._input_key]
        elif stage == TrainingStage.validation:
            return [self._input_key]
        elif stage == TrainingStage.test:
            return [self._input_key]
        elif stage == TrainingStage.predict:
            return []
        else:
            raise ValueError(f"Stage {stage} not supported")

    def training_step(self, batch: BatchDict, **kwargs) -> ModelOutput:
        input = self._prepare_input(batch)
        reconstruction, posterior = self._model_forward(input)
        loss, rec_loss, kl_loss = self._compute_loss(
            input=input, reconstruction=reconstruction, posterior=posterior
        )
        return VarAutoEncoderModelOutput(
            loss=loss,
            rec_loss=rec_loss,
            kl_loss=kl_loss,
            real=input,
            reconstructed=reconstruction,
            posterior=posterior,
        )

    def evaluation_step(self, batch: BatchDict, **kwargs) -> ModelOutput:
        input = self._prepare_input(batch)
        reconstruction, posterior = self._model_forward(input)
        loss, rec_loss, kl_loss = self._compute_loss(
            input=input, reconstruction=reconstruction, posterior=posterior
        )
        return VarAutoEncoderModelOutput(
            loss=loss,
            rec_loss=rec_loss,
            kl_loss=kl_loss,
            real=input,
            reconstructed=reconstruction,
            posterior=posterior,
        )

    def predict_step(self, batch: BatchDict, **kwargs) -> ModelOutput:
        input = self._prepare_input(batch)
        reconstruction, posterior = self._model_forward(input)
        return VarAutoEncoderModelOutput(
            real=input, reconstructed=reconstruction, posterior=posterior
        )
