# pytorch_diffusion + derived encoder decoder
from functools import partial
from typing import Dict, List, Mapping, Optional, Tuple, Union

import torch
from atria.core.constants import DataKeys
from atria.core.data.datasets.dataset_metadata import DatasetMetadata
from atria.core.models.losses.contperceptual import LPIPSWithDiscriminator
from atria.core.models.model_outputs import ModelOutput, VarAutoEncoderGANModelOutput
from atria.core.models.task_modules.atria_task_module import (
    AtriaTaskModule,
    CheckpointConfig,
    TorchModelDict,
)
from atria.core.models.torch_model_builders.base import TorchModelBuilderBase
from atria.core.training.utilities.constants import GANStage
from atria.core.utilities.logging import get_logger
from atria.core.utilities.typing import BatchDict
from atria.models.autoencoding.compvis_vae import CompvisAutoencoderKL
from diffusers import AutoencoderKL
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Engine
from torch import nn

logger = get_logger(__name__)


class GANWrapper(nn.Module):
    def __init__(
        self,
        model: Union[AutoencoderKL, CompvisAutoencoderKL],
        loss: LPIPSWithDiscriminator,
    ):
        super().__init__()
        self.encoder_decoder_model = model
        self.loss = loss

    def forward(self, x):
        return self.encoder_decoder_model(x)


class VarAutoEncodingGANModule(AtriaTaskModule):
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
        loss_type: str = "lpips_disc",
        rec_loss_weight: float = 1.0,
        kl_loss_weight: float = 0.000001,
        disc_start: float = 10001,
        disc_weight: float = 0.5,
    ):
        super().__init__(
            torch_model_builder=torch_model_builder,
            checkpoint_configs=checkpoint_configs,
            dataset_metadata=dataset_metadata,
            tb_logger=tb_logger,
        )

        # vae params
        self._input_key = input_key
        self._loss_type = loss_type
        self._rec_loss_weight = rec_loss_weight
        self._kl_loss_weight = kl_loss_weight

        # disc params if using lpips_disc
        self._disc_start = disc_start
        self._disc_weight = disc_weight

    def _get_last_layer(self):
        assert isinstance(
            self.torch_model.encoder_decoder_model,
            (AutoencoderKL, CompvisAutoencoderKL),
        ), (
            "This method is only supported for models that are instances of "
            "AutoencoderKL or CompvisAutoencoderKL"
        )
        return self.torch_model.encoder_decoder_model.encoder.conv_out.weight

    def _build_model(
        self,
    ) -> Union[torch.nn.Module, TorchModelDict, AutoencoderKL, CompvisAutoencoderKL]:
        encoder_decoder_model: torch.nn.Module = super()._build_model()

        assert isinstance(
            encoder_decoder_model, (AutoencoderKL, CompvisAutoencoderKL)
        ), (
            "This method is only supported for models that are instances of "
            "AutoencoderKL or CompvisAutoencoderKL"
        )

        # initialize classification related stuff
        if self._loss_type == "lpips_disc":
            loss = LPIPSWithDiscriminator(
                disc_start=self._disc_start,
                kl_weight=self._kl_loss_weight,
                pixelloss_weight=self._rec_loss_weight,
                disc_weight=self._disc_weight,
                disc_in_channels=encoder_decoder_model._z_channels,
            )
        else:
            raise NotImplementedError()

        encoder_decoder_model = GANWrapper(encoder_decoder_model, loss)
        return encoder_decoder_model

    @property
    def torch_model(self) -> GANWrapper:
        return self._torch_model

    @property
    def generator(self):
        return self.torch_model.encoder_decoder_model

    @property
    def discriminator(self):
        return self.torch_model.loss.discriminator

    def optimized_parameters(self) -> Mapping[str, List[nn.Parameter]]:
        self.validate_model_built()
        return {
            "generator": list(self.torch_model.encoder_decoder_model.parameters()),
            "discriminator": list(self.torch_model.loss.discriminator.parameters()),
        }

    def _prepare_input(self, batch: BatchDict) -> torch.Tensor:
        return batch[self._input_key]

    def _compute_loss(
        self,
        input: torch.FloatTensor,
        reconstruction: torch.FloatTensor,
        posterior: DiagonalGaussianDistribution,
        gan_stage: GANStage,
        global_step: int,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        # train teh generator or discriminator depending upon the stage
        if gan_stage == GANStage.train_generator:
            loss, logged_outputs = self.torch_model.loss(
                inputs=input,
                reconstructions=reconstruction,
                posteriors=posterior,
                optimizer_idx=0,
                global_step=global_step,
                last_layer=self._get_last_layer(),
            )

        elif gan_stage == GANStage.train_discriminator:
            loss, logged_outputs = self.torch_model.loss(
                inputs=input,
                reconstructions=reconstruction,
                posteriors=posterior,
                optimizer_idx=1,
                global_step=global_step,
                last_layer=self._get_last_layer(),
            )
        return loss, logged_outputs

    def _model_forward(
        self,
        input: torch.FloatTensor,
        sample_posterior: bool = True,
        generator: Optional[torch.Generator] = None,
    ) -> Tuple[torch.FloatTensor, DiagonalGaussianDistribution]:
        posterior = self.torch_model.encoder_decoder_model.encode(input).latent_dist
        if sample_posterior:
            z = posterior.sample(generator=generator)
        else:
            z = posterior.mode()
        dec = self.torch_model.encoder_decoder_model.decode(z).sample
        return dec, posterior, z

    def training_step(
        self, batch: BatchDict, training_engine: Engine, **kwargs
    ) -> ModelOutput:
        assert (
            "gan_stage" in kwargs
        ), "gan_stage must be passed to training_step for this module"
        gan_stage = kwargs["gan_stage"]
        input = self._prepare_input(batch)
        reconstruction, posterior, z = self._model_forward(input)
        loss, logged_outputs = self._compute_loss(
            input=input,
            reconstruction=reconstruction,
            posterior=posterior,
            gan_stage=gan_stage,
            global_step=training_engine.state.iteration,
        )
        reconstruction = reconstruction.clamp(input.min(), input.max())
        for key, value in logged_outputs.items():
            training_engine.state.metrics[key] = value
        return VarAutoEncoderGANModelOutput(
            loss=loss,
            real=input,
            reconstructed=reconstruction,
        )

    def evaluation_step(self, batch: BatchDict, **kwargs) -> ModelOutput:
        input = self._prepare_input(batch)
        reconstruction, posterior, z = self._model_forward(input)
        latent_noise = torch.randn_like(z)
        generated = self.torch_model.encoder_decoder_model.decode(
            latent_noise
        ).sample.clamp(input.min(), input.max())
        reconstruction = reconstruction.clamp(input.min(), input.max())
        return VarAutoEncoderGANModelOutput(
            loss=-1,
            real=input,
            reconstructed=reconstruction,
            generated=generated,
        )

    def predict_step(self, batch: BatchDict, **kwargs) -> ModelOutput:
        input = self._prepare_input(batch)
        reconstruction, posterior, z = self._model_forward(input)
        generated = self.torch_model.encoder_decoder_model.decode(
            torch.randn_like(z)
        ).sample.clamp(input.min(), input.max())
        reconstruction = reconstruction.clamp(input.min(), input.max())
        return VarAutoEncoderGANModelOutput(
            real=input,
            reconstructed=reconstruction,
            generated=generated,
        )
