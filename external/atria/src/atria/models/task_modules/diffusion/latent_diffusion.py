# pytorch_diffusion + derived encoder decoder
import math
from dataclasses import field
from functools import partial
from typing import Dict, List, Optional, Union

import ignite.distributed as idist
import torch
import torchvision
from atria.core.constants import DataKeys
from atria.core.data.datasets.dataset_metadata import DatasetMetadata
from atria.core.models.model_outputs import DiffusionModelOutput, ModelOutput
from atria.core.models.task_modules.atria_task_module import (
    CheckpointConfig,
    TorchModelDict,
)
from atria.core.models.torch_model_builders.base import TorchModelBuilderBase
from atria.core.training.utilities.constants import TrainingStage
from atria.core.utilities.logging import get_logger
from atria.core.utilities.typing import BatchDict
from atria.models.task_modules.diffusion.diffusion import DiffusionModule
from atria.models.task_modules.diffusion.pipelines.diffusion_sampling import (
    DiffusionSamplingPipeline,
)
from atria.models.task_modules.diffusion.utilities import _unnormalize
from diffusers.models.autoencoders.autoencoder_kl import AutoencoderKL
from ignite.contrib.handlers import TensorboardLogger
from ignite.engine import Engine

logger = get_logger(__name__)


class LatentDiffusionModule(DiffusionModule):
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
        enable_xformers_memory_efficient_attention: bool = False,
        gradient_checkpointing: bool = False,
        # diffusion args
        scheduler: str = "diffusers.schedulers.scheduling_ddpm.DDPMScheduler",
        objective: str = "epsilon",
        diffusion_steps: int = 1000,
        inference_diffusion_steps: int = 200,
        noise_schedule: str = "linear",
        snr_gamma: Optional[float] = None,
        clip_sample: bool = False,
        clip_sample_range: float = 1.0,
        unnormalize_output: bool = True,
        noise_multiplicity: Optional[int] = None,
        weighted_timestep_sampling: bool = False,
        weighted_timestep_config: dict = field(
            default_factory=lambda: {
                "distributions": [
                    {"low": 0, "high": 200},
                    {"low": 200, "high": 800},
                    {"low": 800, "high": 1000},
                ],
                "coefficients": [0.015, 0.785, 0.2],
            }
        ),
        reinit_keys_patterns: Optional[List[str]] = None,
        schedule_sampler: str = "uniform",
        # class conditioning args
        enable_class_conditioning: bool = False,
        use_fixed_class_labels: bool = True,
        use_cfg: bool = False,  # by default we set use_cfg to True to generate num_classes+1 embeddings
        # this is the probability of dropping the class label during training, by default class conditional latent diffusion trains without classes used
        cond_drop_prob: float = 0.1,
        guidance_scale: float = 1.0,  # this is the scale of the guidance term, 1 means no guidance, > 1 means guidance
        custom_generated_class_label: Optional[int] = None,
        # final test sampling args
        use_batch_labels: bool = True,
        generate_dataset_on_test: bool = False,
        save_outputs_to_msgpack: bool = True,
        # ldm args
        latent_input_key: str = DataKeys.LATENT_IMAGE,
        compute_scale_factor: bool = False,
        use_precomputed_latents_if_available: bool = False,
        features_scale_factor: Optional[float] = None,
    ):
        super().__init__(
            torch_model_builder=torch_model_builder,
            checkpoint_configs=checkpoint_configs,
            dataset_metadata=dataset_metadata,
            tb_logger=tb_logger,
            input_key=input_key,
            loss_type=loss_type,
            enable_xformers_memory_efficient_attention=enable_xformers_memory_efficient_attention,
            gradient_checkpointing=gradient_checkpointing,
            # diffusion args
            scheduler=scheduler,
            objective=objective,
            diffusion_steps=diffusion_steps,
            inference_diffusion_steps=inference_diffusion_steps,
            noise_schedule=noise_schedule,
            snr_gamma=snr_gamma,
            clip_sample=clip_sample,
            clip_sample_range=clip_sample_range,
            unnormalize_output=unnormalize_output,
            noise_multiplicity=noise_multiplicity,
            weighted_timestep_sampling=weighted_timestep_sampling,
            weighted_timestep_config=weighted_timestep_config,
            reinit_keys_patterns=reinit_keys_patterns,
            schedule_sampler=schedule_sampler,
            enable_class_conditioning=enable_class_conditioning,
            use_fixed_class_labels=use_fixed_class_labels,
            use_cfg=use_cfg,
            cond_drop_prob=cond_drop_prob,
            guidance_scale=guidance_scale,
            custom_generated_class_label=custom_generated_class_label,
            use_batch_labels=use_batch_labels,
            generate_dataset_on_test=generate_dataset_on_test,
            save_outputs_to_msgpack=save_outputs_to_msgpack,
        )

        # ldm params
        self._latent_input_key = latent_input_key
        self._compute_scale_factor = compute_scale_factor
        self._use_precomputed_latents_if_available = (
            use_precomputed_latents_if_available
        )
        self._features_scale_factor = features_scale_factor

    def required_keys_in_batch(self, stage: TrainingStage) -> List[str]:
        required_keys = (
            [self._input_key]
            if not self._use_precomputed_latents_if_available
            else [self._latent_input_key]
        )
        if self._enable_class_conditioning:
            required_keys.append(DataKeys.LABEL)
        if stage == TrainingStage.train:
            return required_keys
        elif stage == TrainingStage.validation:
            return required_keys
        elif stage == TrainingStage.test:
            return required_keys
        elif stage == TrainingStage.visualization:
            return required_keys
        elif stage == TrainingStage.predict:
            return []
        elif stage == "FeatureExtractor":
            return [self._input_key]
        else:
            raise ValueError(f"Stage {stage} not supported")

    def _set_scaling_factor(self, vae, scaling_factor):
        if isinstance(vae, AutoencoderKL):
            vae.register_to_config(scaling_factor=scaling_factor)
        else:
            vae.scaling_factor = scaling_factor

    def _get_scaling_factor(self, vae):
        if isinstance(vae, AutoencoderKL):
            return vae.config.scaling_factor
        else:
            return vae.scaling_factor

    def _build_model(
        self,
    ) -> Union[torch.nn.Module, TorchModelDict]:
        model: TorchModelDict = super()._build_model()

        # make sure the underlying model got an encode and decode method
        assert hasattr(
            model.non_trainable_models, "vae"
        ), "The non_trainable_models models must contain a `vae` underlying model."
        model.non_trainable_models.vae.eval()
        model.non_trainable_models.vae.requires_grad_(False)
        if self._features_scale_factor is not None:
            self._set_scaling_factor(
                model.non_trainable_models.vae, self._features_scale_factor
            )

        return model

    def _build_sampling_pipeline(
        self, return_intermediate_samples: bool = False
    ) -> DiffusionSamplingPipeline:
        return DiffusionSamplingPipeline(
            model=self.reverse_diffusion_model,
            scheduler=self._forward_noise_scheduler,
            vae=self._torch_model.non_trainable_models.vae,
            unnormalize_output=self._unnormalize_output,
            return_intermediate_samples=return_intermediate_samples,
            enable_class_conditioning=self._enable_class_conditioning,
            use_cfg=self._use_cfg,
            guidance_scale=self._guidance_scale,
        )

    @torch.no_grad()
    def compute_scale_factor(
        self, batch
    ):  # compute_scale_factor does not work correctly at the moment with resume checkpoint,
        # after computing, we need to save it in checkpoint for later runs but this does not work atm
        if (
            self._compute_scale_factor
        ):  # and self.current_epoch == 0 and self.global_step == 0 and batch_idx == 0 and not self.restarted_from_ckpt:
            # get data
            input = self._prepare_input(batch, scale_latents=False).detach()
            old_scaling_factor = self._get_scaling_factor(
                self._torch_model.non_trainable_models.vae
            )
            new_scaling_factor = 1.0 / input.flatten().std()
            self._set_scaling_factor(
                self._torch_model.non_trainable_models.vae, new_scaling_factor
            )
            logger.info(
                f"Using std scale factor: {new_scaling_factor} instead of checkpointed scale factor {old_scaling_factor}"
            )

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["scale_factor"] = self._get_scaling_factor(
            self._torch_model.non_trainable_models.vae
        )
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if "scale_factor" in state_dict:
            self._set_scaling_factor(
                self._torch_model.non_trainable_models.vae, state_dict["scale_factor"]
            )

    def encode(self, input: torch.Tensor, scale_latents: bool = True) -> torch.Tensor:
        with torch.cuda.amp.autocast(
            enabled=False
        ):  # with fp16 forward pass on latent, we get nans
            latents = self._torch_model.non_trainable_models.vae.encode(
                input
            ).latent_dist.sample()
            if scale_latents:
                latents = latents * self._get_scaling_factor(
                    self._torch_model.non_trainable_models.vae
                )
            return latents

    @torch.no_grad()
    def _prepare_input(
        self, batch: BatchDict, scale_latents: bool = True
    ) -> torch.Tensor:
        if self._use_precomputed_latents_if_available:
            assert (
                self._latent_input_key in batch
            ), f"Key {self._latent_input_key} not found in batch. "
            latents = batch[self._latent_input_key]
            if scale_latents:
                latents = latents * self._get_scaling_factor(
                    self._torch_model.non_trainable_models.vae
                )
        else:
            input = batch[self._input_key]
            latents = self.encode(input, scale_latents=scale_latents)
        return latents

    def training_step(
        self, batch: BatchDict, training_engine: Engine, **kwargs
    ) -> ModelOutput:
        # we compute scaling factor on first input batch and first iteration
        if training_engine.state.iteration == 1 and training_engine.state.epoch == 1:
            if idist.get_rank() > 0:
                idist.barrier()
            self.compute_scale_factor(batch)

            if idist.get_rank() == 0:
                idist.barrier()
        return super().training_step(
            batch=batch, training_engine=training_engine, **kwargs
        )

    def feature_extractor_step(self, batch: BatchDict, engine, **kwargs) -> ModelOutput:
        self._use_precomputed_latents_if_available = False
        latents = self._prepare_input(batch, scale_latents=False)

        if engine.state.iteration == 1:
            logger.info("Adding feature extraction images to tensorboard")
            reconstructed = self._torch_model.non_trainable_models.vae.decode(
                latents[:4]
            ).sample
            self._tb_logger.writer.add_image(
                f"feature_extractor/{self._input_key}",
                torchvision.utils.make_grid(
                    _unnormalize(batch[self._input_key][:4]),
                    normalize=False,
                    nrow=int(math.sqrt(batch[self._input_key].shape[0])),
                ),
                engine.state.iteration,
            )
            self._tb_logger.writer.add_image(
                f"feature_extractor/reconstructed/{self._input_key}",
                torchvision.utils.make_grid(
                    _unnormalize(reconstructed[:4]),
                    normalize=False,
                    nrow=int(math.sqrt(reconstructed.shape[0])),
                ),
                engine.state.iteration,
            )
            self._tb_logger.writer.flush()
        return {
            self._latent_input_key: latents,
        }

    def evaluation_step(
        self, batch: BatchDict, evaluation_engine, **kwargs
    ) -> ModelOutput:
        if (
            kwargs["stage"] == TrainingStage.test
            and self._generate_dataset_on_test
            and self.check_output_msgpack_file_exists()
        ):
            evaluation_engine.terminate()
            return DiffusionModelOutput(loss=-1)

        latents = self._prepare_input(batch)
        assert (
            self._input_shape == latents.shape[1:]
        ), f"Input shape mismatch. Expected {self._input_shape}, got {latents.shape[1:]}"
        model_kwargs = self._prepare_model_kwargs(batch, stage=kwargs["stage"])
        generated_samples = self._generate_data(
            num_samples=latents.shape[0],
            **model_kwargs,
        ).generated_samples
        if kwargs.get("test_run", False) and self._input_key == DataKeys.IMAGE:
            import matplotlib.pyplot as plt

            # plotting noisy images
            plt.title("Generated images")
            generated_grid = torchvision.utils.make_grid(
                generated_samples.clone().detach().cpu(),
                normalize=False,
                nrow=int(math.sqrt(generated_samples.shape[0])),
            )
            plt.imshow(generated_grid.permute(1, 2, 0).float())
            plt.show()

        # return outputs
        if self._unnormalize_output:
            if self._input_key in batch:
                real = batch[self._input_key]
            else:
                real = self._torch_model.non_trainable_models.vae.decode(
                    latents
                    / self._get_scaling_factor(
                        self._torch_model.non_trainable_models.vae
                    )
                ).sample
            real = _unnormalize(real)

        if idist.get_rank() == 0 and evaluation_engine.state.iteration == 1:
            logger.info("Adding image batch to tensorboard")
            self._tb_logger.writer.add_image(
                f"evaluation/generated/{self._input_key}",
                torchvision.utils.make_grid(
                    generated_samples,
                    normalize=False,
                    nrow=int(math.sqrt(generated_samples.shape[0])),
                ),
                evaluation_engine.state.iteration,
            )
            self._tb_logger.writer.flush()

        # save outputs
        if kwargs["stage"] == TrainingStage.test and self._generate_dataset_on_test:
            self.save_outputs(
                engine=evaluation_engine,
                generated_samples=generated_samples,
                batch=batch,
                model_kwargs=model_kwargs,
            )

        return DiffusionModelOutput(loss=-1, real=real, generated=generated_samples)

    def visualization_step(
        self, batch, evaluation_engine=None, training_engine=None, **kwargs
    ):
        inputs = self._prepare_input(batch)

        # decode latents of original images to visualize
        reconstructed = self.torch_model.non_trainable_models.vae.decode(
            inputs / self._get_scaling_factor(self.torch_model.non_trainable_models.vae)
        ).sample

        if idist.get_rank() == 0:
            logger.info("Adding image batch to tensorboard")
            if self._input_key in batch:
                self._tb_logger.writer.add_image(
                    f"visualization/{self._input_key}",
                    torchvision.utils.make_grid(
                        _unnormalize(batch[self._input_key]),
                        normalize=False,
                        nrow=int(math.sqrt(batch[self._input_key].shape[0])),
                    ),
                    training_engine.state.iteration,
                )
            self._tb_logger.writer.add_image(
                f"visualization/{self._latent_input_key}",
                torchvision.utils.make_grid(
                    inputs,
                    normalize=False,
                    nrow=int(math.sqrt(inputs.shape[0])),
                ),
                training_engine.state.iteration,
            )
            self._tb_logger.writer.add_image(
                f"visualization/reconstructed/{self._input_key}",
                torchvision.utils.make_grid(
                    _unnormalize(reconstructed),
                    normalize=False,
                    nrow=int(math.sqrt(reconstructed.shape[0])),
                ),
                training_engine.state.iteration,
            )
            self._tb_logger.writer.flush()

        super().visualization_step(
            batch=batch,
            evaluation_engine=evaluation_engine,
            training_engine=training_engine,
            **kwargs,
        )
