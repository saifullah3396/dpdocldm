# pytorch_diffusion + derived encoder decoder
import math
from dataclasses import field
from functools import partial
from typing import Dict, List, Optional, Union

import torch
import torchvision
from atria.core.constants import DataKeys
from atria.core.data.datasets.dataset_metadata import DatasetMetadata
from atria.core.models.model_outputs import ModelOutput
from atria.core.models.task_modules.atria_task_module import (
    CheckpointConfig,
)
from atria.core.models.torch_model_builders.base import TorchModelBuilderBase
from atria.core.training.utilities.constants import TrainingStage
from atria.core.utilities.logging import get_logger
from atria.core.utilities.typing import BatchDict
from atria.models.task_modules.diffusion.latent_diffusion import LatentDiffusionModule
from atria.models.task_modules.diffusion.utilities import (
    _dropout_label_for_cfg_training,
    _unnormalize,
)
from ignite.contrib.handlers import TensorboardLogger
from torch import nn

logger = get_logger(__name__)


class LayoutConditionedLatentDiffusionModule(LatentDiffusionModule):
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
        # layout cond args
        layout_cond_input_key: str = DataKeys.LAYOUT_MASK,
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
            latent_input_key=latent_input_key,
            compute_scale_factor=compute_scale_factor,
            use_precomputed_latents_if_available=use_precomputed_latents_if_available,
            features_scale_factor=features_scale_factor,
        )

        # ldm params
        self._layout_cond_input_key = layout_cond_input_key
        self._layout_cond_features_scale_factor = None

    def _get_input_shape(self, model: nn.Module):
        config = model.config if hasattr(model, "config") else model
        if hasattr(config, "sample_size") and hasattr(config, "in_channels"):
            return (
                config["in_channels"],
                *config["sample_size"],
            )
        elif hasattr(config, "input_shape"):
            return tuple(config["input_shape"])
        elif hasattr(config, "in_channels") and hasattr(config, "image_size"):
            return (config.in_channels // 2, config.image_size, config.image_size)
        else:
            raise ValueError("Could not determine input shape")

    def required_keys_in_batch(self, stage: TrainingStage) -> List[str]:
        required_keys = (
            [self._input_key, self._layout_cond_input_key]
            if not self._use_precomputed_latents_if_available
            else [self._latent_input_key, self._layout_cond_input_key]
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

    @torch.no_grad()
    def _prepare_layout_cond_input(
        self, batch: BatchDict, scale_latents: bool = True
    ) -> torch.Tensor:
        if self._use_precomputed_latents_if_available:
            assert (
                self._layout_cond_input_key in batch
            ), f"Key {self._layout_cond_input_key} not found in batch. "
            cond_latents = batch[self._layout_cond_input_key]
        else:
            cond_input = batch[self._layout_cond_input_key]
            with torch.cuda.amp.autocast(
                enabled=False
            ):  # with fp16 forward pass on latent, we get nans
                cond_latents = self._torch_model.non_trainable_models.vae.encode(
                    cond_input
                ).latent_dist.sample()
        if scale_latents:
            if self._layout_cond_features_scale_factor is None:
                self._layout_cond_features_scale_factor = (
                    1.0 / cond_latents.flatten().std()
                )
                logger.info(
                    f"Using std scale factor for layout condition: {self._layout_cond_features_scale_factor}"
                )
            cond_latents = cond_latents * self._layout_cond_features_scale_factor
        return cond_latents

    def feature_extractor_step(self, batch: BatchDict, engine, **kwargs) -> ModelOutput:
        self._use_precomputed_latents_if_available = False

        # for image, layout_mask in zip(
        #     batch[DataKeys.IMAGE][:4], batch[DataKeys.LAYOUT_MASK][:4]
        # ):
        #     import matplotlib.pyplot as plt

        #     fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        #     axes[0].imshow(image.cpu().permute(1, 2, 0), cmap="gray")
        #     axes[0].set_title("Original Image")
        #     axes[0].axis("off")

        #     axes[1].imshow(layout_mask.cpu().permute(1, 2, 0).float(), cmap="gray")
        #     axes[1].set_title("Layout Mask")
        #     axes[1].axis("off")

        #     plt.show()

        latents = self._prepare_input(batch, scale_latents=False)
        cond_latents = self._prepare_layout_cond_input(batch, scale_latents=False)

        if engine.state.iteration == 1:
            logger.info("Adding feature extraction images to tensorboard")
            reconstructed = self._torch_model.non_trainable_models.vae.decode(
                latents[:4]
            ).sample
            reconstructed_cond = self._torch_model.non_trainable_models.vae.decode(
                cond_latents[:4]
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
            self._tb_logger.writer.add_image(
                f"feature_extractor/{self._layout_cond_input_key}",
                torchvision.utils.make_grid(
                    _unnormalize(batch[self._layout_cond_input_key][:4]),
                    normalize=False,
                    nrow=int(math.sqrt(batch[self._layout_cond_input_key].shape[0])),
                ),
                engine.state.iteration,
            )
            self._tb_logger.writer.add_image(
                f"feature_extractor/reconstructed/{self._layout_cond_input_key}",
                torchvision.utils.make_grid(
                    _unnormalize(reconstructed_cond[:4]),
                    normalize=False,
                    nrow=int(math.sqrt(reconstructed_cond.shape[0])),
                ),
                engine.state.iteration,
            )
            self._tb_logger.writer.flush()
        return {
            self._latent_input_key: latents,
            self._layout_cond_input_key: cond_latents,
        }

    def _prepare_model_kwargs(
        self, batch: BatchDict, stage: TrainingStage
    ) -> torch.Tensor:
        model_kwargs = super()._prepare_model_kwargs(batch, stage)
        model_kwargs["channel_wise_condition"] = self._prepare_layout_cond_input(batch)
        return model_kwargs

    def _prepare_model_kwargs(
        self, batch: BatchDict, stage: TrainingStage
    ) -> torch.Tensor:
        model_kwargs = {}
        if self._enable_class_conditioning:
            class_labels = batch[DataKeys.LABEL]
            if stage == TrainingStage.train:
                if self._use_cfg:
                    class_labels = _dropout_label_for_cfg_training(
                        class_labels=class_labels,
                        num_classes=len(self._dataset_metadata.labels),
                        probability=self._cond_drop_prob,
                        device=class_labels.device,
                    )
            if stage == TrainingStage.visualization:
                class_labels = batch[DataKeys.LABEL]
                logger.info(f"Using fixed class labels for evaluation: {class_labels}")
            if stage == TrainingStage.test:
                class_labels = batch[DataKeys.LABEL]
                logger.info(f"Using fixed class labels for evaluation: {class_labels}")

            model_kwargs["class_labels"] = class_labels
        model_kwargs["channel_wise_condition"] = self._prepare_layout_cond_input(batch)
        return model_kwargs

    def state_dict(self):
        state_dict = super().state_dict()
        state_dict["layout_cond_scale_factor"] = self._layout_cond_features_scale_factor
        return state_dict

    def load_state_dict(self, state_dict):
        super().load_state_dict(state_dict)
        if "layout_cond_scale_factor" in state_dict:
            self._layout_cond_features_scale_factor = state_dict[
                "layout_cond_scale_factor"
            ]
