# pytorch_diffusion + derived encoder decoder
from dataclasses import field
from functools import partial
from typing import Dict, List, Optional, Union

import torch
from atria.core.constants import DataKeys
from atria.core.data.datasets.dataset_metadata import DatasetMetadata
from atria.core.models.task_modules.atria_task_module import (
    CheckpointConfig,
)
from atria.core.models.torch_model_builders.base import TorchModelBuilderBase
from atria.core.training.utilities.constants import TrainingStage
from atria.core.utilities.logging import get_logger
from atria.core.utilities.typing import BatchDict
from atria.models.task_modules.diffusion.diffusion import DiffusionModule
from ignite.contrib.handlers import TensorboardLogger

logger = get_logger(__name__)


class LayoutConditionedDiffusionModule(DiffusionModule):
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
        )

        # ldm params
        self._layout_cond_input_key = layout_cond_input_key

    def _prepare_model_kwargs(
        self, batch: BatchDict, stage: TrainingStage
    ) -> torch.Tensor:
        model_kwargs = super()._prepare_model_kwargs(batch, stage)
        model_kwargs["channel_wise_condition"] = batch[self._layout_cond_input_key]
        return model_kwargs

    def required_keys_in_batch(self, stage: TrainingStage) -> List[str]:
        required_keys = (
            [self._input_key, self._layout_cond_input_key, DataKeys.LABEL]
            if self._enable_class_conditioning
            else [self._input_key, self._layout_cond_input_key]
        )
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
        else:
            raise ValueError(f"Stage {stage} not supported")
