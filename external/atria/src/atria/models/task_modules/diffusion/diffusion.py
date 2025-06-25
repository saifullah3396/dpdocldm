# pytorch_diffusion + derived encoder decoder
import math
from dataclasses import field
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

import ignite.distributed as idist
import torch
import torch.nn.functional as F
import torchvision
from atria.core.constants import DataKeys
from atria.core.data.datasets.dataset_metadata import DatasetMetadata
from atria.core.models.model_outputs import DiffusionModelOutput, ModelOutput
from atria.core.models.task_modules.atria_task_module import (
    AtriaTaskModule,
    CheckpointConfig,
    TorchModelDict,
)
from datadings.writer import FileWriter
from atria.core.models.torch_model_builders.base import TorchModelBuilderBase
from atria.core.training.utilities.constants import TrainingStage
from atria.core.utilities.common import _resolve_module_from_path
from atria.core.utilities.logging import get_logger, print_once
from atria.core.utilities.typing import BatchDict
from atria.models.task_modules.diffusion.pipelines.diffusion_sampling import (
    DiffusionSamplingPipeline,
    DiffusionSamplingPipelineOutput,
)
from ignite.engine import Engine
from atria.models.task_modules.diffusion.resample import (
    LossAwareSampler,
    LossSecondMomentResampler,
    SpeedDiffusionSampler,
    UniformSampler,
)
from atria.models.task_modules.diffusion.utilities import (
    _compute_snr,
    _dropout_label_for_cfg_training,
    _tensor_image_to_bytes,
    _unnormalize,
)
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from ignite.contrib.handlers import TensorboardLogger
from matplotlib import pyplot as plt
from torch import nn

logger = get_logger(__name__)


def sample_weighted_timesteps(config: dict, sample_size: int, device: torch.device):
    import numpy as np

    distributions = config["distributions"]
    coefficients = np.array(config["coefficients"])
    coefficients /= coefficients.sum()  # in case these did not add up to 1

    num_distr = len(distributions)
    data = np.zeros((sample_size, num_distr))
    for idx, distr in enumerate(distributions):
        data[:, idx] = np.random.uniform(size=(sample_size,), **distr)
    random_idx = np.random.choice(
        np.arange(num_distr), size=(sample_size,), p=coefficients
    )
    return torch.from_numpy(data[np.arange(sample_size), random_idx]).long().to(device)


def _convert_to_lora(model: nn.Module, lora_rank: int = 4) -> nn.Module:
    from peft import LoraConfig, get_peft_model

    model.requires_grad_(False)
    diffusers_lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        init_lora_weights="gaussian",
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
    )
    model = get_peft_model(model, diffusers_lora_config)
    logger.info("Applied LoRA to the model")
    trainable_params = {
        name: param.requires_grad
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    logger.info(f"Current trainable parameters: {trainable_params}")
    return model


class DiffusionModule(AtriaTaskModule):
    _REQUIRES_BUILDER_DICT = True
    _SUPPORTED_BUILDERS = [
        "LocalTorchModelBuilder",
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
        enable_xformers_memory_efficient_attention: bool = False,
        gradient_checkpointing: bool = False,
        # diffusion args
        scheduler: str = "diffusers.schedulers.scheduling_ddpm.DDPMScheduler",
        objective: str = "epsilon",
        diffusion_steps: int = 1000,
        inference_diffusion_steps: int = 200,
        noise_schedule: str = "linear",
        snr_gamma: Optional[float] = None,
        clip_sample: bool = True,
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
    ):
        super().__init__(
            torch_model_builder=torch_model_builder,
            checkpoint_configs=checkpoint_configs,
            dataset_metadata=dataset_metadata,
            tb_logger=tb_logger,
        )
        self._input_key = input_key
        self._loss_type = loss_type

        # params
        self._enable_xformers_memory_efficient_attention = (
            enable_xformers_memory_efficient_attention
        )
        self._gradient_checkpointing = gradient_checkpointing

        # diffusion params
        self._scheduler = scheduler
        self._objective = objective
        self._diffusion_steps = diffusion_steps
        self._inference_diffusion_steps = inference_diffusion_steps
        self._noise_schedule = noise_schedule
        self._snr_gamma = snr_gamma
        self._clip_sample = clip_sample
        self._clip_sample_range = clip_sample_range
        self._unnormalize_output = unnormalize_output
        self._noise_multiplicity = noise_multiplicity
        self._weighted_timestep_sampling = weighted_timestep_sampling
        self._weighted_timestep_config = weighted_timestep_config
        self._apply_lora = False
        self._reinit_keys_patterns = reinit_keys_patterns
        self._schedule_sampler = schedule_sampler

        # class conditioning params
        self._enable_class_conditioning = enable_class_conditioning
        self._use_fixed_class_labels = use_fixed_class_labels
        self._use_cfg = use_cfg
        self._cond_drop_prob = cond_drop_prob
        self._guidance_scale = guidance_scale
        self._custom_generated_class_label = custom_generated_class_label

        # final test sampling args
        self._use_labels_from_batch = use_batch_labels
        self._generate_dataset_on_test = generate_dataset_on_test
        self._save_outputs_to_msgpack = save_outputs_to_msgpack
        self._msgpack_filewriter = None
        self._is_save_outputs_done = False

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @guidance_scale.setter
    def guidance_scale(self, value):
        self._guidance_scale = value
        if self._save_outputs_to_msgpack:
            self._msgpack_filewriter = None
            self._is_save_outputs_done = False

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
            return (config.in_channels, config.image_size, config.image_size)
        else:
            raise ValueError("Could not determine input shape")

    def _build_model(
        self,
    ) -> Union[torch.nn.Module, TorchModelDict]:
        model: TorchModelDict = super()._build_model()

        # make sure the underlying model got an encode and decode method
        assert hasattr(
            model.trainable_models, "reverse_diffusion_model"
        ), "The trainable_models models must contain a `reverse_diffusion_model` underlying model."

        if self._enable_xformers_memory_efficient_attention:
            from diffusers.utils.import_utils import is_xformers_available
            from packaging import version

            if is_xformers_available():
                import xformers

                xformers_version = version.parse(xformers.__version__)
                if xformers_version == version.parse("0.0.16"):
                    logger.warn(
                        "xFormers 0.0.16 cannot be used for training in some GPUs. If you observe problems "
                        "during training, please update xFormers to at least 0.0.17. "
                        "See https://huggingface.co/docs/diffusers/main/en/optimization/xformers for more details."
                    )
                model.trainable_models.reverse_diffusion_model.enable_xformers_memory_efficient_attention()

        if self._gradient_checkpointing:
            if hasattr(
                model.trainable_models.reverse_diffusion_model,
                "enable_gradient_checkpointing",
            ):
                model.trainable_models.reverse_diffusion_model.enable_gradient_checkpointing()

        # build forward noise scheduler
        self._forward_noise_scheduler = self.build_forward_noise_scheduler()

        # initialize classification related stuff
        if self._loss_type == "l2":
            self._loss_fn = F.mse_loss
        elif self._loss_type == "l1":
            self._loss_fn = F.l1_loss
        else:
            raise NotImplementedError()

        # define the input shape for the model from model config
        self._input_shape = self._get_input_shape(
            model.trainable_models.reverse_diffusion_model
        )

        logger.info(
            "Setting up diffusion with the following parameters:\n"
            f"  input_key: {self._input_key}\n"
            f"  loss_type: {self._loss_type}\n"
            f"  scheduler: {self._scheduler}\n"
            f"  objective: {self._objective}\n"
            f"  diffusion_steps: {self._diffusion_steps}\n"
            f"  inference_diffusion_steps: {self._inference_diffusion_steps}\n"
            f"  noise_schedule: {self._noise_schedule}\n"
            f"  snr_gamma: {self._snr_gamma}\n"
            f"  clip_sample: {self._clip_sample}\n"
            f"  clip_sample_range: {self._clip_sample_range}\n"
            f"  unnormalize_output: {self._unnormalize_output}\n"
            f"  schedule_sampler: {self._schedule_sampler}\n"
            f"  enable_class_conditioning: {self._enable_class_conditioning}\n"
            f"  use_fixed_class_labels: {self._use_fixed_class_labels}\n"
            f"  use_cfg: {self._use_cfg}\n"
            f"  cond_drop_prob: {self._cond_drop_prob}\n"
            f"  guidance_scale: {self._guidance_scale}\n"
            f"  custom_generated_class_label: {self._custom_generated_class_label}"
        )

        return model

    @property
    def reverse_diffusion_model(self):
        if hasattr(self._torch_model.trainable_models, "base_model"):
            return (
                self._torch_model.trainable_models.base_model.model.reverse_diffusion_model
            )
        else:
            return self._torch_model.trainable_models.reverse_diffusion_model

    @property
    def forward_noise_scheduler(self):
        return self._forward_noise_scheduler

    def build_model(self) -> Union[torch.nn.Module, TorchModelDict]:
        super().build_model()
        if self._apply_lora:
            if isinstance(self._torch_model, TorchModelDict):
                self._torch_model.trainable_models = _convert_to_lora(
                    self._torch_model.trainable_models
                )
            else:
                self._torch_model = _convert_to_lora(self._torch_model)

        if self._reinit_keys_patterns is not None:
            logger.info(
                f"Reinitializing model layers based on patterns: {self._reinit_keys_patterns}"
            )
            reinitialized_layers = []
            for name, layer in self._torch_model.trainable_models.named_modules():
                for pattern in self._reinit_keys_patterns:
                    if pattern in name:
                        if isinstance(layer, (torch.nn.Conv2d, torch.nn.Linear)):
                            layer.reset_parameters()
                            reinitialized_layers.append(name)
                        break
            logger.info(f"Reinitialized trainable parameters: {reinitialized_layers}")

    def build_forward_noise_scheduler(self, scheduler_override: partial = None):
        scheduler_class = scheduler_override if scheduler_override else self._scheduler

        if isinstance(scheduler_class, str):
            scheduler_class = _resolve_module_from_path(scheduler_class)

        # initialize the scheduler
        forward_noise_scheduler: DDPMScheduler = scheduler_class(
            num_train_timesteps=self._diffusion_steps,
            beta_schedule=self._noise_schedule,
            prediction_type=self._objective,
            clip_sample=self._clip_sample,
            clip_sample_range=self._clip_sample_range,
        )

        # set the number of timesteps for inference
        forward_noise_scheduler.set_timesteps(self._inference_diffusion_steps)

        # Get the target for loss depending on the prediction type
        if self._objective is not None:
            # set objective of scheduler if defined
            forward_noise_scheduler.register_to_config(prediction_type=self._objective)

        logger.info(
            f"Loaded {scheduler_class} forward noise scheduler with parameters:"
        )
        logger.info(f"Number of diffusion steps: {self._diffusion_steps}")
        logger.info(f"Objective: {self._objective}")
        logger.info(f"Schedule type: {self._noise_schedule}")
        logger.info(f"Clip sample: {self._clip_sample}")
        logger.info(f"Clip sample range: {self._clip_sample_range}")

        if self._schedule_sampler == "uniform":
            self._schedule_sampler = UniformSampler(self._diffusion_steps)
        elif self._schedule_sampler == "loss-second-moment":
            self._schedule_sampler = LossSecondMomentResampler(self._diffusion_steps)
        elif self._schedule_sampler == "speed_diffusion":
            self._schedule_sampler = SpeedDiffusionSampler(
                self._diffusion_steps, forward_noise_scheduler
            )
        else:
            raise NotImplementedError(
                f"Schedule sampler {self._schedule_sampler} not implemented"
            )

        return forward_noise_scheduler

    def _build_sampling_pipeline(
        self, return_intermediate_samples: bool = False
    ) -> DiffusionSamplingPipeline:
        return DiffusionSamplingPipeline(
            model=self.reverse_diffusion_model,
            scheduler=self._forward_noise_scheduler,
            unnormalize_output=self._unnormalize_output,
            return_intermediate_samples=return_intermediate_samples,
            enable_class_conditioning=self._enable_class_conditioning,
            use_cfg=self._use_cfg,
            guidance_scale=self._guidance_scale,
        )

    def _prepare_input(self, batch: BatchDict) -> torch.Tensor:
        return batch[self._input_key]

    def _prepare_model_kwargs(
        self, batch: BatchDict, stage: TrainingStage
    ) -> torch.Tensor:
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
                if self._use_fixed_class_labels:
                    bs = min(64, batch[DataKeys.LABEL].shape[0])
                    num_labels = (
                        len(self._dataset_metadata.labels) + 1
                        if self._use_cfg
                        else len(self._dataset_metadata.labels)
                    )
                    if num_labels < bs:
                        class_labels = (
                            torch.arange(0, num_labels)
                            .repeat_interleave(bs // num_labels)
                            .to(idist.device())
                        )
                    else:
                        class_labels = torch.randperm(num_labels).repeat_interleave(4)
                        class_labels = class_labels[:bs].to(idist.device())

                    if self._custom_generated_class_label is not None:
                        if self._custom_generated_class_label == -1:
                            assert (
                                self._use_cfg
                            ), "Custom generated class label can only be set to -1 with CFG"
                            self._custom_generated_class_label = len(
                                self._dataset_metadata.labels
                            )
                        class_labels = (
                            torch.ones_like(class_labels, device=class_labels.device)
                            * self._custom_generated_class_label
                        )
                    logger.info(
                        f"Using fixed class labels for evaluation: {class_labels}"
                    )

            if stage == TrainingStage.test:
                if self._use_labels_from_batch:
                    class_labels = batch[DataKeys.LABEL]
                else:
                    class_labels = torch.randperm(num_labels)[:bs].to(idist.device())

            return {
                "class_labels": class_labels,
            }
        else:
            return {}

    def _setup_model_config(self) -> Dict[str, Dict[str, Any]]:
        if not isinstance(self._torch_model_builder, dict):
            raise NotImplementedError(
                self.__class__.__name__ + " requires a dict of TorchModelBuilders"
            )
        if (
            self._enable_class_conditioning
            and self._dataset_metadata.labels is not None
        ):
            num_class_embeds = len(self._dataset_metadata.labels)
            num_class_embeds = (
                num_class_embeds + 1 if self._use_cfg else num_class_embeds
            )
            return {
                key: dict(
                    num_class_embeds=num_class_embeds,
                    num_labels=num_class_embeds,
                    num_classes=num_class_embeds,
                )
                for key in self._torch_model_builder
            }
        else:
            return {key: {} for key in self._torch_model_builder}

    def _generate_timesteps(
        self,
        bsz: int,
        noise_multiplicity: Optional[int] = None,
        device: torch.device = torch.device("cpu"),
    ) -> torch.Tensor:
        timestep_weights = None
        if noise_multiplicity is not None:
            if self._weighted_timestep_sampling:
                timesteps = sample_weighted_timesteps(
                    self._weighted_timestep_config,
                    int(bsz * self._noise_multiplicity),
                    device=device,
                ).long()
            else:
                timesteps, timestep_weights = self._schedule_sampler.sample(
                    int(bsz * self._noise_multiplicity), device=device
                )
        else:
            if self._weighted_timestep_sampling:
                timesteps = sample_weighted_timesteps(
                    self._weighted_timestep_config,
                    bsz,
                    device=device,
                ).long()
            else:
                timesteps, timestep_weights = self._schedule_sampler.sample(
                    bsz, device=device
                )
        return timesteps, timestep_weights

    def _nm_model_forward(
        self, input: torch.Tensor, **model_kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz = input.shape[0]

        # repeat inputs for noise multiplier
        input = input.repeat_interleave(self._noise_multiplicity, dim=0)
        model_kwargs = {
            k: (
                v.repeat_interleave(self._noise_multiplicity, dim=0)
                if isinstance(v, torch.Tensor)
                else v
            )
            for k, v in model_kwargs.items()
        }
        noise = torch.randn_like(input)

        # Sample a random timestep for each image
        timesteps, timestep_weights = self._generate_timesteps(
            bsz, self._noise_multiplicity, input.device
        )

        # Add noise to the images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_inputs = self._forward_noise_scheduler.add_noise(input, noise, timesteps)

        # Predict the noise residual
        model_output = self.reverse_diffusion_model(
            noisy_inputs, timesteps, **model_kwargs
        )

        # get the 'sample' object
        model_output = (
            model_output.sample if hasattr(model_output, "sample") else model_output
        )
        return noise, noisy_inputs, timesteps, model_output, timestep_weights

    def _model_forward(
        self, input: torch.Tensor, **model_kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Sample noise that we'll add to the images
        noise = torch.randn_like(input)
        bsz = input.shape[0]

        # Sample a random timestep for each image
        timesteps, timestep_weights = self._generate_timesteps(bsz, None, input.device)

        # Add noise to the images according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_inputs = self._forward_noise_scheduler.add_noise(input, noise, timesteps)

        # Predict the noise residual
        model_output = self.reverse_diffusion_model(
            noisy_inputs, timesteps, **model_kwargs
        )

        # get the 'sample' object
        model_output = (
            model_output.sample if hasattr(model_output, "sample") else model_output
        )

        return noise, noisy_inputs, timesteps, model_output, timestep_weights

    def _compute_loss(
        self,
        input: torch.FloatTensor,
        noise: torch.FloatTensor,
        timesteps: torch.LongTensor,
        model_output: torch.FloatTensor,
        timestep_weights: Optional[torch.FloatTensor] = None,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        # compute loss
        if self._snr_gamma is None:
            if self._forward_noise_scheduler.config.prediction_type == "epsilon":
                target = noise
            elif self._forward_noise_scheduler.config.prediction_type == "sample":
                target = input
            elif self._forward_noise_scheduler.config.prediction_type == "v_prediction":
                target = self._forward_noise_scheduler.get_velocity(
                    input, noise, timesteps
                )
            else:
                raise ValueError(
                    f"Unknown training objective {self._forward_noise_scheduler.config.objective}"
                )
            loss = self._loss_fn(model_output, target, reduction="none")
            loss = loss.mean(dim=list(range(1, len(loss.shape))))

            if isinstance(self._schedule_sampler, LossAwareSampler):
                self._schedule_sampler.update_with_local_losses(
                    timesteps, loss.detach()
                )
            return (loss * timestep_weights).mean()
        else:
            snr = _compute_snr(self._forward_noise_scheduler, timesteps)
            snr_weights = torch.stack(
                [snr, self._snr_gamma * torch.ones_like(timesteps)], dim=1
            ).min(dim=1)[0]
            if self._forward_noise_scheduler.config.prediction_type == "epsilon":
                target = noise
                snr_weights = snr_weights / snr
                loss = self._loss_fn(model_output, target, reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * snr_weights
                return loss.mean()
            elif self._forward_noise_scheduler.config.prediction_type == "sample":
                target = input
                loss = self._loss_fn(model_output, target, reduction="none")
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * snr_weights
                return loss.mean()
            elif self._forward_noise_scheduler.config.prediction_type == "v_prediction":
                target = self._forward_noise_scheduler.get_velocity(
                    input, noise, timesteps
                )
                snr_weights = snr_weights / (snr + 1)
                loss = self._loss_fn(
                    model_output.sample.float(), target.float(), reduction="none"
                )
                loss = loss.mean(dim=list(range(1, len(loss.shape)))) * snr_weights
                return loss.mean()
            else:
                raise ValueError(
                    f"Unknown training objective {self._forward_noise_scheduler.config.objective}"
                )

    def _generate_data(
        self,
        num_samples: int = 64,
        return_intermediate_samples: bool = False,
        **model_kwargs,
    ) -> DiffusionSamplingPipelineOutput:
        num_samples = (
            model_kwargs["class_labels"].shape[0]
            if self._enable_class_conditioning
            else num_samples
        )

        sampling_pipeline = self._build_sampling_pipeline(
            return_intermediate_samples=return_intermediate_samples,
        )
        return sampling_pipeline(
            batch_size=num_samples,
            num_inference_steps=self._inference_diffusion_steps,
            input_shape=self._input_shape,
            **model_kwargs,
        )

    def required_keys_in_batch(self, stage: TrainingStage) -> List[str]:
        required_keys = (
            [self._input_key, DataKeys.LABEL]
            if self._enable_class_conditioning
            else [self._input_key]
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

    def training_step(self, batch: BatchDict, training_engine, **kwargs) -> ModelOutput:
        input = self._prepare_input(batch)
        assert (
            self._input_shape == input.shape[1:]
        ), f"Input shape mismatch. Expected {self._input_shape}, got {input.shape[1:]}"
        if self._noise_multiplicity is not None:
            noise, noisy_inputs, timesteps, model_output, timestep_weights = (
                self._nm_model_forward(
                    input,
                    **self._prepare_model_kwargs(batch, stage=TrainingStage.train),
                )
            )
        else:
            noise, noisy_inputs, timesteps, model_output, timestep_weights = (
                self._model_forward(
                    input,
                    **self._prepare_model_kwargs(batch, stage=TrainingStage.train),
                )
            )
        # if training_engine.state.iteration % 10 == 0:
        # if self._schedule_sampler._warmed_up():
        #     logger.info(f"Training iteration: {training_engine.state.iteration}")
        # import matplotlib.pyplot as plt

        # plt.plot(self._schedule_sampler.weights.cpu().numpy())
        # plt.show()

        # print and plot some stuff if test run is used
        if kwargs.get("test_run", False) and self._input_key == DataKeys.IMAGE:
            # printing timesteps
            bsz = input.shape[0]
            logger.debug(f"Noise timesteps in batch {timesteps}")
            logger.debug(f"kwargs sent to model:")
            for k, v in kwargs.items():
                logger.debug(f"{k}: {v}")

            # plotting images
            plt.title("Input images")
            image_grid = torchvision.utils.make_grid(
                input.clone().detach().cpu().float(),
                normalize=False,
                nrow=int(math.sqrt(bsz)),
            )
            plt.imshow(image_grid.permute(1, 2, 0))
            plt.show()

            # plotting noise
            plt.title("Noise")
            noise_grid = torchvision.utils.make_grid(
                noise.clone().detach().cpu().float(),
                normalize=False,
                nrow=int(math.sqrt(bsz)),
            )
            plt.imshow(noise_grid.permute(1, 2, 0))
            plt.show()

            # plotting noisy images
            plt.title("Noisy Images")
            noise_inputs_grid = torchvision.utils.make_grid(
                noisy_inputs.clone().detach().cpu().float(),
                normalize=False,
                nrow=int(math.sqrt(bsz)),
            )
            plt.imshow(noise_inputs_grid.permute(1, 2, 0))
            plt.show()

        loss = self._compute_loss(
            input=input,
            noise=noise,
            timesteps=timesteps,
            model_output=model_output,
            timestep_weights=timestep_weights,
        )
        self._tb_logger.add_scalar(
            "train/loss", loss.item(), training_engine.state.iteration
        )
        return DiffusionModelOutput(loss=loss)

    def get_output_dir(self):
        output_dir = (
            Path(self._tb_logger.logdir).parent
            / "samples"
            / self._forward_noise_scheduler.__class__.__name__
        )
        if self._use_cfg:
            output_dir = output_dir / f"cfg-{self._guidance_scale}"
        output_dir.mkdir(exist_ok=True, parents=True)
        return output_dir

    def check_output_msgpack_file_exists(self):
        output_msgpack_file_path = self.get_output_dir() / "samples.msgpack"
        if self._save_outputs_to_msgpack and self._generate_dataset_on_test:
            if (
                output_msgpack_file_path.exists()
                and output_msgpack_file_path.with_suffix(".msgpack.keys").exists()
            ):
                logger.info("Msgpack file already exists. Skipping saving outputs.")
                return True
        return False

    def save_outputs(
        self,
        engine: Engine,
        generated_samples: torch.Tensor,
        batch: BatchDict,
        model_kwargs: dict,
    ) -> None:
        assert (
            not self._is_save_outputs_done
        ), "save_outputs should only be called once on the task module. "
        output_dir = self.get_output_dir()

        print_once(logger, f"Saving generated samples to {output_dir}")
        for idx, sample in enumerate(generated_samples):
            class_label = None
            if "class_labels" in model_kwargs:
                class_label = model_kwargs["class_labels"][idx].item()

            if self._save_outputs_to_msgpack:
                output_msgpack_file_path = output_dir / "samples.msgpack"
                if self._msgpack_filewriter is None:
                    self._msgpack_filewriter = FileWriter(
                        output_msgpack_file_path,
                        overwrite=False,
                    )
                saved_sample = {}
                saved_sample["key"] = str(uuid.uuid4())
                saved_sample[f"{DataKeys.IMAGE}.mp"] = {
                    "path": None,
                    "bytes": _tensor_image_to_bytes(sample),
                }
                if class_label is not None:
                    saved_sample[f"{DataKeys.LABEL}.mp"] = class_label
                elif (
                    DataKeys.LABEL in batch
                ):  # this is a hack, essentially in unconditional sampling, the label will get attached to the batch
                    saved_sample[f"{DataKeys.LABEL}.mp"] = (
                        batch[DataKeys.LABEL][idx].cpu().numpy()
                    )
                self._msgpack_filewriter.write(saved_sample)
                if engine.state.iteration == 1:
                    torchvision.utils.save_image(
                        sample,
                        output_dir / f"{batch['__key__'][idx]}-{class_label}.png",
                        normalize=True,
                        nrow=1,
                    )
            else:
                torchvision.utils.save_image(
                    sample,
                    output_dir / f"{batch['__key__'][idx]}-{class_label}.png",
                    normalize=True,
                    nrow=1,
                )

        if (
            engine.state.iteration
            == engine.state.epoch_length * engine.state.max_epochs
        ):
            self._is_save_outputs_done = True
            if self._msgpack_filewriter is not None:
                self._msgpack_filewriter.close()

    def evaluation_step(
        self, batch: BatchDict, evaluation_engine: Engine, **kwargs
    ) -> ModelOutput:
        input = self._prepare_input(batch)
        assert (
            self._input_shape == input.shape[1:]
        ), f"Input shape mismatch. Expected {self._input_shape}, got {input.shape[1:]}"
        model_kwargs = self._prepare_model_kwargs(batch, stage=kwargs["stage"])
        self._progress_bar.pbar.set_description("Generating samples")
        generated_samples = self._generate_data(
            num_samples=input.shape[0],
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
            input = _unnormalize(input)

        # save outputs
        if kwargs["stage"] == TrainingStage.test and self._generate_dataset_on_test:
            self.save_outputs(
                engine=evaluation_engine,
                generated_samples=generated_samples,
                batch=batch,
                model_kwargs=model_kwargs,
            )

        return DiffusionModelOutput(loss=-1, real=input, generated=generated_samples)

    def visualization_step(
        self, batch, evaluation_engine=None, training_engine=None, **kwargs
    ):
        pipeline_outputs = self._generate_data(
            num_samples=len(batch["__key__"]),
            **self._prepare_model_kwargs(batch, stage=TrainingStage.visualization),
            return_intermediate_samples=False,
        )
        generated_samples = pipeline_outputs.generated_samples
        intermediate_samples = pipeline_outputs.intermediate_samples_at_xt
        if idist.get_rank() == 0:
            logger.info("Adding image batch to tensorboard")
            self._tb_logger.writer.add_image(
                f"visualization/generated/{self._input_key}",
                torchvision.utils.make_grid(
                    generated_samples,
                    normalize=False,
                    nrow=int(math.sqrt(generated_samples.shape[0])),
                ),
                training_engine.state.iteration,
            )

            # add intermediate samples to tensorboard
            if intermediate_samples is not None:
                for idx, intermediate_sample in enumerate(intermediate_samples):
                    self._tb_logger.writer.add_image(
                        f"visualization/generated/intermediate_sample_{training_engine.state.iteration}",
                        torchvision.utils.make_grid(
                            intermediate_sample,
                            normalize=False,
                            nrow=int(math.sqrt(intermediate_sample.shape[0])),
                        ),
                        idx,
                    )

            self._tb_logger.writer.flush()

    def predict_step(self, batch: BatchDict, **kwargs) -> ModelOutput:
        generated = self._generate_data(
            **self._prepare_model_kwargs(batch, stage=TrainingStage.predict)
        )
        return DiffusionModelOutput(loss=-1, generated=generated)
