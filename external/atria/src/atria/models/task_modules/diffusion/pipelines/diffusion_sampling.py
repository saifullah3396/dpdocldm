from dataclasses import dataclass
from typing import List, Optional, Union

import torch
import tqdm
from atria.models.task_modules.diffusion.utilities import (
    _guidance_wrapper,
    _unnormalize,
)
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


@dataclass
class DiffusionSamplingPipelineOutput:
    generated_samples: torch.FloatTensor
    intermediate_samples_at_xt: List[torch.FloatTensor] = None


class DiffusionSamplingPipeline:
    def __init__(
        self,
        model: torch.nn.Module,
        scheduler: DDPMScheduler,
        vae: torch.nn.Module = None,
        unnormalize_output: bool = True,
        return_intermediate_samples: bool = False,
        total_intermediate_samples: int = 20,
        enable_class_conditioning: bool = False,
        use_cfg: bool = False,
        guidance_scale: int = 1,
    ):
        super().__init__()
        self._model = model
        self._scheduler = scheduler
        self._vae = vae
        self._unnormalize_output = unnormalize_output
        self._return_intermediate_samples = return_intermediate_samples
        self._total_intermediate_samples = total_intermediate_samples
        self._use_cfg = use_cfg
        self._guidance_scale = guidance_scale
        if enable_class_conditioning:
            self._model = _guidance_wrapper(
                self._model, guidance_scale=self._guidance_scale, use_cfg=self._use_cfg
            )

    def _post_process(self, x):
        if self._vae is not None:
            scaling_factor = (
                self._vae.config.scaling_factor
                if hasattr(self._vae, "config")
                else self._vae.scaling_factor
            )
            x = x.cuda()
            x = 1 / scaling_factor * x
            x = self._vae.decode(x).sample
        if self._unnormalize_output:
            x = _unnormalize(x)
        return x

    def _model_forward(self, x: torch.Tensor, t: int, **model_kwargs):
        t = torch.full(
            (x.shape[0],),
            t,
            device=x.device,
            dtype=torch.long,
        )
        model_output = self._model(x, t, **model_kwargs)
        if hasattr(model_output, "sample"):
            model_output = model_output.sample
        return model_output

    def _generate(
        self,
        x: torch.FloatTensor,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        **model_kwargs,
    ) -> torch.FloatTensor:
        intermediate_samples_at_xt = []
        if self._return_intermediate_samples:
            intermediate_samples_at_xt.append(x)

        for idx, t in tqdm.tqdm(enumerate(self._scheduler.timesteps), disable=False):
            # 1. predict noise model_output
            model_output = self._model_forward(x, t, **model_kwargs)

            # 2. compute previous image: x_t -> x_t-1
            x = self._scheduler.step(
                model_output,
                t,
                x,
                generator=generator,
                # eta=1.0,
            ).prev_sample

            # store intermediate steps
            if self._return_intermediate_samples:
                total_steps = len(self._scheduler.timesteps)
                save_every = total_steps // self._total_intermediate_samples
                if idx % save_every == 0 or (idx == len(self._scheduler.timesteps) - 1):
                    intermediate_samples_at_xt.append(x)
        return x, intermediate_samples_at_xt

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        input_shape: List[int] = [3, 256, 256],
        num_inference_steps: int = 1000,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        **model_kwargs,
    ) -> DiffusionSamplingPipelineOutput:
        import ignite.distributed as idist

        input_shape = (batch_size, *input_shape)
        input = torch.randn(input_shape, generator=generator, device=idist.device())

        # set step values
        self._scheduler.set_timesteps(num_inference_steps)

        # generate from noise
        generated_samples, intermediate_samples_at_xt = self._generate(
            x=input, generator=generator, **model_kwargs
        )

        # post process
        if len(intermediate_samples_at_xt) > 0:
            return DiffusionSamplingPipelineOutput(
                generated_samples=self._post_process(generated_samples),
                intermediate_samples_at_xt=[
                    self._post_process(x) for x in intermediate_samples_at_xt
                ],
            )
        return DiffusionSamplingPipelineOutput(
            generated_samples=self._post_process(generated_samples)
        )
