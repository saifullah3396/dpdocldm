from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from atria.core.models.model_outputs import ModelOutput
from atria.core.models.task_modules.atria_task_module import AtriaTaskModule
from atria.core.models.utilities.common import _validate_keys_in_batch
from atria.core.training.configs.gradient_config import GradientConfig
from atria.core.training.engines.engine_steps.base import BaseEngineStep
from atria.core.training.engines.events import OptimizerEvents
from atria.core.training.engines.utilities import _detach_tensor
from atria.core.training.utilities.constants import GANStage, TrainingStage
from atria.core.utilities.logging import get_logger
from ignite.engine import Engine
from torch.amp import autocast
from torch.optim import Optimizer

logger = get_logger(__name__)


class TrainingStep(BaseEngineStep):
    def __init__(
        self,
        task_module: AtriaTaskModule,
        device: Union[str, torch.device],
        optimizers: Dict[str, Optimizer],
        gradient_config: GradientConfig,
        grad_scaler: Optional["torch.cuda.amp.GradScaler"] = None,
        output_transform: Callable[[Any], Any] = _detach_tensor,
        non_blocking_tensor_conv: bool = False,
        with_amp: bool = False,
        test_run: bool = False,
    ):
        super().__init__(
            task_module=task_module,
            device=device,
            non_blocking_tensor_conv=non_blocking_tensor_conv,
            with_amp=with_amp,
            test_run=test_run,
        )
        self._optimizers = optimizers
        self._grad_scaler = grad_scaler
        self._output_transform = output_transform
        self._gradient_config = gradient_config
        self._step_update_validated = False
        self._setup_amp()

    @property
    def stage(self) -> TrainingStage:
        return TrainingStage.train

    @property
    def gradient_config(self) -> GradientConfig:
        return self._gradient_config

    def _setup_amp(self) -> None:
        if self._with_amp:
            try:
                pass
            except ImportError:
                raise ImportError("Please install torch>=1.6.0 to use amp_mode='amp'.")

            from torch.cuda.amp.grad_scaler import GradScaler

            self._grad_scaler = GradScaler(enabled=True)

    def _validate_inputs(self, batch: Dict[str, torch.Tensor]) -> None:
        if not self._step_update_validated:
            self._task_module.validate_model_built()
            _validate_keys_in_batch(
                keys=self._task_module.required_keys_in_batch(
                    stage=TrainingStage.train
                ),
                batch=batch,
            )
            self._step_update_validated = True

    def _validate_gradient_config(self) -> None:
        if self._gradient_config.gradient_accumulation_steps <= 0:
            raise ValueError(
                "Gradient_accumulation_steps must be strictly positive. "
                "No gradient accumulation if the value set to one (default)."
            )

    def _reset_optimizers(self, engine: Engine) -> None:
        # perform optimizers zero_grad() operation with gradient accumulation
        if (
            engine.state.iteration - 1
        ) % self._gradient_config.gradient_accumulation_steps == 0:
            for opt in self._optimizers.values():
                opt.zero_grad()

    def _call_forward(self, engine: Engine, batch: Dict[str, torch.Tensor]) -> None:
        with autocast(device_type=self._device.type, enabled=self._with_amp):
            # forward pass
            model_output = self._task_module.training_step(
                training_engine=engine, batch=batch, test_run=self._test_run
            )

            # make sure we get a dict from the model
            assert isinstance(
                model_output, ModelOutput
            ), "Model must return an instance of ModelOutput."
            assert (
                model_output.loss is not None
            ), f"Model output loss must not be None during the training step. "

            # get the loss
            loss = model_output.loss

            # accumulate loss if required
            if self._gradient_config.gradient_accumulation_steps > 1:
                loss = loss / self._gradient_config.gradient_accumulation_steps

            return loss, model_output

    def _update_optimizers_with_grad_scaler(
        self,
        engine: Engine,
        loss: torch.Tensor,
        optimizer_key: Optional[str] = None,
    ) -> None:
        self._grad_scaler.scale(loss).backward()

        # perform optimizer update for correct gradient accumulation step
        if (
            engine.state.iteration % self._gradient_config.gradient_accumulation_steps
            == 0
        ):
            # perform gradient clipping if needed
            if self._gradient_config.enable_grad_clipping:
                # Unscales the gradients of optimizer's assigned params in-place
                for key, opt in self._optimizers.items():
                    if optimizer_key is None or key == optimizer_key:
                        self._grad_scaler.unscale_(opt)

                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                torch.nn.utils.clip_grad_norm_(
                    self._task_module.torch_model.parameters(),
                    self._gradient_config.max_grad_norm,
                )

            for key, opt in self._optimizers.items():
                if optimizer_key is None or key == optimizer_key:
                    self._grad_scaler.step(opt)

            # scaler update should be called only once. See https://pytorch.org/docs/stable/amp.html
            self._grad_scaler.update()

    def _update_optimizers_standard(
        self,
        engine: Engine,
        loss: torch.Tensor,
        optimizer_key: Optional[str] = None,
    ) -> None:
        # backward pass
        loss.backward()

        # perform optimizer update for correct gradient accumulation step
        if (
            engine.state.iteration % self._gradient_config.gradient_accumulation_steps
            == 0
        ):
            # perform gradient clipping if needed
            if self._gradient_config.enable_grad_clipping:
                # Since the gradients of optimizer's assigned params are unscaled, clips as usual:
                torch.nn.utils.clip_grad_norm_(
                    self._task_module.torch_model.parameters(),
                    self._gradient_config.max_grad_norm,
                )

            for key, opt in self._optimizers.items():
                if optimizer_key is None or key == optimizer_key:
                    opt.step()

    def _update_optimizers(
        self,
        engine: Engine,
        loss: torch.Tensor,
        optimizer_key: Optional[str] = None,
    ) -> None:
        if self._grad_scaler:
            self._update_optimizers_with_grad_scaler(
                engine=engine, loss=loss, optimizer_key=optimizer_key
            )
        else:
            self._update_optimizers_standard(
                engine=engine, loss=loss, optimizer_key=optimizer_key
            )

    def __call__(
        self, engine: Engine, batch: Dict[str, torch.Tensor]
    ) -> Union[Any, Tuple[torch.Tensor]]:
        self._validate_inputs(batch)
        self._validate_gradient_config()
        self._reset_optimizers(engine=engine)
        self._task_module.train()
        self._convert_batch_to_device(batch)
        loss, model_output = self._call_forward(engine=engine, batch=batch)
        self._update_optimizers(engine=engine, loss=loss)
        if (
            engine.state.iteration % self._gradient_config.gradient_accumulation_steps
            == 0
        ):
            # call update
            engine.fire_event(OptimizerEvents.optimizer_step)
            engine.state.optimizer_step += 1

        return model_output


class GANTrainingStep(TrainingStep):
    def __init__(
        self,
        task_module: AtriaTaskModule,
        device: Union[str, torch.device],
        optimizers: Dict[str, Optimizer],
        gradient_config: GradientConfig,
        grad_scaler: Optional["torch.cuda.amp.GradScaler"] = None,
        output_transform: Callable[[Any], Any] = _detach_tensor,
        non_blocking_tensor_conv: bool = False,
        with_amp: bool = False,
        test_run: bool = False,
    ):
        super().__init__(
            task_module=task_module,
            device=device,
            optimizers=optimizers,
            gradient_config=gradient_config,
            grad_scaler=grad_scaler,
            output_transform=output_transform,
            non_blocking_tensor_conv=non_blocking_tensor_conv,
            with_amp=with_amp,
            test_run=test_run,
        )

    def _validate_optimizers(self) -> None:
        # make sure that optimizers are set up for generator and discriminator
        assert (
            "generator" in self._optimizers.keys()
        ), f"You must define and optimizer with key 'generator' to use the {self.__class__.__name__}. "
        assert (
            "discriminator" in self._optimizers.keys()
        ), f"You must define and optimizer with key 'discriminator' to use the {self.__class__.__name__}. "

    def _toggle_optimizer(self, opt: torch.optim.Optimizer) -> None:
        # disable all parameters
        for param in self._task_module.torch_model.parameters():
            param.requires_grad = False

        # enable the parameters of the optimizer
        for group in opt.param_groups:
            for param in group["params"]:
                param.requires_grad = True

        # # print the parameters that are enabled
        # for name, param in self._task_module.torch_model.named_parameters():
        #     if param.requires_grad:
        #         logger.debug(f"Enabled parameter: {name}")

    def _call_forward(
        self, engine: Engine, batch: Dict[str, torch.Tensor], gan_stage: GANStage
    ) -> None:
        with autocast(device_type=self._device.type, enabled=self._with_amp):
            # forward pass
            model_output = self._task_module.training_step(
                training_engine=engine,
                batch=batch,
                test_run=self._test_run,
                gan_stage=gan_stage,
            )

            # make sure we get a dict from the model
            assert isinstance(
                model_output, ModelOutput
            ), "Model must return an instance of ModelOutput."
            assert (
                model_output.loss is not None
            ), f"Model output loss must not be None during the training step. "

            # get the loss
            loss = model_output.loss

            # accumulate loss if required
            if self._gradient_config.gradient_accumulation_steps > 1:
                loss = loss / self._gradient_config.gradient_accumulation_steps

            return loss, model_output

    def __call__(
        self, engine: Engine, batch: Dict[str, torch.Tensor]
    ) -> Union[Any, Tuple[torch.Tensor]]:
        self._validate_inputs(batch)
        self._validate_gradient_config()
        self._reset_optimizers(engine=engine)
        self._task_module.train()
        self._convert_batch_to_device(batch)
        step_outputs = {}
        for gan_stage in [GANStage.train_generator, GANStage.train_discriminator]:
            # only train the generator or discriminator depending upon the stage
            optimizer_key = (
                "generator"
                if gan_stage == GANStage.train_generator
                else "discriminator"
            )

            # only compute gradients for this optimizer parameters
            self._toggle_optimizer(self._optimizers[optimizer_key])

            # perform forward pass for the given stage
            loss, model_output = self._call_forward(
                engine=engine, batch=batch, gan_stage=gan_stage
            )
            self._update_optimizers(
                engine=engine, loss=loss, optimizer_key=optimizer_key
            )

            output_key = "gen" if gan_stage == GANStage.train_generator else "disc"
            step_outputs[f"{output_key}_loss"] = model_output["loss"]
        if (
            engine.state.iteration % self._gradient_config.gradient_accumulation_steps
            == 0
        ):
            # call update
            engine.fire_event(OptimizerEvents.optimizer_step)
            engine.state.optimizer_step += 1

        return step_outputs
