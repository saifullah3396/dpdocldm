from itertools import chain
from typing import List, Tuple, Union

import torch
from atria.core.training.utilities.ddp_model_proxy import ModuleProxyWrapper
from atria.core.utilities.logging import get_logger
from opacus import GradSampleModule, PrivacyEngine
from opacus.distributed import DifferentiallyPrivateDistributedDataParallel as DPDDP
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

from opacus.optimizers.optimizer import DPOptimizer
from opacus.privacy_engine import forbid_accumulation_hook

logger = get_logger()


class ExtendedPrivacyEngine(PrivacyEngine):
    def state_dict(self):
        return {"privacy_accountant": self.accountant.state_dict()}

    def load_state_dict(self, state_dict):
        self.accountant.load_state_dict(state_dict["privacy_accountant"])

    def make_private(
        self,
        *,
        module: nn.Module,
        optimizer: optim.Optimizer,
        data_loader: DataLoader,
        noise_multiplier: float,
        max_grad_norm: Union[float, List[float]],
        batch_first: bool = True,
        loss_reduction: str = "mean",
        poisson_sampling: bool = True,
        clipping: str = "flat",
        noise_generator=None,
        grad_sample_mode: str = "hooks",
        noise_multiplicity: int = 1,
    ) -> Tuple[GradSampleModule, DPOptimizer, DataLoader]:
        """
        Add privacy-related responsibilites to the main PyTorch training objects:
        model, optimizer, and the data loader.

        All of the returned objects act just like their non-private counterparts
        passed as arguments, but with added DP tasks.

        - Model is wrapped to also compute per sample gradients.
        - Optimizer is now responsible for gradient clipping and adding noise to the gradients.
        - DataLoader is updated to perform Poisson sampling.

        Notes:
            Using any other models, optimizers, or data sources during training
            will invalidate stated privacy guarantees.

        Args:
            module: PyTorch module to be used for training
            optimizer: Optimizer to be used for training
            data_loader: DataLoader to be used for training
            noise_multiplier: The ratio of the standard deviation of the Gaussian noise to
                the L2-sensitivity of the function to which the noise is added
                (How much noise to add)
            max_grad_norm: The maximum norm of the per-sample gradients. Any gradient with norm
                higher than this will be clipped to this value.
            batch_first: Flag to indicate if the input tensor to the corresponding module
                has the first dimension representing the batch. If set to True, dimensions on
                input tensor are expected be ``[batch_size, ...]``, otherwise
                ``[K, batch_size, ...]``
            loss_reduction: Indicates if the loss reduction (for aggregating the gradients)
                is a sum or a mean operation. Can take values "sum" or "mean"
            poisson_sampling: ``True`` if you want to use standard sampling required
                for DP guarantees. Setting ``False`` will leave provided data_loader
                unchanged. Technically this doesn't fit the assumptions made by
                privacy accounting mechanism, but it can be a good approximation when
                using Poisson sampling is unfeasible.
            clipping: Per sample gradient clipping mechanism ("flat" or "per_layer" or "adaptive").
                Flat clipping calculates the norm of the entire gradient over
                all parameters, per layer clipping sets individual norms for
                every parameter tensor, and adaptive clipping updates clipping bound per iteration.
                Flat clipping is usually preferred, but using per layer clipping in combination
                with distributed training can provide notable performance gains.
            noise_generator: torch.Generator() object used as a source of randomness for
                the noise
            grad_sample_mode: mode for computing per sample gradients. Determines the
                implementation class for the wrapped ``module``. See
                :class:`~opacus.grad_sample.gsm_base.AbstractGradSampleModule` for more
                details

        Returns:
            Tuple of (model, optimizer, data_loader).

            Model is a wrapper around the original model that also computes per sample
                gradients
            Optimizer is a wrapper around the original optimizer that also does
            gradient clipping and noise addition to the gradients
            DataLoader is a brand new DataLoader object, constructed to behave as
                equivalent to the original data loader, possibly with updated
                sampling mechanism. Points to the same dataset object.
        """
        if noise_generator and self.secure_mode:
            raise ValueError("Passing seed is prohibited in secure mode")

        # compare module parameter with optimizer parameters
        model_parameters = set(module.parameters())
        for p in chain.from_iterable(
            [param_group["params"] for param_group in optimizer.param_groups]
        ):
            if p not in model_parameters:
                raise ValueError(
                    "Module parameters are different than optimizer Parameters"
                )
        if isinstance(module, ModuleProxyWrapper):
            distributed = isinstance(module.module, (DPDDP, DDP))
        else:
            distributed = isinstance(module, (DPDDP, DDP))

        module = self._prepare_model(
            module,
            batch_first=batch_first,
            loss_reduction=loss_reduction,
            grad_sample_mode=grad_sample_mode,
        )
        if poisson_sampling:
            module.register_backward_hook(forbid_accumulation_hook)

        data_loader = self._prepare_data_loader(
            data_loader, distributed=distributed, poisson_sampling=poisson_sampling
        )

        sample_rate = 1 / len(data_loader)
        expected_batch_size = int(len(data_loader.dataset) * sample_rate)

        # expected_batch_size is the *per worker* batch size
        if distributed:
            world_size = torch.distributed.get_world_size()
            expected_batch_size /= world_size

        logger.info(f"Per client sample rate = {sample_rate}")
        logger.info(f"Per client batch size = {expected_batch_size}")

        optimizer = self._prepare_optimizer(
            optimizer,
            noise_multiplier=noise_multiplier,
            max_grad_norm=max_grad_norm,
            expected_batch_size=expected_batch_size,
            loss_reduction=loss_reduction,
            noise_generator=noise_generator,
            distributed=distributed,
            clipping=clipping,
            grad_sample_mode=grad_sample_mode,
            noise_multiplicity=noise_multiplicity,
        )

        optimizer.attach_step_hook(
            self.accountant.get_optimizer_hook_fn(sample_rate=sample_rate)
        )

        return module, optimizer, data_loader
