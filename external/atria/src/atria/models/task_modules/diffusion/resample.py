from abc import ABC, abstractmethod
import copy
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler


def create_named_schedule_sampler(name):
    """
    Create a ScheduleSampler from a library of pre-defined samplers.

    :param name: the name of the sampler.
    """
    if name == "uniform":
        return UniformSampler()
    elif name == "loss-second-moment":
        return LossSecondMomentResampler()
    else:
        raise NotImplementedError(f"unknown schedule sampler: {name}")


class ScheduleSampler(ABC):
    """
    A distribution over timesteps in the diffusion process, intended to reduce
    variance of the objective.

    By default, samplers perform unbiased importance sampling, in which the
    objective's mean is unchanged.
    However, subclasses may override sample() to change how the resampled
    terms are reweighted, allowing for actual changes in the objective.
    """

    def __init__(self, step_filter_range: Optional[Tuple[int, int]] = None):
        super().__init__()
        self.step_filter_range = step_filter_range

    @abstractmethod
    def weights(self):
        """
        Get a numpy array of weights, one per diffusion step.

        The weights needn't be normalized, but must be positive.
        """

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        w = self.weights()
        if self.step_filter_range is not None:
            w = copy.deepcopy(w)
            w[: self.step_filter_range[0]] = 0
            w[self.step_filter_range[1] :] = 0

        p = w / np.sum(w)
        indices_np = np.random.choice(len(p), size=(batch_size,), p=p)
        indices = torch.from_numpy(indices_np).long().to(device)
        weights_np = 1 / (len(p) * p[indices_np])
        weights = torch.from_numpy(weights_np).float().to(device)
        return indices, weights


class UniformSampler(ScheduleSampler):
    def __init__(self, num_train_timesteps):
        super().__init__()
        self._weights = np.ones([num_train_timesteps])

    def weights(self):
        return self._weights


class LossAwareSampler(ScheduleSampler):
    def update_with_local_losses(self, timesteps: torch.Tensor, losses: torch.Tensor):
        """
        Update the reweighting using losses from a model.

        Call this method from each rank with a batch of timesteps and the
        corresponding losses for each of those timesteps.
        This method will perform synchronization to make sure all of the ranks
        maintain the exact same reweighting.

        :param local_ts: an integer Tensor of timesteps.
        :param local_losses: a 1D Tensor of losses.
        """
        self.update_with_all_losses(
            timesteps.cpu().detach().tolist(), losses.cpu().detach().tolist()
        )

    @abstractmethod
    def update_with_all_losses(self, ts, losses):
        """
        Update the reweighting using losses from a model.

        Sub-classes should override this method to update the reweighting
        using losses from the model.

        This method directly updates the reweighting without synchronizing
        between workers. It is called by update_with_local_losses from all
        ranks with identical arguments. Thus, it should have deterministic
        behavior to maintain state across workers.

        :param ts: a list of int timesteps.
        :param losses: a list of float losses, one per timestep.
        """


class LossSecondMomentResampler(LossAwareSampler):
    def __init__(self, num_train_timesteps, history_per_term=1, uniform_prob=0.001):
        super().__init__()
        self.num_train_timesteps = num_train_timesteps
        self.history_per_term = history_per_term
        self.uniform_prob = uniform_prob
        self._loss_history = np.zeros(
            [num_train_timesteps, history_per_term], dtype=np.float64
        )
        self._loss_counts = np.zeros([num_train_timesteps], dtype=int)

    def weights(self):
        if not self._warmed_up():
            return np.ones([self.num_train_timesteps], dtype=np.float64)
        weights = np.sqrt(np.mean(self._loss_history**2, axis=-1))
        weights /= np.sum(weights)
        weights *= 1 - self.uniform_prob
        weights += self.uniform_prob / len(weights)
        return weights

    def update_with_all_losses(self, ts, losses):
        for t, loss in zip(ts, losses):
            if self._loss_counts[t] == self.history_per_term:
                # Shift out the oldest loss term.
                self._loss_history[t, :-1] = self._loss_history[t, 1:]
                self._loss_history[t, -1] = loss
            else:
                self._loss_history[t, self._loss_counts[t]] = loss
                self._loss_counts[t] += 1

    def _warmed_up(self):
        return (self._loss_counts == self.history_per_term).all()


class SpeedDiffusionSampler:
    def __init__(
        self,
        num_train_timesteps,
        noise_scheduler: DDPMScheduler,
        k=1,
        lam=0.6,
        tau=700,
    ):
        super().__init__()
        self.num_timesteps = num_train_timesteps
        self.alphas_cumprod = noise_scheduler.alphas_cumprod
        self.sqrt_one_minus_alphas_cumprod = (1 - self.alphas_cumprod) ** 0.5
        grad = np.gradient(self.sqrt_one_minus_alphas_cumprod)
        self.meaningful_steps = np.argmax(grad < 1e-4) + 1
        self.lam = lam
        self.k = k
        self.tau = tau
        higher = self.k
        lower = 1
        p = [higher] * self.tau + [lower] * (self.num_timesteps - self.tau)
        self.p = F.normalize(torch.tensor(p, dtype=torch.float32), p=1, dim=0)

        self.weights = self._weights()

    def _weights(self):
        weights = np.gradient(1 - self.alphas_cumprod)
        k = 1 - self.lam
        p = weights
        weights = k + (1 - 2 * k) * (p - p.min()) / (p.max() - p.min())
        weights = weights[: self.tau].tolist() + [1] * (self.num_timesteps - self.tau)
        weights = np.array(weights)
        return torch.from_numpy(weights)

    def sample(self, batch_size, device):
        """
        Importance-sample timesteps for a batch.

        :param batch_size: the number of timesteps.
        :param device: the torch device to save to.
        :return: a tuple (timesteps, weights):
                 - timesteps: a tensor of timestep indices.
                 - weights: a tensor of weights to scale the resulting losses.
        """
        t = torch.multinomial(self.p, batch_size // 2 + 1, replacement=True).to(device)
        dual_t = torch.where(
            t < self.meaningful_steps,
            self.meaningful_steps - t,
            t - self.meaningful_steps,
        )
        t = torch.cat([t, dual_t], dim=0)[:batch_size]
        self.weights = self.weights.to(device)
        return t, self.weights[t]
