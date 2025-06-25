import warnings
from functools import partial
from pathlib import Path
from typing import Callable, Optional, Sequence, Union

import torch
from atria.core.metrics.generative.utilities import compute_fid
from atria.core.training.utilities.constants import TrainingStage
from atria.core.utilities.logging import get_logger
from ignite.metrics.gan.utils import InceptionModel, _BaseInceptionMetric
from ignite.metrics.metric import reinit__is_reduced, sync_all_reduce
from packaging.version import Version
from torch import nn
from torchvision.transforms.functional import resize

if Version(torch.__version__) <= Version("1.7.0"):
    torch_outer = torch.ger
else:
    torch_outer = torch.outer


logger = get_logger(__name__)


# wrapper class as feature_extractor
class WrapperInceptionV3(nn.Module):
    def __init__(self, fid_incv3):
        super().__init__()
        self.fid_incv3 = fid_incv3
        self.warned = False

    @torch.no_grad()
    def forward(self, x):
        y = self.fid_incv3(x)
        y = y[0]
        y = y[:, :, 0, 0]
        return y


class FID(_BaseInceptionMetric):
    def __init__(
        self,
        num_features: Optional[int] = None,
        feature_extractor: Optional[torch.nn.Module] = None,
        output_transform: Callable = lambda x: x,
        device: Union[str, torch.device] = torch.device("cpu"),
        ckpt_path: Optional[str] = None,
    ) -> None:
        try:
            import numpy as np  # noqa: F401
        except ImportError:
            raise ModuleNotFoundError("This module requires numpy to be installed.")

        try:
            import scipy  # noqa: F401
        except ImportError:
            raise ModuleNotFoundError("This module requires scipy to be installed.")

        if num_features is None and feature_extractor is None:
            num_features = 1000
            feature_extractor = InceptionModel(return_features=False, device=device)

        self._ckpt_path = Path(ckpt_path) if ckpt_path is not None else None
        self._stats_loaded_from_ckpt = False
        self._eps = 1e-6

        super(FID, self).__init__(
            num_features=num_features,
            feature_extractor=feature_extractor,
            output_transform=output_transform,
            device=device,
        )

    @staticmethod
    def _online_update(
        features: torch.Tensor, total: torch.Tensor, sigma: torch.Tensor
    ) -> None:
        total += features
        sigma += torch_outer(features, features)

    def _get_covariance(self, sigma: torch.Tensor, total: torch.Tensor) -> torch.Tensor:
        r"""
        Calculates covariance from mean and sum of products of variables
        """

        sub_matrix = torch_outer(total, total)
        sub_matrix = sub_matrix / self._num_examples

        return (sigma - sub_matrix) / (self._num_examples - 1)

    @reinit__is_reduced
    def reset(self) -> None:
        self._train_sigma = torch.zeros(
            (self._num_features, self._num_features),
            dtype=torch.float64,
            device=self._device,
        )

        self._train_total = torch.zeros(
            self._num_features, dtype=torch.float64, device=self._device
        )

        if self._ckpt_path is not None and self._ckpt_path.exists():
            ckpt = torch.load(
                self._ckpt_path,
                map_location=self._device,
            )
            self._test_sigma: torch.Tensor = ckpt["test_sigma"]
            self._test_total: torch.Tensor = ckpt["test_total"]
            self._num_examples_in_ckpt: int = ckpt["num_examples"]
            self._stats_loaded_from_ckpt = True

            logger.debug("Loaded FID statistics from checkpoint.")
            logger.debug(
                "Number of examples in checkpoint: %d", self._num_examples_in_ckpt
            )

            assert self._test_sigma.shape == (self._num_features, self._num_features)
            assert self._test_total.shape == (self._num_features,)
        else:
            self._test_sigma = torch.zeros(
                (self._num_features, self._num_features),
                dtype=torch.float64,
                device=self._device,
            )
            self._test_total = torch.zeros(
                self._num_features, dtype=torch.float64, device=self._device
            )
            self._num_examples_in_ckpt = None
        self._num_examples: int = 0

        super(FID, self).reset()

    @reinit__is_reduced
    def update(self, output: Sequence[torch.Tensor]) -> None:
        # train features are the predicted features, test features refer to the real features from original dataset
        # in case they are precomputed, the test features are loaded from the file and the train features are predicted
        # by the generative model
        train, test = output
        train_features = self._extract_features(train)

        # Updates the mean and covariance for the train features
        for features in train_features:
            self._online_update(features, self._train_total, self._train_sigma)

        if not self._stats_loaded_from_ckpt:
            test_features = self._extract_features(test)

            if (
                train_features.shape[0] != test_features.shape[0]
                or train_features.shape[1] != test_features.shape[1]
            ):
                raise ValueError(
                    f"""
                    Number of Training Features and Testing Features should be equal ({train_features.shape} != {test_features.shape})
                    """
                )

            # Updates the mean and covariance for the test features
            for features in test_features:
                self._online_update(features, self._test_total, self._test_sigma)

        self._num_examples += train_features.shape[0]

    @sync_all_reduce(
        "_num_examples", "_test_total", "_train_total", "_test_sigma", "_train_sigma"
    )
    def compute(self) -> float:
        if self._ckpt_path is not None and not self._ckpt_path.exists():
            self._ckpt_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "test_sigma": self._test_sigma,
                    "test_total": self._test_total,
                    "num_examples": self._num_examples,
                },
                self._ckpt_path,
            )

        if self._num_examples_in_ckpt is not None:
            if self._num_examples != self._num_examples_in_ckpt:
                logger.warning(
                    "The number of examples used in evaluation are different "
                    "from the number of examples in saved dataset statistics. FID computation may be incorrect."
                )

        fid = compute_fid(
            mu1=self._train_total / self._num_examples,
            mu2=self._test_total / self._num_examples,
            sigma1=self._get_covariance(self._train_sigma, self._train_total),
            sigma2=self._get_covariance(self._test_sigma, self._test_total),
            eps=self._eps,
        )

        if torch.isnan(torch.tensor(fid)) or torch.isinf(torch.tensor(fid)):
            warnings.warn(
                "The product of covariance of test and train features is out of bounds."
            )

        return fid

    def _extract_features(self, inputs: torch.Tensor) -> torch.Tensor:
        # we resize it again once for model input as FID is computed with 299, 299 size
        inputs = resize(inputs, (299, 299))

        inputs = inputs.detach()

        if inputs.device != torch.device(self._device):
            inputs = inputs.to(self._device)

        with torch.no_grad():
            outputs = self._feature_extractor(inputs).to(
                self._device, dtype=torch.float64
            )
        self._check_feature_shapes(outputs)

        return outputs


def default_fid_score(
    stage: Union[TrainingStage, str],
    dataset_cache_dir: Union[Path, str],
    output_transform: Callable,
    device: Union[str, torch.device],
    use_reconstruction: bool = False,
) -> FID:
    from pytorch_fid.inception import InceptionV3

    # # make a fid_stats_path given the stage
    # fid_ckpt_path = (
    #     Path(dataset_cache_dir) / "fid_stats" / f"{stage}-{uuid.uuid4()}.npz"
    # )
    # if not fid_ckpt_path.parent.exists():
    #     fid_ckpt_path.parent.mkdir(parents=True, exist_ok=True)

    # # log information
    # logger.info(
    #     f"Setting FID stats checkpoint path to: {fid_ckpt_path}. "
    #     "The dataset stats will be reused from this path after first run."
    # )

    # pytorch_fid model
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])

    # wrapper model to pytorch_fid model
    wrapper_model = WrapperInceptionV3(model)
    wrapper_model.requires_grad_(False)
    wrapper_model.eval()

    return FID(
        num_features=dims,
        feature_extractor=wrapper_model,
        output_transform=partial(
            output_transform, use_reconstruction=use_reconstruction
        ),
        # ckpt_path=fid_ckpt_path,
        device=device,
    )
