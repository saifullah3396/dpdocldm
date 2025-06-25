import math
from pathlib import Path

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from ignite.contrib.handlers.tqdm_logger import ProgressBar
from ignite.engine import Engine
from pytorch_fid.inception import InceptionV3
from torch.utils.data import DataLoader

from atria.core.constants import DataKeys
from atria.core.task_modules.utilities.data_collators import PassThroughCollator
from atria.core.training.metrics.fid_metric import FIDSaver, WrapperInceptionV3


def calculate_inception_features(self, samples):
    if self.channels == 1:
        samples = repeat(samples, "b 1 ... -> b c ...", c=3)

    self._inception_model.eval()
    features = self._inception_model(samples)[0]

    if features.size(2) != 1 or features.size(3) != 1:
        features = F.adaptive_avg_pool2d(features, output_size=(1, 1))
    features = rearrange(features, "... 1 1 -> ...")
    return features


def load_or_precalc_dataset_stats(
    dataset,
    cache_dir: Path,
    split: str = "train",
    batch_size: int = 12,
    dataset_statistics_n_samples: int = 5000,
    stats_filename: str = "stats",
    logger=None,
):
    n_samples = min(dataset_statistics_n_samples, len(dataset))
    batch_size = min(batch_size, n_samples)
    ckpt_path = f"{cache_dir}/{split}-{n_samples}-{stats_filename}.pth"
    if Path(ckpt_path).exists():
        if logger is not None:
            logger.info(f"Dataset stats cache {ckpt_path} already exists.")
        return

    assert (
        DataKeys.IMAGE in dataset[0]
    ), f"Dataset must have image key {DataKeys.IMAGE} to compute dataset stats."

    # log shape info
    logger.info(
        f"Computing dataset stats for images of shape: {dataset[0][DataKeys.IMAGE].shape} with {n_samples} samples.",
    )

    # pytorch_fid model
    dims = 2048
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx])

    # wrapper model to pytorch_fid model
    wrapper_model = WrapperInceptionV3(model)
    wrapper_model.requires_grad_(False)
    wrapper_model.eval()
    metric = FIDSaver(
        num_features=dims,
        feature_extractor=wrapper_model,
        ckpt_path=ckpt_path,
        device=0,
    )

    def eval_step(engine, batch):
        image = torch.stack(batch[DataKeys.IMAGE]).cuda()
        if len(image.shape) == 3:  # convert grayscale to rgb
            image = image.unsqueeze(1).repeat(1, 3, 1, 1)
        return image

    default_evaluator = Engine(eval_step)
    num_batches = int(math.ceil(n_samples / batch_size))
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=PassThroughCollator(),
    )
    metric.attach(default_evaluator, "fid")
    ProgressBar(
        desc="Evaluating dataset FID stats",
    ).attach(
        default_evaluator,
    )
    default_evaluator.run(dl, epoch_length=num_batches)
    if logger is not None:
        logger.info(f"Dataset stats cached to {ckpt_path} for future use.")

    if hasattr(dataset, "close"):
        dataset.close()
