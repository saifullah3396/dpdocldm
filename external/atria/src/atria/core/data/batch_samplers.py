from functools import partial
from typing import Optional

from torch.utils.data import BatchSampler


class BatchSamplersDict:
    def __init__(
        self,
        train: Optional[partial[BatchSampler]] = None,
        evaluation: Optional[partial[BatchSampler]] = None,
    ):
        self._train = train
        self._evaluation = evaluation

    @property
    def train(self):
        return self._train

    @property
    def evaluation(self):
        return self._evaluation
