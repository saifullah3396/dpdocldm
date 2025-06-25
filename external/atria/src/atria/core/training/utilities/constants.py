"""Holds the constants related to training."""

from typing import Any


class TrainingStage:
    train = "train"
    validation = "validation"
    test = "test"
    predict = "predict"
    visualization = "visualization"

    @classmethod
    def get(cls, name: str) -> Any:
        return getattr(cls, name)


class GANStage:
    train_generator = "train_gen"
    train_discriminator = "train_disc"
