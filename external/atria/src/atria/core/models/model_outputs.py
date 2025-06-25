import dataclasses
from typing import List, Optional

import torch
from diffusers.models.autoencoders.vae import DiagonalGaussianDistribution
from transformers.utils import ModelOutput as TransformersModelOutput


@dataclasses.dataclass
class ModelOutput(TransformersModelOutput):
    pass


@dataclasses.dataclass
class ClassificationModelOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    prediction: Optional[torch.Tensor] = None
    label: Optional[torch.Tensor] = None


@dataclasses.dataclass
class TokenClassificationModelOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    predicted_labels: Optional[List[str]] = None
    target_labels: Optional[List[str]] = None


@dataclasses.dataclass
class LayoutTokenClassificationModelOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    label: Optional[torch.Tensor] = None
    token_bboxes: Optional[List[torch.Tensor]] = None
    prediction: Optional[torch.Tensor] = None


@dataclasses.dataclass
class QAModelOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    pred_answers: Optional[List[str]] = None
    target_answers: Optional[List[str]] = None


@dataclasses.dataclass
class SequenceQAModelOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    start_logits: Optional[torch.Tensor] = None
    end_logits: Optional[torch.Tensor] = None
    predicted_answers: Optional[List[str]] = None
    target_answers: Optional[List[str]] = None
    words: Optional[List[str]] = None
    word_ids: Optional[List[int]] = None
    sequence_ids: Optional[List[int]] = None
    question_id: Optional[int] = None
    gold_answers: Optional[List[str]] = None


@dataclasses.dataclass
class AutoEncoderModelOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    real: Optional[torch.Tensor] = None
    reconstructed: Optional[torch.Tensor] = None


@dataclasses.dataclass
class VarAutoEncoderModelOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    real: Optional[torch.Tensor] = None
    reconstructed: Optional[torch.Tensor] = None
    posterior: Optional[DiagonalGaussianDistribution] = None
    kl_loss: Optional[torch.Tensor] = None
    rec_loss: Optional[torch.Tensor] = None


@dataclasses.dataclass
class VarAutoEncoderGANModelOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    real: Optional[torch.Tensor] = None
    reconstructed: Optional[torch.Tensor] = None
    generated: Optional[torch.Tensor] = None

    # different types of gan losses
    kl_loss: Optional[torch.Tensor] = None
    nll_loss: Optional[torch.Tensor] = None
    rec_loss: Optional[torch.Tensor] = None
    d_weight: Optional[torch.Tensor] = None
    disc_factor: Optional[torch.Tensor] = None
    g_loss: Optional[torch.Tensor] = None


@dataclasses.dataclass
class DiffusionModelOutput(ModelOutput):
    loss: Optional[torch.Tensor] = None
    real: Optional[torch.Tensor] = None
    generated: Optional[torch.Tensor] = None
