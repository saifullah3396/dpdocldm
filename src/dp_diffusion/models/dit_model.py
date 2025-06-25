from transformers import (
    BeitModel,
)
import torch
from torch import nn


class DitModel(nn.Module):
    def __init__(self, drop_rate: float = 0.5, num_labels: int = 16):
        super().__init__()
        self.model = BeitModel.from_pretrained(
            "microsoft/dit-base", add_pooling_layer=True
        )
        self.classifier = (
            nn.Linear(self.model.config.hidden_size, num_labels)
            if num_labels > 0
            else nn.Identity()
        )
        self.dropout = nn.Dropout(drop_rate)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        outputs = self.model(inputs)
        return self.classifier(self.dropout(outputs.pooler_output))
