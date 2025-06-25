from typing import Any, Dict, TYPE_CHECKING, Union

if TYPE_CHECKING:
    import torch

BatchDict = Dict[str, Union["torch.Tensor", Any]]
TorchNNModule = Union[Dict[str, "torch.nn.Module"], "torch.nn.Module"]
