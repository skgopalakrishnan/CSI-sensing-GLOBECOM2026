from typing import Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


class _TinyLinear(nn.Module):
    """A minimal linear model producing 2-class logits.

    The model flattens the input and applies a learned linear projection.
    Gradients backpropagate to inputs so adversarial steps are non-trivial.
    """

    def __init__(self, in_features: int, num_classes: int = 3) -> None:
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features, num_classes, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(self.flatten(x))

def _build_loader(shape: Tuple[int, int, int, int], num_classes: int = 2) -> Tuple[DataLoader, torch.Tensor]:
    """Create a small CPU DataLoader with random inputs and labels.

    Returns the loader and the stacked inputs tensor for shape checks.
    """
    N, C, H, W = shape
    x = torch.randn(N, C, H, W)
    y = torch.randint(low=0, high=num_classes, size=(N,))
    ds = TensorDataset(x, y)
    return DataLoader(ds, batch_size=N, shuffle=False), x
