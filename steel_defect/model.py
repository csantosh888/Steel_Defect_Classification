"""
ResNet18 model definition for steel defect classification.

"""
import torch
import torch.nn as nn
from torchvision import models

from steel_defect.utils import NUM_CLASSES, setup_logging

logger = setup_logging(__name__)


class SteelResNet18(nn.Module):
    

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        # ┌──────────────────────────────────────────────┐
        # │  MODEL-1: Write your code below              │
        # └──────────────────────────────────────────────┘
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.model = models.resnet18(weights=weights)

        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        
        # raise NotImplementedError("MODEL-1: Define CNN layers in __init__")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        # ┌──────────────────────────────────────────────┐
        # │  MODEL-2: Write your code below              │
        # └──────────────────────────────────────────────┘
        
        return self.model(x)
    
        raise NotImplementedError("MODEL-2: Implement forward pass")

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return f"SteelResNet18(params={self.num_parameters:,})"
