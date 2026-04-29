"""
ResNet18 model definition for steel defect classification.

"""
import torch
import torch.nn as nn
from torchvision import models

from steel_defect.utils import NUM_CLASSES, setup_logging

logger = setup_logging(__name__)


class SteelResNet18(nn.Module):
    """ResNet18-based classifier for steel defect classification."""

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        # ┌──────────────────────────────────────────────┐
        # │  MODEL-1: Write your code below              │
        # └──────────────────────────────────────────────┘
        super().__init__()

        # Load ImageNet-pretrained ResNet18 weights if pretrained=True
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        self.model = models.resnet18(weights=weights)

        # Replace the original ImageNet classification head
        # ResNet18 originally predicts 1000 classes; we change it to NUM_CLASSES
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

        
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run the input image batch through ResNet18 and return class logits."""
        # ┌──────────────────────────────────────────────┐
        # │  MODEL-2: Write your code below              │
        # └──────────────────────────────────────────────┘
        # x shape expected: [batch_size, channels, height, width]
        # output shape: [batch_size, num_classes]
        return self.model(x)
    
        

    @property
    def num_parameters(self) -> int:
        """Total number of trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        """Readable model summary when printing the model object."""
        return f"SteelResNet18(params={self.num_parameters:,})"
