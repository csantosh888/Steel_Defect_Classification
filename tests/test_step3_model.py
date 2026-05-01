"""
Step 3 Tests — MODEL-1 and MODEL-2.

Run with:
    pytest tests/test_step3_model.py -v

Prerequisites: None.
"""
import pytest
import torch

from steel_defect.utils import NUM_CLASSES, IMAGE_SIZE
from steel_defect.model import SteelResNet18


@pytest.fixture
def model():
    """Create a SteelResNet18 instance."""
    return SteelResNet18(num_classes=NUM_CLASSES)


@pytest.fixture
def dummy_batch():
    """Create a random input batch matching expected dimensions."""
    return torch.randn(2, 3, IMAGE_SIZE[1], IMAGE_SIZE[0])


class TestModel1Init:
    """MODEL-1: __init__ layer definitions."""

    def test_is_nn_module(self, model):
        assert isinstance(model, torch.nn.Module)

       
    def test_has_trainable_parameters(self, model):
        assert model.num_parameters > 0, "Model should have trainable parameters"

    

class TestModel2Forward:
    """MODEL-2: forward pass."""

    def test_output_shape(self, model, dummy_batch):
        output = model(dummy_batch)
        assert output.shape == (2, NUM_CLASSES), (
            f"Expected output shape (2, {NUM_CLASSES}), got {tuple(output.shape)}"
        )

    def test_output_dtype(self, model, dummy_batch):
        output = model(dummy_batch)
        assert output.dtype == torch.float32

    def test_single_image(self, model):
        """Forward pass with batch size 1."""
        single = torch.randn(1, 3, IMAGE_SIZE[0], IMAGE_SIZE[1])
        output = model(single)
        assert output.shape == (1, NUM_CLASSES)

    def test_no_softmax_applied(self, model, dummy_batch):
        """Output should be raw logits — NOT softmax probabilities."""
        output = model(dummy_batch)
        row_sums = output.sum(dim=1)
        is_probability = torch.allclose(
            row_sums, torch.ones_like(row_sums), atol=0.01
        )
        assert not is_probability, (
            "Output rows sum to ~1.0 — looks like softmax was applied. "
            "forward() should return raw logits, not probabilities."
        )

    def test_custom_num_classes(self):
        """Model should work with a different number of classes."""
        custom_model = SteelResNet18(num_classes=3)
        x = torch.randn(1, 3, IMAGE_SIZE[1], IMAGE_SIZE[0])
        output = custom_model(x)
        assert output.shape == (1, 3)
