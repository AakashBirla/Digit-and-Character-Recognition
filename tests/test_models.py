"""Tests for neural network model architectures.

Tests MLP model initialization, forward pass, output shapes,
loss calculation, and training step.
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from models.mlp import MLP
from models.cnn import CNN


class TestMLPInitialization:
    """Test MLP model initialization."""

    def test_default_initialization(self):
        model = MLP(num_classes=10)
        assert model is not None
        assert model.num_classes == 10

    def test_custom_num_classes(self):
        model = MLP(num_classes=47)
        assert model.num_classes == 47

    def test_custom_hidden_sizes(self):
        model = MLP(num_classes=10, hidden_sizes=(512, 256, 128))
        assert model.hidden_sizes == (512, 256, 128)

    def test_parameter_count(self):
        model = MLP(num_classes=10, hidden_sizes=(256, 128))
        count = model.count_parameters()
        assert count > 0
        # 784*256 + 256 + 256*128 + 128 + 128*10 + 10 = ~235K
        assert 230000 < count < 240000


class TestMLPForwardPass:
    """Test MLP forward pass and output shapes."""

    def test_forward_4d_input(self):
        """Test with standard image tensor (batch, C, H, W)."""
        model = MLP(num_classes=10)
        x = torch.randn(32, 1, 28, 28)
        output = model(x)
        assert output.shape == (32, 10)

    def test_forward_2d_input(self):
        """Test with pre-flattened input (batch, 784)."""
        model = MLP(num_classes=10)
        x = torch.randn(16, 784)
        output = model(x)
        assert output.shape == (16, 10)

    def test_forward_single_sample(self):
        """Test with a single sample."""
        model = MLP(num_classes=10)
        x = torch.randn(1, 1, 28, 28)
        output = model(x)
        assert output.shape == (1, 10)

    def test_forward_emnist_classes(self):
        """Test with EMNIST number of classes (47)."""
        model = MLP(num_classes=47)
        x = torch.randn(8, 1, 28, 28)
        output = model(x)
        assert output.shape == (8, 47)

    def test_output_not_nan(self):
        model = MLP(num_classes=10)
        x = torch.randn(4, 1, 28, 28)
        output = model(x)
        assert not torch.isnan(output).any()


class TestMLPTrainingStep:
    """Test a single training step with the MLP."""

    def test_loss_calculation(self):
        model = MLP(num_classes=10)
        criterion = nn.CrossEntropyLoss()
        x = torch.randn(16, 1, 28, 28)
        labels = torch.randint(0, 10, (16,))
        output = model(x)
        loss = criterion(output, labels)
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_backward_pass(self):
        model = MLP(num_classes=10)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        x = torch.randn(16, 1, 28, 28)
        labels = torch.randint(0, 10, (16,))

        # Forward
        output = model(x)
        loss = criterion(output, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify gradients were computed
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None

    def test_training_reduces_loss(self):
        """Verify that training on the same batch reduces loss."""
        model = MLP(num_classes=10)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

        x = torch.randn(32, 1, 28, 28)
        labels = torch.randint(0, 10, (32,))

        # Initial loss
        output = model(x)
        initial_loss = criterion(output, labels).item()

        # Train for a few steps on same data
        for _ in range(20):
            output = model(x)
            loss = criterion(output, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Final loss should be lower
        output = model(x)
        final_loss = criterion(output, labels).item()
        assert final_loss < initial_loss, \
            f"Loss didn't decrease: {initial_loss:.4f} → {final_loss:.4f}"

class TestCNNInitialization:
    """Test CNN model initialization."""

    def test_default_initialization(self):
        model = CNN(num_classes=10)
        assert model is not None
        assert model.num_classes == 10

    def test_custom_num_classes(self):
        model = CNN(num_classes=47)
        assert model.num_classes == 47

    def test_parameter_count(self):
        model = CNN(num_classes=10)
        count = model.count_parameters()
        assert count > 0
        # Expected params:
        # Conv1: 32 * 1 * 3 * 3 + 32 = 320
        # Conv2: 64 * 32 * 3 * 3 + 64 = 18,496
        # FC1: 128 * 3136 + 128 = 401,536
        # FC2: 10 * 128 + 10 = 1,290
        # Total: ~421,642
        assert 400000 < count < 450000


class TestCNNForwardPass:
    """Test CNN forward pass and output shapes."""

    def test_forward_4d_input(self):
        """Test with standard image tensor (batch, C, H, W)."""
        model = CNN(num_classes=10)
        x = torch.randn(32, 1, 28, 28)
        output = model(x)
        assert output.shape == (32, 10)

    def test_forward_single_sample(self):
        """Test with a single sample."""
        model = CNN(num_classes=10)
        x = torch.randn(1, 1, 28, 28)
        output = model(x)
        assert output.shape == (1, 10)

    def test_forward_emnist_classes(self):
        """Test with EMNIST number of classes (47)."""
        model = CNN(num_classes=47)
        x = torch.randn(8, 1, 28, 28)
        output = model(x)
        assert output.shape == (8, 47)

    def test_output_not_nan(self):
        model = CNN(num_classes=10)
        x = torch.randn(4, 1, 28, 28)
        output = model(x)
        assert not torch.isnan(output).any()


class TestCNNTrainingStep:
    """Test a single training step with the CNN."""

    def test_loss_calculation(self):
        model = CNN(num_classes=10)
        criterion = nn.CrossEntropyLoss()
        x = torch.randn(16, 1, 28, 28)
        labels = torch.randint(0, 10, (16,))
        output = model(x)
        loss = criterion(output, labels)
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_backward_pass(self):
        model = CNN(num_classes=10)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        x = torch.randn(16, 1, 28, 28)
        labels = torch.randint(0, 10, (16,))

        # Forward
        output = model(x)
        loss = criterion(output, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Verify gradients were computed
        for param in model.parameters():
            if param.requires_grad:
                assert param.grad is not None
