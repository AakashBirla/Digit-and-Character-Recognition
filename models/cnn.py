"""Convolutional Neural Network (CNN) for digit and character classification.

Architecture:
    Input (1×28×28)
    → Conv2D(1 → 32, 3×3) → ReLU → MaxPool(2×2)
    → Conv2D(32 → 64, 3×3) → ReLU → MaxPool(2×2)
    → Flatten
    → Linear(3136 → 128) → ReLU
    → Linear(128 → num_classes)

CNNs exploit spatial structure via convolutions and are translation-invariant,
making them much better suited for image data than MLPs.
"""

import torch
import torch.nn as nn


class CNN(nn.Module):
    """Convolutional Neural Network classifier for 28×28 grayscale images.

    Args:
        num_classes: Number of output classes (e.g. 10 for MNIST, 47 for EMNIST).

    Example:
        >>> model = CNN(num_classes=10)
        >>> x = torch.randn(32, 1, 28, 28)
        >>> output = model(x)
        >>> output.shape
        torch.Size([32, 10])
    """

    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.num_classes = num_classes

        # Feature extractor (Convolutional layers)
        self.features = nn.Sequential(
            # Block 1
            # Input: (batch, 1, 28, 28)
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            # Output: (batch, 32, 28, 28)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Output: (batch, 32, 14, 14)

            # Block 2
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            # Output: (batch, 64, 14, 14)
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            # Output: (batch, 64, 7, 7)
        )

        # Classifier (Fully connected layers)
        # Flattened size: 64 channels * 7 height * 7 width = 3136
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Input: (batch, 3136)
            nn.Linear(64 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            # Output: (batch, 128)
            nn.Linear(128, num_classes)
            # Output: (batch, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, 1, 28, 28).

        Returns:
            Logits tensor of shape (batch, num_classes).
        """
        x = self.features(x)
        x = self.classifier(x)
        return x

    def count_parameters(self) -> int:
        """Count total trainable parameters.

        Returns:
            Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"CNN("
            f"num_classes={self.num_classes}, "
            f"params={self.count_parameters():,})"
        )
