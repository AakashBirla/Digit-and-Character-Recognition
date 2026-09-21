"""Multi-Layer Perceptron (MLP) for digit and character classification.

Architecture:
    Input (28×28) → Flatten (784) → Linear(784, 256) → ReLU
    → Linear(256, 128) → ReLU → Linear(128, num_classes)

This is a baseline fully-connected network. It treats each pixel
independently and does not exploit spatial structure — CNNs (Stage 3)
will address that limitation.
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Multi-Layer Perceptron classifier for 28×28 grayscale images.

    Args:
        num_classes: Number of output classes (e.g. 10 for MNIST, 47 for EMNIST).
        input_size: Flattened input dimension. Default 784 (28×28).
        hidden_sizes: Tuple of hidden layer sizes. Default (256, 128).
        dropout_rate: Dropout probability between layers. Default 0.2.

    Example:
        >>> model = MLP(num_classes=10)
        >>> x = torch.randn(32, 1, 28, 28)
        >>> output = model(x)
        >>> output.shape
        torch.Size([32, 10])
    """

    def __init__(
        self,
        num_classes: int = 10,
        input_size: int = 784,
        hidden_sizes: tuple[int, ...] = (256, 128),
        dropout_rate: float = 0.2,
    ) -> None:
        super().__init__()

        self.num_classes = num_classes
        self.input_size = input_size
        self.hidden_sizes = hidden_sizes

        # Build layers dynamically
        layers: list[nn.Module] = []
        prev_size = input_size

        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
            ])
            prev_size = hidden_size

        # Output layer (no activation — handled by CrossEntropyLoss)
        layers.append(nn.Linear(prev_size, num_classes))

        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.

        Args:
            x: Input tensor of shape (batch, 1, 28, 28) or (batch, 784).

        Returns:
            Logits tensor of shape (batch, num_classes).

        Tensor dimensions walkthrough (default architecture):
            Input:   (batch, 1, 28, 28)
            Flatten: (batch, 784)
            Linear:  (batch, 256)
            ReLU:    (batch, 256)
            Linear:  (batch, 128)
            ReLU:    (batch, 128)
            Linear:  (batch, num_classes)
        """
        # Flatten spatial dimensions: (batch, C, H, W) → (batch, C*H*W)
        x = x.view(x.size(0), -1)
        return self.network(x)

    def count_parameters(self) -> int:
        """Count total trainable parameters.

        Returns:
            Number of trainable parameters.
        """
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self) -> str:
        return (
            f"MLP(input_size={self.input_size}, "
            f"hidden_sizes={self.hidden_sizes}, "
            f"num_classes={self.num_classes}, "
            f"params={self.count_parameters():,})"
        )
