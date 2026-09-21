"""Image preprocessing utilities for inference.

Provides functions to convert hand-drawn images into the format
expected by trained models (28x28 normalized tensors).

Placeholder for Stage 5 implementation.
"""

import logging
from typing import Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def preprocess_image(image: np.ndarray) -> torch.Tensor:
    """Convert a raw image to a normalized 28x28 tensor.

    Placeholder — full implementation in Stage 5.

    Args:
        image: Input image as a NumPy array.

    Returns:
        Preprocessed tensor of shape (1, 1, 28, 28).
    """
    raise NotImplementedError("Full preprocessing will be implemented in Stage 5.")
