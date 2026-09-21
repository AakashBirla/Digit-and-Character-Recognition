"""Reproducibility utilities for setting random seeds."""

import random
import logging

import numpy as np
import torch

logger = logging.getLogger(__name__)


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility across all frameworks.

    Args:
        seed: The seed value to use. Defaults to 42.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    logger.info(
        "Seeds set — Python: %d, NumPy: %d, PyTorch: %d, CUDA: %s",
        seed, seed, seed, torch.cuda.is_available()
    )


def get_device() -> torch.device:
    """Detect and return the best available compute device.

    Priority: CUDA > MPS > CPU

    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info("Using CUDA device: %s", torch.cuda.get_device_name(0))
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("Using MPS device (Apple Silicon)")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU device")
    return device
