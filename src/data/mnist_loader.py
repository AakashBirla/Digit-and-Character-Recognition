"""MNIST dataset loading and DataLoader creation.

This module handles downloading, loading, and creating DataLoaders
for the MNIST handwritten digit dataset.
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

# Default data directory
DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# MNIST class mapping: index -> digit character
MNIST_CLASS_MAPPING = {i: str(i) for i in range(10)}


def get_mnist_transforms(normalize: bool = True) -> transforms.Compose:
    """Create the transform pipeline for MNIST images.

    Args:
        normalize: Whether to apply mean/std normalization.
            MNIST mean=0.1307, std=0.3081 (computed from training set).

    Returns:
        A composed transform pipeline.
    """
    transform_list = [transforms.ToTensor()]
    if normalize:
        transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
    return transforms.Compose(transform_list)


def load_mnist(
    data_dir: Optional[Path] = None,
    normalize: bool = True,
) -> tuple[datasets.MNIST, datasets.MNIST]:
    """Download and load MNIST training and test datasets.

    Args:
        data_dir: Directory to store/load data. Defaults to project data/ dir.
        normalize: Whether to normalize images.

    Returns:
        Tuple of (train_dataset, test_dataset).
    """
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    data_dir = Path(data_dir)

    transform = get_mnist_transforms(normalize=normalize)

    logger.info("Loading MNIST training dataset from %s", data_dir)
    train_dataset = datasets.MNIST(
        root=str(data_dir),
        train=True,
        download=True,
        transform=transform,
    )

    logger.info("Loading MNIST test dataset from %s", data_dir)
    test_dataset = datasets.MNIST(
        root=str(data_dir),
        train=False,
        download=True,
        transform=transform,
    )

    logger.info(
        "MNIST loaded — Train: %d samples, Test: %d samples",
        len(train_dataset), len(test_dataset)
    )
    return train_dataset, test_dataset


def create_mnist_dataloaders(
    batch_size: int = 64,
    data_dir: Optional[Path] = None,
    normalize: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Create DataLoaders for MNIST training and test sets.

    Args:
        batch_size: Number of samples per batch.
        data_dir: Directory to store/load data.
        normalize: Whether to normalize images.
        num_workers: Number of worker processes for data loading.
        pin_memory: Whether to pin memory for faster GPU transfer.

    Returns:
        Tuple of (train_loader, test_loader).
    """
    train_dataset, test_dataset = load_mnist(
        data_dir=data_dir, normalize=normalize
    )

    # Use pin_memory only when CUDA is available
    use_pin_memory = pin_memory and torch.cuda.is_available()

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
    )

    logger.info(
        "DataLoaders created — Batch size: %d, Train batches: %d, Test batches: %d",
        batch_size, len(train_loader), len(test_loader)
    )
    return train_loader, test_loader


def get_mnist_info(train_dataset: datasets.MNIST, test_dataset: datasets.MNIST) -> dict:
    """Extract and return dataset information.

    Args:
        train_dataset: MNIST training dataset.
        test_dataset: MNIST test dataset.

    Returns:
        Dictionary with dataset metadata.
    """
    # Get a sample to determine shape
    sample_image, sample_label = train_dataset[0]

    # Compute class distribution
    train_labels = train_dataset.targets
    class_counts = torch.bincount(train_labels, minlength=10)

    info = {
        "dataset_name": "MNIST",
        "image_shape": tuple(sample_image.shape),
        "num_train_samples": len(train_dataset),
        "num_test_samples": len(test_dataset),
        "num_classes": 10,
        "class_mapping": MNIST_CLASS_MAPPING,
        "class_distribution": {str(i): int(class_counts[i]) for i in range(10)},
    }
    return info


def print_mnist_info(info: dict) -> None:
    """Print MNIST dataset information in a formatted manner.

    Args:
        info: Dictionary returned by get_mnist_info().
    """
    print("=" * 50)
    print(f"Dataset: {info['dataset_name']}")
    print("=" * 50)
    print(f"Image shape:           {info['image_shape']}")
    print(f"Number of train samples: {info['num_train_samples']}")
    print(f"Number of test samples:  {info['num_test_samples']}")
    print(f"Number of classes:       {info['num_classes']}")
    print()
    print("Class Distribution (Training Set):")
    print("-" * 30)
    for class_id, count in info["class_distribution"].items():
        label = info["class_mapping"][int(class_id)]
        print(f"  {label}: {count:>6,}")
    print("=" * 50)
