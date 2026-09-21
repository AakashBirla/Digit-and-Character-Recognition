"""EMNIST dataset loading and DataLoader creation.

This module handles downloading, loading, and creating DataLoaders
for the EMNIST (Extended MNIST) handwritten character dataset.

EMNIST Splits Available:
    - ByClass:  62 classes (0-9, A-Z, a-z) — unbalanced
    - ByMerge:  47 classes (merges similar upper/lowercase)
    - Balanced: 47 classes (balanced version of ByMerge)
    - Letters:  26 classes (A-Z, case-insensitive)
    - Digits:   10 classes (0-9)
    - MNIST:    10 classes (same as original MNIST)

Selected Split: Balanced
    - 47 classes with balanced training samples
    - Classes: 0-9, A-Z, and select lowercase letters
    - Well-suited for combined digit + character recognition

IMPORTANT: EMNIST images are transposed relative to MNIST.
    The raw EMNIST images from torchvision are stored in a
    transposed orientation and need to be flipped along the
    diagonal (transpose) to appear correctly oriented.
    This is handled automatically by applying
    `transforms.Lambda(lambda img: img.transpose(Image.TRANSPOSE))`
    BEFORE ToTensor().
"""

import logging
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

logger = logging.getLogger(__name__)

DEFAULT_DATA_DIR = Path(__file__).resolve().parent.parent.parent / "data"

# EMNIST Balanced class mapping (47 classes)
# Classes 0-9: digits '0'-'9'
# Classes 10-35: uppercase 'A'-'Z'
# Classes 36-46: lowercase letters that are visually distinct from uppercase:
#   a, b, d, e, f, g, h, n, q, r, t
# The "Balanced" split merges visually ambiguous lowercase letters
# (c, i, j, k, l, m, o, p, s, u, v, w, x, y, z) with their uppercase forms.
EMNIST_BALANCED_CLASS_MAPPING = {}

# Digits 0-9
for i in range(10):
    EMNIST_BALANCED_CLASS_MAPPING[i] = str(i)

# Uppercase A-Z (classes 10-35)
for i in range(26):
    EMNIST_BALANCED_CLASS_MAPPING[10 + i] = chr(ord('A') + i)

# Lowercase visually-distinct letters (classes 36-46)
_BALANCED_LOWERCASE = ['a', 'b', 'd', 'e', 'f', 'g', 'h', 'n', 'q', 'r', 't']
for i, ch in enumerate(_BALANCED_LOWERCASE):
    EMNIST_BALANCED_CLASS_MAPPING[36 + i] = ch

# Number of classes for each split
EMNIST_SPLIT_NUM_CLASSES = {
    "byclass": 62,
    "bymerge": 47,
    "balanced": 47,
    "letters": 26,
    "digits": 10,
    "mnist": 10,
}


def get_emnist_transforms(
    normalize: bool = True,
    correct_orientation: bool = True,
) -> transforms.Compose:
    """Create the transform pipeline for EMNIST images.

    Args:
        normalize: Whether to apply mean/std normalization.
            Uses MNIST-compatible values: mean=0.1307, std=0.3081.
        correct_orientation: Whether to transpose images to correct
            the EMNIST orientation issue.

    Returns:
        A composed transform pipeline.
    """
    transform_list = []
    if correct_orientation:
        transform_list.append(
            transforms.Lambda(lambda img: img.transpose(Image.TRANSPOSE))
        )
    transform_list.append(transforms.ToTensor())
    if normalize:
        transform_list.append(transforms.Normalize((0.1307,), (0.3081,)))
    return transforms.Compose(transform_list)


def load_emnist(
    split: str = "balanced",
    data_dir: Optional[Path] = None,
    normalize: bool = True,
    correct_orientation: bool = True,
) -> tuple[datasets.EMNIST, datasets.EMNIST]:
    """Download and load EMNIST training and test datasets.

    Args:
        split: EMNIST split to use. One of:
            'byclass', 'bymerge', 'balanced', 'letters', 'digits', 'mnist'.
        data_dir: Directory to store/load data.
        normalize: Whether to normalize images.
        correct_orientation: Whether to transpose images.

    Returns:
        Tuple of (train_dataset, test_dataset).

    Raises:
        ValueError: If an invalid split name is provided.
    """
    valid_splits = list(EMNIST_SPLIT_NUM_CLASSES.keys())
    if split not in valid_splits:
        raise ValueError(
            f"Invalid EMNIST split '{split}'. Must be one of: {valid_splits}"
        )

    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
    data_dir = Path(data_dir)

    transform = get_emnist_transforms(
        normalize=normalize,
        correct_orientation=correct_orientation,
    )

    logger.info("Loading EMNIST '%s' training dataset from %s", split, data_dir)
    train_dataset = datasets.EMNIST(
        root=str(data_dir),
        split=split,
        train=True,
        download=True,
        transform=transform,
    )

    logger.info("Loading EMNIST '%s' test dataset from %s", split, data_dir)
    test_dataset = datasets.EMNIST(
        root=str(data_dir),
        split=split,
        train=False,
        download=True,
        transform=transform,
    )

    logger.info(
        "EMNIST '%s' loaded — Train: %d samples, Test: %d samples",
        split, len(train_dataset), len(test_dataset)
    )
    return train_dataset, test_dataset


def create_emnist_dataloaders(
    split: str = "balanced",
    batch_size: int = 64,
    data_dir: Optional[Path] = None,
    normalize: bool = True,
    correct_orientation: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
) -> tuple[DataLoader, DataLoader]:
    """Create DataLoaders for EMNIST training and test sets.

    Args:
        split: EMNIST split to use.
        batch_size: Number of samples per batch.
        data_dir: Directory to store/load data.
        normalize: Whether to normalize images.
        correct_orientation: Whether to transpose images.
        num_workers: Number of worker processes.
        pin_memory: Whether to pin memory.

    Returns:
        Tuple of (train_loader, test_loader).
    """
    train_dataset, test_dataset = load_emnist(
        split=split,
        data_dir=data_dir,
        normalize=normalize,
        correct_orientation=correct_orientation,
    )

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
        "EMNIST DataLoaders created — Batch size: %d, Train batches: %d, Test batches: %d",
        batch_size, len(train_loader), len(test_loader)
    )
    return train_loader, test_loader


def get_class_mapping(split: str = "balanced") -> dict[int, str]:
    """Get the class ID to character mapping for a given EMNIST split.

    Args:
        split: The EMNIST split name.

    Returns:
        Dictionary mapping class IDs to character labels.
    """
    if split == "balanced":
        return EMNIST_BALANCED_CLASS_MAPPING.copy()
    elif split == "digits" or split == "mnist":
        return {i: str(i) for i in range(10)}
    elif split == "letters":
        # Letters split: classes 1-26 map to A-Z (class 0 is N/A)
        return {i: chr(ord('A') + i - 1) for i in range(1, 27)}
    elif split in ("byclass", "bymerge"):
        mapping = {}
        for i in range(10):
            mapping[i] = str(i)
        for i in range(26):
            mapping[10 + i] = chr(ord('A') + i)
        for i in range(26):
            mapping[36 + i] = chr(ord('a') + i)
        return mapping
    else:
        raise ValueError(f"Unknown split: {split}")


def get_emnist_info(
    train_dataset: datasets.EMNIST,
    test_dataset: datasets.EMNIST,
    split: str = "balanced",
) -> dict:
    """Extract and return EMNIST dataset information.

    Args:
        train_dataset: EMNIST training dataset.
        test_dataset: EMNIST test dataset.
        split: The EMNIST split used.

    Returns:
        Dictionary with dataset metadata.
    """
    sample_image, sample_label = train_dataset[0]
    class_mapping = get_class_mapping(split)
    num_classes = EMNIST_SPLIT_NUM_CLASSES[split]

    train_labels = train_dataset.targets
    class_counts = torch.bincount(train_labels, minlength=num_classes)

    info = {
        "dataset_name": f"EMNIST ({split})",
        "split": split,
        "image_shape": tuple(sample_image.shape),
        "num_train_samples": len(train_dataset),
        "num_test_samples": len(test_dataset),
        "num_classes": num_classes,
        "class_mapping": class_mapping,
        "class_distribution": {
            str(i): int(class_counts[i]) for i in range(num_classes)
        },
        "orientation_corrected": True,
        "differences_from_mnist": [
            "EMNIST images are transposed relative to MNIST and require correction",
            f"EMNIST Balanced has {num_classes} classes vs MNIST's 10",
            "EMNIST includes alphabetic characters in addition to digits",
            "The Balanced split merges visually similar upper/lowercase letters",
            "Some lowercase letters (a, b, d, e, f, g, h, n, q, r, t) are kept separate",
        ],
    }
    return info


def print_emnist_info(info: dict) -> None:
    """Print EMNIST dataset information in a formatted manner.

    Args:
        info: Dictionary returned by get_emnist_info().
    """
    print("=" * 60)
    print(f"Dataset: {info['dataset_name']}")
    print("=" * 60)
    print(f"Split:                   {info['split']}")
    print(f"Image shape:             {info['image_shape']}")
    print(f"Number of train samples: {info['num_train_samples']}")
    print(f"Number of test samples:  {info['num_test_samples']}")
    print(f"Number of classes:       {info['num_classes']}")
    print(f"Orientation corrected:   {info['orientation_corrected']}")
    print()
    print("Class Mapping:")
    print("-" * 30)
    for class_id, char in info["class_mapping"].items():
        count = info["class_distribution"].get(str(class_id), 0)
        print(f"  Class {class_id:>2d} -> '{char}': {count:>6,}")
    print()
    print("Key Differences from MNIST:")
    print("-" * 30)
    for diff in info["differences_from_mnist"]:
        print(f"  • {diff}")
    print("=" * 60)
