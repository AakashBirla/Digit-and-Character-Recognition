"""Tests for data loading modules.

Verifies that MNIST and EMNIST datasets load correctly,
images have correct shapes, labels are valid, and
DataLoaders function properly.
"""

import sys
from pathlib import Path

import pytest
import torch

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.data.mnist_loader import (
    MNIST_CLASS_MAPPING,
    create_mnist_dataloaders,
    get_mnist_info,
    get_mnist_transforms,
    load_mnist,
)


class TestMNISTTransforms:
    """Test MNIST transform pipeline."""

    def test_transforms_with_normalization(self):
        transform = get_mnist_transforms(normalize=True)
        assert transform is not None

    def test_transforms_without_normalization(self):
        transform = get_mnist_transforms(normalize=False)
        assert transform is not None


class TestMNISTLoading:
    """Test MNIST dataset loading."""

    @pytest.fixture(scope="class")
    def mnist_datasets(self):
        train_ds, test_ds = load_mnist()
        return train_ds, test_ds

    def test_dataset_loads(self, mnist_datasets):
        train_ds, test_ds = mnist_datasets
        assert train_ds is not None
        assert test_ds is not None

    def test_train_size(self, mnist_datasets):
        train_ds, _ = mnist_datasets
        assert len(train_ds) == 60000

    def test_test_size(self, mnist_datasets):
        _, test_ds = mnist_datasets
        assert len(test_ds) == 10000

    def test_image_shape(self, mnist_datasets):
        train_ds, _ = mnist_datasets
        image, _ = train_ds[0]
        assert image.shape == (1, 28, 28), f"Expected (1, 28, 28), got {image.shape}"

    def test_image_is_tensor(self, mnist_datasets):
        train_ds, _ = mnist_datasets
        image, _ = train_ds[0]
        assert isinstance(image, torch.Tensor)

    def test_labels_valid_range(self, mnist_datasets):
        train_ds, _ = mnist_datasets
        for i in range(100):  # Check first 100 samples
            _, label = train_ds[i]
            assert 0 <= label <= 9, f"Label {label} out of range"

    def test_normalization_applied(self, mnist_datasets):
        """After normalization, pixel values should not all be in [0, 1]."""
        train_ds, _ = mnist_datasets
        image, _ = train_ds[0]
        # Normalized images can have negative values
        assert image.min() < 0 or image.max() > 1.0, \
            "Image doesn't appear to be normalized"


class TestMNISTDataLoader:
    """Test MNIST DataLoader creation."""

    @pytest.fixture(scope="class")
    def dataloaders(self):
        train_loader, test_loader = create_mnist_dataloaders(
            batch_size=32, num_workers=0
        )
        return train_loader, test_loader

    def test_dataloader_creation(self, dataloaders):
        train_loader, test_loader = dataloaders
        assert train_loader is not None
        assert test_loader is not None

    def test_batch_shape(self, dataloaders):
        train_loader, _ = dataloaders
        images, labels = next(iter(train_loader))
        assert images.shape == (32, 1, 28, 28)
        assert labels.shape == (32,)

    def test_batch_iteration(self, dataloaders):
        train_loader, _ = dataloaders
        batch_count = 0
        for images, labels in train_loader:
            assert images.ndim == 4
            assert labels.ndim == 1
            batch_count += 1
            if batch_count >= 3:  # Only check first 3 batches
                break
        assert batch_count == 3


class TestMNISTInfo:
    """Test MNIST dataset info extraction."""

    def test_get_info(self):
        train_ds, test_ds = load_mnist()
        info = get_mnist_info(train_ds, test_ds)

        assert info["dataset_name"] == "MNIST"
        assert info["image_shape"] == (1, 28, 28)
        assert info["num_train_samples"] == 60000
        assert info["num_test_samples"] == 10000
        assert info["num_classes"] == 10
        assert len(info["class_mapping"]) == 10
        assert len(info["class_distribution"]) == 10


class TestMNISTClassMapping:
    """Test MNIST class mapping."""

    def test_mapping_size(self):
        assert len(MNIST_CLASS_MAPPING) == 10

    def test_mapping_values(self):
        for i in range(10):
            assert MNIST_CLASS_MAPPING[i] == str(i)
