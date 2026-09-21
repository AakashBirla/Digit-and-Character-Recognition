#!/usr/bin/env python3
"""Stage 1: MNIST Data Exploration and Visualization.

This script demonstrates the complete data pipeline:
    MNIST download → Dataset loading → DataLoader → Normalization → Visualization

It prints dataset statistics and saves visualization plots to results/figures/.

Usage:
    python explore_data.py
"""

import logging
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.mnist_loader import (
    MNIST_CLASS_MAPPING,
    create_mnist_dataloaders,
    get_mnist_info,
    load_mnist,
    print_mnist_info,
)
from src.utils.seed import get_device, set_seed
from src.utils.visualization import (
    plot_batch_grid,
    plot_class_distribution,
    plot_image_statistics,
    plot_sample_images,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"


def main() -> None:
    """Run the MNIST data exploration pipeline."""
    print()
    print("=" * 60)
    print("  STAGE 1: MNIST Data Exploration and Visualization")
    print("=" * 60)
    print()

    # --- Reproducibility ---
    set_seed(42)
    device = get_device()
    print(f"Device: {device}\n")

    # --- Load MNIST ---
    print("Loading MNIST dataset...")
    train_dataset, test_dataset = load_mnist()

    # --- Dataset Info ---
    info = get_mnist_info(train_dataset, test_dataset)
    print()
    print_mnist_info(info)
    print()

    # --- Create DataLoaders ---
    print("Creating DataLoaders...")
    train_loader, test_loader = create_mnist_dataloaders(
        batch_size=64, num_workers=0
    )

    # Print batch info
    images, labels = next(iter(train_loader))
    print(f"\nBatch shape:  {images.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Pixel range:  [{images.min():.4f}, {images.max():.4f}]")
    print()

    # --- Visualizations ---
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("Generating visualizations...")

    # 1. Sample images
    plot_sample_images(
        dataset=train_dataset,
        class_mapping=MNIST_CLASS_MAPPING,
        num_samples=25,
        title="MNIST Sample Images",
        save_path=FIGURES_DIR / "mnist_sample_images.png",
    )
    print("  ✓ Saved: mnist_sample_images.png")

    # 2. Class distribution
    plot_class_distribution(
        class_distribution=info["class_distribution"],
        class_mapping=MNIST_CLASS_MAPPING,
        title="MNIST Class Distribution",
        save_path=FIGURES_DIR / "mnist_class_distribution.png",
    )
    print("  ✓ Saved: mnist_class_distribution.png")

    # 3. Batch grid
    plot_batch_grid(
        dataloader=train_loader,
        class_mapping=MNIST_CLASS_MAPPING,
        title="MNIST Batch Samples",
        save_path=FIGURES_DIR / "mnist_batch_samples.png",
    )
    print("  ✓ Saved: mnist_batch_samples.png")

    # 4. Pixel intensity distribution
    plot_image_statistics(
        dataset=train_dataset,
        num_samples=1000,
        title="MNIST Pixel Intensity Distribution",
        save_path=FIGURES_DIR / "mnist_pixel_distribution.png",
    )
    print("  ✓ Saved: mnist_pixel_distribution.png")

    print()
    print("=" * 60)
    print("  Stage 1 complete. Figures saved to results/figures/")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
