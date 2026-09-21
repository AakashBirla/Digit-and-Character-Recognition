"""Visualization utilities for dataset exploration and model analysis.

Provides functions to display sample images, class distributions,
and save plots to the results directory.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)

# Default output directory for figures
DEFAULT_FIGURES_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "figures"


def plot_sample_images(
    dataset: torch.utils.data.Dataset,
    class_mapping: dict[int, str],
    num_samples: int = 25,
    title: str = "Sample Images",
    save_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    """Plot a grid of sample images from the dataset.

    Args:
        dataset: A PyTorch dataset returning (image_tensor, label).
        class_mapping: Dictionary mapping class IDs to character labels.
        num_samples: Number of samples to display. Should be a perfect square.
        title: Title for the plot.
        save_path: Path to save the figure. If None, saves to default dir.
        show: Whether to call plt.show() (set False for non-interactive).
    """
    cols = int(np.ceil(np.sqrt(num_samples)))
    rows = int(np.ceil(num_samples / cols))

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 1.5, rows * 1.8))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    axes_flat = axes.flatten() if hasattr(axes, "flatten") else [axes]

    for i in range(len(axes_flat)):
        ax = axes_flat[i]
        if i < num_samples and i < len(dataset):
            image, label = dataset[i]
            # Handle normalized images: convert tensor to displayable format
            if isinstance(image, torch.Tensor):
                image_np = image.squeeze().numpy()
            else:
                image_np = np.array(image)

            label_str = class_mapping.get(label, str(label))
            ax.imshow(image_np, cmap="gray")
            ax.set_title(f"{label_str}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()

    if save_path is None:
        save_dir = DEFAULT_FIGURES_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{title.lower().replace(' ', '_')}.png"
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("Saved figure to %s", save_path)

    if show:
        plt.show()
    plt.close(fig)


def plot_class_distribution(
    class_distribution: dict[str, int],
    class_mapping: dict[int, str],
    title: str = "Class Distribution",
    save_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    """Plot a bar chart showing the distribution of classes.

    Args:
        class_distribution: Dict mapping class ID (str) to count.
        class_mapping: Dict mapping class ID (int) to character label.
        title: Title for the plot.
        save_path: Path to save the figure.
        show: Whether to call plt.show().
    """
    class_ids = sorted(class_distribution.keys(), key=int)
    labels = [class_mapping.get(int(cid), str(cid)) for cid in class_ids]
    counts = [class_distribution[cid] for cid in class_ids]

    fig, ax = plt.subplots(figsize=(max(12, len(labels) * 0.4), 5))
    bars = ax.bar(range(len(labels)), counts, color="steelblue", edgecolor="black", linewidth=0.5)

    ax.set_xlabel("Class", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45 if len(labels) > 15 else 0, ha="right" if len(labels) > 15 else "center", fontsize=8)

    # Add count labels on bars if not too many
    if len(labels) <= 20:
        for bar, count in zip(bars, counts):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                f"{count:,}",
                ha="center", va="bottom", fontsize=7,
            )

    plt.tight_layout()

    if save_path is None:
        save_dir = DEFAULT_FIGURES_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{title.lower().replace(' ', '_')}.png"
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("Saved figure to %s", save_path)

    if show:
        plt.show()
    plt.close(fig)


def plot_batch_grid(
    dataloader: DataLoader,
    class_mapping: dict[int, str],
    title: str = "Batch Samples",
    save_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    """Plot a grid of images from the first batch of a DataLoader.

    Args:
        dataloader: A PyTorch DataLoader.
        class_mapping: Dict mapping class IDs to character labels.
        title: Title for the plot.
        save_path: Path to save the figure.
        show: Whether to call plt.show().
    """
    images, labels = next(iter(dataloader))
    num_display = min(16, images.size(0))

    fig, axes = plt.subplots(4, 4, figsize=(8, 8))
    fig.suptitle(title, fontsize=14, fontweight="bold")

    for i, ax in enumerate(axes.flatten()):
        if i < num_display:
            image_np = images[i].squeeze().numpy()
            label_str = class_mapping.get(labels[i].item(), str(labels[i].item()))
            ax.imshow(image_np, cmap="gray")
            ax.set_title(f"{label_str}", fontsize=9)
        ax.axis("off")

    plt.tight_layout()

    if save_path is None:
        save_dir = DEFAULT_FIGURES_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{title.lower().replace(' ', '_')}.png"
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("Saved figure to %s", save_path)

    if show:
        plt.show()
    plt.close(fig)


def plot_image_statistics(
    dataset: torch.utils.data.Dataset,
    num_samples: int = 1000,
    title: str = "Pixel Intensity Distribution",
    save_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    """Plot histogram of pixel intensities from dataset samples.

    Args:
        dataset: A PyTorch dataset.
        num_samples: Number of samples to use for the histogram.
        title: Title for the plot.
        save_path: Path to save the figure.
        show: Whether to call plt.show().
    """
    pixel_values = []
    for i in range(min(num_samples, len(dataset))):
        image, _ = dataset[i]
        if isinstance(image, torch.Tensor):
            pixel_values.append(image.numpy().flatten())
        else:
            pixel_values.append(np.array(image).flatten())

    all_pixels = np.concatenate(pixel_values)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(all_pixels, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    ax.set_xlabel("Pixel Value", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.axvline(all_pixels.mean(), color="red", linestyle="--", label=f"Mean: {all_pixels.mean():.3f}")
    ax.axvline(all_pixels.std(), color="green", linestyle="--", label=f"Std: {all_pixels.std():.3f}")
    ax.legend()

    plt.tight_layout()

    if save_path is None:
        save_dir = DEFAULT_FIGURES_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{title.lower().replace(' ', '_')}.png"
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info("Saved figure to %s", save_path)

    if show:
        plt.show()
    plt.close(fig)
