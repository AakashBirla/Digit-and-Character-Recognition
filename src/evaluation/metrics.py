"""Evaluation metrics and visualization for trained models.

Provides functions to compute classification reports and plot
confusion matrices.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Default output directory for metrics
DEFAULT_METRICS_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "metrics"
DEFAULT_FIGURES_DIR = Path(__file__).resolve().parent.parent.parent / "results" / "figures"


def get_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Get all model predictions for a dataset.

    Args:
        model: Trained neural network model.
        dataloader: DataLoader for the dataset.
        device: Device to run inference on.

    Returns:
        Tuple of (true_labels, predicted_labels) as NumPy arrays.
    """
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(dataloader, desc="Evaluating", leave=False)
        for images, labels in progress_bar:
            images = images.to(device)
            outputs = model(images)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())

    return np.array(all_labels), np.array(all_preds)


def generate_classification_report(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    class_mapping: dict[int, str],
    save_path: Optional[Path] = None,
) -> str:
    """Generate and optionally save a detailed classification report.

    Args:
        true_labels: Ground truth labels.
        predicted_labels: Model predictions.
        class_mapping: Mapping of class IDs to string labels.
        save_path: Path to save the text report.

    Returns:
        The classification report as a string.
    """
    class_ids = sorted(list(class_mapping.keys()))
    target_names = [class_mapping[i] for i in class_ids]

    report = classification_report(
        true_labels,
        predicted_labels,
        labels=class_ids,
        target_names=target_names,
        digits=4,
    )

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            f.write(report)
        logger.info(f"Saved classification report to {save_path}")

    return report


def plot_confusion_matrix(
    true_labels: np.ndarray,
    predicted_labels: np.ndarray,
    class_mapping: dict[int, str],
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None,
    show: bool = False,
) -> None:
    """Plot and save a confusion matrix.

    Args:
        true_labels: Ground truth labels.
        predicted_labels: Model predictions.
        class_mapping: Mapping of class IDs to string labels.
        title: Plot title.
        save_path: Path to save the plot.
        show: Whether to display the plot.
    """
    class_ids = sorted(list(class_mapping.keys()))
    labels = [class_mapping[i] for i in class_ids]

    cm = confusion_matrix(true_labels, predicted_labels, labels=class_ids)

    # Calculate percentages for the diagonal
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_normalized = np.nan_to_num(cm_normalized)

    # Plot
    # Adjust figure size based on number of classes
    fig_size = max(10, len(labels) * 0.4)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size * 0.8))

    # Using seaborn for a nicer heatmap
    sns.heatmap(
        cm,
        annot=True if len(labels) <= 20 else False,  # Hide numbers if too many classes
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        cbar_kws={'label': 'Count'}
    )

    ax.set_title(title, fontsize=14, fontweight="bold", pad=20)
    ax.set_xlabel("Predicted Label", fontsize=12, labelpad=10)
    ax.set_ylabel("True Label", fontsize=12, labelpad=10)

    # Rotate tick labels if there are many classes
    if len(labels) > 15:
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)

    plt.tight_layout()

    if save_path is None:
        save_dir = DEFAULT_FIGURES_DIR
        save_dir.mkdir(parents=True, exist_ok=True)
        save_path = save_dir / f"{title.lower().replace(' ', '_')}.png"
    else:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    logger.info(f"Saved confusion matrix to {save_path}")

    if show:
        plt.show()
    plt.close(fig)
