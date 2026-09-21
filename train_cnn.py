#!/usr/bin/env python3
"""Stage 3: Train CNN on MNIST and EMNIST.

Trains a Convolutional Neural Network on digit and character datasets
and generates training curves and evaluation summaries.

Usage:
    python train_cnn.py [--dataset mnist|emnist] [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--lr LR]
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
import torch.nn as nn

from models.cnn import CNN
from src.data.mnist_loader import MNIST_CLASS_MAPPING, create_mnist_dataloaders
from src.data.emnist_loader import get_class_mapping, create_emnist_dataloaders
from src.training.train import train_model
from src.utils.seed import get_device, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

FIGURES_DIR = PROJECT_ROOT / "results" / "figures"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


def plot_training_curves(history: dict, save_dir: Path, dataset_name: str) -> None:
    """Generate and save training loss and accuracy plots.

    Args:
        history: Training history dictionary from train_model().
        save_dir: Directory to save the plots.
        dataset_name: Name of dataset (e.g., 'mnist', 'emnist') for file naming.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_losses"]) + 1)

    # Loss plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, history["train_losses"], "b-o", label="Train Loss", markersize=4)
    ax.plot(epochs, history["val_losses"], "r-o", label="Val Loss", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"CNN Training Loss ({dataset_name.upper()})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f"cnn_{dataset_name}_training_loss.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved cnn_{dataset_name}_training_loss.png")

    # Accuracy plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, history["train_accuracies"], "b-o", label="Train Acc", markersize=4)
    ax.plot(epochs, history["val_accuracies"], "r-o", label="Val Acc", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"CNN Training Accuracy ({dataset_name.upper()})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / f"cnn_{dataset_name}_training_accuracy.png", dpi=150)
    plt.close(fig)
    logger.info(f"Saved cnn_{dataset_name}_training_accuracy.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train CNN on MNIST or EMNIST")
    parser.add_argument("--dataset", type=str, choices=["mnist", "emnist"], default="mnist", help="Dataset to train on")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print()
    print("=" * 60)
    print(f"  STAGE 3: CNN Training on {args.dataset.upper()}")
    print("=" * 60)
    print()

    # Setup
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print()

    # Data
    if args.dataset == "mnist":
        train_loader, test_loader = create_mnist_dataloaders(
            batch_size=args.batch_size, num_workers=2
        )
        class_mapping = MNIST_CLASS_MAPPING
        num_classes = 10
    elif args.dataset == "emnist":
        train_loader, test_loader = create_emnist_dataloaders(
            split="balanced", batch_size=args.batch_size, num_workers=2
        )
        class_mapping = get_class_mapping(split="balanced")
        num_classes = 47
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # Model
    model = CNN(num_classes=num_classes)
    print(f"Model: {model}")
    print(f"Parameters: {model.count_parameters():,}")
    print()

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # Train
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=test_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=args.epochs,
        checkpoint_dir=CHECKPOINT_DIR,
        model_name=f"{args.dataset}_cnn",
        class_mapping=class_mapping,
    )

    # Plot results
    plot_training_curves(history, FIGURES_DIR, args.dataset)

    # Summary
    print()
    print("=" * 60)
    print("  Results Summary")
    print("=" * 60)
    print(f"  Best Val Accuracy: {history['best_val_accuracy']:.2f}%")
    print(f"  Best Epoch:        {history['best_epoch']}")
    print(f"  Training Time:     {history['training_time']:.1f}s")
    print()
    print("  CNN Advantages over MLP:")
    print("  • Exploits 2D spatial structure via convolutional filters")
    print("  • Weight sharing reduces parameters and improves generalization")
    print("  • Max pooling provides translation invariance")
    print("  • Hierarchical feature learning (edges -> shapes -> digits/chars)")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
