#!/usr/bin/env python3
"""Stage 2: Train MLP baseline on MNIST.

Trains a Multi-Layer Perceptron on MNIST digits and generates
training curves (loss and accuracy plots).

Usage:
    python train_mlp.py [--epochs EPOCHS] [--batch-size BATCH_SIZE] [--lr LR]
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

from models.mlp import MLP
from src.data.mnist_loader import MNIST_CLASS_MAPPING, create_mnist_dataloaders
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


def plot_training_curves(history: dict, save_dir: Path) -> None:
    """Generate and save training loss and accuracy plots.

    Args:
        history: Training history dictionary from train_model().
        save_dir: Directory to save the plots.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_losses"]) + 1)

    # Loss plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, history["train_losses"], "b-o", label="Train Loss", markersize=4)
    ax.plot(epochs, history["val_losses"], "r-o", label="Val Loss", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("MLP Training Loss (MNIST)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "mlp_training_loss.png", dpi=150)
    plt.close(fig)
    logger.info("Saved mlp_training_loss.png")

    # Accuracy plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(epochs, history["train_accuracies"], "b-o", label="Train Acc", markersize=4)
    ax.plot(epochs, history["val_accuracies"], "r-o", label="Val Acc", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("MLP Training Accuracy (MNIST)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_dir / "mlp_training_accuracy.png", dpi=150)
    plt.close(fig)
    logger.info("Saved mlp_training_accuracy.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train MLP on MNIST")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    print()
    print("=" * 60)
    print("  STAGE 2: MLP Baseline Training on MNIST")
    print("=" * 60)
    print()

    # Setup
    set_seed(args.seed)
    device = get_device()
    print(f"Device: {device}")
    print(f"Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print()

    # Data
    train_loader, test_loader = create_mnist_dataloaders(
        batch_size=args.batch_size, num_workers=2
    )

    # Model
    model = MLP(num_classes=10)
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
        model_name="mnist_mlp",
        class_mapping=MNIST_CLASS_MAPPING,
    )

    # Plot results
    plot_training_curves(history, FIGURES_DIR)

    # Summary
    print()
    print("=" * 60)
    print("  Results Summary")
    print("=" * 60)
    print(f"  Best Val Accuracy: {history['best_val_accuracy']:.2f}%")
    print(f"  Best Epoch:        {history['best_epoch']}")
    print(f"  Training Time:     {history['training_time']:.1f}s")
    print()
    print("  Limitations of MLP for image data:")
    print("  • Treats each pixel independently — no spatial awareness")
    print("  • Loses 2D structure by flattening to 1D vector")
    print("  • More parameters than necessary (784 inputs per neuron)")
    print("  • Not translation-invariant")
    print("  • CNNs (Stage 3) exploit spatial structure via convolutions")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
