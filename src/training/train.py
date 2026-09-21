"""Training loop for neural network models.

Provides a generic training function that works with any model
(MLP, CNN) and any dataset (MNIST, EMNIST).
"""

import logging
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    epoch: int,
    total_epochs: int,
) -> tuple[float, float]:
    """Train the model for one epoch.

    Args:
        model: The neural network model.
        train_loader: DataLoader for training data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on.
        epoch: Current epoch number (1-indexed).
        total_epochs: Total number of epochs.

    Returns:
        Tuple of (average_loss, accuracy_percentage).
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(
        train_loader,
        desc=f"Epoch {epoch}/{total_epochs} [Train]",
        leave=False,
    )

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Track metrics
        running_loss += loss.item() * images.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        progress_bar.set_postfix({
            "loss": f"{loss.item():.4f}",
            "acc": f"{100.0 * correct / total:.2f}%",
        })

    avg_loss = running_loss / total
    accuracy = 100.0 * correct / total
    return avg_loss, accuracy


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: Optimizer,
    device: torch.device,
    epochs: int = 10,
    checkpoint_dir: Optional[Path] = None,
    model_name: str = "model",
    class_mapping: Optional[dict] = None,
) -> dict:
    """Complete training loop with validation and checkpointing.

    Args:
        model: The neural network model.
        train_loader: DataLoader for training data.
        val_loader: DataLoader for validation/test data.
        criterion: Loss function.
        optimizer: Optimizer.
        device: Device to train on.
        epochs: Number of training epochs.
        checkpoint_dir: Directory to save model checkpoints.
        model_name: Name prefix for checkpoint files.
        class_mapping: Optional class ID to label mapping to save with checkpoint.

    Returns:
        Dictionary containing training history with keys:
            train_losses, train_accuracies, val_losses, val_accuracies,
            best_val_accuracy, best_epoch, training_time.
    """
    from src.training.validate import validate

    model.to(device)

    history = {
        "train_losses": [],
        "train_accuracies": [],
        "val_losses": [],
        "val_accuracies": [],
        "best_val_accuracy": 0.0,
        "best_epoch": 0,
    }

    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    start_time = time.time()

    logger.info(
        "Starting training: %d epochs, %d train batches, %d val batches",
        epochs, len(train_loader), len(val_loader),
    )

    for epoch in range(1, epochs + 1):
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch, epochs
        )

        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        # Record history
        history["train_losses"].append(train_loss)
        history["train_accuracies"].append(train_acc)
        history["val_losses"].append(val_loss)
        history["val_accuracies"].append(val_acc)

        logger.info(
            "Epoch %d/%d — Train Loss: %.4f, Train Acc: %.2f%%, "
            "Val Loss: %.4f, Val Acc: %.2f%%",
            epoch, epochs, train_loss, train_acc, val_loss, val_acc,
        )

        # Checkpointing
        if checkpoint_dir is not None:
            checkpoint = {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "val_loss": val_loss,
                "val_accuracy": val_acc,
                "class_mapping": class_mapping,
            }

            # Save last model
            last_path = checkpoint_dir / f"{model_name}_last.pth"
            torch.save(checkpoint, last_path)

            # Save best model
            if val_acc > history["best_val_accuracy"]:
                history["best_val_accuracy"] = val_acc
                history["best_epoch"] = epoch
                best_path = checkpoint_dir / f"{model_name}_best.pth"
                torch.save(checkpoint, best_path)
                logger.info(
                    "New best model saved (Val Acc: %.2f%%) → %s",
                    val_acc, best_path,
                )

    elapsed = time.time() - start_time
    history["training_time"] = elapsed

    logger.info(
        "Training complete in %.1fs — Best Val Acc: %.2f%% (epoch %d)",
        elapsed, history["best_val_accuracy"], history["best_epoch"],
    )

    return history
