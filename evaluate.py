#!/usr/bin/env python3
"""Stage 4: Evaluate trained models.

Loads a saved model checkpoint, runs inference on the test set,
and generates evaluation metrics including a classification report
and confusion matrix.

Usage:
    python evaluate.py --model cnn --dataset mnist --checkpoint checkpoints/mnist_cnn_best.pth
"""

import argparse
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch

from models.cnn import CNN
from models.mlp import MLP
from src.data.emnist_loader import create_emnist_dataloaders
from src.data.mnist_loader import create_mnist_dataloaders
from src.evaluation.metrics import (
    generate_classification_report,
    get_predictions,
    plot_confusion_matrix,
)
from src.utils.seed import get_device, set_seed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
METRICS_DIR = RESULTS_DIR / "metrics"


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained model")
    parser.add_argument("--model", type=str, choices=["mlp", "cnn"], required=True, help="Model architecture")
    parser.add_argument("--dataset", type=str, choices=["mnist", "emnist"], required=True, help="Dataset to evaluate on")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size for evaluation")
    args = parser.parse_args()

    print()
    print("=" * 60)
    print(f"  STAGE 4: Evaluation ({args.model.upper()} on {args.dataset.upper()})")
    print("=" * 60)
    print()

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        sys.exit(1)

    set_seed(42)
    device = get_device()
    print(f"Device: {device}")
    print(f"Loading checkpoint: {checkpoint_path}")
    print()

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    class_mapping = checkpoint.get("class_mapping")
    
    if class_mapping is None:
        logger.error("Checkpoint does not contain 'class_mapping'")
        sys.exit(1)
        
    num_classes = len(class_mapping)

    # Convert keys to int (JSON/save might have converted them to string)
    class_mapping = {int(k): str(v) for k, v in class_mapping.items()}

    # Initialize model
    if args.model == "mlp":
        model = MLP(num_classes=num_classes)
    else:
        model = CNN(num_classes=num_classes)
        
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    print(f"Model loaded. Best validation accuracy during training: {checkpoint.get('val_accuracy', 0):.2f}%")
    print()

    # Load data
    print("Loading test data...")
    if args.dataset == "mnist":
        _, test_loader = create_mnist_dataloaders(batch_size=args.batch_size, num_workers=2)
    else:
        _, test_loader = create_emnist_dataloaders(split="balanced", batch_size=args.batch_size, num_workers=2)

    # Evaluate
    print("Running inference on test set...")
    true_labels, predicted_labels = get_predictions(model, test_loader, device)

    # Generate Metrics
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = METRICS_DIR / f"{args.model}_{args.dataset}_classification_report.txt"
    
    print("\nClassification Report:\n")
    report = generate_classification_report(
        true_labels, predicted_labels, class_mapping, save_path=report_path
    )
    print(report)

    # Confusion Matrix
    cm_path = FIGURES_DIR / f"{args.model}_{args.dataset}_confusion_matrix.png"
    plot_confusion_matrix(
        true_labels,
        predicted_labels,
        class_mapping,
        title=f"{args.model.upper()} Confusion Matrix ({args.dataset.upper()})",
        save_path=cm_path,
    )

    print("=" * 60)
    print(f"  Evaluation complete.")
    print(f"  Report saved to: {report_path}")
    print(f"  Confusion matrix saved to: {cm_path}")
    print("=" * 60)
    print()


if __name__ == "__main__":
    main()
