# Digit and Character Recognition System

A complete handwritten digit and character recognition system built from scratch using **PyTorch**. The project demonstrates the full machine learning pipeline from data loading through model training to real-time inference.

## Features

- ✅ MNIST digit dataset loading and visualization
- ✅ EMNIST character dataset preparation and documentation
- ✅ Configurable data pipeline with DataLoaders
- ✅ Image normalization and preprocessing
- ✅ Dataset analysis and visualization utilities
- ✅ Reproducible experiments with seed management
- ✅ Automatic GPU/MPS/CPU device detection
- ✅ MLP baseline classifier with training pipeline
- ✅ Training/validation loops with checkpointing
- ✅ CNN digit and character classifier
- 🔲 Full training pipeline with metrics (Stage 4)
- 🔲 Real-time drawing interface (Stage 5)

## Project Structure

```text
digit-character-recognition/
│
├── data/                        # Downloaded datasets (gitignored)
│   └── README.md                # Dataset documentation
│
├── models/                      # Neural network architectures
│   ├── __init__.py
│   ├── mlp.py                   # MLP baseline model
│   └── cnn.py                   # CNN classifier model
│
├── src/                         # Source code
│   ├── data/                    # Data loading modules
│   │   ├── mnist_loader.py      # MNIST dataset pipeline
│   │   └── emnist_loader.py     # EMNIST dataset pipeline
│   │
│   ├── training/                # Training modules (Stage 2+)
│   │   ├── train.py             # Generic training loop
│   │   └── validate.py          # Validation loop
│   │
│   ├── evaluation/              # Evaluation metrics (Stage 4+)
│   ├── inference/               # Prediction pipeline (Stage 5)
│   │
│   └── utils/                   # Utility modules
│       ├── visualization.py     # Plotting and visualization
│       ├── preprocessing.py     # Image preprocessing
│       └── seed.py              # Reproducibility utilities
│
├── app/                         # Drawing application (Stage 5)
├── tests/                       # Test suite
├── checkpoints/                 # Model checkpoints (Stage 2+)
├── results/
│   ├── figures/                 # Generated plots
│   └── metrics/                 # Experiment results
│
├── explore_data.py              # Stage 1: Data exploration script
├── train_mlp.py                 # Stage 2: MLP training script
├── train_cnn.py                 # Stage 3: CNN training script
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore rules
├── LICENSE                      # MIT License
└── README.md                    # This file
```

## Installation

```bash
# Clone the repository
git clone https://github.com/AakashBirla/Digit-and-Character-Recognition.git
cd Digit-and-Character-Recognition

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Datasets

### MNIST (Handwritten Digits)

| Property | Value |
|---|---|
| Images | 70,000 (60,000 train + 10,000 test) |
| Image size | 28×28 grayscale |
| Classes | 10 (digits 0–9) |
| Normalization | mean=0.1307, std=0.3081 |

### EMNIST Balanced (Handwritten Characters)

| Property | Value |
|---|---|
| Split | Balanced |
| Classes | 47 |
| Class breakdown | 10 digits + 26 uppercase + 11 lowercase |
| Image size | 28×28 grayscale |
| Orientation | Transposed (auto-corrected in loader) |

**EMNIST Balanced class mapping:**
- Classes 0–9: Digits `0`–`9`
- Classes 10–35: Uppercase `A`–`Z`
- Classes 36–46: Visually distinct lowercase: `a, b, d, e, f, g, h, n, q, r, t`

**Key differences from MNIST:**
- EMNIST images are transposed and require orientation correction
- EMNIST Balanced has 47 classes vs MNIST's 10
- The Balanced split merges visually ambiguous lowercase letters with their uppercase counterparts

## Stage 1: Data Pipeline

### How to Run

```bash
# Activate virtual environment
source .venv/bin/activate

# Run the data exploration script
python explore_data.py

# Run tests
python -m pytest tests/ -v
```

### What it does

1. Downloads the MNIST dataset automatically
2. Loads training and test sets with normalization
3. Creates DataLoaders with configurable batch sizes
4. Prints dataset statistics:
   - Image shape
   - Number of training/test samples
   - Number of classes
   - Class distribution
5. Generates visualization plots saved to `results/figures/`:
   - Sample images grid
   - Class distribution bar chart
   - Batch samples grid
   - Pixel intensity distribution histogram

### Results

Plots are saved to `results/figures/`:
- `mnist_sample_images.png` — Grid of sample digit images
- `mnist_class_distribution.png` — Class distribution bar chart
- `mnist_batch_samples.png` — Sample batch from DataLoader
- `mnist_pixel_distribution.png` — Pixel intensity histogram

## Stage 2: MLP Baseline Classifier

### MLP Architecture

```text
Input (28×28)
     ↓
Flatten (784)
     ↓
Linear(784, 256) → ReLU → Dropout(0.2)
     ↓
Linear(256, 128) → ReLU → Dropout(0.2)
     ↓
Linear(128, num_classes)
```

- **Parameters**: ~235K trainable
- **Loss**: CrossEntropyLoss
- **Optimizer**: Adam (lr=0.001)
- Supports configurable `num_classes` (10 for MNIST, 47 for EMNIST)

### How to Run

```bash
# Train MLP on MNIST (10 epochs)
python train_mlp.py --epochs 10 --batch-size 64 --lr 0.001
```

### Results (MNIST)

| Metric | Value |
|---|---|
| Best Val Accuracy | 98.02% |
| Best Epoch | 8 / 10 |
| Training Time | 48.1s |

### Limitations

The MLP is a **baseline** — it has inherent limitations for image data:
- Treats each pixel independently with no spatial awareness
- Loses 2D structure by flattening to a 1D vector
- Not translation-invariant
- More parameters than necessary

**CNNs (Stage 3) address these limitations** by using convolutional filters that exploit spatial structure, weight sharing, and local connectivity.

## Stage 3: CNN Digit and Character Classifier

### CNN Architecture

```text
Input (1×28×28)
     ↓
Conv2D (1 → 32, 3×3) → ReLU → MaxPool (2×2)
     ↓
Conv2D (32 → 64, 3×3) → ReLU → MaxPool (2×2)
     ↓
Flatten (3136)
     ↓
Linear(3136, 128) → ReLU → Dropout(0.3)
     ↓
Linear(128, num_classes)
```

- **Parameters**: ~421K trainable
- **Loss**: CrossEntropyLoss
- **Optimizer**: Adam (lr=0.001)
- Supports configurable `num_classes` (10 for MNIST, 47 for EMNIST)

### How to Run

```bash
# Train CNN on MNIST (5 epochs)
python train_cnn.py --dataset mnist --epochs 5 --batch-size 64 --lr 0.001

# Train CNN on EMNIST (5 epochs)
python train_cnn.py --dataset emnist --epochs 5 --batch-size 64 --lr 0.001
```

### Results

| Metric | MNIST | EMNIST (Balanced) |
|---|---|---|
| Best Val Accuracy | 99.25% | 87.41% |
| Best Epoch | 5 / 5 | 5 / 5 |
| Training Time | 23.9s | 52.6s |

### CNN Advantages over MLP
- Exploits 2D spatial structure via convolutional filters
- Weight sharing reduces parameters and improves generalization
- Max pooling provides translation invariance
- Hierarchical feature learning (edges -> shapes -> digits/chars)

## Project Evolution

| Stage | Description | Status |
|---|---|---|
| **Stage 1** | MNIST/EMNIST data pipeline and visualization | ✅ Complete |
| **Stage 2** | MLP baseline classifier | ✅ Complete |
| **Stage 3** | CNN digit and character classifier | ✅ Complete |
| **Stage 4** | Training pipeline and evaluation metrics | 🔲 Pending |
| **Stage 5** | Real-time drawing recognition app | 🔲 Pending |

## Technology Stack

- **Python** 3.12+
- **PyTorch** 2.14+ (with CUDA 13.0 support)
- **Torchvision** 0.29+
- **NumPy**, **Matplotlib**, **Pandas**
- **OpenCV** (headless)
- **Pillow**, **scikit-learn**, **tqdm**

## Future Improvements

- Data augmentation (rotation, scaling, elastic deformation)
- Better CNN architectures (ResNet, EfficientNet)
- Batch normalization and dropout
- Learning-rate scheduling
- Transfer learning
- Transformer-based OCR
- Word/sentence recognition
- ONNX export for production
- TensorRT inference optimization
- Web deployment (Flask/FastAPI + React)

## License

MIT License — see [LICENSE](LICENSE) for details.
