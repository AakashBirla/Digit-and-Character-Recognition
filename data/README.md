# Data Directory

This directory stores downloaded datasets. The datasets are downloaded automatically by torchvision.

## Datasets Used

### MNIST
- **Source**: Yann LeCun's MNIST database
- **Content**: 70,000 handwritten digit images (60,000 train + 10,000 test)
- **Image size**: 28×28 grayscale
- **Classes**: 10 (digits 0-9)
- **Downloaded to**: `data/MNIST/`

### EMNIST
- **Source**: Extended MNIST (Cohen et al., 2017)
- **Content**: Handwritten character images derived from NIST Special Database 19
- **Image size**: 28×28 grayscale
- **Split used**: Balanced (47 classes)
- **Downloaded to**: `data/EMNIST/`

## Note

These directories are excluded from git tracking via `.gitignore` since the datasets are large and can be re-downloaded automatically.
