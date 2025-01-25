# LoRA and SVD Experiments Repository

This repository contains two Jupyter notebooks demonstrating:
1. **Low-Rank Adaptation (LoRA)** for efficient fine-tuning of neural networks.
2. **Singular Value Decomposition (SVD)** for low-rank matrix approximation.

## Contents
- `main.ipynb`: Implements LoRA for MNIST classification using PyTorch.
- `svd.ipynb`: Demonstrates SVD-based matrix decomposition and reconstruction.

---

## 1. `main.ipynb`: LoRA for MNIST Classification

### Overview
- Trains a neural network on MNIST and applies LoRA to reduce trainable parameters during fine-tuning.
- **Key Features**:
  - Model architecture: 3 fully connected layers (28x28 → 1000 → 2000 → 10).
  - LoRA parameterization for weight matrices.
  - Freezes original weights and biases, updating only LoRA parameters.
  - Compares accuracy and parameter counts before/after LoRA.

### Key Results
- **Original Model**: 2,807,010 parameters.
- **LoRA-Enhanced Model**: Adds 6,794 parameters (0.242% increase).
- Wrongly Classified Samples For Digit 9:
  - **Baseline**: 137
  - **After LoRA Fine-Tuning**: 10 

### Usage
1. **Dependencies**: 
   ```bash
   pip install torch torchvision tqdm
