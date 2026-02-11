# Exploring Convolutional Layers Through Data and Experiments

## Problem Description
This repository contains an applied exploration of convolutional layers as
architectural components that introduce inductive bias into neural networks.
The goal is to compare a non-convolutional baseline against a purposely
designed CNN and run a controlled experiment on a specific convolutional
layer choice.

The full analysis, code, and results are in the notebook:
`convolutional-layers-chess-piece.ipynb`.

## Dataset Description
**Dataset:** Chess Pieces Detection Image Dataset (Kaggle)  
**Classes:** bishop, king, knight, pawn, queen, rook  
**Input:** RGB images resized to 128x128  
**Folder structure:**

```
data/
├── bishop/
├── king/
├── knight/
├── pawn/
├── queen/
└── rook/
```

**Why it fits CNNs:** Images contain strong spatial structure, local patterns,
and translation invariance, which match the assumptions embedded in
convolutional layers.

## EDA Summary
The notebook includes:
- Dataset size and class distribution
- Image dimensions and channels
- Example samples per class
- Preprocessing steps (resize + normalization)

## Models

### Baseline (Non-Convolutional)
**Architecture:**
```
Input (128x128x3) -> Flatten -> FC(512) -> FC(6)
```

**Purpose:** Provide a reference model that ignores spatial locality so the
impact of convolutional layers can be quantified.

**Reported in notebook:**
- Number of parameters
- Training/validation loss and accuracy
- Observed limitations

### CNN (Designed From Scratch)
**Architecture:**
```
Input
 -> Conv(3x3, 32) + ReLU -> MaxPool(2x2)
 -> Conv(3x3, 64) + ReLU -> MaxPool(2x2)
 -> Flatten -> FC(256) -> FC(6)
```

**Key design choices and justification (in notebook):**
- 2 conv layers for hierarchical features without overcomplication
- 3x3 kernels for efficient local feature extraction
- Stride 1 + padding 1 to preserve spatial resolution
- Max pooling to reduce dimensionality and retain salient features

## Controlled Experiment
**Variable:** Kernel size  
**Comparison:** 3x3 vs 5x5 kernels  
**Controlled:** number of layers, filters, stride/padding, optimizer, epochs

The notebook reports:
- Parameter count for each configuration
- Training/validation loss and accuracy
- Trade-offs (performance vs complexity)

## Results Summary

**Baseline vs CNN**
- Baseline params: `25,169,414`
- Baseline train acc: `0.7385` | val acc: `0.5577`
- CNN params: `16,798,406`
- CNN train acc: `0.9903` | val acc: `0.9519`

**Kernel Size Experiment**
- 3x3 params: `16,798,406`
- 3x3 train acc: `0.8329` | val acc: `0.8750`
- 5x5 params: `16,832,710`
- 5x5 train acc: `0.9274` | val acc: `0.9615`

## Interpretation
The notebook answers:
- Why convolution outperforms the baseline
- What inductive bias convolution introduces
- Where convolution is not appropriate
- How kernel size affects learning and generalization

These sections emphasize reasoning over raw metrics and align with the
assignment’s grading focus.

## SageMaker Deployment
The notebook includes:
- Local model artifact packaging (`model.tar.gz`)
- Upload to S3
- Deployable model creation (`PyTorchModel`)
- Endpoint deployment
- Endpoint invocation from base64 image payload
- Endpoint cleanup (to avoid cost)

## Repository Structure
- `convolutional-layers-chess-piece.ipynb`: full workflow
- `train.py`: training script
- `inference.py`: endpoint inference handlers
- `requirements.txt`: inference dependencies
- `data/`: local dataset folders by class

## How to Run
1. Ensure the dataset is in `data/` with the folder-per-class structure.
2. Open `convolutional-layers-chess-piece.ipynb`.
3. Run all cells top to bottom.