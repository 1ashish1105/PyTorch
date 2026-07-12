# PyTorch Notes 

This repository contains my learning notes on **PyTorch**, covering both theoretical concepts and practical implementation.

## Table of Contents

1. Introduction to PyTorch
2. Tensors
3. Autograd
4. torch.nn Module
5. Building Neural Networks
6. Dataset and DataLoader
7. Training Loop
8. Model Evaluation
9. Saving and Loading Models

---

# 1. Introduction to PyTorch

PyTorch is an open-source deep learning framework developed by Facebook (Meta).

It is mainly used for:

- Deep Learning
- Computer Vision
- Natural Language Processing
- Reinforcement Learning
- Research and Production

---

# 2. Tensors

## What are Tensors?

A tensor is a multidimensional array optimized for mathematical operations and GPU computation.

It is the fundamental data structure used in PyTorch.

### Types of Tensors

### 0-D Tensor (Scalar)

A single value.

Example

```python
5
```

Applications

- Temperature
- Age
- Loss value

---

### 1-D Tensor (Vector)

A list of values.

Example

```python
[1,2,3,4,5]
```

Applications

- Word embeddings (NLP)
- Feature vectors

---

### 2-D Tensor (Matrix)

Rows and columns.

Example

```text
[
 [1,2],
 [3,4]
]
```

Applications

- Grayscale Images
- Tabular Data

---

### 3-D Tensor

Stack of matrices.

Applications

- RGB Images

Shape

```
Height × Width × Channels
```

---

### 4-D Tensor

Batch of RGB images.

Shape

```
Batch × Height × Width × Channels
```

Example

```
32 × 224 × 224 × 3
```

---

### 5-D Tensor

Sequence of RGB images.

Applications

- Videos

Shape

```
Batch × Frames × Height × Width × Channels
```

---

## Why do we use tensors?

- Efficient mathematical operations
- GPU acceleration
- Store images, text, videos
- Build deep learning models

---

# 3. Autograd

## What is Autograd?

Autograd is PyTorch's automatic differentiation engine.

It automatically computes gradients during backpropagation.

Without Autograd, we would need to manually compute derivatives for every parameter in a neural network, which becomes impractical for deep models.

### Why is Autograd needed?

Training a neural network involves:

```
Forward Pass
      ↓
Calculate Loss
      ↓
Compute Gradients
      ↓
Update Weights
```

Autograd automatically performs the gradient computation.

---

# 4. torch.nn Module

## What is torch.nn?

The `torch.nn` module provides all the building blocks required to build neural networks.

Instead of implementing layers and activation functions from scratch, PyTorch provides ready-to-use implementations.

---

## Components of torch.nn

### 1. Layers

Common layers include

- nn.Linear
- nn.Conv2d
- nn.LSTM
- nn.GRU
- nn.Embedding

---

### 2. Activation Functions

Activation functions introduce non-linearity.

Common activation functions

- ReLU
- Sigmoid
- Tanh
- Softmax
- LeakyReLU

---

### 3. Loss Functions

Loss functions measure how far predictions are from the actual values.

Common loss functions

- nn.CrossEntropyLoss
- nn.MSELoss
- nn.BCELoss
- nn.NLLLoss

---

### 4. Container Modules

PyTorch provides modules to organize layers.

Example

```python
nn.Sequential(
    nn.Linear(784,128),
    nn.ReLU(),
    nn.Linear(128,10)
)
```

---

### 5. Regularization Layers

To reduce overfitting:

- nn.Dropout
- nn.BatchNorm1d
- nn.BatchNorm2d

---

# What's Next?

After understanding the above concepts, we'll move on to:

- Building Custom Datasets
- DataLoader
- Training Loop
- Optimizers
- Model Evaluation
- Saving Models






















   






    
