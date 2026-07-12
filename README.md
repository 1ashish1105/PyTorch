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

# Dataset and DataLoader

`Dataset` and `DataLoader` are two of the most important abstractions in PyTorch. They separate **how data is stored** from **how data is fed to the model during training**.

---

## Dataset

The `Dataset` class acts as a **blueprint** for your data. It tells PyTorch:

- Where the data is stored.
- How to load the data.
- How to return a single sample when requested.

To create a custom dataset, you inherit from `torch.utils.data.Dataset` and implement three methods.

### 1. `__init__()`

This method loads or initializes the dataset.

```python
def __init__(self, features, labels):
    self.features = features
    self.labels = labels
```

---

### 2. `__len__()`

Returns the total number of samples in the dataset.

```python
def __len__(self):
    return len(self.features)
```

---

### 3. `__getitem__(index)`

Returns a single sample (features and corresponding label) at the specified index.

```python
def __getitem__(self, index):
    return self.features[index], self.labels[index]
```

---

### Dataset Workflow

```text
CSV File
      │
      ▼
Pandas DataFrame
      │
      ▼
Custom Dataset
      │
      ├── __init__()
      ├── __len__()
      └── __getitem__()
```

---

## DataLoader

The `DataLoader` wraps a `Dataset` and provides an efficient way to iterate through the data during training.

Instead of manually accessing samples one by one, the DataLoader automatically:

- Creates mini-batches
- Shuffles the data (optional)
- Loads data efficiently
- Supports parallel data loading using multiple workers

Example:

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)
```

---

## DataLoader Control Flow

At the beginning of every epoch:

1. Shuffle the dataset indices (if `shuffle=True`).
2. Divide the indices into batches.
3. Request each sample from the `Dataset` using `__getitem__()`.
4. Combine the samples into a batch using `collate_fn`.
5. Return the batch to the training loop.

---

## Data Flow

```text
Dataset
   │
   ▼
DataLoader
   │
   ├── Shuffle (Optional)
   ├── Create Batches
   ├── Load Samples
   └── Combine into Batch
   │
   ▼
Training Loop
   │
   ▼
Neural Network
```

---

## Example

```python
train_dataset = CustomDataset(X_train, y_train)

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True
)

for features, labels in train_loader:
    outputs = model(features)
```

---

## Why do we use Dataset and DataLoader?

Without `Dataset` and `DataLoader`, you would have to:

- Read the data manually.
- Select samples one by one.
- Create batches yourself.
- Shuffle the data manually.

Using these classes makes the training pipeline cleaner, faster, and easier to maintain.

---

## Key Takeaways

- **Dataset** defines **how data is loaded**.
- **DataLoader** defines **how data is fed to the model**.
- `Dataset` returns **one sample at a time**.
- `DataLoader` returns **one batch at a time**.
- Together, they make data handling efficient and scalable.





















   






    
