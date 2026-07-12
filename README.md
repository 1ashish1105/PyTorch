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


## Parallel Data Loading

Loading data from disk can become a bottleneck during training, especially for large datasets. While the model is processing one batch, the CPU can prepare the next batch in parallel.

PyTorch's `DataLoader` supports parallel data loading using the `num_workers` parameter.

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)
```

Here,

- **batch_size=32** → Each batch contains 32 samples.
- **shuffle=True** → Shuffles the dataset at the beginning of every epoch.
- **num_workers=4** → Four CPU worker processes load and prepare batches in parallel.

---

## Why use Multiple Workers?

Suppose your model trains on the GPU.

If there is only one worker (`num_workers=0`), the workflow looks like:

```text
Load Batch 1
      │
      ▼
Train Batch 1
      │
      ▼
Load Batch 2
      │
      ▼
Train Batch 2
      │
      ▼
Load Batch 3
      │
      ▼
Train Batch 3
```

Here, the GPU often waits while the CPU loads data.

---

With multiple workers:

```text
Worker 1 ──► Batch 1
Worker 2 ──► Batch 2
Worker 3 ──► Batch 3
Worker 4 ──► Batch 4
                 │
                 ▼
            DataLoader
                 │
                 ▼
          Training Loop
```

While the model is training on **Batch 1**, the workers are already preparing **Batch 2**, **Batch 3**, and **Batch 4**.

This keeps the GPU busy and reduces idle time.

---

## Data Loading Pipeline

```text
Dataset
   │
   ▼
Worker 1 ─┐
Worker 2 ─┼──► DataLoader ───► Model
Worker 3 ─┤
Worker 4 ─┘
```

Each worker independently calls the `Dataset.__getitem__()` method to fetch samples and prepares batches concurrently.

---

## Choosing `num_workers`

| num_workers | Description |
|-------------|-------------|
| 0 | No parallel loading (default) |
| 2 | Two worker processes |
| 4 | Four worker processes |
| 8 | Eight worker processes |

Generally,

- Small datasets → `num_workers = 0 or 2`
- Medium datasets → `num_workers = 4`
- Large datasets → `num_workers = 8` (depending on CPU cores)

There is no universal best value—it depends on your hardware and dataset.

---

## Complete DataLoader Flow

```text
Dataset
   │
   ▼
Shuffle Indices
   │
   ▼
Split into Mini-Batches
   │
   ▼
Multiple Workers Load Data
   │
   ▼
Collate Samples into Batch
   │
   ▼
Return Batch to Training Loop
   │
   ▼
Neural Network
```

---

## Key Takeaways

- **Dataset** defines how to access the data.
- **DataLoader** creates mini-batches from the dataset.
- **shuffle=True** randomizes the order of samples each epoch.
- **batch_size** controls the number of samples per batch.
- **num_workers** enables parallel data loading, reducing data loading time and improving GPU utilization.


# Samplers in PyTorch

## What is a Sampler?

A **Sampler** determines the order in which samples are selected from a dataset.

Instead of fetching the data itself, a sampler simply decides **which indices should be returned** to the `DataLoader`.

The DataLoader then uses those indices to retrieve the actual samples from the Dataset using the `__getitem__()` method.

---

## How DataLoader Uses a Sampler

Suppose our dataset contains **10 samples**.

```text
Dataset

Index:
[0] [1] [2] [3] [4] [5] [6] [7] [8] [9]
```

The sampler decides the order in which these indices are accessed.

For example,

```text
Random Sampler

[4] [6] [1] [3] [7] [8] [2] [0] [9] [5]
```

The DataLoader will now call

```python
dataset[4]
dataset[6]
dataset[1]
dataset[3]
...
```

instead of

```python
dataset[0]
dataset[1]
dataset[2]
...
```

---

# Types of Samplers

PyTorch provides several built-in samplers.

---

## 1. SequentialSampler

Returns the samples in sequential order.

```text
0 → 1 → 2 → 3 → 4 → ...
```

Example:

```python
from torch.utils.data import DataLoader

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=False
)
```

Output indices

```text
Batch 1 : 0 1 2 3

Batch 2 : 4 5 6 7

Batch 3 : 8 9
```

This is the default sampler when

```python
shuffle=False
```

---

## 2. RandomSampler

Returns the samples in random order.

Example:

```python
loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True
)
```

Possible output

```text
Batch 1 : 7 3 1 5

Batch 2 : 0 6 9 2

Batch 3 : 8 4
```

Every epoch the order changes.

This is the default sampler when

```python
shuffle=True
```

---

## 3. SubsetRandomSampler

Used when you want to train on only a subset of the dataset.

Example

```python
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler

indices = [0, 5, 8, 11, 20]

sampler = SubsetRandomSampler(indices)

loader = DataLoader(
    dataset,
    batch_size=2,
    sampler=sampler
)
```

Only the specified indices will be sampled.

---

## 4. WeightedRandomSampler

Useful for **imbalanced datasets**.

Suppose your dataset is

```text
Cats : 900

Dogs : 100
```

The model may become biased toward cats.

WeightedRandomSampler assigns a higher sampling probability to the minority class.

Example

```python
from torch.utils.data import WeightedRandomSampler

weights = [0.1, 0.9, ...]

sampler = WeightedRandomSampler(
    weights,
    num_samples=len(weights),
    replacement=True
)
```

This helps create more balanced batches.

---

# DataLoader Pipeline

```text
Dataset
   │
   ▼
Sampler
   │
   ▼
Indices

0 5 3 7 2 1 ...

   │
   ▼
Dataset.__getitem__()

   │
   ▼
DataLoader

   │
   ▼
Mini Batch

   │
   ▼
Training Loop
```

---

# Relationship Between Dataset, Sampler and DataLoader

```text
Dataset
│
├── __len__()
│
└── __getitem__()

        ▲
        │
Sampler chooses indices

        │
        ▼

DataLoader
│
├── Creates batches
├── Shuffles data (optional)
├── Uses multiple workers
└── Returns batches to the model
```

---

# Key Points

- Dataset stores the data.
- Sampler decides **which sample to pick next**.
- DataLoader groups samples into mini-batches.
- `shuffle=True` internally uses a **RandomSampler**.
- `shuffle=False` internally uses a **SequentialSampler**.

Dataset and DataLoader
│
├── Dataset
│     ├── __init__()
│     ├── __len__()
│     └── __getitem__()
│
├── Samplers
│     ├── SequentialSampler
│     ├── RandomSampler
│     ├── SubsetRandomSampler
│     └── WeightedRandomSampler
│
├── BatchSampler
│
├── Parallel Data Loading (num_workers)
│
├── collate_fn
│
├── pin_memory
│
└── persistent_workers














   






    
