# Module 5: Simple Neural Network (from scratch)

A 2-layer neural network built with only **NumPy** that learns the **XOR** pattern.

## Files

| File | Description |
|---|---|
| `Module_5_Neural_Network.ipynb` | Assignment notebook with blanks to fill in |
| `Module_5_Neural_Network_Solution.ipynb` | Completed solution |

## The Task

| Input 1 | Input 2 | Output |
|---|---|---|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

The third input column is always `1`. It acts as a **bias**.

## Network Architecture

```
3 inputs  ->  4 hidden nodes  ->  1 output
```

## Key Concepts

| Concept | Formula / Idea |
|---|---|
| Sigmoid activation | `σ(x) = 1 / (1 + e^-x)` |
| Sigmoid derivative | `σ'(a) = a · (1 - a)` |
| Forward propagation | `a1 = σ(X · w0)`, `a2 = σ(a1 · w1)` |
| Error | `error = a2 - y` |
| Backpropagation | `δ2 = error · σ'(a2)`, `δ1 = (δ2 · w1ᵀ) · σ'(a1)` |
| Gradient descent | `w1 -= a1ᵀ · δ2`, `w0 -= a0ᵀ · δ1` |
| Epoch | One forward pass + backprop + weight update (5000 total) |

## How It Works

**Guess -> measure the mistake -> find which weights caused it -> adjust them -> repeat.**

## Results

- Error drops from about **0.50** to about **0.014** after 5000 epochs.
- Final predictions: `[0.011, 0.986, 0.988, 0.016]`, which is very close to the target `[0, 1, 1, 0]`.

## Why a Hidden Layer?

XOR cannot be solved with a single straight line. The hidden layer lets the network combine inputs to learn non-linear patterns.

## Real-Life Uses

- **Spam filters:** sigmoid output as a probability
- **Fraud detection:** flags combinations of features (e.g. new device AND foreign location)
- **Image recognition:** hidden layers detect edges, then shapes, then faces

## Requirements

```
numpy
pandas
matplotlib
jupyter
```

## Run It

```bash
jupyter notebook
```

Open `Module_5_Neural_Network_Solution.ipynb` and run all cells.
