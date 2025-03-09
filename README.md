# Feedforward Neural Networks - Image Classification Using Keras and TensorFlow

## Overview
Consider two models:

1. A 2-layer feedforward neural network (i.e., 1 hidden layer) with:

   \[
   f(x, W_1, b_1, W_2, b_2) = W_2 \max(0, W_1x + b_1) + b_2
   \]

2. The same as above, but using **Leaky ReLU** as the activation function:

   \[
   f(x) =
   \begin{cases}
   x, & \text{if } x > 0 \\
   0.01 \cdot x, & \text{otherwise}
   \end{cases}
   \]

---

## Tasks

### a) Model Implementation
- Build the above classifiers using **Keras** and **TensorFlow**.
- Solve the classification problem for **MNIST** / **Fashion MNIST** datasets.

### b) Optimizer Choice
- Discuss how different optimizers (e.g., SGD, Adam, RMSprop) influence model performance.

### c) Hidden Units Analysis
- Investigate the effect of varying the number of hidden units:
  - What happens when the number of hidden units is much smaller?
  - What happens when the number of hidden units is much higher?

---

## Requirements
Ensure you have the following dependencies installed before running the code:

```bash
pip install numpy tensorflow keras matplotlib
