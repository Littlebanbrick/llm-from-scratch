# Lab 2: NumPy Fundamentals and Manual Neural Network Implementation

## 1. Objective

This lab aims to deepen your understanding of the core numerical operations and gradient-based learning algorithms that underpin modern deep learning frameworks. By completing this lab, you will:

- Master fundamental NumPy operations on multi-dimensional arrays.
- Implement forward and backward passes for linear and logistic regression from scratch.
- Build a single-hidden-layer neural network using only NumPy and manual gradient derivation.
- Compare your manually computed gradients with PyTorch's automatic differentiation to validate correctness.
- Develop an intuitive grasp of backpropagation and gradient descent without relying on high-level APIs.

## 2. Prerequisites

You are expected to have completed Lab 1 (MNIST classification with PyTorch). Familiarity with Python, basic linear algebra (matrix multiplication, transposition), and calculus (chain rule) is assumed.

## 3. Software Environment

- Python 3.8 or above
- NumPy (latest version)
- Matplotlib (for visualisation)
- PyTorch (optional, for gradient checking in Task 4)

All experiments must be run on CPU. Use a fixed random seed (`np.random.seed(42)`) for reproducibility.

## 4. Tasks

The lab is divided into four incremental tasks. Complete each task in order; each builds upon the previous.

### Task 1: NumPy Array Operations and Vectorisation

Familiarise yourself with NumPy's core data structure, the `ndarray`. Write a script that performs the following operations:

1. Create a 2D array `X` of shape (100, 5) filled with random numbers drawn from a standard normal distribution.
2. Compute the mean and standard deviation of each column (axis=0) and each row (axis=1).
3. Normalise the array so that each column has zero mean and unit variance (standardisation).
4. Compute the matrix product `X @ X.T` and `X.T @ X`. Explain the shape of each result.
5. Implement a function that computes the sigmoid activation `σ(z) = 1 / (1 + exp(-z))` in a numerically stable way (handle large positive/negative values to avoid overflow).
6. Create a binary label vector `y` of shape (100,) with values 0 or 1. Compute the binary cross-entropy loss for a given prediction vector `p` (also shape (100,)) using the formula: `L = - (y * log(p) + (1-y) * log(1-p))`. Use a small epsilon to avoid log(0).

*Deliverable*: A Python script `task1_numpy_basics.py` that executes the above steps and prints the results. Include comments explaining each operation.

### Task 2: Linear Regression with Gradient Descent

Implement a simple linear regression model with a single feature (or multiple features) using NumPy. Use a synthetic dataset:

- Generate 200 samples with one input feature `x` uniformly distributed in [-3, 3].
- Generate target `y = 2 * x + 1 + noise`, where noise is drawn from a normal distribution with standard deviation 0.5.
- Split the data into training (80%) and validation (20%) sets.

Implement the following:

- Define the model: `y_pred = w * x + b`.
- Define the mean squared error (MSE) loss function.
- Compute the gradients of the loss with respect to `w` and `b` analytically.
- Run mini-batch gradient descent for 1000 epochs with a batch size of 32. Use a fixed learning rate (start with 0.01 and adjust if necessary).
- Record the training and validation losses per epoch. Plot the loss curves.
- Report the final learned parameters `w` and `b` and compare them to the true values (2 and 1).

*Deliverable*: A script `task2_linear_regression.py` that runs the experiment and produces a plot of loss curves. Include a brief analysis in comments or a separate text file.

### Task 3: Logistic Regression for Binary Classification

Extend Task 2 to logistic regression for binary classification. Use a synthetic dataset that is not linearly separable (e.g., two interleaving half-moons or a circle dataset). You can generate such data using `sklearn.datasets.make_moons` or `make_circles` with 300 samples.

- Use only the first two features.
- Split into training and test sets (80/20).

Implement logistic regression without any hidden layers:

- Model: `z = X @ w + b`, `p = sigmoid(z)`.
- Loss: binary cross-entropy.
- Compute gradients analytically: `∂L/∂w = X.T @ (p - y) / m`, `∂L/∂b = mean(p - y)`.
- Train using mini-batch gradient descent (batch size 32) for a sufficient number of epochs. Plot training and validation loss and accuracy over epochs.
- Evaluate on the test set and report the final accuracy.
- Visualise the decision boundary (plot the data points and the contour of `p=0.5`).

*Deliverable*: A script `task3_logistic_regression.py` that includes data generation, training, evaluation, and decision boundary visualisation.

### Task 4: Single-Hidden-Layer Neural Network

Implement a neural network with one hidden layer (e.g., 10 neurons) and ReLU activation, with a sigmoid output for binary classification. Use the same dataset as in Task 3 (or a more challenging one, e.g., `make_circles` with noise).

- Architecture: `h = relu(X @ W1 + b1)`, `z = h @ W2 + b2`, `p = sigmoid(z)`.
- Initialise weights using He initialisation: `W1` with standard deviation `sqrt(2 / n_in)`, `W2` with `sqrt(1 / n_hidden)`.
- Implement forward and backward passes manually (do not use PyTorch autograd for this part).
  - You must derive and code the gradients for `W1`, `b1`, `W2`, `b2` using the chain rule.
  - Use the derivatives: `dL/dz = p - y`, `dL/dW2 = h.T @ (p - y) / m`, `dL/db2 = mean(p - y)`, etc.
  - For the hidden layer, you need to backpropagate through ReLU: `dL/dh = (p - y) @ W2.T`, then `dL/dz1 = dL/dh * (h > 0)` (elementwise), and finally `dL/dW1 = X.T @ dL/dz1 / m`, `dL/db1 = mean(dL/dz1, axis=0)`.
- Train with mini-batch gradient descent. Monitor loss and accuracy on validation set. Plot curves.
- After training, re-implement the exact same network architecture using PyTorch (with `nn.Linear`, `nn.ReLU`, `nn.Sigmoid`, and BCELoss). Train it with the same hyperparameters (learning rate, batch size, epochs) and using PyTorch's autograd. Compare the final test accuracies and the gradient values (at a chosen layer) between your manual implementation and PyTorch's to confirm correctness. Print the maximum relative difference.

*Deliverable*: 
- A script `task4_manual_nn.py` with the manual implementation and training.
- A script `task4_pytorch_nn.py` for the PyTorch version.
- A short report (in Markdown) that compares the gradients and final performance, and explains any discrepancies.

## 5. Submission Instructions

Submit a single compressed archive (`.zip` or `.tar.gz`) containing:

- All Python scripts (`task1_*.py`, `task2_*.py`, `task3_*.py`, `task4_*.py`).
- The loss/accuracy plots generated (PNG or PDF).
- A `report.md` file that briefly describes your approach for each task, any difficulties encountered, and the gradient comparison results for Task 4.
- Do not include the datasets (they are synthetic and can be generated).

## 6. Evaluation Criteria

- Correctness of NumPy operations and vectorisation (Task 1).
- Proper implementation of gradient descent and loss computation (Tasks 2–4).
- Correct derivation and coding of backpropagation (Task 4).
- Quality of visualisations and analysis.
- Clarity of comments and code structure.
- Comparison with PyTorch autograd and discussion of findings.

## 7. Hints and Notes

- For Task 4, the manual implementation may be slow; it is acceptable to train for a moderate number of epochs (e.g., 2000) and report results.
- When implementing ReLU backward, remember that the derivative is 1 for positive inputs and 0 otherwise.
- Use `np.clip` to avoid log(0) in cross-entropy loss.
- For numerical stability in sigmoid, handle large negative inputs by returning 0 and large positive by returning 1, or use the stable `expit` function from `scipy.special`.
- In Task 4, to compare gradients, you can capture the gradients after a single forward-backward pass on a small batch, print the maximum absolute difference between your manual gradients and PyTorch's `.grad` values. A difference of `1e-6` or less is acceptable.

## 8. Additional Resources

- NumPy documentation: https://numpy.org/doc/stable/
- "The Matrix Calculus You Need For Deep Learning" (useful for derivative derivations)
- Andrew Ng's lecture notes on backpropagation.
