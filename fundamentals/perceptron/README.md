# Perceptron Algorithm

A from-scratch implementation and empirical evaluation of the classical single-layer Perceptron algorithm (Rosenblatt, 1957), following concepts from *Machine Learning with PyTorch and Scikit-Learn* by Sebastian Raschka.

---

## Learning Objective

- Understand the mathematical foundations of biological-inspired artificial neurons.
- Implement the Perceptron learning rule from first principles using only **NumPy**.
- Analyze convergence behavior on linearly separable data (Iris) versus multi-dimensional non-linearly separable data (Banknote Authentication).
- Explore dimensionality reduction by implementing Principal Component Analysis (**PCA from scratch**) to visualize high-dimensional feature distributions.

---

## Core Concept & Mathematical Formulation

The Perceptron is a binary classification algorithm for linearly separable patterns.

### 1. Net Input Function
For an input vector $\mathbf{x} = [x_1, x_2, \dots, x_m]$ and weight vector $\mathbf{w} = [w_1, w_2, \dots, w_m]$ with bias $b$:

$$z = \mathbf{w}^T \mathbf{x} + b = \sum_{j=1}^{m} w_j x_j + b$$

### 2. Decision Function (Unit Step Function)
The activation function $\phi(z)$ outputs a discrete class label:

$$\phi(z) = \begin{cases} 1 & \text{if } z \ge 0 \\ 0 & \text{otherwise} \end{cases}$$

### 3. Weight & Bias Update Rule
Weights and bias are updated after each sample $(\mathbf{x}^{(i)}, y^{(i)})$ according to the error between the true target and prediction:

$$\Delta w_j = \eta \left(y^{(i)} - \hat{y}^{(i)}\right) x_j^{(i)}$$

$$\Delta b = \eta \left(y^{(i)} - \hat{y}^{(i)}\right)$$

Where:
- $\eta \in (0.0, 1.0]$ is the learning rate hyperparameter.
- $\hat{y}^{(i)} = \phi(z^{(i)})$ is the predicted class label.
- Updates occur **only** when misclassification happens ($y^{(i)} \neq \hat{y}^{(i)}$).

---

## Files in this Directory

| File | Description |
| :--- | :--- |
| [`perceptron.py`](file:///c:/Users/laksh/Downloads/journey_of_ML/fundamentals/perceptron/perceptron.py) | Standalone object-oriented implementation with detailed educational docstrings explaining weight initialization (`RandomState`), hyperparameters (`eta`, `n_iter`), and learned attributes (`w_`, `b_`, `errors_`). |
| [`perceptron_iris.ipynb`](file:///c:/Users/laksh/Downloads/journey_of_ML/fundamentals/perceptron/perceptron_iris.ipynb) | End-to-end evaluation on the Iris dataset (Iris-setosa vs. Iris-versicolor). Includes convergence plot (epochs vs. misclassification updates) and decision boundary visualization (`plot_decision_regions`). |
| [`banknote_pca_ppn.ipynb`](file:///c:/Users/laksh/Downloads/journey_of_ML/fundamentals/perceptron/banknote_pca_ppn.ipynb) | **PCA from Scratch** (standardization, covariance matrix, eigenvalue decomposition, top-2 eigenvector projection) followed by training custom Perceptron on 4 features with epoch convergence plots. |
| [`practice_scratch.py`](file:///c:/Users/laksh/Downloads/journey_of_ML/fundamentals/perceptron/practice_scratch.py) | An unpolished scratchpad used for testing recall and writing algorithms from memory during study sessions. |

---

## How to Run

### Standalone Script
You can import and use the class in your Python scripts:

```python
import numpy as np
from fundamentals.perceptron.perceptron import Perceptron

# Sample binary data
X = np.array([[1.0, 2.0], [2.0, 3.0], [3.0, 1.0], [4.0, 3.0]])
y = np.array([0, 0, 1, 1])

# Initialize and fit
ppn = Perceptron(eta=0.01, n_iter=10, random_state=1)
ppn.fit(X, y)

print("Learned Weights:", ppn.w_)
print("Learned Bias:", ppn.b_)
print("Predictions:", ppn.predict(X))
```

### Notebooks
Launch Jupyter to explore the visualizations and PCA experiments:

```bash
jupyter notebook fundamentals/perceptron/perceptron_iris.ipynb
jupyter notebook fundamentals/perceptron/banknote_pca_ppn.ipynb
```
