# Logistic Regression & Sigmoid Function

Exploratory mathematical analysis and early from-scratch implementation of Logistic Regression for binary classification, referencing Chapter 3 of *Machine Learning with PyTorch and Scikit-Learn* by Sebastian Raschka.

---

## Learning Objective

- Understand the transition from linear classification (Perceptron) to probabilistic classification (Logistic Regression).
- Explore the **Sigmoid (logistic) activation function** and its asymptotic properties.
- Analyze the convexity of the **Logistic Loss (Binary Cross-Entropy)** function compared to Mean Squared Error.
- Practice deriving and coding gradient descent optimization for logistic regression.

---

## Core Mathematical Concepts

### 1. The Logit and Sigmoid Function
While linear regression outputs unconstrained continuous values $z = \mathbf{w}^T \mathbf{x} + b$, logistic regression maps $z$ to a probability $p \in [0, 1]$ using the sigmoid function $\sigma(z)$:

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

Where:
- $\lim_{z \to \infty} \sigma(z) = 1$
- $\lim_{z \to -\infty} \sigma(z) = 0$
- $\sigma(0) = 0.5$ (the default decision threshold)

### 2. Logistic Cost Function (Log-Loss)
To maintain a convex optimization surface that avoids local minima, the cost function for a single training example is defined as:

$$L(\mathbf{w}, b) = \begin{cases} -\log(\sigma(z)) & \text{if } y = 1 \\ -\log(1 - \sigma(z)) & \text{if } y = 0 \end{cases}$$

Combined into a single equation:

$$L(\mathbf{w}, b) = -y \log(\sigma(z)) - (1 - y) \log(1 - \sigma(z))$$

This heavily penalizes confident but incorrect predictions (as $\sigma(z) \to 0$ when $y=1$, cost $\to \infty$).

---

## Files in this Directory

| File | Status | Description |
| :--- | :--- | :--- |
| [`sigmoid_analysis.ipynb`](file:///c:/Users/laksh/Downloads/journey_of_ML/fundamentals/logistic_regression/sigmoid_analysis.ipynb) | Complete | Visual analysis plotting $\sigma(z)$ across varying inputs and graphing the individual loss curves for $y=1$ and $y=0$ to demonstrate penalty behavior. |
| [`logistic_regression_scratch.ipynb`](file:///c:/Users/laksh/Downloads/journey_of_ML/fundamentals/logistic_regression/logistic_regression_scratch.ipynb) | *In Progress (WIP)* | Early draft implementation of batch gradient descent logistic regression (`LogisticRegressionGD`). Currently contains initial class architecture and dataset loading. |

---

## How to Explore

Open the visual analysis notebook to view the plots and mathematical breakdown:

```bash
jupyter notebook fundamentals/logistic_regression/sigmoid_analysis.ipynb
```
