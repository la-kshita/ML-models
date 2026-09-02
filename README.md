# Machine Learning Learning Repository

A personal, hands-on repository documenting my journey learning the foundations of Machine Learning — from mathematical formulations and from-scratch implementations to data preprocessing, exploratory data analysis, and model evaluations.

---

## About This Repository

This repository documents my hands-on learning journey in Machine Learning. As a second-year B.Tech Computer Science and Engineering student specializing in AI & ML, I created this space to practice writing algorithms from first principles, explore real-world datasets, and build intuition for how learning systems operate behind the scenes.

Rather than treating machine learning as a black box of library calls, this repository focuses on understanding the mechanics of algorithms, exploring optimization behavior, and understanding the practical impact of preprocessing and feature scaling.

---

## Goals

The primary objective of this repository is to develop a deep, practical understanding of machine learning by:
- **Implementing algorithms from scratch** using NumPy to understand internal weight updates, loss functions, and optimization mathematics.
- **Conducting empirical experiments** on convergence, decision boundaries, and linear separability.
- **Documenting exploratory data analysis (EDA)** and handling messy real-world data (missing values, categorical encoding, feature standardization).
- **Benchmarking scikit-learn models** against from-scratch baselines to see how production libraries implement these algorithms.

---

## Current Topics and Implementations

The following table reflects **only** concepts that are genuinely implemented, tested, or analyzed in this repository:

| Topic | Category | Type | Repository Location | Summary |
| :--- | :--- | :--- | :--- | :--- |
| **Perceptron Algorithm** | Supervised Learning | Implementation & Experiments | [`fundamentals/perceptron/`](fundamentals/perceptron/) | Complete single-layer Perceptron implemented in pure NumPy. Includes standalone class ([`perceptron.py`](fundamentals/perceptron/perceptron.py)), Iris binary classification notebook ([`perceptron_iris.ipynb`](fundamentals/perceptron/perceptron_iris.ipynb)), and decision region plots. |
| **Principal Component Analysis (PCA)** | Dimensionality Reduction | From-Scratch Implementation | [`fundamentals/perceptron/banknote_pca_ppn.ipynb`](fundamentals/perceptron/banknote_pca_ppn.ipynb) | Implemented PCA from mathematical first principles (data standardization, covariance matrix, eigenvalue/eigenvector decomposition, 2D projection) to visualize banknote linear separability. |
| **Sigmoid & Logistic Loss** | Mathematical Foundations | Visual Analysis & Experiment | [`fundamentals/logistic_regression/sigmoid_analysis.ipynb`](fundamentals/logistic_regression/sigmoid_analysis.ipynb) | Graphical analysis of the Sigmoid activation function and the convexity of the binary cross-entropy loss function ($L(w,b)$ for $y=1$ vs. $y=0$). |
| **Logistic Regression (GD)** | Supervised Learning | In-Progress Scratch Work | [`fundamentals/logistic_regression/logistic_regression_scratch.ipynb`](fundamentals/logistic_regression/logistic_regression_scratch.ipynb) | Early draft of batch gradient descent logistic regression (`LogisticRegressionGD`) implementing weight update mathematics. |
| **NumPy Foundations** | Scientific Computing | Learning Notebook | [`notebooks/foundations/01_numpy_fundamentals.ipynb`](notebooks/foundations/01_numpy_fundamentals.ipynb) | Core array operations, multi-dimensional slicing, axis reductions (`.reduce()`), and variance with Bessel's correction. |
| **Data Cleaning & Imputation** | Data Preprocessing | Learning Notebook | [`notebooks/foundations/02_data_cleaning_and_imputation.ipynb`](notebooks/foundations/02_data_cleaning_and_imputation.ipynb) | Missing value handling, `SimpleImputer`, categorical mapping, `LabelEncoder`, `OneHotEncoder`, and `ColumnTransformer`. |
| **Feature Scaling** | Preprocessing / Feature Eng. | Experiment Notebook | [`notebooks/foundations/03_feature_scaling_standardization.ipynb`](notebooks/foundations/03_feature_scaling_standardization.ipynb) | Practical demonstration of `StandardScaler` (z-score normalization) on the Social Network Ads dataset. |
| **Titanic EDA** | Exploratory Data Analysis | Project / Case Study | [`notebooks/eda/titanic/01_titanic_eda.ipynb`](notebooks/eda/titanic/01_titanic_eda.ipynb) | Distribution analysis (Age, Fare), zero-fare passenger analysis, and outlier inspection using boxplots. |
| **Titanic Missing Data** | Missing Value Imputation | Experiment Notebook | [`notebooks/eda/titanic/02_titanic_missing_data.ipynb`](notebooks/eda/titanic/02_titanic_missing_data.ipynb) | Comparing mean, median, and random sample imputation techniques using KDE distribution overlays and `ColumnTransformer`. |
| **Scikit-Learn Benchmarks** | Model Evaluation | Benchmark Experiments | [`notebooks/sklearn_practice/`](notebooks/sklearn_practice/) | Applying and comparing `Perceptron`, `LogisticRegression`, `SVC` (Support Vector Classifier), and `DecisionTreeClassifier` across Iris, Wine, and Banknote datasets. |

---

## Repository Structure

```text
ML-models/
│
├── fundamentals/                                # From-scratch algorithms & mathematical foundations
│   ├── perceptron/
│   │   ├── perceptron.py                        # Standalone Perceptron class with detailed docstrings
│   │   ├── perceptron_iris.ipynb                # Training on Iris, error curves & decision boundaries
│   │   ├── banknote_pca_ppn.ipynb               # Banknote: PCA from scratch + Perceptron training
│   │   ├── practice_scratch.py                  # Scratchpad for testing recall & manual implementation
│   │   └── README.md                            # Theory, math formulation, and usage guide
│   │
│   └── logistic_regression/
│       ├── sigmoid_analysis.ipynb               # Sigmoid curve and logistic loss visual experiments
│       ├── logistic_regression_scratch.ipynb    # In-progress gradient descent implementation
│       └── README.md                            # Mathematical notes on activation & loss functions
│
├── notebooks/
│   ├── foundations/                             # Core Python & Data Preprocessing practice
│   │   ├── 01_numpy_fundamentals.ipynb          # Arrays, slicing, axis reductions, Bessel's correction
│   │   ├── 02_data_cleaning_and_imputation.ipynb# Missing values, imputers, categorical encoding
│   │   └── 03_feature_scaling_standardization.ipynb # StandardScaler on Social Network Ads
│   │
│   ├── eda/                                     # Exploratory Data Analysis & Case Studies
│   │   └── titanic/
│   │       ├── 01_titanic_eda.ipynb             # Demographics, distributions & outlier detection
│   │       └── 02_titanic_missing_data.ipynb    # Imputation strategies (mean/median/random sample)
│   │
│   └── sklearn_practice/                        # Standard library benchmarks & model comparisons
│       ├── 01_sklearn_perceptron_iris.ipynb     # Scikit-Learn Perceptron baseline on Iris
│       ├── 02_sklearn_perceptron_banknote.ipynb # Scikit-Learn Perceptron on Banknote dataset
│       ├── 03_sklearn_model_comparison_iris.ipynb # Comparing Logistic Regression, SVM, Decision Tree
│       └── 04_sklearn_model_comparison_wine.ipynb # Multi-classifier benchmark on Wine dataset
│
├── datasets/                                    # Benchmark datasets (< 130 KB combined)
│   ├── 50_Startups.csv                          # Multi-variable regression dataset
│   ├── Data.csv                                 # Small 10-row dataset for preprocessing demos
│   ├── Social_Network_Ads.csv                   # Feature scaling practice data
│   ├── Titanic-Dataset.csv                      # Titanic passenger dataset
│   ├── data_banknote_authentication.txt         # Banknote authentication features
│   ├── iris.data                                # Iris benchmark dataset
│   ├── iris.names                               # Iris attribute documentation
│   ├── bezdekIris.data                          # Corrected Iris dataset
│   ├── Index                                    # UCI Iris index
│   └── README.md                                # Complete data catalog & provenance documentation
│
├── .gitignore                                   # Ignore rules (bytecode, virtualenvs, local references)
├── requirements.txt                             # Genuine dependencies actually used
└── README.md                                    # Repository documentation
```

---

## Learning Approach

My learning methodology is built around **active implementation**:
1. **Derive and code from scratch**: Before relying on library abstractions, I attempt to write the core algorithm or transformation (e.g., the Perceptron learning rule, covariance matrix and eigenvectors for PCA) using NumPy.
2. **Visualize and interpret**: Using Matplotlib and Seaborn to visualize decision surfaces, loss curves, and data distributions to verify mathematical intuition.
3. **Validate with standard libraries**: Compare custom implementations against Scikit-Learn to assess performance, edge cases, and standard API designs.
4. **Document honestly**: Keep detailed notes in code comments and Markdown cells explaining hyperparameters, why certain design choices were made, and where further study is needed.

---

## Resources

This repository follows concepts and exercises from the following primary textbooks:
- **Machine Learning with PyTorch and Scikit-Learn** — Sebastian Raschka, Yuxi Liu, Vahid Mirjalili
- **An Introduction to Statistical Learning** — Gareth James, Daniela Witten, Trevor Hastie, Robert Tibshirani

*Note: I am actively studying these texts and working through topics sequentially; this repository represents ongoing study rather than completion of all material.*

---

## Technologies Used

Only technologies actually imported and utilized in this repository:
- **Python 3.10+** (Core programming language)
- **NumPy** (Vectorized math, array operations, linear algebra for PCA, custom model implementations)
- **Pandas** (DataFrames, CSV loading, data cleaning, aggregation, exploratory analysis)
- **Matplotlib & Seaborn** (Data visualization, decision boundaries, KDE plots, loss surface graphs)
- **Scikit-Learn** (Preprocessing tools, imputation, pipeline benchmarks, standard model baselines)
- **Jupyter Notebooks** (Interactive experimentation and documentation environment)

---

## Current Progress

This repository is an **actively evolving learning record**:
- **Completed**: Single-layer Perceptron from scratch, PCA from scratch on Banknote data, Sigmoid & logistic loss mathematical analysis, Data cleaning and encoding workflows, Feature standardization experiments, Titanic exploratory analysis and imputation comparison, Scikit-Learn benchmark comparisons (Iris, Wine).
- **In Progress**: Completing the from-scratch Gradient Descent Logistic Regression implementation (`LogisticRegressionGD`), studying Adaline (Adaptive Linear Neuron), and expanding evaluation metrics (confusion matrices, precision/recall, ROC-AUC).
- **Upcoming**: Linear Regression from scratch, Support Vector Machine mathematics, and Decision Tree implementations.

---

## How to Use This Repository

### 1. Clone the Repository
```bash
git clone https://github.com/la-kshita/ML-models.git
cd ML-models
```

### 2. Set Up Environment & Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Explore
- **Run the custom Perceptron in Python**:
  ```bash
  python -c "from fundamentals.perceptron.perceptron import Perceptron; print('Perceptron class loaded successfully!')"
  ```
- **Launch Jupyter to view notebooks**:
  ```bash
  jupyter notebook
  ```
  Navigate to [`fundamentals/`](fundamentals/) for from-scratch algorithms, [`notebooks/eda/titanic/`](notebooks/eda/titanic/) for data analysis, or [`notebooks/sklearn_practice/`](notebooks/sklearn_practice/) for model comparisons.

---

## Author

**Lakshita Rawat**  
B.Tech in Computer Science and Engineering (Specialization in AI & ML)  
GitHub: [@la-kshita](https://github.com/la-kshita)
