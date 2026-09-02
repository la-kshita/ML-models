# Datasets Directory

This directory contains the benchmark datasets used across the implementations, notebooks, and experiments in this repository.

---

## Dataset Overview & Catalog

| Dataset | File(s) | Source | Records / Shape | Target Variable | Used In |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Iris Flower** | `iris.data`, `iris.names`, `bezdekIris.data`, `Index` | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/53/iris) | 150 rows × 4 features | `class` (Setosa, Versicolor, Virginica) | `fundamentals/perceptron/perceptron_iris.ipynb`<br>`notebooks/sklearn_practice/01_sklearn_perceptron_iris.ipynb`<br>`notebooks/sklearn_practice/03_sklearn_model_comparison_iris.ipynb` |
| **Banknote Authentication** | `data_banknote_authentication.txt` | [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/267/banknote+authentication) | 1,372 rows × 4 features | `class` (0: authentic, 1: forged) | `fundamentals/perceptron/banknote_pca_ppn.ipynb`<br>`notebooks/sklearn_practice/02_sklearn_perceptron_banknote.ipynb` |
| **Social Network Ads** | `Social_Network_Ads.csv` | Kaggle Community | 400 rows × 5 columns | `Purchased` (0 or 1) | `notebooks/foundations/03_feature_scaling_standardization.ipynb` |
| **Titanic Passengers** | `Titanic-Dataset.csv` | [Kaggle Titanic Competition](https://www.kaggle.com/competitions/titanic) | 891 rows × 12 columns | `Survived` (0: no, 1: yes) | `notebooks/eda/titanic/01_titanic_eda.ipynb`<br>`notebooks/eda/titanic/02_titanic_missing_data.ipynb` |
| **50 Startups** | `50_Startups.csv` | Benchmark ML Dataset | 50 rows × 5 columns | `Profit` (continuous) | Reserved for multiple linear regression experiments |
| **Toy Preprocessing** | `Data.csv` | Educational Dataset | 10 rows × 4 columns | `Purchased` (categorical) | Basic missing value and encoding demonstrations |

---

## Dataset Details

### 1. Iris Flower (`iris.data`)
- **Features**: `sepal_length`, `sepal_width`, `petal_length`, `petal_width` (all in cm).
- **Target**: 3 classes of 50 instances each.
- **Notes**: In `fundamentals/perceptron/perceptron_iris.ipynb`, the first 100 samples are extracted to form a linearly separable binary classification problem (Iris-setosa vs. Iris-versicolor) using petal length and sepal length.

### 2. Banknote Authentication (`data_banknote_authentication.txt`)
- **Features**: Continuous values extracted from images via Wavelet Transform:
  1. Variance of Wavelet Transformed image
  2. Skewness of Wavelet Transformed image
  3. Curtosis of Wavelet Transformed image
  4. Entropy of image
- **Target**: Column index 4 (`0` = genuine, `1` = forged).
- **Notes**: Used to test Perceptron on a 4-dimensional feature space, accompanied by **PCA from scratch** to project the data into 2D for linear separability inspection.

### 3. Social Network Ads (`Social_Network_Ads.csv`)
- **Features**: `User ID`, `Gender`, `Age`, `EstimatedSalary`.
- **Target**: `Purchased` (0: didn't buy, 1: bought).
- **Notes**: Used to study the empirical effect of `StandardScaler` (zero mean, unit variance) on features with vastly different scales (`Age` vs `EstimatedSalary`).

### 4. Titanic Passengers (`Titanic-Dataset.csv`)
- **Features**: `PassengerId`, `Pclass`, `Name`, `Sex`, `Age`, `SibSp`, `Parch`, `Ticket`, `Fare`, `Cabin`, `Embarked`.
- **Target**: `Survived` (0 = died, 1 = survived).
- **Notes**: Used for detailed EDA (passenger distributions, zero-fare investigations, outlier detection) and missing data imputation strategies (mean, median, random sample, and scikit-learn `ColumnTransformer`).

---

## Version Control & Reproducibility Policy

- **Current Datasets**: All datasets in this directory are small, lightweight benchmark files (totaling under **130 KB** combined). They are committed directly to the repository so that every notebook and script can be cloned and run immediately without broken file references or external API tokens.
- **Future Large Datasets**: For datasets exceeding **1 MB** or compressed archives (`.zip`, `.tar.gz`), add the raw data file to `.gitignore` and provide a reproducible download script or Kaggle API instructions here.
