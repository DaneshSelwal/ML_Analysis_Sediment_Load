# 🏞️ ML Analysis of Sediment Load: Advanced Uncertainty Quantification

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-XGBoost%20%7C%20LightGBM%20%7C%20CatBoost-orange?style=for-the-badge)
![Uncertainty Quantification](https://img.shields.io/badge/Uncertainty-Adaptive%20CP%20%7C%20NEXCP%20%7C%20Quantile-green?style=for-the-badge)
![Status](https://img.shields.io/badge/Status-Active-success?style=for-the-badge)

A comprehensive machine learning framework designed to predict **Suspended Sediment Load ($S_t$)** in hydrological systems. This project moves beyond standard point predictions by implementing rigorous **Uncertainty Quantification (UQ)** workflows—specifically tailored for time-series data using **Adaptive Conformal Prediction**—ensuring that environmental engineering decisions are backed by statistical confidence.

---

## 📑 Table of Contents (Navigation)

1. [📌 Project Overview](#-project-overview)
2. [📂 Repository Structure](#-repository-structure)
3. [📊 Dataset Details](#-dataset-details)
4. [🛠️ Workflow & Methodology](#-workflow--methodology)
    - [Phase 1: Hyperparameter Tuning](#phase-1-hyperparameter-tuning)
    - [Phase 2: Quantile Regression](#phase-2-quantile-regression)
    - [Phase 3: Probabilistic Distribution](#phase-3-probabilistic-distribution)
    - [Phase 4: Standard Conformal Predictions](#phase-4-standard-conformal-predictions)
    - [Phase 5: Adaptive & Non-Exchangeable CP](#phase-5-adaptive--non-exchangeable-cp)

---

## 📌 Project Overview

Accurate sediment load prediction is vital for reservoir management and hydraulic design. This repository provides a robust pipeline for analyzing hydrometric data using advanced Machine Learning regressors. Key features include:

* **Advanced Optimization**: Automated hyperparameter tuning using **Optuna** with various pruners and samplers.
* **Interval Estimation**: **Quantile Regression** to estimate the range of probable sediment loads.
* **Full Distribution Modeling**: Using **NGBoost** and **PGBM** to predict the entire probability distribution of the target.
* **Time-Series Uncertainty**: Specialized **NEXCP (Non-Exchangeable Conformal Prediction)** and **Adaptive CP** methods to handle data drift and temporal dependencies, ensuring valid coverage even during extreme events.

---

## 📂 Repository Structure

The project is organized into modular directories representing the analysis pipeline.

### 1. 🗂️ [Data](./Data)
Contains the hydrological time-series dataset.
* **[`train.csv`](./Data/train.csv)**: Historical data with lagged features used for model training.
* **[`test.csv`](./Data/test.csv)**: Held-out data for final model evaluation.

### 2. 🎛️ [Hyperparameter Tuning](./Hyperparameter%20Tuning)
Before training UQ models, base regressors are optimized to minimize error.
* **[`Optuna_autosampler_SEDIMENT.ipynb`](./Hyperparameter%20Tuning/Optuna_autosampler_SEDIMENT.ipynb)**: Implements Bayesian Optimization via Optuna to find optimal parameters.
* **`Models/`**: Stores serialized (`.pkl`) model files (e.g., `CatBoost_HyperbandPruner_BEST.pkl`).
* **`test_results.xlsx`**: Detailed comparative scores of different pruners and samplers.

### 3. 📉 [Quantile Regression](./Quantile%20Regression)
Focuses on predicting conditional quantiles (e.g., $Q_{0.05}$ and $Q_{0.95}$) rather than just the mean.
* **[`Quantile_Regression_SEDIMENT.ipynb`](./Quantile%20Regression/Quantile_Regression_SEDIMENT.ipynb)**: Trains models to minimize Pinball Loss.
* **`Results/`**: Contains prediction files for **XGBoost, LightGBM, CatBoost, HGBM, GPBoost,** and **PGBM**.

### 4. 📊 [Probabilistic Distribution](./Probabilistic%20Distribution)
Treats the target variable as a distribution to capture aleatoric uncertainty.
* **[`Probabilistic__Distribution_SEDIMENT.ipynb`](./Probabilistic%20Distribution/Probabilistic__Distribution_SEDIMENT.ipynb)**: Implements **NGBoost** and **PGBM** to output $\mu$ (mean) and $\sigma$ (standard deviation).
* **`Results/`**: Includes CRPS (Continuous Ranked Probability Score) metrics and Calibration plots.

### 5. 🛡️ [Conformal Prediction (Standard & Adaptive)](./Conformal_Predictions(NEXCP,AdaptiveCP,mfcs))
Applies rigorous statistical calibration. This module is split into standard methods and advanced time-series methods.
* **[`Conformal Predictions(MAPIE,PUNCC)_SEDIMENT.ipynb`](./Conformal%20Prediction(MAPIE,PUNC)/Conformal%20Predictions(MAPIE,PUNCC)_SEDIMENT.ipynb)**: Standard Split Conformal Prediction and CV+.
* **[`Conformal_Predictions(NEXCP, Adaptive CP, mfcs)_SEDIMENT.ipynb`](./Conformal_Predictions(NEXCP,AdaptiveCP,mfcs)/Conformal_Predictions(NEXCP,%20Adaptive%20CP,%20mfcs)_SEDIMENT.ipynb)**: Implements **NEXCP** and **Adaptive Conformal Prediction** to handle distribution shifts in sediment data.

---

## 📊 Dataset Details

The analysis is based on hydrometric time-series data containing current flow and historical lags:

| Feature | Description | Role |
| :--- | :--- | :--- |
| **Qt** | Discharge (Flow) at current time $t$ | Predictor |
| **Qt-1** | Discharge at previous time step $t-1$ | Predictor (Lag) |
| **St-1** | Sediment Load at previous time step $t-1$ | Predictor (Lag) |
| **St** | **Target Variable**: Sediment Load at time $t$ | **Target** |

---

## 🛠️ Workflow & Methodology

### Phase 1: Hyperparameter Tuning
We utilize **Optuna** to optimize regressors (XGBoost, CatBoost, etc.).
1.  **Search Space**: Defined for Learning Rate, Depth, Regularization, and Estimators.
2.  **Pruning**: Tested various pruners (Hyperband, Median, SuccessiveHalving) to efficiently discard unpromising trials.
3.  **Outcome**: The best models are saved in the `Models/` directory for subsequent UQ phases.

### Phase 2: Quantile Regression
Standard regression predicts the conditional mean $E[Y|X]$. Quantile regression predicts $Q_\tau(Y|X)$.
* **Application**: We predict the 5th and 95th percentiles to create a 90% prediction interval.
* **Metric**: Pinball Loss (Quantile Loss).

### Phase 3: Probabilistic Distribution
This approach assumes $Y|X \sim \mathcal{D}(\theta)$.
* **Models**: **NGBoost** (Natural Gradient Boosting) and **PGBM**.
* **Analysis**: Evaluates the probabilistic fit using **NLL** (Negative Log-Likelihood) and **PIT** (Probability Integral Transform) histograms to ensure the model is well-calibrated.

### Phase 4: Standard Conformal Predictions
Uses libraries like `MAPIE` and `PUNCC` for exchangeable data.
* **Methods**: Jackknife+, CV+, and Split Conformal.
* **Guarantee**: Provides marginal coverage guarantees (e.g., 90%) over the entire dataset.

### Phase 5: Adaptive & Non-Exchangeable CP
**Crucial for Sediment Data**: Sediment load data often violates the "exchangeability" assumption due to temporal dependencies.
* **NEXCP**: Weights recent observations higher to account for distribution drift.
* **Adaptive CP**: Dynamically adjusts the prediction interval width ($C_t$) based on recent errors.
* **Benefit**: Maintains valid coverage even during "bursty" periods like floods or seasonal shifts.

---
