# 💳 Credit Default Risk Prediction Pipeline

An advanced machine learning classification pipeline to forecast client loan default risks using the Kaggle Home Credit Default Risk dataset and a high-performing gradient-boosting ensemble. 

Developed for an in-class data science competition held in the GCI World 2026 course. 

---

## 📌 Project Overview

Predicting credit default risk is a critical challenge in the financial sector, balancing risk management with financial inclusion. This project implements an end-to-end machine learning classification pipeline designed to process complex financial datasets, engineer high-signal features, and accurately predict the probability of loan default. 

The final solution leverages a robust **gradient-boosting ensemble** combining **CatBoost, XGBoost, and LightGBM**, achieving a **ROC-AUC score of 0.763**.

---

## ⚙️ Dataset & Preprocessing

* **Source:** Kaggle Home Credit Default Risk Dataset (with select columns excluded for the competition).
* **Missing Value Handling:** Imputed missing numerical and categorical values using a strategic combination of column means and zeros.
* **Categorical Encoding:** Applied **Label Encoding**, **One-Hot Encoding**, and **Count Encoding** to transform categorical variables into high-utility numerical features.
* **Target Encoding (Leakage Prevention):** Implemented strict **out-of-fold target encoding** calculated independently within each cross-validation fold to completely prevent data leakage.
* **Feature Engineering:** Created novel interaction and combined features to capture complex financial behaviors and risk indicators.

---

## 🚀 Modeling & Validation Methodology

* **Cross-Validation:** Utilized **Stratified 5-Fold Cross-Validation** to maintain class balance across training and validation splits and ensure stable generalization.
* **Gradient-Boosting Ensemble:** Integrated three state-of-the-art boosting frameworks (**CatBoost**, **XGBoost**, and **LightGBM**) to capture non-linear relationships and maximize predictive performance.
* **Hyperparameter Tuning:** Utilized **Optuna** for automated hyperparameter tuning across ensemble estimators to maximize predictive stability and performance.

---

## 🛠️ Tech Stack

* **Language:** Python (Jupyter Notebook)
* **Machine Learning & Boosting:** CatBoost, XGBoost, LightGBM, scikit-learn
* **Data Manipulation & Analysis:** pandas, NumPy

---

## ⚙️ Installation

```
pip install numpy pandas scikit-learn lightgbm xgboost catboost
```

---

## 📂 Project Structure

```
Credit-Default-Risk-Prediction-Model/
│
├── input/
│   ├── sample_submission.csv
│   ├── test.csv
│   └── train.csv
├── output/
│   └── submission.csv
├── tutorial.ipynb
│
└── README.md
```

---

## 📊 Results & Performance
* **Primary Metric:** ROC-AUC
* **Best Model:** Weighted Ensemble (CatBoost + XGBoost + LightGBM)
* **Final Score:** **0.76 ROC-AUC**
