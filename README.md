# Telecom Customer Churn Prediction – End-to-End ML Project

An industry-grade, machine learning solution to predict customer churn in the telecom sector using a fully automated preprocessing pipeline and serialized model.

> 📌 End-to-End ML Pipeline | 🎯 Business-Oriented | ✅ Deployment Ready

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Problem Statement](#-problem-statement)
- [Dataset Information](#-dataset-information)
- [Tech Stack & Libraries](#-tech-stack--libraries)
- [Project Workflow](#-project-workflow)
- [Model Comparison](#-model-comparison)
- [Final Model Performance (Random Forest)](#-final-model-performance-random-forest)
- [Project Structure](#-project-structure)

---

## 📌 Overview

Customer churn is a major challenge in the telecom industry. Retaining customers is significantly cheaper than acquiring new ones. This project implements a robust machine learning pipeline to predict customer churn using demographic, behavioral, and account-based features.

> 🎯 Goal: Build a production-ready ML model that accurately predicts customer churn using automated preprocessing and powerful ML algorithms.

---

## Problem Statement

The objective is to predict whether a customer will churn or not, using structured data. Early churn prediction enables businesses to take preventive actions and retain customers.

---

## Dataset Information

- **Source**: [Telco Customer Churn - Kaggle](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Records**: 7,043 customers
- **Features**:
  - Demographics: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
  - Services: `PhoneService`, `InternetService`, `StreamingTV`, etc.
  - Account Info: `tenure`, `Contract`, `MonthlyCharges`, `TotalCharges`
- **Target Variable**: `Churn` (Binary: Yes/No)

---

## 🛠️ Tech Stack & Libraries

- **Language**: Python 3.10+
- **Libraries**: 
  - `pandas`, `numpy`, `seaborn`, `matplotlib`
  - `scikit-learn`
  - `cloudpickle` for model serialization

---

## 🔄 Project Workflow

1. **Data Cleaning**
   - Removed missing and inconsistent entries
   - Converted column types where necessary

2. **Exploratory Data Analysis (EDA)**
   - Visualized churn trends by contract, payment method, and demographics
   - Identified key churn indicators

3. **Feature Engineering**
   - Label encoding and one-hot encoding of categorical features
   - **Bucketizing** tenure and other continuous variables
   - Feature scaling for numerical columns

4. **Preprocessing Pipeline**
   - Built using `ColumnTransformer` and `Pipeline` from `scikit-learn`
   - Includes imputation, encoding, scaling, and bucketing steps
   - Fully automated for both training and inference

5. **Model Training**
   - Trained multiple ML models:
     - Logistic Regression
     - Random Forest
     - Bagging Classifier
     - Boosting (GradientBoosting, XGBoost, AdaBoost)
     - ExtraTrees Classifier
     - KNN
   - Hyperparameter tuning cross-validation using RandomizedSearchCV and optuna.

6. **Model Evaluation**
   - Accuracy, Precision, Recall, F1-score
   - **ROC AUC Score** and Confusion Matrix
   - Compared models using visual metrics

7. **Serialization**
   - Complete preprocessing + model pipeline saved as:
     ```
     model_pipeline.pkl
     ```

---

## 🔬 Model Comparison

Below is a comparative analysis of various machine learning models evaluated using cross-validation:

| Model              | Train Accuracy | CV Accuracy | Train F1 Score | CV F1 Score | Train ROC AUC | CV ROC AUC |
|--------------------|----------------|-------------|----------------|-------------|----------------|-------------|
| Logistic Regression | 0.81           | 0.81        | 0.61           | 0.60        | 0.86           | 0.85        |
| Decision Tree       | 0.84           | 0.78        | 0.68           | 0.56        | 0.91           | 0.79        |
| KNN                 | 0.81           | 0.79        | 0.63           | 0.59        | 0.87           | 0.83        |
| Random Forest       | 0.83           | 0.78        | 0.73           | 0.64        | 0.92           | 0.85        |
| Gradient Boosting   | 0.83           | 0.80        | 0.64           | 0.58        | 0.88           | 0.85        |
| XGBoost             | 0.83           | 0.81        | 0.63           | 0.59        | 0.88           | 0.85        |
| AdaBoost            | 0.81           | 0.81        | 0.60           | 0.60        | 0.85           | 0.85        |
| Extra Trees         | 0.80           | 0.76        | 0.69           | 0.64        | 0.90           | 0.85        |
| Bagging             | **0.99**       | 0.79        | **0.98**       | 0.56        | **1.00**       | 0.83        |

> ⚠️ **Note**: Although Bagging performed extremely well on training data, its generalization on unseen data was limited due to overfitting.

✅ *Final model was chosen based on balanced generalization performance, interpretability, and robustness. Then tuning choosen model again*


## 🎯 Final Model Performance (Random Forest)

The final model selected for this project is **Random Forest**, chosen for its balance between performance and interpretability, along with strong generalization capabilities on unseen data.

| Metric        | Precision | Recall | F1 Score | Accuracy |
|---------------|-----------|--------|----------|----------|
| **Train**     | 0.5582    | 0.8368 | 0.6697   | 0.7806   |
| **Test**      | 0.5059    | 0.7968 | 0.6189   | 0.7392   |

📌 **Key Insights:**
- High recall indicates the model is good at identifying churn customers (true positives).
- The slight drop in test metrics compared to training is expected and shows healthy generalization.
- F1 score provides a good balance between precision and recall, crucial for churn classification problems.

✅ **Why Random Forest?**
- Balanced performance on both training and test sets.
- Less prone to overfitting compared to other ensemble methods in this context.
- Handles non-linearity and feature interactions well.

---

## Project Structure

```bash
├── dataset/                   # Raw dataset (optional)
├── model                      # Serialized pipeline + model
├── churn_prediction           # EDA & model training notebook
└── README.md                  # Project documentation
