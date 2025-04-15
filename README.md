# 🧠 Telecom Customer Churn Prediction (ML Project)

Predicting customer churn in the telecom industry using machine learning. This project includes a full preprocessing pipeline along with the trained model, both serialized for deployment and reuse.

> 🚀 Status: Completed  
> 📦 Includes Preprocessing Pipeline + Trained ML Model (Saved as `.joblib`)  
> 🔍 Focus: Data Preprocessing, Feature Engineering, Model Training, Evaluation, and Deployment Readiness

---

## 📌 Overview

Customer churn prediction is crucial for telecom companies to identify customers likely to leave their services. This project utilizes machine learning to predict churn using customer demographic, account, and usage data. The final pipeline is serialized, enabling direct application on unseen data.

---

## 🧩 Problem Statement

Telecom companies face high losses due to customer churn. Identifying customers at risk of leaving allows businesses to take proactive retention measures. Our model addresses this issue by learning from historical data to make accurate churn predictions.

---

## 📂 Dataset Information

- **Source**: [Telco Customer Churn Dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- **Samples**: 7,043 customers
- **Features**:
  - Demographics: `gender`, `SeniorCitizen`, `Partner`, `Dependents`
  - Services: `PhoneService`, `InternetService`, `StreamingTV`, etc.
  - Account Info: `tenure`, `Contract`, `MonthlyCharges`, etc.
- **Target**: `Churn` (Yes/No)

---

## 🛠️ Tech Stack & Libraries

- Python 3.10+
- Pandas, NumPy
- Scikit-learn
- Matplotlib, Seaborn
- Joblib

---

## 🔄 Project Workflow

1. **Data Cleaning**
   - Removed duplicates and handled missing values
2. **Exploratory Data Analysis (EDA)**
   - Visualizations using Seaborn/Matplotlib
3. **Feature Engineering**
   - Label Encoding, One
