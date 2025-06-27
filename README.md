# Lead Scoring Model using Logistic Regression

This repository contains a complete solution for building a lead scoring model using Logistic Regression. The objective is to identify high-potential leads that are more likely to convert, helping X Education improve their sales conversion rates.

## Problem Statement

X Education aims to increase its lead-to-sale conversion rate by building a machine learning model that predicts the probability of a lead being converted. The dataset includes around 9000 leads with information such as Lead Source, Website Visits, Last Activity, Occupation, etc.

## Methodology

The entire solution involves:

- Data cleaning and preprocessing
- Handling missing values and irrelevant features
- Encoding categorical variables
- Feature scaling
- Feature selection using RFE (Recursive Feature Elimination)
- Model building with `statsmodels` and `sklearn`
- Evaluation using metrics like accuracy, ROC-AUC, sensitivity, and specificity
- Threshold tuning for optimal probability cutoff
- Final testing and reporting

## Tools and Libraries

- Python
- Pandas, NumPy
- Seaborn, Matplotlib
- Scikit-learn
- Statsmodels

## Model Summary

- **Model**: Logistic Regression
- **Feature Selection**: Recursive Feature Elimination (RFE)
- **Scaling**: MinMaxScaler
- **Evaluation Metrics**:
  - Accuracy
  - Sensitivity & Specificity
  - ROC-AUC
  - Confusion Matrix

## Results

- Achieved balanced sensitivity and specificity at a custom cutoff of 0.42
- ROC-AUC Score: ~0.86 on training data
- Generalized well on the test data with good classification performance

## Visualization

- ROC Curve
- Cutoff vs Accuracy, Sensitivity, Specificity plot

## Learnings

- How to handle high-cardinality categorical data
- Use of statistical significance (p-values) and multicollinearity (VIF)
- Best practices in logistic regression modeling
- Data-driven threshold tuning for binary classification
