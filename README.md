# Bank Churn Prediction

This repository contains the code for predicting customer churn in a bank using various machine learning models. The data used for training and testing the models is stored in `Bank_Churn.csv`.

---

## Table of Contents

- [Overview](#overview)
- [Problem Statement](#problem-statement)
- [Project Objective](#project-objective)
- [Dataset](#dataset)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Data Preprocessing](#data-preprocessing)
- [Model Training](#model-training)
- [Model Evaluation](#model-evaluation)
- [Deployment Process](#deployment-process)
- [Conclusion](#conclusion)
- [Acknowledgements](#acknowledgements)

---

## Overview

The goal of this project is to predict if a bank customer will churn (exit) using various machine learning techniques. The dataset includes customer information such as credit score, geography, gender, age, balance, and other relevant features.

---
  
## Problem Statement

The bank is experiencing a significant churn rate among its customers, which is impacting its revenue and overall business performance. Identifying the key factors that contribute to customer churn and developing a predictive model to forecast which customers are likely to exit can help the bank proactively address these issues. The goal of this project is to analyze customer data and build a machine learning model that can accurately predict customer churn, enabling the bank to implement targeted retention strategies and improve customer satisfaction.

---
  
## Project Objective

The objective of this project is to **develop and deploy a predictive machine learning model** that accurately identifies bank customers who are at high risk of churning. By leveraging historical customer data, the model will uncover key factors influencing customer attrition and generate actionable insights. This will enable the bank to:

* **Predict** which customers are likely to exit in the near future.
* **Understand** the main drivers behind customer churn through feature analysis.
* **Support** the development of targeted retention strategies aimed at reducing churn rates and enhancing customer loyalty.
* **Improve** overall business performance by minimizing revenue loss associated with customer attrition.

---


## Dataset

The dataset `Bank_Churn.csv` consists of the following columns:

- `CustomerId`: Unique identifier for the customer
- `Surname`: Customer's surname
- `CreditScore`: Customer's credit score
- `Geography`: Customer's geography (country)
- `Gender`: Customer's gender
- `Age`: Customer's age
- `Tenure`: Number of years the customer has been with the bank
- `Balance`: Customer's account balance
- `NumOfProducts`: Number of bank products the customer is using
- `HasCrCard`: Whether the customer has a credit card (1 = Yes, 0 = No)
- `IsActiveMember`: Whether the customer is an active member (1 = Yes, 0 = No)
- `EstimatedSalary`: Customer's estimated salary
- `Exited`: Whether the customer has exited the bank (1 = Yes, 0 = No)

---

## Exploratory Data Analysis (EDA)

We performed some EDA by plotting the distributions of numerical variables and count plots for categorical variables:

- `CreditScore`, `Age`, `Balance`, `EstimatedSalary`, and `Tenure` distributions using histograms
- `Geography`, `Gender`, `NumOfProducts`, `HasCrCard`, `IsActiveMember`, and `Exited` distributions using count plots
- Bivariate relationships between `Age` & `CreditScore` vs `Exited` using scatter plots
- Correlation heatmap for numerical variables

---

## Data Preprocessing

Data preprocessing steps were also performed and these included:

1. Encoding categorical variables (`Geography` and `Gender`) using Label Encoding
2. Splitting the dataset into features (`x`) and target variable (`y`), we also dropped the customer id and surmane columns as they were irrelevant to our model training.
3. Performing train-test split
4. Feature scaling using Standard Scaler
5. Balancing the feature and target variable for training (`x_train`, `y_train`) using SMOTE (Synthetic Minority Over-sampling Technique).

---

## Model Training

Multiple models were trained and compared for performance:

1. **Logistic Regression**
              Accuracy: 0.7032
              Classification Report:
              precision,  recall,  f1-score

           0       0.90      0.71      0.79
           1       0.37      0.68      0.48
   
   
2. **Support Vector Machine (SVC)**
              Accuracy: 0.7884
              Classification Report:
              precision,   recall,  f1-score

           0       0.93      0.80      0.86
           1       0.48      0.74      0.58
   

3. **K-Nearest Neighbors (KNN)**
              Accuracy: 0.742
              Classification Report:
              precision,    recall,  f1-score

           0       0.90      0.76      0.83
           1       0.41      0.67      0.51
 

4. **Naive Bayes**
              Accuracy: 0.7432
              Classification Report:
              precision,    recall,  f1-score

           0       0.91      0.75      0.82
           1       0.42      0.71      0.52


5. **Random Forest**
              Accuracy: 0.838
              Classification Report:
              precision,   recall,  f1-score

           0       0.90      0.90      0.90
           1       0.59      0.58      0.59


6. **XGBoost**
              Accuracy: 0.8372
              Classification Report:
              precision,    recall,  f1-score

           0       0.91      0.88      0.90
           1       0.58      0.67      0.62


The XGBoost model was chosen based on the following reasons:

 - **Better recall for class 1**: Although the accuracy is similar for both models, the recall for class 1 (which represents the positive class, typically the minority class in many datasets) is higher in XGBoost (0.67 vs. 0.58 in Random Forest). This indicates that XGBoost is better at identifying true positives for class 1, making it more effective since our priority is to reduce false negatives for this class.

 - **Higher F1-Score for Class 1**: The F1-score for class 1 is also better with XGBoost (0.62 vs. 0.59). Since the F1-score balances precision and recall, this suggests that XGBoost offers a more balanced performance for this class, which is important given that we need to account for both false positives and false negatives.


## Model Evaluation

The models were evaluated based on:

1. Accuracy Score
2. Classification Report
3. Confusion Matrix
4. ROC Curve and AUC Score (for XGBoost)

---

## Saving the Models

The XGBoost model was chosen after a proper cross validation was done.
The following components were saved for future use:

1. Trained XGBoost model (`model.pkl`)
2. Scaler (`scaler.pkl`)
3. Label encoder (`label_encoder.pkl`)

Make sure to have the following dependencies installed and ready for the deployment process listed in requirements.txt:
`gradio`
`joblib`
`matplotlib`
`seaborn`
`numpy`
`pandas`
`scikit-learn`
`xgboost`

---

## Deployment Process
This project uses Gradio to create an interactive web interface and is deployed on Hugging Face Spaces. Below are the steps taken to achieve this deployment:

---

**Prerequisites**  
Ensure you have the following files from your trained machine learning project:

- model.pkl (Trained Machine Learning Model)
- scaler.pkl (Feature Scaler)
- label_encoder.pkl (Label Encoder)
- requirements.txt (python dependencies)

---

**Building the Gradio Interface**  
Create an Application Script

Create a Python script (e.g., `deploy.py`) that includes code to load your model, scaler, and label encoder, and defines the Gradio interface for user interaction.

---

**Deploying on Hugging Face Spaces**  

**Create a New Space**  
Navigate to Hugging Face Spaces and create a new Space. Select Gradio as the SDK for your project.

**Add Files to Your Space**  
Upload the `deploy.py`, `model.pkl`, `requirements.txt`, `scaler.pkl`, and `label_encoder.pkl` files to your Space.

**Configuration**  
Ensure that the `deploy.py` file is set as the main entry point for the application.

---

**Deployment**  
Once the files are added and configured, your Space will automatically build and deploy your Gradio application. The deployment status can be tracked on the Hugging Face Spaces interface.

---

**Accessing the Application**  
After successful deployment, you can access your Gradio interface through the link provided by Hugging Face Spaces. Users can interact with the interface to make predictions using the deployed machine learning model.
You can access the web application [HERE](https://huggingface.co/spaces/phenomkay/Bank_Churn)

---

## Conclusion

This project demonstrates how to preprocess data, train multiple machine learning models, and evaluate their performance for predicting customer churn in a bank. The XGBoost model achieved the highest accuracy and AUC score.

## Acknowledgements

- [Pandas Documentation](https://pandas.pydata.org/pandas-docs/stable/)
- [Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [Seaborn Documentation](https://seaborn.pydata.org/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/en/latest/)
