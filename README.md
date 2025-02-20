# 🧠 ANN Customer Churn Classification End-to-End Deployment Project
For more details, check out the [project repository on GitHub](https://github.com/dannyyqyq/ANN_Classification_churn).

## 🚀 Web Application
Experience the Customer Churn Prediction live!  
[Live Demo: Customer Churn Prediction](https://ann-churn-prediction.streamlit.app/)

## 📌 Project Overview
This project uses Artificial Neural Networks (ANN) to predict customer churn for a bank, helping businesses retain customers by identifying at-risk individuals. It leverages a dataset of customer records from a bank, processed with TensorFlow/Keras for classification. The project covers:

- **Data Ingestion**: Loading and preparing customer data from CSV files for analysis.
- **Data Transformation**: Preprocessing and encoding categorical data for model training.
- **Model Training**: Training an ANN to classify customers as likely to churn or not.
- **Prediction Pipeline**: Applying the trained model to predict churn for new customer data.
- **Web Application**: A Streamlit-based interface for users to input customer data and get churn predictions.
- **Deployment**: Deployed on Streamlit Cloud for live access.

## 🛠 Tools and Technologies Used

### 🚀 Deployment
- **Streamlit**: 
  - Provides an interactive web interface for real-time churn predictions.

### 📊 Machine Learning
- **Classification Model**: 
  - Artificial Neural Network (ANN) built with TensorFlow/Keras, including multiple dense layers, batch normalization, and dropout for regularization.
- **Feature Engineering**: 
  - StandardScaler for numerical features, OneHotEncoder/FunctionTransformer for categorical features. The model’s performance is detailed in the repository’s documentation or logs.
- **Model Evaluation**:

  | Metric              | Value    |
  | ------------------- | -------- |
  | Accuracy            | 0.8687   |
  | Precision (Churn=1) | 0.7513   |
  | Recall (Churn=1)    | 0.4863   |
  | F1-Score (Churn=1)  | 0.5904   |

  *Note: Values may vary slightly based on dataset splits and hyperparameters.*
