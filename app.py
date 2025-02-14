from src.exception import CustomException
from src.logger import logging
from src.utils import load_object
import tensorflow as tf
import sys
import os
from src.component.data_transformation import DataTransformationConfig
from src.component.model_trainer import ModelTrainerConfig
import streamlit as st
import pandas as pd

# Initialize config
data_transformation_config = DataTransformationConfig()
model_trainer_config = ModelTrainerConfig()
try:
    # Check if preprocessor file exists
    if not os.path.exists(data_transformation_config.preprocessor_obj_file_path):
        raise FileNotFoundError(
            f"Preprocessor file not found: {data_transformation_config.preprocessor_obj_file_path}"
        )

    # Load preprocessor
    logging.info("Loading preprocessor...")
    preprocessor = load_object(data_transformation_config.preprocessor_obj_file_path)
    logging.info("Preprocessor loaded successfully.")

    # Check if model file exists
    if not os.path.exists(model_trainer_config.trained_model_file_path):
        raise FileNotFoundError(
            f"Trained model file not found: {model_trainer_config.trained_model_file_path}"
        )

    # Load model
    logging.info("Loading model...")
    model = tf.keras.models.load_model(model_trainer_config.trained_model_file_path)
    logging.info("Model loaded successfully.")

except Exception as e:
    logging.error(f"Error loading preprocessor or model: {str(e)}")
    raise CustomException(e, sys)

# streamlit app title
st.title("Customer Churn Prediction")

# User input
credit_score = st.number_input("Credit Score", min_value=0)
geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.slider("Age", min_value=18, max_value=92)
tenure = st.slider("Tenure", min_value=0, max_value=10)
balance = st.number_input("Balance", min_value=0)
num_of_products = st.slider("Number of Products", min_value=1, max_value=4)
has_credit_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", min_value=0)

# Prepare DataFrame
input_data = pd.DataFrame(
    {
        "CreditScore": [credit_score],
        "Geography": [geography],
        "Gender": [gender],
        "Age": [age],
        "Tenure": [tenure],
        "Balance": [balance],
        "NumOfProducts": [num_of_products],
        "HasCrCard": [has_credit_card],
        "IsActiveMember": [is_active_member],
        "EstimatedSalary": [estimated_salary],
    }
)
try:
    # transformed data
    input_data_transformed = preprocessor.transform(input_data)

    # Prediction Churn
    prediction = model.predict(input_data_transformed)
    prediction_proba = prediction[0][0]

    # Display prediction
    if prediction_proba < 0.5:
        st.write("The customer is not likely to churn.")
    else:
        st.write("The customer is likely to churn.")
except Exception as e:
    raise CustomException(e, sys)
