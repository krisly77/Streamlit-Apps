import streamlit as st
import pandas as pd
import numpy as np
import pickle
import sklearn
from sklearn.linear_model import LogisticRegression

# Loading the Model
with open('/Users/MY LAPTOP/Desktop/Vs Code Intro/SL/Streamlit Checkpoint 2.pkl', 'rb') as file:
    saved_objects = pickle.load(file)

logreg = saved_objects["LRmodel"]  # trained model
encoders = saved_objects["encoders"]  # dictionary of LabelEncoders

st.title("Bank Account Ownership Prediction App")
st.write("Fill in the details below and click **Validate & Predict**.")


# Input Fields

country = st.selectbox("Country", encoders["country"].classes_)

location_type = st.selectbox("Location Type", encoders["location_type"].classes_)

cellphone_access = st.selectbox("Cellphone Access", encoders["cellphone_access"].classes_)

household_size = st.number_input("Household Size", min_value=1, max_value=20, value=3)

age_of_respondent = st.number_input("Age of Respondent", min_value=15, max_value=100, value=30)

gender = st.selectbox("Gender", encoders["gender_of_respondent"].classes_)

relationship = st.selectbox("Relationship With Head", encoders["relationship_with_head"].classes_)

marital_status = st.selectbox("Marital Status", encoders["marital_status"].classes_)

education_level = st.selectbox("Education Level", encoders["education_level"].classes_)

job_type = st.selectbox("Job Type", encoders["job_type"].classes_)

# Encode inputs using LabelEncoders

encoded_features = [
    encoders["country"].transform([country])[0],
    2018,  # fixed year
    encoders["location_type"].transform([location_type])[0],
    encoders["cellphone_access"].transform([cellphone_access])[0],
    household_size,
    age_of_respondent,
    encoders["gender_of_respondent"].transform([gender])[0],
    encoders["relationship_with_head"].transform([relationship])[0],
    encoders["marital_status"].transform([marital_status])[0],
    encoders["education_level"].transform([education_level])[0],
    encoders["job_type"].transform([job_type])[0],
]

input_array = np.array(encoded_features).reshape(1, -1)

# Predict Button

if st.button("Validate & Predict"):
    try:
        raw_pred = logreg.predict(input_array)[0]
        prediction = 'Yes, owns a bank account' if raw_pred == 1 else 'No, does not own a bank account'
        st.success(f"Prediction: **{prediction}**")
    except Exception as e:
        st.error("An error occurred during prediction. Check your inputs or encoders.")
        st.write(e)