import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn
from sklearn.linear_model import LogisticRegression


# Load model, scaler and encoder
model = joblib.load(r"C:\Users\MY LAPTOP\Desktop\Vs Code Intro\logistic_model.pkl")
scaler = joblib.load(r"C:\Users\MY LAPTOP\Desktop\Vs Code Intro\scaler.pkl")
encoder = joblib.load(r"C:\Users\MY LAPTOP\Desktop\Vs Code Intro\OneHotEncoder.pkl")


st.title("Loan Payback Predictor App")
st.write("Enter borrower details to predict probability of default:")

# Inputs
credit_policy = st.selectbox("Credit Policy", [0,1])
purpose = st.selectbox("Loan Purpose", ["credit_card", "debt_consolidation", "educational","major_purchase", "small_business", "all_other"])
installment = st.number_input("Installment Amount", 0.0)
log_annual_inc = st.number_input("Log Annual Income", 0.0)
dti = st.number_input("Debt-to-Income Ratio", 0.0)
fico = st.number_input("FICO Score", 0)
delinq_2yrs = st.number_input("Delinquent past 2 yrs", 0)

# Create dataframe
input_df = pd.DataFrame([[credit_policy, installment, log_annual_inc,dti, fico, delinq_2yrs]],
                        columns=['credit.policy','installment','log.annual.inc','dti','fico','delinq.2yrs'])

# Manual one-hot encode purpose
purpose_df = pd.DataFrame({'purpose':[purpose]})
purpose_encoded = encoder.transform(purpose_df)
purpose_cols = encoder.get_feature_names_out(['purpose'])
input_df[purpose_cols] = purpose_encoded

# Scale numeric columns
scaled_cols = ['installment','log.annual.inc','dti','fico','delinq.2yrs']
input_df[scaled_cols] = scaler.transform(input_df[scaled_cols])

# Prediction
if st.button("Validate & Predict"):
    try:
        prob = model.predict(input_df)[0]
        prediction = 'Loan is risky' if prob == 1 else 'Loan is safe'
        st.success(f"Prediction: **{prediction}**")
    except Exception as e:
        st.error("An error occurred during prediction. Check your inputs.")
        st.write(e)