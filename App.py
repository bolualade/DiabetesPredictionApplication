import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the model
model = XGBClassifier().fit(X_train, y_train)
dump(model, 'diabetes.joblib')

# Title
st.title("Diabetes Risk Prediction App")

# Sidebar for user input
st.sidebar.header("Patient Information")

def user_input_features():
    age = st.sidebar.slider("Age", 1, 100, 30)
    gender = st.sidebar.selectbox("Gender", ['Male', 'Female', 'Other'])
    hypertension = st.sidebar.selectbox("Hypertension", [0, 1])
    heart_disease = st.sidebar.selectbox("Heart Disease", [0, 1])
    smoking_history = st.sidebar.selectbox("Smoking History", 
                                           ['No Info', 'never', 'former', 'current', 'not current', 'ever'])
    bmi = st.sidebar.slider("BMI", 10.0, 50.0, 25.0)
    hba1c_level = st.sidebar.slider("HbA1c Level", 3.5, 9.0, 5.5)
    glucose = st.sidebar.slider("Blood Glucose Level", 80, 300, 120)

    data = {
        'age': age,
        'hypertension': hypertension,
        'heart_disease': heart_disease,
        'bmi': bmi,
        'HbA1c_level': hba1c_level,
        'blood_glucose_level': glucose,
        'gender_Male': 1 if gender == 'Male' else 0,
        'gender_Other': 1 if gender == 'Other' else 0,
        'smoking_history_current': 1 if smoking_history == 'current' else 0,
        'smoking_history_ever': 1 if smoking_history == 'ever' else 0,
        'smoking_history_former': 1 if smoking_history == 'former' else 0,
        'smoking_history_never': 1 if smoking_history == 'never' else 0,
        'smoking_history_not current': 1 if smoking_history == 'not current' else 0,
        'bmi_category_Normal': 1 if 18.5 <= bmi < 25 else 0,
        'bmi_category_Overweight': 1 if 25 <= bmi < 30 else 0,
        'bmi_category_Obese': 1 if bmi >= 30 else 0
    }

    return pd.DataFrame([data])

input_df = user_input_features()

# Display input
st.subheader("Patient Data Summary")
st.write(input_df)

# Predict
prediction = model.predict(input_df)[0]
probability = model.predict_proba(input_df)[0][1]

st.subheader("Prediction Result")
st.write("Diabetes Risk:", "**Positive**" if prediction else "**Negative**")
st.write(f"Probability of having diabetes: **{probability:.2%}**")