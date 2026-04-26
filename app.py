import streamlit as st
import joblib
import numpy as np

# Load saved model and scaler
model = joblib.load("model.joblib")
scaler = joblib.load("scaler.joblib")

st.title("📘 Student Performance Evaluator (ANN)")

attendance = st.number_input("Attendance", 0, 100)
assignment = st.number_input("Assignment", 0, 100)
quiz = st.number_input("Quiz", 0, 100)
mid = st.number_input("Mid", 0, 100)
study_hours = st.number_input("Study Hours", 0, 24)

if st.button("Predict"):
    data = np.array([[attendance, assignment, quiz, mid, study_hours]])
    data = scaler.transform(data)
    result = model.predict(data)

    if result[0] == 1:
        st.success("PASS 🟢")
    else:
        st.error("FAIL 🔴")