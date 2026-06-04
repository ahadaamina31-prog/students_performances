import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("student_model.pkl")

st.title("🎓 Student Performance Prediction System")
st.write("Predict whether a student will PASS or FAIL")

# --- Encoding maps (SAFE & CONSISTENT) ---
gender_map = {"female": 0, "male": 1}

race_map = {
    "group A": 0,
    "group B": 1,
    "group C": 2,
    "group D": 3,
    "group E": 4
}

parental_map = {
    "some high school": 0,
    "high school": 1,
    "some college": 2,
    "associate's degree": 3,
    "bachelor's degree": 4,
    "master's degree": 5
}

lunch_map = {"standard": 0, "free/reduced": 1}
test_map = {"none": 0, "completed": 1}

# --- Inputs ---
gender = st.selectbox("Gender", list(gender_map.keys()))
race = st.selectbox("Race/Ethnicity", list(race_map.keys()))
parental_edu = st.selectbox("Parental Level of Education", list(parental_map.keys()))
lunch = st.selectbox("Lunch", list(lunch_map.keys()))
test_prep = st.selectbox("Test Preparation Course", list(test_map.keys()))

reading_score = st.number_input("Reading Score", 0, 100, value=50)
writing_score = st.number_input("Writing Score", 0, 100, value=50)

# --- Prediction ---
if st.button("Predict Result"):

    input_data = np.array([[
        gender_map[gender],
        race_map[race],
        parental_map[parental_edu],
        lunch_map[lunch],
        test_map[test_prep],
        reading_score,
        writing_score
    ]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Student will PASS")
    else:
        st.error("❌ Student will FAIL")