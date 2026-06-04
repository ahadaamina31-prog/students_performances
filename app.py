import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("student_model.pkl")

st.title("🎓 Student Performance Prediction System")
st.write("Predict whether a student will PASS or FAIL")

# Inputs (must match training features order)
gender = st.selectbox("Gender", ["female", "male"])

race = st.selectbox(
    "Race/Ethnicity",
    ["group A", "group B", "group C", "group D", "group E"]
)

parental_edu = st.selectbox(
    "Parental Level of Education",
    [
        "some high school",
        "high school",
        "some college",
        "associate's degree",
        "bachelor's degree",
        "master's degree"
    ]
)

lunch = st.selectbox("Lunch", ["standard", "free/reduced"])

test_prep = st.selectbox("Test Preparation Course", ["none", "completed"])

reading_score = st.number_input("Reading Score", min_value=0, max_value=100)
writing_score = st.number_input("Writing Score", min_value=0, max_value=100)

# Simple encoding (IMPORTANT: must match training LabelEncoder order)
def encode(value, options):
    return options.index(value)

if st.button("Predict Result"):

    input_data = np.array([[
        encode(gender, ["female", "male"]),
        encode(race, ["group A", "group B", "group C", "group D", "group E"]),
        encode(parental_edu, [
            "some high school",
            "high school",
            "some college",
            "associate's degree",
            "bachelor's degree",
            "master's degree"
        ]),
        encode(lunch, ["free/reduced", "standard"]),
        encode(test_prep, ["none", "completed"]),
        reading_score,
        writing_score
    ]])

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.success("✅ Student will PASS")
    else:
        st.error("❌ Student will FAIL")