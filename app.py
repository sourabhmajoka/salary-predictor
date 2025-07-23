import streamlit as st
import pandas as pd
import joblib

# Load the trained model
model = joblib.load('salary_prediction_model.pkl')

#load the encoders
label_encoders = joblib.load('label_encoders.pkl')

st.set_page_config(
    page_title="Employee Salary Predictor",
    page_icon="💼",
    layout="centered"
)

# Title
st.title("Employee Salary Predictor")
st.write("This app predicts whether an employee earns >50K or ≤50K per year.")

# Sidebar inputs
st.sidebar.header("Input Employee Details")

#age
age = st.sidebar.slider("Age", 18, 75, 30)

#gender
gender = st.sidebar.radio("Select Gender: ", label_encoders['gender'].classes_)
gender_encoded = label_encoders['gender'].transform([gender])[0]

#workclass
custom_workclass_labels = {
    'Private Sector': 'Private',
    'Self Employed (Incorporated)': 'Self-emp-inc',
    'Self Employed (Not Incorporated)': 'Self-emp-not-inc',
    'Center-Government': 'Federal-gov',
    'Local-Government': 'Local-gov',
    'State-Government': 'State-gov',
    'Other': 'Other'
}
workclass = st.sidebar.selectbox("Workclass", list(custom_workclass_labels.keys()))
workclass_original = custom_workclass_labels[workclass]
workclass_encoded = label_encoders['workclass'].transform([workclass_original])[0]

#occupation
job_role = st.sidebar.selectbox("Select Job Role", label_encoders['occupation'].classes_)
job_role_encoded = label_encoders['occupation'].transform([job_role])[0]

#hours-per-week
hours = st.sidebar.slider("Job Hours-Per-Week", 30, 70, 40)

#country
country = st.sidebar.selectbox("Native Country", label_encoders['native-country'].classes_)
country_encoded = label_encoders['native-country'].transform([country])[0]

#marital-status
marital = st.sidebar.selectbox("Marital Status", label_encoders['marital-status'].classes_)
marital_encoded = label_encoders['marital-status'].transform([marital])[0]

#education
education = st.sidebar.selectbox("Select Education", label_encoders['education'].classes_)
education_encoded = label_encoders['education'].transform([education])[0]

#net-capital
capital = st.sidebar.number_input("Net Capital (Capital Gain - Capital Loss)", value=0, step=1)

# Prepare input data
input_data = pd.DataFrame({
    'age': [age],
    'workclass': [workclass_encoded],
    'education': [education_encoded],
    'marital-status': [marital_encoded],
    'occupation': [job_role_encoded],
    'gender': [gender_encoded],
    'net-capital': [capital],
    'hours-per-week': [hours],
    'native-country': [country_encoded]
})

st.write("### 🔎 Input Data")
st.write(input_data)

# Predict
if st.button("Predict Salary Class"):
    try:
        prediction = model.predict(input_data)
        result = ">50K" if prediction[0] == 1 else "<=50K"
        st.subheader(f"Predicted Salary Range: {result}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")