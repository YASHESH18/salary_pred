
import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Load the trained model
with open('random_forest_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the label encoders
with open('label_encoders.pkl', 'rb') as file:
    label_encoders = pickle.load(file)

# Set page config
st.set_page_config(page_title="Salary Prediction App", layout="centered")

st.title('Salary Prediction for Data Professionals')
st.write('Enter the details below to predict the salary.')

# Input fields
age = st.slider('Age', 18, 65, 30)
gender_options = ['Male', 'Female', 'Other']
gender_selected = st.selectbox('Gender', gender_options)
education_options = ['High School', 'Bachelor's Degree', 'Master's Degree', 'PhD']
education_selected = st.selectbox('Education Level', education_options)
job_title = st.text_input('Job Title (e.g., Data Scientist, Software Engineer)', 'Data Scientist')
years_of_experience = st.slider('Years of Experience', 0.0, 30.0, 5.0, 0.5)

# Prepare input for prediction
if st.button('Predict Salary'):
    try:
        gender_encoded = label_encoders['Gender'].transform([gender_selected])[0]
    except ValueError:
        st.warning(f"Gender '{gender_selected}' not seen during training. Using default value.")
        gender_encoded = 0 # Default to 0 or handle as appropriate

    try:
        education_encoded = label_encoders['Education Level'].transform([education_selected])[0]
    except ValueError:
        st.warning(f"Education Level '{education_selected}' not seen during training. Using default value.")
        education_encoded = 0 # Default to 0 or handle as appropriate

    # For Job Title, we need to handle potential new job titles not in training data
    # A simple approach is to map unseen job titles to a default or a numerically close value.
    # A more advanced approach would use embeddings or a robust 'unknown' category.
    # For now, if a job title is not found, we'll assign it a default value (e.g., 0).
    if job_title in label_encoders['Job Title'].classes_:
        job_title_encoded = label_encoders['Job Title'].transform([job_title])[0]
    else:
        st.warning(f"Job Title '{job_title}' not seen during training. Assigning a default value.")
        job_title_encoded = 0

    input_data = pd.DataFrame([[age, gender_encoded, education_encoded, job_title_encoded, years_of_experience]],
                              columns=['Age', 'Gender', 'Education Level', 'Job Title', 'Years of Experience'])

    prediction = model.predict(input_data)[0]
    st.success(f'Predicted Salary: ${prediction:,.2f}')
