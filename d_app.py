import numpy as np
import pickle
import streamlit as st
import os

# Get the directory of the current script
script_dir = os.path.dirname(__file__)
# Construct the full path to the trained model
model_path = os.path.join(script_dir, 'trained_model.sav')

# --- 1. Loading the Saved Model ---
try:
    loaded_model = pickle.load(open(model_path, 'rb'))
except FileNotFoundError:
    st.error(f"Error: Model file '{model_path}' not found. Please run 'train_model.py' first.")
    st.stop() # Stop execution if model isn't found
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- 2. Create a Prediction Function ---
def diabetes_prediction(input_data):
    # Convert input_data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # Reshape the array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)

    prediction = loaded_model.predict(input_data_reshaped)

    if (prediction[0] == 0):
        return 'The person is not diabetic'
    else:
        return 'The person is diabetic'

# --- 3. Streamlit Web App Interface ---
def main():
    # Giving a title for the web page
    st.title('DIABETES PREDICTION')

    # Getting the input data from the user
    # Using st.number_input is generally safer for numerical inputs
    # You can add min_value, max_value, and value (default)
    pregnancies = st.number_input('Number of Pregnancies', min_value=0, max_value=20, value=0)
    glucose = st.number_input('Glucose Level (mg/dL)', min_value=0, max_value=200, value=190)
    blood_pressure = st.number_input('Blood Pressure value (mm Hg)', min_value=0, max_value=150, value=70)
    skin_thickness = st.number_input('Skin Thickness value (mm)', min_value=0, max_value=100, value=20)
    insulin = st.number_input('Insulin Level (mu U/ml)', min_value=0, max_value=900, value=80)
    bmi = st.number_input('BMI value', min_value=0.0, max_value=70.0, value=25.0)
    diabetes_pedigree_function = st.number_input('Diabetes Pedigree Function value', min_value=0.0, max_value=2.5, value=0.5)
    age = st.number_input('Age of the Person', min_value=0, max_value=120, value=30)

    # Code for Prediction
    diagnosis = ''

    # Creating a button for Prediction
    if st.button('Diabetes Test Result'):
        input_data = [
            pregnancies,
            glucose,
            blood_pressure,
            skin_thickness,
            insulin,
            bmi,
            diabetes_pedigree_function,
            age
        ]
        diagnosis = diabetes_prediction(input_data)

    st.success(diagnosis)

# This ensures the main() function runs only when the script is executed directly
if __name__ == '__main__':
    main()