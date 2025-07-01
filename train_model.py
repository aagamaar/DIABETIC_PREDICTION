import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC # Support Vector Classifier
from sklearn.metrics import accuracy_score
import pickle
import os # To check if the file exists

# Get the directory of the current script
script_dir = os.path.dirname(__file__)
# Construct the full path to the dataset
data_path = os.path.join(script_dir, 'diabetes.csv')

# --- 1. Load the Dataset ---
try:
    diabetes_df = pd.read_csv(data_path)
except FileNotFoundError:
    print(f"Error: '{data_path}' not found. Make sure diabetes.csv is in the same directory as this script.")
    exit() # Exit if the file isn't found

# Separate features (X) and target (Y)
X = diabetes_df.drop(columns='Outcome', axis=1)
Y = diabetes_df['Outcome']

# --- 2. Split Data into Training and Test Sets ---
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)

print(f"X_train shape: {X_train.shape}")
print(f"X_test shape: {X_test.shape}")

# --- 3. Train the Machine Learning Model (Support Vector Classifier) ---
classifier = SVC(kernel='linear') # Using a linear kernel for simplicity
classifier.fit(X_train, Y_train)

# --- 4. Evaluate the Model (Optional, but good practice) ---
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print(f"Accuracy on training data: {training_data_accuracy}")

X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print(f"Accuracy on test data: {test_data_accuracy}")

# --- 5. Save the Trained Model using Pickle ---
model_file_name = 'trained_model.sav'
model_save_path = os.path.join(script_dir, model_file_name)
pickle.dump(classifier, open(model_save_path, 'wb'))
print(f"Model saved as {model_save_path}")

# --- 6. (Optional) Test Loading the Saved Model and Making a Prediction ---
loaded_model_test = pickle.load(open(model_save_path, 'rb'))

input_data = (1,189,60,23,846,30.1,0.398,59,1)
import numpy as np
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
prediction = loaded_model_test.predict(input_data_reshaped)

if (prediction[0] == 0):
    print('The person is not diabetic (test load)')
else:
    print('The person is diabetic (test load)')