# 🩺 Diabetes Prediction using Machine Learning (PIMA Dataset)

This project is a complete end-to-end Machine Learning solution that predicts whether a person is diabetic or not using the [PIMA Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database).

The solution includes:

- Training a Support Vector Machine model using Scikit-learn
- Saving the trained model using Pickle
- Deploying a web application using Streamlit

---

## 📁 Project Structure

diabetes-prediction/
│
├── diabetes.csv # Dataset from Kaggle
├── train_model.py # Python script to train the model
├── trained_model.sav # Saved model (generated after training)
├── diabetes_app.py # Streamlit web app for predictions
└── requirements.txt # Required libraries (optional but recommended)

yaml


---

## 🧰 Tools & Libraries Used

- Python 3.8+
- pandas
- numpy
- scikit-learn
- pickle (standard lib)
- streamlit

---

## 🛠️ Setup Instructions

1️⃣ Clone the Repository

git clone https://github.com/yourusername/diabetes-prediction.git
cd diabetes-prediction

2️⃣ Set Up a Virtual Environment (Recommended)
To keep dependencies isolated, create a virtual environment:

python -m venv venv
source venv/bin/activate     # On Linux/Mac
venv\Scripts\activate        # On Windows

3️⃣ Install Required Libraries

pip install -r requirements.txt
If you don’t have requirements.txt, you can install manually:

pip install pandas numpy scikit-learn streamlit


🎯 Step 1: Train the Model
Run the training script to train and save your model:

python train_model.py

This will:

Load the diabetes.csv dataset

Train a Support Vector Classifier

Evaluate accuracy

Save the trained model as trained_model.sav



🌐 Step 2: Launch the Web App
Now run the Streamlit app:

streamlit run diabetes_app.py
This will open a browser window with an interactive form to input medical values and get a diabetes prediction.

🧪 Sample Input Fields in the App

Pregnancies

Glucose Level

Blood Pressure

Skin Thickness

Insulin Level

BMI

Diabetes Pedigree Function

Age

☁️ Deploying to Streamlit Cloud (Optional)
Want to deploy online?

Push this project to a public GitHub repository

Go to Streamlit Cloud

Connect your GitHub repo and deploy diabetes_app.py



📎 License
This project is open-source under the MIT License — free to use and modify.



🙋‍♀️ Author
Aagama AR


If you found this helpful, please ⭐ the repo and share it!

yaml


---

### ✅ Also include a `requirements.txt`

You can generate it with:

```bash

pip freeze > requirements.txt
Or just create it manually with:

pandas
numpy
scikit-learn
streamlit
