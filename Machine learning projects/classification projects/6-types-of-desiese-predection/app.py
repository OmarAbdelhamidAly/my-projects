from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained models
skin_cancer_model = joblib.load('skin_cancer.pkl')
other_cancer_model = joblib.load('other_cancer.pkl')
depression_model = joblib.load('depression.pkl')
arthritis_model = joblib.load('arthritis.pkl')
diabetes_model = joblib.load('diabetes.pkl')
heart_disease_model = joblib.load('heart_disease.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    features = [float(request.form.get(f)) for f in ['Height_(cm)', 'Weight_(kg)', 'BMI', 'Alcohol_Consumption', 'Fruit_Consumption', 'Green_Vegetables_Consumption', 'FriedPotato_Consumption']]
    data = pd.DataFrame([features], columns=['Height_(cm)', 'Weight_(kg)', 'BMI', 'Alcohol_Consumption', 'Fruit_Consumption', 'Green_Vegetables_Consumption', 'FriedPotato_Consumption'])

    skin_cancer_pred = skin_cancer_model.predict(data)[0]
    other_cancer_pred = other_cancer_model.predict(data)[0]
    depression_pred = depression_model.predict(data)[0]
    arthritis_pred = arthritis_model.predict(data)[0]
    diabetes_pred = diabetes_model.predict(data)[0]
    heart_disease_pred = heart_disease_model.predict(data)[0]

    results = {
        'Skin_Cancer': 'Yes' if skin_cancer_pred == 1 else 'No',
        'Other_Cancer': 'Yes' if other_cancer_pred == 1 else 'No',
        'Depression': 'Yes' if depression_pred == 1 else 'No',
        'Arthritis': 'Yes' if arthritis_pred == 1 else 'No',
        'Diabetes': 'Yes' if diabetes_pred == 1 else 'No',
        'Heart_Disease': 'Yes' if heart_disease_pred == 1 else 'No',
    }

    return render_template('result.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)
