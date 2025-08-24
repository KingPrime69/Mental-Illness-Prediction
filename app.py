from flask import Flask, render_template, request, session, redirect, url_for
from sklearn.preprocessing import LabelEncoder
import joblib
import pandas as pd
import numpy as np
import random
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.preprocessing import preprocessing

app = Flask(__name__)
app.secret_key = os.urandom(24)

model = joblib.load('./data/models/random_forest_model.pkl')
with open("./data/models/model_features.txt") as f:
    MODEL_FEATURES = [line.strip() for line in f]

# Use to see model require
# data = pd.read_csv('./data/raw/Patients.csv', sep=";", header=0)

# print("=== INFORMATIONS SUR LE MODÈLE ===")
# print(f"Nombre de champs/inputs attendus : {model.n_features_in_}")

# if hasattr(model, 'feature_names_in_'):
#     print(f"\nFeatures dans l'ordre :")
#     for i, feature in enumerate(model.feature_names_in_):
#         print(f"  {i+1}. {feature}")

# print(f"\nValeurs encodées par feature :")
# for col in data.select_dtypes(include=["object"]).columns:
#     le = LabelEncoder()
#     le.fit(data[col].astype(str))
#     print(f"\n{col} :")
#     for valeur, code in zip(le.classes_, range(len(le.classes_))):
#         print(f"  '{valeur}' -> {code}")

# print("=" * 50)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/form', methods=['GET', 'POST'])
def form():
    if request.method == 'POST':
        form_data = {}
        for key, value in request.form.items():
            form_data[key] = int(value) if value.isdigit() else value

        session['form_data'] = form_data
        
        try:
            input_data = pd.DataFrame([form_data])
            input_data = input_data.astype(str).apply(lambda x: x.str.upper())
            input_data = preprocessing(None,input_data,mode="inference")

            expected = list(model.feature_names_in_)
            input_data = input_data.reindex(columns=expected, fill_value=0)

            probabilities = model.predict_proba(input_data)
            prediction_class = model.predict(input_data)[0]
            
            print("Probabilities:", probabilities)
            print("Prediction class:", prediction_class)

            if prediction_class == 1:
                prediction = "HIGH"
                trust = round(probabilities[0][1] * 100, 1)
            else:
                prediction = "LOW"
                trust = round(probabilities[0][0] * 100, 1)
                
        except Exception as e:
            print("Error with model:", str(e))
            prediction = "ERROR"
            trust = 0

        session['prediction'] = prediction
        session['trust'] = trust

        return redirect(url_for('result'))
    else:
        form_data = session.get('form_data', {})
        has_previous = session.get('has_previous_data', False)
        
        return render_template('form.html', form_data=form_data, has_previous=has_previous)
    
@app.route('/result')
def result():
    prediction = session.get('prediction', 'UNKNOWN')
    trust = session.get('trust', 0)

    return render_template('result.html', prediction=prediction, trust=trust)



if __name__ == '__main__':
    app.run(debug=True)