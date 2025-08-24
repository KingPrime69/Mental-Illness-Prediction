from flask import Flask, render_template, request, session, redirect, url_for
from sklearn.preprocessing import LabelEncoder
import joblib
import pandas as pd
import numpy as np
import random
import os

app = Flask(__name__)
app.secret_key = os.urandom(24)

model = joblib.load('./data/models/random_forest_model_v0.pkl')

# data = pd.read_csv('./data/processed/data.csv')

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

FEATURE_ORDER = [
    'program_category', 'region_served', 'age', 'Sex', 'transgender', 
    'sexual_orientation', 'hispanic_ethnicity', 'race', 'living_situation', 
    'household_composition', 'preferred_language', 'religious_preference', 
    'veteran_status', 'employment_status', 'education_status', 0, 'intellectual_disability', 
    'autism_spectrum', 'other_developmental_disability', 'alcohol_disorder', 
    'drug_disorder', 'opioid_disorder', 'mobility_impairment', 'hearing_impairment', 
    'vision_impairment', 'speech_impairment', 'Hyperlipidemia', 'high_blood_pressure', 
    'diabetes', 'obesity', 'heart_attack', 'stroke', 'other_cardiac', 
    'pulmonary_asthma', 'alzheimer_dementia', 'kidney_disease', 'liver_disease', 
    'endocrine', 'neurological', 'traumatic_brain_injury', 'joint_disease', 
    'cancer', 'other_chronic_med', 'no_chronic_med', 0, 'cannabis_recreational_use', 
    'cannabis_medicinal_use', 'smokes', 'received_smoking_medication', 
    'received_smoking_counseling', 'alcohol_12m_service', 'opioid_12m_service', 
    'drug_substance_12m_service', 'ssi_cash_assistance', 'ssdi_cash_assistance', 
    'veterans_disability_benefits', 'veteran_cash_assistance', 'public_assistance_cash_program', 
    'other_cash_benefits', 'medicaid_medicare', 'no_insurance', 'medicaid', 
    'medicaid_managed_insurance', 'medicare', 'private_insurance', 'child_health_plus', 
    'other_insurance', 'criminal_justice_status'
]

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
            ordered_data = []
            for feature in FEATURE_ORDER:
                if feature in form_data:
                    ordered_data.append(form_data[feature])
                else:
                    ordered_data.append(0)

            input_data = np.array(ordered_data).reshape(1, -1)

            # call preprocessing function

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