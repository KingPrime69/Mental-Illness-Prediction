# app.py
from flask import Flask, request, render_template_string, redirect, url_for
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# === À ajuster si besoin ===
MODEL_PATH = r"C:/Users/maud.busserolles/OneDrive - VINCI Energies/Bureau/Projet_Mental_Illness/Mental-Illness-Prediction/notebooks/random_forest_model.pkl"
DATA_PATH  = r"C:/Users/maud.busserolles/OneDrive - VINCI Energies/Bureau/Projet_Mental_Illness/Mental-Illness-Prediction/data/processed/data/data_clean.csv"

TARGET = "Mental Illness"
VALID_TARGET = {"YES": 1, "NO": 0}

# --- Chargements ---
pipe = pickle.load(open(MODEL_PATH, "rb"))
df = pd.read_csv(DATA_PATH)

# Garder uniquement YES/NO
df = df[df[TARGET].isin(VALID_TARGET.keys())].copy()
y_true_bin = df[TARGET].map(VALID_TARGET)

# X brut (les mêmes colonnes que lors du training)
X = df.drop(columns=[TARGET])

# Détection colonnes
cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
num_cols = X.select_dtypes(include=["number"]).columns.tolist()

# === Encodage LabelEncoder appris sur X complet ===
label_encoders = {}         # col -> LabelEncoder
label_mappings = {}         # col -> dict(valeur_str -> int)
X_le = X.copy()

for col in cat_cols:
    le = LabelEncoder()
    # On convertit tout en string pour stabilité (inclut NaN -> "nan")
    series_as_str = X[col].astype(str)
    le.fit(series_as_str)
    label_encoders[col] = le
    mapping = {cls: i for i, cls in enumerate(le.classes_)}
    label_mappings[col] = mapping
    # Remplace par codes
    X_le[col] = series_as_str.map(mapping).astype(int)

# Colonnes numériques: s'assurer que c'est bien numérique
for col in num_cols:
    X_le[col] = pd.to_numeric(X_le[col], errors="coerce")

# Optionnel: imputations simples côté app (au cas où)
X_le = X_le.fillna(-1)

# Conserver ordre des colonnes attendu par le modèle si dispo
if hasattr(pipe, "feature_names_in_"):
    FEATURE_ORDER = list(pipe.feature_names_in_)
else:
    FEATURE_ORDER = list(X_le.columns)

# Pour l’UI
X_display = X.reset_index(drop=True)            # valeurs brutes pour affichage
X_le = X_le.reset_index(drop=True)              # valeurs encodées pour prédiction
y_true_bin = y_true_bin.reset_index(drop=True)
y_true_text = df[TARGET].reset_index(drop=True)

app = Flask(__name__)

HTML = """
<!doctype html>
<html lang="fr">
<head>
  <meta charset="utf-8">
  <title>Vérification du modèle - Mental Illness</title>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <style>
    body{font-family:system-ui,-apple-system,Segoe UI,Roboto,Ubuntu,"Helvetica Neue",Arial; margin:24px; line-height:1.5}
    .card{max-width:1000px;margin:auto;padding:24px;border:1px solid #e5e7eb;border-radius:16px;box-shadow:0 6px 16px rgba(0,0,0,.06)}
    .row{display:flex;gap:12px;align-items:center;flex-wrap:wrap}
    select,input,button{padding:10px;border:1px solid #d1d5db;border-radius:10px}
    .pill{display:inline-block;padding:6px 10px;border-radius:999px;font-weight:600}
    .ok{background:#d1fae5;color:#065f46}
    .ko{background:#fee2e2;color:#991b1b}
    table{border-collapse:collapse;width:100%}
    th,td{border:1px solid #e5e7eb;padding:8px;text-align:left}
    th{background:#f3f4f6}
  </style>
</head>
<body>
  <div class="card">
    <h2>Vérifier la prédiction du modèle</h2>
    <form class="row" method="get" action="/">
      <label for="row_id">Observation (ligne) :</label>
      <select id="row_id" name="row_id">
        {% for rid in sample_ids %}
          <option value="{{ rid }}" {% if rid == row_id %}selected{% endif %}>Ligne {{ rid }}</option>
        {% endfor %}
      </select>
      <button type="submit">Prédire</button>
      <a href="{{ url_for('random_row') }}"><button type="button">Ligne aléatoire</button></a>
    </form>

    {% if predicted is not none %}
      <hr>
      <h3>Résultat</h3>
      <p><strong>Vérité terrain :</strong> {{ true_text }} ({{ true_bin }})</p>
      <p><strong>Prédiction :</strong> {{ pred_text }} ({{ predicted }})</p>
      {% if proba is not none %}
        <p><strong>Proba classe YES (1):</strong> {{ '{:.3f}'.format(proba) }}</p>
      {% endif %}
      <p>
        <span class="pill {% if correct %}ok{% else %}ko{% endif %}">
          {% if correct %}Correct{% else %}Incorrect{% endif %}
        </span>
      </p>

      <h4>Caractéristiques de la ligne {{ row_id }}</h4>
      <table>
        <thead><tr><th>Colonne</th><th>Valeur</th></tr></thead>
        <tbody>
          {% for col, val in row_items %}
            <tr><td>{{ col }}</td><td>{{ val }}</td></tr>
          {% endfor %}
        </tbody>
      </table>
    {% endif %}
  </div>
</body>
</html>
"""

def encode_single_row_with_labelencoders(x_row_raw: pd.DataFrame) -> pd.DataFrame:
    """Transforme une ligne brute en features numériques encodées LabelEncoder."""
    x = x_row_raw.copy()

    # Encoder chaque col catégorielle avec le mapping appris
    for col in cat_cols:
        s = x[col].astype(str)
        mapping = label_mappings[col]
        # valeurs inédites -> -1
        x[col] = s.map(mapping).fillna(-1).astype(int)

    # Colonnes numériques en float, NaN -> -1
    for col in num_cols:
        x[col] = pd.to_numeric(x[col], errors="coerce")

    x = x.fillna(-1)

    # Aligner l'ordre des features comme au training (si connu)
    # et ne garder que ces colonnes
    x = x.reindex(columns=FEATURE_ORDER, fill_value=-1)

    return x

@app.route("/")
def index():
    n = len(X_display)
    try:
        row_id = int(request.args.get("row_id", 0))
    except:
        row_id = 0
    row_id = max(0, min(row_id, n-1))

    sample_ids = list(range(min(100, n)))

    predicted = None
    proba = None
    true_bin = None
    true_text = None
    pred_text = None
    correct = None
    row_items = []

    if n > 0:
        # Lignes brute (pour affichage) et encodée (pour prédiction)
        x_row_raw = X_display.iloc[[row_id]]      # DataFrame (1 ligne)
        x_row_le  = encode_single_row_with_labelencoders(x_row_raw)

        # Prédiction
        y_hat = pipe.predict(x_row_le)[0]
        predicted = int(y_hat)

        if hasattr(pipe, "predict_proba"):
            proba = float(pipe.predict_proba(x_row_le)[0][1])

        true_bin = int(y_true_bin.iloc[row_id])
        true_text = str(y_true_text.iloc[row_id])
        pred_text = "YES" if predicted == 1 else "NO"
        correct = (predicted == true_bin)
        row_items = list(zip(X_display.columns.tolist(), x_row_raw.iloc[0].tolist()))

    return render_template_string(
        HTML,
        sample_ids=sample_ids,
        row_id=row_id,
        predicted=predicted,
        proba=proba,
        true_bin=true_bin,
        true_text=true_text,
        pred_text=pred_text,
        correct=correct,
        row_items=row_items
    )

@app.route("/random")
def random_row():
    if len(X_display) == 0:
        return redirect(url_for("index"))
    return redirect(url_for("index", row_id=int(pd.Series(range(len(X_display))).sample(1).iloc[0])))

if __name__ == "__main__":
    app.run(debug=True)
