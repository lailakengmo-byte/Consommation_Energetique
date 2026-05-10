"""
save_model.py — Script de sauvegarde du modèle
À exécuter une fois depuis la racine du projet pour (re)générer model.pkl.
"""
import os
import warnings
import numpy as np
import pandas as pd
import joblib
warnings.filterwarnings("ignore")

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Chemins
BASE       = os.path.dirname(os.path.abspath(__file__))
DATA_PATH  = os.path.join(BASE, "data", "dataset.csv")
MODEL_DIR  = os.path.join(BASE, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Chargement ───────────────────────────────────────────
data = pd.read_csv(DATA_PATH)
print(f"Dataset chargé : {data.shape}")

# ── Preprocessing (fidèle au notebook) ──────────────────
data["jour_semaine"]    = data["jour_semaine"].fillna(data["jour_semaine"].mode()[0])
data["type_habitation"] = data["type_habitation"].fillna(data["type_habitation"].mode()[0])

data = pd.get_dummies(data, columns=["jour_semaine"],    drop_first=True)
data = pd.get_dummies(data, columns=["type_habitation"], drop_first=True)

data = data.dropna(subset=["consommation (kW)"])

for col in ["temperature (°C)", "humidite (%)", "vitesse_vent (km/h)",
            "heure", "nombre_personnes", "consommation (kW)"]:
    data[col].fillna(data[col].mean(), inplace=True)

X = data.drop("consommation (kW)", axis=1)
y = data["consommation (kW)"]

print(f"Après preprocessing : {X.shape[0]} lignes · {X.shape[1]} features")
print(f"Features : {list(X.columns)}")

# ── Division ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# ── Entraînement ─────────────────────────────────────────
model = DecisionTreeRegressor(max_depth=7, min_samples_leaf=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# ── Métriques ────────────────────────────────────────────
mae  = mean_absolute_error(y_test, y_pred)
mse  = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2   = r2_score(y_test, y_pred)

print(f"\n=== Métriques sur le jeu de test ===")
print(f"  MAE  : {mae:.4f} kW")
print(f"  MSE  : {mse:.4f}")
print(f"  RMSE : {rmse:.4f} kW")
print(f"  R²   : {r2:.4f}")

# ── Sauvegarde ───────────────────────────────────────────
joblib.dump(model, os.path.join(MODEL_DIR, "model.pkl"))
joblib.dump(list(X.columns), os.path.join(MODEL_DIR, "feature_names.pkl"))

print(f"\n✅ model.pkl sauvegardé dans {MODEL_DIR}/")
print(f"✅ feature_names.pkl sauvegardé dans {MODEL_DIR}/")
print("\nNote : un R² négatif ou proche de 0 est attendu sur ce dataset.")
print("Cela reflète le faible signal prédictif des données, pas un bug de code.")
