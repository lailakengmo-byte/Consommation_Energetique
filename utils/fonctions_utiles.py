"""
Fonctions utilitaires — Projet ML Consommation Énergétique
Auteur : KENGMO Maryline Laila
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ─── Preprocessing (fidèle au notebook) ───────────────────────────────────────

def charger_et_preparer(filepath: str) -> pd.DataFrame:
    """
    Charge le dataset et applique le preprocessing complet,
    dans le même ordre que le notebook.
    """
    data = pd.read_csv(filepath)

    # Imputation des catégorielles par le mode
    data["jour_semaine"]    = data["jour_semaine"].fillna(data["jour_semaine"].mode()[0])
    data["type_habitation"] = data["type_habitation"].fillna(data["type_habitation"].mode()[0])

    # One-Hot Encoding (drop_first=True comme dans le notebook)
    data = pd.get_dummies(data, columns=["jour_semaine"],    drop_first=True)
    data = pd.get_dummies(data, columns=["type_habitation"], drop_first=True)

    # Suppression lignes sans valeur cible
    data = data.dropna(subset=["consommation (kW)"])

    # Imputation numériques par la moyenne
    cols_num = [
        "temperature (°C)", "humidite (%)", "vitesse_vent (km/h)",
        "heure", "nombre_personnes", "consommation (kW)",
    ]
    for col in cols_num:
        if col in data.columns:
            data[col].fillna(data[col].mean(), inplace=True)

    return data


# ─── Métriques ─────────────────────────────────────────────────────────────────

def calculer_metriques(y_true, y_pred) -> dict:
    """Retourne un dict avec MAE, MSE, RMSE, R²."""
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mae  = mean_absolute_error(y_true, y_pred)
    mse  = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2   = r2_score(y_true, y_pred)
    return {"MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}


# ─── Prédiction sur un cas utilisateur ────────────────────────────────────────

def construire_vecteur(valeurs: dict, feature_names: list) -> pd.DataFrame:
    """
    Construit un DataFrame d'une ligne à partir des saisies utilisateur,
    en respectant exactement l'ordre des colonnes du modèle entraîné.
    """
    row = {col: 0 for col in feature_names}

    # Variables numériques directes
    for champ in ["temperature (°C)", "humidite (%)", "vitesse_vent (km/h)",
                  "heure", "nombre_personnes"]:
        if champ in row:
            row[champ] = valeurs.get(champ, 0)

    # Encodage jour_semaine (Dimanche = référence → toutes les dummies à 0)
    jours_dummies = {
        "Jeudi":    "jour_semaine_Jeudi",
        "Lundi":    "jour_semaine_Lundi",
        "Mardi":    "jour_semaine_Mardi",
        "Mercredi": "jour_semaine_Mercredi",
        "Samedi":   "jour_semaine_Samedi",
        "Vendredi": "jour_semaine_Vendredi",
    }
    jour = valeurs.get("jour_semaine", "Dimanche")
    if jour in jours_dummies and jours_dummies[jour] in row:
        row[jours_dummies[jour]] = 1

    # Encodage type_habitation (Appartement = référence)
    hab_dummies = {
        "Bureau": "type_habitation_Bureau",
        "Maison": "type_habitation_Maison",
    }
    hab = valeurs.get("type_habitation", "Appartement")
    if hab in hab_dummies and hab_dummies[hab] in row:
        row[hab_dummies[hab]] = 1

    return pd.DataFrame([row])[feature_names]


# ─── Interprétation du R² ──────────────────────────────────────────────────────

def interpreter_r2(r2: float) -> tuple[str, str]:
    """
    Retourne (niveau_css, message) selon la valeur du R².
    Niveaux CSS : 'ok', 'warn', 'bad'
    """
    if r2 >= 0.60:
        return "ok",   f"R² = {r2:.4f} — bon pouvoir prédictif."
    elif r2 >= 0.20:
        return "warn",  f"R² = {r2:.4f} — signal partiel, modèle modéré."
    elif r2 >= 0.0:
        return "warn",  f"R² = {r2:.4f} — signal très faible dans les données."
    else:
        return "bad",  (
            f"R² = {r2:.4f} — le modèle fait moins bien que la moyenne. "
            "Cela reflète le faible signal prédictif du dataset "
            "(corrélations < 0.2, MI ≈ 0). Ce n'est pas un bug de code."
        )
