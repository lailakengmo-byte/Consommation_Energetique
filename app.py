"""
Prédiction de Consommation Énergétique
Auteur : KENGMO Maryline Laila — Session Normale 2026
Modèle : DecisionTreeRegressor (max_depth=7, min_samples_leaf=100)
"""

import os
import sys
import warnings

import joblib
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor, plot_tree

warnings.filterwarnings("ignore")
matplotlib.use("Agg")

# ─── Chemins ──────────────────────────────────────────────────────────────────
BASE        = os.path.dirname(os.path.abspath(__file__))
DATA_PATH   = os.path.join(BASE, "data",  "dataset.csv")
MODEL_PATH  = os.path.join(BASE, "model", "model.pkl")
FEAT_PATH   = os.path.join(BASE, "model", "feature_names.pkl")

sys.path.insert(0, BASE)
from utils.fonctions_utiles import (
    charger_et_preparer,
    calculer_metriques,
    construire_vecteur,
    interpreter_r2,
)

# ─── Config page ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Consommation Énergétique — ML",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS minimal et propre ────────────────────────────────────────────────────
st.markdown("""
<style>
    /* titres de section */
    .section-title {
        font-size: 1.15rem; font-weight: 600; color: #1d3461;
        padding: 4px 0 4px 12px;
        border-left: 4px solid #4a90d9;
        margin: 1.2rem 0 0.8rem;
    }
    /* bandeaux d'info */
    .info-block  { background:#eef4fb; border-left:4px solid #4a90d9;
                   padding:.75rem 1rem; border-radius:3px; font-size:.9rem;
                   margin:.5rem 0; color:#1d3461; }
    .warn-block  { background:#fff8ec; border-left:4px solid #e8a020;
                   padding:.75rem 1rem; border-radius:3px; font-size:.9rem;
                   margin:.5rem 0; color:#7a4b00; }
    .bad-block   { background:#fdf0f0; border-left:4px solid #c0392b;
                   padding:.75rem 1rem; border-radius:3px; font-size:.9rem;
                   margin:.5rem 0; color:#7b1e1e; }
    .ok-block    { background:#f0faf3; border-left:4px solid #27ae60;
                   padding:.75rem 1rem; border-radius:3px; font-size:.9rem;
                   margin:.5rem 0; color:#145a32; }
    /* header principal */
    .main-header { font-size:2rem; font-weight:700; color:#1d3461; margin-bottom:.2rem; }
    .sub-header  { font-size:.92rem; color:#6b7280; margin-bottom:1.5rem; }
    /* footer */
    .footer { text-align:center; font-size:.78rem; color:#9ca3af; margin-top:2rem; }
</style>
""", unsafe_allow_html=True)


# ─── Chargement données & modèle (cachés) ─────────────────────────────────────

@st.cache_data
def load_raw():
    return pd.read_csv(DATA_PATH)

@st.cache_data
def load_prepared():
    return charger_et_preparer(DATA_PATH)

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_resource
def load_features():
    return joblib.load(FEAT_PATH)


raw   = load_raw()
data  = load_prepared()
model = load_model()
feats = load_features()

X = data.drop("consommation (kW)", axis=1)
y = data["consommation (kW)"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)
y_pred     = model.predict(X_test)
metriques  = calculer_metriques(y_test, y_pred)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("## ⚡ Navigation")
    page = st.radio("", [
        "🏠 Accueil",
        "📊 Exploration",
        "🤖 Modélisation",
        "📈 Résultats",
        "🔑 Variables importantes",
        "🔮 Prédiction",
        "📝 Conclusion",
    ], label_visibility="collapsed")

    st.markdown("---")
    st.caption("**Auteur :** KENGMO Maryline Laila")
    st.caption("**Session :** Normale 2026")
    st.caption("**Modèle :** DecisionTreeRegressor")
    st.caption(f"**Dataset :** {raw.shape[0]} lignes · {raw.shape[1]} colonnes")


# ═══════════════════════════════════════════════════════════
# PAGE 1 — ACCUEIL
# ═══════════════════════════════════════════════════════════

if page == "🏠 Accueil":

    st.markdown('<div class="main-header">⚡ Prédiction de Consommation Énergétique</div>',
                unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Projet de Machine Learning — Session Normale 2026 · KENGMO Maryline Laila</div>',
                unsafe_allow_html=True)

    col1, col2 = st.columns([1.6, 1])

    with col1:
        st.markdown('<div class="section-title">Contexte du projet</div>',
                    unsafe_allow_html=True)
        st.write(
            "Ce projet s'inscrit dans le cadre de l'examen de session normale "
            "de Machine Learning. L'objectif est de construire un modèle de régression "
            "capable de **prédire la consommation électrique** (en kWh) d'un logement "
            "ou local à partir de variables contextuelles : météo, heure, type "
            "d'habitation et nombre d'occupants."
        )

        st.markdown('<div class="section-title">Modèle utilisé</div>',
                    unsafe_allow_html=True)
        st.write(
            "Le modèle imposé est le **Decision Tree Regressor** de scikit-learn. "
            "Il a été entraîné avec `max_depth=7` et `min_samples_leaf=100` pour "
            "limiter l'overfitting sur un dataset à signal prédictif faible."
        )

        st.markdown('<div class="section-title">Honnêteté sur les performances</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        <div class="warn-block">
        Le R² obtenu est légèrement négatif (–0.027). Cela ne traduit pas un défaut
        de code, mais la réalité des données : les features disponibles n'expliquent
        pas suffisamment la variabilité de la consommation. Ce constat est documenté
        et argumenté dans la section Conclusion.
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="section-title">Vue rapide du dataset</div>',
                    unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Lignes",         raw.shape[0])
        c2.metric("Colonnes",       raw.shape[1])
        c1.metric("Valeurs manq.",  int(raw.isna().sum().sum()))
        c2.metric("% manquant",     f"{raw.isna().mean().mean()*100:.1f}%")

        st.markdown('<div class="section-title">Métriques clés</div>',
                    unsafe_allow_html=True)
        st.metric("R²",   f"{metriques['R2']:.4f}")
        st.metric("MAE",  f"{metriques['MAE']:.4f} kW")
        st.metric("RMSE", f"{metriques['RMSE']:.4f} kW")

    st.markdown('<div class="section-title">Variables du dataset</div>',
                unsafe_allow_html=True)
    desc = pd.DataFrame({
        "Colonne": [
            "temperature (°C)", "humidite (%)", "vitesse_vent (km/h)",
            "heure", "nombre_personnes", "jour_semaine",
            "type_habitation", "consommation (kW)",
        ],
        "Type": [
            "Numérique", "Numérique", "Numérique",
            "Numérique", "Numérique", "Catégorielle",
            "Catégorielle", "Numérique (cible)",
        ],
        "Description": [
            "Température extérieure", "Taux d'humidité", "Vitesse du vent",
            "Heure de la journée (0–23)", "Personnes présentes",
            "Lundi à Dimanche", "Appartement / Maison / Bureau",
            "Consommation électrique — variable à prédire",
        ],
    })
    st.dataframe(desc, use_container_width=True, hide_index=True)


# ═══════════════════════════════════════════════════════════
# PAGE 2 — EXPLORATION
# ═══════════════════════════════════════════════════════════

elif page == "📊 Exploration":

    st.markdown('<div class="section-title">Aperçu du dataset brut</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Lignes",   raw.shape[0])
    c2.metric("Colonnes", raw.shape[1])
    c3.metric("Valeurs manquantes", int(raw.isna().sum().sum()))
    c4.metric("Doublons", int(raw.duplicated().sum()))

    st.dataframe(raw.head(10), use_container_width=True)

    # Statistiques descriptives
    with st.expander("📋 Statistiques descriptives", expanded=True):
        st.dataframe(raw.describe().round(3), use_container_width=True)

    # Valeurs manquantes
    with st.expander("❓ Valeurs manquantes par colonne"):
        miss = pd.DataFrame({
            "Valeurs manquantes": raw.isna().sum(),
            "% manquant": (raw.isna().mean() * 100).round(1),
        }).sort_values("Valeurs manquantes", ascending=False)
        st.dataframe(miss, use_container_width=True)

    # Distribution de la cible
    st.markdown('<div class="section-title">Distribution de la consommation (kW)</div>',
                unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    conso = raw["consommation (kW)"].dropna()
    axes[0].hist(conso, bins=30, color="#4a90d9", edgecolor="white", alpha=0.85)
    axes[0].set_title("Distribution — consommation (kW)")
    axes[0].set_xlabel("kW"); axes[0].set_ylabel("Fréquence")
    axes[1].boxplot(conso, vert=True, patch_artist=True,
                    boxprops=dict(facecolor="#a8c8f0", color="#1d3461"))
    axes[1].set_title("Boxplot — détection des valeurs aberrantes")
    axes[1].set_ylabel("kW")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Heatmap corrélation
    st.markdown('<div class="section-title">Matrice de corrélation</div>',
                unsafe_allow_html=True)
    num_data = data.select_dtypes(include="number")
    fig, ax = plt.subplots(figsize=(14, 8))
    mask = np.triu(np.ones_like(num_data.corr(), dtype=bool))
    sns.heatmap(num_data.corr(), annot=True, fmt=".2f", cmap="Blues",
                mask=mask, ax=ax, linewidths=0.4, annot_kws={"size": 8})
    ax.set_title("Corrélation entre les variables (après encodage)")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("""
    <div class="warn-block">
    <b>Observation :</b> Les corrélations avec la cible <code>consommation (kW)</code>
    sont toutes inférieures à 0.2, ce qui indique un signal prédictif très limité.
    La variable la plus corrélée est <em>vitesse_vent</em> et <em>heure</em>,
    mais cette corrélation reste insuffisante pour un modèle précis.
    </div>
    """, unsafe_allow_html=True)

    # Consommation par variables catégorielles
    st.markdown('<div class="section-title">Consommation selon les catégories</div>',
                unsafe_allow_html=True)
    raw_clean = raw.dropna(subset=["consommation (kW)", "type_habitation", "jour_semaine"])
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    sns.boxplot(data=raw_clean, x="type_habitation", y="consommation (kW)",
                palette="Blues", ax=axes[0])
    axes[0].set_title("Consommation par type d'habitation")
    order = ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"]
    ex_ord = [j for j in order if j in raw_clean["jour_semaine"].unique()]
    sns.boxplot(data=raw_clean, x="jour_semaine", y="consommation (kW)",
                order=ex_ord, palette="Blues", ax=axes[1])
    axes[1].set_title("Consommation par jour de la semaine")
    axes[1].tick_params(axis="x", rotation=30)
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Scatterplot heure vs consommation
    st.markdown('<div class="section-title">Consommation vs Heure de la journée</div>',
                unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    sub = raw.dropna(subset=["consommation (kW)", "heure"])
    ax.scatter(sub["heure"], sub["consommation (kW)"],
               alpha=0.4, s=14, color="#4a90d9")
    moy_heure = sub.groupby("heure")["consommation (kW)"].mean()
    ax.plot(moy_heure.index, moy_heure.values, color="red",
            linewidth=2, label="Moyenne/heure")
    ax.set_xlabel("Heure"); ax.set_ylabel("Consommation (kW)")
    ax.set_title("Consommation vs Heure — recherche de patterns")
    ax.legend()
    plt.tight_layout()
    st.pyplot(fig); plt.close()


# ═══════════════════════════════════════════════════════════
# PAGE 3 — MODÉLISATION
# ═══════════════════════════════════════════════════════════

elif page == "🤖 Modélisation":

    st.markdown('<div class="section-title">Démarche de modélisation</div>',
                unsafe_allow_html=True)
    st.write(
        "Conformément aux consignes, le modèle retenu est le **DecisionTreeRegressor** "
        "de scikit-learn. Il a l'avantage d'être interprétable et ne nécessite pas "
        "de mise à l'échelle des données."
    )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-title">Hyperparamètres</div>',
                    unsafe_allow_html=True)
        params = pd.DataFrame({
            "Paramètre": ["max_depth", "min_samples_leaf", "random_state", "criterion"],
            "Valeur":    ["7",          "100",               "42",           "squared_error"],
            "Rôle": [
                "Profondeur maximale de l'arbre — limite l'overfitting",
                "Minimum d'observations par feuille — évite les nœuds trop spécifiques",
                "Reproductibilité des résultats",
                "Critère de division des nœuds (MSE)",
            ],
        })
        st.dataframe(params, use_container_width=True, hide_index=True)

    with col2:
        st.markdown('<div class="section-title">Pipeline ML</div>',
                    unsafe_allow_html=True)
        st.markdown("""
        1. **Chargement** du dataset (1000 lignes)
        2. **Imputation** catégorielles → mode, numériques → moyenne
        3. **Encodage** One-Hot (`drop_first=True`)
        4. **Split** 75% train / 25% test (`random_state=42`)
        5. **Entraînement** `DecisionTreeRegressor`
        6. **Évaluation** : MAE, MSE, RMSE, R²
        """)

    # Division données
    st.markdown('<div class="section-title">Division des données</div>',
                unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Lignes totales (après nettoyage)", data.shape[0])
    c2.metric("Lignes train (75%)",  X_train.shape[0])
    c3.metric("Lignes test (25%)",   X_test.shape[0])

    # Preprocessing — rappel
    st.markdown('<div class="section-title">Preprocessing appliqué</div>',
                unsafe_allow_html=True)
    with st.expander("Voir les étapes de preprocessing"):
        st.markdown("""
        **Variables catégorielles :**
        - `jour_semaine` : imputation par le mode, puis One-Hot Encoding
        - `type_habitation` : imputation par le mode, puis One-Hot Encoding
        - `drop_first=True` → référence : Dimanche / Appartement

        **Variables numériques :**
        - Lignes sans `consommation (kW)` → supprimées (on ne peut pas imputer la cible)
        - Autres colonnes numériques → imputation par la moyenne

        **Résultat :** 852 lignes × 13 features après encodage
        """)

    # Visualisation de l'arbre
    st.markdown('<div class="section-title">Visualisation de l\'arbre (profondeur 3)</div>',
                unsafe_allow_html=True)
    st.caption("Affichage limité à max_depth=3 pour la lisibilité.")
    dt_viz = DecisionTreeRegressor(max_depth=3, min_samples_leaf=100, random_state=42)
    dt_viz.fit(X_train, y_train)
    fig, ax = plt.subplots(figsize=(20, 8))
    plot_tree(dt_viz, feature_names=X.columns, filled=True,
              rounded=True, fontsize=9, max_depth=3, ax=ax)
    ax.set_title("Structure de l'arbre de décision (max_depth=3)")
    plt.tight_layout()
    st.pyplot(fig); plt.close()


# ═══════════════════════════════════════════════════════════
# PAGE 4 — RÉSULTATS
# ═══════════════════════════════════════════════════════════

elif page == "📈 Résultats":

    st.markdown('<div class="section-title">Métriques d\'évaluation</div>',
                unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("R²",   f"{metriques['R2']:.4f}")
    c2.metric("MAE",  f"{metriques['MAE']:.4f} kW")
    c3.metric("RMSE", f"{metriques['RMSE']:.4f} kW")
    c4.metric("MSE",  f"{metriques['MSE']:.4f}")

    # Bandeau d'interprétation
    niveau, msg = interpreter_r2(metriques["R2"])
    st.markdown(f'<div class="{niveau}-block">{msg}</div>', unsafe_allow_html=True)

    # Interprétation de chaque métrique
    with st.expander("📖 Interprétation détaillée des métriques"):
        st.markdown(f"""
        - **MAE = {metriques['MAE']:.4f} kW** : En moyenne, les prédictions s'écartent
          de {metriques['MAE']:.2f} kW des valeurs réelles.
        - **MSE = {metriques['MSE']:.4f}** : L'erreur quadratique pénalise davantage
          les grandes erreurs. Valeur de {metriques['MSE']:.2f} ici.
        - **RMSE = {metriques['RMSE']:.4f} kW** : Exprimé dans la même unité que la cible,
          l'erreur typique est d'environ {metriques['RMSE']:.2f} kW.
        - **R² = {metriques['R2']:.4f}** : Un R² négatif signifie que le modèle performe
          moins bien qu'une simple prédiction par la moyenne. Cela reflète l'absence
          de signal prédictif dans les données disponibles.
        """)

    # Graphiques prédictions vs réelles
    st.markdown('<div class="section-title">Prédictions vs Valeurs réelles</div>',
                unsafe_allow_html=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].scatter(y_test, y_pred, alpha=0.4, color="#4a90d9", s=18)
    mn = min(float(y_test.min()), float(y_pred.min()))
    mx = max(float(y_test.max()), float(y_pred.max()))
    axes[0].plot([mn, mx], [mn, mx], "r--", lw=2, label="Idéal")
    axes[0].axhline(float(y_train.mean()), color="orange", linestyle=":",
                    lw=1.5, label=f"Moy. train ({y_train.mean():.2f} kW)")
    axes[0].set_xlabel("Valeurs réelles (kW)")
    axes[0].set_ylabel("Prédictions (kW)")
    axes[0].set_title(f"Prédictions vs Réelles — R²={metriques['R2']:.3f}")
    axes[0].legend()

    residuals = y_test.values - y_pred
    axes[1].scatter(y_pred, residuals, alpha=0.4, color="#e07b39", s=18)
    axes[1].axhline(0, color="red", linestyle="--", lw=2)
    axes[1].set_xlabel("Prédictions (kW)")
    axes[1].set_ylabel("Résidus (kW)")
    axes[1].set_title("Analyse des résidus")

    plt.tight_layout()
    st.pyplot(fig); plt.close()

    st.markdown("""
    <div class="warn-block">
    <b>Lecture des graphiques :</b> Si le modèle était précis, les points du graphique
    de gauche suivraient la diagonale rouge. Ici, ils sont dispersés horizontalement,
    ce qui confirme que le modèle prédit une valeur proche de la moyenne pour tous les cas.
    Les résidus ne montrent pas de structure particulière — le bruit est aléatoire.
    </div>
    """, unsafe_allow_html=True)

    # Courbe overfitting
    st.markdown('<div class="section-title">Courbe d\'overfitting (max_depth)</div>',
                unsafe_allow_html=True)
    depths = [1, 2, 3, 4, 5, 6, 7, 8, 10, 12]
    r2_tr_list, r2_te_list = [], []
    for d in depths:
        m = DecisionTreeRegressor(max_depth=d, min_samples_leaf=100, random_state=42)
        m.fit(X_train, y_train)
        r2_tr_list.append(r2_score(y_train, m.predict(X_train)))
        r2_te_list.append(r2_score(y_test,  m.predict(X_test)))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(depths, r2_tr_list, "b-o", label="Train R²", lw=2)
    ax.plot(depths, r2_te_list, "r-o", label="Test R²",  lw=2)
    ax.axhline(0, color="gray", linestyle="--", lw=1, label="R²=0")
    ax.axvline(7, color="green", linestyle=":", lw=1.5, label="max_depth choisi (7)")
    ax.set_xlabel("max_depth"); ax.set_ylabel("R²")
    ax.set_title("Évolution du R² selon la profondeur")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig); plt.close()


# ═══════════════════════════════════════════════════════════
# PAGE 5 — IMPORTANCE DES VARIABLES
# ═══════════════════════════════════════════════════════════

elif page == "🔑 Variables importantes":

    st.markdown('<div class="section-title">Importance des variables — Decision Tree</div>',
                unsafe_allow_html=True)

    imp_df = pd.DataFrame({
        "Variable":  feats,
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    # Graphique
    fig, ax = plt.subplots(figsize=(10, 6))
    imp_sorted = imp_df.sort_values("Importance", ascending=True)
    colors = ["#4a90d9" if v > 0 else "#d0dce8" for v in imp_sorted["Importance"]]
    ax.barh(imp_sorted["Variable"], imp_sorted["Importance"], color=colors)
    ax.set_title("Importance des variables — DecisionTreeRegressor")
    ax.set_xlabel("Importance (somme des réductions d'impureté)")
    plt.tight_layout()
    st.pyplot(fig); plt.close()

    # Tableau
    st.dataframe(
        imp_df.assign(Importance=imp_df["Importance"].round(4)),
        use_container_width=True, hide_index=True
    )

    # Interprétation
    top_var = imp_df.iloc[0]["Variable"]
    top_val = imp_df.iloc[0]["Importance"]
    zero_count = int((imp_df["Importance"] == 0).sum())

    st.markdown(f"""
    <div class="info-block">
    <b>Interprétation :</b><br>
    La variable la plus utilisée par l'arbre est <b>{top_var}</b>
    (importance = {top_val:.4f}). {zero_count} variables ont une importance nulle,
    ce qui signifie que l'arbre ne les utilise pas pour ses divisions —
    signe supplémentaire du signal prédictif très faible du dataset.
    </div>
    """, unsafe_allow_html=True)

    with st.expander("ℹ️ Comment lire l'importance des variables ?"):
        st.markdown("""
        L'importance d'une variable dans un arbre de décision est calculée comme la **somme
        des réductions d'impureté** (ici MSE) obtenues à chaque nœud où cette variable est
        utilisée, pondérée par le nombre d'observations.

        - Une importance élevée → la variable est souvent choisie pour diviser les données
        - Une importance nulle → la variable n'est jamais sélectionnée par l'arbre
        - Les importances sont normalisées pour sommer à 1
        """)


# ═══════════════════════════════════════════════════════════
# PAGE 6 — PRÉDICTION
# ═══════════════════════════════════════════════════════════

elif page == "🔮 Prédiction":

    st.markdown('<div class="section-title">Simuler une prédiction</div>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="warn-block">
    <b>Note :</b> Compte tenu du R² quasi nul, les prédictions resteront proches de la
    <b>moyenne générale (~5.3 kW)</b> quelle que soit la saisie. Cette interface illustre
    le fonctionnement du modèle.
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        jour = st.selectbox("Jour de la semaine",
            ["Lundi","Mardi","Mercredi","Jeudi","Vendredi","Samedi","Dimanche"])
        hab  = st.selectbox("Type d'habitation", ["Appartement","Maison","Bureau"])

    with col2:
        temp     = st.slider("Température (°C)",     -10, 45,  15)
        humidite = st.slider("Humidité (%)",            0, 100, 50)

    with col3:
        vent    = st.slider("Vitesse du vent (km/h)", 0,  60, 15)
        heure   = st.slider("Heure de la journée",    0,  23, 12)
        nb_pers = st.slider("Nombre de personnes",    1,   7,  3)

    if st.button("⚡ Calculer la prédiction", type="primary"):
        valeurs = {
            "temperature (°C)":   temp,
            "humidite (%)":       humidite,
            "vitesse_vent (km/h)": vent,
            "heure":              heure,
            "nombre_personnes":   nb_pers,
            "jour_semaine":       jour,
            "type_habitation":    hab,
        }
        inp = construire_vecteur(valeurs, feats)
        pred = float(model.predict(inp)[0])
        moy  = float(y.mean())

        st.markdown('<div class="section-title">Résultat</div>', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        c1.metric("Consommation prédite", f"{pred:.2f} kW")
        c2.metric("Moyenne dataset",      f"{moy:.2f} kW",
                  delta=f"{pred - moy:+.2f} kW")

        # Mini visualisation
        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.barh(["Prédiction", "Moyenne dataset"],
                [pred, moy], color=["#4a90d9", "#a8c8f0"], height=0.4)
        ax.set_xlabel("kW")
        ax.set_title("Prédiction vs Moyenne")
        ax.axvline(moy, color="red", linestyle="--", lw=1.5, alpha=0.6)
        plt.tight_layout()
        st.pyplot(fig); plt.close()

        st.markdown(f"""
        <div class="info-block">
        <b>Contexte :</b> {nb_pers} personne(s) dans un(e) <b>{hab}</b>,
        un <b>{jour}</b> à <b>{heure}h</b>, {temp}°C.<br>
        La prédiction ({pred:.2f} kW) est très proche de la moyenne
        ({moy:.2f} kW), ce qui est attendu avec un R² ≈ 0.
        </div>
        """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# PAGE 7 — CONCLUSION
# ═══════════════════════════════════════════════════════════

elif page == "📝 Conclusion":

    st.markdown('<div class="section-title">Analyse et conclusion</div>',
                unsafe_allow_html=True)

    st.markdown("### Pourquoi un R² négatif ?")
    st.markdown("""
    Un R² négatif signifie que le modèle est moins performant que de prédire
    simplement la **moyenne** pour chaque observation. Cela peut paraître surprenant,
    mais s'explique par plusieurs facteurs :

    - **Les features ne capturent pas les vrais déterminants** de la consommation :
      superficie du logement, type d'équipements, isolation thermique...
    - **La relation entre variables est essentiellement aléatoire** dans ce dataset :
      les corrélations sont toutes inférieures à 0.2, et l'information mutuelle proche de 0.
    - **14–17% de valeurs manquantes** par colonne dégradent le signal disponible.
    """)

    st.markdown("### Ce que le modèle fait correctement")
    st.markdown("""
    Malgré les performances limitées, la démarche est rigoureuse :
    - Preprocessing complet (imputation, encodage, vérification des doublons)
    - Entraînement avec `random_state=42` pour la reproductibilité
    - Évaluation sur un jeu de test séparé (25%)
    - Analyse de l'importance des variables
    """)

    st.markdown("### Recommandations pour améliorer le modèle")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        **Sur les données :**
        - Collecter des features plus pertinentes (superficie, équipements)
        - Réduire le taux de valeurs manquantes
        - Vérifier la qualité des mesures de consommation
        """)
    with col2:
        st.markdown("""
        **Sur le modèle :**
        - Essayer Random Forest ou Gradient Boosting (XGBoost)
        - Explorer des approches de séries temporelles si données ordonnées
        - Optimiser les hyperparamètres via GridSearchCV
        """)

    st.markdown("### Conclusion académique")
    st.markdown("""
    <div class="info-block">
    Ce projet a appliqué une démarche Machine Learning complète et honnête.
    Le DecisionTreeRegressor a été entraîné, évalué et interprété correctement.
    Les performances obtenues (R² ≈ –0.027, RMSE ≈ 2.85 kW) sont fidèles à la
    réalité des données disponibles — elles n'ont pas été artificiellement
    améliorées ni dissimulées.<br><br>
    La limite principale identifiée est la <strong>faiblesse du signal prédictif</strong>
    dans le dataset, et non un défaut dans la méthodologie appliquée.
    </div>
    """, unsafe_allow_html=True)

    # Tableau de synthèse
    st.markdown('<div class="section-title">Tableau de synthèse</div>',
                unsafe_allow_html=True)
    synthese = pd.DataFrame({
        "Étape":   ["Chargement", "Preprocessing", "Modélisation",
                    "Évaluation", "Déploiement"],
        "Statut":  ["✅ OK", "✅ OK", "✅ OK", "✅ Métriques réelles", "✅ Streamlit"],
        "Détail":  [
            "1000 lignes, 8 colonnes, ~15% manquants",
            "Imputation mode/moyenne, One-Hot Encoding",
            "DecisionTreeRegressor(max_depth=7, min_samples_leaf=100)",
            f"R²={metriques['R2']:.3f}, MAE={metriques['MAE']:.3f}, RMSE={metriques['RMSE']:.3f}",
            "Application Streamlit multi-pages",
        ],
    })
    st.dataframe(synthese, use_container_width=True, hide_index=True)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">Projet Machine Learning · KENGMO Maryline Laila · Session Normale 2026</div>',
    unsafe_allow_html=True,
)
