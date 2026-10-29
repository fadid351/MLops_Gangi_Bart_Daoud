"""Constantes du projet : chemins, colonnes, graine aléatoire.

Tout ce qui est « réglage global » est centralisé ici pour éviter
les valeurs codées en dur dans les notebooks et les scripts.
"""

import os
from pathlib import Path

# --- Chemins -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
# La variable d'environnement ADULT_DATA_DIR permet de pointer vers un autre dossier.
DATA_RAW = Path(os.environ.get("ADULT_DATA_DIR", PROJECT_ROOT / "data" / "raw"))
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "model.joblib"            # chargé par l'API
MODEL_META_PATH = MODELS_DIR / "model_meta.json"    # nom, seuil, métriques du modèle
MLFLOW_EXPERIMENT = "adult-income"

UCI_BASE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/"
RAW_FILES = {"train": "adult.data", "test": "adult.test"}

# --- Reproductibilité --------------------------------------------------------
RANDOM_STATE = 42

# --- Colonnes ----------------------------------------------------------------
COLUMNS = [
    "age",
    "workclass",
    "fnlwgt",
    "education",
    "education-num",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "capital-gain",
    "capital-loss",
    "hours-per-week",
    "native-country",
    "income",
]
TARGET = "income"
POSITIVE_LABEL = ">50K"
NEGATIVE_LABEL = "<=50K"

# Variables numériques très asymétriques (beaucoup de 0) -> transformation log.
SKEWED_FEATURES = ["capital-gain", "capital-loss"]
NUMERIC_FEATURES = ["age", "education-num", "hours-per-week"]
CATEGORICAL_FEATURES = [
    "workclass",
    "marital-status",
    "occupation",
    "relationship",
    "race",
    "sex",
    "native-country",
]
# fnlwgt : poids de sondage du recensement, pas une caractéristique de l'individu.
# education : redondante avec education-num (bijection).
DROPPED_FEATURES = ["fnlwgt", "education"]

# Variables utilisées pour l'analyse d'équité (fairness) du modèle final.
SENSITIVE_FEATURES = ["sex", "race"]
