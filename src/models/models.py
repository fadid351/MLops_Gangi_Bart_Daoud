"""Définition des modèles candidats, de leurs grilles d'hyperparamètres et du tuning."""

from __future__ import annotations

from pathlib import Path

import joblib
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.config import RANDOM_STATE
from src.features.build_features import build_preprocessor


def get_models(random_state: int = RANDOM_STATE) -> dict[str, BaseEstimator]:
    """Modèles candidats, du plus simple au plus complexe.

    - Baseline : prédit toujours la classe majoritaire (le score à battre).
    - Régression logistique : linéaire, rapide, interprétable.
    - Arbre de décision : non linéaire, interprétable, tendance au surapprentissage.
    - KNN : basé sur les distances, sensible au scaling.
    - Random Forest : moyenne de nombreux arbres (bagging), plus stable.
    - Gradient Boosting : arbres construits séquentiellement pour corriger les erreurs.
    """
    return {
        "Baseline (majoritaire)": DummyClassifier(strategy="most_frequent"),
        "Régression logistique": LogisticRegression(max_iter=2000),
        "Arbre de décision": DecisionTreeClassifier(max_depth=8, random_state=random_state),
        "KNN": KNeighborsClassifier(n_neighbors=25),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=2, n_jobs=-1, random_state=random_state
        ),
        "Gradient Boosting": HistGradientBoostingClassifier(random_state=random_state),
    }


# Espaces de recherche pour RandomizedSearchCV (préfixe "model__" = étape du pipeline).
PARAM_DISTRIBUTIONS: dict[str, dict[str, list]] = {
    "Régression logistique": {
        "model__C": [0.01, 0.03, 0.1, 0.3, 1, 3, 10],
        "model__class_weight": [None, "balanced"],
    },
    "Arbre de décision": {
        "model__max_depth": [4, 6, 8, 10, 12, None],
        "model__min_samples_leaf": [1, 5, 10, 20, 50],
        "model__criterion": ["gini", "entropy"],
    },
    "Random Forest": {
        "model__n_estimators": [200, 300, 500],
        "model__max_depth": [10, 15, 20, None],
        "model__min_samples_leaf": [1, 2, 5, 10],
        "model__max_features": ["sqrt", 0.3, 0.5],
    },
    "Gradient Boosting": {
        "model__learning_rate": [0.03, 0.05, 0.1, 0.2],
        "model__max_iter": [200, 300, 500],
        "model__max_leaf_nodes": [15, 31, 63],
        "model__min_samples_leaf": [10, 20, 50],
        "model__l2_regularization": [0.0, 0.1, 1.0],
    },
}


def make_pipeline(model: BaseEstimator) -> Pipeline:
    """Enchaîne le preprocessing et un modèle dans un seul objet scikit-learn."""
    return Pipeline([("prep", build_preprocessor()), ("model", model)])


def get_pipelines(random_state: int = RANDOM_STATE) -> dict[str, Pipeline]:
    """Tous les modèles candidats, chacun dans son pipeline complet."""
    return {name: make_pipeline(model) for name, model in get_models(random_state).items()}


def tune(
    name: str,
    X,
    y,
    n_iter: int = 20,
    cv: int = 5,
    scoring: str = "roc_auc",
    random_state: int = RANDOM_STATE,
    n_jobs: int = -1,
) -> RandomizedSearchCV:
    """Optimise les hyperparamètres d'un modèle par recherche aléatoire + validation croisée.

    Args:
        name: clé de get_models() présente dans PARAM_DISTRIBUTIONS.
        n_iter: nombre de combinaisons testées.
        scoring: métrique optimisée.

    Returns:
        L'objet RandomizedSearchCV entraîné (best_estimator_ réentraîné sur tout X).
    """
    if name not in PARAM_DISTRIBUTIONS:
        raise KeyError(f"Pas d'espace de recherche défini pour {name!r}")
    search = RandomizedSearchCV(
        make_pipeline(get_models(random_state)[name]),
        PARAM_DISTRIBUTIONS[name],
        n_iter=n_iter,
        scoring=scoring,
        cv=StratifiedKFold(cv, shuffle=True, random_state=random_state),
        random_state=random_state,
        n_jobs=n_jobs,
        refit=True,
    )
    return search.fit(X, y)


def save_model(model, path: Path | str) -> Path:
    """Sauvegarde un modèle entraîné avec joblib."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    return path


def load_model(path: Path | str):
    """Recharge un modèle sauvegardé avec save_model."""
    return joblib.load(path)
