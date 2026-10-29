"""Évaluation des modèles : métriques, validation croisée, seuil, équité, graphiques."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    f1_score,
    make_scorer,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

from src.config import RANDOM_STATE

# Métriques calculées en validation croisée (noms scikit-learn).
CV_SCORING = {
    "accuracy": "accuracy",
    # zero_division=0 : la baseline ne prédit jamais >50K, sa precision vaut 0 sans warning.
    "precision": make_scorer(precision_score, zero_division=0),
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
    "pr_auc": "average_precision",
}


def compute_metrics(y_true, y_pred, y_score) -> dict[str, float]:
    """Métriques de classification binaire.

    - accuracy : part de bonnes prédictions (trompeuse ici : 76 % en prédisant toujours <=50K) ;
    - precision : parmi les prédits >50K, part qui le sont vraiment ;
    - recall : parmi les vrais >50K, part retrouvée ;
    - f1 : moyenne harmonique precision / recall ;
    - roc_auc : capacité à classer un >50K au-dessus d'un <=50K (indépendant du seuil) ;
    - pr_auc : aire sous la courbe precision-recall, adaptée aux classes déséquilibrées.
    """
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_score),
        "pr_auc": average_precision_score(y_true, y_score),
    }


def cross_validate_models(
    pipelines: dict, X, y, cv: int = 5, random_state: int = RANDOM_STATE, n_jobs: int = -1
) -> pd.DataFrame:
    """Validation croisée stratifiée de plusieurs pipelines.

    Returns:
        Un DataFrame (une ligne par modèle) avec moyenne et écart-type de chaque
        métrique, plus le temps d'entraînement moyen, trié par ROC-AUC décroissante.
    """
    splitter = StratifiedKFold(cv, shuffle=True, random_state=random_state)
    rows = []
    for name, pipe in pipelines.items():
        res = cross_validate(pipe, X, y, cv=splitter, scoring=CV_SCORING, n_jobs=n_jobs)
        row = {"modele": name}
        for metric in CV_SCORING:
            scores = res[f"test_{metric}"]
            row[f"{metric}_mean"] = scores.mean()
            row[f"{metric}_std"] = scores.std()
        row["fit_time_s"] = res["fit_time"].mean()
        rows.append(row)
    return pd.DataFrame(rows).sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)


def best_threshold(y_true, y_score) -> tuple[float, float]:
    """Seuil de décision qui maximise le F1 (au lieu du seuil 0,5 par défaut).

    À calculer sur des prédictions hors échantillon (cross_val_predict sur le train),
    jamais sur le jeu de test.

    Returns:
        (seuil, f1 obtenu avec ce seuil)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-12, None)
    # La dernière valeur de precision/recall n'a pas de seuil associé.
    idx = int(np.argmax(f1[:-1]))
    return float(thresholds[idx]), float(f1[idx])


def fairness_report(y_true, y_pred, groups: pd.Series) -> pd.DataFrame:
    """Comportement du modèle par groupe (ex. par sexe).

    - taux_reel_50k : proportion réelle de >50K dans le groupe ;
    - taux_predit_50k : proportion prédite >50K (parité démographique) ;
    - tpr (recall) : part des vrais >50K détectés (égalité des chances) ;
    - fpr : part des <=50K classés à tort >50K.
    """
    df = pd.DataFrame(
        {"y": np.asarray(y_true), "p": np.asarray(y_pred), "g": groups.fillna("Manquant").values}
    )
    rows = []
    for g, d in df.groupby("g"):
        pos, neg = d[d.y == 1], d[d.y == 0]
        rows.append(
            {
                "groupe": g,
                "effectif": len(d),
                "taux_reel_50k": d.y.mean(),
                "taux_predit_50k": d.p.mean(),
                "tpr": pos.p.mean() if len(pos) else np.nan,
                "fpr": neg.p.mean() if len(neg) else np.nan,
                "precision": d[d.p == 1].y.mean() if (d.p == 1).any() else np.nan,
            }
        )
    return pd.DataFrame(rows).sort_values("effectif", ascending=False).reset_index(drop=True)


def permutation_importance_df(model, X, y, scoring: str = "roc_auc", n_repeats: int = 5) -> pd.DataFrame:
    """Importance de chaque variable *d'origine* par permutation.

    On mélange une colonne et on mesure la baisse du score : plus la baisse est
    forte, plus le modèle dépend de cette variable. Fonctionne pour tout modèle.
    """
    res = permutation_importance(
        model, X, y, scoring=scoring, n_repeats=n_repeats, random_state=RANDOM_STATE, n_jobs=-1
    )
    return (
        pd.DataFrame(
            {"variable": X.columns, "importance": res.importances_mean, "std": res.importances_std}
        )
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )


# --- Graphiques --------------------------------------------------------------


def plot_cv_comparison(cv_results: pd.DataFrame, metrics=("roc_auc", "f1", "pr_auc"), ax=None):
    """Diagramme en barres (avec écart-type) des scores de validation croisée."""
    ax = ax or plt.subplots(figsize=(10, 5))[1]
    df = cv_results.set_index("modele")
    means = df[[f"{m}_mean" for m in metrics]].rename(columns=lambda c: c.replace("_mean", ""))
    stds = df[[f"{m}_std" for m in metrics]].rename(columns=lambda c: c.replace("_std", ""))
    means.plot.barh(xerr=stds, ax=ax, capsize=3)
    ax.set_xlabel("Score moyen en validation croisée (barre = écart-type)")
    ax.set_ylabel("")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    ax.legend(title="Métrique", loc="lower right")
    ax.set_title("Comparaison des modèles")
    return ax


def plot_roc_curves(scores: dict, y_true, ax=None):
    """Courbes ROC de plusieurs modèles. scores : {nom: probabilités prédites}."""
    ax = ax or plt.subplots(figsize=(6, 6))[1]
    for name, s in scores.items():
        fpr, tpr, _ = roc_curve(y_true, s)
        ax.plot(fpr, tpr, label=f"{name} (AUC={roc_auc_score(y_true, s):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Hasard")
    ax.set(xlabel="Taux de faux positifs", ylabel="Taux de vrais positifs", title="Courbes ROC")
    ax.legend(loc="lower right", fontsize=8)
    return ax


def plot_pr_curves(scores: dict, y_true, ax=None):
    """Courbes precision-recall. La ligne horizontale = taux de >50K (modèle au hasard)."""
    ax = ax or plt.subplots(figsize=(6, 6))[1]
    for name, s in scores.items():
        p, r, _ = precision_recall_curve(y_true, s)
        ax.plot(r, p, label=f"{name} (AP={average_precision_score(y_true, s):.3f})")
    ax.axhline(np.mean(y_true), color="k", ls="--", lw=1, label="Hasard")
    ax.set(xlabel="Recall", ylabel="Precision", title="Courbes precision-recall")
    ax.legend(loc="lower left", fontsize=8)
    return ax


def plot_confusion(y_true, y_pred, title: str = "Matrice de confusion", ax=None):
    """Matrice de confusion avec effectifs et pourcentages par ligne."""
    ax = ax or plt.subplots(figsize=(5, 4))[1]
    ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=["<=50K", ">50K"], cmap="Blues", ax=ax, colorbar=False
    )
    ax.grid(False)
    ax.set(title=title, xlabel="Classe prédite", ylabel="Classe réelle")
    return ax


def plot_importance(importances: pd.DataFrame, top: int = 15, ax=None):
    """Barres horizontales des importances par permutation."""
    ax = ax or plt.subplots(figsize=(7, 5))[1]
    d = importances.head(top).iloc[::-1]
    ax.barh(d["variable"], d["importance"], xerr=d["std"], capsize=3, color="#4C72B0")
    ax.set_xlabel("Baisse de ROC-AUC quand la variable est mélangée")
    ax.set_title("Importance des variables (permutation)")
    return ax
