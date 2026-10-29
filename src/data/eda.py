"""Fonctions de statistiques descriptives utilisées dans le notebook d'EDA."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency


def missing_report(df: pd.DataFrame) -> pd.DataFrame:
    """Nombre et pourcentage de valeurs manquantes par colonne (colonnes concernées uniquement)."""
    n_missing = df.isna().sum()
    report = pd.DataFrame({"n_manquants": n_missing, "pct_manquants": 100 * n_missing / len(df)})
    return report[report["n_manquants"] > 0].sort_values("n_manquants", ascending=False)


def target_rate_by(df: pd.DataFrame, col: str, target: pd.Series, min_count: int = 1) -> pd.DataFrame:
    """Effectif et taux de revenu >50K pour chaque modalité de `col`.

    Args:
        df: données.
        col: variable catégorielle à analyser.
        target: cible binaire alignée sur df (1 = >50K).
        min_count: modalités plus rares que ce seuil ignorées.
    """
    grouped = (
        pd.DataFrame({col: df[col].fillna("Manquant"), "target": target.values})
        .groupby(col)["target"]
        .agg(effectif="size", taux_50k="mean")
    )
    grouped["taux_50k"] = 100 * grouped["taux_50k"]
    return grouped[grouped["effectif"] >= min_count].sort_values("taux_50k", ascending=False)


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """V de Cramér (corrigé du biais, Bergsma 2013) entre deux variables catégorielles.

    0 = aucune association, 1 = association parfaite.
    """
    table = pd.crosstab(x, y)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    chi2 = chi2_contingency(table, correction=False)[0]
    n = table.to_numpy().sum()
    phi2 = chi2 / n
    r, k = table.shape
    phi2_corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    r_corr = r - (r - 1) ** 2 / (n - 1)
    k_corr = k - (k - 1) ** 2 / (n - 1)
    denom = min(k_corr - 1, r_corr - 1)
    return float(np.sqrt(phi2_corr / denom)) if denom > 0 else 0.0


def cramers_v_matrix(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Matrice symétrique des V de Cramér entre les colonnes catégorielles."""
    mat = pd.DataFrame(np.eye(len(cols)), index=cols, columns=cols)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            v = cramers_v(df[a].fillna("Manquant"), df[b].fillna("Manquant"))
            mat.loc[a, b] = mat.loc[b, a] = v
    return mat
