"""Téléchargement, chargement et nettoyage du jeu de données Adult."""

from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd

from src.config import (
    COLUMNS,
    DATA_PROCESSED,
    DATA_RAW,
    NEGATIVE_LABEL,
    POSITIVE_LABEL,
    RAW_FILES,
    TARGET,
    UCI_BASE_URL,
)


def download_data(dest: Path | str = DATA_RAW, overwrite: bool = False) -> list[Path]:
    """Télécharge adult.data et adult.test depuis le site UCI.

    Args:
        dest: dossier de destination.
        overwrite: si False, les fichiers déjà présents ne sont pas retéléchargés.

    Returns:
        La liste des chemins des fichiers présents après l'appel.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    paths = []
    for filename in RAW_FILES.values():
        path = dest / filename
        if overwrite or not path.exists():
            print(f"Téléchargement de {filename}...")
            urllib.request.urlretrieve(UCI_BASE_URL + filename, path)
        paths.append(path)
    return paths


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Nettoie un DataFrame brut.

    - supprime les espaces parasites dans les colonnes texte ;
    - harmonise la cible : le fichier de test écrit ">50K." au lieu de ">50K" ;
    - vérifie qu'il ne reste que les deux modalités attendues.
    """
    df = df.copy()
    # Compatible pandas 2 (dtype object) et pandas 3 (dtype str).
    text_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    for col in text_cols:
        df[col] = df[col].str.strip()
    df[TARGET] = df[TARGET].str.rstrip(".")

    labels = set(df[TARGET].dropna().unique())
    unexpected = labels - {POSITIVE_LABEL, NEGATIVE_LABEL}
    if unexpected:
        raise ValueError(f"Modalités inattendues dans la cible : {unexpected}")
    return df.dropna(subset=[TARGET]).reset_index(drop=True)


def load_raw(split: str = "train", data_dir: Path | str | None = None) -> pd.DataFrame:
    """Charge et nettoie un des deux fichiers ("train" ou "test").

    Les valeurs manquantes, codées "?" dans les fichiers, deviennent des NaN.
    """
    if split not in RAW_FILES:
        raise ValueError(f"split doit valoir 'train' ou 'test', pas {split!r}")
    path = Path(data_dir or DATA_RAW) / RAW_FILES[split]
    if not path.exists():
        raise FileNotFoundError(
            f"{path} introuvable. Lancez `python scripts/download_data.py` "
            "ou copiez les fichiers dans data/raw/."
        )
    df = pd.read_csv(
        path,
        header=None,
        names=COLUMNS,
        skipinitialspace=True,
        na_values="?",
        # La 1re ligne de adult.test est un commentaire ("|1x3 Cross validator").
        skiprows=1 if split == "test" else 0,
    )
    return clean(df)


def load_train_test(data_dir: Path | str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Charge le découpage officiel train / test fourni par UCI."""
    return load_raw("train", data_dir), load_raw("test", data_dir)


def split_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Sépare les variables explicatives et la cible binaire (1 = revenu >50K)."""
    y = (df[TARGET] == POSITIVE_LABEL).astype(int).rename("target")
    X = df.drop(columns=[TARGET])
    return X, y


def save_processed_data(df: pd.DataFrame, filename: str) -> Path:
    """Sauvegarde des données prétraitées dans data/processed/ (au format CSV)."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    path = DATA_PROCESSED / filename
    df.to_csv(path, index=False)
    return path
