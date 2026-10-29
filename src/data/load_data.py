"""
Module de chargement et préparation des données.
"""
import pandas as pd
from pathlib import Path

RAW_DATA_DIR = Path("data/raw")
PROCESSED_DATA_DIR = Path("data/processed")


def load_raw_data(filename: str) -> pd.DataFrame:
    """Charge les données brutes depuis data/raw/."""
    filepath = RAW_DATA_DIR / filename
    return pd.read_csv(filepath)


def save_processed_data(df: pd.DataFrame, filename: str) -> None:
    """Sauvegarde les données prétraitées dans data/processed/."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_DIR / filename, index=False)
