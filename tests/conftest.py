"""Données synthétiques au format exact des fichiers UCI, pour tester sans téléchargement."""

import numpy as np
import pandas as pd
import pytest

from src.api.predictor import Predictor
from src.config import COLUMNS
from src.data.load_data import load_raw, split_X_y
from src.models.models import get_pipelines


def make_fake_adult(n: int = 600, seed: int = 0, test_format: bool = False) -> pd.DataFrame:
    """Génère un faux jeu Adult : mêmes colonnes, mêmes modalités, "?" pour les manquants.

    La cible dépend de l'âge, de l'éducation et du statut marital pour que
    les modèles aient quelque chose à apprendre.
    """
    rng = np.random.default_rng(seed)
    age = rng.integers(17, 90, n)
    edu_num = rng.integers(1, 17, n)
    married = rng.choice(["Married-civ-spouse", "Never-married", "Divorced"], n)
    score = 0.04 * (age - 40) + 0.35 * (edu_num - 10) + 1.5 * (married == "Married-civ-spouse") - 1.2
    income = np.where(rng.random(n) < 1 / (1 + np.exp(-score)), ">50K", "<=50K")
    if test_format:
        income = np.char.add(income.astype(str), ".")

    def with_missing(values):
        values = values.astype(object)
        values[rng.random(n) < 0.05] = "?"
        return values

    return pd.DataFrame(
        {
            "age": age,
            "workclass": with_missing(rng.choice(["Private", "Self-emp-not-inc", "State-gov"], n)),
            "fnlwgt": rng.integers(10_000, 500_000, n),
            "education": rng.choice(["Bachelors", "HS-grad", "Masters"], n),
            "education-num": edu_num,
            "marital-status": married,
            "occupation": with_missing(rng.choice(["Tech-support", "Sales", "Exec-managerial"], n)),
            "relationship": rng.choice(["Husband", "Wife", "Not-in-family"], n),
            "race": rng.choice(["White", "Black", "Asian-Pac-Islander"], n),
            "sex": rng.choice(["Male", "Female"], n),
            "capital-gain": np.where(rng.random(n) < 0.1, rng.integers(1, 99_999, n), 0),
            "capital-loss": np.where(rng.random(n) < 0.05, rng.integers(1, 4_000, n), 0),
            "hours-per-week": rng.integers(10, 80, n),
            "native-country": with_missing(rng.choice(["United-States", "Mexico", "India"], n)),
            "income": income,
        },
        columns=COLUMNS,
    )


def write_fake_files(directory, n_train: int = 600, n_test: int = 300) -> None:
    """Écrit adult.data et adult.test dans `directory`, au format brut UCI."""
    train = make_fake_adult(n_train, seed=0)
    test = make_fake_adult(n_test, seed=1, test_format=True)
    train.to_csv(directory / "adult.data", header=False, index=False, sep=",")
    with open(directory / "adult.test", "w") as f:
        f.write("|1x3 Cross validator\n")
        test.to_csv(f, header=False, index=False, sep=",")


@pytest.fixture
def raw_dir(tmp_path):
    """Dossier temporaire contenant de faux fichiers adult.data / adult.test."""
    write_fake_files(tmp_path)
    return tmp_path


PERSON = {
    "age": 45,
    "workclass": "Private",
    "education-num": 13,
    "marital-status": "Married-civ-spouse",
    "occupation": "Exec-managerial",
    "relationship": "Husband",
    "race": "White",
    "sex": "Male",
    "capital-gain": 5000,
    "capital-loss": 0,
    "hours-per-week": 50,
    "native-country": "United-States",
}


@pytest.fixture
def predictor(raw_dir):
    """Modèle rapide entraîné sur les données synthétiques."""
    X, y = split_X_y(load_raw("train", raw_dir))
    model = get_pipelines()["Régression logistique"].fit(X, y)
    return Predictor(model, {"modele": "test", "seuil": 0.4})
