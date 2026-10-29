"""Préparation des variables (preprocessing) sous forme de pipeline scikit-learn.

Tout le preprocessing est appris *dans* la validation croisée (fit sur les plis
d'entraînement uniquement), ce qui évite les fuites de données.
"""

from __future__ import annotations

import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from src.config import CATEGORICAL_FEATURES, NUMERIC_FEATURES, SKEWED_FEATURES


def build_preprocessor(
    numeric: list[str] = NUMERIC_FEATURES,
    skewed: list[str] = SKEWED_FEATURES,
    categorical: list[str] = CATEGORICAL_FEATURES,
    min_frequency: float = 0.005,
) -> ColumnTransformer:
    """Construit le ColumnTransformer appliqué avant chaque modèle.

    - numériques : imputation par la médiane puis standardisation ;
    - asymétriques (capital-gain / capital-loss) : log(1 + x) puis standardisation ;
    - catégorielles : imputation par la modalité la plus fréquente puis one-hot.
      Les modalités représentant moins de `min_frequency` des lignes sont
      regroupées (utile pour native-country qui a ~40 pays très rares).

    Les colonnes non listées (fnlwgt, education) sont ignorées.
    """
    numeric_pipe = Pipeline(
        [("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]
    )
    skewed_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("log", FunctionTransformer(np.log1p, feature_names_out="one-to-one")),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=min_frequency,
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_pipe, numeric),
            ("skew", skewed_pipe, skewed),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
