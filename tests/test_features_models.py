import numpy as np
import pandas as pd

from src.data.load_data import load_raw, split_X_y
from src.data.eda import cramers_v, missing_report, target_rate_by
from src.models.evaluate import best_threshold, compute_metrics, cross_validate_models, fairness_report
from src.features.build_features import build_preprocessor
from src.models.models import get_pipelines, tune


def test_preprocessor_output_has_no_nan(raw_dir):
    X, _ = split_X_y(load_raw("train", raw_dir))
    Xt = build_preprocessor().fit_transform(X)
    assert not np.isnan(Xt).any()
    assert Xt.shape[0] == len(X)


def test_preprocessor_drops_fnlwgt_and_education(raw_dir):
    X, _ = split_X_y(load_raw("train", raw_dir))
    names = build_preprocessor().fit(X).get_feature_names_out()
    assert "fnlwgt" not in names
    assert not any(n.startswith("education_") for n in names)


def test_preprocessor_handles_unknown_category(raw_dir):
    X, _ = split_X_y(load_raw("train", raw_dir))
    prep = build_preprocessor().fit(X)
    X_new = X.head(3).copy()
    X_new["native-country"] = "Atlantis"
    assert prep.transform(X_new).shape[0] == 3


def test_all_pipelines_fit_and_predict(raw_dir):
    X, y = split_X_y(load_raw("train", raw_dir))
    for name, pipe in get_pipelines().items():
        proba = pipe.fit(X, y).predict_proba(X)[:, 1]
        assert proba.shape == (len(X),), name


def test_cross_validation_table(raw_dir):
    X, y = split_X_y(load_raw("train", raw_dir))
    pipes = {k: v for k, v in get_pipelines().items() if k != "KNN"}
    table = cross_validate_models(pipes, X, y, cv=3, n_jobs=1)
    assert len(table) == len(pipes)
    assert table["roc_auc_mean"].between(0, 1).all()
    # Sur ces données, un vrai modèle doit battre la baseline.
    baseline = table.set_index("modele").loc["Baseline (majoritaire)", "roc_auc_mean"]
    assert table["roc_auc_mean"].max() > baseline


def test_tune_returns_fitted_search(raw_dir):
    X, y = split_X_y(load_raw("train", raw_dir))
    search = tune("Régression logistique", X, y, n_iter=2, cv=3, n_jobs=1)
    assert 0 <= search.best_score_ <= 1
    assert search.best_estimator_.predict(X.head()).shape == (5,)


def test_metrics_and_threshold():
    y = np.array([0, 0, 1, 1, 1, 0])
    s = np.array([0.1, 0.4, 0.35, 0.8, 0.9, 0.2])
    m = compute_metrics(y, (s >= 0.5).astype(int), s)
    assert set(m) == {"accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc"}
    t, f1 = best_threshold(y, s)
    assert 0 < t < 1 and f1 >= m["f1"]


def test_fairness_report_per_group():
    y = np.array([1, 0, 1, 0])
    p = np.array([1, 0, 0, 0])
    report = fairness_report(y, p, pd.Series(["Male", "Male", "Female", "Female"]))
    assert set(report["groupe"]) == {"Male", "Female"}
    assert report.set_index("groupe").loc["Male", "tpr"] == 1.0


def test_eda_helpers(raw_dir):
    X, y = split_X_y(load_raw("train", raw_dir))
    assert "workclass" in missing_report(X).index
    rates = target_rate_by(X, "sex", y)
    assert rates["taux_50k"].between(0, 100).all()
    assert cramers_v(X["sex"], X["sex"]) > 0.9
