"""Pipeline d'entraînement complet, avec suivi des expériences dans MLflow.

Étapes :
    1. Chargement du découpage officiel train / test.
    2. Validation croisée (5 plis) de tous les modèles avec leurs réglages par défaut.
    3. Optimisation des hyperparamètres des candidats.
    4. Choix du modèle final (meilleure ROC-AUC en validation croisée).
    5. Choix du seuil de décision qui maximise le F1 (sur le train, hors échantillon).
    6. Évaluation unique sur le jeu de test + importance des variables + équité.
    7. Sauvegarde du modèle pour l'API (models/model.joblib + models/model_meta.json).

Chaque modèle testé devient un « run » MLflow (paramètres, métriques, figures, modèle).
Interface : `mlflow ui` puis http://localhost:5000.

Usage (depuis la racine du repo) :
    python -m src.models.train            # complet (~5-15 min)
    python -m src.models.train --fast     # rapide, pour vérifier que tout tourne
    python -m src.models.train --no-mlflow
"""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext

import matplotlib

matplotlib.use("Agg")  # pas de fenêtre graphique : on enregistre les figures
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.model_selection import StratifiedKFold, cross_val_predict  # noqa: E402

from src.config import (  # noqa: E402
    FIGURES_DIR,
    MLFLOW_EXPERIMENT,
    MODEL_META_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    REPORTS_DIR,
    SENSITIVE_FEATURES,
)
from src.data.load_data import load_train_test, split_X_y  # noqa: E402
from src.models.evaluate import (  # noqa: E402
    CV_SCORING,
    best_threshold,
    compute_metrics,
    cross_validate_models,
    fairness_report,
    permutation_importance_df,
    plot_confusion,
    plot_cv_comparison,
    plot_importance,
    plot_pr_curves,
    plot_roc_curves,
)
from src.models.models import PARAM_DISTRIBUTIONS, get_pipelines, save_model, tune  # noqa: E402

try:
    import mlflow
    import mlflow.sklearn
except ImportError:  # MLflow est optionnel : le script fonctionne sans
    mlflow = None


class Tracker:
    """Petite surcouche de MLflow qui ne fait rien si MLflow est absent ou désactivé."""

    def __init__(self, enabled: bool):
        self.enabled = enabled and mlflow is not None
        if enabled and mlflow is None:
            print("MLflow non installé : pas de suivi des expériences (pip install mlflow).")
        if self.enabled:
            mlflow.set_experiment(MLFLOW_EXPERIMENT)

    def run(self, name: str, nested: bool = False):
        return mlflow.start_run(run_name=name, nested=nested) if self.enabled else nullcontext()

    def log(self, params: dict | None = None, metrics: dict | None = None, tags: dict | None = None):
        if not self.enabled:
            return
        if params:
            mlflow.log_params({k.replace("model__", ""): v for k, v in params.items()})
        if metrics:
            mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        if tags:
            mlflow.set_tags(tags)

    def artifact(self, path):
        if self.enabled:
            mlflow.log_artifact(str(path))

    def model(self, model, name: str = "model"):
        if not self.enabled:
            return
        try:
            mlflow.sklearn.log_model(model, name=name, serialization_format="cloudpickle")
        except Exception as e:  # selon la version de MLflow
            print(f"Modèle non enregistré par MLflow ({type(e).name}) : ajout du fichier joblib.")
            mlflow.log_artifact(str(MODEL_PATH))


def save_figure(draw, filename: str, tracker: Tracker) -> None:
    fig, ax = plt.subplots(figsize=(9, 6))
    draw(ax)
    fig.tight_layout()
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=120)
    plt.close(fig)
    tracker.artifact(path)


def main(fast: bool = False, use_mlflow: bool = True) -> dict:
    start = time.time()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    n_iter, cv = (3, 3) if fast else (20, 5)
    tracker = Tracker(use_mlflow)

    # 1. Données ---------------------------------------------------------------
    train, test = load_train_test()
    X_train, y_train = split_X_y(train)
    X_test, y_test = split_X_y(test)
    print(f"Train : {X_train.shape}, test : {X_test.shape}, taux >50K train : {y_train.mean():.1%}")

    with tracker.run("comparaison-modeles"):
        tracker.log(params={"cv_folds": cv, "n_iter": n_iter, "n_train": len(X_train)})

        # 2. Comparaison des modèles par défaut ----------------------------------
        print("\n[2] Validation croisée des modèles par défaut...")
        pipelines = get_pipelines()
        cv_default = cross_validate_models(pipelines, X_train, y_train, cv=cv)
        cv_default.to_csv(REPORTS_DIR / "cv_default.csv", index=False)
        for _, row in cv_default.iterrows():
            with tracker.run(f"defaut - {row['modele']}", nested=True):
                tracker.log(
                    params=pipelines[row["modele"]].named_steps["model"].get_params(),
                    metrics={f"cv_{m}": row[f"{m}_mean"] for m in CV_SCORING},
                    tags={"etape": "defaut", "modele": row["modele"]},
                )
        print(cv_default[["modele", "roc_auc_mean", "f1_mean", "pr_auc_mean", "fit_time_s"]].round(4))
        tracker.artifact(REPORTS_DIR / "cv_default.csv")

        # 3. Tuning --------------------------------------------------------------
        print(f"\n[3] Tuning ({n_iter} combinaisons par modèle)...")
        searches, tuned_rows = {}, []
        for name in PARAM_DISTRIBUTIONS:
            search = tune(name, X_train, y_train, n_iter=n_iter, cv=cv)
            searches[name] = search
            i = search.best_index_
            row = {
                "modele": name,
                "roc_auc_mean": search.best_score_,
                "roc_auc_std": search.cv_results_["std_test_score"][i],
                "fit_time_s": search.cv_results_["mean_fit_time"][i],
                "best_params": json.dumps(search.best_params_, default=str),
            }
            tuned_rows.append(row)
            with tracker.run(f"tune - {name}", nested=True):
                tracker.log(
                    params=search.best_params_,
                    metrics={"cv_roc_auc": row["roc_auc_mean"], "cv_roc_auc_std": row["roc_auc_std"]},
                    tags={"etape": "tuning", "modele": name},
                )
            print(f"  {name:25s} ROC-AUC = {search.best_score_:.4f}")
        cv_tuned = pd.DataFrame(tuned_rows).sort_values("roc_auc_mean", ascending=False)
        cv_tuned.to_csv(REPORTS_DIR / "cv_tuned.csv", index=False)
        tracker.artifact(REPORTS_DIR / "cv_tuned.csv")

        # 4. Choix du modèle final -----------------------------------------------
        best_name = cv_tuned.iloc[0]["modele"]
        best_model = searches[best_name].best_estimator_
        print(f"\n[4] Modèle retenu : {best_name}")

        # 5. Seuil de décision (prédictions hors échantillon sur le train) --------
        oof_scores = cross_val_predict(
            best_model,
            X_train,
            y_train,
            cv=StratifiedKFold(cv, shuffle=True, random_state=RANDOM_STATE),
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]
        threshold, oof_f1 = best_threshold(y_train, oof_scores)
        print(f"[5] Seuil optimal F1 : {threshold:.3f} (F1 hors échantillon = {oof_f1:.4f})")

        # 6. Évaluation finale sur le test (une seule fois) ----------------------
        test_scores = {n: s.best_estimator_.predict_proba(X_test)[:, 1] for n, s in searches.items()}
        y_score = test_scores[best_name]
        y_pred = (y_score >= threshold).astype(int)
        results = {
            "seuil_0.5": compute_metrics(y_test, (y_score >= 0.5).astype(int), y_score),
            "seuil_optimal": compute_metrics(y_test, y_pred, y_score),
        }
        test_metrics = pd.DataFrame(results).T
        test_metrics.to_csv(REPORTS_DIR / "test_metrics.csv")
        print("\n[6] Performances sur le jeu de test :")
        print(test_metrics.round(4))

        pd.DataFrame(
            {n: compute_metrics(y_test, (s >= 0.5).astype(int), s) for n, s in test_scores.items()}
        ).T.to_csv(REPORTS_DIR / "test_metrics_all_models.csv")

        for col in SENSITIVE_FEATURES:
            path = REPORTS_DIR / f"fairness_{col}.csv"
            fairness_report(y_test, y_pred, X_test[col]).to_csv(path, index=False)
            tracker.artifact(path)

        importances = permutation_importance_df(best_model, X_test, y_test, n_repeats=2 if fast else 5)
        importances.to_csv(REPORTS_DIR / "permutation_importance.csv", index=False)

        save_figure(lambda ax: plot_cv_comparison(cv_default, ax=ax), "cv_comparison.png", tracker)
        save_figure(lambda ax: plot_roc_curves(test_scores, y_test, ax=ax), "roc_curves.png", tracker)
        save_figure(lambda ax: plot_pr_curves(test_scores, y_test, ax=ax), "pr_curves.png", tracker)
        save_figure(
            lambda ax: plot_confusion(y_test, y_pred, f"{best_name} (seuil {threshold:.2f})", ax=ax),
            "confusion_matrix.png",
            tracker,
        )
        save_figure(lambda ax: plot_importance(importances, ax=ax), "permutation_importance.png", tracker)

        # 7. Sauvegarde pour l'API -----------------------------------------------
        save_model(best_model, MODEL_PATH)
        meta = {
            "modele": best_name,
            "hyperparametres": searches[best_name].best_params_,
            "seuil": threshold,
            "cv_roc_auc": float(cv_tuned.iloc[0]["roc_auc_mean"]),
            "test": {k: {m: round(float(v), 4) for m, v in d.items()} for k, d in results.items()},
            "duree_s": round(time.time() - start, 1),
        }
        with open(MODEL_META_PATH, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=str)
        with open(REPORTS_DIR / "summary.json", "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False, default=str)

        tracker.log(
            params={"modele_retenu": best_name, "seuil": round(threshold, 4)},
            metrics={f"test_{m}": v for m, v in results["seuil_optimal"].items()},
        )
        tracker.artifact(MODEL_META_PATH)
        tracker.model(best_model)

    print(f"\nTerminé en {meta['duree_s']} s. Modèle : {MODEL_PATH}")
    return meta


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--fast", action="store_true", help="moins d'itérations, pour tester")
    parser.add_argument("--no-mlflow", action="store_true", help="désactive le suivi MLflow")
    args = parser.parse_args()
    main(fast=args.fast, use_mlflow=not args.no_mlflow)
