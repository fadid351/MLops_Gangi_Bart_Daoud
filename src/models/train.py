"""
Script d'entraînement du modèle avec tracking MLflow.
"""
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import joblib
from pathlib import Path

MODEL_DIR = Path("models")


def train(X, y, model, params: dict = None):
    """
    Entraîne un modèle et logue les métriques avec MLflow.

    Args:
        X: Features
        y: Target
        model: Instance de modèle sklearn
        params: Hyperparamètres à logger
    """
    MODEL_DIR.mkdir(exist_ok=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    with mlflow.start_run():
        if params:
            mlflow.log_params(params)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        metrics = {
            "accuracy": accuracy_score(y_test, y_pred),
            "f1_score": f1_score(y_test, y_pred, average="weighted"),
        }
        mlflow.log_metrics(metrics)
        mlflow.sklearn.log_model(model, "model")

        print(f"Accuracy : {metrics['accuracy']:.4f}")
        print(f"F1 Score : {metrics['f1_score']:.4f}")

        # Sauvegarde locale
        joblib.dump(model, MODEL_DIR / "model.joblib")
        print("Modèle sauvegardé dans models/model.joblib")

    return model, metrics


if __name__ == "__main__":
    # Exemple de lancement — à adapter selon vos données
    print("Lancez ce script avec vos données chargées depuis src/data/load_data.py")
