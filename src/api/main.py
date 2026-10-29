"""
API FastAPI pour servir les prédictions du modèle.
"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import numpy as np
from pathlib import Path

app = FastAPI(
    title="MLOps API",
    description="API de prédiction — Projet MLOps M1 Data Science",
    version="0.1.0",
)

MODEL_PATH = Path("models/model.joblib")
model = None


@app.on_event("startup")
def load_model():
    global model
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        print("Modèle chargé.")
    else:
        print("Aucun modèle trouvé. Entraînez d'abord avec `make train`.")


class PredictionInput(BaseModel):
    features: list[float]


class PredictionOutput(BaseModel):
    prediction: float
    model_version: str = "0.1.0"


@app.get("/")
def root():
    return {"status": "ok", "message": "MLOps API opérationnelle"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}


@app.post("/predict", response_model=PredictionOutput)
def predict(input_data: PredictionInput):
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    features = np.array(input_data.features).reshape(1, -1)
    prediction = model.predict(features)[0]

    return PredictionOutput(prediction=float(prediction))
