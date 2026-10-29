"""API FastAPI qui sert le modèle de prédiction du revenu (>50K$).

Lancement (depuis la racine du repo, après `python -m src.models.train`) :
    uvicorn src.api.main:app --reload
Documentation interactive : http://localhost:8000/docs
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from src.api.predictor import Predictor
from src.config import MODEL_META_PATH, MODEL_PATH

# Variables d'environnement utiles pour les tests ou un déploiement.
model_path = Path(os.environ.get("MODEL_PATH", MODEL_PATH))
meta_path = Path(os.environ.get("MODEL_META_PATH", MODEL_META_PATH))
state: dict = {"predictor": None}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Charge le modèle au démarrage de l'API."""
    if model_path.exists():
        state["predictor"] = Predictor.from_files(model_path, meta_path)
        print(f"Modèle chargé : {model_path}")
    else:
        print("Aucun modèle trouvé. Entraînez d'abord avec `make train`.")
    yield


app = FastAPI(
    title="Adult Income API",
    description="Prédit si une personne gagne plus de 50 000 $/an (dataset UCI Adult).",
    version="1.0.0",
    lifespan=lifespan,
)


class Person(BaseModel):
    """Caractéristiques d'une personne (mêmes noms que les colonnes du dataset)."""

    model_config = ConfigDict(
        populate_by_name=True,
        json_schema_extra={
            "example": {
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
        },
    )

    age: int = Field(ge=17, le=100)
    workclass: str | None = None
    education_num: int = Field(alias="education-num", ge=1, le=16)
    marital_status: str = Field(alias="marital-status")
    occupation: str | None = None
    relationship: str
    race: str
    sex: str
    capital_gain: float = Field(0, alias="capital-gain", ge=0)
    capital_loss: float = Field(0, alias="capital-loss", ge=0)
    hours_per_week: int = Field(alias="hours-per-week", ge=1, le=99)
    native_country: str | None = Field(None, alias="native-country")


class Prediction(BaseModel):
    probabilite_50k: float
    prediction: str
    seuil: float


@app.get("/")
def root():
    return {"status": "ok", "message": "Adult Income API opérationnelle", "docs": "/docs"}


@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": state["predictor"] is not None}


@app.get("/model-info")
def model_info():
    """Nom du modèle, hyperparamètres, seuil et métriques de test."""
    predictor = _get_predictor()
    return predictor.meta or {"seuil": predictor.threshold}


@app.post("/predict", response_model=Prediction)
def predict(person: Person):
    """Prédiction pour une personne."""
    return _get_predictor().predict([person.model_dump(by_alias=True)])[0]


@app.post("/predict/batch", response_model=list[Prediction])
def predict_batch(persons: list[Person]):
    """Prédictions pour plusieurs personnes en un appel."""
    return _get_predictor().predict([p.model_dump(by_alias=True) for p in persons])


def _get_predictor() -> Predictor:
    if state["predictor"] is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé : lancez `make train`.")
    return state["predictor"]
