# MLops_Gangi_Bart_Daoud

Projet MLOps — Master 1 Data Science  
Équipe : Gangi · Bart · Daoud

---

## Objectif

Développer, versionner et déployer un modèle de Machine Learning en suivant les bonnes pratiques MLOps : reproductibilité, versioning des données et modèles, automatisation des pipelines, et monitoring.

## Structure du projet

```
MLops_Gangi_Bart_Daoud/
├── data/
│   ├── raw/            # Données brutes (non modifiées)
│   └── processed/      # Données prétraitées
├── models/             # Modèles entraînés sérialisés
├── notebooks/          # Exploration et expérimentation
├── src/
│   ├── data/           # Scripts de chargement et preprocessing
│   ├── features/       # Feature engineering
│   ├── models/         # Entraînement et évaluation
│   └── api/            # API de serving (FastAPI)
├── tests/              # Tests unitaires et d'intégration
├── .github/workflows/  # CI/CD pipelines
├── requirements.txt
├── Makefile
└── README.md
```

## Stack technique

| Composant       | Outil               |
|----------------|---------------------|
| Versioning ML   | MLflow              |
| Data versioning | DVC                 |
| API serving     | FastAPI             |
| CI/CD           | GitHub Actions      |
| Tests           | pytest              |
| Packaging       | Docker (optionnel)  |

## Installation

```bash
git clone https://github.com/fadid351/MLops_Gangi_Bart_Daoud.git
cd MLops_Gangi_Bart_Daoud
pip install -r requirements.txt
```

## Workflow Git

- `main` — branche stable, déploiement
- `develop` — branche d'intégration
- `feature/<nom>` — une branche par fonctionnalité ou membre

## Membres

| Membre | GitHub |
|--------|--------|
| Daoud  | @fadid351 |
| Gangi  | à compléter |
| Bart   | à compléter |
