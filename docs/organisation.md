# Organisation du travail en groupe

## Répartition des rôles

Chaque membre a un domaine principal, mais **chaque Pull Request est relue par un autre membre**.

| | Daoud | Gangi | Bart |
|---|---|---|---|
| **Domaine** | Infrastructure MLOps & API | Données & EDA | Modélisation & évaluation |
| **Code** | `src/api/`, CI, Makefile, tracking MLflow (`train.py`) | `config.py`, `src/data/`, `scripts/` | `src/features/`, `models.py`, `evaluate.py` |
| **Notebook** | — | `01_eda.ipynb` | `02_modeling.ipynb` |
| **Tests** | `test_api.py`, `test_predictor.py` | `test_data.py`, `conftest.py` | `test_features_models.py` |
| **Relit** | les PR de Gangi | les PR de Bart | les PR de Daoud |

## Déroulé

| Étape | Contenu | Livrable |
|---|---|---|
| 1. Mise en place | Structure du repo, branches `main` / `develop`, CI | Squelette MLOps |
| 2. Données & EDA | Chargement, nettoyage, statistiques descriptives | `01_eda.ipynb`, décisions de preprocessing |
| 3. Modélisation | Pipeline, 6 modèles, validation croisée, tuning, MLflow | `02_modeling.ipynb`, runs MLflow |
| 4. Évaluation | Test final, interprétation, équité, choix argumenté | `reports/`, conclusion |
| 5. Mise à disposition | API FastAPI branchée sur le modèle retenu | `make serve`, `/docs` |
| 6. Finalisation | README, relecture croisée, merge `develop` → `main`, tag `v1.0` | Release |

## Décisions prises en équipe

| Décision | Raison |
|---|---|
| Métrique principale ROC-AUC | Indépendante du seuil, robuste au déséquilibre des classes |
| Découpage officiel train / test UCI | Test jamais vu pendant le développement |
| Preprocessing dans un `Pipeline` sklearn | Pas de fuite de données ; l'API reçoit les données brutes |
| Code dans `src/`, notebooks pour l'analyse | Code testable et réutilisable |
| Tests sur données synthétiques | La CI tourne sans télécharger le dataset |
| MLflow optionnel dans `train.py` | Le script tourne même sans serveur MLflow |
| Analyse d'équité (sexe, origine) | Variables sensibles identifiées pendant l'EDA |
