# Organisation du travail en groupe

## Répartition des rôles

Chaque membre a un domaine principal. Tout le code passe par une branche `feature/*` et une
Pull Request vers `develop`, relue par un autre membre.

| | Daoud | Bart | Gangi |
|---|---|---|---|
| **Domaine** | Socle MLOps | Modélisation & API | Données, qualité & résultats |
| **Code** | Structure du repo, CI, Makefile, `config.py`, `src/data/` | `src/features/`, `src/models/`, `src/api/` | `tests/`, entraînement final, correctif MLflow |
| **Notebook** | — | `02_modeling.ipynb` | `01_eda.ipynb` |
| **Documentation** | README initial | — | README, `docs/` |
| **Pull Requests** | `feature/socle` | `feature/modeles` | `feature/eda-tests-doc`, `feature/resultats` |

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
