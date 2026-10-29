# MLops_Gangi_Bart_Daoud

Projet MLOps — Master 2 Data Science
Équipe : Gangi · Bart · Daoud

---

## Objectif

Prédire si une personne gagne **plus de 50 000 $ par an** à partir de ses caractéristiques
socio-démographiques, en suivant les bonnes pratiques MLOps : code versionné et testé,
expériences suivies dans MLflow, modèle servi par une API, CI automatisée.

- **Données** : [UCI Adult (Census Income)](https://archive.ics.uci.edu/dataset/2/adult),
  recensement américain de 1994, 48 842 individus, 14 variables.
- **Tâche** : classification binaire (`<=50K` / `>50K`), classes déséquilibrées (~76 % / 24 %).
- **Démarche** : statistiques descriptives → comparaison de 6 modèles → optimisation →
  choix argumenté → évaluation finale, interprétation et analyse d'équité → API.

## Résultats

Validation croisée stratifiée à 5 plis sur le train (32 561 lignes), puis évaluation unique
sur le jeu de test officiel (16 281 lignes). Scores de test au seuil 0,5.

| Modèle | ROC-AUC (CV) | ROC-AUC (test) | PR-AUC (test) | F1 (test) | Accuracy (test) |
|---|---|---|---|---|---|
| Baseline (classe majoritaire) | 0,500 | — | — | 0,000 | 0,759 (CV) |
| KNN (défaut) | 0,898 | — | — | — | — |
| Régression logistique | 0,898 | 0,897 | 0,728 | 0,664 | 0,800 |
| Arbre de décision | 0,907 | 0,905 | 0,761 | 0,640 | 0,858 |
| Random Forest | 0,920 | 0,917 | 0,806 | 0,681 | 0,867 |
| **Gradient Boosting** | **0,929** | **0,927** | **0,825** | **0,708** | **0,873** |

**Modèle retenu : Gradient Boosting** (`HistGradientBoostingClassifier`), avec un seuil de
décision de **0,42** optimisé pour le F1 : sur le test, **F1 = 0,718**, precision = 0,72,
recall = 0,71.

Pourquoi : meilleur sur toutes les métriques, très stable entre les plis (écart-type 0,0015),
même score en test qu'en validation croisée (pas de surapprentissage), entraînement en 2,5 s.
Les variables les plus importantes sont le statut marital, les gains en capital, l'âge et le
niveau d'études. Justification complète dans [`notebooks/02_modeling.ipynb`](notebooks/02_modeling.ipynb),
section 9.

**Équité** : le modèle détecte moins bien les femmes qui gagnent plus de 50K (recall 0,65 contre
0,72 pour les hommes) ; il reproduit les inégalités présentes dans les données de 1994.

## Installation

```bash
git clone https://github.com/fadid351/MLops_Gangi_Bart_Daoud.git
cd MLops_Gangi_Bart_Daoud
python -m venv .venv && source .venv/bin/activate   # Windows : .venv\Scripts\activate
make install                                        # ou : pip install -r requirements.txt
```

## Utilisation

Toutes les commandes se lancent depuis la racine du repo.

```bash
make data          # télécharge adult.data et adult.test dans data/raw/
make train         # comparaison, tuning, choix, évaluation -> models/ + reports/ + MLflow
make train-fast    # même chose en version rapide, pour vérifier que tout tourne
make mlflow-ui     # interface MLflow : http://localhost:5000
make serve         # API : http://localhost:8000/docs
make test          # tests + couverture
make lint          # vérification du style (ruff)
```

Analyse détaillée et commentée dans les notebooks :
- [`01_eda.ipynb`](notebooks/01_eda.ipynb) : statistiques descriptives et décisions de preprocessing ;
- [`02_modeling.ipynb`](notebooks/02_modeling.ipynb) : comparaison, tuning, choix, test, interprétation, équité.

### Exemple d'appel à l'API

```bash
curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" -d '{
  "age": 45, "workclass": "Private", "education-num": 13,
  "marital-status": "Married-civ-spouse", "occupation": "Exec-managerial",
  "relationship": "Husband", "race": "White", "sex": "Male",
  "capital-gain": 5000, "capital-loss": 0, "hours-per-week": 50,
  "native-country": "United-States"}'
# -> {"probabilite_50k": 0.93, "prediction": ">50K", "seuil": 0.38}
```

Routes : `GET /health`, `GET /model-info`, `POST /predict`, `POST /predict/batch`.

## Structure du projet

```
MLops_Gangi_Bart_Daoud/
├── data/
│   ├── raw/                  # adult.data, adult.test (non versionnés)
│   └── processed/
├── models/                   # model.joblib + model_meta.json (générés par make train)
├── notebooks/                # 01_eda, 02_modeling
├── reports/                  # résultats (CSV, summary.json) et figures
├── scripts/download_data.py
├── src/
│   ├── config.py             # chemins, colonnes, graine aléatoire
│   ├── data/
│   │   ├── load_data.py      # téléchargement, chargement, nettoyage
│   │   └── eda.py            # fonctions de statistiques descriptives
│   ├── features/
│   │   └── build_features.py # pipeline de preprocessing
│   ├── models/
│   │   ├── models.py         # modèles candidats, grilles, tuning
│   │   ├── evaluate.py       # métriques, CV, seuil, équité, graphiques
│   │   └── train.py          # pipeline complet + suivi MLflow
│   └── api/
│       ├── predictor.py      # logique de prédiction
│       └── main.py           # API FastAPI
├── tests/                    # pytest, sur données synthétiques (pas de téléchargement)
├── docs/organisation.md      # répartition du travail
├── .github/                  # CI, templates de PR et d'issues
├── requirements.txt
├── Makefile
└── README.md
```

## Méthodologie

**Préparation** (justifiée par l'EDA) :
- valeurs manquantes (`?`) imputées par la modalité la plus fréquente ;
- `capital-gain` / `capital-loss` transformées en log(1+x) ;
- `fnlwgt` (poids de sondage) et `education` (redondante avec `education-num`) supprimées ;
- one-hot encoding avec regroupement des modalités rares ; standardisation.

Tout le preprocessing est dans un `Pipeline` scikit-learn réappris à chaque pli :
**aucune fuite de données**, et l'API reçoit les données brutes.

**Évaluation** :
- découpage officiel train (32 561) / test (16 281) ; le test n'est utilisé qu'une fois ;
- validation croisée stratifiée à 5 plis ;
- métrique principale **ROC-AUC**, complétée par **F1** et **PR-AUC**
  (l'accuracy est trompeuse avec 76 % de `<=50K`) ;
- seuil de décision optimisé pour le F1 sur des prédictions hors échantillon du train.

**Choix du modèle** : performance, stabilité entre les plis, temps de calcul, interprétabilité.

**Suivi MLflow** : un run par modèle testé (paramètres, scores de CV), et un run final
(métriques de test, figures, modèle).

**Limites** : données de 1994 ; le modèle reproduit des inégalités selon le sexe et l'origine
(voir l'analyse d'équité) ; seuil à adapter à l'usage.

## Workflow Git

- `main` — branche stable, protégée
- `develop` — branche d'intégration
- `feature/<nom>` — une branche par tâche, PR vers `develop` relue par un autre membre

Détails : [`CONTRIBUTING.md`](CONTRIBUTING.md) et [`docs/organisation.md`](docs/organisation.md).

## Membres

| Membre | GitHub | Rôle principal |
|--------|--------|----------------|
| Daoud  | @fadid351 | Socle MLOps : structure du repo, CI, chargement des données |
| Gangi  | @silviogng | Statistiques descriptives, tests, documentation, entraînement final |
| Bart   | @frantrab08 | Preprocessing, modèles, suivi MLflow, API |
