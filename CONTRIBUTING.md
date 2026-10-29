# Règles de contribution

## Branches

| Branche | Rôle | Règle |
|---|---|---|
| `main` | Version stable, présentée | Protégée ; reçoit uniquement `develop` via PR |
| `develop` | Intégration | Protégée ; reçoit les `feature/*` via PR relue |
| `feature/<sujet>` | Une tâche | Créée depuis `develop` |

## Cycle d'une tâche

1. Une **issue** par tâche, assignée, dans le tableau GitHub Projects.
2. Branche depuis `develop` à jour :
   ```bash
   git switch develop && git pull
   git switch -c feature/preprocessing
   ```
3. **Commits petits et explicites** ([Conventional Commits](https://www.conventionalcommits.org/fr/)) :
   ```
   feat(data): nettoyage des labels du fichier de test
   fix(api): gestion des valeurs manquantes
   docs(eda): interprétation des variables catégorielles
   test(models): test du tuning sur données synthétiques
   ```
4. **Pull Request vers `develop`**, template rempli, `Closes #<issue>`.
5. **Relecture par un autre membre** + CI verte (ruff + pytest) avant le merge.
6. En fin de projet : PR `develop` → `main`, puis tag `v1.0`.

## Règles de code

- Le code réutilisable va dans `src/` ; les notebooks l'importent et racontent l'analyse.
- Chaque fonction a une docstring.
- Aucune valeur en dur : chemins, colonnes et graine aléatoire sont dans `src/config.py`.
- Pas de données ni de modèles entraînés dans Git (voir `.gitignore`).
- Avant de pousser : `make lint` et `make test`.
- Un notebook modifié est réexécuté entièrement (*Restart & Run All*) avant le commit.
