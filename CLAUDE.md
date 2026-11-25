## Projet

Repo du cours *poc-to-prod* : un lab par dossier (`lab1/`, …). Vocabulaire du domaine dans `GLOSSARY.md`, décisions dans `docs/adr/`.

### Lab 1 — Suggestion de tags Stack Overflow

Spécification complète : `lab1/task`. EDA : `lab1/eda.ipynb`.

- **But** : pour le **titre** d'une question Stack Overflow, suggérer les **3 tags** les plus probables, avec leur score.
- **Données** : `lab1/stackoverflow_posts.csv`, format long (une ligne par post et par tag), à regrouper par `post_id`. Toujours lire avec `keep_default_na=False`, car le tag `null` existe.
- **Formulation** : multi-label, vocabulaire = **top-50 tags** (paramétrable). Les posts sans tag dans le vocabulaire sont exclus. Split 80/10/10 par `post_id`, même graine pour les deux modèles.
- **Deux modèles** :
  - classique : TF-IDF + régression logistique one-vs-rest (ADR-0001) ;
  - BERT : `bert-base-uncased` fine-tuné (ADR-0002).
- **Métriques** : précision@3, rappel@3, F1 micro et F1 macro, sur le même test set.
- **Code** : des scripts dans `lab1/scripts/` (pas de notebook pour la modélisation).
  - `data.py`, `metrics.py` : modules partagés.
  - `classic.py`, `bert.py` : les deux modèles, avec la même interface `fit` / `suggest(titles, k)`.
  - `python scripts/train.py --model {classic,bert}` : entraînement, évaluation, sauvegarde dans `lab1/artifacts/<model>/` (non versionné).
  - `python scripts/predict.py --model {classic,bert} "<titre>"` : suggestion.
- **Tests** : `pytest` dans `lab1/tests/`, sur `data.py` et `metrics.py`. **Écrits par l'utilisateur** : ne pas les écrire à sa place.

### Environnement

`.venv/` à la racine (Python 3.12), dépendances dans `requirements.txt`. GPU : RTX 4060, 8 Go (CUDA).

## Agent skills

### Issue tracker

Issues are tracked in GitHub Issues on ENISSAY39/poc-to-prod (via `gh`). See `docs/agents/issue-tracker.md`.

### Triage labels

Default canonical labels: needs-triage, needs-info, ready-for-agent, ready-for-human, wontfix. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `GLOSSARY.md` + `docs/adr/` at the repo root. See `docs/agents/domain.md`.
