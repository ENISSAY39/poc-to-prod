# Modèle BERT : fine-tuning complet de bert-base-uncased avec une tête multi-label

Le second modèle doit être « un BERT » et aller plus loin que le modèle classique, surtout sur les titres où le tag n'apparaît pas littéralement. C'est le cas d'environ un tiers des posts (ex. `UITableView` → `iphone`). On fine-tune entièrement `bert-base-uncased` avec une tête de classification à 50 sorties, une sigmoïde et une loss BCE. `max_length` vaut 64, car les titres font au plus 28 mots. Les autres réglages sont lr 2e-5, batch 32 et 5 epochs au maximum, en gardant la meilleure epoch sur la validation. L'entraînement tient sur la RTX 4060 (8 Go) de la machine de développement.

## Considered Options

- **`distilbert-base-uncased`** : environ 2 fois plus rapide, mais ce n'est pas le BERT de référence demandé. Le GPU disponible rend le gain de vitesse inutile.
- **BERT spécialisé code / Stack Overflow (ex. BERTOverflow)** : vocabulaire plus adapté, mais modèle moins standard et moins bien maintenu. Ça brouille la comparaison « classique vs BERT ».
- **BERT figé + classifieur sur les embeddings `[CLS]`** : moins cher, mais nettement moins bon que le fine-tuning en classification, et ça perd l'intérêt de la comparaison.

## Consequences

Le modèle sauvegardé (`save_pretrained`, environ 400 Mo) va dans `lab1/artifacts/bert/`, qui n'est pas versionné. `predict.py` doit donc avoir lancé `train.py --model bert` au préalable sur la machine. Le tokenizer `uncased` met les titres en minuscules, ce qui fait disparaître la casse (`Java` / `java`), sans effet sur des tags qui sont tous en minuscules.
