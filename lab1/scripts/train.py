"""Entraîne un suggesteur, l'évalue sur le test et le sauvegarde dans artifacts/<model>/.

    python scripts/train.py --model classic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import data
import metrics

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
K = 3


def _classic():
    from classic import ClassicSuggester

    return ClassicSuggester


def _bert():
    raise NotImplementedError("le modèle BERT n'est pas encore implémenté (ticket #2)")


# Nom du modèle -> fonction qui renvoie sa classe. L'import est paresseux : BERT tire torch.
SUGGESTERS = {"classic": _classic, "bert": _bert}
MODEL_NAMES = tuple(SUGGESTERS)


def suggester_class(name: str):
    """Renvoie la classe du suggesteur `name`."""
    return SUGGESTERS[name]()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--data", type=Path, default=data.DEFAULT_CSV)
    parser.add_argument("--vocabulary-size", type=int, default=data.DEFAULT_VOCABULARY_SIZE)
    parser.add_argument("--seed", type=int, default=data.DEFAULT_SEED)
    args = parser.parse_args()

    dataset = data.load_dataset(args.data, args.vocabulary_size, args.seed)
    print(
        f"{len(dataset.vocabulary)} tags, couverture {dataset.coverage:.1%}, "
        f"train/val/test = {len(dataset.train)}/{len(dataset.val)}/{len(dataset.test)} posts"
    )

    model = suggester_class(args.model)(dataset.vocabulary)
    model.fit(
        dataset.train["title"].tolist(),
        dataset.train["tags"].tolist(),
        dataset.val["title"].tolist(),
        dataset.val["tags"].tolist(),
    )

    suggested = metrics.tags_only(model.suggest(dataset.test["title"].tolist(), K))
    results = {
        **metrics.evaluate(dataset.test["tags"].tolist(), suggested, dataset.vocabulary, K),
        **model.summary(),
        "coverage": dataset.coverage,
        "vocabulary_size": len(dataset.vocabulary),
        "seed": args.seed,
        "n_train": len(dataset.train),
        "n_val": len(dataset.val),
        "n_test": len(dataset.test),
    }

    output = ARTIFACTS_DIR / args.model
    model.save(output)
    (output / "tags.json").write_text(json.dumps(dataset.vocabulary, indent=2), encoding="utf-8")
    (output / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    for name, value in results.items():
        print(f"  {name}: {value}")
    print(f"Sauvegardé dans {output}")


if __name__ == "__main__":
    main()
