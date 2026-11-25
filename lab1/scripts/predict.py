"""Suggère des tags pour un titre, avec un modèle entraîné par train.py.

    python scripts/predict.py --model classic "How to declare a variable in python"
"""

from __future__ import annotations

import argparse

from train import ARTIFACTS_DIR, K, MODEL_NAMES, suggester_class


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("-k", type=int, default=K, help="nombre de tags suggérés")
    parser.add_argument("title", help="titre du post Stack Overflow")
    args = parser.parse_args()

    directory = ARTIFACTS_DIR / args.model
    if not directory.exists():
        parser.error(f"{directory} n'existe pas : lancez d'abord train.py --model {args.model}")

    model = suggester_class(args.model).load(directory)
    for rank, (tag, score) in enumerate(model.suggest([args.title], args.k)[0], start=1):
        print(f"{rank}. {tag}  {score:.3f}")


if __name__ == "__main__":
    main()
