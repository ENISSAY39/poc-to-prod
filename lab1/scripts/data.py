"""Données du lab 1 : lecture du CSV, regroupement par post, vocabulaire de tags, split.

Ce module ne dépend d'aucun modèle : les modèles classique et BERT l'utilisent tel quel.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DEFAULT_CSV = Path(__file__).resolve().parent.parent / "stackoverflow_posts.csv"
DEFAULT_VOCABULARY_SIZE = 50
DEFAULT_SEED = 42


@dataclass
class Dataset:
    """Posts restreints au vocabulaire de tags et découpés en train / validation / test.

    Chaque DataFrame a les colonnes `post_id`, `title` et `tags` (liste de tags du
    vocabulaire, dans l'ordre de `tag_position`).
    """

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    vocabulary: list[str]
    coverage: float


def load_posts(path: str | Path = DEFAULT_CSV) -> pd.DataFrame:
    """Lit le CSV au format long et renvoie un post par ligne : `post_id`, `title`, `tags`.

    `keep_default_na=False` est indispensable : sinon pandas lit le tag `null` comme NaN.
    """
    rows = pd.read_csv(path, keep_default_na=False)
    rows = rows.sort_values(["post_id", "tag_position"])
    posts = rows.groupby("post_id", sort=True).agg(
        title=("title", "first"),
        tags=("tag_name", list),
    )
    return posts.reset_index()


def build_vocabulary(posts: pd.DataFrame, size: int = DEFAULT_VOCABULARY_SIZE) -> list[str]:
    """Renvoie les `size` tags portés par le plus de posts, du plus fréquent au moins fréquent.

    Les ex aequo sont départagés par ordre alphabétique, pour un résultat déterministe.
    """
    counts = Counter(tag for tags in posts["tags"] for tag in set(tags))
    ranked = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
    return [tag for tag, _ in ranked[:size]]


def restrict_to_vocabulary(posts: pd.DataFrame, vocabulary: list[str]) -> pd.DataFrame:
    """Retire des posts les tags hors vocabulaire, puis exclut les posts qui n'en ont plus."""
    known = set(vocabulary)
    restricted = posts.assign(tags=[[t for t in tags if t in known] for tags in posts["tags"]])
    return restricted[restricted["tags"].map(len) > 0].reset_index(drop=True)


def split_posts(
    posts: pd.DataFrame,
    seed: int = DEFAULT_SEED,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Découpe les posts en train / validation / test (80/10/10 par défaut).

    Le split se fait par post : un même post ne peut être que dans une seule partie.
    La même graine donne toujours le même découpage.
    """
    order = np.random.default_rng(seed).permutation(len(posts))
    n_train = int(len(posts) * train_ratio)
    n_val = int(len(posts) * val_ratio)
    parts = np.split(order, [n_train, n_train + n_val])
    train, val, test = (posts.iloc[idx].reset_index(drop=True) for idx in parts)
    return train, val, test


def load_dataset(
    path: str | Path = DEFAULT_CSV,
    vocabulary_size: int = DEFAULT_VOCABULARY_SIZE,
    seed: int = DEFAULT_SEED,
) -> Dataset:
    """Enchaîne lecture, vocabulaire, restriction et split.

    `coverage` est la part des posts qui gardent au moins un tag du vocabulaire.
    """
    posts = load_posts(path)
    vocabulary = build_vocabulary(posts, vocabulary_size)
    kept = restrict_to_vocabulary(posts, vocabulary)
    train, val, test = split_posts(kept, seed)
    return Dataset(train, val, test, vocabulary, coverage=len(kept) / len(posts))
