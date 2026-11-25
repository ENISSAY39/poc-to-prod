"""Métriques de suggestion : précision@k, rappel@k, F1 micro et F1 macro.

Les fonctions prennent des listes de tags (pas de scores) et ne dépendent d'aucun modèle.
`true_tags[i]` est la liste des tags réels du post i, `suggested[i]` la suggestion
classée pour ce post. Pour les F1, l'ensemble suggéré pour un post est son top-k.
"""

from __future__ import annotations

from collections.abc import Sequence

TagLists = Sequence[Sequence[str]]
Suggestion = list[tuple[str, float]]


def tags_only(suggestions: Sequence[Suggestion]) -> list[list[str]]:
    """Retire les scores des suggestions, pour les passer aux métriques."""
    return [[tag for tag, _ in s] for s in suggestions]


def _check_lengths(true_tags: TagLists, suggested: TagLists) -> None:
    if len(true_tags) != len(suggested):
        raise ValueError(
            f"{len(true_tags)} listes de tags réels pour {len(suggested)} suggestions"
        )
    if len(true_tags) == 0:
        raise ValueError("aucun post à évaluer")


def precision_at_k(true_tags: TagLists, suggested: TagLists, k: int = 3) -> float:
    """Moyenne sur les posts de |top-k ∩ tags réels| / k."""
    _check_lengths(true_tags, suggested)
    hits = (len(set(s[:k]) & set(t)) for t, s in zip(true_tags, suggested))
    return sum(h / k for h in hits) / len(true_tags)


def recall_at_k(true_tags: TagLists, suggested: TagLists, k: int = 3) -> float:
    """Moyenne sur les posts de |top-k ∩ tags réels| / |tags réels|."""
    _check_lengths(true_tags, suggested)
    if any(len(t) == 0 for t in true_tags):
        raise ValueError("un post sans tag réel n'a pas de rappel défini")
    hits = (len(set(s[:k]) & set(t)) / len(set(t)) for t, s in zip(true_tags, suggested))
    return sum(hits) / len(true_tags)


def _f1(tp: int, fp: int, fn: int) -> float:
    denominator = 2 * tp + fp + fn
    return 2 * tp / denominator if denominator else 0.0


def _counts_per_tag(
    true_tags: TagLists, suggested: TagLists, k: int
) -> dict[str, tuple[int, int, int]]:
    """Renvoie, pour chaque tag rencontré, ses (vrais positifs, faux positifs, faux négatifs)."""
    counts: dict[str, list[int]] = {}
    for t, s in zip(true_tags, suggested):
        truth, guessed = set(t), set(s[:k])
        for tag in truth | guessed:
            c = counts.setdefault(tag, [0, 0, 0])
            c[0] += tag in truth and tag in guessed
            c[1] += tag in guessed and tag not in truth
            c[2] += tag in truth and tag not in guessed
    return {tag: (c[0], c[1], c[2]) for tag, c in counts.items()}


def f1_micro(true_tags: TagLists, suggested: TagLists, k: int = 3) -> float:
    """F1 calculé sur les vrais positifs, faux positifs et faux négatifs cumulés."""
    _check_lengths(true_tags, suggested)
    counts = _counts_per_tag(true_tags, suggested, k).values()
    return _f1(*(sum(c[i] for c in counts) for i in range(3)))


def f1_macro(
    true_tags: TagLists, suggested: TagLists, vocabulary: Sequence[str], k: int = 3
) -> float:
    """Moyenne des F1 par tag, sur tous les tags du vocabulaire.

    Un tag du vocabulaire jamais réel ni suggéré compte pour un F1 de 0.
    """
    _check_lengths(true_tags, suggested)
    counts = _counts_per_tag(true_tags, suggested, k)
    return sum(_f1(*counts.get(tag, (0, 0, 0))) for tag in vocabulary) / len(vocabulary)


def evaluate(
    true_tags: TagLists, suggested: TagLists, vocabulary: Sequence[str], k: int = 3
) -> dict[str, float]:
    """Calcule les quatre métriques du lab sur le top-k."""
    return {
        f"precision@{k}": precision_at_k(true_tags, suggested, k),
        f"recall@{k}": recall_at_k(true_tags, suggested, k),
        "f1_micro": f1_micro(true_tags, suggested, k),
        "f1_macro": f1_macro(true_tags, suggested, vocabulary, k),
    }
