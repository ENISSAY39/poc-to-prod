"""Modèle classique : TF-IDF (mots + n-grammes de caractères) et régression logistique
one-vs-rest. Voir docs/adr/0001-modele-classique-tfidf-logreg.md.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import MultiLabelBinarizer

import metrics
from metrics import Suggestion

MODEL_FILE = "model.joblib"

# Un token est une suite de lettres, chiffres, `#`, `+`, `.` et `-` qui ne finit ni par
# `.` ni par `-` : on garde `c#`, `c++`, `.net` et `asp.net-mvc`, mais `python.` en fin
# de phrase donne `python`.
WORD_TOKEN_PATTERN = r"[\w#+.\-]*[\w#+]"


def _pipeline(C: float) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "words",
                TfidfVectorizer(
                    token_pattern=WORD_TOKEN_PATTERN, ngram_range=(1, 2), sublinear_tf=True
                ),
            ),
            (
                "chars",
                TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True),
            ),
        ]
    )
    classifier = OneVsRestClassifier(LogisticRegression(C=C, max_iter=1000))
    return Pipeline([("features", features), ("classifier", classifier)])


class ClassicSuggester:
    """Suggère les tags d'un titre, triés par probabilité de la régression logistique."""

    def __init__(self, vocabulary: Sequence[str], C_grid: Sequence[float] = (0.1, 1, 10)):
        self.vocabulary = list(vocabulary)
        self.C_grid = list(C_grid)
        self.C: float | None = None
        self.val_f1_micro_by_C: dict[float, float] = {}
        self._binarizer = MultiLabelBinarizer(classes=self.vocabulary)
        self._pipeline: Pipeline | None = None

    def fit(
        self,
        train_titles: Sequence[str],
        train_tags: Sequence[Sequence[str]],
        val_titles: Sequence[str],
        val_tags: Sequence[Sequence[str]],
        k: int = 3,
    ) -> ClassicSuggester:
        """Entraîne un modèle par valeur de C et garde celui au meilleur F1 micro de validation."""
        y_train = self._binarizer.fit_transform(train_tags)
        best_score = -1.0
        for C in self.C_grid:
            self._pipeline = _pipeline(C).fit(list(train_titles), y_train)
            suggested = metrics.tags_only(self.suggest(val_titles, k))
            score = metrics.f1_micro(val_tags, suggested, k)
            self.val_f1_micro_by_C[C] = score
            if score > best_score:
                best_score, best_C, best_pipeline = score, C, self._pipeline
        self.C, self._pipeline = best_C, best_pipeline
        return self

    def suggest(self, titles: Sequence[str], k: int = 3) -> list[Suggestion]:
        """Renvoie, pour chaque titre, les k tags les plus probables avec leur probabilité."""
        if self._pipeline is None:
            raise RuntimeError("le modèle n'est pas entraîné")
        probabilities = self._pipeline.predict_proba(list(titles))
        top = np.argsort(-probabilities, axis=1, kind="stable")[:, :k]
        return [
            [(self.vocabulary[j], float(row[j])) for j in indices]
            for row, indices in zip(probabilities, top)
        ]

    def summary(self) -> dict:
        """Hyperparamètres retenus, pour metrics.json."""
        return {
            "C": self.C,
            "val_f1_micro_by_C": {str(C): s for C, s in self.val_f1_micro_by_C.items()},
        }

    def save(self, directory: str | Path) -> None:
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, directory / MODEL_FILE)

    @classmethod
    def load(cls, directory: str | Path) -> ClassicSuggester:
        model = joblib.load(Path(directory) / MODEL_FILE)
        if not isinstance(model, cls):
            raise TypeError(f"{directory} ne contient pas un {cls.__name__}")
        return model
