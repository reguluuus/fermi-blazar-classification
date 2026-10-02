from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from .evaluation import classification_metrics


@dataclass
class BaselineResult:
    name: str
    metrics: dict[str, float | list[list[int]]]
    predictions: np.ndarray
    probabilities: np.ndarray


def evaluate_baselines(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    random_state: int = 42,
) -> list[BaselineResult]:
    """Train lightweight sklearn baselines on the same split as the neural network."""
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=500,
            class_weight="balanced",
            n_jobs=-1,
            random_state=random_state,
        ),
    }

    results: list[BaselineResult] = []
    y_train_int = np.asarray(y_train).astype(int)
    y_test_int = np.asarray(y_test).astype(int)

    for name, model in models.items():
        model.fit(X_train, y_train_int)
        probabilities = model.predict_proba(X_test)[:, 1]
        predictions = (probabilities >= 0.5).astype(int)
        metrics = classification_metrics(
            y_test_int,
            predictions,
            probabilities,
            test_loss=None,
        )
        results.append(BaselineResult(name, metrics, predictions, probabilities))

    return results
