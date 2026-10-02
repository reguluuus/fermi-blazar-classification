from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn
from torch.utils.data import DataLoader


def predict_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    threshold: float = 0.5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    model.eval()
    loss_fn = nn.BCEWithLogitsLoss(reduction="sum")
    probabilities: list[float] = []
    predictions: list[int] = []
    targets: list[int] = []
    total_loss = 0.0
    n_examples = 0

    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += loss_fn(logits, y).item()
            n_examples += y.numel()
            probs = torch.sigmoid(logits)

            probabilities.extend(probs.cpu().numpy().ravel().tolist())
            predictions.extend((probs >= threshold).int().cpu().numpy().ravel().tolist())
            targets.extend(y.int().cpu().numpy().ravel().tolist())

    return (
        np.asarray(targets),
        np.asarray(predictions),
        np.asarray(probabilities),
        total_loss / max(n_examples, 1),
    )


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    test_loss: float | None,
) -> dict[str, float | list[list[int]]]:
    metrics: dict[str, float | list[list[int]]] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier_score": float(brier_score_loss(y_true, y_prob)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }
    if test_loss is not None:
        metrics["test_loss"] = float(test_loss)
    return metrics


def find_best_f1_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    lower: float = 0.05,
    upper: float = 0.95,
    steps: int = 181,
) -> tuple[float, float]:
    """Choose a decision threshold on validation data only by maximizing F1."""
    thresholds = np.linspace(lower, upper, steps)
    scores = [
        f1_score(y_true, (y_prob >= threshold).astype(int), zero_division=0)
        for threshold in thresholds
    ]
    best_index = int(np.argmax(scores))
    return float(thresholds[best_index]), float(scores[best_index])


def predict_array(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    threshold: float = 0.5,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    tensor = torch.from_numpy(X.astype(np.float32))
    loader = DataLoader(tensor, batch_size=batch_size, shuffle=False)

    probs_all: list[np.ndarray] = []
    model.eval()
    with torch.inference_mode():
        for x in loader:
            probs = torch.sigmoid(model(x.to(device))).cpu().numpy().ravel()
            probs_all.append(probs)

    probabilities = np.concatenate(probs_all) if probs_all else np.array([])
    predictions = (probabilities >= threshold).astype(int)
    return predictions, probabilities
