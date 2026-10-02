from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    PrecisionRecallDisplay,
    RocCurveDisplay,
)


def plot_losses(train_loss: list[float], val_loss: list[float], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(7, 5))
    plt.plot(range(1, len(train_loss) + 1), train_loss, label="Training loss")
    plt.plot(range(1, len(val_loss) + 1), val_loss, label="Validation loss")
    plt.xlabel("Epoch")
    plt.ylabel("BCE loss")
    plt.title("Training history")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["FSRQ", "BLL"])
    disp.plot(values_format="d")
    disp.ax_.set_title("Confusion matrix — test set")
    disp.figure_.tight_layout()
    disp.figure_.savefig(path, dpi=200)
    plt.close(disp.figure_)


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    RocCurveDisplay.from_predictions(y_true, y_prob, name="MLP", ax=ax)
    ax.plot([0, 1], [0, 1], linestyle="--", label="Random classifier")
    ax.set_title("ROC curve — test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_precision_recall_curve(y_true: np.ndarray, y_prob: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 5))
    PrecisionRecallDisplay.from_predictions(y_true, y_prob, name="MLP", ax=ax)
    ax.set_title("Precision–recall curve — test set")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_calibration(y_true: np.ndarray, y_prob: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    prob_true, prob_pred = calibration_curve(
        y_true,
        y_prob,
        n_bins=10,
        strategy="quantile",
    )
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot(prob_pred, prob_true, marker="o", label="MLP")
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability of BLL")
    ax.set_ylabel("Observed BLL fraction")
    ax.set_title("Probability calibration — test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_model_comparison(
    metrics_by_model: dict[str, dict[str, float | list[list[int]]]],
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    metric_names = ["f1", "roc_auc", "pr_auc", "balanced_accuracy"]
    model_names = list(metrics_by_model)
    x = np.arange(len(metric_names))
    width = 0.8 / max(len(model_names), 1)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    for i, model_name in enumerate(model_names):
        values = [float(metrics_by_model[model_name][metric]) for metric in metric_names]
        ax.bar(x + (i - (len(model_names) - 1) / 2) * width, values, width, label=model_name)

    ax.set_xticks(x)
    ax.set_xticklabels(["F1", "ROC-AUC", "PR-AUC", "Balanced accuracy"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model comparison — held-out test set")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_class_distribution(
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    labels = ["Train", "Validation", "Test"]
    fsrq = [int((y == 0).sum()) for y in (y_train, y_val, y_test)]
    bll = [int((y == 1).sum()) for y in (y_train, y_val, y_test)]
    x = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x, fsrq, label="FSRQ")
    ax.bar(x, bll, bottom=fsrq, label="BLL")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Number of sources")
    ax.set_title("Class distribution after stratified split")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def plot_feature_importance(
    model,
    X: np.ndarray,
    feature_names: list[str],
    device,
    path: Path,
    max_samples: int = 256,
    top_k: int = 20,
) -> None:
    """Gradient-based global feature importance on a validation sample."""
    import torch

    if len(X) == 0:
        return
    sample = torch.from_numpy(X[:max_samples].astype(np.float32)).to(device)
    sample.requires_grad_(True)

    model.eval()
    output = model(sample).sum()
    model.zero_grad(set_to_none=True)
    output.backward()

    importance = sample.grad.detach().abs().mean(dim=0).cpu().numpy()
    order = np.argsort(importance)[::-1][:top_k]
    ordered_names = [feature_names[i] for i in order][::-1]
    ordered_values = importance[order][::-1]

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(np.arange(len(order)), ordered_values)
    ax.set_yticks(np.arange(len(order)))
    ax.set_yticklabels(ordered_names, fontsize=8)
    ax.set_xlabel("Mean absolute input gradient")
    ax.set_title(f"Top {len(order)} gradient-based feature importances")
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
