import numpy as np

from agn_classifier.evaluation import classification_metrics, find_best_f1_threshold


def test_classification_metrics_perfect_predictions():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.8, 0.9])
    y_pred = (y_prob >= 0.5).astype(int)

    metrics = classification_metrics(y_true, y_pred, y_prob, test_loss=0.1)

    assert metrics["accuracy"] == 1.0
    assert metrics["f1"] == 1.0
    assert metrics["roc_auc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["confusion_matrix"] == [[2, 0], [0, 2]]


def test_threshold_search_returns_valid_threshold():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.45, 0.9])
    threshold, score = find_best_f1_threshold(y_true, y_prob)

    assert 0.05 <= threshold <= 0.95
    assert 0.0 <= score <= 1.0
