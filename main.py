from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch

from agn_classifier.baselines import evaluate_baselines
from agn_classifier.data import (
    TARGET_COLUMN,
    fit_preprocessor,
    load_catalog,
    make_splits,
    select_numeric_features,
)
from agn_classifier.evaluation import (
    classification_metrics,
    find_best_f1_threshold,
    predict_array,
    predict_loader,
)
from agn_classifier.model import BlazarClassifier
from agn_classifier.plots import (
    plot_calibration,
    plot_class_distribution,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_losses,
    plot_model_comparison,
    plot_precision_recall_curve,
    plot_roc_curve,
)
from agn_classifier.training import make_loader, set_seed, train_model


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def train(args: argparse.Namespace) -> None:
    # Load tuned hyperparameters if provided.
    if args.params is not None:
        params_path = Path(args.params)

        if not params_path.exists():
            raise FileNotFoundError(
                f"Hyperparameter file not found: {params_path}"
            )

        with params_path.open("r", encoding="utf-8") as f:
            tuned_params = json.load(f)

        args.hidden_size = tuned_params.get(
            "hidden_size",
            args.hidden_size,
        )
        args.dropout = tuned_params.get(
            "dropout",
            args.dropout,
        )
        args.learning_rate = tuned_params.get(
            "learning_rate",
            args.learning_rate,
        )
        args.batch_size = tuned_params.get(
            "batch_size",
            args.batch_size,
        )

        print("Loaded tuned hyperparameters:")
        print(f"  hidden_size   = {args.hidden_size}")
        print(f"  dropout       = {args.dropout}")
        print(f"  learning_rate = {args.learning_rate}")
        print(f"  batch_size    = {args.batch_size}")
        print()

    set_seed(args.seed)

    device = resolve_device(args.device)
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # Load and prepare data
    # ------------------------------------------------------------------
    catalog = load_catalog(args.catalog)
    data = select_numeric_features(catalog)

    splits = make_splits(
        data,
        random_state=args.seed,
    )

    # Fit preprocessing ONLY on the training set to avoid data leakage.
    preprocessor = fit_preprocessor(splits.X_train)

    X_train = preprocessor.transform(splits.X_train)
    X_val = preprocessor.transform(splits.X_val)
    X_test = preprocessor.transform(splits.X_test)

    y_train = splits.y_train.to_numpy(dtype=np.float32)
    y_val = splits.y_val.to_numpy(dtype=np.float32)
    y_test = splits.y_test.to_numpy(dtype=np.float32)

    print(
        f"Dataset: "
        f"train={len(X_train)}, "
        f"validation={len(X_val)}, "
        f"test={len(X_test)}, "
        f"BCU={len(splits.bcu)}"
    )
    print(f"Number of features: {X_train.shape[1]}")

    # ------------------------------------------------------------------
    # DataLoaders
    # ------------------------------------------------------------------
    train_loader = make_loader(
        X_train,
        y_train,
        args.batch_size,
        shuffle=True,
    )

    val_loader = make_loader(
        X_val,
        y_val,
        args.batch_size,
        shuffle=False,
    )

    test_loader = make_loader(
        X_test,
        y_test,
        args.batch_size,
        shuffle=False,
    )

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------
    model = BlazarClassifier(
        input_features=X_train.shape[1],
        hidden_size=args.hidden_size,
        dropout=args.dropout,
    ).to(device)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
    )

    # ------------------------------------------------------------------
    # Select threshold using validation data only
    # ------------------------------------------------------------------
    y_val_true, _, y_val_prob, _ = predict_loader(
        model,
        val_loader,
        device,
    )

    if args.threshold is None:
        threshold, validation_f1 = find_best_f1_threshold(
            y_val_true,
            y_val_prob,
        )

        print(
            f"Selected validation threshold="
            f"{threshold:.3f} "
            f"(F1={validation_f1:.4f})"
        )
    else:
        threshold = args.threshold

        print(
            f"Using fixed classification threshold="
            f"{threshold:.3f}"
        )

    # ------------------------------------------------------------------
    # Final test evaluation
    # ------------------------------------------------------------------
    y_true, y_pred, y_prob, test_loss = predict_loader(
        model,
        test_loader,
        device,
        threshold=threshold,
    )

    metrics = classification_metrics(
        y_true,
        y_pred,
        y_prob,
        test_loss,
    )

    metrics["decision_threshold"] = float(threshold)

    # ------------------------------------------------------------------
    # Baseline models
    # ------------------------------------------------------------------
    baseline_results = evaluate_baselines(
        X_train,
        y_train,
        X_test,
        y_test,
        random_state=args.seed,
    )

    comparison_metrics = {
        "PyTorch MLP": metrics,
    }

    for baseline in baseline_results:
        comparison_metrics[baseline.name] = baseline.metrics

    # ------------------------------------------------------------------
    # Output directories
    # ------------------------------------------------------------------
    model_dir = Path(args.model_dir)
    results_dir = Path(args.results_dir)

    model_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # Save trained neural network
    # ------------------------------------------------------------------
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_features": X_train.shape[1],
            "hidden_size": args.hidden_size,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "threshold": threshold,
            "class_mapping": {
                "0": "FSRQ",
                "1": "BLL",
            },
        },
        model_dir / "blazar_classifier.pt",
    )

    # Save preprocessing pipeline.
    joblib.dump(
        preprocessor,
        model_dir / "preprocessor.joblib",
    )

    # ------------------------------------------------------------------
    # Save metrics
    # ------------------------------------------------------------------
    with (
        results_dir / "metrics.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(
            metrics,
            f,
            indent=2,
        )

    with (
        results_dir / "model_comparison.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(
            comparison_metrics,
            f,
            indent=2,
        )

    comparison_rows = []

    for model_name, model_metrics in comparison_metrics.items():
        row = {
            "model": model_name,
        }

        for key, value in model_metrics.items():
            if (
                key != "confusion_matrix"
                and isinstance(value, (int, float))
            ):
                row[key] = value

        comparison_rows.append(row)

    pd.DataFrame(
        comparison_rows
    ).to_csv(
        results_dir / "model_comparison.csv",
        index=False,
    )

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    plot_losses(
        history.train_loss,
        history.val_loss,
        results_dir / "loss_curve.png",
    )

    plot_confusion_matrix(
        np.asarray(
            metrics["confusion_matrix"]
        ),
        results_dir / "confusion_matrix.png",
    )

    plot_feature_importance(
        model,
        X_val,
        preprocessor.feature_names,
        device,
        results_dir / "feature_importance.png",
    )

    plot_roc_curve(
        y_true,
        y_prob,
        results_dir / "roc_curve.png",
    )

    plot_precision_recall_curve(
        y_true,
        y_prob,
        results_dir / "precision_recall_curve.png",
    )

    plot_calibration(
        y_true,
        y_prob,
        results_dir / "calibration_curve.png",
    )

    plot_model_comparison(
        comparison_metrics,
        results_dir / "model_comparison.png",
    )

    plot_class_distribution(
        y_train.astype(int),
        y_val.astype(int),
        y_test.astype(int),
        results_dir / "class_distribution.png",
    )

    # ------------------------------------------------------------------
    # Run summary
    # ------------------------------------------------------------------
    split_summary = {
        "train_samples": len(X_train),
        "validation_samples": len(X_val),
        "test_samples": len(X_test),
        "bcu_samples": len(splits.bcu),
        "features": len(preprocessor.feature_names),
        "device": str(device),
        "decision_threshold": float(threshold),
        "class_mapping": {
            "0": "FSRQ",
            "1": "BLL",
        },
        "training_parameters": {
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "hidden_size": args.hidden_size,
            "dropout": args.dropout,
            "learning_rate": args.learning_rate,
            "patience": args.patience,
            "seed": args.seed,
        },
    }

    with (
        results_dir / "run_summary.json"
    ).open("w", encoding="utf-8") as f:
        json.dump(
            split_summary,
            f,
            indent=2,
        )

    # ------------------------------------------------------------------
    # Console output
    # ------------------------------------------------------------------
    print("\nTest metrics")

    for key in [
        "test_loss",
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "mcc",
        "roc_auc",
        "pr_auc",
        "brier_score",
    ]:
        print(
            f"  {key:>18}: "
            f"{metrics[key]:.4f}"
        )

    print("\nBaseline comparison")

    for baseline in baseline_results:
        print(
            f"  {baseline.name:<20} "
            f"F1={baseline.metrics['f1']:.4f} | "
            f"ROC-AUC={baseline.metrics['roc_auc']:.4f} | "
            f"PR-AUC={baseline.metrics['pr_auc']:.4f}"
        )

    print(
        f"\nSaved model to "
        f"{model_dir / 'blazar_classifier.pt'}"
    )


def classify_bcu(
    args: argparse.Namespace,
) -> None:
    device = resolve_device(
        args.device
    )

    model_dir = Path(
        args.model_dir
    )

    # ------------------------------------------------------------------
    # Load trained model
    # ------------------------------------------------------------------
    checkpoint = torch.load(
        model_dir / "blazar_classifier.pt",
        map_location=device,
        weights_only=False,
    )

    preprocessor = joblib.load(
        model_dir / "preprocessor.joblib"
    )

    model = BlazarClassifier(
        input_features=checkpoint["input_features"],
        hidden_size=checkpoint["hidden_size"],
        dropout=checkpoint["dropout"],
    ).to(device)

    model.load_state_dict(
        checkpoint["state_dict"]
    )

    # ------------------------------------------------------------------
    # Load BCU sources
    # ------------------------------------------------------------------
    catalog = load_catalog(
        args.catalog
    )

    data = select_numeric_features(
        catalog
    )

    bcu = data[
        data[TARGET_COLUMN] == "BCU"
    ].copy()

    if bcu.empty:
        raise RuntimeError(
            "No BCU objects were found in the catalog."
        )

    X_bcu = preprocessor.transform(
        bcu.drop(
            columns=[TARGET_COLUMN]
        )
    )

    # Use the threshold saved during training unless overridden.
    if args.threshold is not None:
        threshold = args.threshold
    else:
        threshold = checkpoint.get(
            "threshold",
            0.5,
        )

    pred, prob_bll = predict_array(
        model,
        X_bcu,
        device,
        threshold=threshold,
    )

    # ------------------------------------------------------------------
    # Preserve original catalog metadata
    # ------------------------------------------------------------------
    original_bcu = catalog[
        catalog[TARGET_COLUMN] == "BCU"
    ].copy().reset_index(drop=True)

    original_bcu["predicted_class"] = np.where(
        pred == 1,
        "BLL",
        "FSRQ",
    )

    original_bcu["probability_BLL"] = (
        prob_bll
    )

    original_bcu["probability_FSRQ"] = (
        1.0 - prob_bll
    )

    # ------------------------------------------------------------------
    # Save predictions
    # ------------------------------------------------------------------
    output = Path(
        args.output
    )

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    original_bcu.to_csv(
        output,
        index=False,
    )

    counts = (
        original_bcu[
            "predicted_class"
        ]
        .value_counts()
        .to_dict()
    )

    print(
        f"Classified "
        f"{len(original_bcu)} "
        f"BCU sources: {counts}"
    )

    print(
        f"Decision threshold: "
        f"{threshold:.3f}"
    )

    print(
        f"Saved predictions to "
        f"{output}"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Classify Fermi-LAT blazars "
            "as BL Lac (BLL) or FSRQ."
        )
    )

    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )

    # ==================================================================
    # TRAIN COMMAND
    # ==================================================================
    train_parser = subparsers.add_parser(
        "train",
        help="Train and evaluate the classifier",
    )

    train_parser.add_argument(
        "--catalog",
        type=Path,
        required=True,
        help="Path to gll_psc_v31.fit",
    )

    train_parser.add_argument(
        "--params",
        type=Path,
        default=None,
        help=(
            "JSON file containing tuned hyperparameters, "
            "for example results/best_params.json"
        ),
    )

    train_parser.add_argument(
        "--epochs",
        type=int,
        default=40,
    )

    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
    )

    train_parser.add_argument(
        "--hidden-size",
        type=int,
        default=128,
    )

    train_parser.add_argument(
        "--dropout",
        type=float,
        default=0.15,
    )

    train_parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )

    train_parser.add_argument(
        "--patience",
        type=int,
        default=8,
    )

    train_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help=(
            "Fixed classification threshold. "
            "If omitted, maximize F1 on validation data."
        ),
    )

    train_parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    train_parser.add_argument(
        "--device",
        default="auto",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
    )

    train_parser.add_argument(
        "--model-dir",
        default="models",
    )

    train_parser.add_argument(
        "--results-dir",
        default="results",
    )

    train_parser.set_defaults(
        func=train
    )

    # ==================================================================
    # BCU PREDICTION COMMAND
    # ==================================================================
    predict_parser = subparsers.add_parser(
        "predict-bcu",
        help="Classify BCU sources",
    )

    predict_parser.add_argument(
        "--catalog",
        type=Path,
        required=True,
    )

    predict_parser.add_argument(
        "--model-dir",
        default="models",
    )

    predict_parser.add_argument(
        "--output",
        default="results/classified_bcu.csv",
    )

    predict_parser.add_argument(
        "--threshold",
        type=float,
        default=None,
    )

    predict_parser.add_argument(
        "--device",
        default="auto",
        choices=[
            "auto",
            "cpu",
            "cuda",
        ],
    )

    predict_parser.set_defaults(
        func=classify_bcu
    )

    return parser


def main() -> None:
    parser = build_parser()

    args = parser.parse_args()

    args.func(args)


if __name__ == "__main__":
    main()