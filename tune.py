from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import torch
from torch import nn

from agn_classifier.data import fit_preprocessor, load_catalog, make_splits, select_numeric_features
from agn_classifier.model import BlazarClassifier
from agn_classifier.training import make_loader, set_seed


def validation_loss(model, loader, device) -> float:
    loss_fn = nn.BCEWithLogitsLoss()
    model.eval()
    total = 0.0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            total += loss_fn(model(x), y).item()
    return total / max(len(loader), 1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna tuning for the blazar classifier")
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--epochs-per-trial", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/best_params.json"))
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    frame = select_numeric_features(load_catalog(args.catalog))
    splits = make_splits(frame, random_state=args.seed)
    prep = fit_preprocessor(splits.X_train)
    X_train = prep.transform(splits.X_train)
    X_val = prep.transform(splits.X_val)
    y_train = splits.y_train.to_numpy(dtype=np.float32)
    y_val = splits.y_val.to_numpy(dtype=np.float32)

    def objective(trial: optuna.Trial) -> float:
        hidden_size = trial.suggest_int("hidden_size", 32, 256, step=32)
        dropout = trial.suggest_float("dropout", 0.0, 0.4)
        learning_rate = trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True)
        batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 128])

        train_loader = make_loader(X_train, y_train, batch_size, shuffle=True)
        val_loader = make_loader(X_val, y_val, batch_size, shuffle=False)
        model = BlazarClassifier(X_train.shape[1], hidden_size, dropout).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        loss_fn = nn.BCEWithLogitsLoss()

        for _ in range(args.epochs_per_trial):
            model.train()
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                loss = loss_fn(model(x), y)
                loss.backward()
                optimizer.step()

        return validation_loss(model, val_loader, device)

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=args.trials)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(study.best_params, f, indent=2)

    print("Best validation loss:", study.best_value)
    print("Best parameters:", study.best_params)
    print("Saved to:", args.output)


if __name__ == "__main__":
    main()
