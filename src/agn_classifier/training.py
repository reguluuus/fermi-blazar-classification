from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


@dataclass
class TrainingHistory:
    train_loss: list[float]
    val_loss: list[float]


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32)).reshape(-1, 1),
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)


def _epoch_loss(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> float:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0

    context = torch.enable_grad() if training else torch.inference_mode()
    with context:
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = loss_fn(logits, y)

            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()

    return total_loss / max(len(loader), 1)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 40,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-2,
    patience: int = 8,
) -> TrainingHistory:
    """Train with BCEWithLogitsLoss and early stopping on validation loss."""
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    history = TrainingHistory([], [])
    best_state = None
    best_val = float("inf")
    stale_epochs = 0

    for epoch in range(1, epochs + 1):
        train_loss = _epoch_loss(model, train_loader, loss_fn, device, optimizer)
        val_loss = _epoch_loss(model, val_loader, loss_fn, device)
        history.train_loss.append(train_loss)
        history.val_loss.append(val_loss)

        print(
            f"Epoch {epoch:03d}/{epochs} | "
            f"train_loss={train_loss:.5f} | val_loss={val_loss:.5f}"
        )

        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= patience:
                print(f"Early stopping after {epoch} epochs.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    return history
