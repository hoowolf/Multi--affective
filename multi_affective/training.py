from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from .io_utils import save_csv, save_json
from .labels import ID2LABEL


def compute_class_weights(index_rows: list[tuple[str, int]], num_classes: int) -> torch.Tensor:
    ys = [y for _, y in index_rows]
    counts = np.bincount(np.array(ys), minlength=num_classes).astype(np.float64)
    counts[counts == 0] = 1.0
    inv = 1.0 / counts
    weights = inv / inv.mean()
    return torch.tensor(weights, dtype=torch.float32)


def freeze_module(m: nn.Module) -> None:
    for p in m.parameters():
        p.requires_grad = False


def split_params(model: nn.Module, head_keywords: tuple[str, ...]) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    encoder_params: list[nn.Parameter] = []
    head_params: list[nn.Parameter] = []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if any(k in name for k in head_keywords):
            head_params.append(p)
        else:
            encoder_params.append(p)
    return encoder_params, head_params


def evaluate(model: nn.Module, loader: Any, device: torch.device) -> dict[str, Any]:
    from sklearn.metrics import accuracy_score, confusion_matrix, f1_score  # type: ignore

    model.eval()
    ys: list[int] = []
    preds: list[int] = []
    losses: list[float] = []
    loss_fn = getattr(model, "_loss_fn", None)
    with torch.no_grad():
        for batch in loader:
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            logits = model(batch)
            y = batch["label"]
            if loss_fn is not None:
                losses.append(float(loss_fn(logits, y).detach().cpu().item()))
            p = torch.argmax(logits, dim=-1)
            ys.extend(y.detach().cpu().tolist())
            preds.extend(p.detach().cpu().tolist())
    acc = float(accuracy_score(ys, preds)) if ys else 0.0
    macro_f1 = float(f1_score(ys, preds, average="macro")) if ys else 0.0
    cm = confusion_matrix(ys, preds, labels=[0, 1, 2]).tolist() if ys else [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    return {"loss": float(np.mean(losses)) if losses else 0.0, "acc": acc, "macro_f1": macro_f1, "confusion_matrix": cm}


def train_one_epoch(model: nn.Module, loader: Any, optimizer: torch.optim.Optimizer, device: torch.device) -> dict[str, Any]:
    from tqdm import tqdm  # type: ignore

    model.train()
    loss_fn = getattr(model, "_loss_fn", None)
    if loss_fn is None:
        raise RuntimeError("Loss function not attached to model")
    losses: list[float] = []
    correct = 0
    total = 0
    for batch in tqdm(loader, desc="train", leave=False):
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch)
        y = batch["label"]
        loss = loss_fn(logits, y)
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach().cpu().item()))
        pred = torch.argmax(logits, dim=-1)
        correct += int((pred == y).sum().detach().cpu().item())
        total += int(y.numel())
    return {"loss": float(np.mean(losses)) if losses else 0.0, "acc": float(correct / max(1, total))}


def plot_curves(history: list[dict[str, Any]], out_path: Any) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_f1 = [h["val_macro_f1"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, train_loss, label="train_loss")
    axes[0].plot(epochs, val_loss, label="val_loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(epochs, val_f1, label="val_macro_f1")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def plot_confusion_matrix(cm: list[list[int]], out_path: Any) -> None:
    import matplotlib.pyplot as plt  # type: ignore

    arr = np.array(cm, dtype=np.int64)
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    im = ax.imshow(arr, cmap="Blues")
    ax.set_xticks([0, 1, 2], labels=[ID2LABEL[0], ID2LABEL[1], ID2LABEL[2]])
    ax.set_yticks([0, 1, 2], labels=[ID2LABEL[0], ID2LABEL[1], ID2LABEL[2]])
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(arr[i, j]), ha="center", va="center", color="black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def save_history(out_dir: Any, history: list[dict[str, Any]]) -> None:
    save_json(out_dir / "history.json", history)
    save_csv(out_dir / "history.csv", history)

