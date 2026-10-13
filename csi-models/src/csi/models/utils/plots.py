from __future__ import annotations

import os
from typing import Iterable, Optional

import matplotlib.pyplot as plt

from .training import TrainingEpochResult

__all__ = ["plot_training_curves"]


def plot_training_curves(
    history: Iterable[TrainingEpochResult],
    out_path: str,
    *,
    title: Optional[str] = None,
) -> Optional[str]:
    epochs = []
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []

    for entry in history:
        epochs.append(entry.epoch)
        train_loss.append(entry.train_loss)
        train_acc.append(entry.train_accuracy)
        val_loss.append(entry.val_loss)
        val_acc.append(entry.val_accuracy)

    if not epochs:
        return None

    has_val = any(v is not None for v in val_loss) or any(v is not None for v in val_acc)

    fig, axs = plt.subplots(2, 2 if has_val else 1, figsize=(12, 8))
    if hasattr(axs, "shape"):
        axes = axs
    else:
        axes = [[axs]]

    ax_loss = axes[0][0] if has_val else axes[0]
    ax_loss.plot(epochs, train_loss, label="train")
    ax_loss.set_title("Training Loss")
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Loss")
    ax_loss.grid(True)

    if has_val:
        ax_val_loss = axes[0][1]
        ax_val_loss.plot(epochs, [v for v in val_loss], label="val")
        ax_val_loss.set_title("Validation Loss")
        ax_val_loss.set_xlabel("Epoch")
        ax_val_loss.set_ylabel("Loss")
        ax_val_loss.grid(True)

        ax_acc = axes[1][0]
        ax_acc.plot(epochs, train_acc, label="train")
        ax_acc.set_title("Training Accuracy")
        ax_acc.set_xlabel("Epoch")
        ax_acc.set_ylabel("Accuracy")
        ax_acc.grid(True)

        ax_val_acc = axes[1][1]
        ax_val_acc.plot(epochs, [v for v in val_acc], label="val")
        ax_val_acc.set_title("Validation Accuracy")
        ax_val_acc.set_xlabel("Epoch")
        ax_val_acc.set_ylabel("Accuracy")
        ax_val_acc.grid(True)
    else:
        if hasattr(axes, "__len__") and len(axes) > 1:
            ax_acc = axes[1]
            ax_acc.plot(epochs, train_acc, label="train")
            ax_acc.set_title("Training Accuracy")
            ax_acc.set_xlabel("Epoch")
            ax_acc.set_ylabel("Accuracy")
            ax_acc.grid(True)

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)
    return out_path
