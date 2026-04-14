from __future__ import annotations

import json
from pathlib import Path


def append_metrics_log(output_dir: Path, metrics: dict):
    metrics_path = output_dir / "metrics.jsonl"
    with metrics_path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(metrics) + "\n")


def load_metrics_history(output_dir: Path) -> list[dict]:
    metrics_path = output_dir / "metrics.jsonl"
    if not metrics_path.exists():
        return []

    history = []
    with metrics_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                history.append(json.loads(line))
    return history


def save_metrics_plot(output_dir: Path):
    history = load_metrics_history(output_dir)
    if not history:
        return

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        print(f"[pipeline] Skipping metrics plot: {exc}")
        return

    epochs = [entry["epoch"] for entry in history if "epoch" in entry]
    train_losses = [entry.get("train_loss", entry.get("train_dice_loss")) for entry in history]
    val_losses = [entry.get("val_loss", entry.get("val_dice_loss")) for entry in history]
    train_ldfm = [entry.get("train_ldfm") for entry in history]
    train_lmrd = [entry.get("train_lmrd") for entry in history]
    train_mprime_loss = [entry.get("train_mprime_loss") for entry in history]
    val_ldfm = [entry.get("val_ldfm") for entry in history]
    val_lmrd = [entry.get("val_lmrd") for entry in history]
    val_mprime_loss = [entry.get("val_mprime_loss") for entry in history]
    val_of1 = [entry.get("val_of1") for entry in history]

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)

    axes[0].plot(epochs, train_losses, marker="o", label="train_loss")
    if any(value is not None for value in val_losses):
        axes[0].plot(epochs, val_losses, marker="o", label="val_loss")
    if any(value is not None for value in train_ldfm):
        axes[0].plot(epochs, train_ldfm, linestyle="--", label="train_ldfm")
    if any(value is not None for value in train_lmrd):
        axes[0].plot(epochs, train_lmrd, linestyle="--", label="train_lmrd")
    if any(value is not None for value in train_mprime_loss):
        axes[0].plot(epochs, train_mprime_loss, linestyle="-.", label="train_mprime_loss")
    if any(value is not None for value in val_ldfm):
        axes[0].plot(epochs, val_ldfm, linestyle=":", label="val_ldfm")
    if any(value is not None for value in val_lmrd):
        axes[0].plot(epochs, val_lmrd, linestyle=":", label="val_lmrd")
    if any(value is not None for value in val_mprime_loss):
        axes[0].plot(epochs, val_mprime_loss, linestyle=(0, (3, 1, 1, 1)), label="val_mprime_loss")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    if any(value is not None for value in val_of1):
        axes[1].plot(epochs, val_of1, marker="o", color="tab:green", label="val_oF1")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Score")
    axes[1].set_title("Validation oF1")
    axes[1].grid(True, alpha=0.3)
    if any(value is not None for value in val_of1):
        axes[1].legend()

    fig.tight_layout()
    plot_path = output_dir / "metrics_plot.png"
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)
