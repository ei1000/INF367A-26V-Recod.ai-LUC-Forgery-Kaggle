from __future__ import annotations

import json
from pathlib import Path


def initialize_metric_accumulator() -> dict[str, float]:
    return {
        "loss": 0.0,
        "ldfm": 0.0,
        "lmrd": 0.0,
        "mprime_loss": 0.0,
        "empty_target_loss": 0.0,
        "empty_refined_loss": 0.0,
        "empty_target_map_loss": 0.0,
        "empty_mprime_map_loss": 0.0,
        "mprime_positive_rate": 0.0,
        "mprime_wins_rate": 0.0,
        "target_positive_rate": 0.0,
    }


def update_metric_accumulator(
    accumulator: dict[str, float],
    loss_terms: tuple,
    branch_stats: dict[str, float],
) -> dict[str, float]:
    (
        loss,
        ldfm,
        lmrd,
        mprime_loss,
        empty_target_loss,
        empty_refined_loss,
        empty_target_map_loss,
        empty_mprime_map_loss,
    ) = loss_terms

    accumulator["loss"] += float(loss.item())
    accumulator["ldfm"] += float(ldfm.item())
    accumulator["lmrd"] += float(lmrd.item())
    accumulator["mprime_loss"] += float(mprime_loss.item())
    accumulator["empty_target_loss"] += float(empty_target_loss.item())
    accumulator["empty_refined_loss"] += float(empty_refined_loss.item())
    accumulator["empty_target_map_loss"] += float(empty_target_map_loss.item())
    accumulator["empty_mprime_map_loss"] += float(empty_mprime_map_loss.item())
    accumulator["mprime_positive_rate"] += branch_stats["mprime_positive_rate"]
    accumulator["mprime_wins_rate"] += branch_stats["mprime_wins_rate"]
    accumulator["target_positive_rate"] += branch_stats["target_positive_rate"]
    return accumulator


def summarize_metric_step(loss_terms: tuple, branch_stats: dict[str, float]) -> dict[str, float]:
    summary = initialize_metric_accumulator()
    return update_metric_accumulator(summary, loss_terms, branch_stats)


def average_metric_accumulator(accumulator: dict[str, float], steps: int) -> dict[str, float]:
    return {
        key: value / max(steps, 1)
        for key, value in accumulator.items()
    }


def initialize_instance_metric_tracker() -> dict[str, float | int]:
    return {
        "of1_sum": 0.0,
        "images": 0,
        "pred_components_sum": 0,
        "authentic_of1_sum": 0.0,
        "authentic_images": 0,
        "authentic_empty_pred_count": 0,
        "authentic_pred_components_sum": 0,
        "forged_of1_sum": 0.0,
        "forged_images": 0,
        "forged_pred_components_sum": 0,
        "forged_gt_components_sum": 0,
    }


def update_instance_metric_tracker(
    tracker: dict[str, float | int],
    *,
    image_of1: float,
    pred_component_count: int,
    gt_component_count: int,
) -> dict[str, float | int]:
    tracker["of1_sum"] += image_of1
    tracker["images"] += 1
    tracker["pred_components_sum"] += pred_component_count

    if gt_component_count == 0:
        tracker["authentic_of1_sum"] += image_of1
        tracker["authentic_images"] += 1
        tracker["authentic_pred_components_sum"] += pred_component_count
        if pred_component_count == 0:
            tracker["authentic_empty_pred_count"] += 1
        return tracker

    tracker["forged_of1_sum"] += image_of1
    tracker["forged_images"] += 1
    tracker["forged_pred_components_sum"] += pred_component_count
    tracker["forged_gt_components_sum"] += gt_component_count
    return tracker


def build_validation_summary(
    accumulator: dict[str, float],
    loss_steps: int,
    segmentation_metrics: dict[str, float],
    instance_tracker: dict[str, float | int],
) -> dict[str, float]:
    summary = average_metric_accumulator(accumulator, loss_steps)
    images = max(int(instance_tracker["images"]), 1)
    authentic_images = max(int(instance_tracker["authentic_images"]), 1)
    forged_images = max(int(instance_tracker["forged_images"]), 1)
    summary.update(
        {
            "of1": instance_tracker["of1_sum"] / images,
            "pred_components_per_image": instance_tracker["pred_components_sum"] / images,
            "authentic_of1": instance_tracker["authentic_of1_sum"] / authentic_images,
            "authentic_empty_pred_rate": instance_tracker["authentic_empty_pred_count"] / authentic_images,
            "authentic_pred_components_per_image": instance_tracker["authentic_pred_components_sum"] / authentic_images,
            "forged_of1": instance_tracker["forged_of1_sum"] / forged_images,
            "forged_pred_components_per_image": instance_tracker["forged_pred_components_sum"] / forged_images,
            "forged_gt_components_per_image": instance_tracker["forged_gt_components_sum"] / forged_images,
            "iou": segmentation_metrics["iou"],
            "dice": segmentation_metrics["dice"],
            "pred_positive_rate": segmentation_metrics["pred_positive_rate"],
            "mask_positive_rate": segmentation_metrics["mask_positive_rate"],
        }
    )
    return summary


def write_split_artifacts(output_dir: Path, split_manifest: dict) -> tuple[Path, Path]:
    split_manifest_path = output_dir / "split_manifest.json"
    split_manifest_path.write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")

    heldout_test_samples = [
        sample
        for dataset_entry in split_manifest["datasets"]
        for sample in dataset_entry["test_samples"]
    ]
    heldout_test_path = output_dir / "heldout_test_samples.txt"
    heldout_test_path.write_text("\n".join(heldout_test_samples), encoding="utf-8")
    return split_manifest_path, heldout_test_path


def format_split_summary(split_summary: dict[str, int], split_seed: int) -> str:
    return (
        "[pipeline] Split summary: "
        f"train={split_summary['train_total']} "
        f"val={split_summary['val_total']} "
        f"test={split_summary['test_total']} "
        f"(seed={split_seed})"
    )


def _format_loss_breakdown(
    loss_label: str,
    summary: dict[str, float],
    mprime_loss_weight: float,
    empty_target_penalty_weight: float,
) -> str:
    return (
        f"{loss_label}: {summary['loss']:.4f} "
        f"(ldfm={summary['ldfm']:.4f}, lmrd={summary['lmrd']:.4f}, "
        f"mprime={summary['mprime_loss']:.4f}, "
        f"empty={summary['empty_target_loss']:.4f}"
        f"[ref={summary['empty_refined_loss']:.4f}, "
        f"se={summary['empty_target_map_loss']:.4f}, "
        f"dlf={summary['empty_mprime_map_loss']:.4f}], "
        f"lambda={mprime_loss_weight:.2f}, "
        f"empty_lambda={empty_target_penalty_weight:.2f}) "
        f"mprime_pos: {summary['mprime_positive_rate']:.4%} "
        f"mprime_wins: {summary['mprime_wins_rate']:.4%} "
        f"target_pos: {summary['target_positive_rate']:.4%}"
    )


def format_train_batch_message(
    epoch_idx: int,
    epochs: int,
    batch_idx: int,
    total_batches: int,
    batch_summary: dict[str, float],
    batch_seconds: float,
    mprime_loss_weight: float,
    empty_target_penalty_weight: float,
    localization_stats: dict[str, float] | None = None,
) -> str:
    message = (
        f"[epoch {epoch_idx + 1}/{epochs}] "
        f"batch {batch_idx}/{total_batches} "
        f"{_format_loss_breakdown('loss', batch_summary, mprime_loss_weight, empty_target_penalty_weight)} "
        f"time spent: {batch_seconds:.2f}"
    )
    if localization_stats is not None:
        message += (
            f" feat: {localization_stats['feature_time_s']:.2f}s"
            f" pm: {localization_stats['patchmatch_time_s']:.2f}s"
            f" dlf: {localization_stats['dlf_time_s']:.2f}s"
        )
        peak_memory_mb = localization_stats.get("localization_peak_memory_mb")
        if peak_memory_mb is not None:
            message += f" loc_peak: {peak_memory_mb:.0f}MB"
    return message


def format_train_epoch_message(
    epoch_idx: int,
    epochs: int,
    train_summary: dict[str, float],
    epoch_seconds: float,
    mprime_loss_weight: float,
    empty_target_penalty_weight: float,
) -> str:
    return (
        f"[epoch {epoch_idx + 1}/{epochs}] "
        f"{_format_loss_breakdown('train_loss', train_summary, mprime_loss_weight, empty_target_penalty_weight)} "
        f"completed in: {epoch_seconds:.2f}s"
    )


def format_validation_message(
    epoch_idx: int,
    epochs: int,
    val_summary: dict[str, float],
    mprime_loss_weight: float,
    empty_target_penalty_weight: float,
) -> str:
    return (
        f"[epoch {epoch_idx + 1}/{epochs}] "
        f"{_format_loss_breakdown('val_loss', val_summary, mprime_loss_weight, empty_target_penalty_weight)} "
        f"val_oF1: {val_summary['of1']:.4f} "
        f"val_pred_pos: {val_summary['pred_positive_rate']:.4%} "
        f"pred_components/img: {val_summary['pred_components_per_image']:.2f} "
        f"auth_oF1: {val_summary['authentic_of1']:.4f} "
        f"auth_empty_pred: {val_summary['authentic_empty_pred_rate']:.2%} "
        f"auth_components/img: {val_summary['authentic_pred_components_per_image']:.2f} "
        f"forged_oF1: {val_summary['forged_of1']:.4f} "
        f"forged_components/img: {val_summary['forged_pred_components_per_image']:.2f} "
        f"forged_gt_components/img: {val_summary['forged_gt_components_per_image']:.2f}"
    )


def build_epoch_metrics(
    epoch_idx: int,
    epoch_seconds: float,
    train_summary: dict[str, float],
    steps: int,
    val_summary: dict[str, float] | None = None,
) -> dict[str, float]:
    metrics = {
        "epoch": epoch_idx + 1,
        "train_loss": train_summary["loss"],
        "train_ldfm": train_summary["ldfm"],
        "train_lmrd": train_summary["lmrd"],
        "train_mprime_loss": train_summary["mprime_loss"],
        "train_empty_target_loss": train_summary["empty_target_loss"],
        "train_empty_refined_loss": train_summary["empty_refined_loss"],
        "train_empty_target_map_loss": train_summary["empty_target_map_loss"],
        "train_empty_mprime_map_loss": train_summary["empty_mprime_map_loss"],
        "train_mprime_positive_rate": train_summary["mprime_positive_rate"],
        "train_mprime_wins_rate": train_summary["mprime_wins_rate"],
        "train_target_positive_rate": train_summary["target_positive_rate"],
        "steps": steps,
        "epoch_seconds": epoch_seconds,
    }

    if val_summary is None:
        return metrics

    metrics.update(
        {
            "val_loss": val_summary["loss"],
            "val_ldfm": val_summary["ldfm"],
            "val_lmrd": val_summary["lmrd"],
            "val_mprime_loss": val_summary["mprime_loss"],
            "val_empty_target_loss": val_summary["empty_target_loss"],
            "val_empty_refined_loss": val_summary["empty_refined_loss"],
            "val_empty_target_map_loss": val_summary["empty_target_map_loss"],
            "val_empty_mprime_map_loss": val_summary["empty_mprime_map_loss"],
            "val_mprime_positive_rate": val_summary["mprime_positive_rate"],
            "val_mprime_wins_rate": val_summary["mprime_wins_rate"],
            "val_target_positive_rate": val_summary["target_positive_rate"],
            "val_of1": val_summary["of1"],
            "val_pred_components_per_image": val_summary["pred_components_per_image"],
            "val_authentic_of1": val_summary["authentic_of1"],
            "val_authentic_empty_pred_rate": val_summary["authentic_empty_pred_rate"],
            "val_authentic_pred_components_per_image": val_summary["authentic_pred_components_per_image"],
            "val_forged_of1": val_summary["forged_of1"],
            "val_forged_pred_components_per_image": val_summary["forged_pred_components_per_image"],
            "val_forged_gt_components_per_image": val_summary["forged_gt_components_per_image"],
            "val_iou": val_summary["iou"],
            "val_dice": val_summary["dice"],
            "val_pred_positive_rate": val_summary["pred_positive_rate"],
            "val_mask_positive_rate": val_summary["mask_positive_rate"],
        }
    )
    return metrics


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
    except ImportError as exc:
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
