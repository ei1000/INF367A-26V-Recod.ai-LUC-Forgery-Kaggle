from __future__ import annotations

from dataclasses import dataclass, fields, replace
from pathlib import Path

from dataset import Datasets


@dataclass(slots=True)
class PipelineConfig:
    """Structured pipeline settings for reproducible training runs."""

    datasets: Datasets = Datasets.TRAIN
    image_size: int = 3000
    epochs: int = 1
    test_run: bool = False
    feature_backbone: str = "dino_single"
    use_dino_transform: bool = False
    batch_size: int = 1
    override_batch_size: bool = False
    dino_model_name: str = "dinov2_vits14"
    cnn_backbone: str = "simple"
    cnn_feature_norm: bool = True
    separate_transforms: bool = True
    pm_iters: int = 24
    pm_beta: float = 10.0
    pm_hard_selection: bool = True
    pm_random_window: int = 50
    pm_use_non_local: bool = True
    pm_non_local_limit: float = 25.0
    pm_flat_threshold: float = 0.15
    pm_margin_threshold: float = 0.10
    pm_topk: int = 1
    pm_reduced_precision: bool = True
    localization_resolution: str = "image"
    log_every: int = 10
    output_dir: str | Path = "artifacts"
    checkpoint_name: str = "latest.pt"
    resume: bool = True
    save_predictions: bool = False
    validation_split: float = 0.0
    test_split: float = 0.0
    validation_seed: int = 42
    learning_rate: float = 1e-3
    mprime_loss_weight: float = 0.5
    empty_target_penalty_weight: float = 0.0
    dlf_error_scaling: str = "log1p"
    do_post_process: bool = True
    post_process_threshold: float = 0.5
    post_process_confident_threshold: float | None = None
    post_process_smooth_probabilities: bool = False
    post_process_fill_holes: bool = True
    post_process_apply_closing: bool = False
    post_process_min_component_area: int = 512

    def with_overrides(self, **overrides) -> "PipelineConfig":
        if not overrides:
            return self

        valid_fields = {field.name for field in fields(self)}
        unknown_fields = sorted(set(overrides) - valid_fields)
        if unknown_fields:
            unknown = ", ".join(unknown_fields)
            raise TypeError(f"Unknown pipeline config override(s): {unknown}")
        return replace(self, **overrides)


def resolve_pipeline_config(
    config: PipelineConfig | None = None,
    **overrides,
) -> PipelineConfig:
    if config is None:
        config = PipelineConfig()
    elif not isinstance(config, PipelineConfig):
        raise TypeError("pipeline config must be a PipelineConfig instance or None")

    return config.with_overrides(**overrides)
