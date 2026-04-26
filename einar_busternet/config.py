from __future__ import annotations

from dataclasses import dataclass

from configs.baseline_config import seed_worker, set_seed


@dataclass(frozen=True)
class BusterNetConfig:
    batch_size: int = 32
    seed: int = 42
    target_size: int = 448
    pred_threshold: float = 0.2
    harden_temperature: float = 0.7
    hard_clip_low: float = 0.1
    hard_clip_high: float = 0.9
    min_component_area: int = 10
    train_subset: int | None = None
    val_subset: int | None = None
    grad_clip_max_norm: float = 1.0
    train_num_workers: int = 2
    val_num_workers: int = 1
    use_rgb: bool = True
    normalize_rgb: bool = True
    dino_mean: tuple[float, float, float] = (0.485, 0.456, 0.406)
    dino_std: tuple[float, float, float] = (0.229, 0.224, 0.225)
    dino_model_name: str = "dinov2_vitb14"
    dino_embed_dim: int = 768
    freeze_dino_encoder: bool = True
    use_amp: bool = True
    sliding_window_size: int | None = 448
    sliding_stride: int | None = 224
    sliding_batch_size: int = 8
    data_root: str = "data"
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    compute_pixel_f1: bool = False
    verify_score_equivalence: bool = False
    validation_inference_mode: str = "direct"
    validation_probability_dtype: str = "float16"
    validation_log_timing: bool = True
    include_supplemental: bool = False
    post_process_confident_threshold: float | None = 0.9
    post_process_smooth_probabilities: bool = True
    post_process_fill_holes: bool = True
    post_process_apply_opening: bool = False
    post_process_apply_closing: bool = True
    post_process_keep_confident_seeded_components: bool = False
    val_batch_size: int | None = None
    validation_transfer_mode: str = "accumulate_gpu"
    resume_checkpoint_path: str | None = None
    save_last_checkpoint: bool = True
    save_last_every_epochs: int = 1

    stage1_epochs: int = 20
    stage2_epochs: int = 10
    stage3_epochs: int = 10
    stage1_lr: float = 1e-3
    stage2_lr: float = 1e-2
    stage3_lr: float = 1e-5
    nb_pools: int = 100
    ce_class_weights: tuple[float, float, float] = (0.3, 1.0, 1.0)
    fusion_mode: str = "three_class"
    union_wrapper_eps: float = 1e-6

    metadata_path: str = "data/train_masks_source_target_metadata.csv"
    allowed_forged_statuses: tuple[str, ...] = ("derived_from_pair",)
    include_authentic: bool = True
    authentic_policy: str = "paired_derived_only"

    # Step 0 thresholds are recorded here for audit; training reads generated masks.
    diff_threshold: float = 5.0
    component_change_fraction: float = 0.25

    checkpoint_dir: str = "einar_busternet/artifacts/checkpoints"
    results_dir: str = "einar_busternet/artifacts/results"
    best_checkpoint_name: str = "best.pt"
    best_balanced_checkpoint_name: str = "best_balanced.pt"
    last_checkpoint_name: str = "last.pt"

    stage3_scheduler_factor: float = 0.5
    stage3_scheduler_patience: int = 1
    early_stop_patience: int = 5

    branch_dice_weight: float = 0.5
    fusion_dice_weight: float = 1.0
    stage3_aux_loss_weight: float = 0.1

    @property
    def total_stage_epochs(self) -> int:
        return self.stage1_epochs + self.stage2_epochs + self.stage3_epochs
