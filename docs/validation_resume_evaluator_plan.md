# Validation Optimization, Resume, And Evaluator Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce remaining validation GPU/CPU synchronization, add robust checkpoint resume support, and create a safe validation-first evaluator workflow for post-processing experiments.

**Architecture:** Keep training unchanged by default, but split training and validation batch sizing, optionally accumulate validation probabilities on GPU before one CPU transfer, and checkpoint both best-by-`kaggle_score` and resumable last-training state. Evaluation should load a checkpoint, rebuild the exact split, reproduce validation scoring first, then support post-processing sweeps and a later one-shot local holdout evaluation.

**Tech Stack:** Python 3.12, PyTorch, NumPy, SciPy, pandas, tqdm, existing `recodai_f1.py`, existing `DinoSegmenter`, existing validation helpers.

---

## Context

The validation performance refactor is complete and working. A full 10-epoch run reached:

```text
Epoch 4: avg_loss=0.0739  kaggle_score=0.4815
  -> New best model saved by kaggle_score=0.4815
```

Remaining observations:

- Validation is much smoother than before, but still synchronizes once per validation batch when probability maps move from GPU to CPU.
- Current validation stores CPU probability maps as `validation_probability_dtype`, default `"float16"`, but the current implementation transfers float32 probabilities to CPU before NumPy casting.
- Current validation uses the shared `batch_size`, so validation cannot independently use a larger batch size despite having no backward pass.
- Current checkpointing saves only best-by-`kaggle_score`. It contains model and optimizer state, but there is no last checkpoint, scheduler state, scaler state, or resume path.
- `docs/implementation_note_2.md` already identifies orphaned post-processing config knobs and the need for local holdout evaluation/submission entrypoints.

## Scope

In scope:

- Add a separate validation batch size.
- Downcast validation probabilities on GPU before CPU transfer when `validation_probability_dtype="float16"`.
- Add optional `validation_transfer_mode="accumulate_gpu"` for current-size validation runs on GPUs with enough memory.
- Add last-checkpoint saving and resume-from-checkpoint support.
- Preserve best checkpoint selection by `kaggle_score`.
- Add a validation evaluator script that can reproduce training validation score and sweep post-processing settings on cached validation probabilities.
- Add an evaluator notebook after the script is trusted.
- Add one-shot local holdout evaluation after validation tuning is complete.

Out of scope for this plan:

- Changing model architecture.
- Tuning on the reserved local test split.
- Using supplemental data.
- Final Kaggle submission generation, TTA, or global/local fusion. Those remain covered by `docs/implementation_note_2.md` and should be planned separately or merged later.
- Automatically rewriting or deleting existing checkpoints.

## Human Checkpoints

Required human approval before:

- Running full training.
- Running a resume-training smoke that writes checkpoints.
- Running one-shot local holdout evaluation.
- Running any submission or test-image inference script.

Validation-only evaluator runs on the validation split are allowed during development after a checkpoint path is provided.

## File Structure

Create:

- `engine/checkpointing.py`: checkpoint save/load helpers for best and last checkpoints.
- `tests/test_checkpointing.py`: synthetic tests for checkpoint metadata, resume state, and best-score preservation.
- `evaluate_validation_postprocess.py`: load checkpoint, rebuild split, cache validation probabilities, reproduce validation score, and sweep post-processing configs.
- `notebooks/evaluate_validation_postprocess.ipynb`: interactive wrapper around the evaluator script, added only after the script is stable.
- `evaluate_baseline.py`: one-shot reserved local holdout evaluation, added only after validation evaluator is trusted.

Modify:

- `configs/baseline_config.py`: add validation transfer, validation batch size, resume, and last-checkpoint config fields.
- `train_baseline.py`: use `val_batch_size`, load resume checkpoint if configured, save last checkpoints, preserve best-score behavior.
- `engine/validation_inference.py`: support GPU downcast before transfer and optional GPU accumulation.
- `engine/validate_loop.py`: pass validation transfer mode to collection.
- `inference/postprocess.py`, `util/pixelmapUtil.py`, `engine/validation_inference.py`, and `train_baseline.py`: wire post-processing knobs before evaluator sweeps, following `docs/implementation_note_2.md`.
- `tests/test_validation_inference.py`: cover downcast and transfer modes.
- `tests/test_postprocess.py`: cover post-processing knob wiring.

---

## Workstream 1: Validation Transfer Optimization

### Task 1: Add Validation Batch Size Config

**Files:**

- Modify: `configs/baseline_config.py`
- Modify: `train_baseline.py`
- Test: compile checks and existing unit tests

- [ ] Add config field:

```python
    val_batch_size: int | None = None
```

- [ ] Use `config.val_batch_size or config.batch_size` for the validation `DataLoader` only.
- [ ] Keep training `DataLoader` on `config.batch_size`.
- [ ] Add a small test if a train-baseline config helper is introduced; otherwise rely on compile and a focused code review.

Expected behavior:

- Default behavior is unchanged when `val_batch_size is None`.
- Users can set `val_batch_size=64`, `96`, or `128` to reduce validation batches and GPU sync points.

### Task 2: Downcast Probabilities Before CPU Transfer

**Files:**

- Modify: `engine/validation_inference.py`
- Test: `tests/test_validation_inference.py`

- [ ] Add a helper that maps `validation_probability_dtype` to both Torch and NumPy dtypes:

```python
def _probability_dtypes(probability_dtype: str) -> tuple[torch.dtype, np.dtype]:
    if probability_dtype == "float16":
        return torch.float16, np.dtype(np.float16)
    if probability_dtype == "float32":
        return torch.float32, np.dtype(np.float32)
    raise ValueError(...)
```

- [ ] In direct mode, after `torch.sigmoid(logits)`, convert to the requested Torch dtype before `.cpu()`:

```python
probs = torch.sigmoid(logits).to(torch_dtype)
probability_batch = _squeeze_probability_batch(probs.detach().cpu().numpy())
```

- [ ] In sliding mode, downcast the stacked probability batch before CPU transfer.
- [ ] Keep `float32` available for exact debugging.

Expected behavior:

- `validation_probability_dtype="float16"` halves validation probability transfer volume.
- Existing tests still pass.

### Task 3: Add Optional GPU Accumulation Transfer Mode

**Files:**

- Modify: `configs/baseline_config.py`
- Modify: `engine/validation_inference.py`
- Modify: `engine/validate_loop.py`
- Modify: `train_baseline.py`
- Test: `tests/test_validation_inference.py`

- [ ] Add config field:

```python
    validation_transfer_mode: str = "per_batch"
```

Allowed values:

- `"per_batch"`: current robust behavior; transfer each validation probability batch to CPU.
- `"accumulate_gpu"`: keep probability batches on GPU during inference, then transfer the concatenated probability tensor to CPU once after all validation batches finish.

- [ ] Add `transfer_mode: str = "per_batch"` argument to `collect_validation_predictions`.
- [ ] Add `validation_transfer_mode` argument to `validate_one_epoch` and pass it through.
- [ ] Wire `config.validation_transfer_mode` from `train_baseline.py`.
- [ ] In `"accumulate_gpu"` mode:
  - keep `torch.Tensor` probability batches in a list,
  - keep matching sample references in order,
  - collect CPU masks only if `collect_masks=True`,
  - concatenate probability batches after the loader finishes,
  - transfer once to CPU,
  - build `ValidationPrediction` records after the transfer.
- [ ] Reject unknown transfer modes with a clear `ValueError`.

Expected behavior:

- On current `448x448` validation with about 514 samples, GPU memory use is approximately:
  - float16 probabilities: ~206 MB,
  - float32 probabilities: ~412 MB.
- `per_batch` remains the default for robustness.
- `accumulate_gpu` is available for local/Colab GPUs with enough memory.

---

## Workstream 2: Resume Training And Last Checkpoints

### Task 4: Add Checkpoint Config Fields

**Files:**

- Modify: `configs/baseline_config.py`
- Test: compile check

- [ ] Add:

```python
    resume_checkpoint_path: str | None = None
    save_last_checkpoint: bool = True
    last_checkpoint_name: str = "last.pt"
    save_last_every_epochs: int = 1
```

Notes:

- For Colab, default `save_last_every_epochs=1` is safest.
- Local runs can set `save_last_every_epochs=3` if write frequency matters.

### Task 5: Extract Checkpoint Save/Load Helpers

**Files:**

- Create: `engine/checkpointing.py`
- Create: `tests/test_checkpointing.py`
- Modify: `train_baseline.py`

- [ ] Create helpers to build checkpoint payloads with:
  - `epoch`,
  - `model_state_dict`,
  - `optimizer_state_dict`,
  - `scheduler_state_dict`,
  - `scaler_state_dict`,
  - `kaggle_score`,
  - `best_kaggle_score`,
  - `validation_result`,
  - `config`,
  - `split_counts`,
  - `model_name`.
- [ ] Save best checkpoint only when the new `kaggle_score` improves over `best_kaggle_score`.
- [ ] Save last checkpoint when:

```python
config.save_last_checkpoint and ((epoch + 1) % config.save_last_every_epochs == 0)
```

- [ ] Validate `save_last_every_epochs >= 1`.

Expected behavior:

- Best checkpoint remains model-selection artifact.
- Last checkpoint is resumable training state.
- Current best checkpoint format remains loadable through fallback behavior.

### Task 6: Resume From Last Or Best Checkpoint

**Files:**

- Modify: `train_baseline.py`
- Modify: `engine/checkpointing.py`
- Test: `tests/test_checkpointing.py`

- [ ] If `config.resume_checkpoint_path` is set, load checkpoint after model, optimizer, scheduler, and scaler are constructed.
- [ ] Restore:
  - model state,
  - optimizer state when present,
  - scheduler state when present,
  - scaler state when present,
  - `start_epoch`,
  - `best_kaggle_score`.
- [ ] Backward-compatible best-score fallback:

```python
best_kaggle_score = checkpoint.get("best_kaggle_score", checkpoint.get("kaggle_score", 0.0))
```

- [ ] Start training loop at `start_epoch`, so a checkpoint saved at epoch 4 continues with epoch index 4.
- [ ] Print a clear resume message:

```text
Resumed from runs/checkpoints/last.pt at epoch=4 best_kaggle_score=0.4815
```

Expected behavior:

- Resuming from current best checkpoints is possible even if they lack `scheduler_state_dict`, `scaler_state_dict`, or `best_kaggle_score`.
- Resuming does not overwrite the best checkpoint unless a later validation score improves it.

### Task 7: Optional RNG State Preservation

**Files:**

- Modify: `engine/checkpointing.py`
- Test: `tests/test_checkpointing.py`

- [ ] Save CPU RNG states:
  - `random.getstate()`,
  - `np.random.get_state()`,
  - `torch.get_rng_state()`.
- [ ] Save CUDA RNG states when CUDA is available:
  - `torch.cuda.get_rng_state_all()`.
- [ ] Restore when present.

Expected behavior:

- Resume is closer to exact reproducibility.
- Missing RNG state in older checkpoints is not an error.

---

## Workstream 3: Validation Evaluator And Post-Processing Experiments

### Task 8: Wire Existing Post-Processing Config Knobs

**Files:**

- Create: `tests/test_postprocess.py`
- Modify: `configs/baseline_config.py`
- Modify: `util/pixelmapUtil.py`
- Modify: `inference/postprocess.py`
- Modify: `engine/validation_inference.py`
- Modify: `engine/validate_loop.py`
- Modify: `train_baseline.py`

Follow `docs/implementation_note_2.md` Tasks 14-15, with one adjustment for the current validation refactor:

- `score_validation_predictions` owns the CPU post-processing call now, so post-processing parameters must be added there and forwarded from `validate_one_epoch`.

Defaults should represent the previous hardcoded behavior:

```python
    post_process_confident_threshold: float = 0.9
    post_process_smooth_probabilities: bool = True
    post_process_fill_holes: bool = True
    post_process_apply_opening: bool = True
    post_process_apply_closing: bool = True
    post_process_keep_confident_seeded_components: bool = False
```

Expected behavior:

- Default validation score should remain close to the current score because defaults match the old hardcoded path.
- Evaluator scripts can alter post-processing without retraining or re-running model inference.

### Task 9: Add Validation Probability Cache/Evaluator Script

**Files:**

- Create: `evaluate_validation_postprocess.py`
- Test: compile check and a small unit test if helpers are factored out

Behavior:

- Load checkpoint from `--checkpoint`, default `runs/checkpoints/best_by_kaggle_score.pt`.
- Reconstruct `BaselineConfig` from checkpoint config, with CLI overrides for safe evaluator-only settings.
- Rebuild the exact train/val/test split using stored or config seed.
- Load the validation split, not the local test split.
- Build model and load checkpoint model weights.
- Run validation inference once and cache probability maps in memory or optionally under `runs/eval_cache/`.
- Reproduce the checkpoint's validation `kaggle_score` using the stored/default post-processing settings.
- Sweep post-processing settings from CLI or a small grid file.
- Print a ranked table:

```text
rank  kaggle_score  pred_threshold  min_component_area  smooth  closing  opening  confident_threshold
1     0.4932        0.45            50                  true    true     true     0.9
```

Safety:

- The evaluator must say clearly that it is using the validation split for tuning.
- It must not touch the local test split.

### Task 10: Add Evaluator Notebook

**Files:**

- Create: `notebooks/evaluate_validation_postprocess.ipynb`

Notebook should:

- Verify project root and checkpoint path.
- Import and call the evaluator script helpers instead of duplicating logic.
- Show baseline validation score replication.
- Run a small post-processing sweep.
- Display several authentic/forged examples with:
  - image,
  - probability map,
  - baseline mask,
  - tuned mask,
  - ground-truth instance/union mask.
- Warn that the local test split must not be used for repeated tuning.

Expected behavior:

- Notebook is an inspection and visualization surface, not the source of truth.

### Task 11: Add One-Shot Local Holdout Evaluation

**Files:**

- Create: `evaluate_baseline.py`
- Test: compile check

Behavior:

- Load checkpoint.
- Rebuild exact split.
- Evaluate only the reserved local test split.
- Print:
  - checkpoint path,
  - epoch,
  - validation `kaggle_score` stored in checkpoint,
  - local holdout `kaggle_score`,
  - sample counts.

Safety:

- Require explicit human approval before running.
- Print a warning:

```text
This evaluates the reserved local holdout test split. Use it for final local review, not tuning.
```

Expected behavior:

- Local holdout evaluation is available, but the workflow keeps validation split as the tuning surface.

---

## Review Checklist

- [ ] Validation can use `val_batch_size` independently of training `batch_size`.
- [ ] `validation_probability_dtype="float16"` downcasts on GPU before CPU transfer.
- [ ] `validation_transfer_mode="per_batch"` remains the default.
- [ ] `validation_transfer_mode="accumulate_gpu"` collects all validation probability batches on GPU and transfers once.
- [ ] `compute_pixel_f1=False` still avoids mask conversion.
- [ ] Best checkpoint is still selected only by `kaggle_score`.
- [ ] Last checkpoint is saved on the configured cadence.
- [ ] Resume restores best score and does not overwrite best unless score improves.
- [ ] Existing best checkpoints without new fields remain loadable.
- [ ] Validation evaluator reproduces the training validation score before sweeps.
- [ ] Post-processing sweeps use validation split only.
- [ ] Local holdout evaluator exists but is clearly one-shot/final-review only.
- [ ] Full unit tests pass without running training.

## Suggested Implementation Order

1. Validation transfer optimization (`val_batch_size`, GPU downcast, transfer mode).
2. Resume/last checkpoint support.
3. Post-processing knob wiring.
4. Validation evaluator script.
5. Evaluator notebook.
6. One-shot local holdout evaluator.

