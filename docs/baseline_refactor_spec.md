# Baseline Refactor Spec

Status: draft spec  
Next document: implementation plan  
Rule: do not implement code changes from this spec until the plan is approved.

## Purpose

The current repository has a working DINOv2-inspired baseline, but the code, validation behavior, and documentation have drifted. This spec captures what needs to be clarified and refactored before implementation starts.

The baseline inspiration is:

https://www.kaggle.com/code/gauravparkhedkar/dinov2-base-0-332-high-res-4500px-robust-inf

The competition is:

https://www.kaggle.com/competitions/recodai-luc-scientific-image-forgery-detection

## Current Baseline Summary

The maintained baseline entrypoint is `train_baseline.py`.

Current high-level flow:

1. Read forged case IDs from `data/train_images/forged`.
2. Split forged IDs into train/validation.
3. Load images and union masks with `ForgeryDataset`.
4. Resize to `BaselineConfig.target_size`.
5. Train a frozen DINOv2 encoder plus small CNN decoder.
6. Validate with DINO-specific sliding-window inference.
7. Post-process probabilities into binary masks.
8. Score with `recodai_f1.calculate_f1_score`.

The current baseline uses a pixel-level F1 helper copied from the Kaggle metric file, not sklearn F1. However, it does not currently use the full `recodai_f1.score` function during validation.

## Non-Goals For The First Refactor

- Do not rewrite the whole training stack.
- Do not optimize leaderboard score before metric and data behavior are correct enough to trust.
- Do not update `docs/pipeline_suggestion.md` as the source of truth before the refactor is complete.
- Do not make the README large. Keep the root README minimal and move detailed explanations into docs.
- Do not use the draft report as source of truth for code decisions.

## Workstream 1: Metric Behavior

### Current State

`engine/validate_loop.py` uses `calculate_f1_score(pred_bin, gt_bin)` from `recodai_f1.py`.

This is pixel-level F1 over one predicted binary mask and one ground-truth binary mask. It is fast and semi-correct as a training signal.

The full copied metric file also contains:

- `rle_encode`
- `rle_decode`
- `oF1_score`
- `evaluate_single_image`
- `score`

These functions represent the fuller Kaggle-style scoring path, including RLE strings, authentic rows, multiple masks, and optimal matching.

### Requirements

- Use Kaggle-equivalent instance scoring for epoch validation.
- A direct array-based scorer may be used during epoch validation to avoid pandas/RLE overhead, but it must be verified against `recodai_f1.score`.
- Use verified `kaggle_score` for model selection.
- Keep a fast pixel-level validation metric available only as a fallback/debug metric.
- Clearly name metrics in logs so nobody confuses fast pixel F1 with Kaggle-equivalent instance scoring.

### Decisions

- Epoch validation should compute and report verified `kaggle_score` by default.
- Model selection should use verified `kaggle_score`.
- `pixel_f1` can remain available as a clearly named fallback/debug metric, not as the default selection metric.
- If `pixel_f1` is reported, logs and docs must state that it is not the official competition score.
- Model-selection logs should state that the selected checkpoint was chosen by `kaggle_score`.
- Empty predictions should be represented as the exact string `authentic`.
- Non-empty predictions should be converted to connected-component instance masks.
- Official-format validation/reference paths should encode non-empty predictions as semicolon-separated JSON RLE strings through the Kaggle-compatible RLE path.
- Forged ground truth should preserve instance masks where possible for Kaggle-equivalent scoring, rather than only using a union mask.
- `recodai_f1.oF1_score` alone is not sufficient as the full validation metric, because it does not handle the competition's authentic exact-match semantics and is not designed as a complete image-level wrapper for empty authentic cases.
- A direct fast scorer should wrap instance oF1 with the official image-level rules:
  - ground truth authentic plus prediction authentic/empty gives `1.0`,
  - ground truth authentic plus any predicted component gives `0.0`,
  - ground truth forged plus prediction authentic/empty gives `0.0`,
  - ground truth forged plus predicted components gives instance oF1.
- Epoch validation may use the direct fast scorer by default after equivalence tests confirm that it matches `recodai_f1.score` on representative cases.
- Official-style solution/submission rows with `annotation` and `shape`, followed by `recodai_f1.score`, should remain the reference/equivalence path and the final submission-compatible path.
- Union-mask pixel F1 can continue using a union ground-truth mask only for fallback/debug use.
- Fallback/debug `pixel_f1` should be opt-in through a config flag or command-line flag, not computed by default.

### Probability-To-Mask Thresholding

The repository already thresholds model probabilities into binary masks during validation. The scoring refactor does not introduce a new concept here; it reuses this probability-to-mask step before connected-component extraction and Kaggle-equivalent scoring.

Current repo behavior:

- `BaselineConfig.pred_threshold = 0.5`.
- Validation applies `torch.sigmoid(logits)`, post-processes probabilities, then thresholds at `pred_threshold`.
- The current post-processing also hardens probabilities with `harden_temperature`, `hard_clip_low`, and `hard_clip_high`.
- Components smaller than `min_component_area` are filtered.

Kaggle notebook behavior:

- Uses `min_mean_conf = 0.19`.
- Combines probability and Sobel-gradient signal into an enhanced probability map.
- Thresholds the enhanced map with `enhanced > min_mean_conf`.
- Runs morphological close/open.
- Filters connected components by area and by blob mean confidence.
- Returns `authentic` when no component survives.

Spec implication:

- Kaggle-equivalent validation needs an explicit probability-to-instance-mask policy.
- This policy should use the current repo post-processing defaults first, because that preserves current behavior while changing the metric/evaluation path.
- Thresholds should be treated as validation/inference hyperparameters and can be tuned on the validation split.
- The first implementation should use the current repo post-processing defaults for continuity.
- Kaggle notebook-inspired post-processing should be treated as a later measured improvement.

### Acceptance Criteria

- It is clear from code and logs which metric is being computed.
- The training baseline no longer implies that pixel F1 is identical to final leaderboard scoring.
- Epoch validation runs verified Kaggle-equivalent `kaggle_score`.
- Checkpoint/model selection uses verified `kaggle_score`.

## Workstream 2: Authentic Data

### Current State

`train_baseline.py` discovers IDs only from `data/train_images/forged`.

`ForgeryDataset` currently assumes every sample has at least one mask because it calls `load_union_mask`.

This means authentic images are not used in the current train/validation loop.

### Requirements

- Use all labeled training data available in the competition training set.
- Include both forged and authentic training images in the data split.
- Support authentic samples during training and validation.
- Train authentic samples with all-zero masks unless later experiments show a better approach.
- Represent authentic ground truth rows for official-format scoring as `annotation = "authentic"` and `shape = "authentic"`.
- Represent authentic predictions for official-format scoring as `annotation = "authentic"`.
- Preserve support for forged images with multiple masks.
- Keep the split reproducible with a fixed seed.
- Do not use supplemental/unlabeled/test-only images for training loss or validation scoring.

### Decisions

- Authentic samples should be included in train and validation.
- Authentic samples should use all-zero masks for segmentation training.
- Kaggle-equivalent validation should apply the exact authentic behavior expected by the official metric.
- Train/validation/test splitting should be stratified while preserving authentic/forged source-image pairs.
- Use an 80/10/10 grouped stratified split over all labeled competition training data: train, validation, and local holdout test.
- Authentic and forged files sharing the same stem should stay in the same split, because they are paired views of the same source image.
- A Max-style label-stratified split that does not preserve pairs may be implemented later only as an explicit diagnostic comparison, not as the default model-selection split.
- Use all labeled competition training data across the train/validation split; do not keep `train_subset`/`val_subset` limits enabled for normal baseline runs.
- Validation is used for model selection, threshold/post-processing tuning, and iterative development.
- The local holdout test split should remain untouched until final local review/reporting.
- Supplemental images can be treated as test/inference-only images by default, not as training or validation labels.
- Supplemental data may become an explicit later configuration option only after verifying label/mask compatibility and competition-rule safety.
- Keep loss handling simple: authentic images contribute to the segmentation loss with all-zero masks.
- Do not introduce class-balanced sampling, loss weighting, or forged/authentic ratio tuning in the first refactor.

### Acceptance Criteria

- The baseline can train and validate with both forged and authentic samples.
- Authentic samples do not crash mask loading.
- Validation includes authentic cases, so false positive behavior is measured.
- Kaggle-equivalent validation works for both forged and authentic samples.
- Official-format reference paths can produce solution/submission rows for both forged and authentic samples.
- Normal baseline runs use all labeled training data assigned to the train/validation/test split.

## Workstream 3: Deprecated Files

### Current State

`main.py` is a scaffold placeholder.

`main.ipynb` is an exploratory/prototype notebook and contains early DINOv2 baseline ideas.

The maintained baseline code has moved into `train_baseline.py`, `configs/`, `models/`, `datasets/`, `engine/`, `inference/`, and utility modules.

### Requirements

- Clearly mark `main.py` and `main.ipynb` as deprecated.
- Do not delete them in the first pass unless the team agrees.
- Make the maintained entrypoints obvious.

### Acceptance Criteria

- A new contributor will not mistake `main.py` or `main.ipynb` for the current baseline.
- The deprecation status is visible without reading this spec.

## Workstream 4: Kaggle Notebook Comparison

### Current State

The older pipeline doc and `main.ipynb` credit the Kaggle notebook as inspiration. Some details in `docs/pipeline_suggestion.md` are stale, speculative, or not implemented.

The Kaggle notebook comparison is important and should happen before detailed pipeline docs are rewritten.

The corrected Kaggle URL was checked through the Kaggle kernel pull endpoint. The notebook source is readable without requiring a local download at the time this spec was written.

### Confirmed Kaggle Notebook Behavior

The Kaggle notebook is primarily an inference/submission notebook, not a training notebook.

Confirmed details:

- Public title: `DINOv2 Base [0.332] | High-Res 4500px | Robust Inf`.
- Uses DINOv2 Base as a frozen visual backbone.
- Loads DINOv2 through Hugging Face `AutoImageProcessor` and `AutoModel`.
- Loads trained segmentation weights from an external Kaggle dataset path, rather than training inside the notebook.
- Uses a tiny CNN decoder with channel shape `768 -> 384 -> 192 -> 96 -> 1`.
- Uses `window_size = 518`.
- Uses `stride = 300`.
- Uses `batch_size = 32`.
- Uses `use_tta = True`.
- Uses `min_mean_conf = 0.19`.
- Uses `alpha_grad = 0.50`.
- Uses `min_pixel_size = 50`.
- Uses `TIME_LIMIT_HOURS = 8.5`.
- The markdown describes 4500px high-resolution inference, but the code inspected uses `MAX_IMG_SIZE = 3000`.
- Uses 4-way TTA in code: original, horizontal flip, vertical flip, and horizontal+vertical flip.
- Runs local sliding-window inference for image detail.
- Runs global resized-image inference for context.
- Fuses global and local probability maps as `0.4 * global + 0.6 * local`.
- Uses Sobel gradient boosting and Gaussian blur before thresholding.
- Uses morphological close with a `7x7` kernel and open with a `3x3` kernel.
- Uses connected-component filtering and hole filling.
- Returns `authentic` when the final mask is empty.
- Encodes non-empty masks as JSON RLE strings.
- Writes `submission.csv`.
- Has a time-limit fallback that fills remaining images as `authentic`.

### Current Repo Compared To Kaggle Notebook

| Area | Kaggle notebook | Current repo baseline | Spec implication |
| --- | --- | --- | --- |
| Primary purpose | Inference/submission | Training plus validation | Keep training baseline separate from final inference/submission path. |
| DINO loader | Hugging Face local model path | `torch.hub` from `facebookresearch/dinov2` | Keep `torch.hub` for training because the repo trains the model itself from the official Meta DINOv2 source. |
| DINO variant | Base | `dinov2_vitb14` base | Same broad model family. |
| Encoder | Frozen | Frozen by default | Aligned for baseline; later improvement may unfreeze selectively. |
| Decoder | Tiny CNN `768 -> 384 -> 192 -> 96 -> 1` | Very similar `DinoTinyDecoder` | Decoder is copied/adapted conceptually; improvement can focus on capacity/structure. |
| Input during training | Not applicable in notebook | `target_size = 448` | Training resolution is much smaller than notebook inference resolution. |
| Inference window | `518` | `448` default sliding window | Final inference can mirror notebook-style settings, but should remain configurable. |
| Sliding stride | `300` | `224` | Final inference can mirror notebook-style settings, but should remain configurable. |
| Patch aggregation | Simple average/count map | Gaussian weighted aggregation | Support both notebook-style count averaging and current Gaussian weighting as configurable options. |
| TTA | 4-way TTA | Not implemented in current DINO sliding-window path | Use for final inference; optional for validation if runtime allows. |
| Global/local fusion | Implemented `0.4/0.6` | Not implemented | Use by default in final inference; expose as an optional validation flag. |
| Post-processing | Sobel boost, blur, threshold `0.19`, morphology, components, fill holes | Probability hardening, morphology, small-component filter | See Workstream 1: current repo defaults first, Kaggle-inspired changes later. |
| Authentic output | Explicit `authentic` for empty mask | Training currently ignores authentic images | See Workstream 2 for authentic-data decisions. |
| RLE/submission | Implemented | RLE helpers exist, no full submission entrypoint | Add final inference/submission path after metric/data refactor. |
| Metric | Optimized for public LB submission behavior | Fast pixel F1 helper during validation | See Workstream 1 for metric decisions. |

### Decisions Suggested By Comparison

- Add a separate inference/submission entrypoint rather than overloading `train_baseline.py`.
- Keep `torch.hub` DINO loading for training; do not switch training to Hugging Face just because the inference notebook uses it.
- Final inference can mirror notebook-style high-resolution settings, but window size and stride should be configurable.
- Global/local fusion should be enabled by default for final inference and exposed as an optional validation flag.
- TTA should be available for final inference; validation use can be optional/configurable.
- Support both current Gaussian weighted patch aggregation and notebook-style count-map averaging.
- The implementation plan should choose concrete default inference values, with notebook-style settings available as configurable options.
- Treat the Kaggle notebook as an inference/post-processing reference, not a complete training baseline.
- Compare Kaggle-inspired post-processing variants later, after Workstream 1 and Workstream 2 make validation trustworthy.

### Requirements

- Compare the repository baseline against the Kaggle inspiration notebook.
- Identify what was copied/adapted, what was intentionally changed, and what is missing.
- Separate confirmed differences from assumptions.
- Use the comparison to inform later performance work.

### Comparison Areas

Metric representation, thresholding/post-processing defaults, and authentic handling are covered by Workstream 1 and Workstream 2. The notebook comparison should focus on architecture and inference choices that are not already decided elsewhere.

- DINOv2 model variant.
- Input resolution and high-resolution inference strategy.
- Encoder freezing or fine-tuning.
- Decoder architecture and capacity.
- Sliding-window size, stride, and weighting.
- Test-time augmentation, including flips if used in the notebook.
- Configurable post-processing variants, including confidence seeding and component filtering after metric/data behavior is trustworthy.
- Submission entrypoint structure.

### Acceptance Criteria

- A comparison document or section exists before performance changes are planned.
- The team can point to concrete reasons for keeping or changing each major baseline design choice.

## Workstream 5: Pipeline Documentation

### Current State

`docs/pipeline_suggestion.md` is older than the current implementation.

It mentions some ideas that are not clearly implemented, such as TTA, global/local fusion, and named functions like `pipeline_fusion` or `enhanced_adaptive_mask`.

### Requirements

- Wait until after the metric/data refactor before rewriting this document.
- Convert it from speculative pipeline suggestion to accurate baseline pipeline documentation.
- Clearly separate implemented behavior from future improvement ideas.
- Keep detailed documentation in `docs/`, not in the root README.

### Acceptance Criteria

- The pipeline doc matches the code after refactor.
- Stale or misleading implementation claims are removed.
- Future ideas are labeled as future ideas.

## Workstream 6: README And Documentation Structure

### Current State

The root README is minimal and does not identify the maintained baseline files.

This is acceptable in spirit, but it should still prevent obvious confusion.

### Requirements

- Keep root README minimal.
- Mention the correct entrypoint.
- Mention deprecated files briefly.
- Link to detailed docs instead of expanding the root README too much.
- Add or update sub-docs after the code refactor, not before.

### Acceptance Criteria

- A new contributor knows where to start.
- Detailed explanations live in docs.
- The root README is not overloaded.

## Workstream 7: Baseline Performance Improvements

### Current State

The current model is a frozen DINOv2 encoder with a small CNN decoder. It is a reasonable baseline, but likely underdeveloped.

Performance work should happen after metric and data behavior are trustworthy.

### Candidate Improvements

- Larger or better-structured decoder.
- Skip connections or multi-scale DINO feature use if available.
- Unfreeze some DINO layers after decoder warmup.
- Better loss function, such as BCE plus Dice/Focal variants.
- Authentic/forged sampling balance.
- Kaggle-inspired post-processing experiments after Workstream 1 and Workstream 2 are implemented.
- Configurable post-processing experiments inspired by `max_individual_project`, such as confidence-seeded component keeping, smoothing, opening/closing, hole filling, and minimum component area.
- Test-time augmentation for validation/inference.
- Higher-resolution validation or inference settings.
- Checkpoint saving and reproducible experiment logging.

### Requirements

- Compare improvement ideas against the Kaggle notebook first.
- Keep changes small enough to measure.
- Do not mix performance changes with metric/data correctness changes.

### Acceptance Criteria

- Performance experiments have named configs or documented settings.
- The validation metric used for comparison is clearly stated.
- Improvements can be compared to the pre-refactor baseline.

## Proposed Chunk Order

1. Spec document. This file.
2. Plan document for the core refactor.
3. Implement data inventory, sample metadata, and 80/10/10 stratified train/validation/local-test splitting.
4. Implement dataset support for authentic samples, all-zero authentic masks, and forged instance-mask metadata.
5. Implement verified Kaggle-equivalent `kaggle_score` validation, with official-style solution/submission rows retained for equivalence checks.
6. Wire model selection and checkpointing to verified `kaggle_score`.
7. Add opt-in fallback/debug `pixel_f1`.
8. Review metric and data behavior together, because Kaggle-equivalent scoring depends on authentic handling and forged instance masks.
9. Mark deprecated files clearly.
10. Add a separate inference/submission entrypoint with configurable notebook-inspired options.
11. Update pipeline docs and root README after code behavior is stable.
12. Plan performance improvements.

## Plan Document Requirements

The plan document must be directly derived from this spec and include:

- File-level implementation details: which files to touch, what to add, and what to modify.
- Data model details for sample metadata, split representation, authentic samples, and forged instance masks.
- Validation details for computing a direct Kaggle-equivalent instance score and verifying it against official-style `solution` and `submission` rows with `recodai_f1.score`.
- Checkpoint/model-selection behavior using verified `kaggle_score`.
- Configuration details for full validation, optional `pixel_f1`, and inference-time options.
- Test and verification steps, including small local smoke tests that do not require a full training run.
- Best-practice notes where choices matter, especially around stratified splitting, held-out local test data, metric-faithful validation, and avoiding validation leakage.
- Explicit non-goals for the implementation chunk, so performance experiments and documentation cleanup do not get mixed into metric/data correctness.

## Immediate Next Step

Draft the plan document for the core refactor. The first implementation chunk should start with data/split/sample metadata support, because verified `kaggle_score` validation depends on it.
