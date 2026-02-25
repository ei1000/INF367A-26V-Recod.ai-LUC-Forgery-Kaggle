# train_baseline.py Improvements

## Changes Made

1. **GPU Auto-Detection** — Use cuda else cpu
2. **DataLoader Optimizations** — num_workers for parallelism and pin_memory for faster data loading
3. **Learning Rate Scheduling** — Reduced initial LR from 1e-3 to 1e-4. Use ReduceLROnPlateau for adaptive LR adjustment
4. **Gradient Clipping** — Added gradient norm clipping (max=1.0). Prevents exploding gradients
5. **GPU Memory Cleanup** — Clear CUDA cache after each epoch to prevent fragmentation. Important to prevent memory leaks
6. **Best Model Checkpointing** — Track and save model when validation F1 improves
7. **Device Consistency** — Ensure all tensors (train + val) are moved to the correct device

