-LUC-Forgery-Kaggle$ uv run python -m einar_busternet.train --fusion-mode binary_union
Using device: cuda
Using seed: 42
{'total': 5128, 'by_label': {'forged': 2751, 'authentic': 2377}, 'by_split': {'train': {'total': 4101, 'forged': 2200, 'authentic': 1901}, 'val': {'total': 514, 'forged': 276, 'authentic': 238}, 'test': {'total': 513, 'forged': 275, 'authentic': 238}}}
Split sizes: train_effective=3802, val=514, test=513 (held out)
Using cache found in /home/einar/.cache/torch/hub/facebookresearch_dinov2_main
stage 1 epoch 1 train: 100%|██████████████████████████| 119/119 [01:08<00:00,  1.74it/s]
Stage 1 epoch 1: loss=1.0996 mani=0.4875 simi=0.6122
stage 1 epoch 2 train: 100%|██████████████████████████| 119/119 [01:05<00:00,  1.81it/s]
Stage 1 epoch 2: loss=0.9522 mani=0.3687 simi=0.5835
stage 1 epoch 3 train: 100%|██████████████████████████| 119/119 [01:07<00:00,  1.76it/s]
Stage 1 epoch 3: loss=0.9221 mani=0.3429 simi=0.5791
stage 1 epoch 4 train: 100%|██████████████████████████| 119/119 [01:06<00:00,  1.78it/s]
Stage 1 epoch 4: loss=0.8890 mani=0.3125 simi=0.5765
stage 1 epoch 5 train: 100%|██████████████████████████| 119/119 [01:07<00:00,  1.77it/s]
Stage 1 epoch 5: loss=0.8762 mani=0.3022 simi=0.5740
stage 1 epoch 6 train: 100%|██████████████████████████| 119/119 [01:07<00:00,  1.77it/s]
Stage 1 epoch 6: loss=0.8488 mani=0.2791 simi=0.5697
stage 1 epoch 7 train: 100%|██████████████████████████| 119/119 [01:08<00:00,  1.73it/s]
Stage 1 epoch 7: loss=0.8368 mani=0.2692 simi=0.5676
stage 1 epoch 8 train: 100%|██████████████████████████| 119/119 [01:08<00:00,  1.75it/s]
Stage 1 epoch 8: loss=0.8226 mani=0.2577 simi=0.5648
stage 1 epoch 9 train: 100%|██████████████████████████| 119/119 [01:08<00:00,  1.74it/s]
Stage 1 epoch 9: loss=0.8234 mani=0.2611 simi=0.5624
stage 1 epoch 10 train: 100%|█████████████████████████| 119/119 [01:08<00:00,  1.73it/s]
Stage 1 epoch 10: loss=0.8000 mani=0.2394 simi=0.5606
stage 1 epoch 11 train: 100%|█████████████████████████| 119/119 [01:08<00:00,  1.73it/s]
Stage 1 epoch 11: loss=0.7926 mani=0.2346 simi=0.5579
stage 1 epoch 12 train: 100%|█████████████████████████| 119/119 [01:10<00:00,  1.69it/s]
Stage 1 epoch 12: loss=0.7857 mani=0.2295 simi=0.5562
stage 1 epoch 13 train: 100%|█████████████████████████| 119/119 [01:07<00:00,  1.77it/s]
Stage 1 epoch 13: loss=0.7714 mani=0.2180 simi=0.5534
stage 1 epoch 14 train: 100%|█████████████████████████| 119/119 [01:10<00:00,  1.70it/s]
Stage 1 epoch 14: loss=0.7634 mani=0.2099 simi=0.5534
stage 1 epoch 15 train: 100%|█████████████████████████| 119/119 [01:09<00:00,  1.70it/s]
Stage 1 epoch 15: loss=0.7560 mani=0.2052 simi=0.5508
stage 1 epoch 16 train: 100%|█████████████████████████| 119/119 [01:07<00:00,  1.75it/s]
Stage 1 epoch 16: loss=0.7519 mani=0.2028 simi=0.5491
stage 1 epoch 17 train: 100%|█████████████████████████| 119/119 [01:07<00:00,  1.77it/s]
Stage 1 epoch 17: loss=0.7464 mani=0.1990 simi=0.5474
stage 1 epoch 18 train: 100%|█████████████████████████| 119/119 [01:09<00:00,  1.71it/s]
Stage 1 epoch 18: loss=0.7484 mani=0.2032 simi=0.5451
stage 1 epoch 19 train: 100%|█████████████████████████| 119/119 [01:07<00:00,  1.76it/s]
Stage 1 epoch 19: loss=0.7374 mani=0.1937 simi=0.5436
stage 1 epoch 20 train: 100%|█████████████████████████| 119/119 [01:08<00:00,  1.74it/s]
Stage 1 epoch 20: loss=0.7351 mani=0.1928 simi=0.5423
epoch 1 train: 100%|██████████████████████████████████| 119/119 [01:07<00:00,  1.77it/s]
epoch 21 val inference: 100%|███████████████████████████| 17/17 [00:21<00:00,  1.29s/it]
Validation timing (epoch 21): inference=22.88s postprocess=5.62s scoring=0.14s
Stage 2 epoch 1: avg_loss=0.6087 kaggle_score=0.5286 balanced_score=0.4666
  -> New best BusterNet saved by kaggle_score=0.5286
  -> New balanced BusterNet saved by balanced_score=0.4666
epoch 2 train: 100%|██████████████████████████████████| 119/119 [01:08<00:00,  1.73it/s]
epoch 22 val inference: 100%|███████████████████████████| 17/17 [00:22<00:00,  1.32s/it]
Validation timing (epoch 22): inference=22.38s postprocess=5.06s scoring=0.12s
Stage 2 epoch 2: avg_loss=0.5443 kaggle_score=0.5321 balanced_score=0.4663
  -> New best BusterNet saved by kaggle_score=0.5321
epoch 3 train: 100%|██████████████████████████████████| 119/119 [01:08<00:00,  1.73it/s]
epoch 23 val inference: 100%|███████████████████████████| 17/17 [00:23<00:00,  1.36s/it]
Validation timing (epoch 23): inference=22.96s postprocess=5.08s scoring=0.12s
Stage 2 epoch 3: avg_loss=0.5364 kaggle_score=0.5251 balanced_score=0.4718
  -> New balanced BusterNet saved by balanced_score=0.4718
epoch 4 train: 100%|██████████████████████████████████| 119/119 [01:07<00:00,  1.75it/s]
epoch 24 val inference: 100%|███████████████████████████| 17/17 [00:22<00:00,  1.33s/it]
Validation timing (epoch 24): inference=22.45s postprocess=4.91s scoring=0.14s
Stage 2 epoch 4: avg_loss=0.5369 kaggle_score=0.5309 balanced_score=0.4700
epoch 5 train: 100%|██████████████████████████████████| 119/119 [01:06<00:00,  1.80it/s]
epoch 25 val inference: 100%|███████████████████████████| 17/17 [00:22<00:00,  1.32s/it]
Validation timing (epoch 25): inference=22.29s postprocess=4.80s scoring=0.11s
Stage 2 epoch 5: avg_loss=0.5400 kaggle_score=0.5349 balanced_score=0.4661
  -> New best BusterNet saved by kaggle_score=0.5349
epoch 6 train: 100%|██████████████████████████████████| 119/119 [01:08<00:00,  1.74it/s]
epoch 26 val inference: 100%|███████████████████████████| 17/17 [00:22<00:00,  1.33s/it]
Validation timing (epoch 26): inference=22.43s postprocess=4.99s scoring=0.12s
Stage 2 epoch 6: avg_loss=0.5301 kaggle_score=0.5329 balanced_score=0.4673
epoch 7 train: 100%|██████████████████████████████████| 119/119 [01:06<00:00,  1.78it/s]
epoch 27 val inference: 100%|███████████████████████████| 17/17 [00:22<00:00,  1.33s/it]
Validation timing (epoch 27): inference=22.35s postprocess=4.96s scoring=0.13s
Stage 2 epoch 7: avg_loss=0.5333 kaggle_score=0.5289 balanced_score=0.4718
  -> New balanced BusterNet saved by balanced_score=0.4718
epoch 8 train: 100%|██████████████████████████████████| 119/119 [01:09<00:00,  1.72it/s]
epoch 28 val inference: 100%|███████████████████████████| 17/17 [00:22<00:00,  1.31s/it]
Validation timing (epoch 28): inference=22.07s postprocess=5.06s scoring=0.15s
Stage 2 epoch 8: avg_loss=0.5324 kaggle_score=0.5271 balanced_score=0.4723
  -> New balanced BusterNet saved by balanced_score=0.4723
epoch 9 train: 100%|██████████████████████████████████| 119/119 [01:08<00:00,  1.72it/s]
epoch 29 val inference: 100%|███████████████████████████| 17/17 [00:22<00:00,  1.31s/it]
Validation timing (epoch 29): inference=22.14s postprocess=5.01s scoring=0.13s
Stage 2 epoch 9: avg_loss=0.5322 kaggle_score=0.5294 balanced_score=0.4742
  -> New balanced BusterNet saved by balanced_score=0.4742
epoch 10 train: 100%|█████████████████████████████████| 119/119 [01:08<00:00,  1.73it/s]
epoch 30 val inference: 100%|███████████████████████████| 17/17 [00:22<00:00,  1.31s/it]
Validation timing (epoch 30): inference=22.23s postprocess=5.11s scoring=0.13s
Stage 2 epoch 10: avg_loss=0.5328 kaggle_score=0.5339 balanced_score=0.4727
stage 3 epoch 1 train: 100%|██████████████████████████| 119/119 [01:06<00:00,  1.79it/s]
epoch 31 val inference: 100%|███████████████████████████| 17/17 [00:22<00:00,  1.31s/it]
Validation timing (epoch 31): inference=22.22s postprocess=5.00s scoring=0.13s
Stage 3 epoch 1: avg_loss=0.5955 fusion=0.5239 mani=0.1781 simi=0.5379 kaggle_score=0.5285 balanced_score=0.4763
  -> New balanced BusterNet saved by balanced_score=0.4763
stage 3 epoch 2 train: 100%|██████████████████████████| 119/119 [01:08<00:00,  1.73it/s]
epoch 32 val inference: 100%|███████████████████████████| 17/17 [00:20<00:00,  1.23s/it]
Validation timing (epoch 32): inference=21.82s postprocess=4.94s scoring=0.12s
Stage 3 epoch 2: avg_loss=0.5898 fusion=0.5186 mani=0.1755 simi=0.5363 kaggle_score=0.5333 balanced_score=0.4762
stage 3 epoch 3 train: 100%|██████████████████████████| 119/119 [01:08<00:00,  1.73it/s]
epoch 33 val inference: 100%|███████████████████████████| 17/17 [00:20<00:00,  1.23s/it]
Validation timing (epoch 33): inference=21.84s postprocess=5.01s scoring=0.12s
Stage 3 epoch 3: avg_loss=0.5819 fusion=0.5109 mani=0.1739 simi=0.5353 kaggle_score=0.5312 balanced_score=0.4807
  -> New balanced BusterNet saved by balanced_score=0.4807
stage 3 epoch 4 train: 100%|██████████████████████████| 119/119 [01:09<00:00,  1.71it/s]
epoch 34 val inference: 100%|███████████████████████████| 17/17 [00:20<00:00,  1.23s/it]
Validation timing (epoch 34): inference=21.86s postprocess=5.01s scoring=0.13s
Stage 3 epoch 4: avg_loss=0.5820 fusion=0.5111 mani=0.1739 simi=0.5351 kaggle_score=0.5322 balanced_score=0.4823
  -> New balanced BusterNet saved by balanced_score=0.4823
stage 3 epoch 5 train: 100%|██████████████████████████| 119/119 [01:07<00:00,  1.76it/s]
epoch 35 val inference: 100%|███████████████████████████| 17/17 [00:20<00:00,  1.22s/it]
Validation timing (epoch 35): inference=21.56s postprocess=5.02s scoring=0.14s
Stage 3 epoch 5: avg_loss=0.5809 fusion=0.5099 mani=0.1744 simi=0.5350 kaggle_score=0.5337 balanced_score=0.4814
stage 3 epoch 6 train: 100%|██████████████████████████| 119/119 [01:06<00:00,  1.80it/s]
epoch 36 val inference: 100%|███████████████████████████| 17/17 [00:20<00:00,  1.21s/it]
Validation timing (epoch 36): inference=21.50s postprocess=5.01s scoring=0.13s
Stage 3 epoch 6: avg_loss=0.5776 fusion=0.5068 mani=0.1732 simi=0.5344 kaggle_score=0.5340 balanced_score=0.4831
  -> New balanced BusterNet saved by balanced_score=0.4831
stage 3 epoch 7 train: 100%|██████████████████████████| 119/119 [01:07<00:00,  1.76it/s]
epoch 37 val inference: 100%|███████████████████████████| 17/17 [00:21<00:00,  1.24s/it]
Validation timing (epoch 37): inference=22.00s postprocess=5.13s scoring=0.13s
Stage 3 epoch 7: avg_loss=0.5755 fusion=0.5049 mani=0.1723 simi=0.5339 kaggle_score=0.5335 balanced_score=0.4851
  -> New balanced BusterNet saved by balanced_score=0.4851
stage 3 epoch 8 train: 100%|██████████████████████████| 119/119 [01:07<00:00,  1.76it/s]
epoch 38 val inference: 100%|███████████████████████████| 17/17 [00:20<00:00,  1.22s/it]
Validation timing (epoch 38): inference=21.68s postprocess=5.11s scoring=0.12s
Stage 3 epoch 8: avg_loss=0.5750 fusion=0.5043 mani=0.1730 simi=0.5340 kaggle_score=0.5330 balanced_score=0.4840
stage 3 epoch 9 train: 100%|██████████████████████████| 119/119 [01:07<00:00,  1.77it/s]
epoch 39 val inference: 100%|███████████████████████████| 17/17 [00:21<00:00,  1.25s/it]
Validation timing (epoch 39): inference=22.24s postprocess=5.05s scoring=0.12s
Stage 3 epoch 9: avg_loss=0.5756 fusion=0.5050 mani=0.1721 simi=0.5340 kaggle_score=0.5328 balanced_score=0.4831
stage 3 epoch 10 train: 100%|█████████████████████████| 119/119 [01:07<00:00,  1.78it/s]
epoch 40 val inference: 100%|███████████████████████████| 17/17 [00:21<00:00,  1.25s/it]
Validation timing (epoch 40): inference=22.21s postprocess=5.09s scoring=0.13s
Stage 3 epoch 10: avg_loss=0.5742 fusion=0.5036 mani=0.1716 simi=0.5339 kaggle_score=0.5329 balanced_score=0.4847