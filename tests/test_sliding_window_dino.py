import unittest

import torch

from inference.sliding_window_dino_impl import compute_window_starts, sliding_window_dino


class CountingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0
        self.batch_sizes: list[int] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.calls += 1
        self.batch_sizes.append(int(x.shape[0]))
        values = torch.arange(
            x.shape[-2] * x.shape[-1], device=x.device, dtype=torch.float32
        ).view(1, 1, x.shape[-2], x.shape[-1])
        return values.repeat(x.shape[0], 1, 1, 1)


class SlidingWindowDinoTests(unittest.TestCase):
    def test_compute_window_starts_exact_size_uses_single_start(self) -> None:
        self.assertEqual(compute_window_starts(length=448, patch_size=448, stride=224), [0])

    def test_compute_window_starts_smaller_than_patch_uses_single_start(self) -> None:
        self.assertEqual(compute_window_starts(length=320, patch_size=448, stride=224), [0])

    def test_compute_window_starts_larger_image_includes_final_aligned_start_without_duplicates(self) -> None:
        starts = compute_window_starts(length=1000, patch_size=448, stride=224)

        self.assertEqual(starts, [0, 224, 448, 552])
        self.assertEqual(len(starts), len(set(starts)))
        self.assertEqual(starts[-1] + 448, 1000)

    def test_sliding_window_exact_size_calls_model_once(self) -> None:
        model = CountingModel()
        img = torch.zeros((3, 448, 448))

        pred = sliding_window_dino(
            img=img,
            model=model,
            device=torch.device("cpu"),
            patch_size=448,
            stride=224,
            batch_size=8,
        )

        self.assertEqual(tuple(pred.shape), (448, 448))
        self.assertTrue(torch.equal(pred, torch.arange(448 * 448, dtype=pred.dtype).view(448, 448)))
        self.assertEqual(model.calls, 1)
        self.assertEqual(model.batch_sizes, [1])


if __name__ == "__main__":
    unittest.main()
