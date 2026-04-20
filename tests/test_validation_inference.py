from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch.utils.data import RandomSampler

from dataset_utils import SampleRecord
from engine.validation_inference import (
    ValidationPrediction,
    collect_validation_predictions,
    score_validation_predictions,
)
from engine.validate_loop import validate_one_epoch


def _sample(sample_id: str, label: str = "authentic") -> SampleRecord:
    image_dir = "forged" if label == "forged" else "authentic"
    mask_paths = (Path(f"data/train_masks/{sample_id}.npy"),) if label == "forged" else tuple()
    return SampleRecord(
        sample_id=f"{label}:{sample_id}",
        case_id=sample_id,
        label=label,
        image_path=Path(f"data/train_images/{image_dir}/{sample_id}.png"),
        mask_paths=mask_paths,
        group_id=sample_id,
        split="val",
    )


class BatchCountingModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.batch_sizes: list[int] = []

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.batch_sizes.append(int(x.shape[0]))
        logits = torch.zeros((x.shape[0], 1, x.shape[-2], x.shape[-1]), device=x.device)
        logits[:, :, 1:3, 1:3] = 4.0
        return logits


class SequentialBatchLoader:
    def __init__(self, batches) -> None:
        self.batches = list(batches)
        self.sampler = object()

    def __iter__(self):
        return iter(self.batches)


class ValidationInferenceTests(unittest.TestCase):
    def test_collect_direct_predictions_runs_one_forward_per_loader_batch(self) -> None:
        model = BatchCountingModel()
        samples = [_sample("1"), _sample("2"), _sample("3")]
        imgs_1 = torch.zeros((2, 3, 8, 8), dtype=torch.float32)
        masks_1 = torch.zeros((2, 1, 8, 8), dtype=torch.float32)
        imgs_2 = torch.zeros((1, 3, 8, 8), dtype=torch.float32)
        masks_2 = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
        loader = [(imgs_1, masks_1), (imgs_2, masks_2)]

        predictions = collect_validation_predictions(
            model=model,
            val_loader=loader,
            val_samples=samples,
            device=torch.device("cpu"),
            inference_mode="direct",
            sliding_window_fn=None,
            probability_dtype="float32",
            collect_masks=False,
        )

        self.assertEqual(model.batch_sizes, [2, 1])
        self.assertEqual([p.sample.sample_id for p in predictions], ["authentic:1", "authentic:2", "authentic:3"])
        self.assertTrue(all(isinstance(p, ValidationPrediction) for p in predictions))
        self.assertTrue(all(p.probability.shape == (8, 8) for p in predictions))
        self.assertTrue(all(p.probability.dtype == np.float32 for p in predictions))
        self.assertTrue(all(p.gt_union_mask is None for p in predictions))

    def test_validate_one_epoch_direct_mode_orchestrates_without_sliding_or_masks(self) -> None:
        model = BatchCountingModel()
        samples = [_sample("1", label="authentic")]
        imgs = torch.zeros((1, 3, 8, 8), dtype=torch.float32)
        loader = SequentialBatchLoader([(imgs, object())])

        def forbidden_sliding_window(*args, **kwargs):
            raise AssertionError("direct validation must not call sliding_window_fn")

        result = validate_one_epoch(
            model=model,
            val_loader=loader,
            val_samples=samples,
            device=torch.device("cpu"),
            sliding_window_fn=forbidden_sliding_window,
            pixel_util=None,
            pred_threshold=0.99,
            harden_temperature=1.0,
            hard_clip_low=0.0,
            hard_clip_high=1.0,
            min_component_area=0,
            epoch_idx=0,
            compute_pixel_f1=False,
            verify_score_equivalence=False,
            inference_mode="direct",
            probability_dtype="float32",
            validation_transfer_mode="per_batch",
            log_timing=False,
        )

        self.assertEqual(model.batch_sizes, [1])
        self.assertEqual(result["kaggle_score"], 1.0)
        self.assertEqual(result["pixel_f1"], None)
        self.assertEqual(result["num_samples"], 1)
        self.assertEqual(result["num_authentic"], 1)
        self.assertEqual(result["num_forged"], 0)
        self.assertIn("inference_seconds", result)
        self.assertIn("postprocess_seconds", result)
        self.assertIn("scoring_seconds", result)
        self.assertEqual(result["validation_inference_mode"], "direct")
        self.assertEqual(result["probability_dtype"], "float32")
        self.assertEqual(result["validation_transfer_mode"], "per_batch")

    def test_collect_direct_predictions_can_keep_cpu_masks_for_pixel_f1(self) -> None:
        model = BatchCountingModel()
        samples = [_sample("1"), _sample("2")]
        imgs = torch.zeros((2, 3, 8, 8), dtype=torch.float32)
        masks = torch.zeros((2, 1, 8, 8), dtype=torch.float32)
        masks[1, 0, 4:6, 4:6] = 1.0
        loader = [(imgs, masks)]

        predictions = collect_validation_predictions(
            model=model,
            val_loader=loader,
            val_samples=samples,
            device=torch.device("cpu"),
            inference_mode="direct",
            sliding_window_fn=None,
            probability_dtype="float16",
            collect_masks=True,
        )

        self.assertEqual(predictions[0].probability.dtype, np.float16)
        self.assertIsNotNone(predictions[0].gt_union_mask)
        self.assertEqual(int(predictions[1].gt_union_mask.sum()), 4)

    def test_collect_direct_predictions_can_accumulate_probabilities_on_gpu(self) -> None:
        model = BatchCountingModel()
        samples = [_sample("1"), _sample("2"), _sample("3")]
        imgs_1 = torch.zeros((2, 3, 8, 8), dtype=torch.float32)
        masks_1 = torch.zeros((2, 1, 8, 8), dtype=torch.float32)
        imgs_2 = torch.zeros((1, 3, 8, 8), dtype=torch.float32)
        masks_2 = torch.zeros((1, 1, 8, 8), dtype=torch.float32)
        loader = [(imgs_1, masks_1), (imgs_2, masks_2)]

        original_cpu = torch.Tensor.cpu
        cpu_calls = {"count": 0}

        def cpu_proxy(self: torch.Tensor, *args, **kwargs):
            cpu_calls["count"] += 1
            return original_cpu(self, *args, **kwargs)

        with patch.object(torch.Tensor, "cpu", new=cpu_proxy):
            predictions = collect_validation_predictions(
                model=model,
                val_loader=loader,
                val_samples=samples,
                device=torch.device("cpu"),
                inference_mode="direct",
                sliding_window_fn=None,
                probability_dtype="float16",
                transfer_mode="accumulate_gpu",
                collect_masks=False,
            )

        self.assertEqual(model.batch_sizes, [2, 1])
        self.assertEqual([p.sample.sample_id for p in predictions], ["authentic:1", "authentic:2", "authentic:3"])
        self.assertTrue(all(p.probability.dtype == np.float16 for p in predictions))
        self.assertTrue(all(p.gt_union_mask is None for p in predictions))
        self.assertEqual(cpu_calls["count"], 1)

    def test_collect_direct_predictions_accumulate_mode_rejects_shape_changes(self) -> None:
        model = BatchCountingModel()
        samples = [_sample("1"), _sample("2"), _sample("3")]
        imgs_1 = torch.zeros((2, 3, 8, 8), dtype=torch.float32)
        masks_1 = torch.zeros((2, 1, 8, 8), dtype=torch.float32)
        imgs_2 = torch.zeros((1, 3, 6, 6), dtype=torch.float32)
        masks_2 = torch.zeros((1, 1, 6, 6), dtype=torch.float32)
        loader = [(imgs_1, masks_1), (imgs_2, masks_2)]

        with self.assertRaisesRegex(ValueError, "Validation probability shape changed"):
            collect_validation_predictions(
                model=model,
                val_loader=loader,
                val_samples=samples,
                device=torch.device("cpu"),
                inference_mode="direct",
                sliding_window_fn=None,
                probability_dtype="float32",
                collect_masks=False,
                transfer_mode="accumulate_gpu",
            )

    def test_collect_predictions_rejects_missing_sliding_window_function_in_sliding_mode(self) -> None:
        model = BatchCountingModel()
        samples = [_sample("1")]
        imgs = torch.zeros((1, 3, 8, 8), dtype=torch.float32)
        masks = torch.zeros((1, 1, 8, 8), dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "sliding_window_fn"):
            collect_validation_predictions(
                model=model,
                val_loader=[(imgs, masks)],
                val_samples=samples,
                device=torch.device("cpu"),
                inference_mode="sliding",
                sliding_window_fn=None,
                probability_dtype="float32",
                collect_masks=False,
            )

    def test_collect_predictions_rejects_unknown_transfer_mode(self) -> None:
        model = BatchCountingModel()
        samples = [_sample("1")]
        imgs = torch.zeros((1, 3, 8, 8), dtype=torch.float32)
        masks = torch.zeros((1, 1, 8, 8), dtype=torch.float32)

        with self.assertRaisesRegex(ValueError, "Unknown transfer_mode"):
            collect_validation_predictions(
                model=model,
                val_loader=[(imgs, masks)],
                val_samples=samples,
                device=torch.device("cpu"),
                inference_mode="direct",
                sliding_window_fn=None,
                probability_dtype="float32",
                transfer_mode="mystery",
                collect_masks=False,
            )

    def test_collect_predictions_rejects_random_sampler_before_iteration(self) -> None:
        model = BatchCountingModel()
        samples = [_sample("1")]

        class LoaderLike:
            def __init__(self) -> None:
                self.sampler = RandomSampler(range(1))

            def __iter__(self):
                raise AssertionError("collect_validation_predictions should reject RandomSampler before iterating")

        with self.assertRaisesRegex(ValueError, "shuffle=False|sequential order"):
            collect_validation_predictions(
                model=model,
                val_loader=LoaderLike(),
                val_samples=samples,
                device=torch.device("cpu"),
                inference_mode="direct",
                sliding_window_fn=None,
                probability_dtype="float32",
                collect_masks=False,
            )

    def test_collect_sliding_predictions_transfers_batch_to_cpu_once(self) -> None:
        model = BatchCountingModel()
        samples = [_sample("1"), _sample("2")]
        imgs = torch.zeros((2, 3, 8, 8), dtype=torch.float32)
        masks = torch.zeros((2, 1, 8, 8), dtype=torch.float32)

        original_cpu = torch.Tensor.cpu

        def sliding_window_fn(img: torch.Tensor, model: torch.nn.Module, device: torch.device) -> torch.Tensor:
            del model, device
            return torch.zeros((1, 1, img.shape[-2], img.shape[-1]), device=img.device)

        cpu_calls = {"count": 0}

        def cpu_proxy(self: torch.Tensor, *args, **kwargs):
            cpu_calls["count"] += 1
            return original_cpu(self, *args, **kwargs)

        with patch.object(torch.Tensor, "cpu", new=cpu_proxy):
            predictions = collect_validation_predictions(
                model=model,
                val_loader=[(imgs, masks)],
                val_samples=samples,
                device=torch.device("cpu"),
                inference_mode="sliding",
                sliding_window_fn=sliding_window_fn,
                probability_dtype="float32",
                collect_masks=False,
            )

        self.assertEqual(len(predictions), 2)
        self.assertEqual(cpu_calls["count"], 1)

    def test_score_validation_predictions_postprocesses_after_prediction_collection(self) -> None:
        authentic = _sample("1", label="authentic")
        forged = _sample("2", label="forged")
        predictions = [
            ValidationPrediction(sample=authentic, probability=np.zeros((8, 8), dtype=np.float32)),
            ValidationPrediction(sample=forged, probability=np.ones((8, 8), dtype=np.float32)),
        ]
        fake_gt = np.ones((8, 8), dtype=np.uint8)

        def fake_load_gt(sample: SampleRecord, shape: tuple[int, int]) -> list[np.ndarray]:
            return [] if sample.label == "authentic" else [fake_gt]

        with patch("engine.validation_inference.load_resized_instance_masks", side_effect=fake_load_gt):
            result = score_validation_predictions(
                predictions=predictions,
                pixel_util=None,
                pred_threshold=0.5,
                harden_temperature=1.0,
                hard_clip_low=0.0,
                hard_clip_high=1.0,
                min_component_area=0,
                compute_pixel_f1=False,
                verify_score_equivalence=False,
            )

        self.assertEqual(result["num_samples"], 2)
        self.assertEqual(result["num_authentic"], 1)
        self.assertEqual(result["num_forged"], 1)
        self.assertEqual(result["pixel_f1"], None)
        self.assertAlmostEqual(result["kaggle_score"], 1.0)

    def test_score_validation_predictions_uses_configured_postprocess_path(self) -> None:
        prediction = ValidationPrediction(
            sample=_sample("1", label="authentic"),
            probability=np.zeros((8, 8), dtype=np.float32),
        )

        with patch(
            "engine.validation_inference.post_process_prediction",
            return_value=np.zeros((8, 8), dtype=np.float32),
        ) as postprocess:
            result = score_validation_predictions(
                predictions=[prediction],
                pixel_util=object(),
                pred_threshold=0.42,
                harden_temperature=0.7,
                hard_clip_low=0.1,
                hard_clip_high=0.9,
                min_component_area=50,
                compute_pixel_f1=False,
                verify_score_equivalence=False,
                confident_threshold=0.77,
                smooth_probabilities=False,
                fill_holes=False,
                apply_opening=False,
                apply_closing=False,
                keep_confident_seeded_components=True,
            )

        self.assertAlmostEqual(result["kaggle_score"], 1.0)
        postprocess.assert_called_once()
        self.assertEqual(postprocess.call_args.kwargs["threshold"], 0.42)
        self.assertEqual(postprocess.call_args.kwargs["harden_temperature"], 0.7)
        self.assertEqual(postprocess.call_args.kwargs["hard_clip_low"], 0.1)
        self.assertEqual(postprocess.call_args.kwargs["hard_clip_high"], 0.9)
        self.assertEqual(postprocess.call_args.kwargs["min_component_area"], 50)
        self.assertEqual(postprocess.call_args.kwargs["confident_threshold"], 0.77)
        self.assertFalse(postprocess.call_args.kwargs["smooth_probabilities"])
        self.assertFalse(postprocess.call_args.kwargs["fill_holes"])
        self.assertFalse(postprocess.call_args.kwargs["apply_opening"])
        self.assertFalse(postprocess.call_args.kwargs["apply_closing"])
        self.assertTrue(postprocess.call_args.kwargs["keep_confident_seeded_components"])

    def test_score_validation_predictions_requires_masks_when_pixel_f1_enabled(self) -> None:
        prediction = ValidationPrediction(sample=_sample("1"), probability=np.zeros((8, 8), dtype=np.float32))

        with self.assertRaisesRegex(ValueError, "gt_union_mask"):
            score_validation_predictions(
                predictions=[prediction],
                pixel_util=None,
                pred_threshold=0.5,
                harden_temperature=1.0,
                hard_clip_low=0.0,
                hard_clip_high=1.0,
                min_component_area=0,
                compute_pixel_f1=True,
                verify_score_equivalence=False,
            )
