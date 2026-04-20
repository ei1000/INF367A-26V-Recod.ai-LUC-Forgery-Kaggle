import contextlib
import io
from types import SimpleNamespace
import unittest

from configs.baseline_config import BaselineConfig
from evaluate_validation_postprocess import (
    config_from_checkpoint,
    iter_postprocess_settings,
    parse_args,
    parse_bool_list,
    parse_float_list,
    parse_int_list,
    parse_optional_float_list,
    validate_reproduced_validation_score,
    validate_model_loading_allowed,
)


class EvaluateValidationPostprocessTests(unittest.TestCase):
    def test_config_from_checkpoint_ignores_unknown_keys_and_keeps_defaults(self) -> None:
        checkpoint = {
            "config": {
                "seed": 123,
                "pred_threshold": 0.6,
                "val_subset": 7,
                "unknown_future_field": "ignore-me",
            }
        }

        config = config_from_checkpoint(checkpoint)

        self.assertIsInstance(config, BaselineConfig)
        self.assertEqual(config.seed, 123)
        self.assertEqual(config.pred_threshold, 0.6)
        self.assertEqual(config.val_subset, 7)
        self.assertEqual(config.post_process_confident_threshold, 0.9)
        self.assertFalse(hasattr(config, "unknown_future_field"))

    def test_parse_float_int_bool_and_optional_float_lists(self) -> None:
        self.assertEqual(parse_float_list("0.1, 0.2"), [0.1, 0.2])
        self.assertEqual(parse_int_list("1, 2,3"), [1, 2, 3])
        self.assertEqual(parse_bool_list("true, false, YES, no, 1, 0"), [True, False, True, False, True, False])
        self.assertEqual(parse_optional_float_list("none, 0.25, NONE"), [None, 0.25, None])

    def test_parse_args_defaults_disallow_torch_hub(self) -> None:
        args = parse_args([])
        self.assertFalse(args.allow_torch_hub)
        self.assertFalse(args.allow_score_mismatch)
        self.assertEqual(args.score_tolerance, 1e-4)

    def test_validate_model_loading_allowed_rejects_false_and_accepts_true(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "torch\\.hub|DINO"):
            validate_model_loading_allowed(False)

        validate_model_loading_allowed(True)

    def test_validate_reproduced_validation_score_checks_mismatch(self) -> None:
        validate_reproduced_validation_score(None, 0.5, tolerance=1e-4, allow_mismatch=False)
        validate_reproduced_validation_score(0.5, 0.50005, tolerance=1e-4, allow_mismatch=False)

        with self.assertRaisesRegex(RuntimeError, "does not match"):
            validate_reproduced_validation_score(0.5, 0.6, tolerance=1e-4, allow_mismatch=False)

        with contextlib.redirect_stdout(io.StringIO()):
            validate_reproduced_validation_score(0.5, 0.6, tolerance=1e-4, allow_mismatch=True)

    def test_iter_postprocess_settings_builds_cartesian_product_and_respects_max_settings(self) -> None:
        config = BaselineConfig(
            harden_temperature=0.7,
            hard_clip_low=0.1,
            hard_clip_high=0.9,
            compute_pixel_f1=True,
        )
        args = SimpleNamespace(
            pred_thresholds=[0.4, 0.5],
            min_component_areas=[10, 20],
            confident_thresholds=[None],
            smooth_options=[True, False],
            opening_options=[True],
            closing_options=[False],
            fill_holes_options=[True],
            keep_confident_seeded_options=[False],
            max_settings=3,
        )

        settings = iter_postprocess_settings(config, args)

        self.assertEqual(len(settings), 3)
        self.assertEqual(
            [setting["pred_threshold"] for setting in settings],
            [0.4, 0.4, 0.4],
        )
        self.assertEqual(
            [setting["min_component_area"] for setting in settings],
            [10, 10, 20],
        )
        self.assertTrue(all(setting["harden_temperature"] == 0.7 for setting in settings))
        self.assertTrue(all(setting["hard_clip_low"] == 0.1 for setting in settings))
        self.assertTrue(all(setting["hard_clip_high"] == 0.9 for setting in settings))
        self.assertTrue(all(setting["compute_pixel_f1"] is True for setting in settings))
        self.assertTrue(all(setting["verify_score_equivalence"] is False for setting in settings))


if __name__ == "__main__":
    unittest.main()
