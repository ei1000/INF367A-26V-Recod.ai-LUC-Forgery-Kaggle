import unittest

from einar_busternet.config import BusterNetConfig
from einar_busternet.evaluate import config_from_checkpoint, parse_args, validate_model_loading_allowed


class BusterNetEvaluateTests(unittest.TestCase):
    def test_parse_args_defaults_to_validation_best_checkpoint(self) -> None:
        args = parse_args([])

        self.assertEqual(str(args.checkpoint), "einar_busternet/artifacts/checkpoints/best.pt")
        self.assertEqual(str(args.output), "einar_busternet/artifacts/results/eval_summary.json")
        self.assertFalse(args.allow_torch_hub)

    def test_validate_model_loading_requires_explicit_allowance(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "allow-torch-hub"):
            validate_model_loading_allowed(False)

        validate_model_loading_allowed(True)

    def test_config_from_checkpoint_filters_to_busternet_fields(self) -> None:
        config = config_from_checkpoint(
            {
                "config": {
                    "seed": 123,
                    "nb_pools": 77,
                    "stage1_epochs": 2,
                    "unknown_field": "ignored",
                }
            }
        )

        self.assertIsInstance(config, BusterNetConfig)
        self.assertEqual(config.seed, 123)
        self.assertEqual(config.nb_pools, 77)
        self.assertEqual(config.stage1_epochs, 2)


if __name__ == "__main__":
    unittest.main()
