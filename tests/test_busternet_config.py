import dataclasses
import unittest

from configs.baseline_config import BaselineConfig, seed_worker as baseline_seed_worker, set_seed as baseline_set_seed
from einar_busternet.config import BusterNetConfig, seed_worker, set_seed


class BusterNetConfigTests(unittest.TestCase):
    def test_defaults_match_busternet_plan(self) -> None:
        config = BusterNetConfig()

        self.assertEqual(config.stage1_lr, 1e-2)
        self.assertEqual(config.stage2_lr, 1e-2)
        self.assertEqual(config.stage3_lr, 1e-5)
        self.assertEqual(config.nb_pools, 100)
        self.assertEqual(config.ce_class_weights, (0.1, 1.0, 1.0))
        self.assertEqual(config.union_wrapper_eps, 1e-6)
        self.assertEqual(config.total_stage_epochs, 16)

    def test_dataset_defaults_use_clean_pairs_and_paired_authentic(self) -> None:
        config = BusterNetConfig()

        self.assertEqual(config.metadata_path, "data/train_masks_source_target_metadata.csv")
        self.assertEqual(config.allowed_forged_statuses, ("derived_from_pair",))
        self.assertTrue(config.include_authentic)
        self.assertEqual(config.authentic_policy, "paired_derived_only")

    def test_baseline_compatible_fields_keep_baseline_defaults(self) -> None:
        config = BusterNetConfig()
        baseline = BaselineConfig()

        for field_name in (
            "batch_size",
            "seed",
            "target_size",
            "pred_threshold",
            "dino_model_name",
            "dino_embed_dim",
            "freeze_dino_encoder",
            "validation_inference_mode",
            "validation_probability_dtype",
            "validation_transfer_mode",
        ):
            self.assertEqual(getattr(config, field_name), getattr(baseline, field_name))

    def test_artifacts_stay_inside_busternet_tree(self) -> None:
        config = BusterNetConfig()

        self.assertEqual(config.checkpoint_dir, "einar_busternet/artifacts/checkpoints")
        self.assertEqual(config.results_dir, "einar_busternet/artifacts/results")
        self.assertEqual(config.best_checkpoint_name, "best.pt")
        self.assertEqual(config.last_checkpoint_name, "last.pt")

    def test_config_serializes_as_dataclass(self) -> None:
        config_dict = dataclasses.asdict(BusterNetConfig(stage1_epochs=1, stage2_epochs=2, stage3_epochs=3))

        self.assertEqual(config_dict["stage1_epochs"], 1)
        self.assertEqual(config_dict["stage2_epochs"], 2)
        self.assertEqual(config_dict["stage3_epochs"], 3)
        self.assertEqual(config_dict["checkpoint_dir"], "einar_busternet/artifacts/checkpoints")

    def test_seed_helpers_are_reused_from_baseline(self) -> None:
        self.assertIs(set_seed, baseline_set_seed)
        self.assertIs(seed_worker, baseline_seed_worker)


if __name__ == "__main__":
    unittest.main()
