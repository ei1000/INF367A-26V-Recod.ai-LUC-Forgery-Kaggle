import json
from pathlib import Path
import unittest


class BusterNetEvaluateNotebookTests(unittest.TestCase):
    def _notebook_source(self) -> str:
        notebook = json.loads(Path("einar_busternet/evaluate_validation_diagnostics.ipynb").read_text())
        return "\n".join(
            line
            for cell in notebook["cells"]
            for line in cell.get("source", [])
        )

    def test_notebook_has_forged_authentic_diagnostics_and_prediction_plots(self) -> None:
        source = self._notebook_source()

        self.assertIn("authentic_fp", source)
        self.assertIn("forged_misses", source)
        self.assertIn("forged_empty_near_threshold", source)
        self.assertIn("forged_successes", source)
        self.assertIn("random_examples", source)
        self.assertIn("pixel_stats", source)
        self.assertIn("gt_size_bucket", source)
        self.assertIn("forged_nonempty_summary", source)
        self.assertIn("size_bucket_summary", source)
        self.assertIn("plot_prediction", source)
        self.assertIn("Cleanup GPU Memory", source)
        self.assertIn("torch.cuda.empty_cache()", source)
        self.assertIn("CUSTOM_SWEEP_SETTINGS = [", source)
        self.assertIn('{"pred_threshold": 0.15, "min_component_area": 10, "apply_opening": False}', source)
        self.assertIn("complete_sweep_setting", source)
        self.assertIn("MAX_SWEEP_SETTINGS = 5", source)
        self.assertIn("PRED_THRESHOLDS = [0.15, 0.2, 0.25, 0.3]", source)
        self.assertIn("MIN_COMPONENT_AREAS = [0, 10, 25, 50]", source)
        self.assertIn("OPENING_OPTIONS = [False, True]", source)
        self.assertIn("USE_BEST_SWEEP_FOR_TABLES = True", source)
        self.assertIn("TABLE_POSTPROCESS_SETTING", source)
        self.assertIn("iter_postprocess_settings", source)
        self.assertIn("CHECKPOINT_CHOICE =", source)
        self.assertIn('"best": Path("einar_busternet/artifacts/checkpoints/best.pt")', source)
        self.assertIn('"last": Path("einar_busternet/artifacts/checkpoints/last.pt")', source)
        self.assertIn('"balanced": Path("einar_busternet/artifacts/checkpoints/best_balanced.pt")', source)
        self.assertIn("validation_diagnostics.csv", source)
        self.assertIn("validation_diagnostics_forged_nonempty_summary.csv", source)
        self.assertIn("validation_diagnostics_size_bucket_summary.csv", source)


if __name__ == "__main__":
    unittest.main()
