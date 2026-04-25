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
        self.assertIn("forged_successes", source)
        self.assertIn("random_examples", source)
        self.assertIn("pixel_stats", source)
        self.assertIn("plot_prediction", source)
        self.assertIn("Cleanup GPU Memory", source)
        self.assertIn("torch.cuda.empty_cache()", source)
        self.assertIn("MAX_SWEEP_SETTINGS = 5", source)
        self.assertIn("iter_postprocess_settings", source)
        self.assertIn('CHECKPOINT_CHOICE = "last"', source)
        self.assertIn('"best": Path("einar_busternet/artifacts/checkpoints/best.pt")', source)
        self.assertIn('"last": Path("einar_busternet/artifacts/checkpoints/last.pt")', source)
        self.assertIn("validation_diagnostics.csv", source)


if __name__ == "__main__":
    unittest.main()
