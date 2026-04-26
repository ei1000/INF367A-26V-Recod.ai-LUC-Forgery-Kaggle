import json
from pathlib import Path
import unittest


class EvaluateValidationNotebookTests(unittest.TestCase):
    def _notebook_source(self) -> str:
        notebook = json.loads(Path("notebooks/evaluate_validation_postprocess.ipynb").read_text())
        return "\n".join(
            line
            for cell in notebook["cells"]
            for line in cell.get("source", [])
        )

    def test_notebook_checks_reproduced_score_before_sweeps(self) -> None:
        source = self._notebook_source()

        self.assertIn("validate_reproduced_validation_score", source)
        self.assertLess(
            source.index("validate_reproduced_validation_score("),
            source.index("for rank, setting in enumerate(iter_postprocess_settings"),
        )

    def test_notebook_has_diagnostics_and_broader_max_inspired_sweep(self) -> None:
        source = self._notebook_source()

        self.assertIn("BROAD_SWEEP_PRESET", source)
        self.assertIn("compute_setting_diagnostics", source)
        self.assertIn("authentic_false_positive_rate", source)
        self.assertIn("forged_empty_prediction_rate", source)
        self.assertIn("Max-inspired", source)


if __name__ == "__main__":
    unittest.main()
