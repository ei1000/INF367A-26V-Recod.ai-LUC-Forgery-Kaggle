import unittest

from evaluate_baseline import parse_args, validate_local_holdout_allowed


class EvaluateBaselineTests(unittest.TestCase):
    def test_parse_args_requires_explicit_holdout_and_torch_hub_flags_by_default(self) -> None:
        args = parse_args([])

        self.assertFalse(args.confirm_local_holdout)
        self.assertFalse(args.allow_torch_hub)

    def test_validate_local_holdout_allowed_rejects_false_and_accepts_true(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "reserved local holdout"):
            validate_local_holdout_allowed(False)

        validate_local_holdout_allowed(True)

    def test_parse_args_accepts_explicit_holdout_and_torch_hub_flags(self) -> None:
        args = parse_args(["--confirm-local-holdout", "--allow-torch-hub"])

        self.assertTrue(args.confirm_local_holdout)
        self.assertTrue(args.allow_torch_hub)


if __name__ == "__main__":
    unittest.main()
