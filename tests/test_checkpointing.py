import tempfile
import unittest

import torch
import torch.nn as nn

from configs.baseline_config import BaselineConfig
from engine.checkpointing import (
    build_checkpoint_payload,
    capture_rng_state,
    load_checkpoint,
    restore_rng_state,
    restore_training_state,
    save_checkpoint,
    validate_checkpoint_cadence,
)


class CheckpointingTests(unittest.TestCase):
    def test_validate_checkpoint_cadence_rejects_zero(self):
        with self.assertRaises(ValueError):
            validate_checkpoint_cadence(0)

    def test_build_checkpoint_payload_includes_required_keys(self):
        model = nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        payload = build_checkpoint_payload(
            epoch=3,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            kaggle_score=0.42,
            best_kaggle_score=0.5,
            validation_result={"kaggle_score": 0.42},
            config=BaselineConfig(),
            split_counts={"train": 1},
            model_name="demo-model",
        )

        required_keys = {
            "epoch",
            "model_state_dict",
            "optimizer_state_dict",
            "scheduler_state_dict",
            "scaler_state_dict",
            "kaggle_score",
            "best_kaggle_score",
            "validation_result",
            "config",
            "split_counts",
            "model_name",
            "rng_state",
        }
        self.assertTrue(required_keys.issubset(payload))
        self.assertEqual(payload["best_kaggle_score"], 0.5)
        self.assertIn("python_random_state", payload["rng_state"])
        self.assertIn("numpy_random_state", payload["rng_state"])
        self.assertIn("torch_rng_state", payload["rng_state"])

    def test_save_and_load_checkpoint_round_trip(self):
        model = nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        payload = build_checkpoint_payload(
            epoch=1,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            kaggle_score=0.25,
            best_kaggle_score=0.25,
            validation_result={"kaggle_score": 0.25},
            config=BaselineConfig(),
            split_counts={"train": 1},
            model_name="demo-model",
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = save_checkpoint(payload, tmpdir, "checkpoint.pt")
            self.assertTrue(checkpoint_path.exists())
            loaded = load_checkpoint(checkpoint_path, trusted=True)

        self.assertEqual(loaded["epoch"], 1)
        self.assertEqual(loaded["best_kaggle_score"], 0.25)
        self.assertEqual(loaded["model_name"], "demo-model")

    def test_load_checkpoint_requires_trusted_opt_in(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_checkpoint({"epoch": 1}, tmpdir, "checkpoint.pt")

            with self.assertRaises(ValueError):
                load_checkpoint(path)

            loaded = load_checkpoint(path, trusted=True)
            self.assertEqual(loaded["epoch"], 1)

    def test_restore_training_state_supports_old_checkpoints(self):
        torch.manual_seed(7)
        model = nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

        inputs = torch.ones(1, 2)
        loss = model(inputs).sum()
        loss.backward()
        optimizer.step()

        checkpoint = {
            "epoch": 4,
            "kaggle_score": 0.4815,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }

        restored_model = nn.Linear(2, 1)
        restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.1, momentum=0.9)

        start_epoch, best_kaggle_score = restore_training_state(
            checkpoint=checkpoint,
            model=restored_model,
            optimizer=restored_optimizer,
            scheduler=None,
            scaler=None,
            restore_rng=False,
        )

        self.assertEqual(start_epoch, 4)
        self.assertEqual(best_kaggle_score, 0.4815)
        torch.testing.assert_close(restored_model.weight, model.weight)
        torch.testing.assert_close(restored_model.bias, model.bias)

        original_state = optimizer.state_dict()["state"]
        restored_state = restored_optimizer.state_dict()["state"]
        self.assertEqual(sorted(original_state.keys()), sorted(restored_state.keys()))
        for key in original_state:
            torch.testing.assert_close(
                restored_state[key]["momentum_buffer"],
                original_state[key]["momentum_buffer"],
            )

    def test_named_torch_generator_state_is_restored(self):
        model = nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        restore_model = nn.Linear(2, 1)
        restore_optimizer = torch.optim.SGD(restore_model.parameters(), lr=0.1)
        generator = torch.Generator().manual_seed(1234)
        checkpoint = build_checkpoint_payload(
            epoch=1,
            model=model,
            optimizer=optimizer,
            scheduler=None,
            scaler=None,
            kaggle_score=0.1,
            best_kaggle_score=0.1,
            validation_result={"kaggle_score": 0.1},
            config=BaselineConfig(),
            split_counts={"train": 1},
            model_name="demo-model",
            torch_generators={"train_loader": generator},
        )

        expected_generator = torch.Generator()
        expected_generator.set_state(checkpoint["rng_state"]["torch_generators"]["train_loader"])
        expected_draw = torch.randint(0, 1000, (5,), generator=expected_generator)

        torch.randint(0, 1000, (9,), generator=generator)
        restore_training_state(
            checkpoint=checkpoint,
            model=restore_model,
            optimizer=restore_optimizer,
            scheduler=None,
            scaler=None,
            restore_rng=True,
            torch_generators={"train_loader": generator},
        )
        actual_draw = torch.randint(0, 1000, (5,), generator=generator)

        torch.testing.assert_close(actual_draw, expected_draw)

    def test_restore_training_state_restores_scheduler_and_scaler(self):
        model = nn.Linear(2, 1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        scaler = torch.cuda.amp.GradScaler(enabled=False)
        optimizer.zero_grad()
        loss = model(torch.ones(1, 2)).sum()
        loss.backward()
        optimizer.step()
        scheduler.step()

        payload = build_checkpoint_payload(
            epoch=2,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            kaggle_score=0.3,
            best_kaggle_score=0.35,
            validation_result={"kaggle_score": 0.3},
            config=BaselineConfig(),
            split_counts={"train": 1},
            model_name="demo-model",
            torch_generators={"train_loader": torch.Generator().manual_seed(7)},
        )

        restored_model = nn.Linear(2, 1)
        restored_optimizer = torch.optim.SGD(restored_model.parameters(), lr=0.1)
        restored_scheduler = torch.optim.lr_scheduler.StepLR(restored_optimizer, step_size=1, gamma=0.5)
        restored_scaler = torch.cuda.amp.GradScaler(enabled=False)

        start_epoch, best_kaggle_score = restore_training_state(
            checkpoint=payload,
            model=restored_model,
            optimizer=restored_optimizer,
            scheduler=restored_scheduler,
            scaler=restored_scaler,
            restore_rng=False,
        )

        self.assertEqual(start_epoch, 2)
        self.assertEqual(best_kaggle_score, 0.35)
        self.assertEqual(restored_scheduler.state_dict(), scheduler.state_dict())
        self.assertEqual(restored_scaler.state_dict(), scaler.state_dict())

    def test_rng_restore_missing_is_noop(self):
        rng_state = capture_rng_state()
        self.assertIn("python_random_state", rng_state)
        self.assertIn("numpy_random_state", rng_state)
        self.assertIn("torch_rng_state", rng_state)

        restore_rng_state(None)
        restore_rng_state({"python_random_state": None, "numpy_random_state": None, "torch_rng_state": None})


if __name__ == "__main__":
    unittest.main()
