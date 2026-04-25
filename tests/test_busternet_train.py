import unittest

import torch

from einar_busternet.model import DinoBusterNet
from einar_busternet.train import build_config_from_args, configure_trainable_parts, train_stage1_epoch
from tests.test_busternet_model import FakeDinoEncoder


class BusterNetTrainTests(unittest.TestCase):
    def test_smoke_args_build_tiny_config(self) -> None:
        config = build_config_from_args(["--smoke"])

        self.assertEqual(config.train_subset, 8)
        self.assertEqual(config.val_subset, 8)
        self.assertEqual(config.batch_size, 2)
        self.assertEqual(config.stage1_epochs, 1)
        self.assertEqual(config.stage2_epochs, 1)
        self.assertEqual(config.stage3_epochs, 0)

    def test_configure_trainable_parts_sets_expected_stage_flags(self) -> None:
        model = DinoBusterNet(FakeDinoEncoder(), embed_dim=8, nb_pools=4)

        configure_trainable_parts(model, stage=2)

        self.assertTrue(all(not param.requires_grad for param in model.encoder.parameters()))
        self.assertTrue(all(not param.requires_grad for param in model.mani_decoder.parameters()))
        self.assertTrue(all(not param.requires_grad for param in model.simi_decoder.parameters()))
        self.assertTrue(any(param.requires_grad for param in model.fusion.parameters()))

        configure_trainable_parts(model, stage=3)

        self.assertTrue(any(param.requires_grad for param in model.mani_decoder.parameters()))
        self.assertTrue(any(param.requires_grad for param in model.simi_decoder.parameters()))
        self.assertTrue(any(param.requires_grad for param in model.fusion.parameters()))
        self.assertTrue(all(not param.requires_grad for param in model.encoder.parameters()))

    def test_train_stage1_epoch_updates_branch_decoders_only(self) -> None:
        torch.manual_seed(7)
        model = DinoBusterNet(FakeDinoEncoder(), embed_dim=8, nb_pools=4)
        configure_trainable_parts(model, stage=1)
        fusion_before = [param.detach().clone() for param in model.fusion.parameters()]
        mani_before = [param.detach().clone() for param in model.mani_decoder.parameters()]
        simi_before = [param.detach().clone() for param in model.simi_decoder.parameters()]

        imgs = torch.randn(2, 3, 16, 16)
        labels = torch.zeros(2, 16, 16, dtype=torch.long)
        labels[:, :4, :4] = 1
        labels[:, 8:12, 8:12] = 2
        train_loader = [(imgs, labels)]
        mani_optimizer = torch.optim.Adam(model.mani_decoder.parameters(), lr=1e-2)
        simi_optimizer = torch.optim.Adam(model.simi_decoder.parameters(), lr=1e-2)

        metrics = train_stage1_epoch(
            model=model,
            train_loader=train_loader,
            mani_optimizer=mani_optimizer,
            simi_optimizer=simi_optimizer,
            loss_fn=torch.nn.BCEWithLogitsLoss(),
            device=torch.device("cpu"),
            grad_clip_max_norm=1.0,
            epoch_idx=0,
        )

        self.assertGreater(metrics["loss"], 0.0)
        self.assertTrue(any(not torch.equal(before, after) for before, after in zip(mani_before, model.mani_decoder.parameters())))
        self.assertTrue(any(not torch.equal(before, after) for before, after in zip(simi_before, model.simi_decoder.parameters())))
        self.assertTrue(all(torch.equal(before, after) for before, after in zip(fusion_before, model.fusion.parameters())))


if __name__ == "__main__":
    unittest.main()
