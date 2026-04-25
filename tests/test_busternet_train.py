import unittest

import torch

from einar_busternet.config import BusterNetConfig
from einar_busternet.model import BinaryFusionDinoBusterNet, DinoBusterNet
from einar_busternet.train import (
    BinaryUnionBCEDiceLoss,
    BinaryUnionBCEWithLogitsLoss,
    BCEDiceLoss,
    build_config_from_args,
    build_fusion_loss,
    build_model,
    configure_trainable_parts,
    foreground_logit_from_three_class,
    train_stage1_epoch,
)
from tests.test_busternet_model import FakeDinoEncoder


class BusterNetTrainTests(unittest.TestCase):
    def test_foreground_logit_matches_source_target_softmax_sum(self) -> None:
        logits = torch.randn(2, 3, 5, 7)

        foreground_prob = foreground_logit_from_three_class(logits).sigmoid()
        expected = logits.softmax(dim=1)[:, 1:3].sum(dim=1)

        torch.testing.assert_close(foreground_prob, expected)

    def test_smoke_args_build_tiny_config(self) -> None:
        config = build_config_from_args(["--smoke"])

        self.assertEqual(config.train_subset, 8)
        self.assertEqual(config.val_subset, 8)
        self.assertEqual(config.batch_size, 2)
        self.assertEqual(config.stage1_epochs, 1)
        self.assertEqual(config.stage2_epochs, 1)
        self.assertEqual(config.stage3_epochs, 0)

    def test_loss_override_args_are_configurable(self) -> None:
        config = build_config_from_args(
            [
                "--stage1-lr",
                "0.002",
                "--branch-dice-weight",
                "0.25",
                "--fusion-dice-weight",
                "0.75",
                "--pred-threshold",
                "0.25",
            ]
        )

        self.assertEqual(config.stage1_lr, 0.002)
        self.assertEqual(config.branch_dice_weight, 0.25)
        self.assertEqual(config.fusion_dice_weight, 0.75)
        self.assertEqual(config.pred_threshold, 0.25)

    def test_binary_fusion_arg_selects_binary_union_mode(self) -> None:
        config = build_config_from_args(["--fusion-mode", "binary_union"])

        self.assertEqual(config.fusion_mode, "binary_union")

    def test_build_fusion_loss_matches_fusion_mode(self) -> None:
        ce_loss = build_fusion_loss(BusterNetConfig(fusion_mode="three_class"), torch.device("cpu"))
        bce_loss = build_fusion_loss(BusterNetConfig(fusion_mode="binary_union"), torch.device("cpu"))

        self.assertIsInstance(ce_loss, torch.nn.CrossEntropyLoss)
        self.assertIsInstance(bce_loss, BinaryUnionBCEDiceLoss)

    def test_binary_union_loss_uses_source_target_union(self) -> None:
        loss_fn = BinaryUnionBCEWithLogitsLoss()
        logits = torch.zeros(1, 1, 2, 3)
        labels = torch.tensor([[[0, 1, 2], [0, 0, 2]]])

        loss = loss_fn(logits, labels)
        expected = torch.nn.functional.binary_cross_entropy_with_logits(
            logits[:, 0],
            (labels > 0).float(),
        )

        torch.testing.assert_close(loss, expected)

    def test_bce_dice_loss_is_bce_plus_one_minus_dice(self) -> None:
        loss_fn = BCEDiceLoss(dice_weight=1.0)
        logits = torch.zeros(1, 2, 3)
        targets = torch.tensor([[[0.0, 1.0, 1.0], [0.0, 0.0, 1.0]]])

        loss = loss_fn(logits, targets)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets)
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(-2, -1))
        denominator = probs.sum(dim=(-2, -1)) + targets.sum(dim=(-2, -1))
        dice_loss = 1.0 - ((2.0 * intersection + 1.0) / (denominator + 1.0)).mean()

        torch.testing.assert_close(loss, bce + dice_loss)

    def test_build_model_selects_binary_fusion_class_without_torch_hub(self) -> None:
        config = BusterNetConfig(fusion_mode="binary_union")
        original = BinaryFusionDinoBusterNet.from_official
        try:
            BinaryFusionDinoBusterNet.from_official = classmethod(
                lambda cls, **kwargs: cls(FakeDinoEncoder(), embed_dim=8, nb_pools=4)
            )
            model = build_model(config)
        finally:
            BinaryFusionDinoBusterNet.from_official = original

        self.assertIsInstance(model, BinaryFusionDinoBusterNet)

    def test_configure_trainable_parts_sets_expected_stage_flags(self) -> None:
        model = DinoBusterNet(FakeDinoEncoder(), embed_dim=8, nb_pools=4)

        configure_trainable_parts(model, stage=2)

        self.assertTrue(all(not param.requires_grad for param in model.encoder.parameters()))
        self.assertTrue(all(not param.requires_grad for param in model.mani_decoder.parameters()))
        self.assertTrue(all(not param.requires_grad for param in model.mani_classifier.parameters()))
        self.assertTrue(all(not param.requires_grad for param in model.simi_decoder.parameters()))
        self.assertTrue(all(not param.requires_grad for param in model.simi_classifier.parameters()))
        self.assertTrue(any(param.requires_grad for param in model.fusion.parameters()))

        configure_trainable_parts(model, stage=3)

        self.assertTrue(any(param.requires_grad for param in model.mani_decoder.parameters()))
        self.assertTrue(any(param.requires_grad for param in model.mani_classifier.parameters()))
        self.assertTrue(any(param.requires_grad for param in model.simi_decoder.parameters()))
        self.assertTrue(any(param.requires_grad for param in model.simi_classifier.parameters()))
        self.assertTrue(any(param.requires_grad for param in model.fusion.parameters()))
        self.assertTrue(all(not param.requires_grad for param in model.encoder.parameters()))

    def test_train_stage1_epoch_updates_branch_decoders_only(self) -> None:
        torch.manual_seed(7)
        model = DinoBusterNet(FakeDinoEncoder(), embed_dim=8, nb_pools=4)
        configure_trainable_parts(model, stage=1)
        fusion_before = [param.detach().clone() for param in model.fusion.parameters()]
        mani_before = [param.detach().clone() for param in model.mani_decoder.parameters()]
        mani_classifier_before = [param.detach().clone() for param in model.mani_classifier.parameters()]
        simi_before = [param.detach().clone() for param in model.simi_decoder.parameters()]
        simi_classifier_before = [param.detach().clone() for param in model.simi_classifier.parameters()]

        imgs = torch.randn(2, 3, 16, 16)
        labels = torch.zeros(2, 16, 16, dtype=torch.long)
        labels[:, :4, :4] = 1
        labels[:, 8:12, 8:12] = 2
        train_loader = [(imgs, labels)]
        mani_optimizer = torch.optim.Adam(
            list(model.mani_decoder.parameters()) + list(model.mani_classifier.parameters()),
            lr=1e-2,
        )
        simi_optimizer = torch.optim.Adam(
            list(model.simi_decoder.parameters()) + list(model.simi_classifier.parameters()),
            lr=1e-2,
        )

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
        self.assertTrue(any(not torch.equal(before, after) for before, after in zip(mani_classifier_before, model.mani_classifier.parameters())))
        self.assertTrue(any(not torch.equal(before, after) for before, after in zip(simi_before, model.simi_decoder.parameters())))
        self.assertTrue(any(not torch.equal(before, after) for before, after in zip(simi_classifier_before, model.simi_classifier.parameters())))
        self.assertTrue(all(torch.equal(before, after) for before, after in zip(fusion_before, model.fusion.parameters())))


if __name__ == "__main__":
    unittest.main()
