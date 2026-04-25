import unittest

import torch

from einar_busternet.model import BinaryFusionDinoBusterNet, BusterNetUnionWrapper, DinoBusterNet, SelfCorrelPercPooling


class FakeDinoEncoder(torch.nn.Module):
    def __init__(self, embed_dim: int = 8, patch_size: int = 4) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.scale = torch.nn.Parameter(torch.tensor(1.0))

    def forward_features(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        grid_h = x.shape[-2] // self.patch_size
        grid_w = x.shape[-1] // self.patch_size
        locations = grid_h * grid_w
        base = torch.arange(locations * self.embed_dim, device=x.device, dtype=x.dtype)
        tokens = base.view(1, locations, self.embed_dim).repeat(x.shape[0], 1, 1)
        return {"x_norm_patchtokens": tokens * self.scale}


class BusterNetModelTests(unittest.TestCase):
    def test_self_correlation_pooling_returns_finite_expected_shape(self) -> None:
        pooling = SelfCorrelPercPooling(nb_pools=3)
        features = torch.randn(2, 4, 3, 3)

        pooled = pooling(features)

        self.assertEqual(tuple(pooled.shape), (2, 3, 3, 3))
        self.assertTrue(torch.isfinite(pooled).all())

    def test_self_correlation_pooling_handles_zero_features_and_one_pool(self) -> None:
        pooling = SelfCorrelPercPooling(nb_pools=1)
        features = torch.zeros(1, 4, 2, 2)

        pooled = pooling(features)

        self.assertEqual(tuple(pooled.shape), (1, 1, 2, 2))
        self.assertTrue(torch.isfinite(pooled).all())
        self.assertTrue(torch.equal(pooled, torch.zeros_like(pooled)))

    def test_self_correlation_pooling_rejects_non_positive_pool_count(self) -> None:
        with self.assertRaisesRegex(ValueError, "nb_pools must be positive"):
            SelfCorrelPercPooling(nb_pools=0)

    def test_model_forward_returns_three_class_logits_for_square_input(self) -> None:
        model = DinoBusterNet(FakeDinoEncoder(), embed_dim=8, nb_pools=4)
        x = torch.randn(2, 3, 16, 16)

        logits = model(x)

        self.assertEqual(tuple(logits.shape), (2, 3, 16, 16))

    def test_model_forward_crops_back_after_patch_padding(self) -> None:
        model = DinoBusterNet(FakeDinoEncoder(), embed_dim=8, nb_pools=4)
        x = torch.randn(1, 3, 15, 15)

        logits = model(x)

        self.assertEqual(tuple(logits.shape), (1, 3, 15, 15))

    def test_forward_branches_returns_full_resolution_branch_logits(self) -> None:
        model = DinoBusterNet(FakeDinoEncoder(), embed_dim=8, nb_pools=4)
        x = torch.randn(1, 3, 16, 16)

        mani_logits, simi_logits = model.forward_branches(x)

        self.assertEqual(tuple(mani_logits.shape), (1, 1, 16, 16))
        self.assertEqual(tuple(simi_logits.shape), (1, 1, 16, 16))

    def test_fusion_uses_decoder_features_and_branch_logits(self) -> None:
        model = DinoBusterNet(FakeDinoEncoder(), embed_dim=8, nb_pools=4)

        self.assertEqual(model.mani_classifier.out_channels, 1)
        self.assertEqual(model.simi_classifier.out_channels, 1)
        self.assertEqual(model.fusion[0].in_channels, 162)
        self.assertEqual(model.fusion[0].out_channels, 128)
        self.assertEqual(model.fusion[3].out_channels, 128)
        self.assertEqual(model.fusion[6].out_channels, 64)

    def test_binary_fusion_model_returns_one_channel_logits(self) -> None:
        model = BinaryFusionDinoBusterNet(FakeDinoEncoder(), embed_dim=8, nb_pools=4)
        x = torch.randn(2, 3, 16, 16)

        logits = model(x)

        self.assertEqual(tuple(logits.shape), (2, 1, 16, 16))
        self.assertEqual(model.fusion[-1].out_channels, 1)

    def test_frozen_encoder_stays_eval_when_model_train_is_called(self) -> None:
        encoder = FakeDinoEncoder()
        model = DinoBusterNet(encoder, embed_dim=8, nb_pools=4, freeze_encoder=True)

        model.train()

        self.assertFalse(model.encoder.training)
        self.assertTrue(all(not param.requires_grad for param in model.encoder.parameters()))

    def test_union_wrapper_returns_logits_matching_source_target_probability(self) -> None:
        model = DinoBusterNet(FakeDinoEncoder(), embed_dim=8, nb_pools=4)
        model.eval()
        wrapper = BusterNetUnionWrapper(model)
        x = torch.randn(1, 3, 16, 16)

        with torch.no_grad():
            logits = model(x)
            expected = logits.softmax(dim=1)[:, 1:2] + logits.softmax(dim=1)[:, 2:3]
            wrapped_prob = torch.sigmoid(wrapper(x))

        self.assertTrue(torch.allclose(wrapped_prob, expected, atol=1e-6))

    def test_union_wrapper_passes_binary_fusion_logits_through(self) -> None:
        model = BinaryFusionDinoBusterNet(FakeDinoEncoder(), embed_dim=8, nb_pools=4)
        model.eval()
        wrapper = BusterNetUnionWrapper(model)
        x = torch.randn(1, 3, 16, 16)

        with torch.no_grad():
            expected = model(x)
            wrapped = wrapper(x)

        torch.testing.assert_close(wrapped, expected)


if __name__ == "__main__":
    unittest.main()
