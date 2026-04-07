import torch
import torch.nn.functional as F


class PixelPropagator:
    def __init__(
        self,
        image: torch.Tensor,
        cnn_features,
        zernike_features,
        random_window: int = 50,
        reduced_precision: bool = True,
    ):
        self.single_image = image.dim() == 3
        if self.single_image:
            image = image.unsqueeze(0)
        elif image.dim() != 4:
            raise ValueError(f"Expected image shape [C,H,W] or [B,C,H,W], got {tuple(image.shape)}")

        self.image = image
        self.device = image.device
        self.random_window = int(random_window)
        self.max_fraction = 1.0

        self.batch_size = self.image.shape[0]
        _, _, H, W = self.image.shape
        self.H = H
        self.W = W
        self.match_dtype = self.resolve_match_dtype(reduced_precision)

        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=self.device, dtype=torch.float32),
            torch.arange(W, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        self.x_grid = x_grid.unsqueeze(0)
        self.y_grid = y_grid.unsqueeze(0)
        self.x_min_full = -self.x_grid
        self.x_max_full = (W - 1.0) - self.x_grid
        self.y_min_full = -self.y_grid
        self.y_max_full = (H - 1.0) - self.y_grid

        self.cnn_features = self.prepare_feature_set(cnn_features)
        self.zernike_features = self.prepare_feature_set(zernike_features)

    def resolve_match_dtype(self, reduced_precision: bool) -> torch.dtype:
        if not reduced_precision or self.device.type != "cuda":
            return torch.float32
        return torch.float16

    def prepare_feature_set(self, features) -> list[torch.Tensor]:
        if isinstance(features, (tuple, list)):
            return [self.prepare_feature_tensor(feature) for feature in features]
        return [self.prepare_feature_tensor(features)]

    def prepare_feature_tensor(self, feature: torch.Tensor) -> torch.Tensor:
        if feature.dim() == 3:
            feature = feature.unsqueeze(0)
        elif feature.dim() != 4:
            raise ValueError(f"Expected feature map shape [C,H,W] or [B,C,H,W], got {tuple(feature.shape)}")

        if feature.shape[0] != self.batch_size:
            raise ValueError(
                f"Batch size mismatch between image batch {self.batch_size} and feature map {tuple(feature.shape)}"
            )

        if feature.shape[-2:] != (self.H, self.W):
            feature = F.interpolate(
                feature,
                size=(self.H, self.W),
                mode="bilinear",
                align_corners=True,
            )

        return feature.to(device=self.device, dtype=self.match_dtype)

    def _scaled_bounds(self):
        if self.max_fraction >= 1:
            return self.x_min_full, self.x_max_full, self.y_min_full, self.y_max_full
        x_min = torch.ceil(self.x_min_full * self.max_fraction)
        x_max = torch.floor(self.x_max_full * self.max_fraction)
        y_min = torch.ceil(self.y_min_full * self.max_fraction)
        y_max = torch.floor(self.y_max_full * self.max_fraction)
        return x_min, x_max, y_min, y_max

    @torch.no_grad()
    def generate_random_offsets(self):
        B, H, W = self.batch_size, self.H, self.W
        x_min, x_max, y_min, y_max = self._scaled_bounds()

        x_range = (x_max - x_min + 1).clamp(min=1)
        y_range = (y_max - y_min + 1).clamp(min=1)

        dx = torch.floor(torch.rand(B, H, W, device=self.device) * x_range + x_min).long()
        dy = torch.floor(torch.rand(B, H, W, device=self.device) * y_range + y_min).long()

        zero_mask = (dx == 0) & (dy == 0)
        if zero_mask.any():
            dy_alt = torch.where(dy + 1 <= y_max, dy + 1, dy - 1)
            dy = dy.clone()
            dy[zero_mask] = dy_alt[zero_mask]

        return dx.float(), dy.float()

    @torch.no_grad()
    def propagation_block(self, dx: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
        def roll_offsets(values: torch.Tensor, shift_h: int, shift_w: int) -> torch.Tensor:
            return torch.roll(values, shifts=(shift_h, shift_w), dims=(-2, -1))

        zero_dx = [
            roll_offsets(dx, 0, 1),
            roll_offsets(dx, 0, -1),
            roll_offsets(dx, 1, 0),
            roll_offsets(dx, -1, 0),
        ]
        zero_dy = [
            roll_offsets(dy, 0, 1),
            roll_offsets(dy, 0, -1),
            roll_offsets(dy, 1, 0),
            roll_offsets(dy, -1, 0),
        ]

        first_dx = [
            2 * roll_offsets(dx, 0, 1) - roll_offsets(dx, 0, 2),
            2 * roll_offsets(dx, 0, -1) - roll_offsets(dx, 0, -2),
            2 * roll_offsets(dx, 1, 0) - roll_offsets(dx, 2, 0),
            2 * roll_offsets(dx, -1, 0) - roll_offsets(dx, -2, 0),
            2 * roll_offsets(dx, 1, 1) - roll_offsets(dx, 2, 2),
            2 * roll_offsets(dx, -1, -1) - roll_offsets(dx, -2, -2),
            2 * roll_offsets(dx, 1, -1) - roll_offsets(dx, 2, -2),
            2 * roll_offsets(dx, -1, 1) - roll_offsets(dx, -2, 2),
        ]
        first_dy = [
            2 * roll_offsets(dy, 0, 1) - roll_offsets(dy, 0, 2),
            2 * roll_offsets(dy, 0, -1) - roll_offsets(dy, 0, -2),
            2 * roll_offsets(dy, 1, 0) - roll_offsets(dy, 2, 0),
            2 * roll_offsets(dy, -1, 0) - roll_offsets(dy, -2, 0),
            2 * roll_offsets(dy, 1, 1) - roll_offsets(dy, 2, 2),
            2 * roll_offsets(dy, -1, -1) - roll_offsets(dy, -2, -2),
            2 * roll_offsets(dy, 1, -1) - roll_offsets(dy, 2, -2),
            2 * roll_offsets(dy, -1, 1) - roll_offsets(dy, -2, 2),
        ]

        cand_dx = torch.stack(zero_dx + first_dx, dim=-1)
        cand_dy = torch.stack(zero_dy + first_dy, dim=-1)
        return torch.stack((cand_dx, cand_dy), dim=-1)

    @torch.no_grad()
    def random_search_block(
        self,
        dx: torch.Tensor,
        dy: torch.Tensor,
        radius: int | None = None,
        num_random: int = 4,
    ) -> torch.Tensor:
        B, H, W = dx.shape
        radius = max(1, self.random_window // 2) if radius is None else int(radius)

        rand_dx = torch.randint(-radius, radius + 1, (B, H, W, num_random), device=self.device)
        rand_dy = torch.randint(-radius, radius + 1, (B, H, W, num_random), device=self.device)

        cand_dx = dx.unsqueeze(-1) + rand_dx
        cand_dy = dy.unsqueeze(-1) + rand_dy

        x_min = self.x_min_full.unsqueeze(-1)
        x_max = self.x_max_full.unsqueeze(-1)
        y_min = self.y_min_full.unsqueeze(-1)
        y_max = self.y_max_full.unsqueeze(-1)
        cand_dx = torch.max(torch.min(cand_dx, x_max), x_min)
        cand_dy = torch.max(torch.min(cand_dy, y_max), y_min)

        zero_mask = (cand_dx.abs() + cand_dy.abs()) < 0.5
        if zero_mask.any():
            cand_dy = torch.where(
                zero_mask,
                torch.where(cand_dy + 1 <= y_max, cand_dy + 1, cand_dy - 1),
                cand_dy,
            )

        return torch.stack((cand_dx, cand_dy), dim=-1)

    @torch.no_grad()
    def non_local_reset(self, dx: torch.Tensor, dy: torch.Tensor, limit_u: float = 25.0):
        local_mask = (dx * dx + dy * dy) <= float(limit_u)
        if not local_mask.any():
            return dx, dy

        rand_dx, rand_dy = self.generate_random_offsets()
        dx = torch.where(local_mask, rand_dx, dx)
        dy = torch.where(local_mask, rand_dy, dy)
        return dx, dy

    def normalize_grid(self, values: torch.Tensor, size: int, dtype: torch.dtype) -> torch.Tensor:
        values = values.clamp(0, size - 1).to(dtype=dtype)
        if size > 1:
            return 2.0 * (values / float(size - 1)) - 1.0
        return torch.zeros_like(values, dtype=dtype)

    def sample_candidates(self, feature_map: torch.Tensor, x_abs: torch.Tensor, y_abs: torch.Tensor) -> torch.Tensor:
        B, _, H, W = feature_map.shape
        K = x_abs.shape[-1]
        grid_dtype = feature_map.dtype

        x_grid = self.normalize_grid(x_abs, W, grid_dtype)
        y_grid = self.normalize_grid(y_abs, H, grid_dtype)

        x_grid = x_grid.permute(0, 1, 3, 2).reshape(B, H, K * W)
        y_grid = y_grid.permute(0, 1, 3, 2).reshape(B, H, K * W)
        grid = torch.stack((x_grid, y_grid), dim=-1)

        sampled = F.grid_sample(
            feature_map,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        sampled = sampled.reshape(B, feature_map.shape[1], H, K, W)
        return sampled.permute(0, 3, 1, 2, 4)

    @torch.no_grad()
    def evaluate_reference(self, candidates: torch.Tensor, feature_list: list[torch.Tensor], beta: float = 10.0, exclude_self: bool = True):
        B, H, W, K = candidates.shape[:4]
        x_abs = self.x_grid.unsqueeze(-1) + candidates[..., 0]
        y_abs = self.y_grid.unsqueeze(-1) + candidates[..., 1]
        oob = (x_abs < 0) | (x_abs > (W - 1)) | (y_abs < 0) | (y_abs > (H - 1))

        l1 = torch.empty((B, K, H, W), device=candidates.device, dtype=torch.float32)
        for candidate_idx in range(K):
            best_pair = None
            x_candidate = x_abs[..., candidate_idx : candidate_idx + 1]
            y_candidate = y_abs[..., candidate_idx : candidate_idx + 1]
            for target_features in feature_list:
                sampled = self.sample_candidates(target_features, x_candidate, y_candidate).squeeze(1)
                for source_features in feature_list:
                    current = (source_features - sampled).abs().sum(dim=1, dtype=torch.float32)
                    best_pair = current if best_pair is None else torch.minimum(best_pair, current)
            l1[:, candidate_idx] = best_pair

        if oob.any():
            l1 = l1.masked_fill(oob.permute(0, 3, 1, 2), 1e6)
        if exclude_self:
            zero_mask = (candidates[..., 0].abs() + candidates[..., 1].abs()) < 0.5
            if zero_mask.any():
                l1 = l1.masked_fill(zero_mask.permute(0, 3, 1, 2), 1e6)

        weights = torch.softmax(-beta * l1, dim=1)
        weighted_candidates = candidates.permute(0, 3, 1, 2, 4)
        return (weighted_candidates * weights.unsqueeze(-1)).sum(dim=1)

    @torch.no_grad()
    def evaluate(self, candidates: torch.Tensor, feature_list: list[torch.Tensor], beta: float = 10.0, exclude_self: bool = True):
        _, H, W, _ = candidates.shape[:4]
        x_abs = self.x_grid.unsqueeze(-1) + candidates[..., 0]
        y_abs = self.y_grid.unsqueeze(-1) + candidates[..., 1]
        oob = (x_abs < 0) | (x_abs > (W - 1)) | (y_abs < 0) | (y_abs > (H - 1))

        l1 = None
        for target_features in feature_list:
            sampled_candidates = self.sample_candidates(target_features, x_abs, y_abs)
            source_pair_costs = []
            for source_features in feature_list:
                current = (source_features.unsqueeze(1) - sampled_candidates).abs().sum(dim=2, dtype=torch.float32)
                source_pair_costs.append(current)
            target_best = torch.amin(torch.stack(source_pair_costs, dim=1), dim=1)
            l1 = target_best if l1 is None else torch.minimum(l1, target_best)

        if oob.any():
            l1 = l1.masked_fill(oob.permute(0, 3, 1, 2), 1e6)
        if exclude_self:
            zero_mask = (candidates[..., 0].abs() + candidates[..., 1].abs()) < 0.5
            if zero_mask.any():
                l1 = l1.masked_fill(zero_mask.permute(0, 3, 1, 2), 1e6)

        weights = torch.softmax(-beta * l1, dim=1)
        weighted_candidates = candidates.permute(0, 3, 1, 2, 4)
        return (weighted_candidates * weights.unsqueeze(-1)).sum(dim=1)

    def _restore_offsets(self, offsets: torch.Tensor) -> torch.Tensor:
        if self.single_image:
            return offsets.squeeze(0)
        return offsets

    @torch.no_grad()
    def run_branch(
        self,
        feature_list: list[torch.Tensor],
        iters: int,
        beta: float,
        use_non_local: bool,
        non_local_limit: float,
        reference_evaluate: bool = False,
    ) -> torch.Tensor:
        evaluate_fn = self.evaluate_reference if reference_evaluate else self.evaluate
        dx, dy = self.generate_random_offsets()
        for iter_idx in range(iters):
            base = torch.stack((dx, dy), dim=-1).unsqueeze(-2)
            propagated = self.propagation_block(dx, dy)
            random_candidates = self.random_search_block(dx, dy, num_random=4)
            candidates = torch.cat((base, propagated, random_candidates), dim=-2)
            best = evaluate_fn(candidates, feature_list, beta=beta, exclude_self=True)
            dx, dy = best[..., 0], best[..., 1]
            if use_non_local and iter_idx < (iters - 1):
                dx, dy = self.non_local_reset(dx, dy, limit_u=non_local_limit)
        return torch.stack((dx, dy), dim=1)

    @torch.no_grad()
    def propagation_layer(
        self,
        iters: int = 24,
        beta: float = 2.5,
        random_window: int | None = None,
        use_non_local: bool = False,
        non_local_limit: float = 25.0,
        reference_evaluate: bool = False,
    ):
        if random_window is not None:
            self.random_window = int(random_window)

        cnn_offsets = self.run_branch(
            self.cnn_features,
            iters=iters,
            beta=beta,
            use_non_local=use_non_local,
            non_local_limit=non_local_limit,
            reference_evaluate=reference_evaluate,
        )
        zernike_offsets = self.run_branch(
            self.zernike_features,
            iters=iters,
            beta=beta,
            use_non_local=use_non_local,
            non_local_limit=non_local_limit,
            reference_evaluate=reference_evaluate,
        )
        return self._restore_offsets(cnn_offsets), self._restore_offsets(zernike_offsets)
