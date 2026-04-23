from __future__ import annotations

import torch
import torch.nn.functional as F

from datatypes import PatchMatchBranchResult


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
        _, _, height, width = self.image.shape
        self.H = height
        self.W = width
        self.match_dtype = self.resolve_match_dtype(reduced_precision)
        self.structure_map = self.compute_structure_map().to(device=self.device, dtype=torch.float32)

        y_grid, x_grid = torch.meshgrid(
            torch.arange(height, device=self.device, dtype=torch.float32),
            torch.arange(width, device=self.device, dtype=torch.float32),
            indexing="ij",
        )
        self.x_grid = x_grid.unsqueeze(0)
        self.y_grid = y_grid.unsqueeze(0)
        self.x_min_full = -self.x_grid
        self.x_max_full = (width - 1.0) - self.x_grid
        self.y_min_full = -self.y_grid
        self.y_max_full = (height - 1.0) - self.y_grid

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

    def compute_structure_map(self, kernel_size: int = 7) -> torch.Tensor:
        with torch.no_grad():
            image = self.image.detach().to(device=self.device, dtype=torch.float32)
            if image.shape[1] >= 3:
                weights = image.new_tensor([0.2989, 0.5870, 0.1140]).view(1, 3, 1, 1)
                gray = (image[:, :3] * weights).sum(dim=1, keepdim=True)
            else:
                gray = image.mean(dim=1, keepdim=True)

            pad = kernel_size // 2
            padded = F.pad(gray, (pad, pad, pad, pad), mode="replicate")
            local_mean = F.avg_pool2d(padded, kernel_size=kernel_size, stride=1)
            local_mean_sq = F.avg_pool2d(padded.square(), kernel_size=kernel_size, stride=1)
            local_std = (local_mean_sq - local_mean.square()).clamp_min(0.0).sqrt()

            sobel_x = gray.new_tensor(
                [[[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]]
            ).unsqueeze(0)
            sobel_y = gray.new_tensor(
                [[[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]]
            ).unsqueeze(0)
            padded_gray = F.pad(gray, (1, 1, 1, 1), mode="replicate")
            grad_x = F.conv2d(padded_gray, sobel_x)
            grad_y = F.conv2d(padded_gray, sobel_y)
            grad_mag = torch.sqrt(grad_x.square() + grad_y.square() + 1e-12)

            std_norm = self.normalize_per_image(local_std)
            grad_norm = self.normalize_per_image(grad_mag)
            return 0.5 * std_norm + 0.5 * grad_norm

    def normalize_per_image(self, tensor: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        flat = tensor.flatten(start_dim=1)
        min_values = flat.min(dim=1, keepdim=True).values.view(-1, 1, 1, 1)
        max_values = flat.max(dim=1, keepdim=True).values.view(-1, 1, 1, 1)
        return (tensor - min_values) / (max_values - min_values + eps)

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
        batch_size, height, width = self.batch_size, self.H, self.W
        x_min, x_max, y_min, y_max = self._scaled_bounds()

        x_range = (x_max - x_min + 1).clamp(min=1)
        y_range = (y_max - y_min + 1).clamp(min=1)

        dx = torch.floor(torch.rand(batch_size, height, width, device=self.device) * x_range + x_min).long()
        dy = torch.floor(torch.rand(batch_size, height, width, device=self.device) * y_range + y_min).long()

        zero_mask = (dx == 0) & (dy == 0)
        if zero_mask.any():
            dy_alt = torch.where(dy + 1 <= y_max, dy + 1, dy - 1)
            dy = dy.clone()
            dy[zero_mask] = dy_alt[zero_mask]

        return dx.float(), dy.float()

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

    def random_search_block(
        self,
        dx: torch.Tensor,
        dy: torch.Tensor,
        radius: int | None = None,
        num_random: int = 4,
    ) -> torch.Tensor:
        batch_size, height, width = dx.shape
        radius = max(1, self.random_window // 2) if radius is None else int(radius)

        rand_dx = torch.randint(-radius, radius + 1, (batch_size, height, width, num_random), device=self.device)
        rand_dy = torch.randint(-radius, radius + 1, (batch_size, height, width, num_random), device=self.device)

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

    def non_local_reset(
        self,
        dx: torch.Tensor,
        dy: torch.Tensor,
        limit_u: float = 25.0,
        ambiguous_mask: torch.Tensor | None = None,
    ):
        limit_sq = float(limit_u) * float(limit_u)
        reset_mask = (dx * dx + dy * dy) <= limit_sq
        if ambiguous_mask is not None:
            reset_mask = reset_mask | ambiguous_mask.to(dtype=torch.bool, device=dx.device)
        if not reset_mask.any():
            return dx, dy

        rand_dx, rand_dy = self.generate_random_offsets()
        dx = torch.where(reset_mask, rand_dx, dx)
        dy = torch.where(reset_mask, rand_dy, dy)
        return dx, dy

    def normalize_grid(self, values: torch.Tensor, size: int, dtype: torch.dtype) -> torch.Tensor:
        values = values.clamp(0, size - 1).to(dtype=dtype)
        if size > 1:
            return 2.0 * (values / float(size - 1)) - 1.0
        return torch.zeros_like(values, dtype=dtype)

    def sample_candidates(self, feature_map: torch.Tensor, x_abs: torch.Tensor, y_abs: torch.Tensor) -> torch.Tensor:
        batch_size, _, height, width = feature_map.shape
        num_candidates = x_abs.shape[-1]
        grid_dtype = feature_map.dtype

        x_grid = self.normalize_grid(x_abs, width, grid_dtype)
        y_grid = self.normalize_grid(y_abs, height, grid_dtype)

        x_grid = x_grid.permute(0, 1, 3, 2).reshape(batch_size, height, num_candidates * width)
        y_grid = y_grid.permute(0, 1, 3, 2).reshape(batch_size, height, num_candidates * width)
        grid = torch.stack((x_grid, y_grid), dim=-1)

        sampled = F.grid_sample(
            feature_map,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        sampled = sampled.reshape(batch_size, feature_map.shape[1], height, num_candidates, width)
        return sampled.permute(0, 3, 1, 2, 4)

    def finalize_candidate_costs(
        self,
        candidates: torch.Tensor,
        l1: torch.Tensor,
        x_abs: torch.Tensor,
        y_abs: torch.Tensor,
        exclude_self: bool,
    ) -> torch.Tensor:
        _, height, width, _ = candidates.shape[:4]
        oob = (x_abs < 0) | (x_abs > (width - 1)) | (y_abs < 0) | (y_abs > (height - 1))
        if oob.any():
            l1 = l1.masked_fill(oob.permute(0, 3, 1, 2), 1e6)
        if exclude_self:
            zero_mask = (candidates[..., 0].abs() + candidates[..., 1].abs()) < 0.5
            if zero_mask.any():
                l1 = l1.masked_fill(zero_mask.permute(0, 3, 1, 2), 1e6)
        return l1

    def compute_candidate_costs_reference(
        self,
        candidates: torch.Tensor,
        feature_list: list[torch.Tensor],
        exclude_self: bool = True,
    ) -> torch.Tensor:
        batch_size, height, width, num_candidates = candidates.shape[:4]
        x_abs = self.x_grid.unsqueeze(-1) + candidates[..., 0]
        y_abs = self.y_grid.unsqueeze(-1) + candidates[..., 1]

        l1 = torch.empty((batch_size, num_candidates, height, width), device=candidates.device, dtype=torch.float32)
        for candidate_idx in range(num_candidates):
            best_pair = None
            x_candidate = x_abs[..., candidate_idx : candidate_idx + 1]
            y_candidate = y_abs[..., candidate_idx : candidate_idx + 1]
            for target_features in feature_list:
                sampled = self.sample_candidates(target_features, x_candidate, y_candidate).squeeze(1)
                for source_features in feature_list:
                    current = (source_features - sampled).abs().sum(dim=1, dtype=torch.float32)
                    best_pair = current if best_pair is None else torch.minimum(best_pair, current)
            l1[:, candidate_idx] = best_pair

        return self.finalize_candidate_costs(candidates, l1, x_abs, y_abs, exclude_self=exclude_self)

    def compute_candidate_costs(
        self,
        candidates: torch.Tensor,
        feature_list: list[torch.Tensor],
        exclude_self: bool = True,
    ) -> torch.Tensor:
        x_abs = self.x_grid.unsqueeze(-1) + candidates[..., 0]
        y_abs = self.y_grid.unsqueeze(-1) + candidates[..., 1]

        l1 = None
        for target_features in feature_list:
            sampled_candidates = self.sample_candidates(target_features, x_abs, y_abs)
            source_pair_costs = []
            for source_features in feature_list:
                current = (source_features.unsqueeze(1) - sampled_candidates).abs().sum(dim=2, dtype=torch.float32)
                source_pair_costs.append(current)
            target_best = torch.amin(torch.stack(source_pair_costs, dim=1), dim=1)
            l1 = target_best if l1 is None else torch.minimum(l1, target_best)

        if l1 is None:
            raise RuntimeError("PatchMatch candidate evaluation received an empty feature list.")
        return self.finalize_candidate_costs(candidates, l1, x_abs, y_abs, exclude_self=exclude_self)

    def select_candidates(
        self,
        candidates: torch.Tensor,
        l1: torch.Tensor,
        beta: float,
        hard_selection: bool,
    ) -> torch.Tensor:
        candidate_values = candidates.permute(0, 3, 1, 2, 4)
        if hard_selection:
            best_idx = torch.argmin(l1, dim=1, keepdim=True)
            gather_idx = best_idx.unsqueeze(-1).expand(-1, -1, -1, -1, candidate_values.shape[-1])
            return torch.gather(candidate_values, dim=1, index=gather_idx).squeeze(1)

        weights = torch.softmax(-beta * l1, dim=1)
        return (candidate_values * weights.unsqueeze(-1)).sum(dim=1)

    def summarize_candidate_costs(self, l1: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if l1.shape[1] == 1:
            best_cost = l1[:, :1]
            second_cost = best_cost.clone()
        else:
            top2_costs, _ = torch.topk(l1, k=2, largest=False, dim=1)
            best_cost = top2_costs[:, :1]
            second_cost = top2_costs[:, 1:2]
        confidence = ((second_cost - best_cost) / (second_cost + 1e-6)).clamp_(0.0, 1.0)
        return best_cost, second_cost, confidence

    def evaluate_reference(
        self,
        candidates: torch.Tensor,
        feature_list: list[torch.Tensor],
        beta: float = 10.0,
        exclude_self: bool = True,
        hard_selection: bool = False,
    ):
        l1 = self.compute_candidate_costs_reference(candidates, feature_list, exclude_self=exclude_self)
        selected = self.select_candidates(candidates, l1, beta=beta, hard_selection=hard_selection)
        best_cost, second_cost, confidence = self.summarize_candidate_costs(l1)
        return {
            "selected": selected,
            "best_cost": best_cost,
            "second_cost": second_cost,
            "confidence": confidence,
        }

    def evaluate(
        self,
        candidates: torch.Tensor,
        feature_list: list[torch.Tensor],
        beta: float = 10.0,
        exclude_self: bool = True,
        hard_selection: bool = False,
    ):
        l1 = self.compute_candidate_costs(candidates, feature_list, exclude_self=exclude_self)
        selected = self.select_candidates(candidates, l1, beta=beta, hard_selection=hard_selection)
        best_cost, second_cost, confidence = self.summarize_candidate_costs(l1)
        return {
            "selected": selected,
            "best_cost": best_cost,
            "second_cost": second_cost,
            "confidence": confidence,
        }

    def _restore_tensor(self, tensor: torch.Tensor | None) -> torch.Tensor | None:
        if tensor is None:
            return None
        if self.single_image:
            return tensor.squeeze(0)
        return tensor

    def _restore_branch_result(self, branch_result: PatchMatchBranchResult) -> PatchMatchBranchResult:
        return PatchMatchBranchResult(
            offsets=self._restore_tensor(branch_result.offsets),
            best_cost=self._restore_tensor(branch_result.best_cost),
            second_cost=self._restore_tensor(branch_result.second_cost),
            confidence=self._restore_tensor(branch_result.confidence),
            structure_map=self._restore_tensor(branch_result.structure_map),
        )

    def run_branch(
        self,
        feature_list: list[torch.Tensor],
        iters: int,
        beta: float,
        use_non_local: bool,
        non_local_limit: float,
        flat_threshold: float,
        margin_threshold: float,
        hard_selection: bool = False,
        reference_evaluate: bool = False,
    ) -> PatchMatchBranchResult:
        evaluate_fn = self.evaluate_reference if reference_evaluate else self.evaluate
        dx, dy = self.generate_random_offsets()
        best_cost = None
        second_cost = None
        confidence = None

        for iter_idx in range(iters):
            base = torch.stack((dx, dy), dim=-1).unsqueeze(-2)
            propagated = self.propagation_block(dx, dy)
            random_candidates = self.random_search_block(dx, dy, num_random=4)
            candidates = torch.cat((base, propagated, random_candidates), dim=-2)
            evaluated = evaluate_fn(
                candidates,
                feature_list,
                beta=beta,
                exclude_self=True,
                hard_selection=hard_selection,
            )
            best = evaluated["selected"]
            dx, dy = best[..., 0], best[..., 1]
            best_cost = evaluated["best_cost"]
            second_cost = evaluated["second_cost"]
            confidence = evaluated["confidence"]
            if use_non_local and iter_idx < (iters - 1):
                ambiguous_flat = (self.structure_map < float(flat_threshold)) & (confidence < float(margin_threshold))
                dx, dy = self.non_local_reset(
                    dx,
                    dy,
                    limit_u=non_local_limit,
                    ambiguous_mask=ambiguous_flat.squeeze(1),
                )

        if best_cost is None or second_cost is None or confidence is None:
            raise RuntimeError("PatchMatch did not complete any iterations.")

        return PatchMatchBranchResult(
            offsets=torch.stack((dx, dy), dim=1),
            best_cost=best_cost,
            second_cost=second_cost,
            confidence=confidence,
            structure_map=self.structure_map.clone(),
        )

    def propagation_layer(
        self,
        iters: int = 24,
        beta: float = 2.5,
        random_window: int | None = None,
        use_non_local: bool = False,
        non_local_limit: float = 25.0,
        flat_threshold: float = 0.15,
        margin_threshold: float = 0.10,
        hard_selection: bool = False,
        reference_evaluate: bool = False,
    ):
        if random_window is not None:
            self.random_window = int(random_window)

        cnn_result = self.run_branch(
            self.cnn_features,
            iters=iters,
            beta=beta,
            use_non_local=use_non_local,
            non_local_limit=non_local_limit,
            flat_threshold=flat_threshold,
            margin_threshold=margin_threshold,
            hard_selection=hard_selection,
            reference_evaluate=reference_evaluate,
        )
        zernike_result = self.run_branch(
            self.zernike_features,
            iters=iters,
            beta=beta,
            use_non_local=use_non_local,
            non_local_limit=non_local_limit,
            flat_threshold=flat_threshold,
            margin_threshold=margin_threshold,
            hard_selection=hard_selection,
            reference_evaluate=reference_evaluate,
        )
        return self._restore_branch_result(cnn_result), self._restore_branch_result(zernike_result)
