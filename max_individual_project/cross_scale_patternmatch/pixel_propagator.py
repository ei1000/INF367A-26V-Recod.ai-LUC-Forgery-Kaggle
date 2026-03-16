import torch
import torch.nn.functional as F


class PixelPropagator:
    def __init__(
        self,
        image: torch.Tensor,
        cnn_features,
        zernike_features,
        random_window: int = 50,
    ):
        self.image = image
        self.cnn_features = cnn_features
        self.zernike_features = zernike_features
        self.random_window = int(random_window)
        self.max_fraction = 1.0

        _, H, W = self.image.size()
        self.H = H
        self.W = W
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=self.image.device),
            torch.arange(W, device=self.image.device),
            indexing="ij",
        )
        self.x_grid = x_grid
        self.y_grid = y_grid
        self.x_min_full = -self.x_grid
        self.x_max_full = (W - 1) - self.x_grid
        self.y_min_full = -self.y_grid
        self.y_max_full = (H - 1) - self.y_grid

    def _scaled_bounds(self):
        if self.max_fraction >= 1:
            return self.x_min_full, self.x_max_full, self.y_min_full, self.y_max_full
        x_min = torch.ceil(self.x_min_full.float() * self.max_fraction)
        x_max = torch.floor(self.x_max_full.float() * self.max_fraction)
        y_min = torch.ceil(self.y_min_full.float() * self.max_fraction)
        y_max = torch.floor(self.y_max_full.float() * self.max_fraction)
        return x_min, x_max, y_min, y_max

    @torch.no_grad()
    def generate_random_offsets(self):
        H, W = self.H, self.W
        x_min, x_max, y_min, y_max = self._scaled_bounds()

        x_range = (x_max - x_min + 1).clamp(min=1).float()
        y_range = (y_max - y_min + 1).clamp(min=1).float()

        dx = torch.floor(torch.rand(H, W, device=self.image.device) * x_range + x_min).long()
        dy = torch.floor(torch.rand(H, W, device=self.image.device) * y_range + y_min).long()

        zero_mask = (dx == 0) & (dy == 0)
        if zero_mask.any():
            dy_alt = torch.where(dy + 1 <= y_max, dy + 1, dy - 1)
            dy = dy.clone()
            dy[zero_mask] = dy_alt[zero_mask]

        return dx.float(), dy.float()

    @torch.no_grad()
    def propagation_block(self, dx: torch.Tensor, dy: torch.Tensor) -> torch.Tensor:
        # Circular-shift propagation (paper + reference implementation style).
        def r(t, sh, sw):
            return torch.roll(t, shifts=(sh, sw), dims=(0, 1))

        zero_dx = [r(dx, 0, 1), r(dx, 0, -1), r(dx, 1, 0), r(dx, -1, 0)]
        zero_dy = [r(dy, 0, 1), r(dy, 0, -1), r(dy, 1, 0), r(dy, -1, 0)]

        first_dx = [
            2 * r(dx, 0, 1) - r(dx, 0, 2),
            2 * r(dx, 0, -1) - r(dx, 0, -2),
            2 * r(dx, 1, 0) - r(dx, 2, 0),
            2 * r(dx, -1, 0) - r(dx, -2, 0),
            2 * r(dx, 1, 1) - r(dx, 2, 2),
            2 * r(dx, -1, -1) - r(dx, -2, -2),
            2 * r(dx, 1, -1) - r(dx, 2, -2),
            2 * r(dx, -1, 1) - r(dx, -2, 2),
        ]
        first_dy = [
            2 * r(dy, 0, 1) - r(dy, 0, 2),
            2 * r(dy, 0, -1) - r(dy, 0, -2),
            2 * r(dy, 1, 0) - r(dy, 2, 0),
            2 * r(dy, -1, 0) - r(dy, -2, 0),
            2 * r(dy, 1, 1) - r(dy, 2, 2),
            2 * r(dy, -1, -1) - r(dy, -2, -2),
            2 * r(dy, 1, -1) - r(dy, 2, -2),
            2 * r(dy, -1, 1) - r(dy, -2, 2),
        ]

        cand_dx = torch.stack(zero_dx + first_dx, dim=2)
        cand_dy = torch.stack(zero_dy + first_dy, dim=2)
        return torch.stack((cand_dx, cand_dy), dim=-1)

    @torch.no_grad()
    def random_search_block(
        self,
        dx: torch.Tensor,
        dy: torch.Tensor,
        radius: int | None = None,
        num_random: int = 4,
    ) -> torch.Tensor:
        H, W = dx.shape
        radius = max(1, self.random_window // 2) if radius is None else int(radius)

        rand_dx = torch.randint(-radius, radius + 1, (H, W, num_random), device=self.image.device)
        rand_dy = torch.randint(-radius, radius + 1, (H, W, num_random), device=self.image.device)

        cand_dx = dx.unsqueeze(-1) + rand_dx
        cand_dy = dy.unsqueeze(-1) + rand_dy

        x_min = self.x_min_full.unsqueeze(-1).float()
        x_max = self.x_max_full.unsqueeze(-1).float()
        y_min = self.y_min_full.unsqueeze(-1).float()
        y_max = self.y_max_full.unsqueeze(-1).float()
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

    def _to_hwc(self, feat: torch.Tensor, H: int, W: int) -> torch.Tensor:
        if feat.dim() != 3:
            raise ValueError(f"Expected 3D feature map, got {feat.shape}")

        if feat.shape[-2:] == (H, W):
            chw = feat
        elif feat.shape[:2] == (H, W):
            chw = feat.permute(2, 0, 1)
        else:
            if feat.shape[0] in (H, W):
                chw = feat.permute(2, 0, 1)
            else:
                chw = feat
            chw = F.interpolate(
                chw.unsqueeze(0),
                size=(H, W),
                mode="bilinear",
                align_corners=True,
            ).squeeze(0)

        return chw.permute(1, 2, 0).contiguous()

    def _sample_candidate(self, feat_hwc: torch.Tensor, x_abs: torch.Tensor, y_abs: torch.Tensor) -> torch.Tensor:
        H, W = feat_hwc.shape[:2]
        feat = feat_hwc.permute(2, 0, 1).unsqueeze(0)

        x = x_abs.clamp(0, W - 1)
        y = y_abs.clamp(0, H - 1)
        if W > 1:
            x = 2.0 * (x / (W - 1.0)) - 1.0
        else:
            x = torch.zeros_like(x)
        if H > 1:
            y = 2.0 * (y / (H - 1.0)) - 1.0
        else:
            y = torch.zeros_like(y)

        grid = torch.stack((x, y), dim=-1).unsqueeze(0)
        sampled = F.grid_sample(
            feat,
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=True,
        )
        return sampled.squeeze(0).permute(1, 2, 0).contiguous()

    @torch.no_grad()
    def evaluate(self, candidates: torch.Tensor, features, beta: float = 2.5, exclude_self: bool = True):
        H, W, K = candidates.shape[:3]

        if isinstance(features, (tuple, list)):
            src_list = [self._to_hwc(f, H, W) for f in features]
            tgt_list = [self._to_hwc(f, H, W) for f in features]
        else:
            feat_hwc = self._to_hwc(features, H, W)
            src_list = [feat_hwc]
            tgt_list = [feat_hwc]

        x_abs = self.x_grid.unsqueeze(2).float() + candidates[..., 0]
        y_abs = self.y_grid.unsqueeze(2).float() + candidates[..., 1]
        oob = (x_abs < 0) | (x_abs > (W - 1)) | (y_abs < 0) | (y_abs > (H - 1))

        l1 = torch.empty((H, W, K), device=candidates.device, dtype=src_list[0].dtype)
        for k in range(K):
            cand_features = [self._sample_candidate(tgt, x_abs[..., k], y_abs[..., k]) for tgt in tgt_list]
            best_pair = None
            for src in src_list:
                for cand in cand_features:
                    cur = (src - cand).abs().sum(dim=-1)
                    best_pair = cur if best_pair is None else torch.minimum(best_pair, cur)
            l1[..., k] = best_pair

        if oob.any():
            l1 = l1.masked_fill(oob, 1e6)
        if exclude_self:
            zero_mask = (candidates[..., 0].abs() + candidates[..., 1].abs()) < 0.5
            if zero_mask.any():
                l1 = l1.masked_fill(zero_mask, 1e6)

        weights = torch.softmax(-beta * l1, dim=2)
        return (candidates * weights.unsqueeze(-1)).sum(dim=2)

    @torch.no_grad()
    def propagation_layer(self, iters: int = 24, beta: float = 2.5, random_window: int | None = None):
        if random_window is not None:
            self.random_window = int(random_window)

        # CNN branch
        dx, dy = self.generate_random_offsets()
        for _ in range(iters):
            base = torch.stack((dx, dy), dim=-1).unsqueeze(2)          # 1
            prop = self.propagation_block(dx, dy)                      # 12
            rand = self.random_search_block(dx, dy, num_random=4)      # 4
            candidates = torch.cat((base, prop, rand), dim=2)          # 17
            best = self.evaluate(candidates, self.cnn_features, beta=beta, exclude_self=True)
            dx, dy = best[..., 0], best[..., 1]
        cnn_offsets = torch.stack((dx, dy))

        # Zernike branch
        dx, dy = self.generate_random_offsets()
        for _ in range(iters):
            base = torch.stack((dx, dy), dim=-1).unsqueeze(2)
            prop = self.propagation_block(dx, dy)
            rand = self.random_search_block(dx, dy, num_random=4)
            candidates = torch.cat((base, prop, rand), dim=2)
            best = self.evaluate(candidates, self.zernike_features, beta=beta, exclude_self=True)
            dx, dy = best[..., 0], best[..., 1]
        zernike_offsets = torch.stack((dx, dy))

        return cnn_offsets, zernike_offsets
