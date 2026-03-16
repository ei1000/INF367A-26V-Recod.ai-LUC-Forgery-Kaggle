import torch
import torch.nn.functional as F


class PixelPropagator:
    def __init__(self, image: torch.Tensor, cnn_features, zernike_features):
        self.image = image
        self.cnn_features = cnn_features
        self.zernike_features = zernike_features
        self.max_fraction = 1

        _, H, W = self.image.size()
        self.H = H
        self.W = W
        y_grid, x_grid = torch.meshgrid(
            torch.arange(H, device=self.image.device),
            torch.arange(W, device=self.image.device),
            indexing='ij',
        )
        self.x_grid = x_grid
        self.y_grid = y_grid

        # Full valid displacement bounds per pixel
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

    def _default_random_radii(self):
        r = max(self.H, self.W) // 2
        radii = []
        while r >= 1 and len(radii) < 4:
            radii.append(int(r))
            r //= 2
        if not radii:
            radii = [1]
        return radii

    '''
    i) Initialization layer: Generate random offsets for each pixel

    Returns relative offsets (dx, dy) for each pixel.
    Does not consider trivial offset (0, 0)

    Image:
    Datatype: torch.Tensor
    Dtype: torch.float32
    '''
    @torch.no_grad()
    def generate_random_offsets(self):
        _, H, W = self.image.size()

        x_min, x_max, y_min, y_max = self._scaled_bounds()

        # Sample per-pixel offsets within valid bounds (inclusive)
        x_range = (x_max - x_min + 1).clamp(min=1).float()
        y_range = (y_max - y_min + 1).clamp(min=1).float()

        rx = torch.rand(H, W, device=self.image.device)
        ry = torch.rand(H, W, device=self.image.device)

        dx = torch.floor(rx * x_range + x_min).long()
        dy = torch.floor(ry * y_range + y_min).long()

        # Avoid the trivial (0,0) offset by nudging dy within bounds
        zero_mask = (dx == 0) & (dy == 0)
        if zero_mask.any():
            dy_alt = torch.where(dy + 1 <= y_max, dy + 1, dy - 1)
            dy = dy.clone()
            dy[zero_mask] = dy_alt[zero_mask]

        return dx.float(), dy.float()

    '''
    ii) Propagation layer: Propagate pixel offsets to neighbouring pixels.
    Utilizes propagation block and random search block

    Returns final pixel offset maps for zernike and cnn features
    '''
    @torch.no_grad()
    def propagation_layer(self, iters=5, beta=5, random_radii=None, random_k=2):
        # CNN
        x, y = self.generate_random_offsets()

        for _ in range(iters):
            basic_candidates = torch.stack((x, y), dim=-1).unsqueeze(2)
            propagation_candidates = self.propagation_block(x, y)

            radii = self._default_random_radii() if random_radii is None else list(random_radii)
            random_list = [self.random_search_block(x, y, radius=r, k=random_k) for r in radii]
            random_candidates = torch.cat(random_list, dim=2)

            candidates = torch.cat((basic_candidates, propagation_candidates, random_candidates), dim=2)
            best = self.evaluate(candidates, self.cnn_features, beta=beta, exclude_self=True)
            x, y = best[..., 0], best[..., 1]

        # store relative offsets (dx, dy)
        cnn_offsets = torch.stack((x, y))

        # ZERNIKE
        x, y = self.generate_random_offsets()

        for _ in range(iters):
            basic_candidates = torch.stack((x, y), dim=-1).unsqueeze(2)
            propagation_candidates = self.propagation_block(x, y)

            radii = self._default_random_radii() if random_radii is None else list(random_radii)
            random_list = [self.random_search_block(x, y, radius=r, k=random_k) for r in radii]
            random_candidates = torch.cat(random_list, dim=2)

            candidates = torch.cat((basic_candidates, propagation_candidates, random_candidates), dim=2)
            best = self.evaluate(candidates, self.zernike_features, beta=beta, exclude_self=True)
            x, y = best[..., 0], best[..., 1]

        zernike_offsets = torch.stack((x, y))

        return (cnn_offsets, zernike_offsets)

    '''
    Propagate to every pixel from surrounding pixels
    Uses a combination of zero order and first order offset

    Every (x, y) entry takes offsets from surrounding pixels as candidates.
    Then directly evaluate all candidates - evaluate them to avoid memory usage.
    Return the best to be checked against the random search candidates.
    '''
    @torch.no_grad()
    def propagation_block(self, x_entries: torch.Tensor, y_entries: torch.Tensor) -> torch.Tensor:
        H, W = x_entries.shape

        # Padding: To include borders
        pad_zero = 1   # for 3x3 windows
        pad_first = 2  # for 5x5 windows

        x_entries_f = x_entries.float()
        y_entries_f = y_entries.float()

        x_zero = F.unfold(
            x_entries_f.unsqueeze(0).unsqueeze(0),
            kernel_size=3,
            padding=pad_zero,
        ).view(1, 9, H, W)
        y_zero = F.unfold(
            y_entries_f.unsqueeze(0).unsqueeze(0),
            kernel_size=3,
            padding=pad_zero,
        ).view(1, 9, H, W)

        # center of zero-order windows (index 4 in 3x3)
        x_center = x_zero[:, 4]
        y_center = y_zero[:, 4]

        # 4 adjacent neighbors in zero-order (up, left, right, down)
        adj_idx = [1, 3, 5, 7]
        adj_x = x_zero[:, adj_idx].permute(2, 3, 1, 0).squeeze(-1)
        adj_y = y_zero[:, adj_idx].permute(2, 3, 1, 0).squeeze(-1)

        # sliding windows for first-order (5x5)
        x_first = F.unfold(
            x_entries_f.unsqueeze(0).unsqueeze(0),
            kernel_size=5,
            padding=pad_first,
        ).view(1, 25, H, W)
        y_first = F.unfold(
            y_entries_f.unsqueeze(0).unsqueeze(0),
            kernel_size=5,
            padding=pad_first,
        ).view(1, 25, H, W)

        # 8 first-order neighbors (corners + midpoints)
        first_idx = [0, 2, 4, 10, 14, 20, 22, 24]
        x_first = x_first[:, first_idx].permute(2, 3, 1, 0).squeeze(-1)
        y_first = y_first[:, first_idx].permute(2, 3, 1, 0).squeeze(-1)

        # first-order candidates: 2*zero_order_center - first_order
        first_x = 2 * x_center.permute(1, 2, 0) - x_first
        first_y = 2 * y_center.permute(1, 2, 0) - y_first

        # stack adjacent neighbors and first-order candidates
        adj = torch.stack((adj_x, adj_y), dim=-1)
        first = torch.stack((first_x, first_y), dim=-1)
        candidates = torch.cat((adj, first), dim=2)

        return candidates

    '''
    Generate k candidates within the radius for each pixel to test similarity to

    '''
    @torch.no_grad()
    def random_search_block(self, x_entries, y_entries, radius=50, k=4):
        H, W = x_entries.shape
        x_entries_f = x_entries.float()
        y_entries_f = y_entries.float()

        # Sample k random offsets per pixel within the radius (avoid large unfold buffers)
        dx = torch.randint(-radius, radius + 1, (H, W, k), device=self.image.device)
        dy = torch.randint(-radius, radius + 1, (H, W, k), device=self.image.device)

        # Avoid the trivial (0,0) offset by nudging dy when both are zero
        zero_mask = (dx == 0) & (dy == 0)
        if zero_mask.any():
            dy = dy.clone()
            dy[zero_mask] = 1

        # Candidate offsets (relative) with per-pixel bounds
        x_candidates = x_entries_f.unsqueeze(-1) + dx
        y_candidates = y_entries_f.unsqueeze(-1) + dy

        x_min = self.x_min_full.unsqueeze(-1).float()
        x_max = self.x_max_full.unsqueeze(-1).float()
        y_min = self.y_min_full.unsqueeze(-1).float()
        y_max = self.y_max_full.unsqueeze(-1).float()

        x_candidates = torch.max(torch.min(x_candidates, x_max), x_min)
        y_candidates = torch.max(torch.min(y_candidates, y_max), y_min)

        candidates = torch.stack((x_candidates, y_candidates), dim=-1)
        return candidates

    @torch.no_grad()
    def evaluate(self, offsets, features, beta=1, exclude_self=True):
        '''
        Evaluate candidates based on feature map in use.
        Returns the best for each pixel.

        Uses relaxed argmax (softmax)
        '''
        H, W = offsets.shape[:2]

        def to_hw_c(feat):
            if feat.dim() != 3:
                raise ValueError(f"features must be 3D (C,H,W) or (H,W,C), got {feat.shape}")

            # Normalize to (C, H, W)
            if feat.shape[-2:] == (H, W):
                feat_chw = feat
            elif feat.shape[:2] == (H, W):
                feat_chw = feat.permute(2, 0, 1)
            else:
                # Resize any mismatched spatial dims (e.g. even-kernel conv output)
                feat_chw = feat if feat.shape[0] != H else feat.permute(2, 0, 1)
                feat_chw = F.interpolate(
                    feat_chw.unsqueeze(0),
                    size=(H, W),
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(0)

            return feat_chw.permute(1, 2, 0).contiguous()

        # Convert relative offsets to absolute coordinates for sampling
        x = self.x_grid.unsqueeze(2).float() + offsets[..., 0]
        y = self.y_grid.unsqueeze(2).float() + offsets[..., 1]

        # Candidate coords and OOB mask (shared across all feature maps)
        oob = (x < 0) | (x > (W - 1)) | (y < 0) | (y > (H - 1))

        x_idx = x.round().long().clamp(0, W - 1)
        y_idx = y.round().long().clamp(0, H - 1)
        idx = (y_idx * W + x_idx).view(-1)

        def gather_cand(feat_hw_c):
            flat = feat_hw_c.view(H * W, -1)
            return flat[idx].view(H, W, -1, flat.shape[-1])  # (H, W, K, C)

        # Compute L1 scores
        if isinstance(features, (tuple, list)):
            # Cross-scale matching: max score == min L1 across all (n, m) pairs
            src_list = [to_hw_c(f) for f in features]
            tgt_list = [to_hw_c(f) for f in features]
            cand_list = [gather_cand(tgt) for tgt in tgt_list]

            l1_list = []
            for src in src_list:
                src_feat = src.unsqueeze(2)  # (H, W, 1, C_src)
                for cand in cand_list:
                    l1_list.append((src_feat - cand).abs().sum(dim=-1))  # (H, W, K)

            l1 = torch.stack(l1_list, dim=0).min(dim=0).values
        else:
            feat_hw_c = to_hw_c(features)
            cand = gather_cand(feat_hw_c)
            src_feat = feat_hw_c.unsqueeze(2)
            l1 = (src_feat - cand).abs().sum(dim=-1)

        # Mask invalid candidates
        if oob.any():
            l1 = l1.masked_fill(oob, 1e6)

        if exclude_self:
            zero_mask = (offsets[..., 0].abs() + offsets[..., 1].abs()) == 0
            if zero_mask.any():
                l1 = l1.masked_fill(zero_mask, 1e6)

        # Relaxed argmax (soft selection)
        weights = torch.softmax(-beta * l1, dim=2)  # (H, W, K)
        best = (offsets * weights.unsqueeze(-1)).sum(dim=2)  # (H, W, 2)
        return best
