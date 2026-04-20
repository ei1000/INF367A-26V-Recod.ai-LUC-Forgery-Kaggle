from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class MultiScaleDLF:
    STATIC_MAP_CACHE: dict[tuple[int, int, str, str], tuple[torch.Tensor, ...]] = {}

    ONE_IDX = 0
    X_IDX = 1
    Y_IDX = 2
    X2_IDX = 3
    XY_IDX = 4
    Y2_IDX = 5
    DX_IDX = 6
    DY_IDX = 7
    X_DX_IDX = 8
    Y_DX_IDX = 9
    X_DY_IDX = 10
    Y_DY_IDX = 11

    def __init__(
        self,
        image: torch.Tensor,
        cnn_offsets: torch.Tensor,
        zernike_offsets: torch.Tensor | None = None,
        kernel_sizes: np.ndarray | None = None,
        cnn_topk_offsets: torch.Tensor | None = None,
        zernike_topk_offsets: torch.Tensor | None = None,
    ):
        self.image = self.as_batched_image(image)
        self.cnn_offsets = self.as_batched_offsets(cnn_offsets).to(dtype=torch.float32)
        self.zernike_offsets = (
            self.as_batched_offsets(zernike_offsets).to(dtype=torch.float32)
            if zernike_offsets is not None
            else None
        )
        self.cnn_topk_offsets = (
            self.as_batched_topk_offsets(cnn_topk_offsets).to(dtype=torch.float32)
            if cnn_topk_offsets is not None
            else None
        )
        self.zernike_topk_offsets = (
            self.as_batched_topk_offsets(zernike_topk_offsets).to(dtype=torch.float32)
            if zernike_topk_offsets is not None
            else None
        )

        if self.image.shape[0] != self.cnn_offsets.shape[0]:
            raise ValueError(
                f"Batch size mismatch between image {self.image.shape} and offsets {self.cnn_offsets.shape}"
            )
        if self.image.shape[-2:] != self.cnn_offsets.shape[-2:]:
            raise ValueError(
                f"Spatial size mismatch between image {self.image.shape} and offsets {self.cnn_offsets.shape}"
            )
        if self.zernike_offsets is not None:
            if self.zernike_offsets.shape[0] != self.cnn_offsets.shape[0]:
                raise ValueError(
                    f"Batch size mismatch between CNN offsets {self.cnn_offsets.shape} and "
                    f"Zernike offsets {self.zernike_offsets.shape}"
                )
            if self.zernike_offsets.shape[-2:] != self.cnn_offsets.shape[-2:]:
                raise ValueError(
                    f"Spatial size mismatch between CNN offsets {self.cnn_offsets.shape} and "
                    f"Zernike offsets {self.zernike_offsets.shape}"
                )

        self.validate_topk_offsets(self.cnn_topk_offsets, self.cnn_offsets, "cnn_topk_offsets")
        self.validate_topk_offsets(self.zernike_topk_offsets, self.zernike_offsets, "zernike_topk_offsets")

        if kernel_sizes is None:
            kernel_sizes = np.array([11, 13, 15])

        self.kernel_sizes = kernel_sizes
        self.device = self.cnn_offsets.device
        self.dtype = torch.float32
        self.batch_size = self.image.shape[0]
        self.height = self.image.shape[-2]
        self.width = self.image.shape[-1]
        self.offset_fields = [self.normalize_offsets(self.cnn_offsets)]
        if self.zernike_offsets is not None:
            self.offset_fields.append(self.normalize_offsets(self.zernike_offsets))
        self.hypothesis_groups = self.build_hypothesis_groups()
        self.has_topk_hypotheses = any(len(group) > 1 for group in self.hypothesis_groups)
        self.ones_map, self.x_map, self.y_map, self.x2_map, self.xy_map, self.y2_map = self.get_static_maps(
            self.height,
            self.width,
            self.device,
            self.dtype,
        )

    def as_batched_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 3:
            return image.unsqueeze(0)
        if image.dim() == 4:
            return image
        raise ValueError(f"Expected image shape [C,H,W] or [B,C,H,W], got {tuple(image.shape)}")

    def as_batched_offsets(self, offsets: torch.Tensor | None) -> torch.Tensor:
        if offsets is None:
            raise ValueError("Offsets tensor cannot be None for this call.")
        if offsets.dim() == 3:
            offsets = offsets.unsqueeze(0)
        if offsets.dim() != 4 or offsets.shape[1] != 2:
            raise ValueError(f"Expected offsets shape [2,H,W] or [B,2,H,W], got {tuple(offsets.shape)}")
        return offsets

    def as_batched_topk_offsets(self, offsets: torch.Tensor) -> torch.Tensor:
        if offsets.dim() == 4:
            offsets = offsets.unsqueeze(0)
        if offsets.dim() != 5 or offsets.shape[2] != 2:
            raise ValueError(f"Expected top-k offsets shape [K,2,H,W] or [B,K,2,H,W], got {tuple(offsets.shape)}")
        return offsets

    def validate_topk_offsets(
        self,
        topk_offsets: torch.Tensor | None,
        reference_offsets: torch.Tensor | None,
        name: str,
    ) -> None:
        if topk_offsets is None or reference_offsets is None:
            return
        if topk_offsets.shape[0] != reference_offsets.shape[0]:
            raise ValueError(
                f"Batch size mismatch between {name} {tuple(topk_offsets.shape)} and offsets {tuple(reference_offsets.shape)}"
            )
        if topk_offsets.shape[-2:] != reference_offsets.shape[-2:]:
            raise ValueError(
                f"Spatial size mismatch between {name} {tuple(topk_offsets.shape)} and offsets {tuple(reference_offsets.shape)}"
            )

    @classmethod
    def get_static_maps(
        cls,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> tuple[torch.Tensor, ...]:
        key = (height, width, str(device), str(dtype))
        cached = cls.STATIC_MAP_CACHE.get(key)
        if cached is not None:
            return cached

        x_coords = (
            torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
            if width > 1
            else torch.zeros(1, device=device, dtype=dtype)
        )
        y_coords = (
            torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
            if height > 1
            else torch.zeros(1, device=device, dtype=dtype)
        )
        y, x = torch.meshgrid(y_coords, x_coords, indexing="ij")
        x = x.unsqueeze(0).unsqueeze(0)
        y = y.unsqueeze(0).unsqueeze(0)
        ones = torch.ones_like(x)
        static_maps = (ones, x, y, x * x, x * y, y * y)
        cls.STATIC_MAP_CACHE[key] = static_maps
        return static_maps

    def normalize_offsets(self, offsets: torch.Tensor) -> torch.Tensor:
        normalized = offsets.to(device=self.device, dtype=self.dtype).clone()
        scale_x = float(max(self.width - 1, 1))
        scale_y = float(max(self.height - 1, 1))
        normalized[:, 0] = normalized[:, 0] / scale_x
        normalized[:, 1] = normalized[:, 1] / scale_y
        return normalized

    def normalize_topk_offsets(self, offsets: torch.Tensor) -> torch.Tensor:
        batch_size, topk, _, height, width = offsets.shape
        flat = offsets.reshape(batch_size * topk, 2, height, width)
        normalized = self.normalize_offsets(flat)
        return normalized.reshape(batch_size, topk, 2, height, width)

    def build_hypothesis_groups(self) -> list[list[torch.Tensor]]:
        groups: list[list[torch.Tensor]] = []
        groups.append(self.build_branch_hypotheses(self.cnn_offsets, self.cnn_topk_offsets))
        if self.zernike_offsets is not None:
            groups.append(self.build_branch_hypotheses(self.zernike_offsets, self.zernike_topk_offsets))
        return groups

    def build_branch_hypotheses(
        self,
        base_offsets: torch.Tensor,
        topk_offsets: torch.Tensor | None,
    ) -> list[torch.Tensor]:
        if topk_offsets is None or topk_offsets.shape[1] <= 1:
            return [self.normalize_offsets(base_offsets)]
        normalized = self.normalize_topk_offsets(topk_offsets)
        return [normalized[:, hypothesis_idx] for hypothesis_idx in range(normalized.shape[1])]

    def box_sum(self, tensor: torch.Tensor, kernel_size: int) -> torch.Tensor:
        pad = kernel_size // 2
        padded = F.pad(tensor, (pad, pad, pad, pad))
        integral = padded.cumsum(dim=-2).cumsum(dim=-1)
        integral = F.pad(integral, (1, 0, 1, 0))
        return (
            integral[..., kernel_size:, kernel_size:]
            - integral[..., :-kernel_size, kernel_size:]
            - integral[..., kernel_size:, :-kernel_size]
            + integral[..., :-kernel_size, :-kernel_size]
        )

    def build_feature_stack(
        self,
        offsets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_map = self.x_map.expand(self.batch_size, -1, -1, -1)
        y_map = self.y_map.expand(self.batch_size, -1, -1, -1)
        ones_map = self.ones_map.expand(self.batch_size, -1, -1, -1)
        x2_map = self.x2_map.expand(self.batch_size, -1, -1, -1)
        xy_map = self.xy_map.expand(self.batch_size, -1, -1, -1)
        y2_map = self.y2_map.expand(self.batch_size, -1, -1, -1)

        dx_map = offsets[:, 0:1, :, :].to(dtype=self.dtype)
        dy_map = offsets[:, 1:2, :, :].to(dtype=self.dtype)

        feature_stack = torch.cat(
            (
                ones_map,
                x_map,
                y_map,
                x2_map,
                xy_map,
                y2_map,
                dx_map,
                dy_map,
                x_map * dx_map,
                y_map * dx_map,
                x_map * dy_map,
                y_map * dy_map,
            ),
            dim=1,
        )
        return feature_stack, x_map, y_map, dx_map, dy_map

    def build_local_equations(self, box_sums: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        sx2 = box_sums[:, self.X2_IDX]
        sxy = box_sums[:, self.XY_IDX]
        sy2 = box_sums[:, self.Y2_IDX]
        sx = box_sums[:, self.X_IDX]
        sy = box_sums[:, self.Y_IDX]
        s1 = box_sums[:, self.ONE_IDX]

        xtx = torch.stack(
            [
                torch.stack([sx2, sxy, sx], dim=-1),
                torch.stack([sxy, sy2, sy], dim=-1),
                torch.stack([sx, sy, s1], dim=-1),
            ],
            dim=-2,
        )

        rhs_x = torch.stack(
            [
                box_sums[:, self.X_DX_IDX],
                box_sums[:, self.Y_DX_IDX],
                box_sums[:, self.DX_IDX],
            ],
            dim=-1,
        )
        rhs_y = torch.stack(
            [
                box_sums[:, self.X_DY_IDX],
                box_sums[:, self.Y_DY_IDX],
                box_sums[:, self.DY_IDX],
            ],
            dim=-1,
        )
        rhs = torch.stack((rhs_x, rhs_y), dim=-1)
        return xtx, rhs

    def solve_coefficients(self, xtx: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        eps = 1e-2
        identity = torch.eye(3, device=self.device, dtype=xtx.dtype).view(1, 1, 1, 3, 3)
        regularized = xtx + eps * identity
        return torch.linalg.solve(regularized, rhs)

    def compute_predictions(
        self,
        theta: torch.Tensor,
        x_map: torch.Tensor,
        y_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        dx_pred = (
            theta[..., 0, 0].unsqueeze(1) * x_map
            + theta[..., 1, 0].unsqueeze(1) * y_map
            + theta[..., 2, 0].unsqueeze(1)
        )
        dy_pred = (
            theta[..., 0, 1].unsqueeze(1) * x_map
            + theta[..., 1, 1].unsqueeze(1) * y_map
            + theta[..., 2, 1].unsqueeze(1)
        )
        return dx_pred, dy_pred

    def compute_single_hypothesis_error(self, offsets: torch.Tensor, kernel_size: int) -> torch.Tensor:
        feature_stack, x_map, y_map, dx_map, dy_map = self.build_feature_stack(offsets)
        box_sums = self.box_sum(feature_stack, kernel_size)
        xtx, rhs = self.build_local_equations(box_sums)
        theta = self.solve_coefficients(xtx, rhs)

        dx_pred, dy_pred = self.compute_predictions(theta, x_map, y_map)
        residuals = torch.cat(
            (
                (dx_map - dx_pred).square(),
                (dy_map - dy_pred).square(),
            ),
            dim=1,
        )
        return self.box_sum(residuals, kernel_size).sum(dim=1)

    def compute_errors_default(self) -> torch.Tensor:
        mean_offsets = torch.stack(self.offset_fields, dim=0).mean(dim=0)
        feature_stack, x_map, y_map, _, _ = self.build_feature_stack(mean_offsets)
        errors = []

        for kernel_size in self.kernel_sizes:
            kernel_size = int(kernel_size)
            box_sums = self.box_sum(feature_stack, kernel_size)
            xtx, rhs = self.build_local_equations(box_sums)
            theta = self.solve_coefficients(xtx, rhs)

            dx_pred, dy_pred = self.compute_predictions(theta, x_map, y_map)
            error = None
            for offsets in self.offset_fields:
                dx_map = offsets[:, 0:1, :, :]
                dy_map = offsets[:, 1:2, :, :]
                residuals = torch.cat(
                    (
                        (dx_map - dx_pred).square(),
                        (dy_map - dy_pred).square(),
                    ),
                    dim=1,
                )
                residual_sums = self.box_sum(residuals, kernel_size).sum(dim=1)
                error = residual_sums if error is None else (error + residual_sums)
            error = error / float(len(self.offset_fields))
            errors.append(error)

        return torch.stack(errors, dim=1)

    def compute_errors_topk(self) -> torch.Tensor:
        errors = []

        for kernel_size in self.kernel_sizes:
            kernel_size = int(kernel_size)
            branch_errors = []
            for hypothesis_group in self.hypothesis_groups:
                branch_hypothesis_errors = [
                    self.compute_single_hypothesis_error(hypothesis_offsets, kernel_size)
                    for hypothesis_offsets in hypothesis_group
                ]
                if len(branch_hypothesis_errors) == 1:
                    branch_error = branch_hypothesis_errors[0]
                else:
                    branch_error = torch.amin(torch.stack(branch_hypothesis_errors, dim=1), dim=1)
                branch_errors.append(branch_error)
            errors.append(torch.stack(branch_errors, dim=0).mean(dim=0))

        return torch.stack(errors, dim=1)

    def compute_errors(self):
        if not self.has_topk_hypotheses:
            return self.compute_errors_default()
        return self.compute_errors_topk()
