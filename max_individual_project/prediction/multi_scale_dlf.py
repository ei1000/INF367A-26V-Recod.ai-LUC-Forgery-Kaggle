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

    def __init__(self, image: torch.Tensor, cnn_offsets: torch.Tensor, kernel_sizes: np.ndarray | None = None):
        self.image = self.as_batched_image(image)
        self.cnn_offsets = self.as_batched_offsets(cnn_offsets)

        if self.image.shape[0] != self.cnn_offsets.shape[0]:
            raise ValueError(
                f"Batch size mismatch between image {self.image.shape} and offsets {self.cnn_offsets.shape}"
            )
        if self.image.shape[-2:] != self.cnn_offsets.shape[-2:]:
            raise ValueError(
                f"Spatial size mismatch between image {self.image.shape} and offsets {self.cnn_offsets.shape}"
            )

        if kernel_sizes is None:
            kernel_sizes = np.array([7, 9, 11])

        self.kernel_sizes = kernel_sizes
        self.device = self.cnn_offsets.device
        self.dtype = self.cnn_offsets.dtype if self.cnn_offsets.is_floating_point() else torch.float32
        self.batch_size = self.image.shape[0]
        self.height = self.image.shape[-2]
        self.width = self.image.shape[-1]
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

    def as_batched_offsets(self, offsets: torch.Tensor) -> torch.Tensor:
        if offsets.dim() == 3:
            offsets = offsets.unsqueeze(0)
        if offsets.dim() != 4 or offsets.shape[1] != 2:
            raise ValueError(f"Expected offsets shape [2,H,W] or [B,2,H,W], got {tuple(offsets.shape)}")
        return offsets

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

        y, x = torch.meshgrid(
            torch.arange(height, device=device, dtype=dtype),
            torch.arange(width, device=device, dtype=dtype),
            indexing="ij",
        )
        x = x.unsqueeze(0).unsqueeze(0)
        y = y.unsqueeze(0).unsqueeze(0)
        ones = torch.ones_like(x)
        static_maps = (ones, x, y, x * x, x * y, y * y)
        cls.STATIC_MAP_CACHE[key] = static_maps
        return static_maps

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

    def build_feature_stack(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        x_map = self.x_map.expand(self.batch_size, -1, -1, -1)
        y_map = self.y_map.expand(self.batch_size, -1, -1, -1)
        ones_map = self.ones_map.expand(self.batch_size, -1, -1, -1)
        x2_map = self.x2_map.expand(self.batch_size, -1, -1, -1)
        xy_map = self.xy_map.expand(self.batch_size, -1, -1, -1)
        y2_map = self.y2_map.expand(self.batch_size, -1, -1, -1)

        dx_map = self.cnn_offsets[:, 0:1, :, :].to(dtype=self.dtype)
        dy_map = self.cnn_offsets[:, 1:2, :, :].to(dtype=self.dtype)

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
        Sx2 = box_sums[:, self.X2_IDX]
        Sxy = box_sums[:, self.XY_IDX]
        Sy2 = box_sums[:, self.Y2_IDX]
        Sx = box_sums[:, self.X_IDX]
        Sy = box_sums[:, self.Y_IDX]
        S1 = box_sums[:, self.ONE_IDX]

        XtX = torch.stack(
            [
                torch.stack([Sx2, Sxy, Sx], dim=-1),
                torch.stack([Sxy, Sy2, Sy], dim=-1),
                torch.stack([Sx, Sy, S1], dim=-1),
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
        return XtX, rhs

    def solve_coefficients(self, XtX: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
        eps = 1e-2
        identity = torch.eye(3, device=self.device, dtype=XtX.dtype).view(1, 1, 1, 3, 3)
        regularized = XtX + eps * identity
        return torch.matmul(torch.linalg.pinv(regularized), rhs)

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

    def compute_errors(self):
        feature_stack, x_map, y_map, dx_map, dy_map = self.build_feature_stack()
        errors = []

        for kernel_size in self.kernel_sizes:
            kernel_size = int(kernel_size)
            box_sums = self.box_sum(feature_stack, kernel_size)
            XtX, rhs = self.build_local_equations(box_sums)
            theta = self.solve_coefficients(XtX, rhs)

            dx_pred, dy_pred = self.compute_predictions(theta, x_map, y_map)
            residuals = torch.cat(
                (
                    (dx_map - dx_pred).square(),
                    (dy_map - dy_pred).square(),
                ),
                dim=1,
            )
            residual_sums = self.box_sum(residuals, kernel_size)
            error = residual_sums.sum(dim=1)
            errors.append(error)

        return torch.stack(errors, dim=1)
