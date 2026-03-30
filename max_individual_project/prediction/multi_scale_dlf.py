import numpy as np
import torch
import torch.nn.functional as F

class MultiScaleDLF:
    def __init__(self, image: torch.Tensor, cnn_offsets: torch.Tensor, kernel_sizes: np.ndarray | None = None): # zernike_offsets likely not used
        self.image = self._as_batched_image(image)
        self.cnn_offsets = self._as_batched_offsets(cnn_offsets)

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

    def _as_batched_image(self, image: torch.Tensor) -> torch.Tensor:
        if image.dim() == 3:
            return image.unsqueeze(0)
        if image.dim() == 4:
            return image
        raise ValueError(f"Expected image shape [C,H,W] or [B,C,H,W], got {tuple(image.shape)}")

    def _as_batched_offsets(self, offsets: torch.Tensor) -> torch.Tensor:
        if offsets.dim() == 3:
            offsets = offsets.unsqueeze(0)
        if offsets.dim() != 4 or offsets.shape[1] != 2:
            raise ValueError(f"Expected offsets shape [2,H,W] or [B,2,H,W], got {tuple(offsets.shape)}")
        return offsets

    '''
    Perform multi-scale dense linear fitting based on P + delta_P ≈ PA, B = A - I

    Solves the linear regression problem using closed form solution:
    B* = (P'P)**-1 P' * delta_P

    Solves for both x and y offsets and combines errors.

    Multi scale as it uses all radiuses in the self.radius array to account for different scalings.

    @return errors for each scale
    '''
    def compute_errors(self):
        # Build feature map once globally to save computation
        features = self.build_maps()
        errors = []

        for k in self.kernel_sizes:
            k = int(k)
            kernel = torch.ones((1, 1, k, k), device=self.device, dtype=self.dtype)
            pad = k // 2

            sums_k = {}
            for name, fmap in features.items():
                sums_k[name] = F.conv2d(fmap, kernel, padding=pad)

            XtX, Xtdx, Xtdy = self.build_local_equations(sums_k)

            # Add a small epsilon to diagonal values to prevent singular matrix
            # + encourages stability and not too large weights
            eps = 1e-2
            I = torch.eye(3, device=self.device, dtype=XtX.dtype).view(1, 1, 1, 3, 3)
            XtX = XtX + eps * I

            # solve for coefficients
            theta_x = torch.matmul(torch.linalg.pinv(XtX), Xtdx)
            theta_y = torch.matmul(torch.linalg.pinv(XtX), Xtdy)
            
            # extract coefficients
            ax = theta_x[..., 0, 0].unsqueeze(1)
            bx = theta_x[..., 1, 0].unsqueeze(1)
            cx = theta_x[..., 2, 0].unsqueeze(1)
            
            ay = theta_y[..., 0, 0].unsqueeze(1)
            by = theta_y[..., 1, 0].unsqueeze(1)
            cy = theta_y[..., 2, 0].unsqueeze(1)
            
            # generate predictions to use in error calcs
            dx_pred = ax * features["x"] + bx * features["y"] + cx
            dy_pred = ay * features["x"] + by * features["y"] + cy

            residual_x = (features["dx"] - dx_pred)**2
            residual_y = (features["dy"] - dy_pred)**2

            # sum over window
            kernel = torch.ones((1,1,k,k), device=self.device, dtype=self.dtype)

            error_x = F.conv2d(residual_x, kernel, padding=k//2)
            error_y = F.conv2d(residual_y, kernel, padding=k//2)
                        
            error = error_x + error_y
            errors.append(error.squeeze(1))
        
        return torch.stack(errors, dim=1)

    def build_local_equations(self, sums):
        Sx2 = sums["x2"].squeeze(1)
        Sxy = sums["xy"].squeeze(1)
        Sy2 = sums["y2"].squeeze(1)
        Sx = sums["x"].squeeze(1)
        Sy = sums["y"].squeeze(1)
        S1 = sums["1"].squeeze(1)

        XtX = torch.stack([
            torch.stack([Sx2, Sxy, Sx], dim=-1),
            torch.stack([Sxy, Sy2, Sy], dim=-1),
            torch.stack([Sx, Sy, S1], dim=-1)
        ], dim=-2) # shape: B, H, W, 3, 3

        Xtdx = torch.stack([
            sums["x_dx"].squeeze(1),
            sums["y_dx"].squeeze(1),
            sums["dx"].squeeze(1)
        ], dim=-1).unsqueeze(-1)

        Xtdy = torch.stack([
            sums["x_dy"].squeeze(1),
            sums["y_dy"].squeeze(1),
            sums["dy"].squeeze(1)
        ], dim=-1).unsqueeze(-1)

        return XtX, Xtdx, Xtdy
    

    '''
    Build feature maps for image.

    '''
    def build_maps(self):
        B, _, H, W = self.image.shape

        y, x = torch.meshgrid(
            torch.arange(H, device=self.device, dtype=self.dtype),
            torch.arange(W, device=self.device, dtype=self.dtype),
            indexing='ij'
        )

        x = x.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)
        y = y.unsqueeze(0).unsqueeze(0).expand(B, -1, -1, -1)

        dx = self.cnn_offsets[:, 0:1, :, :].to(dtype=self.dtype)
        dy = self.cnn_offsets[:, 1:2, :, :].to(dtype=self.dtype)

        ones = torch.ones_like(x)

        # All features for a single pixel needed for the DLF procedure
        features = {
            "1": ones,
            "x": x,
            "y": y,
            "x2": x*x,
            "xy": x*y,
            "y2": y*y,
            "dx": dx,
            "dy": dy,
            "x_dx": x*dx,
            "y_dx": y*dx,
            "x_dy": x*dy,
            "y_dy": y*dy
        }


        return features
