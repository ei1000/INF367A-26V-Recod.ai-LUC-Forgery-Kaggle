import numpy as np
import torch
import torch.nn.functional as F

class MultiScaleDLF:
    def __init__(self, image: torch.Tensor, cnn_offsets: torch.Tensor, radiuses: np.ndarray | None = None): # zernike_offsets likely not used
        self.image = image
        self.cnn_offsets = cnn_offsets

        if radiuses is None:
            radiuses = np.array([7, 9, 11])
        
        self.radiuses = radiuses

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

        for k in self.radiuses:
            kernel = torch.ones((1, 1, k, k), device=self.image.device, dtype=torch.float32)
            pad = k // 2

            sums_k = {}
            for name, fmap in features.items():
                fmap = fmap.unsqueeze(0) # needs batch and channel: 1, 1, H, W
                sums_k[name] = F.conv2d(fmap, kernel, padding=pad)

            XtX, Xtdx, Xtdy = self.build_local_equations(sums_k)

            # Add a small epsilon to diagonal values to prevent singular matrix
            # + encourages stability and not too large weights
            XtX += 1e-4 * torch.eye(3, device=self.image.device).view(1, 1, 1, 3, 3)

            theta_x = torch.linalg.solve(XtX, Xtdx) # solve for X
            theta_y = torch.linalg.solve(XtX, Xtdy) # solve for Y

            # NOTE: continue by getting predictions: https://chatgpt.com/c/69c92af3-6df4-8388-8c53-3195c990c850


    def build_local_equations(sums):
        Sx2 = sums["x2"].squeeze(1)
        Sxy = sums["xy"].squeeze(1)
        Sy2 = sums["x"].squeeze(1)
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

    def _single_prediction(self, x):
        pass


    '''
    Build feature maps for image.

    '''
    def build_maps(self):
        _, H, W = self.image.shape

        x, y = torch.meshgrid(
            torch.arange(W, device=self.image.device, dtype=torch.float32),
            torch.arange(H, device=self.image.device, dtype=torch.float32),
            indexing='ij'
        )

        x = x.unsqueeze(0)
        y = y.unsqueeze(0)

        dx = self.cnn_offsets[0, :, :]
        dy = self.cnn_offsets[1, :, :]

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
            "x_dx": x*dy,
            "y_dy": y*dy
        }


        return features
    
    '''
    Transform ro x ro area into an N x 3 matrix (x, y, 1)
    '''
    def _transform(self, x):
        pass