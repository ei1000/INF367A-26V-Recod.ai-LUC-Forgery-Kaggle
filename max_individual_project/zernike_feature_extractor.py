import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from math import factorial

class ZernikeExtractor(nn.Module):
    def __init__(self, pq_list, kernel_size=64):
        super().__init__()
        self.pq_list = pq_list
        self.kernel_size = kernel_size

        # Precompute polar grid for small kernel
        self.rho, self.theta, self.mask = self.polar_grid(kernel_size)

        # Precompute kernels and register as buffers
        for idx, (p, q) in enumerate(self.pq_list):
            K = self.compute_kernel(p, q, self.rho, self.theta)
            K_real = torch.tensor(K.real, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
            K_imag = torch.tensor(K.imag, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            self.register_buffer(f"K_real_{idx}", K_real)
            self.register_buffer(f"K_imag_{idx}", K_imag)

    def polar_grid(self, size):
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, y)
        rho = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        mask = rho <= 1
        return rho, theta, mask

    def compute_radial_poly(self, p, q, rho):
        R = np.zeros_like(rho)
        m = (p - abs(q)) // 2
        for s in range(m + 1):
            c = (-1)**s * factorial(p - s) / (
                factorial(s) *
                factorial((p + abs(q)) // 2 - s) *
                factorial((p - abs(q)) // 2 - s)
            )
            R += c * rho**(p - 2*s)
        return R

    def compute_kernel(self, p, q, rho, theta):
        R = self.compute_radial_poly(p, q, rho)
        K = np.zeros_like(rho, dtype=np.complex64)
        K[rho <= 1] = R[rho <= 1] * np.exp(-1j * q * theta[rho <= 1])
        return K

    def forward(self, x):
        B, C, H, W = x.shape
        outputs = []

        for idx in range(len(self.pq_list)):
            K_real = getattr(self, f"K_real_{idx}")
            K_imag = getattr(self, f"K_imag_{idx}")

            # Use grouped convolution to avoid repeating kernels
            Kr = K_real.expand(C, 1, *K_real.shape[-2:])  # (C,1,Hk,Wk)
            Ki = K_imag.expand(C, 1, *K_imag.shape[-2:])

            conv_real = F.conv2d(x, Kr, padding=Kr.shape[-1]//2, groups=C)
            conv_imag = F.conv2d(x, Ki, padding=Ki.shape[-1]//2, groups=C)

            outputs.append(torch.sqrt(conv_real**2 + conv_imag**2))

        # Concatenate along channel dimension
        return torch.cat(outputs, dim=1)


class PyramidZernikeExtractor(nn.Module):
    def __init__(self, pq_list, kernel_size=64, rb=0.75, ru=1.5):
        super().__init__()
        self.zernike = ZernikeExtractor(pq_list, kernel_size)
        self.rb = rb
        self.ru = ru

    def forward(self, Io):
        H, W = Io.shape[-2:]
        Ib = F.interpolate(Io, scale_factor=self.rb, mode='bilinear')
        Iu = F.interpolate(Io, scale_factor=self.ru, mode='bilinear')

        Fb = self.zernike(Ib)
        Fo = self.zernike(Io)
        Fu = self.zernike(Iu)

        # Resize features back to original image size
        Fb = F.interpolate(Fb, size=(H, W), mode='bilinear')
        Fu = F.interpolate(Fu, size=(H, W), mode='bilinear')

        return Fb, Fo, Fu
