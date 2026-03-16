import hashlib
from pathlib import Path
from math import factorial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def default_pq_list(max_order: int = 5):
    """
    Generate valid (p, q) pairs up to max_order.
    Validity: |q| <= p and (p - |q|) is even.

    We use non-negative q only because |ZM(p, -q)| == |ZM(p, q)|,
    and the model consumes magnitudes.
    """
    pq = []
    for p in range(max_order + 1):
        for q in range(0, p + 1):
            if (p - q) % 2 == 0:
                pq.append((p, q))
    return pq


class ZernikeExtractor(nn.Module):
    def __init__(self, pq_list, kernel_size=13, cache_dir=None):
        super().__init__()
        self.pq_list = list(pq_list)
        self.kernel_size = kernel_size
        self.pad_size = kernel_size // 2

        # Cache directory for kernels
        if cache_dir is None:
            cache_dir = Path(__file__).resolve().parent / "zernike"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Precompute polar grid for kernel
        self.rho, self.theta, self.mask = self.polar_grid(kernel_size)

        # Load or compute kernels
        K_real, K_imag = self._load_or_compute_kernels()
        # Store as (N,1,H,W); repeat across channels at forward
        self.register_buffer("K_real", torch.tensor(K_real, dtype=torch.float32).unsqueeze(1))
        self.register_buffer("K_imag", torch.tensor(K_imag, dtype=torch.float32).unsqueeze(1))

    def polar_grid(self, size):
        x = np.linspace(-1, 1, size)
        y = np.linspace(-1, 1, size)
        X, Y = np.meshgrid(x, y)
        rho = np.sqrt(X**2 + Y**2)
        theta = np.arctan2(Y, X)
        mask = rho <= 1
        return rho, theta, mask

    def _cache_key(self):
        pq_str = ",".join([f"{p}:{q}" for p, q in self.pq_list])
        digest = hashlib.md5(pq_str.encode("utf-8")).hexdigest()[:10]
        return f"k{self.kernel_size}_n{len(self.pq_list)}_{digest}"

    def _kernel_cache_paths(self):
        key = self._cache_key()
        real_path = self.cache_dir / f"zm_{key}_real.npy"
        imag_path = self.cache_dir / f"zm_{key}_imag.npy"
        return real_path, imag_path

    def _load_or_compute_kernels(self):
        real_path, imag_path = self._kernel_cache_paths()
        if real_path.exists() and imag_path.exists():
            K_real = np.load(real_path)
            K_imag = np.load(imag_path)
            return K_real, K_imag

        K_real = np.zeros((len(self.pq_list), self.kernel_size, self.kernel_size), dtype=np.float32)
        K_imag = np.zeros((len(self.pq_list), self.kernel_size, self.kernel_size), dtype=np.float32)
        for idx, (p, q) in enumerate(self.pq_list):
            K = self.compute_kernel(p, q, self.rho, self.theta)
            K_real[idx] = K.real.astype(np.float32)
            K_imag[idx] = K.imag.astype(np.float32)

        np.save(real_path, K_real)
        np.save(imag_path, K_imag)
        return K_real, K_imag

    def compute_radial_poly(self, p, q, rho):
        if abs(q) > p or (p - abs(q)) % 2 != 0:
            return np.zeros_like(rho)
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
        _, C, _, _ = x.shape

        # Repeat kernels across channels (mix RGB)
        Kr = self.K_real.repeat(1, C, 1, 1)
        Ki = self.K_imag.repeat(1, C, 1, 1)

        conv_real = F.conv2d(x, Kr, padding=self.pad_size, groups=1)
        conv_imag = F.conv2d(x, Ki, padding=self.pad_size, groups=1)

        return torch.sqrt(conv_real**2 + conv_imag**2 + 1e-10)


class PyramidZernikeExtractor(nn.Module):
    def __init__(self, pq_list, kernel_size=13, rb=0.75, ru=1.5, cache_dir=None):
        super().__init__()
        self.zernike = ZernikeExtractor(pq_list, kernel_size, cache_dir=cache_dir)
        self.rb = rb
        self.ru = ru

    def forward(self, Io):
        H, W = Io.shape[-2:]
        Ib = F.interpolate(Io, scale_factor=self.rb, mode='bilinear', align_corners=True)
        Iu = F.interpolate(Io, scale_factor=self.ru, mode='bilinear', align_corners=True)

        Fb = self.zernike(Ib)
        Fo = self.zernike(Io)
        Fu = self.zernike(Iu)

        # Resize features back to original image size
        Fb = F.interpolate(Fb, size=(H, W), mode='bilinear', align_corners=True)
        Fu = F.interpolate(Fu, size=(H, W), mode='bilinear', align_corners=True)

        return Fb, Fo, Fu
