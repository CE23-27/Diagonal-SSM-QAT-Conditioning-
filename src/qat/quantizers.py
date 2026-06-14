"""
Quantization modules for QAT with straight-through estimator.

Implements deterministic and stochastic rounding for Phase 2a.

TODO Phase 2b: Add LSQ (Learned Step Size Quantization) with proper initialization
from data statistics and gradient rescaling per Esser et al. (2020).

TODO Phase 2b: Add per-channel/per-mode quantization for better dynamic range
utilization (currently per-tensor only).
"""

import torch
import torch.nn as nn


class Quantizer(nn.Module):
    """
    Quantizes a tensor to B bits with straight-through estimator.

    Forward: Q(x) = step * round(clamp(x / step))
    where step = scale / (2^(B-1) - 1)

    Backward: straight-through estimator (gradient passes unchanged)

    Modes:
        deterministic: standard round-to-nearest
        stochastic: unbiased stochastic rounding (for LOTION-style smoothing)

    Range:
        symmetric: [-scale, scale]

    Convention:
        For B-bit signed, use levels in [-(2^(B-1)-1), 2^(B-1)-1]
        e.g., 8-bit uses [-127, 127], 4-bit uses [-7, 7]
    """

    def __init__(self, bits, scale=1.0, mode='deterministic'):
        """
        Args:
            bits:  int, number of bits (2-8)
            scale: float OR tensor. If float, the quantization range is
                   [-scale, +scale] and applied per-tensor. If tensor, the
                   scale is broadcast against the input — e.g. shape
                   (1, N//2) gives per-mode quantization with one grid per
                   mode shared across heads. The shape must be
                   broadcast-compatible with the inputs to forward().
            mode:  str, 'deterministic' or 'stochastic'
        """
        super().__init__()
        assert bits >= 2 and bits <= 16, "bits must be in [2, 16]"
        assert mode in ['deterministic', 'stochastic'], \
            "mode must be 'deterministic' or 'stochastic'"

        self.bits = bits
        self.mode = mode

        # Quantization levels: symmetric signed with integer max
        # B-bit: [-(2^(B-1)-1), 2^(B-1)-1]
        self.max_val = 2 ** (bits - 1) - 1  # e.g., 127 for 8-bit, 7 for 4-bit

        # Scale buffer. Accept both scalars and pre-built tensors so
        # callers can pass per-mode scales directly. We `.detach().clone()`
        # tensors to avoid accidentally aliasing autograd state into a
        # buffer.
        if isinstance(scale, torch.Tensor):
            scale_tensor = scale.detach().clone().float()
        else:
            scale_tensor = torch.tensor(float(scale))
        self.register_buffer('scale', scale_tensor)

    def forward(self, x):
        """
        Quantize input tensor.

        Args:
            x: input tensor

        Returns:
            x_q: quantized tensor (same shape as input)
        """
        # Compute step size
        step = self.scale / self.max_val

        # Normalize to quantization grid
        x_scaled = x / step

        # Clamp BEFORE rounding to maintain unbiasedness in stochastic mode
        x_clamped = torch.clamp(x_scaled, -self.max_val, self.max_val)

        # Quantize
        if self.mode == 'deterministic':
            # Round to nearest
            x_q_scaled = torch.round(x_clamped)
        else:  # stochastic
            # Stochastic rounding: unbiased
            # round_up with probability (x - floor(x))
            x_floor = torch.floor(x_clamped)
            prob_up = x_clamped - x_floor
            rand = torch.rand_like(x_clamped)
            x_q_scaled = torch.where(rand < prob_up, x_floor + 1, x_floor)

        # Rescale
        x_q = x_q_scaled * step

        # Straight-through estimator: forward uses quantized value,
        # backward uses gradient of identity
        x_q = x + (x_q - x).detach()

        return x_q

    def extra_repr(self):
        if self.scale.numel() == 1:
            scale_str = f"{self.scale.item():.3f}"
        else:
            scale_str = (
                f"tensor(shape={tuple(self.scale.shape)}, "
                f"min={self.scale.min().item():.3g}, "
                f"max={self.scale.max().item():.3g})"
            )
        return f'bits={self.bits}, scale={scale_str}, mode={self.mode}'


class IdentityQuantizer(nn.Module):
    """
    Identity quantizer (no quantization) for fp32 baseline.
    """

    def forward(self, x):
        return x

    def extra_repr(self):
        return 'bits=32 (fp32)'


def make_quantizer(bits, scale=1.0, mode='deterministic'):
    """
    Factory function for creating quantizers.

    Args:
        bits: int or None. If None or 32, returns IdentityQuantizer
        scale: float, quantization range is [-scale, scale]
        mode: str, 'deterministic' or 'stochastic'

    Returns:
        Quantizer module
    """
    if bits is None or bits == 32:
        return IdentityQuantizer()
    else:
        return Quantizer(bits, scale=scale, mode=mode)
