"""
S4D module: parameterization, initialization wiring, and forward pass.

Implements the diagonal SSM described in Gu et al. (2022)
"On the Parameterization and Initialization of Diagonal State Space Models"
(arXiv:2206.11893), with the parameter layout chosen for QAT legibility.

Design choices (see goals.md §5.1, §5.6 Gotcha 1):
    - Diagonal A is stored as TWO REAL tensors (A_real, A_imag) of shape (H, N//2).
      The complex eigenvalues A = -exp(A_real) + i * A_imag are reconstructed
      only inside the kernel, AFTER quantization, so that quantization noise
      enters in (alpha, omega)-space exactly as the Bauer-Fike analysis assumes.
    - B and C are stored as separate real/imaginary pairs (real tensors).
      This sidesteps PyTorch's incomplete complex autograd and lets us swap
      real-valued quantizers in for B/C in Phase 2b without touching the kernel.
    - Delta (timestep) is log-parameterized, per-channel.
    - D (skip) is a real per-channel scalar.
    - We require H == d_model (the standard S4D convention). This avoids the
      tile-and-truncate footgun that silently drops channels when d_model is
      not a multiple of H.
    - We do NOT call verify_kernel inside forward(). Diagnostics that print
      from inside a forward pass spam logs during eval and bypass downstream
      logging configuration. Use S4D.diagnose() once per epoch instead.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .kernel import s4d_kernel, fft_conv, verify_kernel
from .inits import (
    init_diag_lin,
    init_diag_inv,
    init_diag_legs,
    init_s4d_real,
    compute_kappa,
)


# Registry of available initializations, keyed by the names used in YAML configs
# and in goals.md. Values must match the (H, N) -> (A_real, A_imag, kappa) signature.
INIT_REGISTRY = {
    'fout':       init_diag_lin,    # FouT  == diag-lin
    'diag-lin':   init_diag_lin,
    'skew-hippo': init_diag_legs,   # Skew-HiPPO == diag-legs
    'diag-legs':  init_diag_legs,
    'diag-inv':   init_diag_inv,
    's4d-real':   init_s4d_real,
}


def _init_log_dt(H, mode='nyquist', d_state=64, dt_min=1e-3, dt_max=1e-1,
                 nyquist_jitter=0.1):
    """
    Initialize the per-channel log-discretization-step.

    Modes:
        'nyquist'     — tight cluster around Δ = 2/N. Centers the model in
                        the FouT-friendly band where the κ(V) gap predicted
                        by Phase-1 theory is largest. Per-channel jitter is
                        small so all channels start in the well-conditioned
                        regime; whether they stay there during training is
                        empirical.
        'log_uniform' — Gu et al.'s standard S4D init: log(dt) ~
                        Uniform[log(dt_min), log(dt_max)]. Use for
                        ablations / sanity baselines.

    Args:
        H:              number of channels.
        mode:           'nyquist' or 'log_uniform'.
        d_state:        state dimension N (only used by 'nyquist').
        dt_min, dt_max: bounds for 'log_uniform' (ignored otherwise).
        nyquist_jitter: stddev of the Gaussian jitter on log(dt) for
                        'nyquist'.

    Returns:
        log_dt: (H,) tensor.
    """
    if mode == 'nyquist':
        nyquist_dt = 2.0 / d_state
        return math.log(nyquist_dt) + nyquist_jitter * torch.randn(H)
    elif mode == 'log_uniform':
        return (
            torch.rand(H) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        )
    else:
        raise ValueError(
            f"unknown dt_init mode '{mode}'. Use 'nyquist' or 'log_uniform'."
        )


def _init_BC(H, N_half, dtype=torch.float32):
    """
    Default B, C initialization.

    Convention: B is fixed at all-ones in continuous time (a common S4D choice
    to avoid redundancy with C), C is iid normal. Both are stored as
    real/imaginary pairs so quantization is straightforward later.

    Returns:
        B_real, B_imag, C_real, C_imag, each (H, N_half) real tensors.
    """
    B_real = torch.ones(H, N_half, dtype=dtype)
    B_imag = torch.zeros(H, N_half, dtype=dtype)
    C_real = torch.randn(H, N_half, dtype=dtype)
    C_imag = torch.randn(H, N_half, dtype=dtype)
    return B_real, B_imag, C_real, C_imag


class S4D(nn.Module):
    """
    Diagonal SSM layer with QAT-friendly parameterization.

    Args:
        d_model:    int, channel / hidden dimension. Used as H (number of
                    independent SSM heads); we require H == d_model.
        d_state:    int, state dimension N. Must be even (conjugate-symmetric
                    parameterization stores N//2 modes).
        init:       str, key into INIT_REGISTRY (e.g. 'fout', 'skew-hippo').
        dt_min:     float, minimum discretization step at init.
        dt_max:     float, maximum discretization step at init.
        q_alpha:    optional Quantizer for A_real (alpha). None => fp32.
        q_omega:    optional Quantizer for A_imag (omega). None => fp32.
        conj_sym:   bool, whether to apply the conjugate-symmetry factor of 2.
                    True for FouT/Skew-HiPPO/Inv, False for S4D-Real.

    Inputs:
        u: (B, d_model, L) — batched 1-D sequences, one per channel/head.

    Outputs:
        y: (B, d_model, L) — same shape as input.

    Notes:
        - The actual quantization happens inside `s4d_kernel`, which receives
          q_alpha, q_omega and applies them to A_real, A_imag BEFORE the
          complex reconstruction. See goals.md §5.6 Gotcha 1.
        - B and C are reassembled into complex tensors only at the moment
          they are passed into the kernel; the learnable parameters remain
          real-valued.
        - kappa(V) is computed once at init and stored as a buffer for
          logging. It is NOT recomputed during training.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        init: str = 'fout',
        dt_init: str = 'nyquist',
        dt_min: float = 1e-3,
        dt_max: float = 1e-1,
        q_alpha: nn.Module = None,
        q_omega: nn.Module = None,
        conj_sym: bool = True,
        freeze_log_dt: bool = False,
    ):
        super().__init__()
        assert d_state % 2 == 0, "d_state (N) must be even"
        if init not in INIT_REGISTRY:
            raise ValueError(
                f"unknown init '{init}'. Available: {list(INIT_REGISTRY.keys())}"
            )

        self.d_model = d_model
        self.d_state = d_state
        self.init_name = init
        self.conj_sym = conj_sym

        # Phase 2a convention: one independent SSM per channel. No tiling, no
        # truncation. Keeping this strict avoids the silent-channel-drop bug
        # that comes from `K.repeat(d_model // H, 1)[:d_model]`-style fixes.
        self.H = d_model
        N_half = d_state // 2

        # --- Diagonal A: two real tensors, log-parameterized real part --------
        A_real, A_imag, kappa = INIT_REGISTRY[init](self.H, d_state)
        self.A_real = nn.Parameter(A_real)        # (H, N//2)
        self.A_imag = nn.Parameter(A_imag)        # (H, N//2)

        # Log kappa(V) at init time. This is the quantity that drives the
        # bit-width prediction; we want it visible in logs/checkpoints.
        self.register_buffer('kappa_init', torch.tensor(float(kappa)))

        # --- B, C: stored as real/imaginary pairs ----------------------------
        B_real, B_imag, C_real, C_imag = _init_BC(self.H, N_half)
        self.B_real = nn.Parameter(B_real)        # (H, N//2)
        self.B_imag = nn.Parameter(B_imag)        # (H, N//2)
        self.C_real = nn.Parameter(C_real)        # (H, N//2)
        self.C_imag = nn.Parameter(C_imag)        # (H, N//2)

        # --- Delta (timestep): log-parameterized per-channel -----------------
        # Default 'nyquist' centers all channels at Δ = 2/N where the FouT
        # κ(V) advantage is largest. Use 'log_uniform' for the standard
        # Gu et al. init as an ablation baseline.
        self.dt_init = dt_init
        self.log_dt = nn.Parameter(_init_log_dt(
            self.H, mode=dt_init, d_state=d_state,
            dt_min=dt_min, dt_max=dt_max,
        ))  # (H,)
        # Frozen-dt ablation (Newgoals.md): if requested, freeze log_dt at
        # init. AdamW silently skips params with requires_grad=False, and
        # make_param_groups still places log_dt in the SSM group — no other
        # code paths need to change. The freeze is "set once at construction"
        # so loading a checkpoint of a frozen model keeps log_dt frozen iff
        # the constructor was called with freeze_log_dt=True.
        self.freeze_log_dt = freeze_log_dt
        if freeze_log_dt:
            self.log_dt.requires_grad_(False)

        # --- D (skip): real per-channel --------------------------------------
        self.D = nn.Parameter(torch.ones(self.H))  # (H,)

        # --- Quantizers (optional). Stored on the module so .to(device) /
        # .cuda() propagate any registered buffers. --------------------------
        self.q_alpha = q_alpha if q_alpha is not None else nn.Identity()
        self.q_omega = q_omega if q_omega is not None else nn.Identity()

        # One-shot guard for diagnostic warnings; see diagnose().
        self._verify_warned = False

    @property
    def kappa(self) -> float:
        """Initial-time condition number of the eigenvector matrix."""
        return float(self.kappa_init.item())

    def _make_BC(self):
        """Reassemble complex B, C tensors from stored real/imag pairs."""
        B = torch.complex(self.B_real, self.B_imag)
        C = torch.complex(self.C_real, self.C_imag)
        return B, C

    def kernel(self, L: int) -> torch.Tensor:
        """
        Compute the convolution kernel of length L.

        Returns:
            K: (H, L) real tensor.
        """
        B, C = self._make_BC()
        # Pass quantizers through; nn.Identity() is treated as fp32 (None).
        q_alpha = self.q_alpha if not isinstance(self.q_alpha, nn.Identity) else None
        q_omega = self.q_omega if not isinstance(self.q_omega, nn.Identity) else None

        K = s4d_kernel(
            self.A_real, self.A_imag,
            B, C,
            self.log_dt,
            L,
            q_alpha=q_alpha,
            q_omega=q_omega,
            conj_sym=self.conj_sym,
        )
        return K

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        """
        Args:
            u: (B, d_model, L) input.

        Returns:
            y: (B, d_model, L) output.
        """
        if u.dim() != 3:
            raise ValueError(f"S4D expects (B, H, L); got shape {tuple(u.shape)}")
        if u.shape[1] != self.d_model:
            raise ValueError(
                f"S4D expects channel dim {self.d_model}; got {u.shape[1]}"
            )

        L = u.shape[-1]
        K = self.kernel(L)                                # (H, L)
        y = fft_conv(u, K)                                # (B, H, L)
        y = y + self.D.view(1, -1, 1) * u                 # skip
        return y

    @torch.no_grad()
    def diagnose(self, L: int) -> dict:
        """
        Optional diagnostic. Intended to be called explicitly from a training
        or eval harness (e.g. once per epoch), NOT from forward(): the
        per-channel κ measurement is non-trivial work and we don't want it
        on the hot path.

        Computes:
            - kernel of length L and runs verify_kernel (soft sanity check).
            - per-channel runtime κ at each channel's actual exp(log_dt).
              kappa_init was computed at the canonical Δ = 2/N; the runtime
              κ is what governs actual kernel sensitivity if log_dt has
              drifted during training.

        The kernel-verify warning is gated by a one-shot flag so repeated
        calls during a sweep do not spam logs.

        Returns:
            dict with keys:
                valid:         bool
                msg:           str (verify_kernel diagnostic)
                kappa_init:    float (frozen init-time value at Δ = 2/N)
                kappa_median:  float (median over channels at runtime Δ)
                kappa_max:     float (max over channels at runtime Δ)
        """
        import logging
        import numpy as np

        K = self.kernel(L)
        valid, msg = verify_kernel(K)
        if not valid and not self._verify_warned:
            logging.warning(
                "S4D[init=%s, kappa_init=%.3g] kernel verification failed: %s",
                self.init_name, self.kappa, msg,
            )
            self._verify_warned = True

        # Per-channel runtime κ at each channel's own dt.
        kappas_runtime = []
        for h in range(self.H):
            dt_h = float(torch.exp(self.log_dt[h]))
            k_h = compute_kappa(
                self.A_real[h:h+1].detach().cpu(),
                self.A_imag[h:h+1].detach().cpu(),
                dt=dt_h,
            )
            kappas_runtime.append(k_h)

        return {
            'valid': valid,
            'msg': msg,
            'kappa_init': self.kappa,
            'kappa_median': float(np.median(kappas_runtime)),
            'kappa_max': float(np.max(kappas_runtime)),
        }

    def extra_repr(self) -> str:
        return (
            f"d_model={self.d_model}, d_state={self.d_state}, "
            f"init={self.init_name}, conj_sym={self.conj_sym}, "
            f"kappa_init={self.kappa:.3g}"
        )


class S4Block(nn.Module):
    """
    Canonical S4 residual block (Gu et al. official S4 repo style).

    Architecture (channels-last I/O):
        residual = x
        x = LayerNorm(x)
        x = x.transpose(1, 2)        # (B, L, d_model) -> (B, d_model, L)
        x = S4D(x)
        x = x.transpose(1, 2)        # (B, d_model, L) -> (B, L, d_model)
        x = GELU(Linear(x))          # single post-SSM activation
        x = Dropout(x)
        return residual + x

    No GLU, no input projection. This matches the published LRA configuration.

    Args:
        d_model:   channel dimension.
        d_state:   SSM state dimension N.
        init:      initialization key into INIT_REGISTRY.
        dropout:   residual-path dropout.
        q_alpha:   optional Quantizer for alpha.
        q_omega:   optional Quantizer for omega.
        conj_sym:  see S4D.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        init: str = 'fout',
        dropout: float = 0.0,
        q_alpha: nn.Module = None,
        q_omega: nn.Module = None,
        conj_sym: bool = True,
        freeze_log_dt: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.norm = nn.LayerNorm(d_model)

        self.s4d = S4D(
            d_model=d_model,
            d_state=d_state,
            init=init,
            q_alpha=q_alpha,
            q_omega=q_omega,
            conj_sym=conj_sym,
            freeze_log_dt=freeze_log_dt,
        )

        self.linear = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)

        Returns:
            y: (B, L, d_model)
        """
        residual = x
        x = self.norm(x)
        x = x.transpose(1, 2)                # (B, d_model, L)
        x = self.s4d(x)
        x = x.transpose(1, 2)                # (B, L, d_model)
        x = F.gelu(self.linear(x))           # single post-SSM activation
        x = self.dropout(x)
        return residual + x


def smoke_test():
    """
    Smoke test for S4D and S4Block forward passes.

    Verifies:
        1. fp32 forward pass shape and finiteness for FouT init.
        2. fp32 forward pass shape and finiteness for Skew-HiPPO init.
        3. Backward pass produces finite gradients on A_real, A_imag.
        4. Quantized forward pass (4-bit) is finite.
        5. S4Block end-to-end forward.
        6. diagnose() runs without crashing.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from src.qat.quantizers import make_quantizer

    torch.manual_seed(0)
    B, L, d_model, d_state = 2, 64, 16, 32

    # --- 1. FouT ----------------------------------------------------------
    layer = S4D(d_model=d_model, d_state=d_state, init='fout')
    u = torch.randn(B, d_model, L, requires_grad=True)
    y = layer(u)
    assert y.shape == (B, d_model, L), f"bad shape: {y.shape}"
    assert torch.isfinite(y).all(), "non-finite output (FouT)"
    print(f"FouT       fp32 forward ok, kappa(V) = {layer.kappa:.3g}")

    # --- 2. Skew-HiPPO ----------------------------------------------------
    layer_sh = S4D(d_model=d_model, d_state=d_state, init='skew-hippo')
    y_sh = layer_sh(u)
    assert torch.isfinite(y_sh).all(), "non-finite output (Skew-HiPPO)"
    print(f"Skew-HiPPO fp32 forward ok, kappa(V) = {layer_sh.kappa:.3g}")

    # --- 3. Backward ------------------------------------------------------
    loss = y.pow(2).mean()
    loss.backward()
    assert layer.A_real.grad is not None and torch.isfinite(layer.A_real.grad).all()
    assert layer.A_imag.grad is not None and torch.isfinite(layer.A_imag.grad).all()
    print("backward ok, gradients finite on A_real / A_imag")

    # --- 4. Quantized forward --------------------------------------------
    q_a = make_quantizer(bits=4, scale=4.0, mode='deterministic')
    q_o = make_quantizer(bits=4, scale=math.pi * d_state, mode='deterministic')
    layer_q = S4D(d_model=d_model, d_state=d_state, init='fout',
                  q_alpha=q_a, q_omega=q_o)
    y_q = layer_q(u.detach())
    assert torch.isfinite(y_q).all(), "non-finite output (quantized)"
    print("4-bit quantized forward ok")

    # --- 5. S4Block end-to-end -------------------------------------------
    block = S4Block(d_model=d_model, d_state=d_state, init='fout', dropout=0.1)
    x = torch.randn(B, L, d_model)
    y_block = block(x)
    assert y_block.shape == x.shape, f"S4Block shape: {y_block.shape}"
    assert torch.isfinite(y_block).all(), "non-finite output (S4Block)"
    print("S4Block forward ok")

    # --- 6. diagnose() ---------------------------------------------------
    # We only assert that diagnose runs and returns a dict with the expected
    # fields; the late-peak heuristic in verify_kernel is dt/L-sensitive
    # and is meant as a soft signal, not a gate. Real kernel-quality checks
    # belong in the eval harness with production-scale (dt, L).
    info = layer.diagnose(L)
    for k in ('valid', 'msg', 'kappa_init', 'kappa_median', 'kappa_max'):
        assert k in info, f"diagnose() missing key {k}"
    print(
        f"diagnose() -> valid={info['valid']}, "
        f"kappa_init={info['kappa_init']:.2e}, "
        f"kappa_median={info['kappa_median']:.2e}, "
        f"kappa_max={info['kappa_max']:.2e}"
    )

    print("All S4D / S4Block smoke tests passed!")


if __name__ == "__main__":
    smoke_test()
