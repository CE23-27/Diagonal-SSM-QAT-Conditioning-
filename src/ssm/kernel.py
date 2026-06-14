"""
Direct S4D kernel computation with quantization support.

Critical: Quantization happens at the parameter level (A_real, A_imag)
BEFORE complex reconstruction, as required by the Bauer-Fike analysis.
"""

import torch
import torch.nn as nn


def s4d_kernel(A_real, A_imag, B, C, log_dt, L, q_alpha=None, q_omega=None, conj_sym=True):
    """
    Direct S4D kernel computation, quantization-friendly.

    Computes K_l = sum_n C_n * (A_disc_n)^l * B_n for l = 0, ..., L-1
    using the direct Vandermonde approach (O(NL) complexity).

    Args:
        A_real:  (H, N//2) real tensor, log-parameterized alphas
        A_imag:  (H, N//2) real tensor, omegas (unconstrained)
        B:       (H, N//2) complex tensor, input projection
        C:       (H, N//2) complex tensor, output projection
        log_dt:  (H,) real tensor, log-parameterized step size
        L:       int, kernel length (sequence length)
        q_alpha: Quantizer module for alpha (or None for fp32)
        q_omega: Quantizer module for omega (or None for fp32)
        conj_sym: bool, if True assumes conjugate symmetry and applies factor of 2
                  to complex modes (for FouT, Skew-HiPPO). Set False for S4D-Real.

    Returns:
        K: (H, L) real tensor, the convolution kernel

    Implementation notes:
        - Quantization (if enabled) happens BEFORE complex reconstruction
        - For conj_sym=True: applies factor of 2 to modes with omega != 0
        - For conj_sym=False: no factor of 2 (e.g., S4D-Real)
        - Stable for L <= 4096 in complex64; for longer sequences consider log-domain
    """
    device = A_real.device
    dtype = A_real.dtype

    # Quantize at the parameter level, BEFORE complex reconstruction (Gotcha 1)
    if q_alpha is not None:
        a_real_q = q_alpha(A_real)
    else:
        a_real_q = A_real

    if q_omega is not None:
        a_imag_q = q_omega(A_imag)
    else:
        a_imag_q = A_imag

    # Reconstruct complex continuous-time A
    # A_n = -exp(a_real_q) + i * a_imag_q
    A = -torch.exp(a_real_q) + 1j * a_imag_q  # (H, N//2), complex

    # ZOH discretization: A_disc = exp(dt * A)
    dt = torch.exp(log_dt).unsqueeze(-1)  # (H, 1)
    A_disc = torch.exp(dt * A)  # (H, N//2), complex

    # Direct kernel via Vandermonde powers
    # K_l = sum_n C_n * (A_disc_n)^l * B_n
    lags = torch.arange(L, device=device, dtype=dtype)  # (L,)

    # Compute powers: (A_disc_n)^l for all n, l
    # Shape broadcasting: (H, N//2, 1) ^ (L,) -> (H, N//2, L)
    powers = A_disc.unsqueeze(-1) ** lags  # (H, N//2, L)

    # Compute mode factors (unified for both branches)
    if conj_sym:
        # Apply factor of 2 for conjugate pairs, but handle edge cases
        # Modes with omega ≈ 0 are real and should not be doubled
        is_real_mode = a_imag_q.abs() < 1e-7
        mode_factor = torch.where(is_real_mode, 1.0, 2.0)
    else:
        # S4D-Real or similar: no conjugate symmetry
        mode_factor = torch.ones_like(a_imag_q)

    # Element-wise product and sum over N//2 modes
    K_complex = (C.unsqueeze(-1) * powers * B.unsqueeze(-1) * mode_factor.unsqueeze(-1)).sum(dim=1)  # (H, L)
    K = K_complex.real

    return K


def fft_conv(u, K):
    """
    Convolve input u with kernel K using FFT.

    Args:
        u: (B, H, L) input sequence
        K: (H, L) convolution kernel

    Returns:
        y: (B, H, L) output sequence
    """
    # Pad to next power of 2 for efficient FFT
    L = u.shape[-1]
    fft_size = 2 ** (2 * L - 1).bit_length()

    # FFT convolution: conv(u, K) = ifft(fft(u) * fft(K))
    u_f = torch.fft.rfft(u, n=fft_size, dim=-1)  # (B, H, fft_size//2 + 1)
    K_f = torch.fft.rfft(K, n=fft_size, dim=-1)  # (H, fft_size//2 + 1)

    # Multiply in frequency domain
    y_f = u_f * K_f.unsqueeze(0)  # (B, H, fft_size//2 + 1)

    # IFFT and truncate
    y = torch.fft.irfft(y_f, n=fft_size, dim=-1)  # (B, H, fft_size)
    y = y[..., :L]  # (B, H, L), truncate to original length

    return y


def verify_kernel(K, late_peak_tol=1.2):
    """
    Sanity check for kernel validity.

    Checks for:
    1. NaN/Inf values (numerical instability).
    2. Late peaks meaningfully exceeding early peaks (envelope should decay).

    The late-peak check is intentionally a *soft* heuristic: for stable SSMs
    the kernel envelope must eventually decay, but with small Δ and short L
    the kernel barely moves over the window, so any minor numerical bump can
    register as a late peak under a strict greater-than. The tolerance
    parameter requires the late-quarter peak to exceed the early-quarter
    peak by at least `late_peak_tol`× before flagging.

    Args:
        K:               (H, L) kernel.
        late_peak_tol:   float ≥ 1. Late peak must exceed early peak by
                         this factor to count as a failure. Default 1.2.

    Returns:
        is_valid: bool
        message:  str, diagnostic message
    """
    if torch.isnan(K).any() or torch.isinf(K).any():
        return False, "nan/inf detected"

    # Envelope check: no late peak exceeds early peak by more than tol.
    L = K.shape[-1]
    early_peak = K.abs()[:, :L//4].amax(dim=-1)
    late_peak = K.abs()[:, 3*L//4:].amax(dim=-1)

    if (late_peak > late_peak_tol * early_peak).any():
        return False, (
            f"late peak exceeds early peak by >{late_peak_tol:.2f}x"
        )

    return True, "ok"


def smoke_test():
    """
    Smoke test for s4d_kernel correctness.

    Verifies:
    1. Output shape is correct
    2. No NaN/Inf values
    3. At l=0, K[:,0] = 2 * Re(sum_n C_n * B_n) for FouT (all complex modes)
    4. Kernel passes verification checks
    """
    from .inits import init_fout

    torch.manual_seed(0)
    H, N, L = 4, 64, 1024
    A_real, A_imag = init_fout(H, N)
    B = torch.randn(H, N // 2, dtype=torch.complex64)
    C = torch.randn(H, N // 2, dtype=torch.complex64)
    log_dt = torch.full((H,), -4.0)

    K = s4d_kernel(A_real, A_imag, B, C, log_dt, L, conj_sym=True)

    assert K.shape == (H, L), f"unexpected shape: {K.shape}"
    assert torch.isfinite(K).all(), "non-finite values in K"

    # At l=0, all powers = 1, so K[:,0] = 2 * Re(sum_n C_n * B_n)
    # (assumes FouT modes are all complex; adjust if any have omega=0)
    K0_expected = 2 * (C * B).sum(dim=1).real
    assert torch.allclose(K[:, 0], K0_expected, atol=1e-5), \
        f"K[0] mismatch: got {K[:, 0]}, expected {K0_expected}"

    valid, msg = verify_kernel(K)
    assert valid, msg

    print("smoke test passed")


if __name__ == "__main__":
    smoke_test()
