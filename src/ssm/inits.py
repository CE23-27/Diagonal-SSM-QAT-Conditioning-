"""
S4D initialization schemes with condition number measurement.

All initializations return (A_real, A_imag) as real tensors, plus the
numerically computed condition number κ(V) of the eigenvector matrix.

Formulas based on Gu et al. (2022) "On the Parameterization and
Initialization of Diagonal State Space Models" arXiv:2206.11893.
"""

import torch
import numpy as np
from scipy import linalg
import math


def compute_kappa(A_real, A_imag, dt=None):
    """
    Compute condition number κ(V) of the Vandermonde of discrete-time
    eigenvalues — the quantity that actually drives the QAT bit-width
    formula derived in Phase 1 (goals.md §2).

    The Bauer-Fike penalty applies to perturbations of A_disc_n = exp(Δ λ_n),
    NOT the normalized continuous-time eigenvalues. The discrete-time
    eigenvalues live near the unit circle (|A_disc| = exp(-α Δ) ≈ 1 for
    small Δ) and their phases are Δ ω_n. For FouT at Δ ≈ 2/N these phases
    span [0, 2π) uniformly → DFT-like Vandermonde → κ = O(1). For LegS the
    phase distribution is highly non-uniform → κ very large. The FouT
    advantage is a narrow Nyquist phenomenon — at very small Δ every init
    is ill-conditioned, at very large Δ aliasing destroys conditioning for
    everyone.

    Args:
        A_real: (H, N//2) log-parameterized alphas (α_n = exp(A_real_n)).
        A_imag: (H, N//2) omegas (continuous-time imaginary parts).
        dt:     float or None. Reference discretization step. If None,
                uses Δ = 2/N (places the highest FouT mode at Nyquist).

    Returns:
        kappa: float, max condition number across heads.

    Notes:
        - For real modes (|ω| ≈ 0) we do NOT add a conjugate copy; otherwise
          the Vandermonde would have a duplicate column and κ → ∞ trivially.
        - The returned κ depends on dt. Callers can supply per-channel dt
          for runtime-κ measurement; see S4D.diagnose().
    """
    H, N_half = A_real.shape
    if dt is None:
        # Nyquist-ish default: Δ = 2/N where N = 2 * N_half.
        dt = 2.0 / (2 * N_half)

    kappas = []

    for h in range(H):
        alpha = torch.exp(A_real[h]).numpy()  # (N//2,)
        omega = A_imag[h].numpy()             # (N//2,)
        eigs = -alpha + 1j * omega            # (N//2,)

        # Build the FULL spectrum without duplicating real eigenvalues
        # (a duplicate column makes V trivially singular).
        real_mask = np.abs(omega) < 1e-7
        complex_eigs = eigs[~real_mask]
        real_eigs = eigs[real_mask]
        eigs_full = np.concatenate(
            [real_eigs, complex_eigs, np.conj(complex_eigs)]
        )

        # Discrete-time eigenvalues. We deliberately do NOT renormalize to
        # the unit circle — that step erases the dt-dependent phase
        # structure which is exactly what distinguishes FouT from LegS.
        eigs_disc = np.exp(dt * eigs_full)

        # Vandermonde V[i, j] = A_disc_j^i over a window of length N.
        N = len(eigs_full)
        powers = np.arange(N)[:, None]
        V = eigs_disc[None, :] ** powers      # (N, N)

        s = linalg.svdvals(V)
        kappa_h = s[0] / (s[-1] + 1e-12)
        kappas.append(kappa_h)

    return max(kappas)


def init_diag_lin(H, N, dtype=torch.float32):
    """
    S4D-Lin initialization: linear spacing of frequencies.

    Eigenvalues: A_n = -0.5 + i*pi*n for n = 0, ..., N//2 - 1

    This gives κ(V) = O(1) — the eigenvector matrix is essentially
    the DFT matrix, which is perfectly conditioned.

    Also known as "FouT" in some literature.

    Args:
        H: number of heads
        N: state dimension (even)
        dtype: torch dtype

    Returns:
        A_real: (H, N//2) log-parameterized alphas
        A_imag: (H, N//2) omegas
        kappa: float, condition number
    """
    assert N % 2 == 0, "N must be even"
    N_half = N // 2

    # Real part: constant 0.5
    alpha = 0.5 * torch.ones(H, N_half, dtype=dtype)

    # Imaginary part: pi * n for n = 0, 1, ..., N//2 - 1.
    # `.contiguous()` defensively breaks any stride-0 alias produced by
    # expand(): nn.Parameter wrapping a stride-0 view causes Adam's
    # in-place addcdiv_ to write through aliased storage and crash.
    # (The scalar multiplication here happens to materialize a fresh
    # tensor anyway, but we don't want future refactors to silently
    # reintroduce the alias by reordering ops.)
    n = torch.arange(N_half, dtype=dtype)
    omega = (math.pi * n.unsqueeze(0).expand(H, -1)).contiguous()

    # Log-parameterize alpha: A_real = log(alpha)
    A_real = torch.log(alpha)
    A_imag = omega

    assert A_real.is_contiguous() and A_imag.is_contiguous(), \
        "init_diag_lin: A_real and A_imag must be contiguous (see comment above)"

    kappa = compute_kappa(A_real, A_imag)

    return A_real, A_imag, kappa


def init_diag_inv(H, N, dtype=torch.float32):
    """
    S4D-Inv initialization: inverse spacing based on HiPPO asymptotics.

    Eigenvalues: A_n = -0.5 + i * imag_n
    where imag_n = (1/pi) * N * (N/(1+2n) - 1)

    This has higher κ(V) than diag-lin due to non-uniform spacing.

    Args:
        H: number of heads
        N: state dimension (even)
        dtype: torch dtype

    Returns:
        A_real: (H, N//2) log-parameterized alphas
        A_imag: (H, N//2) omegas
        kappa: float, condition number
    """
    assert N % 2 == 0, "N must be even"
    N_half = N // 2

    # Real part: constant 0.5
    alpha = 0.5 * torch.ones(H, N_half, dtype=dtype)

    # Imaginary part: (1/pi) * N * (N/(1+2n) - 1).
    # `.contiguous()` after expand: nn.Parameter wrapping a stride-0 view
    # crashes Adam — see comment in init_diag_lin.
    n = torch.arange(N_half, dtype=dtype)
    omega = (1.0 / math.pi) * N * (N / (1 + 2*n) - 1)
    omega = omega.unsqueeze(0).expand(H, -1).contiguous()

    # Log-parameterize alpha
    A_real = torch.log(alpha)
    A_imag = omega

    assert A_real.is_contiguous() and A_imag.is_contiguous(), \
        "init_diag_inv: A_real and A_imag must be contiguous"

    kappa = compute_kappa(A_real, A_imag)

    return A_real, A_imag, kappa


def init_diag_legs(H, N, dtype=torch.float32):
    """
    S4D-LegS: HiPPO-LegS eigenvalue initialization.

    Extracts eigenvalues from the normal part of the HiPPO-LegS matrix.
    This is the most principled initialization from function approximation
    theory, but has high κ(V).

    Also known as "Skew-HiPPO" in some literature.

    Real part is fixed at 0.5 (same as diag-lin/diag-inv).
    Imaginary parts come from HiPPO-LegS normal matrix eigenvalues.

    Args:
        H: number of heads
        N: state dimension (even)
        dtype: torch dtype

    Returns:
        A_real: (H, N//2) log-parameterized alphas
        A_imag: (H, N//2) omegas
        kappa: float, condition number
    """
    assert N % 2 == 0, "N must be even"
    N_half = N // 2

    # HiPPO-LegS lower-triangular A (Gu et al. S4 paper, eq. 4)
    # A[n,k] = -sqrt(2n+1) * sqrt(2k+1) for n > k
    # A[n,n] = -(n+1)
    # A[n,k] = 0 for n < k
    n = torch.arange(N, dtype=dtype)
    sqrt_2n1 = torch.sqrt(2 * n + 1)
    A = -sqrt_2n1.unsqueeze(1) * sqrt_2n1.unsqueeze(0)  # outer product
    A = torch.tril(A, diagonal=-1)  # keep strict lower triangle only
    A = A - torch.diag(n + 1)  # set diagonal to -(n+1)

    # The normal part: A_normal = A + (P P^T)/2 where P = sqrt(2n+1)
    # This restores the upper triangle to make the matrix normal
    P = sqrt_2n1
    A_normal = A + 0.5 * P.unsqueeze(1) * P.unsqueeze(0)

    # Now eigenvalues are complex conjugate pairs
    eigs = torch.linalg.eigvals(A_normal)

    # Sort by imaginary part magnitude
    idx = torch.argsort(eigs.imag.abs())
    eigs = eigs[idx]

    # Take first N//2 with positive imaginary parts (conjugate pairs)
    # Filter to positive imaginary eigenvalues
    pos_imag_mask = eigs.imag > 1e-7
    eigs_pos = eigs[pos_imag_mask][:N_half]

    # Real part: fixed at 0.5 (not extracted from eigenvalues)
    alpha = 0.5 * torch.ones(H, N_half, dtype=dtype)
    A_real = torch.log(alpha)

    # Imaginary part: from HiPPO eigenvalues.
    # `.contiguous()` is REQUIRED here (not merely defensive): without it,
    # `expand(H, -1)` returns a view with stride 0 on the head axis. Every
    # head shares the same underlying storage; wrapping that in nn.Parameter
    # makes Adam's in-place addcdiv_ write through aliased storage and
    # PyTorch errors out at the first optimizer step. See init_diag_lin
    # for the full diagnosis.
    omega = eigs_pos.imag.unsqueeze(0).expand(H, -1).contiguous()
    A_imag = omega

    assert A_real.is_contiguous() and A_imag.is_contiguous(), \
        "init_diag_legs: A_real and A_imag must be contiguous"

    kappa = compute_kappa(A_real, A_imag)

    return A_real, A_imag, kappa


def init_s4d_real(H, N, dtype=torch.float32):
    """
    S4D-Real: purely real eigenvalues, no oscillatory modes.

    Eigenvalues: λ_n = -(n+1) for n = 0, ..., N//2 - 1.

    Note that the discrete-time Vandermonde of clustered real points on
    (e^{-N·dt/2}, e^{-dt}) ≈ (0.4, 0.97) at dt = 2/N is exponentially
    ill-conditioned by classical Vandermonde theory — so compute_kappa
    returns a large κ, not 1. The "κ = 1" sometimes claimed for S4D-Real
    refers to the eigenvector matrix of A itself (trivially I for diagonal
    A), which is a different object than the Bauer-Fike-relevant
    Vandermonde of A_disc that compute_kappa now measures.

    Args:
        H: number of heads
        N: state dimension (even)
        dtype: torch dtype

    Returns:
        A_real: (H, N//2) log-parameterized alphas
        A_imag: (H, N//2) omegas (all zeros)
        kappa:  float, Vandermonde condition number at default dt
    """
    assert N % 2 == 0, "N must be even"
    N_half = N // 2

    # Continuous-time eigenvalues: -(n+1).
    # `.contiguous()` defensively, even though `torch.log` below happens to
    # materialize a fresh tensor — keeps the contract uniform across inits
    # and protects against future refactors that might skip torch.log.
    n = torch.arange(N_half, dtype=dtype)
    alpha = (n + 1).unsqueeze(0).expand(H, -1).contiguous()
    omega = torch.zeros(H, N_half, dtype=dtype)

    # Log-parameterize alpha
    A_real = torch.log(alpha)
    A_imag = omega

    assert A_real.is_contiguous() and A_imag.is_contiguous(), \
        "init_s4d_real: A_real and A_imag must be contiguous"

    kappa = compute_kappa(A_real, A_imag)

    return A_real, A_imag, kappa


# Aliases for consistency with goals.md naming
init_fout = init_diag_lin  # FouT = diag-lin
init_skew_hippo = init_diag_legs  # Skew-HiPPO = diag-legs


def smoke_test():
    """
    Smoke test for initialization schemes.

    Verifies:
    1. All initializations run without crashing.
    2. Shapes are correct.
    3. κ(V) values are computed via the discrete-time Vandermonde.
    4. THE THEORETICAL PREDICTION: at the default Nyquist-ish dt,
       FouT κ is O(1) and the FouT-vs-LegS gap is huge (many orders
       of magnitude). If either fails, something is structurally wrong
       with compute_kappa, the LegS construction, or the dt choice —
       the whole experimental program rests on this gap existing.
    """
    H, N = 1, 64

    # diag-lin (FouT)
    A_real_lin, A_imag_lin, kappa_lin = init_diag_lin(H, N)
    assert A_real_lin.shape == (H, N//2)
    assert A_imag_lin.shape == (H, N//2)
    print(f"diag-lin   (FouT)      : κ(V) = {kappa_lin:.2e}")

    # diag-inv
    A_real_inv, A_imag_inv, kappa_inv = init_diag_inv(H, N)
    assert A_real_inv.shape == (H, N//2)
    assert A_imag_inv.shape == (H, N//2)
    print(f"diag-inv               : κ(V) = {kappa_inv:.2e}")

    # diag-legs (Skew-HiPPO)
    A_real_legs, A_imag_legs, kappa_legs = init_diag_legs(H, N)
    assert A_real_legs.shape == (H, N//2)
    assert A_imag_legs.shape == (H, N//2)
    print(f"diag-legs  (Skew-HiPPO): κ(V) = {kappa_legs:.2e}")

    # s4d-real (sanity: under the new discrete-time interpretation, κ is
    # NOT 1 — clustered real points give a poorly-conditioned Vandermonde).
    A_real_real, A_imag_real, kappa_real = init_s4d_real(H, N)
    assert A_real_real.shape == (H, N//2)
    assert A_imag_real.shape == (H, N//2)
    assert (A_imag_real.abs() < 1e-7).all(), \
        "s4d-real should have zero imaginary parts"
    print(f"s4d-real               : κ(V) = {kappa_real:.2e}")

    # --- The two prediction-validating asserts ------------------------------
    # FouT should be roughly O(1) at the canonical Nyquist dt. We pick a
    # generous bound (100) so this doesn't fire on numerically marginal
    # runs; the actual N=64 value should be ~10s.
    assert kappa_lin < 100, (
        f"FouT κ should be O(1) at default dt=2/N; got {kappa_lin:.2e}. "
        f"Check compute_kappa or the FouT init."
    )

    # The whole headline experiment rests on the FouT-vs-LegS gap. If it
    # collapses below ~1e6 something is structurally wrong (likely in the
    # LegS matrix construction or the κ measurement).
    assert kappa_legs > 1e6 * kappa_lin, (
        f"FouT-vs-LegS κ gap collapsed: "
        f"kappa_lin={kappa_lin:.2e}, kappa_legs={kappa_legs:.2e}. "
        f"Theory predicts many orders of magnitude separation."
    )

    print("All smoke tests passed!")


if __name__ == "__main__":
    smoke_test()
