"""
Kernel-sensitivity measurement: the direct empirical test of the QAT
bit-width prediction (goals.md §2, Claim 2).

What this measures
------------------
Take a trained or freshly-initialized S4D layer with fp32 parameters.
Compute its convolution kernel K_fp32 of length L. Then for each bit-width
B in a sweep, replace the layer's α / ω quantizers with B-bit quantizers
and recompute the kernel K_q. Report the relative L1 error
    err(B) = mean(|K_q - K_fp32|) / mean(|K_fp32|).

For stochastic rounding we average over `n_trials` independent draws and
return both mean and std.

Why this is useful
------------------
The bit-width formula B* ≈ max(log2 L, log2(N κ(V))) + O(1) predicts:
    - For FouT (κ = O(1)):   err(B) flat down to ~log2 L bits, then rises.
    - For LegS (κ huge):     err(B) rises sharply at every B in {2..8}.

Plotting err(B) for FouT and LegS at a few values of L is the cleanest
falsifier of the theory that doesn't depend on training to convergence.
The κ measurement supplies the explanation; this curve is the result.

Implementation note
-------------------
We *do not* clone the layer or copy state_dict — we temporarily swap in
new quantizer modules on the existing layer and restore the originals in
a try/finally. This sidesteps the awkwardness of moving quantizer-internal
buffers (`scale`) through a state_dict, and guarantees we are measuring
the *exact* same parameter tensors.
"""

import math
import torch
import torch.nn as nn
import numpy as np

from .quantizers import make_quantizer


def _auto_scale(param: torch.Tensor, headroom: float = 1.05) -> float:
    """
    Per-tensor symmetric quantization scale that covers the parameter range.

    A fixed scale per (init, parameter) is wrong because LegS' ω runs to
    ~1300 while FouT's tops out at ~100 for N=64. Derive from data.

    Args:
        param:    tensor whose values must all fit in [-scale, +scale].
        headroom: multiplicative safety factor on the observed max.

    Returns:
        scale: float ≥ a tiny floor (avoids division-by-zero in the
               quantizer when all values happen to be exactly zero).
    """
    m = float(param.detach().abs().max().item())
    return max(headroom * m, 1e-6)


def _auto_scale_per_mode(param: torch.Tensor, headroom: float = 1.05) -> torch.Tensor:
    """
    Per-mode quantization scale for a tensor of shape (H, N//2):
    one scale per mode, shared across heads.

    Returns a tensor of shape (1, N//2) — broadcasts back against (H, N//2)
    automatically inside the Quantizer.

    For FouT this is exactly what restores frequency separation through
    quantization: each ω_n gets a grid sized to its own magnitude, so
    adjacent modes can no longer collide on the same quantized value. For
    LegS the giant high-magnitude modes still sit on a coarse grid, but
    the small modes that per-tensor quantization erased now get a sensible
    grid of their own.
    """
    m = param.detach().abs().amax(dim=0, keepdim=True)   # (1, N//2)
    return torch.clamp(headroom * m, min=1e-6)


def kernel_error_curve(
    layer: nn.Module,
    bits_list,
    L: int,
    n_trials: int = 5,
    mode: str = 'stochastic',
    scale_alpha=None,
    scale_omega=None,
    per_channel: bool = False,
    seed: int = 0,
):
    """
    Measure relative kernel error vs bit-width for a single S4D layer.

    Args:
        layer:        an S4D instance (or anything with a `.kernel(L)` method
                      and `.q_alpha`, `.q_omega` attributes that get swapped
                      in for quantization).
        bits_list:    iterable of ints, e.g. [2, 3, 4, 5, 6, 8].
        L:            kernel length.
        n_trials:     number of independent quantizer draws per bit-width.
                      For deterministic rounding 1 is sufficient; for
                      stochastic rounding more trials reduces std.
        mode:         'deterministic' or 'stochastic'.
        scale_alpha:  scale for the α quantizer. None ⇒ auto-derive from
                      layer.A_real (per-tensor or per-mode depending on
                      `per_channel`).
        scale_omega:  scale for the ω quantizer. None ⇒ auto-derive from
                      layer.A_imag.
        per_channel:  if True, derive a per-mode scale of shape (1, N//2)
                      so each mode gets its own grid. Defaults to False
                      (per-tensor) for backward compatibility, but per_mode
                      is what the spec's mixed-precision section calls for
                      and is the only setting where the κ theory applies
                      cleanly to *each mode independently*.
        seed:         RNG seed for reproducible stochastic rounding.

    Returns:
        results: dict mapping B → {
            'mean': mean relative L1 error over n_trials,
            'std':  std relative L1 error over n_trials,
            'n':    n_trials,
        }
        Plus a special key 'fp32_kernel_l1_mean' giving the reference
        kernel norm so the caller can sanity-check that the kernel is not
        trivially zero, and 'per_channel' echoing the flag for downstream
        bookkeeping.
    """
    if scale_alpha is None:
        scale_alpha = (_auto_scale_per_mode(layer.A_real) if per_channel
                       else _auto_scale(layer.A_real))
    if scale_omega is None:
        scale_omega = (_auto_scale_per_mode(layer.A_imag) if per_channel
                       else _auto_scale(layer.A_imag))

    # Save and restore the original quantizers so this function is
    # side-effect-free. Wrap everything in try/finally to avoid leaving the
    # caller's layer in a partially-quantized state on exception.
    orig_q_alpha = layer.q_alpha
    orig_q_omega = layer.q_omega
    layer.q_alpha = nn.Identity()
    layer.q_omega = nn.Identity()

    try:
        # fp32 reference. detach() so no autograd graph is retained.
        with torch.no_grad():
            K_fp32 = layer.kernel(L).detach()
        ref_norm = float(K_fp32.abs().mean().item())
        if ref_norm < 1e-12:
            # Avoid divide-by-zero downstream and surface the issue loudly.
            raise RuntimeError(
                f"fp32 reference kernel is ~0 (mean |K| = {ref_norm:.2e}). "
                f"Check init / B,C / dt — relative error is undefined."
            )

        results = {
            'fp32_kernel_l1_mean': ref_norm,
            'per_channel': per_channel,
        }

        # Reproducible per-trial stochastic rounding.
        gen = torch.Generator(device=K_fp32.device).manual_seed(seed)

        for B in bits_list:
            errs = []
            for trial in range(n_trials):
                # Fresh quantizer per trial so independent stochastic draws
                # are produced (the rand_like inside the quantizer uses the
                # default torch RNG; re-seed it for reproducibility).
                torch.manual_seed(seed + trial * 1000 + B)
                q_a = make_quantizer(B, scale=scale_alpha, mode=mode)
                q_w = make_quantizer(B, scale=scale_omega, mode=mode)
                # Move to the layer's device so quantizer buffers are on the
                # right device when the kernel uses them.
                q_a = q_a.to(K_fp32.device)
                q_w = q_w.to(K_fp32.device)

                layer.q_alpha = q_a
                layer.q_omega = q_w

                with torch.no_grad():
                    K_q = layer.kernel(L).detach()
                err = float((K_q - K_fp32).abs().mean().item()) / ref_norm
                errs.append(err)

            results[B] = {
                'mean': float(np.mean(errs)),
                'std':  float(np.std(errs)),
                'n':    n_trials,
            }
    finally:
        # Always restore — even on RuntimeError, KeyboardInterrupt, etc.
        layer.q_alpha = orig_q_alpha
        layer.q_omega = orig_q_omega

    return results


def compare_inits(
    s4d_factory,
    inits,
    bits_list,
    L_values,
    n_trials: int = 5,
    mode: str = 'stochastic',
    per_channel: bool = False,
    seed: int = 0,
):
    """
    Run kernel_error_curve for several (init, L) cells.

    Args:
        s4d_factory: callable (init: str) -> S4D layer. Lets the caller
                     fix d_model / d_state / dt_init etc. without
                     hard-coding them here.
        inits:       list of init names to compare, e.g. ['fout', 'skew-hippo'].
        bits_list:   iterable of bit-widths.
        L_values:    iterable of kernel lengths.
        n_trials:    trials per (init, L, B) cell.
        mode:        'deterministic' or 'stochastic'.
        per_channel: forwarded to kernel_error_curve.
        seed:        RNG seed.

    Returns:
        nested dict: results[init][L] = kernel_error_curve(...) output.
    """
    out = {}
    for init in inits:
        out[init] = {}
        for L in L_values:
            layer = s4d_factory(init)
            out[init][L] = kernel_error_curve(
                layer, bits_list, L,
                n_trials=n_trials, mode=mode,
                per_channel=per_channel, seed=seed,
            )
    return out


def smoke_test():
    """
    Structural smoke test only. Asserts that:
        - The function runs without crashing.
        - Returned dicts have the expected keys for every (init, L, B).
        - All errors are finite, non-negative, and the fp32 reference is
          not trivially zero.
        - Shapes / scales agree under per_channel=True/False.

    No prediction-validating asserts here. The goal of this test is to
    confirm the measurement machinery works; deciding what threshold the
    FouT-vs-LegS gap is "supposed to" hit before we have measurements is
    rationalizing in advance. Run the experiment, eyeball the numbers,
    then write asserts that test the observed pattern at thresholds that
    actually distinguish a real effect from noise.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    from src.ssm.s4d import S4D

    torch.manual_seed(0)

    d_model, d_state = 4, 64
    L_values = [64, 256, 1024]
    bits_list = [2, 3, 4, 5, 6, 8]

    def factory(init: str) -> S4D:
        torch.manual_seed(42)
        return S4D(
            d_model=d_model,
            d_state=d_state,
            init=init,
            dt_init='nyquist',
        )

    for per_channel in (False, True):
        label = "per-channel (per-mode)" if per_channel else "per-tensor"
        print(f"\n=== {label} quantization ===")
        results = compare_inits(
            factory,
            inits=['fout', 'skew-hippo'],
            bits_list=bits_list,
            L_values=L_values,
            n_trials=3,
            mode='stochastic',
            per_channel=per_channel,
        )

        # Header
        print(f"{'init':<12} {'L':>5} " + " ".join(f"B={B:<6}" for B in bits_list))
        for init in ('fout', 'skew-hippo'):
            for L in L_values:
                row = results[init][L]
                cells = " ".join(f"{row[B]['mean']:.2e}" for B in bits_list)
                print(f"{init:<12} {L:>5} {cells}")

        # --- structural asserts only ------------------------------------
        for init in ('fout', 'skew-hippo'):
            for L in L_values:
                row = results[init][L]
                # required top-level keys
                for k in ('fp32_kernel_l1_mean', 'per_channel'):
                    assert k in row, f"missing key {k} in results[{init}][{L}]"
                assert row['per_channel'] == per_channel
                assert row['fp32_kernel_l1_mean'] > 0, (
                    f"fp32 reference vanished for {init} L={L}"
                )
                # per-bit cells
                for B in bits_list:
                    cell = row[B]
                    for k in ('mean', 'std', 'n'):
                        assert k in cell, (
                            f"missing key {k} in results[{init}][{L}][{B}]"
                        )
                    assert math.isfinite(cell['mean']), (
                        f"non-finite mean error at {init}, L={L}, B={B}"
                    )
                    assert math.isfinite(cell['std']), (
                        f"non-finite std at {init}, L={L}, B={B}"
                    )
                    assert cell['mean'] >= 0, (
                        f"negative error at {init}, L={L}, B={B}"
                    )
                    assert cell['n'] == 3

    print("\nStructural smoke test passed.")


if __name__ == "__main__":
    smoke_test()
