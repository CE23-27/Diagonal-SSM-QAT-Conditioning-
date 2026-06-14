"""
Exploratory follow-ups on the kernel-error-vs-bit-width measurement.

Three measurements, no asserts:

1) Extend the bit-width sweep to {10, 12, 16}. If the FouT-vs-LegS gap
   widens at high precision (FouT → 1e-3, LegS stalls at 1e-2 or higher),
   the κ story still has explanatory power at high precision. If both
   curves go to zero together, the residual gap from the original sweep
   really is just dynamic-range noise.

2) Re-run B ∈ {2, 3} with n_trials = 20 to see whether the surprising
   "B=3 sometimes worse than B=2" pattern is real or trial noise.

3) Compare stochastic vs deterministic rounding at the standard sweep.
   Stochastic averaging can hide grid-alignment artifacts; deterministic
   gives a single reproducible kernel.

Run: python -m experiments.explore_sensitivity
"""

import os
import sys

# Make `src` importable when run as a module or as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import torch
import numpy as np

from src.ssm.s4d import S4D
from src.qat.sensitivity import compare_inits, kernel_error_curve


D_MODEL = 4
D_STATE = 64
L_VALUES = [64, 256, 1024]
INITS = ['fout', 'skew-hippo']


def factory(init: str) -> S4D:
    # Same seed for both inits so B/C/log_dt are matched.
    torch.manual_seed(42)
    return S4D(
        d_model=D_MODEL,
        d_state=D_STATE,
        init=init,
        dt_init='nyquist',
    )


def _print_table(results, bits_list, title):
    print(f"\n--- {title} ---")
    print(f"{'init':<12} {'L':>5} " + " ".join(f"B={B:<8}" for B in bits_list))
    for init in INITS:
        for L in L_VALUES:
            row = results[init][L]
            cells = " ".join(f"{row[B]['mean']:.2e}" for B in bits_list)
            print(f"{init:<12} {L:>5} {cells}")


def experiment_high_precision():
    """Extend B to {8, 10, 12, 16} and check whether the gap widens or closes."""
    bits_list = [8, 10, 12, 16]
    results = compare_inits(
        factory,
        inits=INITS,
        bits_list=bits_list,
        L_values=L_VALUES,
        n_trials=5,
        mode='stochastic',
        per_channel=True,
        seed=0,
    )
    _print_table(results, bits_list, "High-precision sweep (per-mode, stochastic, n_trials=5)")

    # Also print FouT/LegS ratio per cell so the eye doesn't have to do log diffs.
    print(f"\n{'L':>5} " + " ".join(f"B={B:<8}" for B in bits_list))
    for L in L_VALUES:
        ratios = []
        for B in bits_list:
            f = results['fout'][L][B]['mean']
            l = results['skew-hippo'][L][B]['mean']
            ratios.append(l / max(f, 1e-12))
        print(f"{L:>5} " + " ".join(f"{r:>9.2e}" for r in ratios) + "    (LegS/FouT)")


def experiment_low_precision_high_trials():
    """20-trial sweep at B=2, 3 only, to test the 'B=3 worse than B=2' anomaly."""
    bits_list = [2, 3]
    n_trials = 20
    results = compare_inits(
        factory,
        inits=INITS,
        bits_list=bits_list,
        L_values=L_VALUES,
        n_trials=n_trials,
        mode='stochastic',
        per_channel=True,
        seed=0,
    )
    print(f"\n--- Low-precision sweep (per-mode, stochastic, n_trials={n_trials}) ---")
    print(f"{'init':<12} {'L':>5} " +
          " ".join(f"B={B} mean±std            " for B in bits_list))
    for init in INITS:
        for L in L_VALUES:
            row = results[init][L]
            cells = " ".join(
                f"{row[B]['mean']:.2e}±{row[B]['std']:.2e}"
                for B in bits_list
            )
            print(f"{init:<12} {L:>5} {cells}")

    # For each (init, L), is mean(B=3) > mean(B=2)? Report the difference
    # and how it compares to the std at B=3.
    print(f"\n{'init':<12} {'L':>5} mean(B=3) - mean(B=2)   (units of std at B=3)")
    for init in INITS:
        for L in L_VALUES:
            r2 = results[init][L][2]
            r3 = results[init][L][3]
            diff = r3['mean'] - r2['mean']
            sigma = r3['std']
            z = diff / max(sigma, 1e-12)
            print(f"{init:<12} {L:>5} {diff:+.2e}            ({z:+.2f} σ)")


def experiment_deterministic_vs_stochastic():
    """Same sweep, deterministic rounding, to remove averaging."""
    bits_list = [2, 3, 4, 5, 6, 8]
    results_det = compare_inits(
        factory,
        inits=INITS,
        bits_list=bits_list,
        L_values=L_VALUES,
        n_trials=1,                # deterministic ⇒ identical every trial
        mode='deterministic',
        per_channel=True,
        seed=0,
    )
    _print_table(results_det, bits_list, "Standard sweep (per-mode, DETERMINISTIC, n_trials=1)")

    # Stochastic for direct visual comparison.
    results_stoch = compare_inits(
        factory,
        inits=INITS,
        bits_list=bits_list,
        L_values=L_VALUES,
        n_trials=5,
        mode='stochastic',
        per_channel=True,
        seed=0,
    )
    _print_table(results_stoch, bits_list, "Standard sweep (per-mode, stochastic,  n_trials=5)")

    # Print stoch / det ratio so it's easy to see if stochastic averaging
    # is materially changing the signal.
    print(f"\n--- ratio stochastic / deterministic ---")
    print(f"{'init':<12} {'L':>5} " + " ".join(f"B={B:<6}" for B in bits_list))
    for init in INITS:
        for L in L_VALUES:
            cells = []
            for B in bits_list:
                d = results_det[init][L][B]['mean']
                s = results_stoch[init][L][B]['mean']
                cells.append(f"{(s / max(d, 1e-12)):.2f}")
            print(f"{init:<12} {L:>5} " + " ".join(f"{c:<6}" for c in cells))


def main():
    print("Sensitivity exploration runs.")
    print(f"d_model={D_MODEL}, d_state={D_STATE}, dt_init='nyquist', "
          f"L_values={L_VALUES}\n")
    experiment_high_precision()
    experiment_low_precision_high_trials()
    experiment_deterministic_vs_stochastic()


if __name__ == "__main__":
    main()
