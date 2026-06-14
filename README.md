# On the Conditioning of Diagonal State Space Models Under Quantization-Aware Training

*Does a 12-order-of-magnitude difference in eigenvector conditioning translate into a 12-order difference in quantization robustness? No — and the way it fails is the result.*

> **Paper:** [On the Conditioning of Diagonal State Space Models Under Quantization-Aware Training](SSMQATConditioning.pdf)  ·  **Status:** research code accompanying the paper.

The project set out to test a clean prediction: that S4D initializations differing by ~12 orders of magnitude in eigenvector conditioning κ(V) — FouT (κ ≈ 14) vs. Skew-HiPPO (κ ≈ 1.3×10¹³) — should differ commensurately in low-bit QAT robustness, with FouT tolerating bit-widths where Skew-HiPPO collapses. **That prediction doesn't hold quantitatively — and what its failure revealed is the actual contribution.**

---

## Key findings

- **Training collapses the κ axis — "destination equivalence."** Initializations 12 orders of magnitude apart in κ(V) converge to the *same* high-κ regime under training. HiPPO-optimal structure is better read as a **destination of diagonal-SSM training** than a property of the parameterization: initialization sets the optimization trajectory, not its endpoint.
- **The Bauer–Fike bound is directionally right but quantitatively loose** — under per-channel quantization the predicted ~10¹²× kernel-error gap collapses to ~3–4×.
- **The task-level gap is a baseline, not a quantization effect** — FouT beats Skew-HiPPO by a roughly constant ~3pp at every bit-width *including fp32*.
- **Per-channel quantization is the fix, and the framework says why** — per-mode scaling shrinks the perturbation κ(V) amplifies, giving a spectral account of the engineering convergence on per-channel quantization for SSMs.
- **The honest upshot** — initialization-time κ is *not* predictive of deployment-time conditioning; analyses keyed to init-time κ, this one's first framing included, overstate the differential.

---

## Results at a glance

Headline numbers; full per-cell tables (all inits × bit-widths × seeds) are in the paper and reproducible into `results.csv` via the commands below.

- **Conditioning at init:** FouT κ(V) ≈ 14; Skew-HiPPO κ(V) ≈ 1.3×10¹³ — ~12 orders of magnitude apart.
- **Kernel-error sensitivity:** the predicted ~10¹²× FouT-vs-Skew-HiPPO gap collapses to **~3–4×** under per-channel quantization.
- **Task accuracy (sCIFAR):** FouT leads Skew-HiPPO by a roughly constant **~3pp across all bit-widths including fp32** — e.g. FouT fp32 ≈ 0.758, FouT 4-bit ≈ 0.743 (a ~1.5pp quantization cost, not the orders-of-magnitude effect the bound would suggest).
- **Conditioning under training:** FouT's κ drifts from ~14 at init toward HiPPO magnitude (~10¹³) by end of training; a **frozen-Δ ablation** shows `A_real`/`A_imag` drift alone drives this within a few epochs — i.e. timestep drift is *not* the mechanism.

---

## Repository structure

```
src/
  ssm/
    s4d.py            # S4D layer: FouT / Skew-HiPPO inits, dt init, per-layer κ diagnostics
    model.py          # S4DClassifier, make_param_groups (SSM/other optimizer split)
  qat/
    sensitivity.py    # kernel-error vs bit-width; compare_inits; per-channel vs per-tensor
  train/
    train.py          # training loop (QAT, fp32 throughout, per-epoch κ logging)
    eval.py           # (init × bit-width × seed) sweep harness with resume mode
  data/
    scifar.py         # sequential CIFAR loaders (seeded split/shuffle)
    smnist.py         # sequential MNIST loaders
experiments/
  run_phase2a.py          # YAML-driven sweep entrypoint
  explore_sensitivity.py  # Phase 1 kernel-error exploration (no training)
configs/
  2a_headline.yaml            # the 36-cell headline sweep
  2a_smoke.yaml               # fast smoke test
  2a_ablation_frozen_dt.yaml  # frozen-Δ ablation
```

---

## Installation

```bash
pip install -r requirements.txt
```

`requirements.txt` pins the CUDA 12.1 PyTorch wheel index. If the CUDA wheel doesn't install, training falls back to CPU — `run_phase2a` prints `device=cpu` at startup, so abort there if you expected GPU. Training is fp32 throughout (no AMP) so that quantization is the only source of floating-point noise.

---

## Reproducing the results

**Phase 1 — kernel-error sensitivity (pre-training)**
```bash
python -m experiments.explore_sensitivity
```
Relative kernel error vs bit-width for both inits across L ∈ {64, 256, 1024}, per-channel, deterministic vs stochastic rounding. This is the measurement behind the "~10¹²× collapses to ~3–4×" result, with no training involved.

**Phase 2a — headline QAT sweep (36 cells)**
```bash
python -m experiments.run_phase2a --config configs/2a_headline.yaml
# fast check: --config configs/2a_smoke.yaml [--cpu]
```
2 inits × {2, 3, 4, 6, 8, fp32} × 3 seeds → one row per cell in `results.csv`. **Resume-safe**: re-running skips cells already present, so a partial run continues cleanly rather than duplicating rows.

**Frozen-Δ ablation**
```bash
python -m experiments.run_phase2a --config configs/2a_ablation_frozen_dt.yaml
```
Freezes `log_dt` at the Nyquist initialization (FouT, 4-bit + fp32, 3 seeds) to test whether timestep drift is what erodes the conditioning advantage. (It isn't.)

Every `(init, bit_width, seed)` cell seeds Python / NumPy / Torch *and* the data split and shuffle order, so a seed reproduces a cell exactly.

---

## How the experiments map to the paper

| Experiment | Command | What it establishes |
|---|---|---|
| Kernel-error sensitivity | `experiments.explore_sensitivity` | the predicted κ-driven gap vs. what per-channel quantization actually leaves |
| Headline QAT sweep | `run_phase2a --config configs/2a_headline.yaml` | the flat ~3pp task gap across bit-widths; destination-equivalence of κ under training |
| Frozen-Δ ablation | `run_phase2a --config configs/2a_ablation_frozen_dt.yaml` | rules out timestep drift as the conditioning-erosion mechanism |

---

## Citation

```bibtex
@misc{enwereji2026conditioning,
  title  = {On the Conditioning of Diagonal State Space Models Under Quantization-Aware Training},
  author = {Enwereji, Chidiebube},
  year   = {2026},
  note   = {Preprint}
}
```

*(Update with the arXiv id and any co-authors before publishing.)*