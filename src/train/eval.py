"""
Evaluation utilities and the Phase 2a bit-width × init × seed sweep harness.

The headline experiment (goals.md §3, §6.1) is a 2 × (5+1) × 3 = 36-cell
sweep:
    inits      = ['fout', 'skew-hippo']
    bit_widths = [2, 3, 4, 6, 8, None]      # None = fp32 reference
    seeds      = [s0, s1, s2]

Each cell builds a fresh model, trains it for `epochs`, evaluates on the
held-out test set, and writes one row to the output CSV with the columns
goals.md §6.4 mandates:
    init, bit_width, seed, val_acc, test_acc, train_loss_final

Plus a few diagnostics for the writeup: best_epoch, kappa_init_layer0,
kappa_max_global_final, total_time_s.

Sweep semantics
---------------
- *Resume mode.* When `run_sweep` finds an existing CSV at the output
  path, it parses it, builds a set of completed `(init, bit_width, seed)`
  triples, and skips those cells. Append-without-skip would duplicate
  every row after a partial re-run, silently inflating the apparent
  sample size.
- *Per-cell seeding.* The seed determines (a) PyTorch / NumPy / Python
  RNGs and (b) the train/val split and shuffle order via
  `data_loaders_fn(seed)`. So `(init, bits, seed)` uniquely identifies
  the cell and the seed alone is sufficient to reproduce it.
- *No CSV mutation.* The harness only writes new rows; it never
  overwrites an existing row. If you actually want to re-run a cell,
  delete it from the CSV first (or wipe the file).

Also provided
-------------
- `bit_width_sweep(model, ...)`: post-training quantization on a *trained*
  model, useful to compare PTQ-only vs QAT separation.
"""

import csv
import os
import time
import random
import numpy as np
import torch

from ..ssm.model import S4DClassifier
from .train import train, evaluate


# ----------------------------------------------------------------------
# Reproducibility helper.
# ----------------------------------------------------------------------

def set_global_seed(seed: int):
    """Seed Python, NumPy, and torch (CPU + CUDA) for a single cell."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ----------------------------------------------------------------------
# bit-width <-> CSV-cell-string conversion (canonical form).
# ----------------------------------------------------------------------

def _bits_to_str(bits) -> str:
    """Canonical CSV representation: 'fp32' for None/32, else 'B'."""
    if bits is None or bits == 32:
        return 'fp32'
    return str(int(bits))


# ----------------------------------------------------------------------
# Single (init, bits, seed) cell.
# ----------------------------------------------------------------------

def run_one_cell(
    *,
    init: str,
    bits,                          # int or None for fp32
    seed: int,
    train_loader,
    val_loader,
    test_loader,
    data_info: dict,
    device,
    model_kwargs: dict,
    train_kwargs: dict,
    quant_kwargs: dict,
) -> dict:
    """
    Build, train, and test one cell of the sweep.

    Returns a dict with the CSV row plus best_state_dict for the caller
    to optionally checkpoint.
    """
    set_global_seed(seed)

    # Build the model with this init.
    model = S4DClassifier(
        in_channels=data_info['in_channels'],
        num_classes=data_info['num_classes'],
        init=init,
        **model_kwargs,
    )

    # Install quantizers BEFORE training so QAT happens during, not after.
    # If bits is None / 32, set_quantizers reverts to Identity (fp32).
    model.set_quantizers(
        bits=bits,
        mode=quant_kwargs.get('mode', 'deterministic'),
        per_channel=quant_kwargs.get('per_channel', True),
        headroom=quant_kwargs.get('headroom', 1.05),
    )

    t0 = time.time()
    result = train(
        model, train_loader, val_loader, device,
        diagnose_L=data_info['L'],
        **train_kwargs,
    )
    elapsed = time.time() - t0

    # Restore the best-val checkpoint before testing.
    if result['best_state_dict'] is not None:
        model.load_state_dict(result['best_state_dict'])
    test_metrics = evaluate(model, test_loader, device)

    # Last-epoch diagnostics for the writeup. `kappa_max_global` is the
    # key the trainer logs (max over all layers); using .get with NaN
    # default keeps this robust to history-format drift, but the value
    # should be real not NaN for well-behaved runs.
    last = result['history'][-1] if result['history'] else {}
    kappas_init = model.kappas_init()
    kappa_init_layer0 = kappas_init[0] if kappas_init else float('nan')

    return {
        'init':                    init,
        'bit_width':               _bits_to_str(bits),
        'seed':                    seed,
        'val_acc':                 result['best_val_acc'],
        'test_acc':                test_metrics['acc'],
        'train_loss_final':        result['final_train_loss'],
        'best_epoch':              result['best_epoch'],
        'kappa_init_layer0':       kappa_init_layer0,
        'kappa_max_global_final':  last.get('kappa_max_global', float('nan')),
        'total_time_s':            elapsed,
        'best_state_dict':         result['best_state_dict'],
    }


# ----------------------------------------------------------------------
# Full sweep over (init, bits, seed) with resume mode.
# ----------------------------------------------------------------------

_CSV_COLUMNS = [
    'init', 'bit_width', 'seed', 'val_acc', 'test_acc', 'train_loss_final',
    'best_epoch', 'kappa_init_layer0', 'kappa_max_global_final', 'total_time_s',
]


def _load_done_keys(csv_path: str) -> set:
    """Return the set of (init, bit_width_str, seed_int) already in the CSV."""
    done = set()
    if not os.path.exists(csv_path):
        return done
    with open(csv_path, 'r', newline='') as fr:
        reader = csv.DictReader(fr)
        for row in reader:
            try:
                key = (row['init'], row['bit_width'], int(row['seed']))
            except (KeyError, ValueError):
                # Skip malformed rows quietly — we don't want a stray
                # header-only or partial line to crash the resume scan.
                continue
            done.add(key)
    return done


def run_sweep(
    *,
    inits,
    bit_widths,                    # list possibly containing None
    seeds,
    data_loaders_fn,               # callable(seed) -> (train, val, test, info)
    device,
    model_kwargs: dict,
    train_kwargs: dict,
    quant_kwargs: dict,
    output_dir: str,
    csv_name: str = 'results.csv',
    save_best: bool = False,
):
    """
    Run the (init, bits, seed) grid and append rows to a CSV. Resumes
    cleanly if the CSV already contains some cells.

    Args:
        inits, bit_widths, seeds: lists defining the sweep grid.
        data_loaders_fn: builds reproducible loaders for a given seed.
                         Signature: (seed:int) -> (train, val, test, info)
        device:          torch.device.
        model_kwargs:    forwarded to S4DClassifier (minus in_channels,
                         num_classes, init — those come from data_info /
                         the sweep loop).
        train_kwargs:    forwarded to train().
        quant_kwargs:    forwarded to model.set_quantizers().
        output_dir:      where to put the CSV (and optional checkpoints).
        csv_name:        name of the CSV file.
        save_best:       if True, write `{init}_b{bits}_s{seed}.pt`
                         checkpoints to output_dir.

    Returns the list of result-row dicts (without the state_dict field)
    for cells run *in this invocation*. Cells skipped because they were
    already in the CSV are not re-added to the return value.
    """
    os.makedirs(output_dir, exist_ok=True)
    csv_path = os.path.join(output_dir, csv_name)

    # Resume-mode scan. The set lookup uses the canonical bit_width string
    # form produced by _bits_to_str, so callers can resume safely whether
    # their python-side bits is `None`, `32`, or `8`.
    done = _load_done_keys(csv_path)
    if done:
        print(f"resume: found {len(done)} completed cells in {csv_path}")

    file_exists = os.path.exists(csv_path)
    f = open(csv_path, 'a', newline='')
    writer = csv.DictWriter(f, fieldnames=_CSV_COLUMNS)
    if not file_exists:
        writer.writeheader()
        f.flush()

    rows = []
    try:
        for init in inits:
            for bits in bit_widths:
                bits_str = _bits_to_str(bits)
                for seed in seeds:
                    key = (init, bits_str, seed)
                    label = f"init={init} bits={bits_str} seed={seed}"
                    if key in done:
                        print(f"--- skip (already in CSV): {label}")
                        continue

                    print(f"\n=== {label} ===")
                    train_loader, val_loader, test_loader, info = data_loaders_fn(seed)
                    cell = run_one_cell(
                        init=init,
                        bits=bits,
                        seed=seed,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        test_loader=test_loader,
                        data_info=info,
                        device=device,
                        model_kwargs=model_kwargs,
                        train_kwargs=train_kwargs,
                        quant_kwargs=quant_kwargs,
                    )
                    state = cell.pop('best_state_dict', None)
                    rows.append(cell)
                    writer.writerow({k: cell[k] for k in _CSV_COLUMNS})
                    f.flush()
                    done.add(key)  # so an intra-run repeat would also be skipped

                    if save_best and state is not None:
                        bw_tag = 'fp32' if bits in (None, 32) else f'b{int(bits)}'
                        ckpt_path = os.path.join(
                            output_dir, f"{init}_{bw_tag}_s{seed}.pt"
                        )
                        torch.save(state, ckpt_path)

                    print(
                        f"--- done: val_acc={cell['val_acc']:.4f}  "
                        f"test_acc={cell['test_acc']:.4f}  "
                        f"time={cell['total_time_s']:.0f}s"
                    )
    finally:
        f.close()

    return rows


# ----------------------------------------------------------------------
# Post-training bit-width sweep (no retraining).
#
# Useful as an ablation: does the κ-driven gap show up purely in
# post-training quantization, or does it require QAT to manifest?
# ----------------------------------------------------------------------

@torch.no_grad()
def bit_width_sweep(
    model: S4DClassifier,
    test_loader,
    device,
    bit_widths,
    *,
    mode: str = 'deterministic',
    per_channel: bool = True,
    headroom: float = 1.05,
) -> list:
    """
    Take a *trained* (assumed fp32) model and report test accuracy at each
    bit-width in `bit_widths`, switching quantizers in-place between calls.
    """
    model.to(device)
    results = []
    for bits in bit_widths:
        model.set_quantizers(
            bits=bits, mode=mode, per_channel=per_channel, headroom=headroom,
        )
        m = evaluate(model, test_loader, device)
        results.append({
            'bit_width': _bits_to_str(bits),
            'test_acc':  m['acc'],
            'test_loss': m['loss'],
        })
    # Leave the model in fp32 state on exit so the caller isn't surprised.
    model.set_quantizers(bits=None)
    return results
