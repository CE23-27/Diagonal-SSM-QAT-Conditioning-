"""
Phase 2a entrypoint.

Reads a YAML config, builds data loaders + model_kwargs + train_kwargs +
quant_kwargs from it, then dispatches to src.train.eval.run_sweep.

Usage:
    python -m experiments.run_phase2a --config configs/2a_headline.yaml
    python -m experiments.run_phase2a --config configs/2a_smoke.yaml [--cpu]
"""

import argparse
import os
import sys
import shutil
import yaml

# Make `src` importable when run as a module or as a script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from src.data import scifar, smnist
from src.train.eval import run_sweep


_DATA_REGISTRY = {
    'scifar': scifar,
    'smnist': smnist,
}


def _build_data_loaders_fn(data_cfg: dict):
    """
    Returns a callable `loaders_fn(seed)` that builds (train, val, test, info).

    All non-seed arguments come from the YAML `data:` block. The seed
    threads through to the train/val split and per-epoch shuffle order
    so a given seed reproduces the same DataLoader content.
    """
    name = data_cfg['name']
    if name not in _DATA_REGISTRY:
        raise ValueError(f"unknown dataset '{name}'. Available: {list(_DATA_REGISTRY)}")
    module = _DATA_REGISTRY[name]

    # Whitelist the kwargs we forward — keeps the YAML schema explicit.
    forwarded = {
        k: data_cfg[k] for k in (
            'data_dir', 'batch_size', 'num_workers', 'val_fraction',
            'download', 'subset_train', 'subset_test',
            'pin_memory', 'persistent_workers',
        )
        if k in data_cfg
    }

    def loaders_fn(seed: int):
        return module.get_loaders(seed=seed, **forwarded)

    return loaders_fn


def _device(cpu_flag: bool):
    if cpu_flag:
        return torch.device('cpu')
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # MPS works for fp32 ops here; if anything breaks fall back to CPU.
        return torch.device('mps')
    return torch.device('cpu')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', required=True, help='YAML config path')
    parser.add_argument('--cpu', action='store_true',
                        help='force CPU even if CUDA / MPS is available')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        cfg = yaml.safe_load(f)

    output_dir = cfg['output']['output_dir']
    os.makedirs(output_dir, exist_ok=True)

    # Snapshot the config alongside the results CSV for reproducibility.
    cfg_snapshot = os.path.join(output_dir, os.path.basename(args.config))
    try:
        shutil.copyfile(args.config, cfg_snapshot)
    except shutil.SameFileError:
        pass

    device = _device(args.cpu)
    print(f"experiment={cfg.get('experiment_name', '<unnamed>')}  device={device}")
    print(f"config={args.config}  output_dir={output_dir}")

    loaders_fn = _build_data_loaders_fn(cfg['data'])

    rows = run_sweep(
        inits=cfg['sweep']['inits'],
        bit_widths=cfg['sweep']['bit_widths'],
        seeds=cfg['sweep']['seeds'],
        data_loaders_fn=loaders_fn,
        device=device,
        model_kwargs=cfg['model'],
        train_kwargs=cfg['training'],
        quant_kwargs=cfg['quant'],
        output_dir=output_dir,
        csv_name=cfg['output'].get('csv_name', 'results.csv'),
        save_best=cfg['output'].get('save_best', False),
    )

    print(f"\nWrote {len(rows)} new rows to {output_dir}/{cfg['output'].get('csv_name', 'results.csv')}")


if __name__ == "__main__":
    main()
