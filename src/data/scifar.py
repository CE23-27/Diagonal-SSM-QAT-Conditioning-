"""
Sequential CIFAR (LRA-style) loader.

LRA's sCIFAR convention: 32x32 CIFAR-10 → grayscaled → flattened to a
length-1024 sequence with one input channel. This is the headline dataset
for Phase 2a (goals.md §3, §6.1).

Returns batches of shape:
    x: (B, L=1024, in_channels=1)  float32, normalized
    y: (B,)                        long (class labels 0..9)

Implementation notes
--------------------
- *No data augmentation.* Standard LRA convention (Tay et al. 2020 and the
  S4 follow-ups). Random crop/flip would make these numbers incomparable
  with published SSM baselines. If you want to ablate augmentation later,
  do it as a separate, clearly-labeled condition.
- *batch_size=50* matches LRA / S4 convention. If comparing against a
  baseline that uses 64 or 100 (some S5 follow-ups), match that for
  effective-LR equivalence — don't silently change.
- *Named transform functions, not Lambdas.* `transforms.Lambda(lambda ...)`
  is not picklable, which breaks DataLoader workers under `spawn`
  (macOS / Windows default). All transform helpers are module-level
  functions so workers can pickle them.
- *Reproducible shuffle.* A DataLoader `generator` is seeded from the same
  seed used for the train/val split, so a given seed determines both the
  split *and* the per-epoch shuffle order.
"""

import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


# CIFAR-10 luminance mean/std after grayscale conversion. These are the values
# used in the LRA / S4 convention pipeline below. Recompute if the upstream
# torchvision transform pipeline ever changes.
_CIFAR_GRAY_MEAN = 0.4810
_CIFAR_GRAY_STD = 0.2392


def _to_seq(t: torch.Tensor) -> torch.Tensor:
    """
    Flatten a (1, H, W) image tensor to (H*W, 1).

    Single named function (picklable) replaces the previous view + transpose
    Lambda pair. The numel is preserved and the resulting layout is
    (sequence_length, in_channels), matching what S4DClassifier expects.
    """
    return t.view(-1, 1)


def _make_transform():
    return transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),                                # (1, 32, 32)
        transforms.Normalize((_CIFAR_GRAY_MEAN,), (_CIFAR_GRAY_STD,)),
        transforms.Lambda(_to_seq),                           # Lambda(named fn) ⇒ picklable
    ])


def get_loaders(
    data_dir: str = './data',
    batch_size: int = 50,
    num_workers: int = 4,
    val_fraction: float = 0.1,
    seed: int = 0,
    download: bool = True,
    subset_train: int = None,
    subset_test: int = None,
    pin_memory: bool = True,
    persistent_workers: bool = True,
):
    """
    Build (train, val, test) DataLoaders for sequential CIFAR-10 (grayscale).

    Args:
        data_dir, batch_size, num_workers, val_fraction, seed, download,
        subset_train, subset_test:  same conventions as smnist.get_loaders.
        pin_memory:         pin tensors in pinned host memory for faster H→D
                            copies. Set True for GPU runs (default), False if
                            running CPU-only or under tight memory pressure.
        persistent_workers: keep DataLoader workers alive between epochs.
                            Avoids the worker-respawn cost that's noticeable
                            on a 50k dataset × ~100 epochs. Only effective
                            when num_workers > 0.

    Returns:
        (train_loader, val_loader, test_loader, info_dict)
        info_dict: {'L': 1024, 'in_channels': 1, 'num_classes': 10}.
    """
    os.makedirs(data_dir, exist_ok=True)
    transform = _make_transform()

    train_full = datasets.CIFAR10(
        root=data_dir, train=True, download=download, transform=transform,
    )
    test_set = datasets.CIFAR10(
        root=data_dir, train=False, download=download, transform=transform,
    )

    # Reproducible train/val split.
    n_total = len(train_full)
    n_val = int(round(val_fraction * n_total))
    n_train = n_total - n_val
    g_split = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n_total, generator=g_split).tolist()
    train_idx = perm[:n_train]
    val_idx = perm[n_train:]

    if subset_train is not None:
        train_idx = train_idx[:subset_train]
    if subset_test is not None:
        test_set = Subset(test_set, list(range(min(subset_test, len(test_set)))))

    train_set = Subset(train_full, train_idx)
    val_set = Subset(train_full, val_idx)

    # Per-epoch shuffle generator, separate from the split generator so the
    # split is invariant to shuffle reseeding. Same seed ⇒ same shuffle order.
    g_shuffle = torch.Generator().manual_seed(seed)

    persistent = persistent_workers and num_workers > 0
    common = dict(
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=persistent,
    )
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True,
        generator=g_shuffle, **common,
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, drop_last=False, **common
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, drop_last=False, **common
    )

    info = {'L': 1024, 'in_channels': 1, 'num_classes': 10}
    return train_loader, val_loader, test_loader, info
