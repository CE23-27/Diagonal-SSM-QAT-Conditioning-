"""
Sequential MNIST loader (sanity-check dataset).

A small, fast dataset to use for end-to-end pipeline checks before launching
the full sCIFAR sweep. 28x28 grayscale → length-784 sequence with 1 channel.

Returns batches of shape:
    x: (B, L=784, in_channels=1)   float32 in [-1, 1]
    y: (B,)                        long (class labels 0..9)

Implementation notes
--------------------
- Named transform functions (no `lambda`s) so workers can pickle them under
  `spawn` start method (macOS / Windows default).
- Reproducible shuffle via a seeded `generator` passed into the train
  DataLoader.
"""

import os
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


def _to_seq(t: torch.Tensor) -> torch.Tensor:
    """(1, 28, 28) → (784, 1). Single named, picklable transform helper."""
    return t.view(-1, 1)


def _scale_pm1(t: torch.Tensor) -> torch.Tensor:
    """[0, 1] → [-1, 1]. Lightweight normalization without depending on
    dataset statistics — sufficient for a sanity-check dataset."""
    return t * 2.0 - 1.0


def _make_transform():
    return transforms.Compose([
        transforms.ToTensor(),                  # (1, 28, 28) in [0, 1]
        transforms.Lambda(_to_seq),             # (784, 1)
        transforms.Lambda(_scale_pm1),          # in [-1, 1]
    ])


def get_loaders(
    data_dir: str = './data',
    batch_size: int = 64,
    num_workers: int = 2,
    val_fraction: float = 0.1,
    seed: int = 0,
    download: bool = True,
    subset_train: int = None,
    subset_test: int = None,
    pin_memory: bool = True,
    persistent_workers: bool = True,
):
    """
    Build (train, val, test) DataLoaders for sequential MNIST.

    Args:
        data_dir:           where torchvision puts the raw files.
        batch_size:         per-loader batch size.
        num_workers:        DataLoader workers.
        val_fraction:       fraction of the training set carved off for val.
        seed:               seed for both the train/val split AND the
                            per-epoch shuffle order.
        download:           passed to torchvision.
        subset_train:       restrict training set to this many samples
                            (smoke tests).
        subset_test:        restrict the test set similarly.
        pin_memory:         pin tensors in pinned host memory for faster H→D.
        persistent_workers: keep workers alive between epochs.

    Returns:
        (train_loader, val_loader, test_loader, info_dict)
        info_dict has keys: 'L' (784), 'in_channels' (1), 'num_classes' (10).
    """
    os.makedirs(data_dir, exist_ok=True)
    transform = _make_transform()

    train_full = datasets.MNIST(
        root=data_dir, train=True, download=download, transform=transform,
    )
    test_set = datasets.MNIST(
        root=data_dir, train=False, download=download, transform=transform,
    )

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

    info = {'L': 784, 'in_channels': 1, 'num_classes': 10}
    return train_loader, val_loader, test_loader, info
