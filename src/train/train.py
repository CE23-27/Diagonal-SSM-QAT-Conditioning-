"""
Training loop for Phase 2a.

Standard supervised image classification on sMNIST / sCIFAR with the
S4DClassifier model, with optional QAT applied to A_real / A_imag in each
S4D layer. Quantizers are inserted via `model.set_quantizers(bits, ...)`
*before* training begins and stay in place throughout — Phase 2a does not
do quantizer scheduling.

Design notes
------------
- AdamW with two parameter groups: SSM params (low LR, no weight decay)
  vs everything else. See model.make_param_groups.
- Cosine LR schedule with a short linear warmup, applied to both groups.
- Optional global gradient clipping (default 1.0).
- AMP is intentionally NOT used. Complex64 inside the kernel does not
  interact cleanly with torch.cuda.amp.autocast; we keep fp32 throughout
  for the quantization claim to be the only floating-point noise source.
- κ diagnostics are logged once per epoch via model.diagnose(L) on a
  single representative L. We collect *every* layer's (median, max), not
  just layer 0 — layer 0 can stay tame while layer 3 drifts. The single
  scalar we print and watch is `kappa_max_global = max over layers`.

Returns from `train(...)`:
    result_dict with keys:
        'best_val_acc':       best validation accuracy seen during training
        'best_epoch':         epoch index of that best val
        'final_train_loss':   last epoch's mean train loss
        'history':            list of per-epoch dicts
                              ({epoch, train_loss, val_loss, val_acc, lr_main,
                                kappa_median_all, kappa_max_all,
                                kappa_max_global, epoch_time_s})
        'best_state_dict':    model.state_dict() at best val
"""

import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR

from ..ssm.model import S4DClassifier, make_param_groups


# ----------------------------------------------------------------------
# LR schedule: linear warmup → cosine decay over remaining epochs.
# ----------------------------------------------------------------------

def _cosine_warmup_lambda(total_epochs: int, warmup_epochs: int):
    def lam(epoch: int) -> float:
        if warmup_epochs > 0 and epoch < warmup_epochs:
            return (epoch + 1) / max(1, warmup_epochs)
        progress = (epoch - warmup_epochs) / max(1, total_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return lam


# ----------------------------------------------------------------------
# Evaluation helper (used inside train() for val and exposed for the
# bit-width sweep harness via eval.py).
# ----------------------------------------------------------------------

@torch.no_grad()
def evaluate(model: nn.Module, loader, device, max_batches: int = None) -> dict:
    """Run one pass over `loader`, return {'loss': mean_loss, 'acc': accuracy}."""
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for i, (x, y) in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction='sum')
        total_loss += float(loss.item())
        total_correct += int((logits.argmax(dim=-1) == y).sum().item())
        total_seen += y.numel()
    if total_seen == 0:
        return {'loss': float('nan'), 'acc': float('nan'), 'n': 0}
    return {
        'loss': total_loss / total_seen,
        'acc': total_correct / total_seen,
        'n': total_seen,
    }


# ----------------------------------------------------------------------
# Train one epoch.
# ----------------------------------------------------------------------

def train_one_epoch(model, loader, optimizer, device, grad_clip: float = 1.0,
                    log_every: int = 0, epoch: int = 0):
    model.train()
    total_loss = 0.0
    total_seen = 0
    t0 = time.time()
    for step, (x, y) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = F.cross_entropy(logits, y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip and grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        bs = y.numel()
        total_loss += float(loss.item()) * bs
        total_seen += bs

        if log_every and step > 0 and step % log_every == 0:
            elapsed = time.time() - t0
            print(
                f"  [epoch {epoch:3d} step {step:5d}] "
                f"loss={total_loss / total_seen:.4f} "
                f"({elapsed:.1f}s)"
            )
    mean_loss = total_loss / max(1, total_seen)
    return mean_loss


# ----------------------------------------------------------------------
# Top-level train() entrypoint.
# ----------------------------------------------------------------------

def train(
    model: S4DClassifier,
    train_loader,
    val_loader,
    device,
    *,
    epochs: int = 200,
    lr: float = 4e-3,
    ssm_lr_factor: float = 0.1,
    weight_decay: float = 0.01,
    warmup_epochs: int = 10,
    grad_clip: float = 1.0,
    log_every: int = 0,
    diagnose_each_epoch: bool = True,
    diagnose_L: int = None,
) -> dict:
    """
    Train `model` end-to-end. Returns a result dict (see module docstring).

    Args:
        model:               S4DClassifier (with quantizers already set, if any).
        train_loader, val_loader:  PyTorch DataLoaders.
        device:              torch.device.
        epochs:              total epochs.
        lr:                  base learning rate (applied to the 'other' group).
        ssm_lr_factor:       multiplier for SSM-param LR.
        weight_decay:        WD for the 'other' group (SSM group is always 0).
        warmup_epochs:       linear warmup epochs at start.
        grad_clip:           max grad-norm. 0 or None disables.
        log_every:           per-step logging frequency (0 = silent).
        diagnose_each_epoch: call model.diagnose() once per epoch.
        diagnose_L:          sequence length to pass to diagnose. If None,
                             infer from a single train_loader batch.
    """
    model.to(device)

    # Optimizer with the canonical SSM / other split.
    param_groups = make_param_groups(
        model, lr=lr, ssm_lr_factor=ssm_lr_factor, weight_decay=weight_decay,
    )
    optimizer = AdamW(param_groups)

    # Scheduler shared across both groups (the per-group base LRs are already
    # set; LambdaLR scales them by the same factor each epoch).
    scheduler = LambdaLR(
        optimizer,
        lr_lambda=[_cosine_warmup_lambda(epochs, warmup_epochs)] * len(param_groups),
    )

    # One-time peek for diagnose_L.
    if diagnose_each_epoch and diagnose_L is None:
        try:
            first_x, _ = next(iter(train_loader))
            diagnose_L = first_x.shape[1]   # (B, L, C)
        except Exception:
            diagnose_L = None

    history = []
    best_val_acc = -1.0
    best_epoch = -1
    best_state = None
    final_train_loss = float('nan')

    for epoch in range(epochs):
        epoch_t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, device,
            grad_clip=grad_clip, log_every=log_every, epoch=epoch,
        )
        final_train_loss = train_loss

        val_metrics = evaluate(model, val_loader, device)

        # ---- All-layer κ diagnostics ------------------------------------
        # Logging only layer 0 leaves us blind to a single drifting deep
        # layer. Collect every layer's (median, max) at each epoch, and
        # surface `kappa_max_global = max over layers` as the one-line
        # health scalar. We also keep `kappa_*_layer0` for backward
        # compatibility with anything that wants layer-0 explicitly.
        kappa_medians, kappa_maxes = [], []
        kappa_median_layer0 = float('nan')
        kappa_max_layer0 = float('nan')
        kappa_max_global = float('nan')
        if diagnose_each_epoch and diagnose_L is not None:
            try:
                diag = model.diagnose(diagnose_L)
                if diag:
                    kappa_medians = [d['kappa_median'] for d in diag]
                    kappa_maxes = [d['kappa_max'] for d in diag]
                    kappa_median_layer0 = kappa_medians[0]
                    kappa_max_layer0 = kappa_maxes[0]
                    kappa_max_global = max(kappa_maxes)
            except Exception:
                # Diagnostics must never kill training.
                kappa_medians, kappa_maxes = [], []
                kappa_median_layer0 = float('nan')
                kappa_max_layer0 = float('nan')
                kappa_max_global = float('nan')

        lr_main = optimizer.param_groups[1]['lr']
        history.append({
            'epoch': epoch,
            'train_loss': train_loss,
            'val_loss': val_metrics['loss'],
            'val_acc': val_metrics['acc'],
            'lr_main': lr_main,
            'kappa_median_layer0': kappa_median_layer0,
            'kappa_max_layer0': kappa_max_layer0,
            'kappa_median_all': kappa_medians,
            'kappa_max_all': kappa_maxes,
            'kappa_max_global': kappa_max_global,
            'epoch_time_s': time.time() - epoch_t0,
        })

        print(
            f"[epoch {epoch:3d}] "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_metrics['loss']:.4f}  "
            f"val_acc={val_metrics['acc']:.4f}  "
            f"lr={lr_main:.2e}  "
            f"k_max_global={kappa_max_global:.2e}  "
            f"({history[-1]['epoch_time_s']:.1f}s)"
        )

        if val_metrics['acc'] > best_val_acc:
            best_val_acc = val_metrics['acc']
            best_epoch = epoch
            # CPU-side state dict so GPU memory is freed for the next sweep cell.
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        scheduler.step()

    return {
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'final_train_loss': final_train_loss,
        'history': history,
        'best_state_dict': best_state,
    }
