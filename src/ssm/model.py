"""
S4D sequence classifier.

Top-level model used for sCIFAR / sMNIST classification in Phase 2a
(goals.md §6.1). Architecture follows the canonical S4 / LRA recipe:

    x: (B, L, in_channels)
        |
        v
    Linear(in_channels -> d_model)            # input embedding
        |
        v
    [ S4Block ] * n_layers                    # stacked S4D residual blocks
        |
        v
    mean over L                                # sequence pooling
        |
        v
    Linear(d_model -> num_classes)            # classifier head

The model itself owns only the embedding, the stack of S4Blocks, and the
classifier head. All quantization machinery lives inside the S4D layer
(via its q_alpha / q_omega submodules). This keeps the classifier wrapper
independent of the QAT details and makes it easy to swap quantizers at
sweep time by walking the stack and replacing per-S4D quantizer modules.

Design choices
--------------
- *Mean pooling.* Standard for LRA classification.
- *Canonical SSM parameter group.* A_real, A_imag, log_dt, *and*
  B_real/B_imag/C_real/C_imag all go into the low-LR, no-weight-decay
  group, matching Albert Gu's reference S4 codebase. Reason: weight
  decay on B (initialized to all-ones in continuous time) would otherwise
  slowly pull B toward zero, collapsing the kernel via a path that did
  not exist in the canonical setup.
- *No GLU, no in_proj inside S4Block.* See s4d.py:S4Block docstring.
"""

import math
import torch
import torch.nn as nn

from .s4d import S4Block, S4D
from ..qat.quantizers import make_quantizer
from ..qat.sensitivity import _auto_scale, _auto_scale_per_mode


class S4DClassifier(nn.Module):
    """
    Stacked-S4D sequence classifier with mean pooling.

    Args:
        in_channels:    input feature dim (1 for sCIFAR/sMNIST grayscale).
        num_classes:    number of output classes.
        d_model:        channel / hidden dim. Same as H (one SSM per channel).
        d_state:        SSM state dim N (must be even).
        n_layers:       number of stacked S4Blocks.
        init:           initialization key into INIT_REGISTRY. Phase 2a
                        compares 'fout' against 'skew-hippo'.
        dropout:        residual-path dropout inside each S4Block.
        conj_sym:       conjugate-symmetric kernel sum factor of 2.
        pooling:        'mean' (default) or 'last'. 'mean' is the LRA
                        convention; 'last' is provided only as an ablation.

    Inputs:
        x: (B, L, in_channels)  float32

    Outputs:
        logits: (B, num_classes)  float32
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        d_model: int = 128,
        d_state: int = 64,
        n_layers: int = 4,
        init: str = 'fout',
        dropout: float = 0.1,
        conj_sym: bool = True,
        pooling: str = 'mean',
        freeze_log_dt: bool = False,
    ):
        super().__init__()
        if pooling not in ('mean', 'last'):
            raise ValueError(f"unknown pooling '{pooling}'. Use 'mean' or 'last'.")

        self.in_channels = in_channels
        self.num_classes = num_classes
        self.d_model = d_model
        self.d_state = d_state
        self.n_layers = n_layers
        self.init = init
        self.pooling = pooling
        self.freeze_log_dt = freeze_log_dt

        # Input embedding: (in_channels) -> (d_model). Acts on the last axis,
        # keeping channels-last for compatibility with the S4Block I/O.
        self.encoder = nn.Linear(in_channels, d_model)

        self.blocks = nn.ModuleList([
            S4Block(
                d_model=d_model,
                d_state=d_state,
                init=init,
                dropout=dropout,
                conj_sym=conj_sym,
                freeze_log_dt=freeze_log_dt,
            )
            for _ in range(n_layers)
        ])

        # Final norm before pooling (matches the canonical S4 head convention).
        self.norm = nn.LayerNorm(d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    # ---- introspection helpers -----------------------------------------

    def s4d_layers(self):
        """Yield every S4D layer in the stack, in order."""
        for block in self.blocks:
            yield block.s4d

    def kappas_init(self) -> list:
        """List of per-layer kappa(V) at init time."""
        return [layer.kappa for layer in self.s4d_layers()]

    # ---- quantizer wiring ----------------------------------------------

    @torch.no_grad()
    def set_quantizers(
        self,
        bits: int,
        mode: str = 'deterministic',
        per_channel: bool = True,
        headroom: float = 1.05,
    ):
        """
        Replace every S4D layer's q_alpha / q_omega with freshly-built
        quantizers at `bits`. Auto-derives per-tensor or per-mode scales
        from the *current* A_real / A_imag values of each layer (so this
        plays nicely with calling it mid-training or post-load).

        Args:
            bits:        target bit-width. Pass `None` or `32` to revert
                         to fp32 (Identity quantizers).
            mode:        'deterministic' or 'stochastic'.
            per_channel: per-mode scales (recommended) vs per-tensor.
            headroom:    multiplicative safety factor on the observed
                         max-abs of A_real / A_imag.
        """
        for layer in self.s4d_layers():
            if bits is None or bits == 32:
                layer.q_alpha = nn.Identity()
                layer.q_omega = nn.Identity()
                continue

            if per_channel:
                s_a = _auto_scale_per_mode(layer.A_real, headroom=headroom)
                s_w = _auto_scale_per_mode(layer.A_imag, headroom=headroom)
            else:
                s_a = _auto_scale(layer.A_real, headroom=headroom)
                s_w = _auto_scale(layer.A_imag, headroom=headroom)

            q_a = make_quantizer(bits, scale=s_a, mode=mode)
            q_w = make_quantizer(bits, scale=s_w, mode=mode)
            device = layer.A_real.device
            layer.q_alpha = q_a.to(device)
            layer.q_omega = q_w.to(device)

    # ---- forward --------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, in_channels)

        Returns:
            logits: (B, num_classes)
        """
        if x.dim() != 3:
            raise ValueError(
                f"S4DClassifier expects (B, L, C); got shape {tuple(x.shape)}"
            )
        if x.shape[-1] != self.in_channels:
            raise ValueError(
                f"S4DClassifier expects in_channels={self.in_channels}; "
                f"got {x.shape[-1]}"
            )

        x = self.encoder(x)                       # (B, L, d_model)
        for block in self.blocks:
            x = block(x)                          # (B, L, d_model)

        x = self.norm(x)
        if self.pooling == 'mean':
            x = x.mean(dim=1)                     # (B, d_model)
        else:  # 'last'
            x = x[:, -1, :]                       # (B, d_model)

        return self.classifier(x)                 # (B, num_classes)

    # ---- diagnostics ----------------------------------------------------

    @torch.no_grad()
    def diagnose(self, L: int) -> list:
        """
        Return per-layer diagnose() dicts. Use sparingly (once per epoch,
        not per step) — the per-channel runtime κ computation is non-trivial.
        """
        return [layer.diagnose(L) for layer in self.s4d_layers()]

    def extra_repr(self) -> str:
        return (
            f"in_channels={self.in_channels}, num_classes={self.num_classes}, "
            f"d_model={self.d_model}, d_state={self.d_state}, "
            f"n_layers={self.n_layers}, init={self.init}, "
            f"pooling={self.pooling}"
        )


# ----------------------------------------------------------------------
# Parameter-group helper for AdamW.
#
# Canonical S4D convention (Albert Gu's reference codebase):
#   - SSM group: A_real, A_imag, log_dt, B_real, B_imag, C_real, C_imag
#     at lr = base_lr * ssm_lr_factor, weight_decay = 0.
#   - Everything else (encoder, LayerNorm, classifier head, block linears,
#     D skip): standard lr and weight_decay.
#
# Reason for putting B and C in the SSM group: weight decay on B (which
# is initialized to ones in continuous time) would slowly pull it toward
# zero, eventually collapsing the kernel. The canonical recipe avoids
# this entirely, and any deviation should be a deliberate, labeled
# ablation rather than a side effect of how the param split is written.
# ----------------------------------------------------------------------

def make_param_groups(model: nn.Module, lr: float, ssm_lr_factor: float = 0.1,
                      weight_decay: float = 0.0):
    """
    Split params into (ssm_params, other_params) parameter groups.

    ssm_params: A_real, A_imag, log_dt, B_real, B_imag, C_real, C_imag
                from every S4D layer. Trained at lr * ssm_lr_factor with
                weight_decay = 0.

    other_params: everything else (encoder, LayerNorm gains, classifier
                  head, block linears, D skip scalar). Trained at lr with
                  the supplied weight_decay.

    D (the SSM skip scalar) is treated as 'other'; it's not part of the
    diagonal eigenvalue parameterization the κ analysis depends on, and
    canonical configs apply normal weight decay to it.
    """
    ssm_param_ids = set()
    ssm_params = []
    for m in model.modules():
        if isinstance(m, S4D):
            for p in (m.A_real, m.A_imag, m.log_dt,
                      m.B_real, m.B_imag, m.C_real, m.C_imag):
                if id(p) not in ssm_param_ids:
                    ssm_param_ids.add(id(p))
                    ssm_params.append(p)

    other_params = [
        p for p in model.parameters()
        if p.requires_grad and id(p) not in ssm_param_ids
    ]

    return [
        {'params': ssm_params,   'lr': lr * ssm_lr_factor, 'weight_decay': 0.0},
        {'params': other_params, 'lr': lr,                 'weight_decay': weight_decay},
    ]


def smoke_test():
    """Structural smoke test for the classifier model and its helpers."""
    torch.manual_seed(0)
    B, L, in_c, n_classes = 4, 64, 1, 10
    model = S4DClassifier(
        in_channels=in_c, num_classes=n_classes,
        d_model=16, d_state=32, n_layers=2, init='fout', dropout=0.1,
    )
    x = torch.randn(B, L, in_c)

    # fp32 forward
    logits = model(x)
    assert logits.shape == (B, n_classes), f"bad shape: {logits.shape}"
    assert torch.isfinite(logits).all()
    print(f"fp32 forward ok. kappas_init = {model.kappas_init()}")

    # Backward
    loss = logits.mean()
    loss.backward()
    for p in model.parameters():
        if p.grad is not None:
            assert torch.isfinite(p.grad).all()
    print("fp32 backward ok, all grads finite")

    # Param groups
    groups = make_param_groups(model, lr=1e-3, ssm_lr_factor=0.1,
                               weight_decay=0.01)
    n_ssm = sum(p.numel() for p in groups[0]['params'])
    n_other = sum(p.numel() for p in groups[1]['params'])
    assert n_ssm > 0 and n_other > 0
    assert groups[0]['weight_decay'] == 0.0
    assert math.isclose(groups[0]['lr'], 1e-4)
    # Sanity-check that B/C are in the SSM group, not 'other'.
    ssm_ids = {id(p) for p in groups[0]['params']}
    for layer in model.s4d_layers():
        for p in (layer.B_real, layer.B_imag, layer.C_real, layer.C_imag):
            assert id(p) in ssm_ids, "B/C must be in the SSM (low-LR, no-WD) group"
    print(f"param groups: ssm={n_ssm}, other={n_other} (B/C verified in ssm group)")

    # Swap in quantizers and re-run.
    model.set_quantizers(bits=4, mode='deterministic', per_channel=True)
    logits_q = model(x)
    assert torch.isfinite(logits_q).all()
    print("4-bit quantized forward ok")

    # diagnose()
    diag = model.diagnose(L)
    assert isinstance(diag, list) and len(diag) == 2
    for d in diag:
        for k in ('valid', 'msg', 'kappa_init', 'kappa_median', 'kappa_max'):
            assert k in d
    print(f"diagnose() ok, layer-0 kappa_max = {diag[0]['kappa_max']:.3g}")

    # Revert to fp32.
    model.set_quantizers(bits=None)
    for layer in model.s4d_layers():
        assert isinstance(layer.q_alpha, nn.Identity)
        assert isinstance(layer.q_omega, nn.Identity)
    print("fp32 revert ok")

    print("All S4DClassifier smoke tests passed!")


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
    smoke_test()
