"""LoRA adapters for the encoder's Linear layers (spec section 6.1, arm B2).

inject_lora() swaps each target nn.Linear for a LoRALinear that keeps the
original layer frozen as `.base` and adds a low-rank update scaled by
alpha / r. B starts at zero, so a freshly injected model computes exactly
what A computed. Load A's weights BEFORE injecting: injection renames
`...q.weight` to `...q.base.weight`.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

TARGETS = ("q", "k", "v", "o", "fc1", "fc2")
RANKS = (8, 4, 2)                 # spec: the largest that keeps trainable <= MAX_FRACTION
MAX_FRACTION = 0.05


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r, self.scale = r, alpha / r
        self.lora_a = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        update = (self.drop(x) @ self.lora_a.t().to(x.dtype)) @ self.lora_b.t().to(x.dtype)
        return self.base(x) + self.scale * update


def inject_lora(module: nn.Module, r: int, dropout: float = 0.05, targets=TARGETS):
    """Replace every nn.Linear whose attribute name is in `targets` with a
    LoRALinear (alpha = 2r). Returns the dotted names that were replaced."""
    replaced = []
    for parent_name, parent in list(module.named_modules()):
        for name, child in list(parent.named_children()):
            if name in targets and type(child) is nn.Linear:
                setattr(parent, name, LoRALinear(child, r, alpha=2 * r, dropout=dropout))
                replaced.append(f"{parent_name}.{name}" if parent_name else name)
    if not replaced:
        raise ValueError(f"no nn.Linear named any of {targets} found")
    return replaced


def lora_parameter_count(module: nn.Module) -> int:
    return sum(p.numel() for n, p in module.named_parameters()
               if n.endswith(("lora_a", "lora_b")))
