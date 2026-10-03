"""The four adaptation arms (spec section 6.1): what each one initialises from,
which parameters it trains, and the learning rate each parameter gets.

    S   random init, everything trained
    LP  A, a new linear head only (A frozen, in eval mode)
    B1  A, head-only warm-up, then everything with layer-wise LR decay
    B2  A, LoRA adapters + norm parameters + head (A frozen)

A run's add-on is the state of its trainable parameters: the head (LP), LoRA
plus norms plus head (B2), or the whole model (S, B1).
"""
from __future__ import annotations

import copy
import hashlib
import re

import torch
import torch.nn as nn

from .model.encoder import PRESETS, SpectralEncoder
from .model.lora import MAX_FRACTION, RANKS, inject_lora
from .model.stem import MaskedGroupNorm

ARMS = ("S", "LP", "B1", "B2")
NORM_TYPES = (nn.LayerNorm, MaskedGroupNorm)


class Classifier(nn.Module):
    """Encoder -> mean-pooled embedding -> linear head."""

    def __init__(self, encoder: SpectralEncoder, n_classes: int):
        super().__init__()
        self.encoder = encoder
        self.head = nn.Linear(encoder.cfg.width, n_classes)

    def forward(self, x, m):
        return self.head(self.encoder.embed(x, m))


def load_a(path):
    """Model A's checkpoint -> (preset, encoder state_dict)."""
    ck = torch.load(path, map_location="cpu", weights_only=True)
    return ck["preset"], ck["encoder"]


def a_fingerprint(path) -> str:
    """SHA-1 hex digest of model A's file. Recorded per run and checked before
    the test pass: LP/B1/B2 add-ons only mean something on top of this exact A."""
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def choose_rank(encoder: SpectralEncoder, n_classes: int):
    """Largest rank in RANKS whose B2 model trains <= MAX_FRACTION of its
    parameters. Returns (rank, fraction)."""
    for r in RANKS:
        m = Classifier(copy.deepcopy(encoder), n_classes)
        inject_lora(m.encoder, r)
        set_trainable(m, "lora")
        frac = trainable_count(m) / total_count(m)
        if frac <= MAX_FRACTION:
            return r, frac
    raise ValueError(f"no LoRA rank in {RANKS} keeps the trainable fraction <= {MAX_FRACTION}")


def build_model(arm, n_classes, preset, a_state=None, seed=0, lora_rank=None):
    """Seeded construction. Arms other than S start from A's weights; B2 then
    gets LoRA (rank from choose_rank unless given). Returns (model, rank)."""
    if arm not in ARMS:
        raise ValueError(f"unknown arm {arm!r}; expected one of {ARMS}")
    if arm != "S" and a_state is None:
        raise ValueError(f"arm {arm} needs model A's weights")
    torch.manual_seed(seed)
    enc = SpectralEncoder(PRESETS[preset])
    if arm != "S":
        enc.load_state_dict(a_state)
    rank = None
    if arm == "B2":
        rank = lora_rank or choose_rank(enc, n_classes)[0]
    model = Classifier(enc, n_classes)
    if arm == "B2":
        inject_lora(model.encoder, rank)
    return model, rank


def set_trainable(model: Classifier, phase: str):
    """phase 'head': head only. 'full': everything. 'lora': LoRA + norms + head."""
    for p in model.parameters():
        p.requires_grad_(phase == "full")
    if phase == "full":
        return
    if phase not in ("head", "lora"):
        raise ValueError(f"unknown phase {phase!r}")
    for p in model.head.parameters():
        p.requires_grad_(True)
    if phase == "lora":
        for name, p in model.named_parameters():
            if name.endswith(("lora_a", "lora_b")):
                p.requires_grad_(True)
        for mod in model.modules():
            if isinstance(mod, NORM_TYPES):
                for p in mod.parameters():
                    p.requires_grad_(True)


def phases(arm):
    """[(phase, is_warmup)] in training order for the FIRST stage of a task;
    later stages (T1 stage ii) use only the last entry."""
    return {"S": [("full", False)], "LP": [("head", False)],
            "B1": [("head", True), ("full", False)], "B2": [("lora", False)]}[arm]


def trainable_count(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def total_count(model):
    return sum(p.numel() for p in model.parameters())


_BLOCK = re.compile(r"^encoder\.blocks\.(\d+)\.")


def lr_scale(name, n_layers, decay):
    """Layer-wise LR decay: head and final norm x1, block i x decay^(L-i),
    stem and patch projection x decay^(L+1) (the smallest)."""
    if name.startswith("head.") or name.startswith("encoder.norm."):
        return 1.0
    m = _BLOCK.match(name)
    if m:
        return decay ** (n_layers - int(m.group(1)))
    return decay ** (n_layers + 1)


def param_groups(model, lr, weight_decay, llrd=None):
    """AdamW groups over trainable parameters. Matrices get weight decay,
    vectors (biases, norms) do not. With llrd, each parameter's LR is
    lr * lr_scale(name)."""
    groups = {}
    n_layers = model.encoder.cfg.layers
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        scale = lr_scale(name, n_layers, llrd) if llrd else 1.0
        wd = weight_decay if p.ndim >= 2 else 0.0
        g = groups.setdefault((scale, wd), {"params": [], "lr": lr * scale, "weight_decay": wd})
        g["params"].append(p)
    return list(groups.values())


def addon_state(model):
    """The trainable parameters (what this run adds on top of A), on the CPU."""
    return {n: p.detach().cpu().clone() for n, p in model.named_parameters() if p.requires_grad}


def load_addon(model, state):
    """Load an add-on into a model built the same way (build_model with A's
    weights). Every key must be one of the model's parameters."""
    names = dict(model.named_parameters())
    unknown = sorted(set(state) - set(names))
    if unknown:
        raise KeyError(f"add-on has parameters the model does not: {unknown[:5]}")
    with torch.no_grad():
        for k, v in state.items():
            if names[k].shape != v.shape:
                raise ValueError(f"shape mismatch for {k}: {tuple(v.shape)} vs "
                                 f"{tuple(names[k].shape)}")
            names[k].copy_(v)
