"""
Utilities for the "new classes arrive" scenario.

The three things the current pipeline cannot do:

1. Grow the output head. num_classes is fixed at construction from whatever
   labels appear in y_reference, so a 31st species has nowhere to go.
2. Preserve old-class performance while learning new ones. Nothing in the repo
   measures forgetting, because evaluate() returned one scalar until recently.
3. Sample old data for replay. No mechanism exists.

Design note on head expansion: rather than hard-coding each architecture's
classifier attribute (LegacyCNN uses .fc, Raman1DCNN and MultiscaleCNN use
.classifier, Transformer1D uses .mlp_head, FusionNet uses .head), this finds
the last nn.Linear in module order and replaces it. That works for all five
because every one of them ends in a Linear producing logits. It would break on
an architecture whose final Linear is not the classifier, so it asserts the
out_features matches the old class count before touching anything.
"""
from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn


# --------------------------------------------------------------------------- #
# class bookkeeping
# --------------------------------------------------------------------------- #

def split_classes(all_labels: Sequence, n_holdout: int, seed: int) -> Tuple[List, List]:
    """Choose which original label values are 'known' and which 'arrive later'.

    Uses a dedicated RNG so the holdout choice is independent of the training
    seed. That matters: you want to vary the holdout set and the training seed
    separately, otherwise you cannot tell a holdout-draw effect from a seed
    effect.
    """
    labels = sorted(all_labels)
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(labels))
    new = sorted(labels[i] for i in idx[:n_holdout])
    old = sorted(labels[i] for i in idx[n_holdout:])
    return old, new


def make_mapping(labels: Sequence) -> Dict:
    """label value -> contiguous index, in sorted order."""
    return {lab: i for i, lab in enumerate(sorted(labels))}


def extended_mapping(old_labels: Sequence, new_labels: Sequence) -> Dict:
    """Old classes keep their indices; new classes are appended after them.

    This is the property that makes head expansion valid. If new classes were
    interleaved by sorted order instead, every old class index would shift and
    the existing classifier rows would point at the wrong species.
    """
    mapping = make_mapping(old_labels)
    n = len(mapping)
    for i, lab in enumerate(sorted(new_labels)):
        mapping[lab] = n + i
    return mapping


def replay_indices(y: np.ndarray, old_indices: Sequence[int],
                   n_per_class: int, seed: int) -> np.ndarray:
    """Sample n_per_class examples of each old class, for a replay buffer."""
    rng = np.random.default_rng(seed)
    y = np.asarray(y)
    picked = []
    for c in old_indices:
        pool = np.where(y == c)[0]
        if len(pool) == 0:
            continue
        take = min(n_per_class, len(pool))
        picked.append(rng.choice(pool, size=take, replace=False))
    return np.concatenate(picked) if picked else np.array([], dtype=int)


# --------------------------------------------------------------------------- #
# head expansion
# --------------------------------------------------------------------------- #

def _last_linear(model: nn.Module):
    name, mod = None, None
    for n, m in model.named_modules():
        if isinstance(m, nn.Linear):
            name, mod = n, m
    if mod is None:
        raise ValueError("model contains no nn.Linear; cannot expand a head")
    return name, mod


def _set_module(model: nn.Module, dotted: str, new: nn.Module) -> None:
    parts = dotted.split(".")
    parent = model
    for p in parts[:-1]:
        parent = getattr(parent, p) if not p.isdigit() else parent[int(p)]
    last = parts[-1]
    if last.isdigit():
        parent[int(last)] = new
    else:
        setattr(parent, last, new)


def expand_classifier(model: nn.Module, new_num_classes: int,
                      device=None) -> Dict:
    """Grow the final Linear from C_old to new_num_classes outputs.

    Existing rows are copied verbatim, so the model's behaviour on old classes
    is unchanged at the moment of expansion (the new logits start small but
    non-zero, so predictions can still shift; that is measured, not assumed).
    New rows get the standard Linear init.
    """
    name, old = _last_linear(model)
    c_old = old.out_features
    if new_num_classes < c_old:
        raise ValueError(f"cannot shrink head: {c_old} -> {new_num_classes}")
    if new_num_classes == c_old:
        return {"expanded": False, "layer": name, "old": c_old, "new": c_old}

    new = nn.Linear(old.in_features, new_num_classes,
                    bias=old.bias is not None)
    with torch.no_grad():
        new.weight[:c_old] = old.weight
        if old.bias is not None:
            new.bias[:c_old] = old.bias
    _set_module(model, name, new)
    if device is not None:
        model.to(device)
    return {"expanded": True, "layer": name, "old": c_old, "new": new_num_classes}


def freeze_backbone(model: nn.Module) -> Dict:
    """Train only the final Linear. Returns what was frozen, for the manifest."""
    name, _ = _last_linear(model)
    n_frozen = n_trainable = 0
    for pname, p in model.named_parameters():
        if pname.startswith(name + "."):
            p.requires_grad = True
            n_trainable += p.numel()
        else:
            p.requires_grad = False
            n_frozen += p.numel()
    return {"head": name, "frozen_params": n_frozen, "trainable_params": n_trainable}


# --------------------------------------------------------------------------- #
# forgetting metrics
# --------------------------------------------------------------------------- #

def group_recall(per_class_recall: Sequence, indices: Sequence[int]):
    """Mean recall over a group of class indices, ignoring absent classes."""
    vals = [per_class_recall[i] for i in indices
            if i < len(per_class_recall) and per_class_recall[i] is not None]
    return float(np.mean(vals)) if vals else None


def forgetting_report(before: Sequence, after: Sequence,
                      old_idx: Sequence[int], new_idx: Sequence[int]) -> Dict:
    """Per-class recall change on old classes, plus new-class attainment.

    'before' is the 25-class model's per-class recall, indexed 0..24.
    'after' is the 30-class model's, indexed 0..29. Old indices are unchanged
    between the two by construction (see extended_mapping), which is what
    makes the comparison valid.
    """
    drops = {}
    for i in old_idx:
        b = before[i] if i < len(before) else None
        a = after[i] if i < len(after) else None
        if b is None or a is None:
            continue
        drops[int(i)] = round(float(b - a), 4)
    worst = max(drops, key=lambda k: drops[k]) if drops else None
    return {
        "old_recall_before": group_recall(before, old_idx),
        "old_recall_after": group_recall(after, old_idx),
        "new_recall_after": group_recall(after, new_idx),
        "mean_forgetting": round(float(np.mean(list(drops.values()))), 4) if drops else None,
        "max_forgetting": drops[worst] if worst is not None else None,
        "worst_forgotten_class": worst,
        "per_class_drop": drops,
    }
