"""
Classification metrics beyond a single accuracy scalar.

Why this is Stage 0 and not Stage 3: the deliverable is a system that retrains
when new data arrives, and the characteristic failure of retraining is
per-class. Overall accuracy can hold steady while six classes collapse. The
current evaluate() returns (loss, accuracy) and nothing else, so that failure
is invisible. You cannot build against a failure mode you cannot see.

Second reason: §10 of RESEARCH_REFERENCE.md shows error is concentrated, not
uniform. Class 2 sits at 12% recall while class 1 sits at 100%. Accuracy
averages that away.

Pure numpy, no sklearn import, so this is cheap to call every epoch.

Convention: macro-F1 here is the unweighted mean of per-class F1 over classes
with non-zero support. Classes absent from y_true are excluded rather than
counted as zero, since counting them as zero would silently penalise any
evaluation on a label subset (the clinical splits, or a held-out-class
experiment).
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence

import numpy as np


def confusion(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """Rows = true class, columns = predicted class."""
    y_true = np.asarray(y_true).astype(np.int64).ravel()
    y_pred = np.asarray(y_pred).astype(np.int64).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    idx = y_true * num_classes + y_pred
    cm = np.bincount(idx, minlength=num_classes * num_classes)
    return cm.reshape(num_classes, num_classes)


def summarize(cm: np.ndarray) -> Dict[str, Any]:
    """Derive every scalar and per-class metric from a confusion matrix."""
    cm = np.asarray(cm, dtype=np.float64)
    support = cm.sum(axis=1)
    predicted = cm.sum(axis=0)
    tp = np.diag(cm)
    total = cm.sum()

    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, tp / support, np.nan)
        precision = np.where(predicted > 0, tp / predicted, np.nan)
        f1 = np.where(
            (precision + recall) > 0,
            2 * precision * recall / (precision + recall),
            0.0,
        )
    present = support > 0
    f1 = np.where(present, f1, np.nan)

    return {
        "accuracy": float(tp.sum() / total) if total else float("nan"),
        "macro_recall": float(np.nanmean(recall)) if present.any() else float("nan"),
        "macro_precision": float(np.nanmean(precision)) if present.any() else float("nan"),
        "macro_f1": float(np.nanmean(f1)) if present.any() else float("nan"),
        "n": int(total),
        "n_classes_present": int(present.sum()),
        "per_class_recall": [None if np.isnan(v) else round(float(v), 4) for v in recall],
        "per_class_precision": [None if np.isnan(v) else round(float(v), 4) for v in precision],
        "per_class_f1": [None if np.isnan(v) else round(float(v), 4) for v in f1],
        "support": [int(v) for v in support],
    }


def worst_classes(stats: Dict[str, Any], k: int = 6) -> Sequence[Dict[str, Any]]:
    """The k lowest-recall classes, which is where your error actually lives."""
    rows = [
        {"class": i, "recall": r, "support": stats["support"][i]}
        for i, r in enumerate(stats["per_class_recall"])
        if r is not None
    ]
    rows.sort(key=lambda d: d["recall"])
    return rows[:k]


def format_summary(stats: Dict[str, Any], phase: str = "Test", k: int = 6) -> str:
    worst = ", ".join(f"{d['class']}:{d['recall']:.2f}" for d in worst_classes(stats, k))
    return (
        f"{phase}: acc {100 * stats['accuracy']:.2f}%  "
        f"macro-F1 {stats['macro_f1']:.4f}  "
        f"macro-recall {stats['macro_recall']:.4f}  "
        f"(n={stats['n']}, classes={stats['n_classes_present']})\n"
        f"{phase}: worst-{k} recall -> {worst}"
    )


def forgetting(
    before: Sequence[Optional[float]],
    after: Sequence[Optional[float]],
    class_ids: Optional[Sequence[int]] = None,
) -> Dict[str, Any]:
    """Per-class recall drop between two evaluations.

    Stage 2 hook, included now because it is six lines and because it defines
    what "accommodated the new data without breaking" has to mean. Positive
    drop = forgetting.
    """
    ids = list(class_ids) if class_ids is not None else list(range(len(before)))
    drops = {}
    for i in ids:
        b, a = before[i], after[i]
        if b is None or a is None:
            continue
        drops[i] = round(float(b - a), 4)
    if not drops:
        return {"per_class_drop": {}, "mean_drop": None, "max_drop": None, "worst_class": None}
    worst = max(drops, key=lambda k: drops[k])
    return {
        "per_class_drop": drops,
        "mean_drop": round(float(np.mean(list(drops.values()))), 4),
        "max_drop": drops[worst],
        "worst_class": worst,
    }
