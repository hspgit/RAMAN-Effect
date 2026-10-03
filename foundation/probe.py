"""Probes on frozen embeddings: the "is A done / good?" signal (spec section 5.4).

Uses the pretraining corpus only (its held-out probe slice and a fixed sample
of pretrain rows); our Bacteria-ID data never appears here.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

KNN_K = 20
SOURCE_SAMPLE_PER_SOURCE = 300


@dataclass
class ProbeTask:
    name: str
    train_idx: np.ndarray
    test_idx: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray


@dataclass
class ProbeSet:
    tasks: list
    sample_idx: np.ndarray        # fixed pretrain rows for RankMe and the source-ID probe
    sample_source: np.ndarray


def build_probe_set(meta, seed=0, per_source=SOURCE_SAMPLE_PER_SOURCE):
    """One task per probe source; a fixed, source-balanced pretrain sample."""
    split = meta["split"].to_numpy()
    source = meta["source"].to_numpy(dtype=object)
    label = meta["label"].to_numpy(dtype=object)
    tasks = []
    for src in sorted(set(source[split != "pretrain"])):
        tr = np.flatnonzero((source == src) & (split == "probe_train"))
        te = np.flatnonzero((source == src) & (split == "probe_test"))
        tasks.append(ProbeTask(src, tr, te, label[tr].astype(str), label[te].astype(str)))
    rng = np.random.default_rng(seed)
    pre = np.flatnonzero(split == "pretrain")
    picks = []
    for src in sorted(set(source[pre])):
        rows = pre[source[pre] == src]
        picks.append(rng.choice(rows, size=min(per_source, rows.size), replace=False))
    sample = np.sort(np.concatenate(picks))
    return ProbeSet(tasks, sample, source[sample].astype(str))


@torch.no_grad()
def embed(encoder, X, M, idx, device, batch=1024):
    encoder.eval()
    out = []
    for s in range(0, len(idx), batch):
        rows = idx[s:s + batch]
        x = torch.as_tensor(np.asarray(X[rows], dtype=np.float32), device=device)
        m = torch.as_tensor(np.asarray(M[rows]), device=device)
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out.append(encoder.embed(x, m).float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, encoder.cfg.width), np.float32)


def knn_f1(tr, y_tr, te, y_te, k=KNN_K):
    # Distance weighting keeps k=20 meaningful when a class has only a few
    # training examples (RRUFF: ~3 per mineral species).
    clf = KNeighborsClassifier(n_neighbors=min(k, len(y_tr)), metric="cosine",
                               weights="distance").fit(tr, y_tr)
    return float(f1_score(y_te, clf.predict(te), average="macro",
                          labels=np.unique(y_te), zero_division=0))


def logreg_f1(tr, y_tr, te, y_te):
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(tr, y_tr)
    return float(f1_score(y_te, clf.predict(te), average="macro",
                          labels=np.unique(y_te), zero_division=0))


def rankme(emb, eps=1e-7):
    """Effective rank of the embedding matrix (Garrido et al. 2023)."""
    s = np.linalg.svd(np.asarray(emb, dtype=np.float64), compute_uv=False)
    p = s / s.sum() + eps
    return float(np.exp(-(p * np.log(p)).sum()))


def source_id_bacc(emb, sources, seed=0):
    """Balanced accuracy of a linear probe predicting the source. High values
    while the chemistry probes stay flat mean A encodes instrument/dataset quirks."""
    keep = np.isin(sources, [s for s in np.unique(sources) if (sources == s).sum() >= 4])
    if len(np.unique(sources[keep])) < 2:
        return float("nan")
    tr, te, ytr, yte = train_test_split(emb[keep], sources[keep], test_size=0.5,
                                        stratify=sources[keep], random_state=seed)
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(tr, ytr)
    return float(balanced_accuracy_score(yte, clf.predict(te)))


def evaluate(encoder, X, M, probe_set, device):
    per_task = {}
    for t in probe_set.tasks:
        tr = embed(encoder, X, M, t.train_idx, device)
        te = embed(encoder, X, M, t.test_idx, device)
        per_task[t.name] = {"knn_f1": knn_f1(tr, t.y_train, te, t.y_test),
                            "logreg_f1": logreg_f1(tr, t.y_train, te, t.y_test)}
    sample = embed(encoder, X, M, probe_set.sample_idx, device)
    knn = float(np.mean([v["knn_f1"] for v in per_task.values()]))
    lr = float(np.mean([v["logreg_f1"] for v in per_task.values()]))
    return {"probe_score": (knn + lr) / 2, "knn_score": knn, "logreg_score": lr,
            "per_task": per_task, "rankme": rankme(sample),
            "source_id_bacc": source_id_bacc(sample, probe_set.sample_source)}


def raw_knn_score(X, M, probe_set):
    """G1 baseline: k-NN on the raw normalised spectra of the same probe slice."""
    scores = []
    for t in probe_set.tasks:
        raw = lambda idx: np.where(M[idx], np.asarray(X[idx], np.float32), 0.0)
        scores.append(knn_f1(raw(t.train_idx), t.y_train, raw(t.test_idx), t.y_test))
    return float(np.mean(scores))
