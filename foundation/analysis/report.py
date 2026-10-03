"""Understand what a pretraining run learned (spec section 7, phase 1).

    python -m foundation.analysis.report RUN_DIR [RUN_DIR ...] [--out report.md]

For each run, writes RUN_DIR/analysis/ with:
  curves.png      loss, probe scores vs the raw-spectra k-NN baseline, RankMe, source-ID
  recon.png       reconstructions of 4 fixed probe spectra at the best epoch
  embedding.png   2-D PCA of the best checkpoint's embeddings, by domain and by source
  summary.json
Then prints one comparison table, the size recommendation (highest best probe
score; a smaller preset wins ties within 0.01) and gate G1 for every run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402

from ..data.harmonize import GRID  # noqa: E402
from ..model.encoder import PATCH, PRESETS, SpectralEncoder  # noqa: E402
from ..pretrain import Corpus  # noqa: E402
from ..probe import build_probe_set, embed  # noqa: E402

SIZE_ORDER = ["tiny", "small", "base"]
TIE = 0.01


def read_metrics(run_dir):
    epochs, probes, end = [], [], None
    for line in (Path(run_dir) / "metrics.jsonl").read_text().splitlines():
        if not line.strip():  # Skip blank lines
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue  # Skip lines that can't be parsed (e.g., truncated)
        if "loss" in rec:
            epochs.append(rec)
        elif "probe" in rec:
            probes.append(rec)
        elif "end" in rec:
            end = rec
    return epochs, probes, end


def summarize(run_dir):
    run_dir = Path(run_dir)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    manifest = json.loads((run_dir / "manifest.json").read_text())
    epochs, probes, end = read_metrics(run_dir)
    # Best epoch: last probe with "improved": true, or max probe_score if none
    best = None
    for p in reversed(probes):
        if p.get("improved", False):
            best = p
            break
    if best is None and probes:
        best = max(probes, key=lambda p: p["probe"]["probe_score"])
    bp = best["probe"] if best else {}
    raw = manifest["raw_knn_score"]
    return {
        "run": run_dir.name, "preset": cfg["preset"],
        "encoder_params": manifest["encoder_params"],
        "epochs_run": epochs[-1]["epoch"] if epochs else 0,
        "hours": sum(e["seconds"] for e in epochs) / 3600,
        "best_epoch": best["epoch"] if best else None,
        "probe_score": bp.get("probe_score"), "knn_score": bp.get("knn_score"),
        "logreg_score": bp.get("logreg_score"), "per_task": bp.get("per_task"),
        "rankme": bp.get("rankme"), "source_id_bacc": bp.get("source_id_bacc"),
        "raw_knn_score": raw,
        "g1_pass": bool(best) and bp["knn_score"] > raw,
        "end": end["end"] if end else "running or interrupted",
    }


def recommend(summaries):
    """Highest best probe score; a smaller preset wins if within TIE of the best."""
    done = [s for s in summaries if s["probe_score"] is not None]
    if not done:
        return None
    top = max(s["probe_score"] for s in done)
    close = [s for s in done if s["probe_score"] >= top - TIE]
    return min(close, key=lambda s: SIZE_ORDER.index(s["preset"]))


def plot_curves(run_dir, out):
    epochs, probes, _ = read_metrics(run_dir)
    raw = json.loads((Path(run_dir) / "manifest.json").read_text())["raw_knn_score"]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot([e["epoch"] for e in epochs], [e["loss"] for e in epochs])
    ax[0].set(title="reconstruction loss", xlabel="epoch")
    pe = [p["epoch"] for p in probes]
    for key in ("probe_score", "knn_score", "logreg_score"):
        ax[1].plot(pe, [p["probe"][key] for p in probes], marker="o", label=key)
    ax[1].axhline(raw, ls="--", c="k", label="raw-spectra k-NN (G1 bar)")
    ax[1].set(title="probe macro-F1", xlabel="epoch", ylim=(0, 1))
    ax[1].legend(fontsize=8)
    ax[2].plot(pe, [p["probe"]["rankme"] for p in probes], marker="o", c="tab:green")
    ax[2].set(title="RankMe (centred) green / source-ID bacc red", xlabel="epoch")
    tw = ax[2].twinx()
    tw.plot(pe, [p["probe"]["source_id_bacc"] for p in probes], marker="s", c="tab:red")
    tw.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def recon_stats(r, i):
    """(hidden MSE, predict-zero MSE) in per-patch normalised space over the valid points
    of spectrum i's hidden tokens; None for an old npz without the normalised arrays."""
    if "pred_normalised" not in r or "target_normalised" not in r:
        return None
    pm = patchify_np(r["mask"][i]) & r["hidden_tokens"][i][:, None]
    if not pm.any():
        return None
    tgt, pred = r["target_normalised"][i], r["pred_normalised"][i]
    return float(((pred - tgt) ** 2)[pm].mean()), float((tgt ** 2)[pm].mean())


def patchify_np(a):
    n_tok = -(-a.shape[0] // PATCH)
    return np.pad(a, (0, n_tok * PATCH - a.shape[0])).reshape(n_tok, PATCH)


def plot_recon(npz_path, out, n=4):
    r = np.load(npz_path)
    picks = np.linspace(0, len(r["clean"]) - 1, n).astype(int)   # spread over probe tasks
    fig, axes = plt.subplots(n, 1, figsize=(12, 2.8 * n), sharex=True)
    for i, ax in zip(picks, np.atleast_1d(axes)):
        m = r["mask"][i]
        ax.plot(GRID[m], r["clean"][i][m], c="k", lw=1, label="clean target")
        cm = r["corrupted_mask"][i]
        ax.plot(GRID[cm], r["corrupted"][i][cm], c="tab:gray", lw=0.6, alpha=0.7, label="corrupted input")
        hid = np.repeat(r["hidden_tokens"][i], PATCH)[:GRID.size] & m
        ax.plot(GRID, np.where(hid, r["reconstruction"][i], np.nan), c="tab:red", lw=1,
                label="reconstruction (hidden; patch mean/scale from target)")
        # What predicting 0 in normalised space gives after de-normalising: each hidden
        # patch's mean, a staircase. The reconstruction has to beat this to have learned anything.
        clean, pm = patchify_np(r["clean"][i].astype(np.float64)), patchify_np(m)
        mean = (clean * pm).sum(-1) / np.maximum(pm.sum(-1), 1)
        stair = np.repeat(mean, PATCH)[:GRID.size]
        ax.plot(GRID, np.where(hid, stair, np.nan), c="tab:gray", ls=":", lw=1.2,
                label="predict-zero baseline")
        stats = recon_stats(r, i)
        if stats is not None:
            ax.set_title(f"hidden MSE {stats[0]:.3f} vs predict-zero {stats[1]:.3f}", fontsize=9)
        for j in np.flatnonzero(r["hidden_tokens"][i]):
            lo = GRID[min(j * PATCH, GRID.size - 1)]
            ax.axvspan(lo, lo + PATCH * (GRID[1] - GRID[0]), color="tab:red", alpha=0.06, lw=0)
    np.atleast_1d(axes)[0].legend(fontsize=8, loc="upper right")
    np.atleast_1d(axes)[-1].set_xlabel("Raman shift (cm$^{-1}$)")
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def plot_embedding(run_dir, corpus, out, device):
    cfg = yaml.safe_load((Path(run_dir) / "config.yaml").read_text())
    seed = cfg.get("seed", 0)
    ck = torch.load(Path(run_dir) / "ckpt_best.pt", map_location=device, weights_only=True)
    enc = SpectralEncoder(PRESETS[ck["preset"]]).to(device)
    enc.load_state_dict(ck["encoder"])
    ps = build_probe_set(corpus.meta, seed=seed)
    emb = embed(enc, corpus.X, corpus.M, ps.sample_idx, device)
    emb = emb - emb.mean(axis=0)
    xy = emb @ np.linalg.svd(emb, full_matrices=False)[2][:2].T
    domain = corpus.meta["domain"].to_numpy(dtype=object)[ps.sample_idx]
    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    for k, (title, groups) in enumerate((("domain", domain), ("source", ps.sample_source))):
        for g in sorted(set(groups)):
            sel = groups == g
            ax[k].scatter(xy[sel, 0], xy[sel, 1], s=4, label=g, alpha=0.6)
        ax[k].set(title=f"PCA of embeddings by {title}")
        ax[k].legend(fontsize=6, markerscale=3, ncol=2 if k else 1)
    fig.tight_layout()
    fig.savefig(out, dpi=110)
    plt.close(fig)


def report_run(run_dir, corpus=None, device=None):
    run_dir = Path(run_dir)
    out = run_dir / "analysis"
    out.mkdir(exist_ok=True)
    s = summarize(run_dir)
    plot_curves(run_dir, out / "curves.png")
    if s["best_epoch"] is not None:
        rec = run_dir / f"recon_epoch{s['best_epoch']:04d}.npz"
        if rec.exists():
            plot_recon(rec, out / "recon.png")
        if corpus is not None:
            plot_embedding(run_dir, corpus, out / "embedding.png", device)
    (out / "summary.json").write_text(json.dumps(s, indent=2))
    return s


def table(summaries):
    head = ("| run | preset | params | epochs | hours | best epoch | probe | k-NN | logreg | "
            "raw k-NN | G1 | RankMe | source-ID | end |")
    rows = [head, "|" + "---|" * 14]
    f = lambda v, d=3: "-" if v is None else f"{v:.{d}f}"
    for s in summaries:
        rows.append(f"| {s['run']} | {s['preset']} | {s['encoder_params'] / 1e6:.2f}M | "
                    f"{s['epochs_run']} | {s['hours']:.1f} | {s['best_epoch']} | "
                    f"{f(s['probe_score'])} | {f(s['knn_score'])} | {f(s['logreg_score'])} | "
                    f"{f(s['raw_knn_score'])} | {'PASS' if s['g1_pass'] else 'FAIL'} | "
                    f"{f(s['rankme'], 1)} | {f(s['source_id_bacc'])} | {s['end']} |")
    return "\n".join(rows)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out")
    ap.add_argument("--no-embedding", action="store_true", help="skip the PCA plot (no GPU work)")
    args = ap.parse_args(argv)
    cfg = yaml.safe_load((Path(args.runs[0]) / "config.yaml").read_text())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    corpus = None if args.no_embedding else Corpus(cfg["corpus_dir"])
    summaries = [report_run(r, corpus, device) for r in args.runs]
    text = table(summaries)
    rec = recommend(summaries)
    text += ("\n\nRecommended size (probe score only; smaller wins ties within "
             f"{TIE}): {rec['preset']} ({rec['run']})" if rec else "\n\nNo finished probe yet.")
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n")


if __name__ == "__main__":
    main()
