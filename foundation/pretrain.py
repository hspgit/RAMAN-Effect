"""Masked-spectrum pretraining of model A (spec sections 5.2-5.4).

    python -m foundation.pretrain --config foundation/configs/pretrain_trial_small.yaml
    python -m foundation.pretrain --resume foundation/runs/pretrain/<run> --max-epochs 300

Runs end only at max_epochs or when the probe score has not improved for
`patience` checkpoints; never on elapsed time. Our Bacteria-ID data is never
loaded here.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import subprocess
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

from .data.corrupt import PRESETS as CORRUPTIONS
from .data.corrupt import corrupt
from .model.decoder import TARGET_EPS, MAEDecoder, mae_loss, patchify
from .model.encoder import PRESETS, SpectralEncoder, count_parameters
from .model.masking import block_mask, visible_order
from .probe import build_probe_set, evaluate, raw_knn_score

DEFAULTS = {
    "tag": "run", "preset": "small", "corpus_dir": "foundation/data/corpus_v1",
    "run_root": "foundation/runs/pretrain", "seed": 0, "device": "cuda",
    "batch_size": 512, "lr": 3.0e-4, "weight_decay": 0.05, "betas": [0.9, 0.95],
    "warmup_epochs": 10, "max_epochs": 200, "grad_clip": 1.0,
    "mask_ratio": 0.5, "mask_block": [2, 6], "deriv_weight": 0.5,
    "corruption": "pretrain", "probe_every": 10, "patience": 3, "amp": True,
    "n_recon": 16, "steps_per_epoch": None, "max_nonfinite_steps": 20,
}


def load_config(path=None, **overrides):
    cfg = dict(DEFAULTS)
    if path:
        cfg.update(yaml.safe_load(Path(path).read_text()) or {})
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


class Corpus:
    """corpus_v1 artefacts held in RAM; only pretrain rows are ever sampled."""

    def __init__(self, corpus_dir):
        d = Path(corpus_dir)
        self.X = np.load(d / "spectra.npy")                 # float16 (N, 1001)
        self.M = np.load(d / "mask.npy")                    # bool (N, 1001)
        self.meta = pd.read_parquet(d / "meta.parquet")
        split = self.meta["split"].to_numpy()
        self.pretrain_idx = np.flatnonzero(split == "pretrain")
        w = self.meta["sample_weight"].to_numpy()[self.pretrain_idx]
        self.weights = torch.as_tensor(w / w.sum(), dtype=torch.float64)
        self.Xt = torch.from_numpy(self.X)
        self.Mt = torch.from_numpy(self.M)
        self.manifest_version = json.loads((d / "manifest.json").read_text()).get("version")

    def sample(self, n, generator, device):
        pick = torch.multinomial(self.weights, n, replacement=True, generator=generator)
        rows = torch.as_tensor(self.pretrain_idx)[pick]
        return (self.Xt[rows].to(device, non_blocking=True).float(),
                self.Mt[rows].to(device, non_blocking=True))


def _git_sha():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True,
                                       stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def _save_atomic(obj, path):
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _fmt_secs(s):
    s = int(s)
    h, m = divmod(s // 60, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s % 60:02d}s"


def lr_factor(step, warmup_steps, total_steps):
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))


def build_models(cfg, device):
    enc = SpectralEncoder(PRESETS[cfg["preset"]]).to(device)
    dec = MAEDecoder(enc.cfg.width).to(device)
    return enc, dec


def make_optimizer(cfg, enc, dec):
    decay, no_decay = [], []
    for name, p in list(enc.named_parameters()) + list(dec.named_parameters()):
        (decay if p.ndim >= 2 and "mask_token" not in name else no_decay).append(p)
    return torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                              {"params": no_decay, "weight_decay": 0.0}],
                             lr=cfg["lr"], betas=tuple(cfg["betas"]))


def train_step(cfg, enc, dec, opt, sched, scaler, corpus, gens, device):
    x, m = corpus.sample(cfg["batch_size"], gens["cpu"], device)
    with torch.no_grad():
        xc, mc = corrupt(x, m, CORRUPTIONS[cfg["corruption"]], generator=gens["dev"])
    enc.train(), dec.train()
    with torch.autocast(device.type, enabled=cfg["amp"] and device.type == "cuda"):
        tokens, valid = enc.tokenize(xc, mc)
        hidden = block_mask(valid, cfg["mask_ratio"], *cfg["mask_block"], generator=gens["dev"])
        keep, keep_valid = visible_order(valid, hidden)
        vis = tokens.gather(1, keep[..., None].expand(-1, -1, tokens.shape[2]))
        pred = dec(enc.encode(vis, keep_valid), keep, keep_valid, valid)
    loss, parts = mae_loss(pred, x, m, hidden, cfg["deriv_weight"])
    opt.zero_grad(set_to_none=True)
    if not torch.isfinite(loss):
        return None
    scaler.scale(loss).backward()
    scaler.unscale_(opt)
    torch.nn.utils.clip_grad_norm_(list(enc.parameters()) + list(dec.parameters()), cfg["grad_clip"])
    scaler.step(opt)
    scaler.update()
    sched.step()
    return loss.item(), parts


@torch.no_grad()
def reconstruct(enc, dec, X, M, rows, cfg, device):
    """Fixed corruption + mask on fixed probe spectra -> arrays for plotting."""
    enc.eval(), dec.eval()
    x = torch.as_tensor(np.asarray(X[rows], np.float32), device=device)
    m = torch.as_tensor(np.asarray(M[rows]), device=device)
    g = torch.Generator(device=device).manual_seed(1234)
    xc, mc = corrupt(x, m, CORRUPTIONS[cfg["corruption"]], generator=g)
    tokens, valid = enc.tokenize(xc, mc)
    hidden = block_mask(valid, cfg["mask_ratio"], *cfg["mask_block"], generator=g)
    keep, keep_valid = visible_order(valid, hidden)
    vis = tokens.gather(1, keep[..., None].expand(-1, -1, tokens.shape[2]))
    pred = dec(enc.encode(vis, keep_valid), keep, keep_valid, valid).float()
    tgt, pm = patchify(x), patchify(m.float()) > 0
    cnt = pm.sum(-1, keepdim=True).clamp_min(1)
    mean = (tgt * pm).sum(-1, keepdim=True) / cnt
    sd = torch.sqrt((((tgt - mean) ** 2) * pm).sum(-1, keepdim=True) / cnt + TARGET_EPS)
    recon = (pred * sd + mean).flatten(1)[:, :x.shape[1]]
    cpu = lambda t: t.cpu().numpy()
    return {"clean": cpu(x), "mask": cpu(m), "corrupted": cpu(xc), "corrupted_mask": cpu(mc),
            "hidden_tokens": cpu(hidden), "reconstruction": cpu(recon)}


def _state(cfg, enc, dec, opt, scaler, sched, epoch, track, gens, device):
    state = {"config": cfg, "epoch": epoch, "track": track,
            "encoder": enc.state_dict(), "decoder": dec.state_dict(),
            "optimizer": opt.state_dict(), "scaler": scaler.state_dict(),
            "scheduler": sched.state_dict(),
            "gen_cpu": gens["cpu"].get_state(), "gen_dev": gens["dev"].get_state(),
            "rng_torch": torch.get_rng_state()}
    if device.type == "cuda":
        state["rng_cuda"] = torch.cuda.get_rng_state()
    return state


def run(cfg, run_dir=None, resume=False, log=print):
    device = torch.device(cfg["device"] if cfg["device"] != "cuda" or torch.cuda.is_available()
                          else "cpu")
    torch.manual_seed(cfg["seed"])
    corpus = Corpus(cfg["corpus_dir"])
    probe_set = build_probe_set(corpus.meta, seed=cfg["seed"])
    enc, dec = build_models(cfg, device)
    opt = make_optimizer(cfg, enc, dec)
    steps = cfg["steps_per_epoch"] or math.ceil(len(corpus.pretrain_idx) / cfg["batch_size"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(
        s, cfg["warmup_epochs"] * steps, cfg["max_epochs"] * steps))
    scaler = torch.amp.GradScaler(device.type, enabled=cfg["amp"] and device.type == "cuda")
    gens = {"cpu": torch.Generator().manual_seed(cfg["seed"]),
            "dev": torch.Generator(device=device).manual_seed(cfg["seed"] + 1)}
    recon_rows = np.concatenate([t.test_idx[:cfg["n_recon"] // max(1, len(probe_set.tasks))]
                                 for t in probe_set.tasks])[:cfg["n_recon"]]

    if resume:
        ck = torch.load(Path(run_dir) / "ckpt_last.pt", map_location="cpu", weights_only=True)
        enc.load_state_dict(ck["encoder"]), dec.load_state_dict(ck["decoder"])
        opt.load_state_dict(ck["optimizer"]), scaler.load_state_dict(ck["scaler"])
        sched.load_state_dict(ck["scheduler"])
        gens["cpu"].set_state(ck["gen_cpu"]), gens["dev"].set_state(ck["gen_dev"])
        torch.set_rng_state(ck["rng_torch"])
        if device.type == "cuda" and "rng_cuda" in ck:
            torch.cuda.set_rng_state(ck["rng_cuda"])
        start, track = ck["epoch"] + 1, ck["track"]
        # Filter metrics.jsonl to remove stale lines and end markers
        metrics_path = Path(run_dir) / "metrics.jsonl"
        lines = metrics_path.read_text(encoding="utf-8").splitlines()
        filtered = [l for l in lines if l and (("epoch" in l and json.loads(l).get("epoch", 0) <= ck["epoch"])
                                                 or ("epoch" in l and "probe" in l))]
        metrics_path.write_text("\n".join(filtered) + ("\n" if filtered else ""), encoding="utf-8")
    else:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        run_dir = Path(cfg["run_root"]) / f"{stamp}_{cfg['preset']}_{cfg['tag']}"
        run_dir.mkdir(parents=True)
        start = 1
        track = {"best_score": None, "best_epoch": None, "bad_checks": 0,
                 "raw_knn_score": raw_knn_score(corpus.X, corpus.M, probe_set)}
        (run_dir / "manifest.json").write_text(json.dumps({
            "git_sha": _git_sha(), "corpus": cfg["corpus_dir"],
            "corpus_version": corpus.manifest_version, "device": str(device),
            "torch": torch.__version__, "python": platform.python_version(),
            "encoder_params": count_parameters(enc), "decoder_params": count_parameters(dec),
            "steps_per_epoch": steps, "raw_knn_score": track["raw_knn_score"]}, indent=2))
    run_dir = Path(run_dir)
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    metrics = open(run_dir / "metrics.jsonl", "a", encoding="utf-8")
    log(f"[pretrain] {run_dir} | {cfg['preset']} | {count_parameters(enc):,} encoder params | "
        f"{steps} steps/epoch | epochs {start}..{cfg['max_epochs']}")

    t0, epoch_secs, reason = time.time(), [], None
    nonfinite_streak = 0
    for epoch in range(start, cfg["max_epochs"] + 1):
        te = time.time()
        losses, parts_sum, skipped = [], {"mse": 0.0, "dmse": 0.0}, 0
        for _ in range(steps):
            out = train_step(cfg, enc, dec, opt, sched, scaler, corpus, gens, device)
            if out is None:
                skipped += 1
                nonfinite_streak += 1
                if nonfinite_streak > cfg["max_nonfinite_steps"]:
                    raise RuntimeError(f"{nonfinite_streak} consecutive non-finite losses at "
                                       f"epoch {epoch}; stopping")
                continue
            nonfinite_streak = 0
            losses.append(out[0])
            for k in parts_sum:
                parts_sum[k] += out[1][k]
        epoch_secs.append(time.time() - te)
        n = max(1, len(losses))
        rec = {"epoch": epoch, "loss": float(np.mean(losses)) if losses else float("nan"),
               "mse": parts_sum["mse"] / n, "dmse": parts_sum["dmse"] / n,
               "lr": opt.param_groups[0]["lr"], "seconds": epoch_secs[-1], "skipped_steps": skipped}
        metrics.write(json.dumps(rec) + "\n")
        metrics.flush()
        left = cfg["max_epochs"] - epoch
        eta = np.mean(epoch_secs) * left
        log(f"[pretrain] epoch {epoch}/{cfg['max_epochs']} ({left} left) | loss {rec['loss']:.4f} "
            f"| lr {rec['lr']:.1e} | {_fmt_secs(time.time() - t0)} elapsed, ~{_fmt_secs(eta)} ETA")

        if epoch % cfg["probe_every"] == 0 or epoch == cfg["max_epochs"]:
            probe = evaluate(enc, corpus.X, corpus.M, probe_set, device)
            improved = track["best_score"] is None or probe["probe_score"] > track["best_score"] + 1e-4
            if improved:
                track.update(best_score=probe["probe_score"], best_epoch=epoch, bad_checks=0)
                _save_atomic({"encoder": enc.state_dict(), "preset": cfg["preset"],
                              "epoch": epoch, "probe": probe}, run_dir / "ckpt_best.pt")
            else:
                track["bad_checks"] += 1
            np.savez_compressed(run_dir / f"recon_epoch{epoch:04d}.npz",
                                **reconstruct(enc, dec, corpus.X, corpus.M, recon_rows, cfg, device))
            metrics.write(json.dumps({"epoch": epoch, "probe": probe, "improved": improved}) + "\n")
            metrics.flush()
            _save_atomic(_state(cfg, enc, dec, opt, scaler, sched, epoch, track, gens, device),
                         run_dir / "ckpt_last.pt")
            log(f"[probe] epoch {epoch} | probe {probe['probe_score']:.4f} "
                f"(knn {probe['knn_score']:.4f}, logreg {probe['logreg_score']:.4f}, "
                f"raw-knn {track['raw_knn_score']:.4f}) | rankme {probe['rankme']:.1f} "
                f"| source-id {probe['source_id_bacc']:.3f} | best {track['best_score']:.4f} "
                f"@ {track['best_epoch']}")
            if track["bad_checks"] >= cfg["patience"]:
                reason = (f"stopped at epoch {epoch}/{cfg['max_epochs']}: probe score flat for "
                          f"{cfg['patience']} checkpoints")
                break
    else:
        if track["bad_checks"] == 0:
            reason = (f"finished {cfg['max_epochs']}/{cfg['max_epochs']}: probe score still "
                      "improving, resume with a larger max_epochs if needed")
        else:
            reason = (f"finished {cfg['max_epochs']}/{cfg['max_epochs']}: best probe score at "
                      f"epoch {track['best_epoch']}")
    metrics.write(json.dumps({"end": reason, "track": track}) + "\n")
    metrics.close()
    log(f"[pretrain] {reason}")
    return run_dir, track


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config")
    ap.add_argument("--resume", help="run directory to continue from its last checkpoint")
    ap.add_argument("--max-epochs", type=int)
    ap.add_argument("--device")
    args = ap.parse_args(argv)
    if args.resume:
        cfg = load_config(Path(args.resume) / "config.yaml", max_epochs=args.max_epochs,
                          device=args.device)
        run(cfg, run_dir=args.resume, resume=True, log=lambda s: print(s, flush=True))
    else:
        if not args.config:
            ap.error("--config or --resume is required")
        cfg = load_config(args.config, max_epochs=args.max_epochs, device=args.device)
        run(cfg, log=lambda s: print(s, flush=True))


if __name__ == "__main__":
    main()
