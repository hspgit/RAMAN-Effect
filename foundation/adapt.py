"""Adapt model A to our tasks with arms S / LP / B1 / B2 (spec section 6).

    python -m foundation.adapt run   --task T1 --arm B1 --lr 3e-4 --seed 1
    python -m foundation.adapt sweep [--task T1 T2] [--arm S LP B1 B2]
    python -m foundation.adapt seeds [--task T1 T2] [--arm ...]
    python -m foundation.adapt curve                (optional, spec 8.5)
    python -m foundation.adapt status

`sweep` trains every arm at each learning rate of its grid on the sweep seed
and writes <run_root>/<task>/selection.json (best validation accuracy, ties
-> lower validation loss). `seeds` trains the remaining seeds at the selected
learning rate. A run whose summary.json exists is skipped, so both commands
can be re-run after an interruption. Test sets are never touched here
(foundation.evaluate scores them once).

Runs end at a stage's max_epochs or when validation loss has not improved for
`patience` epochs; never on elapsed time.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import sklearn
import torch
import torch.nn.functional as F
import yaml
from sklearn.metrics import f1_score

from .arms import (ARMS, a_fingerprint, addon_state, build_model, load_a, param_groups,
                   phases, set_trainable, total_count, trainable_count)
from .data.corrupt import PRESETS as CORRUPTIONS
from .data.corrupt import corrupt
from .data.tasks import Split, load_t1, load_t2
from .pretrain import _fmt_secs, _git_sha, _save_atomic, lr_factor

TASKS = ("T1", "T2")
# (stage name, train split, validation split). The last stage's validation
# split is the one selection uses (spec 6.2: T1 -> stage-ii finetune val).
STAGES = {"T1": [("stage-i", "ref_train", "ref_val"), ("stage-ii", "ft_train", "ft_val")],
          "T2": [("single", "train", "val")]}

DEFAULTS = {
    "a_ckpt": "foundation/runs/pretrain/20261003-123546_small_main/ckpt_best.pt",
    "data_dir": "data", "cache_dir": "foundation/data/cache",
    "run_root": "foundation/runs/adapt", "device": "cuda", "amp": True,
    "batch_size": 256, "weight_decay": 0.05, "betas": [0.9, 0.999],
    "lr_warmup_epochs": 1, "llrd": 0.75, "patience": 5,
    "head_warmup_epochs": 5, "head_warmup_lr_mult": 10.0,
    "max_epochs": {"stage-i": 30, "stage-ii": 50, "single": 50},
    "augment": "mild", "lora_dropout": 0.05,
    "lr_grid": {"S": [1.0e-4, 3.0e-4, 1.0e-3], "B1": [1.0e-4, 3.0e-4, 1.0e-3],
                "LP": [1.0e-3, 3.0e-3, 1.0e-2], "B2": [1.0e-3, 3.0e-3, 1.0e-2]},
    "sweep_seed": 1, "seeds": [1, 2, 3, 4, 5],
    "ft_per_class": None,
    "curve_per_class": [5, 10, 25, 50], "curve_arms": ["S", "B1", "B2"], "curve_seeds": [1, 2, 3],
}


def load_config(path=None, **overrides):
    cfg = json.loads(json.dumps(DEFAULTS))
    if path:
        cfg.update(yaml.safe_load(Path(path).read_text()) or {})
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def load_task(name, cfg):
    loader = {"T1": load_t1, "T2": load_t2}[name]
    task = loader(cfg["data_dir"], cfg["cache_dir"])
    if cfg.get("ft_per_class") and name == "T1":
        task.splits["ft_train"] = subsample_per_class(task.splits["ft_train"],
                                                      cfg["ft_per_class"])
    return task


def subsample_per_class(split: Split, n, seed=0):
    """First n spectra of each class after a fixed shuffle (optional data-efficiency
    curve, spec 8.5). Same rows for every arm and seed."""
    rng = np.random.default_rng(seed)
    keep = []
    for c in np.unique(split.y):
        idx = np.flatnonzero(split.y == c)
        rng.shuffle(idx)
        if idx.size < n:
            raise ValueError(f"class {c} has {idx.size} spectra, fewer than {n}")
        keep.append(idx[:n])
    keep = np.sort(np.concatenate(keep))
    return Split(split.X[keep], split.M[keep], split.y[keep])


def run_dir_for(cfg, task, arm, lr, seed):
    """Run folders are named by a one-significant-digit LR (lr3e-04_seed1), so
    an LR that does not round-trip through that name would share a folder."""
    if not math.isclose(float(f"{lr:.0e}"), lr, rel_tol=1e-9):
        raise ValueError(f"run folders are named by a one-significant-digit LR (e.g. 3e-04); "
                         f"lr {lr!r} would collide with lr {lr:.0e}")
    t = task if not cfg.get("ft_per_class") else f"{task}_pc{cfg['ft_per_class']}"
    return Path(cfg["run_root"]) / t / arm / f"lr{lr:.0e}_seed{seed}"


class GPUSplit:
    def __init__(self, split: Split, device):
        self.X = torch.as_tensor(split.X, dtype=torch.float32, device=device)
        self.M = torch.as_tensor(split.M, device=device)
        self.y = torch.as_tensor(split.y, dtype=torch.long, device=device)

    def __len__(self):
        return self.y.numel()


@torch.no_grad()
def predict(model, data: GPUSplit, device, batch=1024):
    """Logits for every row, in order (eval mode, no augmentation)."""
    model.eval()
    out = []
    for s in range(0, len(data), batch):
        with torch.autocast(device.type, enabled=device.type == "cuda"):
            out.append(model(data.X[s:s + batch], data.M[s:s + batch]).float())
    return torch.cat(out)


def val_metrics(model, data: GPUSplit, device):
    logits = predict(model, data, device)
    loss = F.cross_entropy(logits, data.y).item()
    pred = logits.argmax(dim=1)
    return loss, (pred == data.y).float().mean().item(), pred.cpu().numpy()


def train_phase(model, phase, train, val, lr, max_epochs, early_stop, tag, cfg, gen,
                device, log, metrics):
    """Train one phase. With early_stop, ends after `patience` epochs without a
    lower validation loss and restores the best epoch's trainable weights."""
    set_trainable(model, phase)
    groups = param_groups(model, lr, cfg["weight_decay"],
                          cfg["llrd"] if phase == "full" else None)
    opt = torch.optim.AdamW(groups, betas=tuple(cfg["betas"]))
    bs = cfg["batch_size"]
    steps = math.ceil(len(train) / bs)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: lr_factor(
        s, cfg["lr_warmup_epochs"] * steps, max_epochs * steps))
    use_amp = cfg["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
    aug = CORRUPTIONS[cfg["augment"]]
    params = [p for p in model.parameters() if p.requires_grad]
    best = {"loss": math.inf, "epoch": 0, "state": None}
    bad, secs, reason = 0, [], None
    for epoch in range(1, max_epochs + 1):
        te = time.time()
        model.train()
        if phase == "head":
            model.encoder.eval()          # frozen features: no dropout
        perm = torch.randperm(len(train), generator=gen, device=device)
        total, seen = 0.0, 0
        for s in range(0, len(train), bs):
            idx = perm[s:s + bs]
            x, m = corrupt(train.X[idx], train.M[idx], aug, generator=gen)
            with torch.autocast(device.type, enabled=use_amp):
                loss = F.cross_entropy(model(x, m).float(), train.y[idx])
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            total += loss.item() * idx.numel()
            seen += idx.numel()
        vloss, vacc, _ = val_metrics(model, val, device)
        secs.append(time.time() - te)
        improved = vloss < best["loss"] - 1e-4
        if improved:
            best = {"loss": vloss, "epoch": epoch,
                    "state": {k: v.detach().clone() for k, v in model.state_dict().items()}}
            bad = 0
        else:
            bad += 1
        rec = {"tag": tag, "phase": phase, "epoch": epoch, "max_epochs": max_epochs,
               "train_loss": total / max(1, seen), "val_loss": vloss, "val_acc": vacc,
               "lr": max(g["lr"] for g in opt.param_groups),   # the head's (largest) LR
               "seconds": secs[-1]}
        metrics.write(json.dumps(rec) + "\n")
        metrics.flush()
        left = max_epochs - epoch
        log(f"[{tag}] epoch {epoch}/{max_epochs} ({left} left) | train loss "
            f"{rec['train_loss']:.4f} | val loss {vloss:.4f} | val acc {vacc:.4f} | best "
            f"{best['loss']:.4f} @ {best['epoch']} | ~{_fmt_secs(np.mean(secs) * left)} ETA")
        if early_stop and bad >= cfg["patience"]:
            reason = (f"stopped at epoch {epoch}/{max_epochs}: val loss flat for "
                      f"{cfg['patience']} epochs")
            break
    else:
        reason = f"finished {max_epochs}/{max_epochs}"
    if early_stop and best["state"] is not None:
        model.load_state_dict(best["state"])
        reason += f"; restored epoch {best['epoch']}"
    log(f"[{tag}] {reason}")
    return {"phase": phase, "epochs_run": epoch, "best_epoch": best["epoch"],
            "best_val_loss": best["loss"], "reason": reason, "seconds": float(sum(secs))}


def stage_label(stage, warmup):
    """'stage-i', 'stage-i-warmup', 'stage-ii'; T2's single stage has no label
    except 'warmup'. Used in progress tags: [B1 T1 stage-ii seed 3]."""
    parts = ([] if stage == "single" else [stage]) + (["warmup"] if warmup else [])
    return "-".join(parts)


def _keeps_stage1(cfg, arm, seed):
    """Full-data T1 runs that the optional curve will reuse save their state
    after stage i (stage i never sees finetune data, so it is identical)."""
    return not cfg.get("ft_per_class") and arm in cfg["curve_arms"] and seed in cfg["curve_seeds"]


def _stage1_source(cfg, task_name, arm, lr, seed):
    if task_name != "T1" or not cfg.get("ft_per_class"):
        return None
    path = run_dir_for(dict(cfg, ft_per_class=None), task_name, arm, lr, seed) / "stage1.pt"
    return path if path.exists() else None


def run_one(cfg, task_name, arm, lr, seed, task=None, a=None, log=print, a_sha1=None):
    """One arm x task x LR x seed. Returns the summary dict (also saved).
    `a` is load_a()'s result; when given, `a_sha1` must be the fingerprint of
    the file it was loaded from (taken at the same moment)."""
    if a is not None and a_sha1 is None:
        raise ValueError("run_one got model A's weights without their a_sha1 fingerprint")
    out = run_dir_for(cfg, task_name, arm, lr, seed)
    if (out / "summary.json").exists():
        log(f"[skip] {out} already finished")
        return json.loads((out / "summary.json").read_text())
    device = torch.device(cfg["device"] if cfg["device"] != "cuda" or torch.cuda.is_available()
                          else "cpu")
    task = task or load_task(task_name, cfg)
    if a is None:
        a, a_sha1 = load_a(cfg["a_ckpt"]), a_fingerprint(cfg["a_ckpt"])
    preset, a_state = a
    model, rank = build_model(arm, len(task.classes), preset, a_state, seed)
    if arm == "B2" and cfg["lora_dropout"] != 0.05:
        raise ValueError("spec 6.1 fixes LoRA dropout at 0.05")
    model.to(device)
    gen = torch.Generator(device=device).manual_seed(seed)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({
        **cfg, "task": task_name, "arm": arm, "lr": lr, "seed": seed, "lora_rank": rank,
        "preset": preset, "a_sha1": a_sha1, "git_sha": _git_sha(), "torch": torch.__version__,
        "numpy": np.__version__, "sklearn": sklearn.__version__,
        "python": platform.python_version(), "device": str(device)}, indent=2))
    t0 = time.time()
    stages = []
    reuse = _stage1_source(cfg, task_name, arm, lr, seed)
    with open(out / "metrics.jsonl", "w", encoding="utf-8") as metrics:
        for i, (stage, tr_name, va_name) in enumerate(STAGES[task_name]):
            if stage == "stage-i" and reuse is not None:
                ck = torch.load(reuse, map_location="cpu", weights_only=True)
                model.load_state_dict(ck["model"])
                gen.set_state(ck["gen"].to(gen.get_state().device))
                stages += [dict(s, reused_from=str(reuse)) for s in ck["stages"]]
                log(f"[{arm} {task_name} stage-i seed {seed}] reused from {reuse}")
                continue
            train = GPUSplit(task.splits[tr_name], device)
            val = GPUSplit(task.splits[va_name], device)
            todo = phases(arm) if i == 0 else phases(arm)[-1:]
            for phase, warmup in todo:
                where = stage_label(stage, warmup)
                tag = " ".join(filter(None, (arm, task_name, where, f"seed {seed}")))
                if warmup:
                    info = train_phase(model, phase, train, val, lr * cfg["head_warmup_lr_mult"],
                                       cfg["head_warmup_epochs"], False, tag, cfg, gen, device,
                                       log, metrics)
                else:
                    info = train_phase(model, phase, train, val, lr, cfg["max_epochs"][stage],
                                       True, tag, cfg, gen, device, log, metrics)
                stages.append({"stage": where or stage, **info})
            del train, val
            if stage == "stage-i" and _keeps_stage1(cfg, arm, seed):
                _save_atomic({"model": model.state_dict(), "gen": gen.get_state(),
                              "stages": stages}, out / "stage1.pt")
    vloss, vacc, vpred = val_metrics(model, GPUSplit(task.splits[STAGES[task_name][-1][2]],
                                                       device), device)
    vy = task.splits[STAGES[task_name][-1][2]].y
    np.save(out / "val_pred.npy", vpred)
    _save_atomic(addon_state(model), out / "addon.pt")
    summary = {
        "task": task_name, "arm": arm, "lr": lr, "seed": seed, "lora_rank": rank,
        "ft_per_class": cfg.get("ft_per_class"), "a_sha1": a_sha1,
        "val_acc": vacc, "val_loss": vloss,
        "val_macro_f1": float(f1_score(vy, vpred, average="macro")),
        "trainable_params": trainable_count(model), "total_params": total_count(model),
        "trainable_fraction": trainable_count(model) / total_count(model),
        "addon_bytes": os.path.getsize(out / "addon.pt"),
        "seconds": time.time() - t0, "stages": stages,
    }
    tmp = out / "summary.json.tmp"
    tmp.write_text(json.dumps(summary, indent=2))
    os.replace(tmp, out / "summary.json")
    log(f"[{arm} {task_name} seed {seed}] done | val acc {vacc:.4f} | val loss {vloss:.4f} | "
        f"trainable {summary['trainable_params']:,} ({100 * summary['trainable_fraction']:.2f}%)"
        f" | {_fmt_secs(summary['seconds'])}")
    return summary


def select_lr(summaries):
    """Best validation accuracy; ties broken by lower validation loss."""
    return min(summaries, key=lambda s: (-round(s["val_acc"], 6), s["val_loss"]))["lr"]


def selection_path(cfg, task):
    """Selections are made on full data only; curve runs reuse T1's."""
    return Path(cfg["run_root"]) / task / "selection.json"


def _queue(cfg, jobs, log):
    """Run (task, arm, lr, seed) jobs in order, loading each task and A once."""
    a, a_sha1 = load_a(cfg["a_ckpt"]), a_fingerprint(cfg["a_ckpt"])
    tasks, results = {}, []
    for n, (t, arm, lr, seed) in enumerate(jobs, 1):
        log(f"[queue] run {n}/{len(jobs)}: {arm} {t} lr {lr:.0e} seed {seed}")
        if t not in tasks:
            tasks[t] = load_task(t, cfg)
        results.append(run_one(cfg, t, arm, lr, seed, task=tasks[t], a=a, log=log,
                               a_sha1=a_sha1))
    return results


def sweep(cfg, tasks=TASKS, arms=ARMS, log=print):
    s = cfg["sweep_seed"]
    jobs = [(t, arm, lr, s) for t in tasks for arm in arms for lr in cfg["lr_grid"][arm]]
    results = _queue(cfg, jobs, log)
    for t in tasks:
        path = selection_path(cfg, t)
        sel = json.loads(path.read_text()) if path.exists() else {}
        for arm in arms:
            rs = [r for r in results if r["task"] == t and r["arm"] == arm]
            sel[arm] = {"lr": select_lr(rs), "grid": [
                {"lr": r["lr"], "val_acc": r["val_acc"], "val_loss": r["val_loss"]} for r in rs]}
            log(f"[sweep] {t} {arm}: selected lr {sel[arm]['lr']:.0e}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(sel, indent=2))
    return results


def seeds(cfg, tasks=TASKS, arms=ARMS, log=print):
    jobs = [(t, arm, _selected(cfg, t, arm), s) for t in tasks for arm in arms
            for s in cfg["seeds"]]
    return _queue(cfg, jobs, log)


def _selected(cfg, task, arm):
    path = selection_path(cfg, task)
    if not path.exists():
        raise FileNotFoundError(f"{path} missing: run `python -m foundation.adapt sweep` first")
    sel = json.loads(path.read_text())
    if arm not in sel:
        raise KeyError(f"{path} has no selected LR for arm {arm}")
    return sel[arm]["lr"]


def curve(cfg, log=print):
    """Optional data-efficiency curve (spec 8.5): T1 with stage ii cut to n
    spectra per class, at each arm's full-data LR. The full-data point (90 per
    class) is the seeds run itself."""
    results = []
    for n in cfg["curve_per_class"]:
        c = dict(cfg, ft_per_class=n)
        jobs = [("T1", arm, _selected(cfg, "T1", arm), s)
                for arm in cfg["curve_arms"] for s in cfg["curve_seeds"]]
        log(f"[curve] {n} spectra per class")
        results += _queue(c, jobs, log)
    return results


def status(cfg, log=print):
    """How many sweep and seed runs are finished, per task."""
    for t in TASKS:
        sweep_runs = [(arm, lr, cfg["sweep_seed"]) for arm in ARMS for lr in cfg["lr_grid"][arm]]
        done = sum((run_dir_for(cfg, t, *r) / "summary.json").exists() for r in sweep_runs)
        line = f"[status] {t}: sweep {done}/{len(sweep_runs)}"
        path = selection_path(cfg, t)
        if path.exists():
            sel = json.loads(path.read_text())
            final = [(arm, sel[arm]["lr"], s) for arm in ARMS if arm in sel for s in cfg["seeds"]]
            fdone = sum((run_dir_for(cfg, t, *r) / "summary.json").exists() for r in final)
            line += f" | final seeds {fdone}/{len(ARMS) * len(cfg['seeds'])}"
        log(line)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("command", choices=["run", "sweep", "seeds", "curve", "status"])
    ap.add_argument("--config", default="foundation/configs/adapt.yaml")
    ap.add_argument("--task", nargs="+", choices=TASKS, default=list(TASKS))
    ap.add_argument("--arm", nargs="+", choices=ARMS, default=list(ARMS))
    ap.add_argument("--lr", type=float)
    ap.add_argument("--seed", type=int)
    ap.add_argument("--ft-per-class", type=int, help="T1 stage-ii spectra per class (spec 8.5)")
    ap.add_argument("--device")
    args = ap.parse_args(argv)
    if not Path(args.config).exists():
        ap.error(f"config {args.config} not found")
    if args.ft_per_class is not None and args.command != "run":
        ap.error("--ft-per-class is only for `run`; `curve` sets it per point from the config")
    cfg = load_config(args.config, device=args.device, ft_per_class=args.ft_per_class)
    if args.command == "run":
        if args.lr is None or args.seed is None or len(args.task) != 1 or len(args.arm) != 1:
            ap.error("run needs exactly one --task, one --arm, --lr and --seed")
        run_one(cfg, args.task[0], args.arm[0], args.lr, args.seed)
    elif args.command == "sweep":
        sweep(cfg, args.task, args.arm)
    elif args.command == "seeds":
        seeds(cfg, args.task, args.arm)
    elif args.command == "curve":
        curve(cfg)
    else:
        status(cfg)


if __name__ == "__main__":
    main()
