"""Single final test pass and report (spec section 7 phase 4, section 8).

    python -m foundation.evaluate                 # needs every planned run finished
    python -m foundation.evaluate --skip-curve    # proceed without the optional curve
    python -m foundation.evaluate --report-only   # rebuild the report from saved predictions

Scores T1's test set and T2's 2019clinical set ONCE with every finished,
frozen run (4 arms x 5 seeds per task at the selected LR, plus the optional
curve runs), saves per-sample predictions under <run_root>/final_test/ and
writes scored.json. A second scoring pass is refused; --report-only only
re-reads the saved predictions.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import confusion_matrix, f1_score  # noqa: E402

from .adapt import (TASKS, GPUSplit, load_config, load_task, predict,  # noqa: E402
                    run_dir_for, selection_path)
from .arms import ARMS, build_model, load_a, load_addon  # noqa: E402
from .pretrain import _git_sha  # noqa: E402
from .stats import bootstrap_diff, mcnemar, mean_sd, q1, q3  # noqa: E402

PAIRS = (("B1", "S"), ("B2", "S"), ("B1", "B2"))
FINAL = "final_test"


def planned_runs(cfg):
    """[(key, task, arm, lr, seed, run_dir)] for the main comparison."""
    runs = []
    for t in TASKS:
        path = selection_path(cfg, t)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing: run the LR sweep first")
        sel = json.loads(path.read_text())
        for arm in ARMS:
            for s in cfg["seeds"]:
                lr = sel[arm]["lr"]
                runs.append((f"{t}/{arm}_seed{s}", t, arm, lr, s, run_dir_for(cfg, t, arm, lr, s)))
    return runs


def curve_runs(cfg):
    sel = json.loads(selection_path(cfg, "T1").read_text())
    runs = []
    for n in cfg["curve_per_class"]:
        c = dict(cfg, ft_per_class=n)
        for arm in cfg["curve_arms"]:
            for s in cfg["curve_seeds"]:
                lr = sel[arm]["lr"]
                runs.append((f"T1_pc{n}/{arm}_seed{s}", "T1", arm, lr, s,
                             run_dir_for(c, "T1", arm, lr, s)))
    return runs


def unfinished(runs):
    return [r[0] for r in runs if not (r[5] / "summary.json").exists()]


def score(cfg, runs, out, log=print):
    """Predict each task's test set once per run; save predictions and labels."""
    device = torch.device(cfg["device"] if cfg["device"] != "cuda" or torch.cuda.is_available()
                          else "cpu")
    preset, a_state = load_a(cfg["a_ckpt"])
    tasks = {t: load_task(t, cfg) for t in sorted({r[1] for r in runs})}
    for t, task in tasks.items():
        (out / t).mkdir(parents=True, exist_ok=True)
        np.save(out / t / "y_test.npy", task.splits["test"].y)
    test = {t: GPUSplit(task.splits["test"], device) for t, task in tasks.items()}
    for n, (key, t, arm, lr, seed, rdir) in enumerate(runs, 1):
        summary = json.loads((rdir / "summary.json").read_text())
        model, _ = build_model(arm, len(tasks[t].classes), preset, a_state, seed,
                               lora_rank=summary["lora_rank"])
        load_addon(model, torch.load(rdir / "addon.pt", map_location="cpu", weights_only=True))
        model.to(device)
        pred = predict(model, test[t], device).argmax(dim=1).cpu().numpy()
        (out / key).parent.mkdir(parents=True, exist_ok=True)
        np.save(out / f"{key}.npy", pred)
        log(f"[test] {n}/{len(runs)} {key}: acc {(pred == tasks[t].splits['test'].y).mean():.4f}")


def _load(out, key):
    return np.load(out / f"{key}.npy")


def collect(cfg, runs, out):
    """Per task and arm: seed-ordered predictions, summaries and metrics."""
    res = {}
    for key, t, arm, lr, seed, rdir in runs:
        y = np.load(out / t / "y_test.npy")
        pred = _load(out, key)
        s = json.loads((rdir / "summary.json").read_text())
        r = res.setdefault(t, {}).setdefault(arm, {"lr": lr, "seeds": [], "correct": [],
                                                  "acc": [], "f1": [], "cm": 0, "summ": []})
        r["seeds"].append(seed)
        r["correct"].append(pred == y)
        r["acc"].append(float((pred == y).mean()))
        r["f1"].append(float(f1_score(y, pred, average="macro")))
        r["cm"] = r["cm"] + confusion_matrix(y, pred, labels=np.arange(int(y.max()) + 1))
        r["summ"].append(s)
    return res


def analyse(res):
    """Numbers for every table in the report (JSON-serialisable)."""
    out = {"tasks": {}, "diffs": {}}
    for t, arms in res.items():
        rows = {}
        for arm, r in arms.items():
            rows[arm] = {
                "lr": r["lr"], "seeds": r["seeds"],
                "acc": mean_sd(r["acc"]), "f1": mean_sd(r["f1"]),
                "val_acc": mean_sd([s["val_acc"] for s in r["summ"]]),
                "trainable": r["summ"][0]["trainable_params"],
                "fraction": r["summ"][0]["trainable_fraction"],
                "addon_bytes": r["summ"][0]["addon_bytes"],
                "seconds": mean_sd([s["seconds"] for s in r["summ"]]),
                "recall": (np.diag(r["cm"]) / r["cm"].sum(axis=1).clip(min=1)).tolist(),
                "confusion": r["cm"].tolist(),
            }
        out["tasks"][t] = rows
        diffs = {}
        for a, b in PAIRS:
            if a in arms and b in arms:
                if arms[a]["seeds"] != arms[b]["seeds"]:
                    raise ValueError(f"{t}: {a} and {b} are not paired by seed")
                ca, cb = np.stack(arms[a]["correct"]), np.stack(arms[b]["correct"])
                diffs[f"{a} vs {b}"] = {**bootstrap_diff(ca, cb),
                                        "mcnemar": [mcnemar(x, y) for x, y in zip(ca, cb)]}
        out["diffs"][t] = diffs
    out["q1"] = q1({t: {tuple(k.split(" vs ")): v for k, v in d.items()}
                    for t, d in out["diffs"].items()})
    out["q2"] = {t: {**out["diffs"][t]["B1 vs B2"],
                     "trainable_ratio": rows["B1"]["trainable"] / rows["B2"]["trainable"]}
                 for t, rows in out["tasks"].items()}
    out["q3"] = q3({t: {a: r["acc"][0] for a, r in rows.items()} for t, rows in out["tasks"].items()},
                   {t: {a: r["fraction"] for a, r in rows.items()} for t, rows in out["tasks"].items()})
    return out


def _pct(v, d=2):
    return f"{100 * v:.{d}f}"


def _size(b):
    return f"{b / 1e6:.2f} MB" if b >= 1e6 else f"{b / 1e3:.1f} kB"


def render(a, classes, a_ckpt, curve=None, figs=None):
    """Markdown report from analyse() output."""
    L = [f"# Foundation experiment: adaptation results", "",
         f"Model A: `{a_ckpt}` ({_size(os.path.getsize(a_ckpt))}). Test sets scored once "
         "(spec section 7, phase 4). Accuracies in %; mean +- sd over seeds "
         f"{a['tasks'][TASKS[0]]['S']['seeds']}.", "",
         "**Known limitation (spec 8.4):** one pretraining run of A; variation between "
         "pretraining seeds is not measured.", ""]
    for t, rows in a["tasks"].items():
        L += [f"## {t}", "", "| arm | LR | val acc | test acc | macro-F1 | trainable | add-on | "
              "time per run |", "|---|---|---|---|---|---|---|---|"]
        for arm in ARMS:
            r = rows[arm]
            L.append(f"| {arm} | {r['lr']:.0e} | {_pct(r['val_acc'][0])} | "
                     f"{_pct(r['acc'][0])} +- {_pct(r['acc'][1])} | "
                     f"{_pct(r['f1'][0])} +- {_pct(r['f1'][1])} | "
                     f"{r['trainable']:,} ({_pct(r['fraction'])}%) | {_size(r['addon_bytes'])} | "
                     f"{r['seconds'][0] / 60:.1f} min |")
        L += ["", "| comparison | diff (points) | 95% bootstrap CI | McNemar p per seed |",
              "|---|---|---|---|"]
        for k, d in a["diffs"][t].items():
            ps = ", ".join(f"{m['p']:.3g}" for m in d["mcnemar"])
            L.append(f"| {k} | {_pct(d['diff'])} | [{_pct(d['lo'])}, {_pct(d['hi'])}] | {ps} |")
        names = [str(int(c)) if float(c).is_integer() else str(c) for c in classes[t]]
        L += ["", "Per-class recall (%, pooled over seeds):", "",
              "| class | " + " | ".join(ARMS) + " |", "|---|" + "---|" * len(ARMS)]
        for i, n in enumerate(names):
            L.append(f"| {n} | " + " | ".join(_pct(rows[arm]["recall"][i], 1) for arm in ARMS) + " |")
        if figs and t in figs:
            L += ["", f"![confusion matrices {t}]({figs[t]})"]
        L.append("")
    L += ["## Pre-registered questions (spec 8.3)", ""]
    for t, qs in a["q1"].items():
        for k, v in qs.items():
            L.append(f"- **Q1** {t} {k}: {_pct(v['diff'])} points, CI [{_pct(v['lo'])}, "
                     f"{_pct(v['hi'])}] -> {v['verdict']} ({'success' if v['success'] else 'not met'})")
    for t, v in a["q2"].items():
        L.append(f"- **Q2** {t} B1 vs B2: {_pct(v['diff'])} points, CI [{_pct(v['lo'])}, "
                 f"{_pct(v['hi'])}]; B1 trains {v['trainable_ratio']:.1f}x B2's parameters")
    for c, v in a["q3"]["candidates"].items():
        det = "; ".join(f"{t}: {_pct(x['gap'])} points below best, {_pct(x['fraction'])}% trainable"
                        for t, x in v["tasks"].items())
        L.append(f"- **Q3** {c}: {'PASS' if v['pass'] else 'FAIL'} ({det})")
    L += [f"- **Q3 overall: {'PASS' if a['q3']['pass'] else 'FAIL'}**", "",
          "## Reuse deliverable (spec 6.3)", "",
          f"One A file ({_size(os.path.getsize(a_ckpt))}) serves both tasks; each task stores only "
          "its add-on.", "", "| task | arm | add-on | trainable | test acc |", "|---|---|---|---|---|"]
    for t, rows in a["tasks"].items():
        for arm in ("LP", "B2"):
            r = rows[arm]
            L.append(f"| {t} | {arm} | {_size(r['addon_bytes'])} | {r['trainable']:,} | "
                     f"{_pct(r['acc'][0])} |")
    if curve:
        L += ["", "## Data-efficiency curve (spec 8.5, T1 stage ii)", "",
              "| spectra per class | " + " | ".join(curve["arms"]) + " |",
              "|---|" + "---|" * len(curve["arms"])]
        for n, row in curve["rows"].items():
            L.append(f"| {n} | " + " | ".join(f"{_pct(m)} +- {_pct(s)}" for m, s in row) + " |")
    return "\n".join(L) + "\n"


def plot_confusions(a, t, path):
    rows = a["tasks"][t]
    fig, axes = plt.subplots(1, len(ARMS), figsize=(5 * len(ARMS), 4.6))
    for ax, arm in zip(axes, ARMS):
        cm = np.asarray(rows[arm]["confusion"], float)
        ax.imshow(cm / cm.sum(axis=1, keepdims=True).clip(min=1), vmin=0, vmax=1, cmap="Blues")
        ax.set(title=f"{t} {arm} ({_pct(rows[arm]['acc'][0])}%)", xlabel="predicted",
               ylabel="true")
    fig.tight_layout()
    fig.savefig(path, dpi=100)
    plt.close(fig)


def curve_results(cfg, crun, out):
    """{per-class size: collect() result for T1} over the curve runs."""
    return {n: collect(cfg, [r for r in crun if r[0].startswith(f"T1_pc{n}/")], out)["T1"]
            for n in cfg["curve_per_class"]}


def curve_table(cfg, by_size, res_main):
    """Mean +- sd accuracy per size and curve arm; the last row is the full
    stage-ii set (the main runs, restricted to the curve seeds)."""
    rows = {n: [mean_sd(by_size[n][arm]["acc"]) for arm in cfg["curve_arms"]]
            for n in cfg["curve_per_class"]}
    main_t1 = res_main["T1"]
    rows["all"] = [mean_sd([acc for s, acc in zip(main_t1[arm]["seeds"], main_t1[arm]["acc"])
                                 if s in cfg["curve_seeds"]]) for arm in cfg["curve_arms"]]
    return {"arms": cfg["curve_arms"], "rows": rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", default="foundation/configs/adapt.yaml")
    ap.add_argument("--skip-curve", action="store_true")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--results", default="foundation/results")
    ap.add_argument("--device")
    args = ap.parse_args(argv)
    cfg = load_config(args.config if Path(args.config).exists() else None, device=args.device)
    out = Path(cfg["run_root"]) / FINAL
    runs = planned_runs(cfg)
    missing = unfinished(runs)
    if missing:
        raise SystemExit(f"{len(missing)} planned runs are not finished, e.g. {missing[:3]}; "
                         "run `python -m foundation.adapt sweep` and `... seeds` first")
    crun = curve_runs(cfg)
    have_curve = not unfinished(crun)
    if not have_curve and not args.skip_curve and not args.report_only:
        done = len(crun) - len(unfinished(crun))
        raise SystemExit(f"the optional curve has {done}/{len(crun)} runs finished. The test sets "
                         "are scored only once, so finish it (`python -m foundation.adapt curve`) "
                         "or pass --skip-curve to go without it")
    lock = out / "scored.json"
    if args.report_only:
        if not lock.exists():
            raise SystemExit(f"{lock} missing: the test sets have not been scored yet")
        scored = json.loads(lock.read_text())
        have_curve = scored["curve"]
    else:
        if lock.exists():
            raise SystemExit(f"the test sets were already scored ({lock}); use --report-only")
        score(cfg, runs + (crun if have_curve else []), out)
        lock.write_text(json.dumps({"date": datetime.now().isoformat(timespec="seconds"),
                                    "git_sha": _git_sha(), "curve": have_curve,
                                    "runs": [r[0] for r in runs + (crun if have_curve else [])]},
                                   indent=2))
    res = collect(cfg, runs, out)
    a = analyse(res)
    curve = curve_table(cfg, curve_results(cfg, crun, out), res) if have_curve else None
    results = Path(args.results)
    figdir = results / "adaptation"
    figdir.mkdir(parents=True, exist_ok=True)
    figs = {}
    for t in a["tasks"]:
        plot_confusions(a, t, figdir / f"confusion_{t}.png")
        figs[t] = f"adaptation/confusion_{t}.png"
    classes = {t: load_task(t, cfg).classes for t in a["tasks"]}
    text = render(a, classes, cfg["a_ckpt"], curve, figs)
    (results / "adaptation.md").write_text(text, encoding="utf-8")
    (out / "report.json").write_text(json.dumps({**a, "curve": curve}, indent=2, default=str))
    print(text)


if __name__ == "__main__":
    main()
