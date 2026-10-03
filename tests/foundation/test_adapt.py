import json

import numpy as np
import pytest
import torch

from foundation import adapt
from foundation.adapt import (GPUSplit, curve, load_task, run_dir_for, run_one, seeds,
                              select_lr, stage_label, status, subsample_per_class, sweep,
                              train_phase)
from foundation.arms import build_model, load_a
from foundation.data.tasks import Split


def test_b1_t1_smoke_run_writes_everything(adapt_cfg):
    logs = []
    s = run_one(adapt_cfg, "T1", "B1", 1e-3, 1, log=logs.append)
    out = run_dir_for(adapt_cfg, "T1", "B1", 1e-3, 1)
    for f in ("config.json", "metrics.jsonl", "val_pred.npy", "addon.pt", "summary.json"):
        assert (out / f).exists(), f
    assert [x["stage"] for x in s["stages"]] == ["stage-i-warmup", "stage-i", "stage-ii"]
    assert s["trainable_fraction"] == 1.0 and s["addon_bytes"] > 0 and s["lora_rank"] is None
    assert 0.0 <= s["val_acc"] <= 1.0
    assert any(l.startswith("[B1 T1 stage-i-warmup seed 1] epoch 1/1 (0 left)") for l in logs)
    assert any(l.startswith("[B1 T1 stage-ii seed 1] epoch 1/1 (0 left)") for l in logs)
    assert np.load(out / "val_pred.npy").shape == (3,)            # ft_val: 1 per class
    cfg_saved = json.loads((out / "config.json").read_text())
    assert cfg_saved["arm"] == "B1" and "git_sha" in cfg_saved and "torch" in cfg_saved
    tags = {json.loads(l)["tag"] for l in (out / "metrics.jsonl").read_text().splitlines()}
    assert tags == {"B1 T1 stage-i-warmup seed 1", "B1 T1 stage-i seed 1", "B1 T1 stage-ii seed 1"}


def test_finished_run_is_skipped(adapt_cfg):
    run_one(adapt_cfg, "T2", "LP", 1e-2, 1, log=lambda *_: None)
    logs = []
    run_one(adapt_cfg, "T2", "LP", 1e-2, 1, log=logs.append)
    assert logs and logs[0].startswith("[skip]")


def test_b2_trains_only_its_addon_and_t2_tags(adapt_cfg):
    logs = []
    s = run_one(adapt_cfg, "T2", "B2", 1e-2, 2, log=logs.append)
    assert s["lora_rank"] == 4 and s["trainable_fraction"] <= 0.05
    assert any(l.startswith("[B2 T2 seed 2] epoch 1/1") for l in logs)
    addon = torch.load(run_dir_for(adapt_cfg, "T2", "B2", 1e-2, 2) / "addon.pt", weights_only=True)
    assert any(k.endswith("lora_a") for k in addon) and "head.weight" in addon
    assert not any(k.endswith("base.weight") for k in addon)
    assert sum(v.numel() for v in addon.values()) == s["trainable_params"]


def test_b1_t2_warmup_tag():
    assert stage_label("single", True) == "warmup"
    assert stage_label("single", False) == ""
    assert stage_label("stage-i", True) == "stage-i-warmup"
    assert stage_label("stage-ii", False) == "stage-ii"


def test_lp_keeps_a_frozen(adapt_cfg):
    run_one(adapt_cfg, "T2", "LP", 1e-2, 1, log=lambda *_: None)
    addon = torch.load(run_dir_for(adapt_cfg, "T2", "LP", 1e-2, 1) / "addon.pt", weights_only=True)
    assert set(addon) == {"head.weight", "head.bias"}


def test_early_stopping_restores_best_epoch(adapt_cfg, monkeypatch):
    preset, a_state = load_a(adapt_cfg["a_ckpt"])
    model, _ = build_model("LP", 3, preset, a_state, seed=0)
    task = load_task("T1", adapt_cfg)
    tr = GPUSplit(task.splits["ft_train"], torch.device("cpu"))
    va = GPUSplit(task.splits["ft_val"], torch.device("cpu"))
    losses = iter([0.5, 0.6, 0.7, 0.8, 0.9])
    snapshots = []

    def fake_val(m, data, device):
        snapshots.append(m.head.weight.detach().clone())
        return next(losses), 0.5, np.zeros(3, int)

    monkeypatch.setattr(adapt, "val_metrics", fake_val)
    cfg = dict(adapt_cfg, patience=2)
    logs = []
    with open(adapt_cfg["run_root"] + "_m.jsonl", "w") as metrics:
        info = train_phase(model, "head", tr, va, 1e-2, 10, True, "LP T1 stage-ii seed 0", cfg,
                           torch.Generator().manual_seed(0), torch.device("cpu"), logs.append,
                           metrics)
    assert info["epochs_run"] == 3 and info["best_epoch"] == 1
    assert "val loss flat for 2 epochs" in info["reason"]
    assert torch.equal(model.head.weight, snapshots[0])


def test_select_lr_accuracy_then_loss():
    rs = [{"lr": 1e-4, "val_acc": 0.9, "val_loss": 0.3},
          {"lr": 3e-4, "val_acc": 0.95, "val_loss": 0.4},
          {"lr": 1e-3, "val_acc": 0.95, "val_loss": 0.2}]
    assert select_lr(rs) == 1e-3


def test_subsample_per_class():
    y = np.repeat([0, 1, 2], 9)
    s = Split(np.arange(27)[:, None].astype(np.float32), np.ones((27, 1), bool), y)
    sub = subsample_per_class(s, 2)
    assert np.bincount(sub.y).tolist() == [2, 2, 2]
    assert np.array_equal(subsample_per_class(s, 2).X, sub.X)       # fixed rows
    with pytest.raises(ValueError, match="fewer than"):
        subsample_per_class(s, 10)


def test_sweep_selects_then_seeds_reuse_the_sweep_run(adapt_cfg):
    adapt_cfg["lr_grid"]["LP"] = [1e-2, 1e-3]
    sweep(adapt_cfg, tasks=["T2"], arms=["LP"], log=lambda *_: None)
    sel = json.loads((run_dir_for(adapt_cfg, "T2", "LP", 1e-2, 1).parents[1]
                      / "selection.json").read_text())
    assert sel["LP"]["lr"] in (1e-2, 1e-3) and len(sel["LP"]["grid"]) == 2
    logs = []
    seeds(adapt_cfg, tasks=["T2"], arms=["LP"], log=logs.append)
    assert sum(l.startswith("[skip]") for l in logs) == 1          # seed 1 came from the sweep
    assert (run_dir_for(adapt_cfg, "T2", "LP", sel["LP"]["lr"], 2) / "summary.json").exists()
    status_lines = []
    status(adapt_cfg, log=status_lines.append)
    assert any("T2: sweep 2/" in l and "final seeds 2/8" in l for l in status_lines)


def test_seeds_without_sweep_explains(adapt_cfg):
    with pytest.raises(FileNotFoundError, match="sweep"):
        seeds(adapt_cfg, tasks=["T1"], arms=["S"])


def test_curve_reuses_stage_one(adapt_cfg):
    adapt_cfg.update(curve_arms=["B2"])
    sweep(adapt_cfg, tasks=["T1"], arms=["B2"], log=lambda *_: None)
    assert (run_dir_for(adapt_cfg, "T1", "B2", 1e-2, 1) / "stage1.pt").exists()
    logs = []
    curve(adapt_cfg, log=logs.append)
    assert any("reused from" in l for l in logs)
    out = run_dir_for(dict(adapt_cfg, ft_per_class=2), "T1", "B2", 1e-2, 1)
    s = json.loads((out / "summary.json").read_text())
    assert s["ft_per_class"] == 2 and "reused_from" in s["stages"][0]
    assert [x["stage"] for x in s["stages"]] == ["stage-i", "stage-ii"]


def test_head_phase_runs_the_frozen_encoder_without_dropout(adapt_cfg, tmp_path):
    preset, a_state = load_a(adapt_cfg["a_ckpt"])
    model, _ = build_model("LP", 3, preset, a_state, seed=0)
    task = load_task("T1", adapt_cfg)
    cpu = torch.device("cpu")
    tr, va = GPUSplit(task.splits["ft_train"], cpu), GPUSplit(task.splits["ft_val"], cpu)
    modes = []
    model.encoder.blocks[0].register_forward_hook(lambda m, i, o: modes.append(m.training))
    with open(tmp_path / "m.jsonl", "w") as metrics:
        args = (tr, va, 1e-3, 1, False, "t", adapt_cfg, torch.Generator().manual_seed(0), cpu,
                lambda *_: None, metrics)
        train_phase(model, "head", *args)
        assert modes and not any(modes)
        modes.clear()
        train_phase(model, "full", *args)
        assert any(modes)


def test_run_records_model_a_fingerprint_and_versions(adapt_cfg):
    from foundation.arms import a_fingerprint
    s = run_one(adapt_cfg, "T2", "LP", 1e-2, 1, log=lambda *_: None)
    assert s["a_sha1"] == a_fingerprint(adapt_cfg["a_ckpt"])
    saved = json.loads((run_dir_for(adapt_cfg, "T2", "LP", 1e-2, 1) / "config.json").read_text())
    assert saved["a_sha1"] == s["a_sha1"]
    assert "numpy" in saved and "sklearn" in saved


def test_queue_records_model_a_fingerprint(adapt_cfg):
    from foundation.arms import a_fingerprint
    rs = sweep(adapt_cfg, tasks=["T2"], arms=["LP"], log=lambda *_: None)
    assert rs and all(r["a_sha1"] == a_fingerprint(adapt_cfg["a_ckpt"]) for r in rs)


def test_run_one_with_a_needs_its_fingerprint(adapt_cfg):
    a = load_a(adapt_cfg["a_ckpt"])
    with pytest.raises(ValueError, match="a_sha1"):
        run_one(adapt_cfg, "T2", "LP", 1e-2, 1, a=a, log=lambda *_: None)


def test_a_fingerprint_is_sha1_of_the_file(tmp_path):
    import hashlib

    from foundation.arms import a_fingerprint
    p = tmp_path / "f.bin"
    p.write_bytes(b"x" * 3_000_000)
    assert a_fingerprint(p) == hashlib.sha1(p.read_bytes()).hexdigest()


def test_run_dir_needs_a_one_significant_digit_lr(adapt_cfg):
    with pytest.raises(ValueError, match="one-significant-digit"):
        run_dir_for(adapt_cfg, "T1", "S", 2.5e-4, 1)
    for lr in (1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 0.1 * 3e-3):
        run_dir_for(adapt_cfg, "T1", "S", lr, 1)


def test_logged_lr_is_the_heads_under_layer_decay(adapt_cfg, tmp_path):
    preset, a_state = load_a(adapt_cfg["a_ckpt"])
    model, _ = build_model("B1", 3, preset, a_state, seed=0)
    task = load_task("T1", adapt_cfg)
    cpu = torch.device("cpu")
    tr, va = GPUSplit(task.splits["ft_train"], cpu), GPUSplit(task.splits["ft_val"], cpu)
    cfg = dict(adapt_cfg, lr_warmup_epochs=0)
    with open(tmp_path / "m.jsonl", "w") as metrics:
        train_phase(model, "full", tr, va, 1e-3, 2, False, "t", cfg,
                    torch.Generator().manual_seed(0), cpu, lambda *_: None, metrics)
    recs = [json.loads(l) for l in (tmp_path / "m.jsonl").read_text().splitlines()]
    from foundation.pretrain import lr_factor
    steps = -(-len(tr) // cfg["batch_size"])
    assert recs[0]["lr"] == pytest.approx(1e-3 * lr_factor(steps, 0, 2 * steps))


def test_cli_guards(adapt_cfg, tmp_path):
    import yaml
    p = tmp_path / "adapt.yaml"
    p.write_text(yaml.safe_dump(adapt_cfg))
    with pytest.raises(SystemExit):
        adapt.main(["sweep", "--config", str(p), "--ft-per-class", "5"])
    with pytest.raises(SystemExit):
        adapt.main(["status", "--config", str(tmp_path / "missing.yaml")])
    assert not (tmp_path / "runs").exists()
