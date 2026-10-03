import json
import re
from pathlib import Path

import numpy as np
import pytest
import torch

import foundation.pretrain as pt


@pytest.fixture
def cfg(tmp_path, fake_corpus):
    return pt.load_config(None, preset="tiny", corpus_dir=str(fake_corpus), device="cpu",
                          run_root=str(tmp_path / "runs"), batch_size=8, steps_per_epoch=2,
                          max_epochs=2, warmup_epochs=1, probe_every=1, amp=False, n_recon=4)


def test_lr_warms_up_then_decays_to_zero():
    f = [pt.lr_factor(s, 10, 100) for s in range(100)]
    assert f[0] == pytest.approx(0.1) and f[9] == pytest.approx(1.0)
    assert f[10] == pytest.approx(1.0) and f[99] < 0.01
    assert all(a >= b for a, b in zip(f[10:], f[11:]))


def test_run_writes_artefacts_and_progress_lines(cfg):
    lines = []
    run_dir, track = pt.run(cfg, log=lines.append)
    for name in ("manifest.json", "config.yaml", "metrics.jsonl", "ckpt_last.pt",
                 "ckpt_best.pt", "recon_epoch0001.npz", "recon_epoch0002.npz"):
        assert (run_dir / name).exists(), name
    assert any(re.search(r"\[pretrain\] epoch 1/2 \(1 left\) \| loss [\d.]+ \| lr \S+ \| \S+ "
                         r"elapsed, ~\S+ ETA", l) for l in lines)
    assert lines[-1].startswith("[pretrain] finished 2/2")
    assert "raw_knn_score" in json.loads((run_dir / "manifest.json").read_text())
    rec = np.load(run_dir / "recon_epoch0002.npz")
    assert rec["reconstruction"].shape == rec["clean"].shape == (4, 1001)


def test_resume_continues_from_the_last_checkpoint(cfg):
    run_dir, _ = pt.run(cfg, log=lambda s: None)
    lines = []
    cfg3 = pt.load_config(run_dir / "config.yaml", max_epochs=3)
    pt.run(cfg3, run_dir=run_dir, resume=True, log=lines.append)
    epochs = [json.loads(l)["epoch"] for l in (run_dir / "metrics.jsonl").read_text().splitlines()
              if '"loss"' in l]
    assert epochs == [1, 2, 3]
    assert any("epoch 3/3 (0 left)" in l for l in lines)


def test_early_stopping_when_probe_is_flat(cfg, monkeypatch):
    monkeypatch.setattr(pt, "evaluate", lambda *a, **k: {
        "probe_score": 0.5, "knn_score": 0.5, "logreg_score": 0.5, "per_task": {},
        "rankme": 1.0, "source_id_bacc": 0.5})
    cfg.update(max_epochs=6, patience=2)
    lines = []
    _, track = pt.run(cfg, log=lines.append)
    assert lines[-1] == "[pretrain] stopped at epoch 3/6: probe score flat for 2 checkpoints"
    assert track["best_epoch"] == 1


def test_nonfinite_losses_abort_instead_of_spinning(cfg, monkeypatch):
    import torch
    monkeypatch.setattr(pt, "mae_loss", lambda *a, **k: (torch.tensor(float("nan")), {}))
    cfg.update(max_nonfinite_steps=1, steps_per_epoch=3)
    with pytest.raises(RuntimeError, match="non-finite"):
        pt.run(cfg, log=lambda s: None)


def test_checkpoint_save_is_atomic(tmp_path):
    pt._save_atomic({"a": 1}, tmp_path / "ck.pt")
    assert (tmp_path / "ck.pt").exists() and not (tmp_path / "ck.pt.tmp").exists()


def test_resume_works_on_cuda(cfg, monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("needs CUDA")

    # First run on CUDA for 1 epoch
    cfg_cuda = dict(cfg)
    cfg_cuda["device"] = "cuda"
    cfg_cuda["max_epochs"] = 1
    run_dir, _ = pt.run(cfg_cuda, log=lambda s: None)

    # Resume to epoch 2 on CUDA
    cfg_resume = pt.load_config(run_dir / "config.yaml", device="cuda", max_epochs=2)
    pt.run(cfg_resume, run_dir=run_dir, resume=True, log=lambda s: None)

    # Verify epochs
    epochs = [json.loads(l)["epoch"] for l in (run_dir / "metrics.jsonl").read_text().splitlines()
              if '"loss"' in l]
    assert epochs == [1, 2]


def test_resume_reproduces_an_uninterrupted_run(cfg):
    cfg.update(max_epochs=3, probe_every=1)
    call_count = [0]
    original_evaluate = pt.evaluate

    def evaluate_with_interrupt(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 3:
            raise KeyboardInterrupt()
        return original_evaluate(*args, **kwargs)

    # Run A: interrupted at 3rd evaluate (epoch 3)
    import foundation.pretrain
    old_evaluate = foundation.pretrain.evaluate
    foundation.pretrain.evaluate = evaluate_with_interrupt
    try:
        pt.run(cfg, log=lambda s: None)
    except KeyboardInterrupt:
        pass
    finally:
        foundation.pretrain.evaluate = old_evaluate

    run_a = list(Path(cfg["run_root"]).glob("*_tiny_run"))[0]

    # Resume run A
    cfg_resume = pt.load_config(run_a / "config.yaml", max_epochs=3)
    pt.run(cfg_resume, run_dir=run_a, resume=True, log=lambda s: None)

    # Run B: uninterrupted
    cfg_b = dict(cfg)
    cfg_b["run_root"] = str(Path(cfg["run_root"]).parent / "runs_b")
    call_count[0] = 0  # Reset for run B
    run_b, _ = pt.run(cfg_b, log=lambda s: None)

    # Compare encoders
    ck_a = torch.load(run_a / "ckpt_last.pt", map_location="cpu", weights_only=True)
    ck_b = torch.load(run_b / "ckpt_last.pt", map_location="cpu", weights_only=True)

    for key in ck_a["encoder"]:
        assert torch.allclose(ck_a["encoder"][key], ck_b["encoder"][key], atol=1e-6), f"Mismatch in {key}"


def test_resume_after_a_crash_between_checkpoints_has_no_duplicate_epochs(cfg, monkeypatch):
    cfg.update(probe_every=2, max_epochs=4)
    call_count = [0]
    original_evaluate = pt.evaluate

    def evaluate_with_interrupt(*args, **kwargs):
        call_count[0] += 1
        if call_count[0] == 2:  # 2nd call is at epoch 4 (since probe_every=2, calls at epochs 2 and 4)
            raise KeyboardInterrupt()
        return original_evaluate(*args, **kwargs)

    # Run and interrupt
    monkeypatch.setattr(pt, "evaluate", evaluate_with_interrupt)
    try:
        pt.run(cfg, log=lambda s: None)
    except KeyboardInterrupt:
        pass
    finally:
        monkeypatch.setattr(pt, "evaluate", original_evaluate)

    run_dir = list(Path(cfg["run_root"]).glob("*_tiny_run"))[0]

    # Resume
    cfg_resume = pt.load_config(run_dir / "config.yaml", max_epochs=4)
    pt.run(cfg_resume, run_dir=run_dir, resume=True, log=lambda s: None)

    # Check metrics
    lines = (run_dir / "metrics.jsonl").read_text().splitlines()
    loss_epochs = [json.loads(l)["epoch"] for l in lines if '"loss"' in l]
    end_lines = [l for l in lines if '"end"' in l]

    assert loss_epochs == [1, 2, 3, 4], f"Got epochs: {loss_epochs}"
    assert len(end_lines) == 1


def test_best_checkpoint_holds_the_best_epoch_for_plan3(cfg, monkeypatch):
    from foundation.model.encoder import PRESETS, SpectralEncoder

    scores = [0.5, 0.7, 0.6]
    call_count = [0]

    def evaluate_with_scores(*args, **kwargs):
        score = scores[call_count[0]]
        call_count[0] += 1
        return {
            "probe_score": score, "knn_score": 0.5, "logreg_score": 0.5, "per_task": {},
            "rankme": 1.0, "source_id_bacc": 0.5
        }

    monkeypatch.setattr(pt, "evaluate", evaluate_with_scores)
    cfg.update(max_epochs=3, probe_every=1, patience=3)
    run_dir, _ = pt.run(cfg, log=lambda s: None)

    # Load ckpt_best.pt
    ck_best = torch.load(run_dir / "ckpt_best.pt", weights_only=True)

    # Verify structure
    assert set(ck_best.keys()) == {"encoder", "preset", "epoch", "probe"}
    assert ck_best["epoch"] == 2
    assert ck_best["probe"]["probe_score"] == 0.7

    # Verify encoder can be loaded
    enc = SpectralEncoder(PRESETS[ck_best["preset"]])
    enc.load_state_dict(ck_best["encoder"])  # Should not raise


def test_resuming_a_finished_run_leaves_one_end_line(cfg):
    # Run to completion (finishes normally with end line)
    run_dir, _ = pt.run(cfg, log=lambda s: None)

    # Resume to add more epochs
    cfg_resume = pt.load_config(run_dir / "config.yaml", max_epochs=3)
    pt.run(cfg_resume, run_dir=run_dir, resume=True, log=lambda s: None)

    # Parse metrics
    lines = (run_dir / "metrics.jsonl").read_text().splitlines()
    records = [json.loads(l) for l in lines]

    # Verify exactly one end line, and it's the last one
    end_lines = [r for r in records if "end" in r]
    assert len(end_lines) == 1
    assert records[-1] == end_lines[0]

    # Verify loss epochs are [1, 2, 3]
    loss_epochs = [r["epoch"] for r in records if "loss" in r]
    assert loss_epochs == [1, 2, 3]

    # Verify probe epochs are [1, 2, 3]
    probe_epochs = [r["epoch"] for r in records if "probe" in r]
    assert probe_epochs == [1, 2, 3]


def test_trim_metrics_drops_stale_probe_and_end_lines_and_bad_json(tmp_path):
    # Create a metrics file with good lines, stale lines, end line, and bad JSON
    metrics_file = tmp_path / "metrics.jsonl"
    lines = [
        json.dumps({"epoch": 1, "loss": 0.5}),  # Keep
        json.dumps({"epoch": 1, "probe": {"probe_score": 0.6}, "improved": True}),  # Keep
        json.dumps({"epoch": 2, "loss": 0.4}),  # Keep
        json.dumps({"epoch": 2, "probe": {"probe_score": 0.7}, "improved": True}),  # Keep
        json.dumps({"epoch": 3, "loss": 0.3}),  # Drop (stale)
        json.dumps({"epoch": 3, "probe": {"probe_score": 0.65}, "improved": False}),  # Drop (stale)
        json.dumps({"end": "finished 2/2", "track": {"best_epoch": 2}}),  # Drop (end line)
        '{"epoch": 4, "lo',  # Drop (truncated/bad JSON)
    ]
    metrics_file.write_text("\n".join(lines), encoding="utf-8")

    # Trim to last_epoch=2
    pt._trim_metrics(metrics_file, 2)

    # Verify result
    result = metrics_file.read_text(encoding="utf-8").splitlines()
    kept = [json.loads(l) for l in result if l]

    # Should have exactly 4 records: epochs 1 and 2, loss and probe
    assert len(kept) == 4
    epochs = [r["epoch"] for r in kept]
    assert epochs == [1, 1, 2, 2]

    # Verify no end lines, no stale epochs, no bad JSON
    assert all("end" not in r for r in kept)
    assert all(r["epoch"] <= 2 for r in kept)
