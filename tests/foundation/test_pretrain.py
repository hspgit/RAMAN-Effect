import json
import re

import numpy as np
import pytest

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
