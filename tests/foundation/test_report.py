import json

import numpy as np
import pytest

import foundation.pretrain as pt
from foundation.analysis import report


@pytest.fixture
def finished_run(tmp_path, fake_corpus):
    corpus = fake_corpus
    cfg = pt.load_config(None, preset="tiny", corpus_dir=str(corpus), device="cpu",
                         run_root=str(tmp_path / "runs"), batch_size=8, steps_per_epoch=2,
                         max_epochs=2, warmup_epochs=1, probe_every=1, amp=False, n_recon=4)
    run_dir, _ = pt.run(cfg, log=lambda s: None)
    return run_dir, pt.Corpus(corpus)


def test_report_writes_plots_and_summary(finished_run):
    import torch
    run_dir, corpus = finished_run
    s = report.report_run(run_dir, corpus, torch.device("cpu"))
    for name in ("curves.png", "recon.png", "embedding.png", "summary.json"):
        assert (run_dir / "analysis" / name).exists(), name
    assert s["preset"] == "tiny" and s["epochs_run"] == 2 and s["best_epoch"] in (1, 2)
    assert s["g1_pass"] == (s["knn_score"] > s["raw_knn_score"])
    assert json.loads((run_dir / "analysis" / "summary.json").read_text())["run"] == run_dir.name
    rec = np.load(run_dir / f"recon_epoch{s['best_epoch']:04d}.npz")
    assert "pred_normalised" in rec and "target_normalised" in rec
    assert rec["pred_normalised"].shape == rec["target_normalised"].shape == (4, 101, 10)
    assert rec["pred_normalised"].dtype == rec["target_normalised"].dtype == np.float32


def test_recon_stats_compare_the_model_with_predicting_zero():
    r = {"hidden_tokens": np.zeros((1, 101), bool), "mask": np.ones((1, 1001), bool),
         "target_normalised": np.random.default_rng(0).normal(size=(1, 101, 10)).astype(np.float32)}
    r["hidden_tokens"][0, 10:20] = True
    r["pred_normalised"] = r["target_normalised"].copy()
    mse, zero = report.recon_stats(r, 0)
    assert mse == pytest.approx(0.0, abs=1e-12) and zero > 0.3
    r["pred_normalised"] = np.zeros_like(r["target_normalised"])
    mse, zero2 = report.recon_stats(r, 0)
    assert mse == pytest.approx(zero) and zero2 == pytest.approx(zero)
    assert report.recon_stats({k: v for k, v in r.items() if k != "pred_normalised"}, 0) is None


def test_plot_recon_handles_an_old_npz_without_normalised_arrays(finished_run, tmp_path):
    run_dir, _ = finished_run
    src = np.load(next(run_dir.glob("recon_epoch*.npz")))
    old = {k: src[k] for k in ("clean", "mask", "corrupted", "corrupted_mask",
                               "hidden_tokens", "reconstruction")}
    np.savez(tmp_path / "old.npz", **old)
    report.plot_recon(tmp_path / "old.npz", tmp_path / "old.png")
    assert (tmp_path / "old.png").exists()


def test_recommend_prefers_smaller_preset_within_tie():
    mk = lambda preset, score: {"preset": preset, "probe_score": score, "run": preset}
    assert report.recommend([mk("tiny", 0.60), mk("small", 0.605), mk("base", 0.70)])["preset"] == "base"
    assert report.recommend([mk("tiny", 0.60), mk("small", 0.605), mk("base", 0.609)])["preset"] == "tiny"
    assert report.recommend([mk("small", None)]) is None


def test_table_has_one_row_per_run(finished_run):
    run_dir, _ = finished_run
    text = report.table([report.summarize(run_dir)])
    assert text.count("\n") == 2 and "tiny" in text


def test_summarize_survives_a_truncated_metrics_line(finished_run):
    run_dir, _ = finished_run
    # Get baseline values before truncation
    baseline = report.summarize(run_dir)
    baseline_epochs_run = baseline["epochs_run"]
    baseline_best_epoch = baseline["best_epoch"]

    # Append a truncated JSON line to metrics.jsonl
    metrics_file = run_dir / "metrics.jsonl"
    with open(metrics_file, "a") as f:
        f.write('{"epoch": 9, "lo')

    # Should survive and return same values
    s = report.summarize(run_dir)
    assert s["epochs_run"] == baseline_epochs_run
    assert s["best_epoch"] == baseline_best_epoch


def test_best_epoch_follows_the_improved_flag(tmp_path):
    from pathlib import Path
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()

    # Write config.yaml
    (run_dir / "config.yaml").write_text("preset: tiny\nseed: 0\n")

    # Write manifest.json
    manifest = {"raw_knn_score": 0.5, "encoder_params": 1}
    (run_dir / "manifest.json").write_text(json.dumps(manifest))

    # Write metrics.jsonl with loss and probe lines
    metrics_lines = [
        json.dumps({"epoch": 1, "loss": 0.8, "seconds": 1.0}),
        json.dumps({"epoch": 2, "loss": 0.7, "seconds": 1.0}),
        json.dumps({"epoch": 1, "probe": {"probe_score": 0.6, "knn_score": 0.55, "logreg_score": 0.58,
                                           "per_task": {}, "rankme": 1.0, "source_id_bacc": 0.5}, "improved": True}),
        json.dumps({"epoch": 2, "probe": {"probe_score": 0.60005, "knn_score": 0.55005, "logreg_score": 0.58005,
                                           "per_task": {}, "rankme": 1.0, "source_id_bacc": 0.5}, "improved": False}),
    ]
    (run_dir / "metrics.jsonl").write_text("\n".join(metrics_lines))

    # Summarize should pick epoch 1 (the last improved=true)
    s = report.summarize(run_dir)
    assert s["best_epoch"] == 1
