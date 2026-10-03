import json

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


def test_recommend_prefers_smaller_preset_within_tie():
    mk = lambda preset, score: {"preset": preset, "probe_score": score, "run": preset}
    assert report.recommend([mk("tiny", 0.60), mk("small", 0.605), mk("base", 0.70)])["preset"] == "base"
    assert report.recommend([mk("tiny", 0.60), mk("small", 0.605), mk("base", 0.609)])["preset"] == "tiny"
    assert report.recommend([mk("small", None)]) is None


def test_table_has_one_row_per_run(finished_run):
    run_dir, _ = finished_run
    text = report.table([report.summarize(run_dir)])
    assert text.count("\n") == 2 and "tiny" in text
