import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from foundation import evaluate
from foundation.adapt import seeds, sweep
from foundation.arms import ARMS


def _write_cfg(cfg, tmp_path):
    p = tmp_path / "adapt.yaml"
    p.write_text(yaml.safe_dump(cfg))
    return str(p)


@pytest.fixture
def finished(adapt_cfg, tmp_path):
    """Every planned run (4 arms x 2 tasks x 2 seeds) finished on fake data."""
    sweep(adapt_cfg, log=lambda *_: None)
    seeds(adapt_cfg, log=lambda *_: None)
    return adapt_cfg, _write_cfg(adapt_cfg, tmp_path)


def test_refuses_before_runs_finish(adapt_cfg, tmp_path):
    with pytest.raises(FileNotFoundError, match="sweep"):
        evaluate.main(["--config", _write_cfg(adapt_cfg, tmp_path), "--skip-curve"])
    sweep(adapt_cfg, tasks=["T1", "T2"], arms=["LP"], log=lambda *_: None)
    sel = Path(adapt_cfg["run_root"]) / "T1" / "selection.json"
    s = json.loads(sel.read_text())
    s.update({a: {"lr": 1e-3, "grid": []} for a in ARMS if a != "LP"})
    sel.write_text(json.dumps(s))
    t2 = Path(adapt_cfg["run_root"]) / "T2" / "selection.json"
    t2.write_text(sel.read_text())
    with pytest.raises(SystemExit, match="not finished"):
        evaluate.main(["--config", _write_cfg(adapt_cfg, tmp_path), "--skip-curve"])


def test_curve_must_be_finished_or_explicitly_skipped(finished):
    cfg, path = finished
    with pytest.raises(SystemExit, match="--skip-curve"):
        evaluate.main(["--config", path])


def test_single_test_pass_and_report(finished, tmp_path):
    cfg, path = finished
    results = tmp_path / "results"
    evaluate.main(["--config", path, "--skip-curve", "--results", str(results)])
    out = Path(cfg["run_root"]) / "final_test"
    scored = json.loads((out / "scored.json").read_text())
    assert len(scored["runs"]) == 16 and scored["curve"] is False
    assert np.load(out / "T1" / "B2_seed2.npy").shape == (30,)
    assert np.load(out / "T2" / "y_test.npy").shape == (20,)
    text = (results / "adaptation.md").read_text(encoding="utf-8")
    for s in ("## T1", "## T2", "B1 vs S", "B2 vs S", "B1 vs B2", "**Q1**", "**Q2**",
              "Q3 overall", "Reuse deliverable", "Known limitation", "Per-class recall"):
        assert s in text, s
    assert (results / "adaptation" / "confusion_T1.png").exists()
    rep = json.loads((out / "report.json").read_text())
    assert set(rep["tasks"]["T2"]) == set(ARMS)
    assert len(rep["diffs"]["T1"]["B1 vs S"]["mcnemar"]) == 2       # one test per paired seed
    with pytest.raises(SystemExit, match="already scored"):
        evaluate.main(["--config", path, "--skip-curve", "--results", str(results)])
    (results / "adaptation.md").unlink()
    evaluate.main(["--config", path, "--report-only", "--results", str(results)])
    assert (results / "adaptation.md").read_text(encoding="utf-8") == text


def test_report_only_needs_a_scored_pass(finished):
    _, path = finished
    with pytest.raises(SystemExit, match="not been scored"):
        evaluate.main(["--config", path, "--report-only"])


def test_scored_predictions_match_the_trained_models(finished):
    """The add-on reloaded for testing must reproduce the run's own validation predictions."""
    import torch

    from foundation.adapt import GPUSplit, load_task, predict, run_dir_for
    from foundation.arms import build_model, load_a, load_addon
    cfg, _ = finished
    preset, a_state = load_a(cfg["a_ckpt"])
    task = load_task("T2", cfg)
    sel = json.loads((Path(cfg["run_root"]) / "T2" / "selection.json").read_text())
    for arm in ARMS:
        rdir = run_dir_for(cfg, "T2", arm, sel[arm]["lr"], 2)
        s = json.loads((rdir / "summary.json").read_text())
        model, _ = build_model(arm, 5, preset, a_state, 2, lora_rank=s["lora_rank"])
        load_addon(model, torch.load(rdir / "addon.pt", weights_only=True))
        pred = predict(model, GPUSplit(task.splits["val"], torch.device("cpu")),
                       torch.device("cpu")).argmax(1).numpy()
        assert np.array_equal(pred, np.load(rdir / "val_pred.npy")), arm


def test_unpaired_seeds_raise():
    c = np.ones(4, bool)
    res = {"T1": {a: {"lr": 1e-3, "seeds": [1, 2] if a != "S" else [1, 3], "correct": [c, c],
                      "acc": [1.0, 1.0], "f1": [1.0, 1.0], "cm": np.eye(2, dtype=int),
                      "summ": [{"val_acc": 1, "trainable_params": 10, "trainable_fraction": 0.01,
                                "addon_bytes": 100, "seconds": 1.0}] * 2} for a in ARMS}}
    with pytest.raises(ValueError, match="not paired"):
        evaluate.analyse(res)


def _crash_on_third_load(monkeypatch):
    real, calls = evaluate.load_addon, {"n": 0}

    def flaky(model, state):
        calls["n"] += 1
        if calls["n"] == 3:
            raise RuntimeError("simulated crash while scoring")
        return real(model, state)

    monkeypatch.setattr(evaluate, "load_addon", flaky)
    return real


def _predictions(out, task):
    return sorted(p.name for p in (out / task).glob("*.npy") if p.name != "y_test.npy")


def _interrupted(finished, tmp_path, monkeypatch):
    cfg, path = finished
    results = tmp_path / "results"
    real = _crash_on_third_load(monkeypatch)
    args = ["--config", path, "--skip-curve", "--results", str(results)]
    with pytest.raises(RuntimeError, match="simulated crash"):
        evaluate.main(args)
    monkeypatch.setattr(evaluate, "load_addon", real)
    return cfg, path, args, Path(cfg["run_root"]) / "final_test"


def test_crash_mid_scoring_leaves_a_started_lock_and_resumes(finished, tmp_path, monkeypatch, capsys):
    cfg, path, args, out = _interrupted(finished, tmp_path, monkeypatch)
    lock = json.loads((out / "scored.json").read_text())
    assert lock["status"] == "started" and len(lock["runs"]) == 16
    assert len(_predictions(out, "T1")) == 2 and _predictions(out, "T2") == []
    first = {n: (out / "T1" / n).read_bytes() for n in _predictions(out, "T1")}
    capsys.readouterr()
    evaluate.main(args)
    assert "resuming an interrupted scoring pass" in capsys.readouterr().out
    done = json.loads((out / "scored.json").read_text())
    assert done["status"] == "complete" and done["runs"] == lock["runs"]
    assert len(_predictions(out, "T1")) + len(_predictions(out, "T2")) == 16
    assert all((out / "T1" / n).read_bytes() == b for n, b in first.items())   # kept, not rescored
    with pytest.raises(SystemExit, match="already scored"):
        evaluate.main(args)


def test_started_lock_with_changed_addon_is_refused(finished, tmp_path, monkeypatch):
    import torch
    cfg, path, args, out = _interrupted(finished, tmp_path, monkeypatch)
    key = next(iter(json.loads((out / "scored.json").read_text())["runs"]))
    addon = Path(json.loads((out / "scored.json").read_text())["runs"][key]["run_dir"]) / "addon.pt"
    state = torch.load(addon, weights_only=True)
    name = next(iter(state))
    state[name] = state[name] + 1.0
    torch.save(state, addon)
    with pytest.raises(SystemExit, match="interrupted scoring pass"):
        evaluate.main(args)
    assert json.loads((out / "scored.json").read_text())["status"] == "started"


def test_report_only_refuses_an_interrupted_pass(finished, tmp_path, monkeypatch):
    cfg, path, args, out = _interrupted(finished, tmp_path, monkeypatch)
    with pytest.raises(SystemExit, match="interrupted"):
        evaluate.main(["--config", path, "--report-only", "--results", str(tmp_path / "results")])


def test_model_a_changed_after_training_is_refused(finished, tmp_path):
    from conftest import make_fake_a
    cfg, path = finished
    make_fake_a(Path(cfg["a_ckpt"]), seed=1)
    with pytest.raises(SystemExit, match="model A") as e:
        evaluate.main(["--config", path, "--skip-curve", "--results", str(tmp_path / "results")])
    assert "T1/S_seed1" in str(e.value)
    assert not (Path(cfg["run_root"]) / "final_test" / "scored.json").exists()


def test_rebuilt_model_must_reproduce_its_validation_predictions(finished, tmp_path):
    cfg, path = finished
    sel = json.loads((Path(cfg["run_root"]) / "T1" / "selection.json").read_text())
    from foundation.adapt import run_dir_for
    vp = run_dir_for(cfg, "T1", "LP", sel["LP"]["lr"], 1) / "val_pred.npy"
    np.save(vp, (np.load(vp) + 1) % 3)
    with pytest.raises(RuntimeError, match="does not reproduce") as e:
        evaluate.main(["--config", path, "--skip-curve", "--results", str(tmp_path / "results")])
    assert "T1/LP_seed1" in str(e.value) and "3" in str(e.value)
    out = Path(cfg["run_root"]) / "final_test"
    assert json.loads((out / "scored.json").read_text())["status"] == "started"
    assert not (out / "T1" / "LP_seed1.npy").exists()
    assert _predictions(out, "T1") == ["S_seed1.npy", "S_seed2.npy"]


def test_resume_rescores_a_truncated_prediction(finished, tmp_path, monkeypatch):
    cfg, path, args, out = _interrupted(finished, tmp_path, monkeypatch)
    victim = out / "T1" / _predictions(out, "T1")[0]
    victim.write_bytes(victim.read_bytes()[:10])
    evaluate.main(args)
    assert json.loads((out / "scored.json").read_text())["status"] == "complete"
    assert np.load(victim).shape == (30,)
    assert not list(out.rglob("*.tmp"))


def test_lock_records_model_a_and_logs_the_val_check(finished, tmp_path, capsys):
    from foundation.arms import a_fingerprint
    cfg, path = finished
    evaluate.main(["--config", path, "--skip-curve", "--results", str(tmp_path / "results")])
    assert "T1/S_seed1: val check 3/3 |" in capsys.readouterr().out
    scored = json.loads((Path(cfg["run_root"]) / "final_test" / "scored.json").read_text())
    assert scored["a_sha1"] == a_fingerprint(cfg["a_ckpt"])


def test_started_lock_with_a_different_model_a_is_refused(finished, tmp_path, monkeypatch):
    cfg, path, args, out = _interrupted(finished, tmp_path, monkeypatch)
    lock = json.loads((out / "scored.json").read_text())
    lock["a_sha1"] = "0" * 40
    (out / "scored.json").write_text(json.dumps(lock))
    with pytest.raises(SystemExit, match="interrupted scoring pass"):
        evaluate.main(args)


def test_missing_config_is_an_error(tmp_path):
    with pytest.raises(SystemExit):
        evaluate.main(["--config", str(tmp_path / "missing.yaml"), "--skip-curve"])
