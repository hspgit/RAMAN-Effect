# Raman foundation model experiment (A -> B1 / B2)

Spec: `docs/superpowers/specs/2026-10-01-raman-foundation-model-design.md`.
This package is self-contained and imports nothing from `src/` or `legacy_baseline/`.

## Phase 0: build the pretraining corpus

1. Optional sources:
   - **DeepeR** is a manual download. Put `Train_Outputs.mat`, `Test_Outputs.mat` and `axis.txt` into `foundation/data/raw/deeper/`. The link is in `foundation/data/sources/deeper.py`.
   - **raman_data datasets** are exported in a separate venv:
     ```
     python -m venv foundation/.venv-raman-data
     foundation/.venv-raman-data/Scripts/python.exe -m pip install "git+https://github.com/ml-lab-htw/raman_data@c92289e" pyyaml
     foundation/.venv-raman-data/Scripts/python.exe foundation/tools/export_raman_data.py
     ```
2. Build (downloads the automatic sources the first time):
   `venv/Scripts/python.exe -m foundation.data.build_corpus --config foundation/configs/corpus_v1.yaml`
3. Audit and gate G0:
   `venv/Scripts/python.exe -m foundation.data.audit --corpus foundation/data/corpus_v1`

Our Bacteria-ID files are read only to keep them out of the corpus (leakage guard).

## Phases 1-2: pretrain model A

Size trials (same 50 epochs each; about 0.8 / 1.2 / 2.0 h on an RTX 3060):
```
venv/Scripts/python.exe -m foundation.pretrain --config foundation/configs/pretrain_trial_tiny.yaml
venv/Scripts/python.exe -m foundation.pretrain --config foundation/configs/pretrain_trial_small.yaml
venv/Scripts/python.exe -m foundation.pretrain --config foundation/configs/pretrain_trial_base.yaml
```
Understand and compare them (plots land in each run's `analysis/`):
```
venv/Scripts/python.exe -m foundation.analysis.report foundation/runs/pretrain/<tiny-run> foundation/runs/pretrain/<small-run> foundation/runs/pretrain/<base-run> --out foundation/runs/pretrain/trials.md
```
Runs stop only at `max_epochs` or when the probe score is flat for 3 checkpoints, never on
time. Continue one that was still improving:
```
venv/Scripts/python.exe -m foundation.pretrain --resume foundation/runs/pretrain/<run> --max-epochs 300
```
Plan 3 loads `<run>/ckpt_best.pt` (`{"encoder", "preset", "epoch", "probe"}`).
