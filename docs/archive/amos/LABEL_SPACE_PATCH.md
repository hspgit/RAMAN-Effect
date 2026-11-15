# Label-space guard

The correctness fix I got wrong earlier. My acceptance test asserted the
unseen-label guard would raise on the clinical files. It did not, and it
cannot: `y_2018clinical.npy` contains labels {0, 2, 3, 5, 6}, all of which
exist in the reference mapping, so a value-level check sees nothing wrong.

The clinical files carry **antibiotic-level** labels while reference, finetune
and test carry **isolate-level** labels. Both label spaces start at 0 and use
small integers. A model evaluated on one using the other's mapping produces a
plausible-looking number that means nothing.

Two edits. Apply before anything that touches the clinical data.

---

## Edit 1: `src/dataset.py`, declare the spaces

Insert after `PIPELINE_VARIANTS`:

```python
# Which label space each file's labels live in. Reference, finetune and test
# use isolate IDs (30 bacterial/yeast isolates). The clinical files use
# empiric-treatment group IDs (the 5 most prevalent of 8 groups, which is why
# their labels are the non-contiguous subset {0, 2, 3, 5, 6}).
#
# These spaces reuse the same small integers for different meanings, so no
# value-level check can detect a mismatch. The file name is the only signal
# available without external metadata.
LABEL_SPACES = {
    "y_reference.npy": "isolate30",
    "y_finetune.npy": "isolate30",
    "y_test.npy": "isolate30",
    "y_2018clinical.npy": "treatment",
    "y_2019clinical.npy": "treatment",
}
```

## Edit 2: `src/dataset.py`, enforce it

Add two parameters to `RamanDataset.__init__`:

```python
        label_space=None,
        require_space=None,
```

Then, immediately after `self.label_mapping` is resolved and before the unseen
check:

```python
        self.label_space = label_space or LABEL_SPACES.get(os.path.basename(y_path))
        if require_space is not None and self.label_space != require_space:
            raise ValueError(
                f"{os.path.basename(y_path)} is in label space "
                f"{self.label_space!r}, but a {require_space!r} mapping was "
                "supplied. These spaces reuse the same integers for different "
                "meanings, so the result would be silently meaningless."
            )

        # Generic backstop for files not in the table. Not proof of a mismatch,
        # but a strong smell: a legitimate subset is possible, a 5-of-30
        # coincidence is not the common case.
        if len(present) < 0.5 * len(self.label_mapping):
            print(f"WARNING: {os.path.basename(y_path)} covers only {len(present)} "
                  f"of {len(self.label_mapping)} mapped classes "
                  f"({sorted(float(v) for v in present)}). Confirm this is an "
                  "intended subset and not a label-space mismatch.")
```

`present` is already computed a few lines below as `np.unique(raw_y)`; move that
line above this block.

## Edit 3: `main.py`, use it

On the finetune and test `RamanDataset(...)` calls, add:

```python
                                        require_space=full_ref_dataset.label_space,
```

---

## Verify

```powershell
python -c "from src.dataset import RamanDataset; r=RamanDataset('data/X_reference.npy','data/y_reference.npy'); RamanDataset('data/X_2018clinical.npy','data/y_2018clinical.npy',label_mapping=r.label_mapping,require_space=r.label_space)"
```

Must now raise `ValueError` naming `treatment` and `isolate30`. Without
`require_space` it still loads, which is correct: the guard is opt-in at the
call site, because there are legitimate reasons to load a treatment-labelled
file (with a treatment mapping).

Then confirm nothing regressed:

```powershell
python main.py --model-type legacy_cnn --epochs 1 --finetune-epochs 1 --val-split 0.33 --seed 99 --tag spacecheck
```
