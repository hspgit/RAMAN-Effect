import os

import numpy as np
import ramanspy as rp
import torch
from torch.utils.data import Dataset

from src.cache import file_fingerprint, load_or_compute

# Declarative description of the preprocessing pipeline.
#
# This tuple is BOTH the cache key and the build instruction (see
# build_pipeline), so the two cannot go out of sync. Changing a parameter here
# changes the cache key automatically, which is what makes Stage 1 ablations
# safe: passing a modified spec cannot collide with the baseline's cached array.
#

LEGACY_FULL_PIPELINE = (
    {"module": "despike", "method": "WhitakerHayes", "kwargs": {}},
    {"module": "denoise", "method": "SavGol", "kwargs": {"window_length": 9, "polyorder": 3}},
    {"module": "baseline", "method": "ASPLS", "kwargs": {}},
    {"module": "normalise", "method": "MinMax", "kwargs": {}},
)
DEFAULT_PIPELINE = (
    {"module": "normalise", "method": "MinMax", "kwargs": {}},
)

# Bump this if you change how the array is produced in a way the spec above
# does not capture (dtype, ordering, the container class). Manual escape hatch.
CACHE_FORMAT = 1

def _without(spec, *methods):
    return tuple(s for s in spec if s["method"] not in methods)


PIPELINE_VARIANTS = {
    # Current default, and its equivalents.
    "minmax_only": DEFAULT_PIPELINE,
    "none": (),
    # The old four-step pipeline and its single-removal arms, kept so every
    # number recorded before this change stays reproducible.
    "legacy_full": LEGACY_FULL_PIPELINE,
    "no_aspls": _without(LEGACY_FULL_PIPELINE, "ASPLS"),
    "no_despike": _without(LEGACY_FULL_PIPELINE, "WhitakerHayes"),
    "no_savgol": _without(LEGACY_FULL_PIPELINE, "SavGol"),
}

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

def build_pipeline(spec):
    """Instantiate a ramanspy Pipeline from a declarative spec.

    spec entries are {"module": <submodule of rp.preprocessing>,
                      "method": <class name>, "kwargs": {...}}
    """
    steps = []
    for s in spec:
        module = getattr(rp.preprocessing, s["module"])
        cls = getattr(module, s["method"])
        steps.append(cls(**s.get("kwargs", {})))
    return rp.preprocessing.Pipeline(steps)


class RamanDataset(Dataset):
    def __init__(
        self,
        X_path,
        y_path,
        label_mapping=None,
        apply_preprocessing=True,
        pipeline_spec=None,
        cache_dir=None,
        use_cache=True,
        full_hash=False,
        on_unseen="error",
        label_space=None,
        require_space=None,
    ):
        """
        Loads the Raman spectra data from numpy arrays.
        X shape: (N, L) -> expanded to (N, 1, L) for 1D CNNs
        y shape: (N,)

        pipeline_spec: override DEFAULT_PIPELINE. Pass () or [] for no
            preprocessing at all (equivalent to apply_preprocessing=False).
        cache_dir: where to store preprocessed arrays. Defaults to
            <dir of X_path>/.cache
        use_cache: set False to force recomputation without touching the cache.
        on_unseen: what to do when y contains labels absent from label_mapping.
            "error" (default) raises, "drop" keeps the old silent-drop
            behaviour but names the offending label values.
        label_space: override the name-based lookup in LABEL_SPACES. Use for
            files whose name is not in that table.
        require_space: assert the labels live in this space, and raise if they
            demonstrably do not. Opt-in at the call site, because loading a
            treatment-labelled file with a treatment mapping is legitimate.
        """
        if not os.path.exists(X_path) or not os.path.exists(y_path):
            raise FileNotFoundError(f"Missing data files: {X_path} or {y_path}")

        self.X = np.load(X_path).astype(np.float32)

        if pipeline_spec is None:
            pipeline_spec = DEFAULT_PIPELINE if apply_preprocessing else ()
        self.pipeline_spec = tuple(pipeline_spec)

        if self.pipeline_spec:
            data_dir = os.path.dirname(os.path.abspath(X_path))
            wavenumbers_path = os.path.join(data_dir, "wavenumbers.npy")
            if os.path.exists(wavenumbers_path):
                wavenumbers = np.load(wavenumbers_path)
            else:
                print(f"Warning: {wavenumbers_path} not found, using index axis.")
                wavenumbers = np.arange(self.X.shape[1])

            def _run_pipeline():
                print(f"Applying RamanSPy preprocessing to {X_path}...")
                raman_obj = rp.SpectralContainer(self.X, wavenumbers)
                processed = build_pipeline(self.pipeline_spec).apply(raman_obj)
                out = processed.spectral_data
                if not isinstance(out, np.ndarray):
                    out = np.array(out)
                return out.astype(np.float32)

            if use_cache:
                key_parts = {
                    "X": file_fingerprint(X_path, full_hash=full_hash),
                    "wavenumbers": (
                        file_fingerprint(wavenumbers_path, full_hash=full_hash)
                        if os.path.exists(wavenumbers_path)
                        else None
                    ),
                    "pipeline": list(self.pipeline_spec),
                    "dtype": "float32",
                    "cache_format": CACHE_FORMAT,
                }
                self.X = load_or_compute(
                    cache_dir or os.path.join(data_dir, ".cache"),
                    key_parts,
                    _run_pipeline,
                )
            else:
                self.X = _run_pipeline()

        # Add channel dimension for 1D CNN: (N, C, L) where C=1.
        # Done after the cache so cached arrays stay (N, L) and inspectable.
        self.X = np.expand_dims(self.X, axis=1)

        raw_y = np.load(y_path)
        if label_mapping is None:
            unique_labels = sorted(np.unique(raw_y))
            self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        else:
            self.label_mapping = label_mapping

        self.label_space = (
            label_space if label_space is not None
            else LABEL_SPACES.get(os.path.basename(y_path))
        )
        present = np.unique(raw_y)

        if require_space is not None:
            if self.label_space is None:
                # Unverifiable, not wrong: only the names in LABEL_SPACES can
                # be settled this way. Raising here would reject legitimate
                # files whose only sin is a name the table has not seen.
                print(f"WARNING: label space of {os.path.basename(y_path)} is unknown "
                      f"(not in LABEL_SPACES), so it cannot be checked against the "
                      f"required {require_space!r}. Pass label_space= to assert it.")
            elif self.label_space != require_space:
                raise ValueError(
                    f"{os.path.basename(y_path)} is in label space "
                    f"{self.label_space!r}, but a {require_space!r} mapping was "
                    "supplied. These spaces reuse the same integers for different "
                    "meanings, so the result would be silently meaningless."
                )

        # Generic backstop for files the name table cannot settle. Not proof of
        # a mismatch, but a strong smell: a legitimate subset is possible, a
        # 5-of-30 coincidence is not the common case. Skipped when the check
        # above positively confirmed the space, where it is only noise.
        space_confirmed = (
            require_space is not None and self.label_space == require_space
        )
        if not space_confirmed and len(present) < 0.5 * len(self.label_mapping):
            print(f"WARNING: {os.path.basename(y_path)} covers only {len(present)} "
                  f"of {len(self.label_mapping)} mapped classes "
                  f"({sorted(float(v) for v in present)}). Confirm this is an "
                  "intended subset and not a label-space mismatch.")

        # Unseen labels. The original code dropped them with only a count,
        # which means a new class silently vanishes and the run still reports a
        # plausible accuracy. That is the single most dangerous behaviour in the
        # pipeline for a project whose goal is accommodating new data.
        known = set(self.label_mapping)
        unseen = [u for u in present if u not in known]
        if unseen:
            counts = {float(u): int((raw_y == u).sum()) for u in unseen}
            msg = (
                f"{len(unseen)} label value(s) in {os.path.basename(y_path)} are absent "
                f"from label_mapping (which covers {len(self.label_mapping)} classes): "
                f"{counts}"
            )
            if on_unseen == "error":
                raise ValueError(
                    msg
                    + ". Pass on_unseen='drop' to discard them deliberately. "
                    "Note: the clinical splits carry antibiotic-level labels, not "
                    "isolate labels, so the reference mapping does not apply to them."
                )
            print("WARNING: dropping samples. " + msg)
            valid = np.isin(raw_y, list(known))
            self.X = self.X[valid]
            raw_y = raw_y[valid]

        mapped_y = np.array([self.label_mapping[label] for label in raw_y])
        self.y = mapped_y.astype(np.int64)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])


class AugmentedDataset(Dataset):
    """
    Wraps a PyTorch Dataset (or Subset) to apply on-the-fly augmentation.
    """

    def __init__(self, dataset, augment=False):
        self.dataset = dataset
        self.augment = augment

    def __len__(self):
        return len(self.dataset)

    def labels(self):
        """Label array for the wrapped dataset, for stratified splitting.

        Handles Subset by indexing into the parent's y.
        """
        ds = self.dataset
        if hasattr(ds, "y"):
            return np.asarray(ds.y)
        if hasattr(ds, "indices") and hasattr(ds, "dataset"):
            return np.asarray(ds.dataset.y)[np.asarray(ds.indices)]
        raise AttributeError("wrapped dataset exposes no labels")

    def __getitem__(self, idx):
        x, y = self.dataset[idx]

        if self.augment:
            # Convert back to numpy for augmentation
            x_np = x.numpy()

            # 1. Random shift (roll)
            shift = np.random.randint(-5, 6)
            x_np = np.roll(x_np, shift, axis=-1)

            # 2. Random Gaussian noise
            noise = np.random.normal(0, 0.01, x_np.shape).astype(np.float32)
            x_np = x_np + noise

            # 3. Random scale
            scale = np.random.uniform(0.95, 1.05)
            x_np = x_np * scale

            x = torch.tensor(x_np)

        return x, y


def stratified_split_indices(y, val_fraction, seed):
    """Class-balanced train/val index split.

    Used for the fine-tune validation split. The finetune set is 100 spectra
    per class, so a random 10% split would give some classes 5 validation
    samples and others 15. Stratifying gives every class exactly 10.

    Caveat worth remembering when reading the numbers: 10 samples per class is
    a small validation set. Per-class recall from it moves in steps of 0.1, and
    overall validation accuracy carries roughly +/-2 points of binomial noise
    at n=300. It is good enough to pick a better epoch than "the last one",
    and not good enough to tune hyperparameters on.
    """
    import numpy as np
    from sklearn.model_selection import StratifiedShuffleSplit

    y = np.asarray(y)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=val_fraction,
                                 random_state=seed)
    train_idx, val_idx = next(sss.split(np.zeros(len(y)), y))
    return train_idx, val_idx
