"""Tests for foundation/tools/export_raman_data.py. No raman_data imports."""
import importlib.util
import json
import types
from pathlib import Path

import numpy as np
import pytest


def load_export_module():
    """Load the export script without importing raman_data."""
    # Path: tests/foundation/test_export_raman_data.py -> foundation/tools/export_raman_data.py
    export_script = Path(__file__).resolve().parents[2] / "foundation" / "tools" / "export_raman_data.py"
    spec = importlib.util.spec_from_file_location(
        "export_raman_data",
        str(export_script)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_dataset(spectra_shape=(10, 300), metadata=None):
    """Create a fake dataset object for testing."""
    if metadata is None:
        metadata = {"description": "test dataset"}

    ds = types.SimpleNamespace(
        spectra=np.ones(spectra_shape, dtype=np.float32),
        raman_shifts=np.linspace(400, 1800, spectra_shape[1]),
        targets=None,
        target_names=None,
        group_ids=None,
        info=types.SimpleNamespace(
            task_type="Regression",
            license="MIT",
            metadata=metadata
        )
    )
    return ds


def test_export_one_writes_npz_atomically(tmp_path):
    """export_one writes .npz atomically, no .tmp file left behind."""
    export = load_export_module()

    ds = fake_dataset((4, 300))
    entry = export.export_one("test_id", ds, tmp_path)

    # Check return value
    assert entry["status"] == "ok"
    assert entry["n"] == 4
    assert entry["licence"] == "MIT"
    assert entry["description"] == "test dataset"

    # Check .npz file exists and is readable
    npz_path = tmp_path / "test_id.npz"
    assert npz_path.exists()

    # Check no .tmp file
    assert not (tmp_path / "test_id.npz.tmp").exists()

    # Verify .npz content
    z = np.load(npz_path)
    assert z["spectra"].shape == (4, 300)
    assert z["spectra"].dtype == np.float32
    assert z["axis"].shape == (300,)
    assert z["axis"].dtype == np.float64
    assert z["labels"].size == 0
    assert z["group_ids"].size == 0


def test_failed_export_leaves_no_npz(tmp_path):
    """Failed export deletes stale .npz and .tmp files."""
    export = load_export_module()

    # Pre-create stale .npz and .tmp files to simulate a previous failed run
    stale_npz = tmp_path / "test_id.npz"
    stale_npz.write_text("stale data")
    assert stale_npz.exists()

    stale_tmp = tmp_path / "test_id.npz.tmp"
    stale_tmp.write_text("stale tmp data")
    assert stale_tmp.exists()

    # Call discard (what main() does in the except block on failure)
    export.discard("test_id", tmp_path)

    # Assert both .npz and .tmp are gone
    assert not stale_npz.exists()
    assert not stale_tmp.exists()


def test_discard_missing_ok(tmp_path):
    """discard() handles missing files gracefully (missing_ok)."""
    export = load_export_module()

    # Call discard on non-existent dataset
    export.discard("nonexistent", tmp_path)

    # Should not raise
    assert not (tmp_path / "nonexistent.npz").exists()
    assert not (tmp_path / "nonexistent.npz.tmp").exists()
