import numpy as np
import pandas as pd
import pytest

from foundation.data.sources import base
from foundation.data.sources.base import RawBatch, fetch


def test_rawbatch_normalises_labels_ids_and_laser():
    b = RawBatch(axes=np.arange(10.0), spectra=np.zeros((4, 10)),
                 sample_id=[1, 1, 2, 2], label=[3, None, float("nan"), "quartz"],
                 laser_nm=532)
    assert b.sample_id == ["1", "1", "2", "2"]
    assert b.label == ["3", None, None, "quartz"]
    assert b.laser_nm.tolist() == [532.0] * 4


def test_rawbatch_rejects_length_mismatch():
    with pytest.raises(ValueError):
        RawBatch(axes=np.arange(10.0), spectra=np.zeros((3, 10)),
                 sample_id=["a", "b"], label=[None, None, None], laser_nm=532)


class _FakeResponse:
    def __init__(self, chunks, fail_after=None, headers=None):
        self.chunks, self.fail_after = chunks, fail_after
        self.headers = headers if headers is not None else {}

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def raise_for_status(self):
        pass

    def iter_content(self, chunk_size):
        for i, c in enumerate(self.chunks):
            if self.fail_after is not None and i == self.fail_after:
                raise ConnectionError("connection dropped")
            yield c


def test_interrupted_download_never_looks_complete(tmp_path, monkeypatch):
    dest = tmp_path / "file.zip"
    monkeypatch.setattr(base.requests, "get",
                        lambda *a, **k: _FakeResponse([b"aa", b"bb"], fail_after=1))
    with pytest.raises(ConnectionError):
        fetch("http://example.invalid/file.zip", dest, retries=2, backoff=0)
    assert not dest.exists()
    monkeypatch.setattr(base.requests, "get",
                        lambda *a, **k: _FakeResponse([b"aa", b"bb"]))
    fetch("http://example.invalid/file.zip", dest)
    assert dest.read_bytes() == b"aabb"
    assert not (tmp_path / "file.zip.part").exists()


def test_existing_file_is_not_downloaded_again(tmp_path, monkeypatch):
    dest = tmp_path / "file.zip"
    dest.write_bytes(b"done")

    def boom(*a, **k):
        raise AssertionError("should not download")

    monkeypatch.setattr(base.requests, "get", boom)
    assert fetch("http://example.invalid/file.zip", dest) == dest


def test_checksum_mismatch_is_rejected(tmp_path, monkeypatch):
    dest = tmp_path / "file.zip"
    monkeypatch.setattr(base.requests, "get", lambda *a, **k: _FakeResponse([b"aabb"]))
    with pytest.raises(IOError, match="checksum"):
        fetch("http://example.invalid/file.zip", dest, sha256="0" * 64)
    assert not dest.exists()


def test_transient_failure_is_retried(tmp_path, monkeypatch):
    dest = tmp_path / "file.zip"
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        return _FakeResponse([b"aa", b"bb"], fail_after=1 if calls["n"] == 1 else None)

    monkeypatch.setattr(base.requests, "get", flaky)
    fetch("http://example.invalid/file.zip", dest, retries=3, backoff=0)
    assert dest.read_bytes() == b"aabb" and calls["n"] == 2


def test_label_normalisation_handles_numpy_pandas_bytes_and_integral_floats():
    from foundation.data.sources.base import _norm_label
    assert _norm_label(np.float32('nan')) is None
    assert _norm_label(pd.NA) is None
    assert _norm_label(b'quartz') == 'quartz'
    assert _norm_label(3.0) == '3'
    assert _norm_label(np.int64(3)) == '3'
    assert _norm_label(2.5) == '2.5'
    assert _norm_label('  ') is None


def test_short_read_against_content_length_is_rejected(tmp_path, monkeypatch):
    dest = tmp_path / "file.zip"
    monkeypatch.setattr(base.requests, "get",
                        lambda *a, **k: _FakeResponse([b"aabb"], headers={"Content-Length": "10"}))
    with pytest.raises(IOError, match="truncated"):
        fetch("http://example.invalid/file.zip", dest, retries=1, backoff=0)
    assert not dest.exists()
    assert not (tmp_path / "file.zip.part").exists()


class _StatusResponse(_FakeResponse):
    """A response with an HTTP status; raise_for_status behaves like requests'."""

    def __init__(self, status, chunks=(b"ok",), headers=None):
        super().__init__(list(chunks), headers=headers)
        self.status_code = status

    def raise_for_status(self):
        if self.status_code >= 400:
            raise base.requests.HTTPError(f"{self.status_code} error", response=self)


def test_throttled_request_waits_retry_after_then_succeeds(tmp_path, monkeypatch):
    replies = [_StatusResponse(429, headers={"Retry-After": "7"}), _StatusResponse(200)]
    monkeypatch.setattr(base.requests, "get", lambda *a, **k: replies.pop(0))
    waits = []
    monkeypatch.setattr(base.time, "sleep", waits.append)
    fetch("http://example.invalid/f.txt", tmp_path / "f.txt", retries=1)
    assert (tmp_path / "f.txt").read_bytes() == b"ok"
    assert waits == [7.0]


def test_throttle_without_retry_after_backs_off_exponentially(tmp_path, monkeypatch):
    replies = [_StatusResponse(429), _StatusResponse(503), _StatusResponse(200)]
    monkeypatch.setattr(base.requests, "get", lambda *a, **k: replies.pop(0))
    waits = []
    monkeypatch.setattr(base.time, "sleep", waits.append)
    fetch("http://example.invalid/f.txt", tmp_path / "f.txt", retries=1, throttle_wait=30)
    assert waits == [30, 60]


def test_throttle_gives_up_after_throttle_retries(tmp_path, monkeypatch):
    monkeypatch.setattr(base.requests, "get", lambda *a, **k: _StatusResponse(429))
    monkeypatch.setattr(base.time, "sleep", lambda s: None)
    with pytest.raises(base.requests.HTTPError):
        fetch("http://example.invalid/f.txt", tmp_path / "f.txt", throttle_retries=3)
    assert not (tmp_path / "f.txt").exists()


def test_permanent_client_error_is_not_retried(tmp_path, monkeypatch):
    calls = {"n": 0}

    def not_found(*a, **k):
        calls["n"] += 1
        return _StatusResponse(404)

    monkeypatch.setattr(base.requests, "get", not_found)
    monkeypatch.setattr(base.time, "sleep", lambda s: None)
    with pytest.raises(base.requests.HTTPError):
        fetch("http://example.invalid/f.txt", tmp_path / "f.txt", retries=5)
    assert calls["n"] == 1
