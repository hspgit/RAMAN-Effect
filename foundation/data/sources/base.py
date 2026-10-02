"""Source adapter interface (spec section 4.1).

Adapter contract: iter_batches yields RawBatch objects of at most 50,000
spectra, and all spectra of one sample/map are in the same batch, so that
build_corpus can apply per-sample caps.
"""
from __future__ import annotations

import hashlib
import numbers
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Protocol, Sequence

import numpy as np
import pandas as pd
import requests

DOMAINS = ("bio", "mineral", "chem")
MAX_BATCH = 50_000


def _norm_label(v):
    if v is None:
        return None
    if pd.isna(v):
        return None
    if isinstance(v, bytes):
        v = v.decode(errors="replace")
    elif isinstance(v, numbers.Real) and not isinstance(v, bool):
        if v == int(v):
            v = int(v)
    s = str(v).strip()
    return s or None


@dataclass
class RawBatch:
    axes: object                       # 1D shared axis, or one axis per spectrum
    spectra: object                    # (n, p) array, or a sequence of 1D arrays
    sample_id: Sequence
    label: Sequence
    laser_nm: object                   # scalar or length-n sequence; NaN = unknown

    def __post_init__(self):
        n = len(self.spectra)
        if n > MAX_BATCH:
            raise ValueError(f"batch of {n} spectra exceeds {MAX_BATCH}")
        shared = isinstance(self.axes, np.ndarray) and self.axes.ndim == 1
        if not shared and len(self.axes) != n:
            raise ValueError(f"{len(self.axes)} axes for {n} spectra")
        if len(self.sample_id) != n or len(self.label) != n:
            raise ValueError(f"sample_id/label lengths {len(self.sample_id)}/"
                             f"{len(self.label)} do not match {n} spectra")
        self.sample_id = [str(s) for s in self.sample_id]
        self.label = [_norm_label(v) for v in self.label]
        if np.isscalar(self.laser_nm):
            self.laser_nm = np.full(n, float(self.laser_nm))
        self.laser_nm = np.asarray(self.laser_nm, dtype=np.float64)
        if self.laser_nm.shape != (n,):
            raise ValueError(f"laser_nm has shape {self.laser_nm.shape}, expected ({n},)")


class Source(Protocol):
    name: str
    domain: str                        # one of DOMAINS
    licence: str
    url: str
    probe_label_kind: str | None       # what `label` means, or None if unlabelled

    def download(self, raw_dir: Path) -> None: ...

    def iter_batches(self, raw_dir: Path) -> Iterator[RawBatch]: ...


class ManualDownloadRequired(RuntimeError):
    """The source cannot be fetched automatically; the message says how to get it."""


THROTTLE_STATUSES = (429, 503)


def _retry_after(response):
    value = (getattr(response, "headers", {}) or {}).get("Retry-After")
    try:
        return max(0.0, float(value))
    except (TypeError, ValueError):
        return None


def fetch(url, dest, sha256=None, timeout=60, retries=3, backoff=2.0,
          throttle_retries=8, throttle_wait=30.0, max_throttle_wait=600.0) -> Path:
    """Download to `dest` via `dest.part`; rename only on success.

    Network errors and 5xx replies are retried `retries` times (some hosts,
    e.g. rruff.net, drop connections). A throttled reply (HTTP 429/503) waits
    the server's Retry-After, or an exponential backoff from `throttle_wait`
    seconds, up to `throttle_retries` times without using up `retries`
    (scidb.cn throttles bursts). Other 4xx replies are permanent: no retry.
    """
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    attempt = throttled = 0
    while True:
        hasher = hashlib.sha256() if sha256 is not None else None
        try:
            with requests.get(url, stream=True, timeout=timeout) as r:
                if (getattr(r, "status_code", 200) in THROTTLE_STATUSES
                        and throttled < throttle_retries):
                    wait = _retry_after(r)
                    if wait is None:
                        wait = min(throttle_wait * 2 ** throttled, max_throttle_wait)
                    throttled += 1
                    time.sleep(wait)
                    continue
                r.raise_for_status()
                bytes_written = 0
                with open(part, "wb") as fh:
                    for block in r.iter_content(chunk_size=1 << 20):
                        fh.write(block)
                        if hasher is not None:
                            hasher.update(block)
                        bytes_written += len(block)
                # Check for truncation based on Content-Length header
                headers = getattr(r, "headers", {}) or {}
                content_length = headers.get("Content-Length")
                content_encoding = headers.get("Content-Encoding")
                if content_length is not None and content_encoding is None:
                    expected = int(content_length)
                    if bytes_written != expected:
                        part.unlink(missing_ok=True)
                        raise IOError(f"truncated download for {url}: got {bytes_written}, "
                                    f"expected {expected}")
            break
        except requests.HTTPError as e:
            part.unlink(missing_ok=True)
            status = getattr(e.response, "status_code", None)
            attempt += 1
            if (status is not None and 400 <= status < 500) or attempt >= retries:
                raise
            time.sleep(backoff * attempt)
        except (requests.RequestException, ConnectionError, TimeoutError, IOError):
            part.unlink(missing_ok=True)
            attempt += 1
            if attempt >= retries:
                raise
            time.sleep(backoff * attempt)
    if sha256 is not None:
        digest = hasher.hexdigest()
        if digest != sha256:
            part.unlink()
            raise IOError(f"checksum mismatch for {url}: got {digest}")
    part.replace(dest)
    return dest


REGISTRY: dict = {}


def register(source):
    if source.domain not in DOMAINS:
        raise ValueError(f"{source.name}: domain {source.domain!r} not in {DOMAINS}")
    REGISTRY[source.name] = source
    return source
