"""
Per-run manifest and artifact paths.

Fixes two recorded problems:

1. Config capture. The committed logs do not record their own arguments, so
   batch size had to be reconstructed by counting print strides. That is
   archaeology, not record keeping.
2. Checkpoint collisions. Every run wrote models/raman_reference_model.pth, so
   consecutive runs of different architectures overwrote each other.

Layout:

    runs/20260917-142530_legacy_cnn_a3f1c2/
        manifest.json    args + env, written at start
        metrics.json     appended to as the run progresses
        reference.pth
        finetuned.pth

Plain JSON, no new dependency.
"""
from __future__ import annotations

import json
import os
import time
import uuid
from typing import Any, Dict, Optional

import torch

from src.repro import describe_env


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    return str(obj)


class RunMeta:
    def __init__(
        self,
        args: Any,
        device: torch.device,
        run_root: str = "runs",
        tag: Optional[str] = None,
    ) -> None:
        argd: Dict[str, Any] = dict(args) if isinstance(args, dict) else dict(vars(args))
        stamp = time.strftime("%Y%m%d-%H%M%S")
        parts = [stamp, str(argd.get("model_type", "model"))]
        if tag:
            parts.append(str(tag))
        parts.append(uuid.uuid4().hex[:6])
        self.run_id = "_".join(parts)
        self.dir = os.path.join(run_root, self.run_id)
        os.makedirs(self.dir, exist_ok=True)

        self.manifest: Dict[str, Any] = {
            "run_id": self.run_id,
            "started": stamp,
            "args": _jsonable(argd),
            "env": _jsonable(describe_env(device)),
        }
        self._write("manifest.json", self.manifest)
        self.metrics: Dict[str, Any] = {"run_id": self.run_id}

    def _write(self, name: str, payload: Any) -> None:
        with open(os.path.join(self.dir, name), "w") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)

    def path(self, name: str) -> str:
        return os.path.join(self.dir, name)

    def log(self, key: str, value: Any) -> None:
        """Record a metric block and flush immediately.

        Flushing every call means a run that dies at epoch 60 still leaves 59
        epochs of usable history.
        """
        self.metrics[key] = _jsonable(value)
        self._write("metrics.json", self.metrics)

    def log_epoch(self, phase: str, epoch: int, **values: Any) -> None:
        self.metrics.setdefault("history", []).append(
            _jsonable({"phase": phase, "epoch": epoch, **values})
        )
        self._write("metrics.json", self.metrics)

    def finish(self, **summary: Any) -> None:
        self.metrics["finished"] = time.strftime("%Y%m%d-%H%M%S")
        for k, v in summary.items():
            self.metrics[k] = _jsonable(v)
        self._write("metrics.json", self.metrics)
        print(f"\n[run] artifacts in {self.dir}")
