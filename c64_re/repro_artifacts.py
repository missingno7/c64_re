"""Helpers for reproducible crash/divergence artifacts.

These helpers intentionally live in ``c64_re`` because they are generic runtime
forensics: write a snapshot plus a small manifest explaining why it was captured.
Game-specific code decides when to call them and what metadata to attach.

Origin: faithful port of ``dos_re.repro_artifacts``.  dos_re hand-copied its
VM/DOS state into a detached clone; here :mod:`c64_re.snapshot` already owns
full-machine freeze/thaw, so :func:`clone_runtime_state` delegates to
:func:`c64_re.snapshot.clone_runtime` and the artifact is a directory holding
one ``.c64snap`` file (dos_re snapshots were directories) plus ``repro.json``.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping

from .snapshot import clone_runtime, write_snapshot

SNAPSHOT_FILE_NAME = "snapshot.c64snap"


def safe_artifact_part(text: str) -> str:
    """Return a filesystem-friendly artifact name component."""
    out = []
    for ch in str(text):
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        elif ch in (" ", ":", "/", "\\", "."):
            out.append("_")
    cleaned = "".join(out).strip("_")
    return cleaned or "artifact"


def clone_runtime_state(src, *, install_hooks: bool = False):
    """Return a detached runtime clone suitable for later repro snapshots.

    This intentionally clones full machine state, not frontend callbacks or
    verifier progress hooks (``install_hooks=False`` yields a pure-ASM clone).
    It is used when a verifier must preserve the state *before* a candidate
    hook/frame mutates the live runtime.  Thin delegation:
    :func:`c64_re.snapshot.clone_runtime` is the freeze/thaw owner.
    """
    return clone_runtime(src, install_hooks=install_hooks)


def write_runtime_repro_snapshot(
    rt,
    *,
    root: str | Path,
    name: str,
    status: str,
    metadata: Mapping[str, Any] | None = None,
    trace_tail: Iterable[str] = (),
    timestamp: datetime | None = None,
) -> Path:
    """Write a timestamped runtime snapshot plus a small repro manifest.

    The returned directory holds ``snapshot.c64snap`` — directly loadable with
    :func:`c64_re.snapshot.load_snapshot` — next to ``repro.json``.  The
    manifest is intentionally best-effort metadata for humans/tools; the
    canonical VM state remains the snapshot file.
    """
    stamp = (timestamp or datetime.now()).strftime("%Y%m%d_%H%M%S")
    out = Path(root) / f"{safe_artifact_part(name)}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    write_snapshot(rt, out / SNAPSHOT_FILE_NAME)
    manifest = {
        "version": 1,
        "kind": "runtime_snapshot",
        "status": status,
        "snapshot": SNAPSHOT_FILE_NAME,
        "created_at": stamp,
        "cpu_addr": f"${rt.cpu.addr():04X}",
        "steps": rt.cpu.instr_count,
        "trace_tail": list(trace_tail),
        "metadata": dict(metadata or {}),
    }
    (out / "repro.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    return out
