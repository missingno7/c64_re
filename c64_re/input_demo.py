"""Deterministic input demos: record/replay VM-visible input by frame.

The dos_re design, C64 flavor.  A demo is a directory holding
``input_demo.json`` (manifest + events) and, unless it is a cold-start
demo, a ``snapshot.c64snap`` start snapshot.  Events are keyed by the
**boundary counter** — on C64 the VIC frame counter (``machine.vic.frame``),
the same clock ``run_frames`` steps by — so replaying into any runtime
resumed from the same snapshot reproduces the run bit-exactly.

Event kinds (all delivered through the same machine input API a human
uses, so recorded and live input are indistinguishable to the game):

- ``key_down`` / ``key_up`` — a C64 matrix key name (``cia.MATRIX``)
- ``joy1`` / ``joy2``       — the full 5-bit joystick mask after a change

Cold-start demos (``write_start_snapshot=False``) carry no snapshot:
playback boots a fresh runtime from the recorded ``boot_args`` and replays
from frame 0 — the input-only capture of a whole session from power-on.

Same-boundary ordering is preserved by ``seq``; ``single=True`` delivery
exists for fine-grained poll waits (a release and re-press recorded on the
same frame must be observable in two successive polls or the game's
debounce collapses two taps into one — the proven dos_re rule).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Sequence

from .cia import MATRIX
from .snapshot import load_snapshot, write_snapshot

DEMO_VERSION = 1
MANIFEST_NAME = "input_demo.json"
SNAPSHOT_NAME = "snapshot.c64snap"

KINDS = ("key_down", "key_up", "joy1", "joy2")


@dataclass(frozen=True)
class InputDemoEvent:
    boundary: int          # VIC frame, relative to the demo's start
    seq: int
    kind: str
    key: str = ""          # for key_down/key_up
    mask: int | None = None  # for joy1/joy2

    @classmethod
    def from_json(cls, raw: dict) -> "InputDemoEvent":
        return cls(
            boundary=max(0, int(raw.get("boundary", 0))),
            seq=max(0, int(raw.get("seq", 0))),
            kind=str(raw.get("kind", "")),
            key=str(raw.get("key", "")),
            mask=None if raw.get("mask") is None else int(raw["mask"]) & 0x1F,
        )

    def to_json(self) -> dict:
        out: dict = {"boundary": self.boundary, "seq": self.seq, "kind": self.kind}
        if self.key:
            out["key"] = self.key
        if self.mask is not None:
            out["mask"] = self.mask
        return out


class InputDemoRecorder:
    """Record a start snapshot plus frame-keyed input events.

    ``metadata`` is copied verbatim into the manifest; the runtime's
    ``boot_args`` are always recorded so cold-start playback (and honest
    provenance) need nothing external.
    """

    def __init__(self, *, root: str | Path, name: str,
                 metadata: dict | None = None) -> None:
        self.root = Path(root)
        self.name = _safe_demo_name(name)
        self.metadata = dict(metadata or {})
        self.demo_dir: Path | None = None
        self.start_boundary = 0
        self._seq = 0
        self._events: list[InputDemoEvent] = []
        self._started_at = ""
        self._cold_start = False
        self._boot_args: dict = {}

    @property
    def active(self) -> bool:
        return self.demo_dir is not None

    @property
    def event_count(self) -> int:
        return len(self._events)

    def start(self, rt, *, write_start_snapshot: bool = True) -> Path:
        if self.active:
            raise RuntimeError("input demo recording is already active")
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.demo_dir = self.root / f"demo_{self.name}_{stamp}"
        self.demo_dir.mkdir(parents=True, exist_ok=True)
        self.start_boundary = rt.machine.vic.frame
        self._seq = 0
        self._events.clear()
        self._started_at = datetime.now().isoformat(timespec="seconds")
        self._boot_args = dict(rt.boot_args)
        self._cold_start = not write_start_snapshot
        if write_start_snapshot:
            write_snapshot(rt, self.demo_dir / SNAPSHOT_NAME)
        elif rt.machine.vic.frame != 0 or rt.cpu.instr_count != 0:
            raise RuntimeError(
                "cold-start demos must start at frame 0 of a fresh boot "
                f"(frame={rt.machine.vic.frame}, instr={rt.cpu.instr_count})"
            )
        self._write_manifest(final=False)
        return self.demo_dir

    # ---- event capture (frontends call these alongside the machine API) ----
    def record_key_down(self, *, boundary: int, key: str) -> None:
        self._record(InputDemoEvent(self._rel(boundary), self._seq, "key_down", key=key))

    def record_key_up(self, *, boundary: int, key: str) -> None:
        self._record(InputDemoEvent(self._rel(boundary), self._seq, "key_up", key=key))

    def record_joy(self, *, boundary: int, port: int, mask: int) -> None:
        if port not in (1, 2):
            raise ValueError(f"joystick port must be 1 or 2, got {port}")
        self._record(InputDemoEvent(self._rel(boundary), self._seq,
                                    f"joy{port}", mask=mask & 0x1F))

    def stop(self, *, boundary: int) -> Path:
        if not self.active or self.demo_dir is None:
            raise RuntimeError("input demo recording is not active")
        self._write_manifest(final=True, end_boundary=self._rel(boundary))
        out = self.demo_dir
        self.demo_dir = None
        return out

    def _rel(self, boundary: int) -> int:
        return max(0, int(boundary) - self.start_boundary)

    def _record(self, event: InputDemoEvent) -> None:
        if not self.active:
            return
        if event.kind in ("key_down", "key_up") and event.key not in MATRIX:
            raise KeyError(f"unknown C64 key {event.key!r}")
        self._events.append(event)
        self._seq += 1
        self._write_manifest(final=False)  # crash-safe: manifest always current

    def _write_manifest(self, *, final: bool, end_boundary: int | None = None) -> None:
        if self.demo_dir is None:
            return
        manifest = {
            "version": DEMO_VERSION,
            "status": "complete" if final else "recording",
            "created_at": self._started_at,
            "snapshot": None if self._cold_start else SNAPSHOT_NAME,
            "boot_args": self._boot_args,
            "metadata": self.metadata,
            "start_boundary": 0,
            "end_boundary": end_boundary,
            "event_count": len(self._events),
            "events": [e.to_json() for e in self._events],
        }
        (self.demo_dir / MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2), encoding="utf-8")


class InputDemoPlayback:
    """Replay a recorded demo into one or more runtimes."""

    def __init__(self, *, demo_dir: Path, manifest: dict) -> None:
        self.demo_dir = demo_dir
        self.manifest = manifest
        self.events = sorted(
            (InputDemoEvent.from_json(raw) for raw in manifest.get("events", [])),
            key=lambda e: (e.boundary, e.seq),
        )
        self._index = 0
        # Event boundaries are relative to the demo's start; a runtime resumed
        # from the start snapshot reports absolute frames.  make_runtime sets
        # this base; set it yourself when wiring a runtime manually.
        self.start_base = 0

    @classmethod
    def load(cls, path: str | Path) -> "InputDemoPlayback":
        p = Path(path)
        manifest_path = p / MANIFEST_NAME if p.is_dir() else p
        demo_dir = manifest_path.parent
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if int(manifest.get("version", 0)) != DEMO_VERSION:
            raise ValueError(f"unsupported input demo version {manifest.get('version')!r}")
        return cls(demo_dir=demo_dir, manifest=manifest)

    # ---- runtime setup -------------------------------------------------------
    @property
    def is_cold_start(self) -> bool:
        return self.manifest.get("snapshot") is None

    def snapshot_path(self) -> Path:
        snap = self.manifest.get("snapshot")
        if snap is None:
            raise ValueError("cold-start demo has no start snapshot "
                             "(check .is_cold_start first)")
        return self.demo_dir / str(snap)

    def make_runtime(self, *, install_hooks: bool = True):
        """Build the runtime this demo replays into: resume the start
        snapshot, or cold-boot from the recorded boot_args.  Also sets
        :attr:`start_base` so absolute frame numbers key correctly."""
        if not self.is_cold_start:
            rt = load_snapshot(self.snapshot_path(), install_hooks=install_hooks)
            self.start_base = rt.machine.vic.frame
            return rt
        from .runtime import create_runtime
        args = self.manifest.get("boot_args") or {}
        if not args.get("image_path"):
            raise ValueError("cold-start demo manifest has no boot_args.image_path")
        rt = create_runtime(args["image_path"], file=args.get("file", "*"),
                            entry=args.get("entry"),
                            roms_dir=args.get("roms_dir"),
                            install_hooks=install_hooks)
        self.start_base = 0
        return rt

    # ---- replay ------------------------------------------------------------------
    def reset(self) -> None:
        self._index = 0

    @property
    def next_event_index(self) -> int:
        return self._index

    @property
    def exhausted(self) -> bool:
        return self._index >= len(self.events)

    @property
    def end_boundary(self) -> int | None:
        raw = self.manifest.get("end_boundary")
        return None if raw is None else max(0, int(raw))

    def finished(self, boundary: int) -> bool:
        """Replay reached the demo's end: prefer the recorded end boundary so
        trailing idle frames still play back; fall back to all-applied.
        ``boundary`` is an absolute frame; :attr:`start_base` is applied."""
        end = self.end_boundary
        if end is not None:
            return max(0, int(boundary) - self.start_base) >= end
        return self.exhausted

    def apply_to_runtime(self, boundary: int, rt, *, single: bool = False) -> int:
        return self.apply_to_runtimes(boundary, (rt,), single=single)

    def apply_to_runtimes(self, boundary: int, runtimes: Sequence, *,
                          single: bool = False) -> int:
        """Deliver due events (relative boundary <= ``boundary`` - start_base)
        to each runtime.  ``single=True`` delivers at most one event — for
        poll-wait seams where same-frame release/press pairs must be observed
        separately."""
        boundary = max(0, int(boundary) - self.start_base)
        applied = 0
        while self._index < len(self.events) and self.events[self._index].boundary <= boundary:
            event = self.events[self._index]
            for rt in runtimes:
                _apply_event(rt, event)
            self._index += 1
            applied += 1
            if single:
                break
        return applied

    # ---- suffix extraction (reproducibility helper for long demos) ----------------
    def write_suffix(self, rt, *, root: str | Path, name: str, boundary: int,
                     metadata: dict | None = None) -> Path:
        """A new demo starting from ``rt``'s current state that replays only
        the not-yet-applied events, rebased to the new snapshot's frame 0."""
        root = Path(root)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out = root / f"demo_{_safe_demo_name(name)}_{stamp}"
        out.mkdir(parents=True, exist_ok=True)
        write_snapshot(rt, out / SNAPSHOT_NAME)
        base = max(0, int(boundary) - self.start_base)  # rebase to relative
        events = [
            InputDemoEvent(boundary=max(0, e.boundary - base), seq=i,
                           kind=e.kind, key=e.key, mask=e.mask)
            for i, e in enumerate(self.events[self._index:])
        ]
        end = self.end_boundary
        manifest = {
            "version": DEMO_VERSION,
            "status": "complete",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "snapshot": SNAPSHOT_NAME,
            "boot_args": dict(rt.boot_args),
            "metadata": {
                **dict(self.manifest.get("metadata", {})),
                **dict(metadata or {}),
                "source_demo": str(self.demo_dir),
                "source_boundary": base,
                "source_next_event_index": self._index,
                "suffix_kind": "remaining_input_from_cursor",
            },
            "start_boundary": 0,
            "end_boundary": None if end is None else max(0, end - base),
            "event_count": len(events),
            "events": [e.to_json() for e in events],
        }
        (out / MANIFEST_NAME).write_text(json.dumps(manifest, indent=2),
                                         encoding="utf-8")
        return out


def _apply_event(rt, event: InputDemoEvent) -> None:
    m = rt.machine
    if event.kind == "key_down":
        m.key_down(event.key)
    elif event.kind == "key_up":
        m.key_up(event.key)
    elif event.kind == "joy1":
        if event.mask is None:
            raise ValueError("joy1 event missing mask")
        m.set_joy1(event.mask)
    elif event.kind == "joy2":
        if event.mask is None:
            raise ValueError("joy2 event missing mask")
        m.set_joy2(event.mask)
    else:
        raise ValueError(f"unknown input demo event kind {event.kind!r}")


def _safe_demo_name(name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_"
                      for ch in str(name).strip())
    return cleaned or "input"
