"""The lift proof ledger — JSON-backed, per-function status.

Ladder: ``REFUSED`` / ``LIFTED`` -> ``ORACLE_PASSING`` -> ``INSTALLED``,
with ``DIVERGED`` as the off-ladder failure state — a routine that lifted
statically but did NOT reproduce the oracle under verification (almost
always runtime-patched code a single-snapshot static scan couldn't see; a
candidate for :mod:`c64_re.runtime_code`, not a plain static hook).
Deliberately disjoint from the recovery status ladder (GUESS..CANONICAL):
**lifted is not recovered** — a lifted hook is a verified mechanical
artifact to refactor from, not understanding (the metrics-honesty rule).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

STATUSES = ("REFUSED", "DIVERGED", "LIFTED", "ORACLE_PASSING", "INSTALLED")


@dataclass
class LiftRecord:
    entry: int
    name: str
    status: str = "LIFTED"
    refusal_reason: str = ""
    size_bytes: int = 0
    instructions: int = 0
    calls_seen: int = 0
    verified_calls: int = 0
    diverged_calls: int = 0
    notes: str = ""


@dataclass
class LiftManifest:
    records: dict[str, LiftRecord] = field(default_factory=dict)  # key: $XXXX

    @staticmethod
    def _key(entry: int) -> str:
        return f"${entry & 0xFFFF:04X}"

    def update(self, record: LiftRecord) -> None:
        if record.status not in STATUSES:
            raise ValueError(f"unknown lift status {record.status!r}")
        self.records[self._key(record.entry)] = record

    def get(self, entry: int) -> LiftRecord | None:
        return self.records.get(self._key(entry))

    def save(self, path: str | Path) -> None:
        data = {k: asdict(r) for k, r in sorted(self.records.items())}
        Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "LiftManifest":
        p = Path(path)
        if not p.exists():
            return cls()
        data = json.loads(p.read_text(encoding="utf-8"))
        return cls(records={k: LiftRecord(**v) for k, v in data.items()})

    def summary(self) -> str:
        by_status: dict[str, int] = {}
        for r in self.records.values():
            by_status[r.status] = by_status.get(r.status, 0) + 1
        return " ".join(f"{s}={by_status[s]}" for s in STATUSES if s in by_status) or "(empty)"
