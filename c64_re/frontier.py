"""Explicit cold-start frontier classification — a triage manifest for the last
interpreted addresses.

Late in a port, coverage reports converge on a small residue of addresses that
never landed a hook.  Left unclassified, they become an undifferentiated
"unknown" bucket that quietly erodes trust in the coverage numbers.  This
module gives each leftover an explicit identity: a real hook candidate, an
intentionally interpreted bootstrap fragment, a bounded-original rare branch
owned by a larger hook, or a harmless scratch/tail inside an already-lifted
block.

The manifest is *data the game adapter owns* — a tuple of
:class:`FrontierEntry` — and a triage record, not an execution dependency.

Origin: faithful port of ``dos_re.frontier`` (itself generalized from the
Overkill port's ``frontier_manifest.py``); addressing translated from CS:IP
pairs to flat 16-bit PCs, written ``$XXXX``.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Iterable

Addr = int


class FrontierCategory(StrEnum):
    FINAL_ORCHESTRATOR = "final-orchestrator"
    SAME_PC_LOOP_GATE = "same-pc-loop-gate"
    DO_NOT_HOOK_BOOTSTRAP = "do-not-hook-bootstrap"
    BOUNDED_ORIGINAL_RARE_BRANCH = "bounded-original-rare-branch"
    UNCLASSIFIED_HARMLESS_TAIL = "unclassified-harmless-scratch-tail"
    HOOK_CANDIDATE = "hook-candidate"


@dataclass(frozen=True)
class FrontierEntry:
    addr: Addr
    name: str
    island: str
    category: FrontierCategory
    status: str
    owner: Addr | None = None
    notes: str = ""


def by_addr(manifest: Iterable[FrontierEntry]) -> dict[Addr, FrontierEntry]:
    return {entry.addr: entry for entry in manifest}


def fmt_addr(addr: Addr) -> str:
    return f"${addr:04X}"


def frontier_summary_lines(manifest: Iterable[FrontierEntry]) -> list[str]:
    lines = ["== explicit cold-start frontier manifest =="]
    for entry in manifest:
        # ``is not None``: 0x0000 is a valid flat PC (dos_re's (cs, ip) tuples
        # were never falsy, a plain int can be)
        owner = f" owner={fmt_addr(entry.owner)}" if entry.owner is not None else ""
        lines.append(
            f"  - {fmt_addr(entry.addr)} {entry.category.value:<34} "
            f"{entry.status:<24} {entry.name}{owner}"
        )
    return lines
