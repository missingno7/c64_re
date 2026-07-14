"""Checkpoint stepping: VM-until-checkpoint handoff (dos_re port).

The source-port runtime resumes only from stable logical boundaries (frame /
render / object-update / input).  Given a runtime positioned at *any*
instruction, step the VM (instruction-exact oracle) until it reaches the
next compatible checkpoint, where game state is consistent and a native
phase-system could take over.

The checkpoint set is the game adapter's curated, evidence-based phase map
(PC -> "kind: description", e.g. ``0x6DE7: "frame: main loop head"``).
Categorising by ``kind`` (the text before the first ``:``) lets a caller
wait for a specific phase boundary.
"""
from __future__ import annotations

from typing import Mapping


def _kind(desc: str) -> str:
    # Descriptions are "frame: ...", "render: ...", "object-update: ...", "input: ...".
    return desc.split(":", 1)[0].strip()


def checkpoints_by_kind(checkpoint_hooks: Mapping[int, str]) -> dict[str, frozenset[int]]:
    """Group a checkpoint table (PC -> "kind: description") by kind."""
    out: dict[str, set[int]] = {}
    for pc, desc in checkpoint_hooks.items():
        out.setdefault(_kind(desc), set()).add(pc)
    return {k: frozenset(v) for k, v in out.items()}


def checkpoints_for(
    checkpoint_hooks: Mapping[int, str],
    kinds: "str | tuple[str, ...] | None",
) -> frozenset[int]:
    """Resolve a kind (or kinds) to its checkpoint PC set; None == all."""
    if kinds is None:
        return frozenset(checkpoint_hooks)
    if isinstance(kinds, str):
        kinds = (kinds,)
    by_kind = checkpoints_by_kind(checkpoint_hooks)
    out: set[int] = set()
    for k in kinds:
        if k not in by_kind:
            raise KeyError(f"unknown checkpoint kind {k!r}; known: {sorted(by_kind)}")
        out |= by_kind[k]
    return frozenset(out)


def run_to_next_checkpoint(
    cpu,
    checkpoint_hooks: Mapping[int, str],
    *,
    kinds: "str | tuple[str, ...] | None" = None,
    max_steps: int = 5_000_000,
    skip_current: bool = True,
) -> int:
    """Step the VM until it reaches the next compatible checkpoint; return it.

    ``kinds`` filters which phase boundaries count (None = any).
    ``skip_current`` steps once first so a call made while already *at* a
    checkpoint advances to the following one.  Raises ``TimeoutError`` if no
    checkpoint is reached within ``max_steps``.
    """
    targets = checkpoints_for(checkpoint_hooks, kinds)
    if skip_current:
        cpu.step()
    for _ in range(max_steps):
        if cpu.addr() in targets:
            return cpu.addr()
        cpu.step()
    raise TimeoutError(f"no checkpoint in {kinds or 'any'} within {max_steps} steps")
