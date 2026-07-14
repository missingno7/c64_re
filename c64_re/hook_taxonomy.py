"""Hook taxonomy: classify replacement hooks by their *role*, not their address.

The original 6502 ASM (the VM) stays instruction-level snapshotable/stepable as
the oracle.  The source-port runtime, by contrast, is meant to be *checkpoint*-
level snapshotable: it resumes only from stable logical boundaries (frame,
object-update, render, input).  Between two checkpoints, lifted source-like code
may run as one atomic deterministic chain - it does NOT need to preserve every
historical PC bounce or support arbitrary mid-chain resume.  A snapshot
requested mid-chain is deferred to the next checkpoint (or represented as the
previous checkpoint + deterministic replay).

So a registered hook address is one of four things:

* ``checkpoint``      - a real logical boundary the source-port loop resumes from.
* ``env_wait``        - a hardware/environment wait the interpreter can't satisfy
                        natively (raster-line poll, CIA timer/IRQ-flag wait) and
                        that must stay a hook even in the oracle reference.
* ``debug_probe``     - exists only for verification/observation, not behaviour.
* ``glue``            - accidental ASM-boundary plumbing: behaviours, tails,
                        helpers, per-object/per-row scan steps.  These are the
                        collapse target - they should fuse into source-like code
                        between checkpoints, with correctness protected by the
                        semantic frame/state verifier, not by their PC.

The curated address sets are game knowledge, so the *adapter* supplies them:

    TAXONOMY = HookTaxonomy(
        checkpoints={0x080D: "frame: gameplay main-loop dispatcher"},
        env_waits={0x0913: "wait for raster line (frame pacing)"},
    )
    TAXONOMY.classify(0x1234)   # -> "glue"

Curated sets are intentionally small and explicit; everything not named is
``glue`` by default (the honest majority).  Refine the curated sets as logical
boundaries are confirmed; do not pad them.

Origin: faithful port of ``dos_re.hook_taxonomy`` (itself generalized from the
Overkill port); addressing translated from CS:IP pairs to flat 16-bit PCs.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

Addr = int

CATEGORIES = ("checkpoint", "env_wait", "debug_probe", "glue")


@dataclass
class HookTaxonomy:
    """Adapter-supplied curated hook-role sets for one game."""

    # Stable logical boundaries the native loop can resume from (frame top,
    # render-phase entries, object-update phases, input poll).
    checkpoints: dict[Addr, str] = field(default_factory=dict)
    # Hardware/environment waits the interpreter must keep hooked (raster-line
    # polls on $D012, CIA timer/IRQ-flag waits the host satisfies); the
    # verifier keeps these on the reference side too.
    env_waits: dict[Addr, str] = field(default_factory=dict)
    # Hooks that exist only to observe/verify, not to produce behaviour.
    debug_probes: dict[Addr, str] = field(default_factory=dict)

    def classify(self, addr: Addr) -> str:
        """Return the taxonomy category for a hook address."""
        if addr in self.checkpoints:
            return "checkpoint"
        if addr in self.env_waits:
            return "env_wait"
        if addr in self.debug_probes:
            return "debug_probe"
        return "glue"

    def classify_registry(self, replacements: Iterable[Addr]) -> dict[str, list[Addr]]:
        """Group an iterable of registered hook addresses by taxonomy category."""
        out: dict[str, list[Addr]] = {c: [] for c in CATEGORIES}
        for addr in replacements:
            out[self.classify(addr)].append(addr)
        for cat in out:
            out[cat].sort()
        return out
