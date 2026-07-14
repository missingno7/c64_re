"""Fail-loud gap exceptions + hybrid-runtime bookkeeping (dos_re port).

**The gap exception.** When the hybrid or native runtime reaches behaviour
that is not yet recovered, it raises :class:`HybridGap` — loudly, with a
precise message — instead of silently falling back to original ASM or
guessing.  A silent fallback hides missing recovery work; a loud gap *is*
the next work item.

**Transition signals.** Multi-frame transitions (death/respawn, level-end,
game-over) are declared as *subclasses* of :class:`HybridGap` in the game
adapter and raised where the transition begins; the flow driver catches the
specific signal first, every generic ``except HybridGap`` still treats an
unhandled transition as a loud gap.

This module stays pure — no cpu/mem imports — so a game's native (VM-less)
layer can import it without pulling in the VM.
"""
from __future__ import annotations

from dataclasses import dataclass, field


class HybridGap(RuntimeError):
    """The hybrid/native runtime reached something not yet recovered."""


@dataclass
class HookVerifyStats:
    """Verified/diverged tallies for checkpoint verifiers (see :func:`report`)."""
    verified: int = 0
    diverged: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class HookTraceStats:
    """Per-hook invocation counts for the live hybrid runtime — which
    recovered systems actually fire (and, by absence, which screens are
    still pure ASM).  No oracle, no diff: just a tally."""
    counts: dict = field(default_factory=dict)

    def bump(self, name: str) -> None:
        self.counts[name] = self.counts.get(name, 0) + 1

    def total(self) -> int:
        return sum(self.counts.values())

    def snapshot(self) -> dict:
        return dict(self.counts)

    def window_total(self, since: dict | None) -> int:
        if since is None:
            return self.total()
        return sum(max(0, v - since.get(k, 0)) for k, v in self.counts.items())

    def summary(self, group=None, top: int | None = None, since: dict | None = None) -> str:
        src = self.counts
        if since is not None:
            src = {k: v - since.get(k, 0) for k, v in self.counts.items()
                   if v - since.get(k, 0) > 0}
        agg: dict[str, int] = {}
        for name, c in src.items():
            g = group(name) if group else name
            agg[g] = agg.get(g, 0) + c
        items = sorted(agg.items(), key=lambda kv: -kv[1])
        if top is not None:
            items = items[:top]
        empty = "(idle)" if since is not None else "(no recovered hooks fired)"
        return " ".join(f"{n}={c}" for n, c in items) or empty


def report(stats: HookVerifyStats, on_result, raise_on_divergence, name: str, reason):
    """Record one verify outcome: ``reason is None`` means the contract matched."""
    if reason is None:
        stats.verified += 1
        if on_result is not None:
            on_result(name, True, None)
    else:
        stats.diverged.append((name, reason))
        if on_result is not None:
            on_result(name, False, reason)
        if raise_on_divergence:
            raise AssertionError(f"hook verify divergence on {name}: {reason}")
