"""The differential hook oracle — every replacement is proven, per call.

The dos_re pattern, 6502 flavor: when a replacement hook is about to run,
clone the full machine state into a pure-ASM oracle runtime, run the
*original* instructions there to the hook's continuation, run the *hook*
on the live runtime, then diff EVERYTHING — CPU registers and flags, all
64 KB of RAM, color RAM, VIC, both CIAs, SID (including the OSC3 LFSR).
Any mismatch raises :class:`HookDivergence` with the precise differences.
Full-state diffs are the default and there is no narrow mode (dos_re
pitfall #7: narrowing the diff hides divergence in scratch space).

Continuation modes:

- **strict (default)** — the hook boundary is a JSR-entered subroutine:
  the return address sits on top of the stack at entry; the oracle runs
  until it returns to the caller (PC == return address and SP has popped
  the frame).  No metadata needed.
- **metadata** — a :class:`HookStop` per address gives explicit
  continuation PCs for boundaries that are not plain subroutines
  (JMP-entered code, fall-through seams).

Machine-time contract: the oracle consumes real cycles (raster and CIA
timers advance).  After the hook runs, the verifier advances the live
machine by the same cycle count and syncs the counters, so both sides sit
at the same machine time before the diff.

Interrupt contract: while the oracle runs the routine body, pending IRQ/NMI
stay *latched but undelivered* (``cpu.inhibit_interrupts``) — a Python hook
is atomic and cannot be interrupted mid-body, so mid-routine interrupt
interleaving is out of scope for the per-hook proof; both sides end with
the same latched-pending chip state and the interrupt then delivers at the
next step on either side.  Whether that delivery-point skew ever *matters*
is exactly what the frame/tick oracles judge at their boundaries (dos_re's
proven division of labor).  A routine that spins waiting for an IRQ effect
therefore exceeds the oracle budget and fails loud — such a wait loop is an
input-wait/checkpoint seam, not a hookable leaf.

``C64_RE_TRACE_HOOK=$XXXX`` prints the oracle's instruction trace for that
hook on divergence (the "read what the original actually did" lever).
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from .gaps import HookVerifyStats
from .snapshot import capture, clone_runtime, restore


class HookDivergence(AssertionError):
    """A replacement did not reproduce the original's effects."""


@dataclass(frozen=True)
class HookStop:
    """Continuation metadata for one hooked address (metadata mode)."""
    continuations: tuple[int, ...] = ()
    max_instructions: int = 2_000_000


@dataclass
class VerifyOutcome:
    name: str
    pc: int
    oracle_instructions: int
    oracle_cycles: int
    reasons: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.reasons


def _parse_trace_env() -> int | None:
    text = os.environ.get("C64_RE_TRACE_HOOK", "").strip().lstrip("$")
    if not text:
        return None
    return int(text, 16) & 0xFFFF


class HookOracle:
    """Holds one persistent oracle runtime and verifies hook calls against it.

    Install with :func:`install_live_verifier`, or call :meth:`verify_call`
    directly for offline single-shot checks.
    """

    def __init__(self, rt, *, metadata: dict[int, HookStop] | None = None,
                 on_result=None, raise_on_divergence: bool = True,
                 max_ram_diffs_reported: int = 16,
                 strict_cycles: bool = False) -> None:
        self.rt = rt
        self.metadata = dict(metadata or {})
        self.on_result = on_result
        self.raise_on_divergence = raise_on_divergence
        self.max_ram_diffs_reported = max_ram_diffs_reported
        # Hand-written recovery hooks tick 0 cycles; the verifier makes up the
        # machine-time deficit after the diff.  Lifted hooks tick the exact
        # interpreter cycles themselves — strict_cycles=True turns any
        # remaining deficit/excess into a reported divergence (the lift
        # driver's setting; catches emitter cycle-model bugs).
        self.strict_cycles = strict_cycles
        self.stats = HookVerifyStats()
        self.trace_pc = _parse_trace_env()
        self._oracle = None
        self._busy = False

    # ---- the cpu.hook_verifier entry point ----------------------------------
    def __call__(self, cpu, pc: int, handler, name: str) -> None:
        if self._busy:
            # Nested hook (a parent hook composing a child): the parent's
            # transaction already covers the child's effects; run it plain.
            handler(cpu)
            return
        self._busy = True
        try:
            outcome = self.verify_call(pc, handler, name)
        finally:
            self._busy = False
        if outcome.ok:
            self.stats.verified += 1
            if self.on_result is not None:
                self.on_result(name, True, None)
        else:
            reason = "; ".join(outcome.reasons)
            self.stats.diverged.append((name, reason))
            if self.on_result is not None:
                self.on_result(name, False, reason)
            if self.raise_on_divergence:
                raise HookDivergence(
                    f"hook {name} at ${pc:04X} diverged from the ASM oracle "
                    f"(after {outcome.oracle_instructions} original instructions):\n  "
                    + "\n  ".join(outcome.reasons)
                )

    # ---- core ------------------------------------------------------------------
    def verify_call(self, pc: int, handler, name: str) -> VerifyOutcome:
        rt = self.rt
        state = capture(rt)
        if self._oracle is None:
            self._oracle = clone_runtime(rt, install_hooks=False)
        else:
            restore(self._oracle, state)
        oracle = self._oracle
        stop = self.metadata.get(pc, HookStop())

        trace_lines = [] if self.trace_pc == (pc & 0xFFFF) else None
        outcome = VerifyOutcome(name=name, pc=pc, oracle_instructions=0, oracle_cycles=0)

        # ---- oracle side: run the original to the continuation ----
        start_instr = oracle.cpu.instr_count
        start_cycles = oracle.cpu.cycle_count
        try:
            self._run_original(oracle, stop, trace_lines)
        except Exception as exc:  # noqa: BLE001 - reported as divergence context
            outcome.reasons.append(f"oracle run failed: {type(exc).__name__}: {exc}")
            self._maybe_dump_trace(trace_lines)
            return outcome
        outcome.oracle_instructions = oracle.cpu.instr_count - start_instr
        outcome.oracle_cycles = oracle.cpu.cycle_count - start_cycles

        # ---- live side: run the replacement, then resync machine time ----
        live_cycles_before = rt.cpu.cycle_count
        try:
            handler(rt.cpu)
        except Exception as exc:  # noqa: BLE001
            outcome.reasons.append(f"hook raised: {type(exc).__name__}: {exc}")
            restore(rt, state)  # leave the live runtime unpoisoned
            self._maybe_dump_trace(trace_lines)
            return outcome
        hook_cycles = rt.cpu.cycle_count - live_cycles_before
        deficit = outcome.oracle_cycles - hook_cycles
        if self.strict_cycles and deficit != 0:
            outcome.reasons.append(
                f"cycle model: oracle={outcome.oracle_cycles} hook={hook_cycles} "
                f"(deficit {deficit})"
            )
        if deficit > 0:
            rt.machine.tick(deficit)
        elif deficit < 0 and not self.strict_cycles:
            outcome.reasons.append(
                f"hook advanced machine time {-deficit} cycles PAST the oracle "
                f"(oracle={outcome.oracle_cycles} hook={hook_cycles}) — cannot resync"
            )
        rt.cpu.cycle_count = oracle.cpu.cycle_count
        rt.cpu.instr_count = oracle.cpu.instr_count

        # ---- diff ----
        outcome.reasons = self._diff(oracle, rt)
        if outcome.reasons:
            self._maybe_dump_trace(trace_lines)
        return outcome

    def _run_original(self, oracle, stop: HookStop, trace_lines) -> None:
        cpu = oracle.cpu
        cpu.inhibit_interrupts = True  # see module docstring: latched, not delivered
        if trace_lines is not None:
            from .dis6502 import disassemble_one
            cpu.trace_fn = lambda c, pc, op: trace_lines.append(
                disassemble_one(c.mem.rb, pc)[0]
            )
        try:
            if stop.continuations:
                targets = {t & 0xFFFF for t in stop.continuations}
                for _ in range(stop.max_instructions):
                    cpu.step()
                    if cpu.s.pc in targets:
                        return
                raise HookDivergence(
                    f"oracle never reached continuation {sorted(hex(t) for t in targets)} "
                    f"within {stop.max_instructions} instructions (PC=${cpu.s.pc:04X})"
                )
            # strict mode: JSR-entered subroutine — return address on top of stack
            s = cpu.s
            entry_sp = s.sp
            ret_lo = cpu.mem.rb(0x0100 | ((entry_sp + 1) & 0xFF))
            ret_hi = cpu.mem.rb(0x0100 | ((entry_sp + 2) & 0xFF))
            ret = ((ret_lo | (ret_hi << 8)) + 1) & 0xFFFF
            done_sp = (entry_sp + 2) & 0xFF
            for _ in range(stop.max_instructions):
                cpu.step()
                if s.pc == ret and s.sp == done_sp:
                    return
            raise HookDivergence(
                f"oracle never returned to caller ${ret:04X} within "
                f"{stop.max_instructions} instructions (PC=${s.pc:04X} SP=${s.sp:02X}) — "
                "if this boundary is not a JSR-entered subroutine, give it a HookStop"
            )
        finally:
            cpu.trace_fn = None
            cpu.inhibit_interrupts = False

    def _maybe_dump_trace(self, trace_lines) -> None:
        if trace_lines:
            tail = trace_lines[-200:]
            print(f"--- oracle trace (last {len(tail)} of {len(trace_lines)}):")
            for line in tail:
                print("  " + line)

    # ---- the diff ----------------------------------------------------------------
    def _diff(self, oracle, live) -> list[str]:
        reasons: list[str] = []
        a, b = oracle.cpu.s, live.cpu.s
        for reg in ("pc", "a", "x", "y", "sp", "n", "v", "d", "i", "z", "c"):
            va, vb = getattr(a, reg), getattr(b, reg)
            if va != vb:
                reasons.append(f"cpu.{reg}: oracle={va:#04x} hook={vb:#04x}")
        if oracle.cpu.nmi_pending != live.cpu.nmi_pending:
            reasons.append(f"cpu.nmi_pending: oracle={oracle.cpu.nmi_pending} "
                           f"hook={live.cpu.nmi_pending}")

        reasons += _diff_bytes("ram", oracle.mem.ram, live.mem.ram,
                               self.max_ram_diffs_reported)
        reasons += _diff_bytes("color_ram", oracle.mem.color_ram, live.mem.color_ram,
                               self.max_ram_diffs_reported, base=0xD800)
        if oracle.mem.cpu_port_ddr != live.mem.cpu_port_ddr:
            reasons.append(f"$00 DDR: oracle={oracle.mem.cpu_port_ddr:#04x} "
                           f"hook={live.mem.cpu_port_ddr:#04x}")
        if oracle.mem.cpu_port_data != live.mem.cpu_port_data:
            reasons.append(f"$01 port: oracle={oracle.mem.cpu_port_data:#04x} "
                           f"hook={live.mem.cpu_port_data:#04x}")

        for label, sa, sb in (
            ("vic", oracle.machine.vic.get_state(), live.machine.vic.get_state()),
            ("cia1", oracle.machine.cia1.get_state(), live.machine.cia1.get_state()),
            ("cia2", oracle.machine.cia2.get_state(), live.machine.cia2.get_state()),
            ("sid", oracle.machine.sid.get_state(), live.machine.sid.get_state()),
        ):
            for key in sa:
                if sa[key] != sb[key]:
                    reasons.append(f"{label}.{key}: oracle={sa[key]!r} hook={sb[key]!r}")
        return reasons


def _diff_bytes(label: str, a, b, max_reported: int, base: int = 0) -> list[str]:
    if a == b:
        return []
    diffs = []
    count = 0
    for i in range(len(a)):
        if a[i] != b[i]:
            count += 1
            if len(diffs) < max_reported:
                diffs.append(f"${base + i:04X}: oracle={a[i]:02X} hook={b[i]:02X}")
    return [f"{label}: {count} byte(s) differ — " + ", ".join(diffs)
            + ("" if count <= max_reported else ", ...")]


def install_live_verifier(rt, *, metadata: dict[int, HookStop] | None = None,
                          on_result=None, raise_on_divergence: bool = True,
                          passthrough: set[int] | None = None,
                          strict_cycles: bool = False) -> HookOracle:
    """Route every installed replacement hook through the oracle differ.

    Expensive by design (a full state capture + original re-execution per
    hook call): this is the offline **verify** execution mode, not the
    everyday hybrid workbench.
    """
    oracle = HookOracle(rt, metadata=metadata, on_result=on_result,
                        raise_on_divergence=raise_on_divergence,
                        strict_cycles=strict_cycles)
    rt.cpu.hook_verifier = oracle
    rt.cpu.hook_verifier_passthrough = set(passthrough or ())
    return oracle
