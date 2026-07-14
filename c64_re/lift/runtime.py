"""Support imported by generated hooks.

``emulate_call`` runs a JSR callee through the VM: if the callee (or
anything it reaches) is itself hooked, the CPU's normal dispatch applies —
hooks compose, so lifting order is irrelevant.  While a lifted hook body
runs, interrupts stay latched-but-undelivered (the same atomicity contract
the differential verifier proves against; delivery happens at the next VM
step after the hook returns).

``interp_one`` executes exactly one instruction through the interpreter —
the emitter's exact fallback tier for anything it chooses not to inline.
"""
from __future__ import annotations


def emulate_call(cpu, target: int, return_pc: int, *, max_instructions: int = 2_000_000) -> None:
    """Run ``JSR target`` semantics to completion on the live VM."""
    s = cpu.s
    ret = (return_pc - 1) & 0xFFFF
    cpu.push((ret >> 8) & 0xFF)
    cpu.push(ret & 0xFF)
    done_sp = s.sp  # frame popped when SP is back here + 2
    s.pc = target & 0xFFFF
    prev_inhibit = cpu.inhibit_interrupts
    cpu.inhibit_interrupts = True
    try:
        for _ in range(max_instructions):
            cpu.step()
            if s.pc == return_pc and s.sp == ((done_sp + 2) & 0xFF):
                return
        raise RuntimeError(
            f"emulate_call ${target:04X} never returned to ${return_pc:04X} "
            f"within {max_instructions} instructions (PC=${s.pc:04X})"
        )
    finally:
        cpu.inhibit_interrupts = prev_inhibit


def interp_one(cpu, pc: int) -> None:
    """Execute the single instruction at ``pc`` through the interpreter."""
    cpu.s.pc = pc & 0xFFFF
    cpu._execute(pc & 0xFFFF)
