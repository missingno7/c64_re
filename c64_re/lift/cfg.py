"""Function-region discovery from an entry PC + the refusal taxonomy.

A function region is every instruction reachable from the entry through
static control flow, stopping at RTS/RTI.  JSR targets are *dependencies*,
not part of the region — generated hooks run callees through the VM
(``lift.runtime.emulate_call``), so hooks compose and lifting order is
irrelevant (dos_re's proven design).

Refusals are structured and honest — a refused function is simply not
liftable *yet*, never lifted wrong:

- ``jmp_ind``       — dynamic successor (jump tables need recovery, not lifting)
- ``brk``           — software-interrupt convention in the region
- ``bad_opcode``    — JAM/unstable/unknown byte reached
- ``budget``        — region exceeded ``max_instructions`` (runaway trace)
- ``no_exit``       — no RTS/RTI reachable (an eternal loop is a driver seam,
                      not a hookable routine)
- ``nonlocal_return`` — an ``RTS`` pops more bytes than this function itself
                      pushed (see below): it returns past its own caller.
- ``calls_nonlocal_return`` — this function JSRs to one already refused as
                      ``nonlocal_return`` (see :func:`refuse_unsafe_callers`).

Overlapping instructions are NOT a refusal: the 6502 BIT-skip idiom
(``$2C``/``$24`` swallowing the next op, with a branch landing on the
swallowed instruction) makes the same bytes decode two ways by entry
point.  ``scan_function`` allows a byte to be part of more than one decoded
instruction; the emitter dispatches each by its own start PC and the
differential oracle proves the result.  ``scan.overlaps`` counts how many
instructions reused another's bytes (0 for ordinary code).

**The non-local return idiom** (another legitimate, idiomatic 6502 trick;
found in Stix's collision-abort routine, $7166): a function pops a return
address it never pushed — typically ``PLA; PLA`` discarding the word its
own caller's JSR pushed — so its own eventual RTS lands not in its caller,
but in the *caller's caller*.  On real hardware this "cascading return" is
free: RTS just uses whatever is on top of the stack.  It is NOT safe to
lift as an independently hookable unit, though: :mod:`lift.runtime`'s
``emulate_call`` wraps every JSR in a nested envelope that waits for a
*specific* return PC and stack depth, and a callee that skips past its
caller's envelope makes that wait hang forever.  ``scan_function`` tracks
each function's own PHA/PHP (+1) vs PLA/PLP (-1) balance along every path
(JSR is depth-neutral locally — the callee's own RTS is assumed to consume
its own pushed return address); an RTS reached at a negative local depth
means this function is popping bytes it never pushed, and is refused.
The refusal is inherited exactly one level up by :func:`refuse_unsafe_callers`:
a function that JSRs to a ``nonlocal_return`` target is *also* unsafe to
lift standalone (its own emitted ``emulate_call`` for that JSR is the one
that hangs) — but functions further up the call chain are unaffected,
because once the unsafe callee is left uninstalled, plain interpretation
handles the whole skip correctly with no bookkeeping to violate.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .decode import (
    BAD, BRANCH, BRK_CLASS, CALL, JMP_ABS, JMP_IND, RET, SEQ, Insn, decode_one,
)


@dataclass
class Refusal:
    entry: int
    reason: str
    detail: str

    def __bool__(self) -> bool:  # a Refusal is falsy as a "scan"
        return False


@dataclass
class FunctionScan:
    entry: int
    insns: dict[int, Insn] = field(default_factory=dict)   # pc -> Insn
    block_starts: set[int] = field(default_factory=set)
    exits: set[int] = field(default_factory=set)            # PCs of RTS/RTI
    calls: set[int] = field(default_factory=set)            # JSR dependency targets
    byte_ranges: list[tuple[int, int]] = field(default_factory=list)  # for SMC guard
    overlaps: int = 0  # count of instructions that reuse a byte of another
    #                    (the 6502 BIT-skip idiom: $2C/$24 swallowing the next op)

    def __bool__(self) -> bool:
        return True

    @property
    def size_bytes(self) -> int:
        return sum(hi - lo for lo, hi in self.byte_ranges)


# Local PHA/PHP (+1) / PLA/PLP (-1) stack-depth effect, tracked relative to
# entry (0), to catch the non-local-return idiom.  JSR/RTS/RTI are excluded:
# a normal call is depth-neutral locally (the callee's own RTS consumes the
# return address it pushed), and RTI's implicit P+PC pop is the hardware
# interrupt-return convention, not a local imbalance to flag.
_STACK_EFFECT = {"PHA": 1, "PHP": 1, "PLA": -1, "PLP": -1}


def scan_function(read, entry: int, *, max_instructions: int = 768):
    """Discover the region.  ``read(addr)`` supplies the static bytes the
    function is expected to run as (typically the live RAM at lift time).
    Returns a :class:`FunctionScan` or a :class:`Refusal`."""
    entry &= 0xFFFF
    scan = FunctionScan(entry=entry)
    scan.block_starts.add(entry)
    work = [(entry, 0)]  # (pc, local stack depth relative to entry)
    covered: dict[int, int] = {}  # every byte pc -> owning insn pc

    while work:
        pc, depth = work.pop()
        if pc in scan.insns:
            continue
        if len(scan.insns) >= max_instructions:
            return Refusal(entry, "budget",
                           f"region exceeded {max_instructions} instructions")
        insn = decode_one(read, pc)
        if insn.flow == BAD:
            return Refusal(entry, "bad_opcode",
                           f"{insn.mnemonic} byte ${insn.opcode:02X} at ${pc:04X}")
        if insn.flow == BRK_CLASS:
            return Refusal(entry, "brk", f"BRK at ${pc:04X}")
        if insn.flow == JMP_IND:
            return Refusal(entry, "jmp_ind", f"JMP (indirect) at ${pc:04X}")
        # Overlapping instructions are LEGAL on the 6502 and idiomatic: the
        # BIT-skip trick uses a bare $2C (BIT abs) / $24 (BIT zp) opcode to
        # swallow the next 2- or 1-byte instruction, and a branch lands on
        # that swallowed instruction to "un-skip" it — so the same bytes
        # decode two ways depending on the entry point.  We allow a byte to
        # belong to more than one decoded instruction: each is keyed by its
        # own start PC and dispatched independently by the emitter, the SMC
        # entry guard still covers the union of all decoded bytes (a
        # runtime-patched variant is refused at call time), and the
        # differential oracle is the final proof the overlap was read right.
        if any(covered.get(b & 0xFFFF, pc) != pc for b in range(pc, pc + insn.size)):
            scan.overlaps += 1
        for b in range(pc, pc + insn.size):
            covered[b & 0xFFFF] = pc
        scan.insns[pc] = insn

        depth = depth + _STACK_EFFECT.get(insn.mnemonic, 0)
        nxt = (pc + insn.size) & 0xFFFF
        if insn.flow == SEQ:
            work.append((nxt, depth))
        elif insn.flow == CALL:
            scan.calls.add(insn.target)
            work.append((nxt, depth))
        elif insn.flow == BRANCH:
            scan.block_starts.add(insn.target)
            scan.block_starts.add(nxt)
            work.append((insn.target, depth))
            work.append((nxt, depth))
        elif insn.flow == JMP_ABS:
            scan.block_starts.add(insn.target)
            work.append((insn.target, depth))
        elif insn.flow == RET:
            scan.exits.add(pc)
            if insn.mnemonic == "RTS" and depth < 0:
                return Refusal(
                    entry, "nonlocal_return",
                    f"RTS at ${pc:04X} pops {-depth} more byte(s) than this "
                    "function itself pushed — it returns past its own caller "
                    "(the 6502 cascading-return idiom); unsafe to lift as an "
                    "independently callable unit (see module docstring)")

    if not scan.exits:
        return Refusal(entry, "no_exit", "no RTS/RTI reachable from entry")

    # contiguous byte ranges for the SMC entry guard
    pcs = sorted(covered)
    lo = prev = pcs[0]
    for b in pcs[1:]:
        if b != prev + 1:
            scan.byte_ranges.append((lo, prev + 1))
            lo = b
        prev = b
    scan.byte_ranges.append((lo, prev + 1))
    return scan


def refuse_unsafe_callers(scans: "dict[int, FunctionScan | Refusal]") -> "dict[int, Refusal]":
    """Tier-2 refusal: a function that JSRs to a ``nonlocal_return`` target
    is also unsafe to lift standalone — refuse it as ``calls_nonlocal_return``.

    Takes a batch of already-computed per-address :func:`scan_function`
    results (as a liftgen census naturally produces) and returns Refusals
    for the *additional* addresses this catches; it does not modify
    ``scans`` or re-scan anything.  Deliberately stops at one level: a
    function calling one of THESE newly-refused functions is unaffected,
    because once the unsafe callee is left uninstalled, plain interpretation
    handles the whole non-local return correctly with no bookkeeping to
    violate (see the module docstring's worked example).
    """
    nonlocal_return = {addr for addr, r in scans.items()
                       if isinstance(r, Refusal) and r.reason == "nonlocal_return"}
    out: dict[int, Refusal] = {}
    for addr, r in scans.items():
        if not isinstance(r, Refusal) and (r.calls & nonlocal_return):
            hit = sorted(r.calls & nonlocal_return)[0]
            out[addr] = Refusal(
                addr, "calls_nonlocal_return",
                f"JSRs to ${hit:04X}, which returns past ITS caller "
                "(nonlocal_return) — this function's own emitted JSR wrapper "
                "would hang waiting for a return that skips it; unsafe to "
                "lift standalone (callers of this function are unaffected)")
    return out
