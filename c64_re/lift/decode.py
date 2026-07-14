"""Static 6502 decode for the lifter — over the interpreter's own table.

The one rule (dos_re lifting_design): the lifter must never carry a second
opinion about instruction semantics or length.  ``cpu.OPCODES`` is the
single source of truth; this module only adds the control-flow *class* of
each opcode, which the interpreter encodes positionally in its dispatch.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..cpu import (
    ABS, IND, JAM_OPCODES, OPCODES, REL, UNSTABLE_OPCODES,
)
from ..dis6502 import MODE_SIZE

# control-flow classes
SEQ = "seq"            # falls through to pc+size
BRANCH = "branch"      # REL: taken target + fall-through
JMP_ABS = "jmp"        # unconditional, one static successor
JMP_IND = "jmp_ind"    # refusal: dynamic successor
CALL = "call"          # JSR: dependency + fall-through
RET = "ret"            # RTS/RTI: function exit
BRK_CLASS = "brk"      # refusal: software-interrupt convention
BAD = "bad"            # JAM/unstable/unknown: refusal

_FLOW = {"JSR": CALL, "RTS": RET, "RTI": RET, "BRK": BRK_CLASS}
_BRANCHES = frozenset(("BPL", "BMI", "BVC", "BVS", "BCC", "BCS", "BNE", "BEQ"))


@dataclass(frozen=True)
class Insn:
    pc: int
    opcode: int
    mnemonic: str
    mode: int
    size: int
    flow: str
    target: int | None  # static branch/jmp/call target, if any


def decode_one(read, pc: int) -> Insn:
    """Decode the instruction at ``pc`` via ``read(addr)`` (static bytes)."""
    opcode = read(pc)
    entry = OPCODES.get(opcode)
    if entry is None:
        kind = BAD
        return Insn(pc, opcode, "JAM" if opcode in JAM_OPCODES else
                    ("UNSTABLE" if opcode in UNSTABLE_OPCODES else "???"),
                    0, 1, kind, None)
    mn, mode, _, _ = entry
    size = MODE_SIZE[mode]
    target = None
    if mn in _BRANCHES:
        flow = BRANCH
        off = read((pc + 1) & 0xFFFF)
        if off >= 0x80:
            off -= 0x100
        target = (pc + 2 + off) & 0xFFFF
    elif mn == "JMP":
        if mode == IND:
            flow = JMP_IND
        else:
            flow = JMP_ABS
            target = read((pc + 1) & 0xFFFF) | (read((pc + 2) & 0xFFFF) << 8)
    elif mn in _FLOW:
        flow = _FLOW[mn]
        if flow == CALL:
            target = read((pc + 1) & 0xFFFF) | (read((pc + 2) & 0xFFFF) << 8)
    else:
        flow = SEQ
    return Insn(pc, opcode, mn, mode, size, flow, target)
