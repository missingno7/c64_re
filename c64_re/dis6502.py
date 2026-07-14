"""6502 disassembler over the interpreter's own opcode table.

Deliberately NOT a second semantic model (dos_re's lifter rule): lengths and
mnemonics come from cpu.OPCODES, so the disassembly can never disagree with
what the interpreter executes.
"""
from __future__ import annotations

from .cpu import (
    ABS, ABX, ABY, ACC, IMM, IMP, IND, IZX, IZY, JAM_OPCODES, OPCODES, REL,
    UNSTABLE_OPCODES, ZP, ZPX, ZPY,
)

MODE_SIZE = {
    IMP: 1, ACC: 1, IMM: 2, ZP: 2, ZPX: 2, ZPY: 2,
    ABS: 3, ABX: 3, ABY: 3, IND: 3, IZX: 2, IZY: 2, REL: 2,
}


def instruction_length(opcode: int) -> int:
    entry = OPCODES.get(opcode)
    if entry is None:
        return 1
    return MODE_SIZE[entry[1]]


def disassemble_one(read, pc: int) -> tuple[str, int]:
    """(text, length).  ``read(addr)`` supplies bytes — pass cpu.mem.rb for
    the banked view or a bytes-getter for raw dumps."""
    opcode = read(pc)
    entry = OPCODES.get(opcode)
    if entry is None:
        tag = "JAM" if opcode in JAM_OPCODES else (
            "UNSTABLE" if opcode in UNSTABLE_OPCODES else "???")
        return f"{pc:04X}  {opcode:02X}        .byte ${opcode:02X}  ; {tag}", 1
    mn, mode, _, _ = entry
    size = MODE_SIZE[mode]
    b = [read((pc + i) & 0xFFFF) for i in range(size)]
    raw = " ".join(f"{x:02X}" for x in b).ljust(9)
    if mode == IMP:
        operand = ""
    elif mode == ACC:
        operand = "A"
    elif mode == IMM:
        operand = f"#${b[1]:02X}"
    elif mode == ZP:
        operand = f"${b[1]:02X}"
    elif mode == ZPX:
        operand = f"${b[1]:02X},X"
    elif mode == ZPY:
        operand = f"${b[1]:02X},Y"
    elif mode == ABS:
        operand = f"${b[1] | (b[2] << 8):04X}"
    elif mode == ABX:
        operand = f"${b[1] | (b[2] << 8):04X},X"
    elif mode == ABY:
        operand = f"${b[1] | (b[2] << 8):04X},Y"
    elif mode == IND:
        operand = f"(${b[1] | (b[2] << 8):04X})"
    elif mode == IZX:
        operand = f"(${b[1]:02X},X)"
    elif mode == IZY:
        operand = f"(${b[1]:02X}),Y"
    else:  # REL
        off = b[1] - 0x100 if b[1] >= 0x80 else b[1]
        operand = f"${(pc + 2 + off) & 0xFFFF:04X}"
    return f"{pc:04X}  {raw} {mn} {operand}".rstrip(), size


def disassemble(read, pc: int, count: int = 16) -> str:
    lines = []
    for _ in range(count):
        text, size = disassemble_one(read, pc)
        lines.append(text)
        pc = (pc + size) & 0xFFFF
    return "\n".join(lines)
