"""The 6502/6510 interpreter — the heart of the c64_re oracle VM.

Mirrors dos_re's ``cpu.py`` role: a deterministic, stdlib-only interpreter
with per-address replacement-hook dispatch (game recovery), service hooks
(framework HLE seams such as the KERNAL shim), an optional per-instruction
trace, and instruction/cycle telemetry that the machine clock (VIC raster,
CIA timers) is driven from.

Scope rule (from dos_re's AGENTS.md, unchanged): the interpreter models what
real programs exercised, not the datasheet.  All documented NMOS 6502
opcodes are implemented including decimal mode; the *stable* illegal opcodes
(LAX/SAX/DCP/ISC/SLO/RLA/SRE/RRA/ANC/ALR/ARR/SBX/USBC and the NOP variants)
are implemented because crunchers and copy protections use them; the
*unstable* ones (ANE/LXA/LAS/TAS/SHA/SHX/SHY) and the JAM opcodes fail loud
with precise context, to be implemented only when a concrete program
exercises them with an observed contract.

Timing model: per-instruction base cycles + the documented page-cross and
taken-branch penalties.  This is instruction-level, not cycle-exact within
an instruction (no bus-accurate sub-instruction timing); ``docs/
hardware_support.md`` records what that does and does not affect.  RMW
instructions perform the NMOS double write (old value, then new) because
VIC interrupt acknowledgment (``INC $D019``) depends on it.
"""
from __future__ import annotations

from typing import Callable


class CPUJam(RuntimeError):
    """A JAM/KIL opcode was executed — the real chip would halt here."""


class UnimplementedOpcode(RuntimeError):
    """An opcode outside the supported set (see module docstring)."""


class CPUState:
    """Architectural 6502 state.  Flags are stored unpacked (0/1 ints)."""

    __slots__ = ("a", "x", "y", "sp", "pc", "n", "v", "d", "i", "z", "c")

    def __init__(self, *, a=0, x=0, y=0, sp=0xFD, pc=0,
                 n=0, v=0, d=0, i=1, z=0, c=0) -> None:
        self.a = a; self.x = x; self.y = y
        self.sp = sp; self.pc = pc
        self.n = n; self.v = v; self.d = d
        self.i = i; self.z = z; self.c = c

    # ---- status register packing (bit5 always reads 1) ----
    def get_p(self, *, b: int = 0) -> int:
        return (
            (self.n << 7) | (self.v << 6) | 0x20 | (b << 4)
            | (self.d << 3) | (self.i << 2) | (self.z << 1) | self.c
        )

    def set_p(self, p: int) -> None:
        self.n = (p >> 7) & 1
        self.v = (p >> 6) & 1
        self.d = (p >> 3) & 1
        self.i = (p >> 2) & 1
        self.z = (p >> 1) & 1
        self.c = p & 1

    def clone(self) -> "CPUState":
        s = CPUState()
        for f in CPUState.__slots__:
            setattr(s, f, getattr(self, f))
        return s

    def as_dict(self) -> dict:
        return {f: getattr(self, f) for f in CPUState.__slots__}

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"A={self.a:02X} X={self.x:02X} Y={self.y:02X} SP={self.sp:02X} "
            f"PC={self.pc:04X} P={self.get_p():02X}"
        )


# Addressing modes.
IMP, ACC, IMM, ZP, ZPX, ZPY, ABS, ABX, ABY, IND, IZX, IZY, REL = range(13)

# opcode -> (mnemonic, mode, base_cycles, page_cross_penalty)
# Stores and RMW ops never take the page-cross penalty (they always pay it).
OPCODES: dict[int, tuple[str, int, int, bool]] = {}


def _op(code, mn, mode, cyc, pagep=False):
    OPCODES[code] = (mn, mode, cyc, pagep)


# ---- loads / stores ----
_op(0xA9, "LDA", IMM, 2); _op(0xA5, "LDA", ZP, 3); _op(0xB5, "LDA", ZPX, 4)
_op(0xAD, "LDA", ABS, 4); _op(0xBD, "LDA", ABX, 4, True); _op(0xB9, "LDA", ABY, 4, True)
_op(0xA1, "LDA", IZX, 6); _op(0xB1, "LDA", IZY, 5, True)
_op(0xA2, "LDX", IMM, 2); _op(0xA6, "LDX", ZP, 3); _op(0xB6, "LDX", ZPY, 4)
_op(0xAE, "LDX", ABS, 4); _op(0xBE, "LDX", ABY, 4, True)
_op(0xA0, "LDY", IMM, 2); _op(0xA4, "LDY", ZP, 3); _op(0xB4, "LDY", ZPX, 4)
_op(0xAC, "LDY", ABS, 4); _op(0xBC, "LDY", ABX, 4, True)
_op(0x85, "STA", ZP, 3); _op(0x95, "STA", ZPX, 4); _op(0x8D, "STA", ABS, 4)
_op(0x9D, "STA", ABX, 5); _op(0x99, "STA", ABY, 5)
_op(0x81, "STA", IZX, 6); _op(0x91, "STA", IZY, 6)
_op(0x86, "STX", ZP, 3); _op(0x96, "STX", ZPY, 4); _op(0x8E, "STX", ABS, 4)
_op(0x84, "STY", ZP, 3); _op(0x94, "STY", ZPX, 4); _op(0x8C, "STY", ABS, 4)
# ---- transfers ----
_op(0xAA, "TAX", IMP, 2); _op(0xA8, "TAY", IMP, 2); _op(0x8A, "TXA", IMP, 2)
_op(0x98, "TYA", IMP, 2); _op(0xBA, "TSX", IMP, 2); _op(0x9A, "TXS", IMP, 2)
# ---- stack ----
_op(0x48, "PHA", IMP, 3); _op(0x68, "PLA", IMP, 4)
_op(0x08, "PHP", IMP, 3); _op(0x28, "PLP", IMP, 4)
# ---- ALU ----
for base, mn in ((0x69, "ADC"), (0xE9, "SBC"), (0x29, "AND"),
                 (0x09, "ORA"), (0x49, "EOR"), (0xC9, "CMP")):
    _op(base, mn, IMM, 2)
    _op(base - 4, mn, ZP, 3); _op(base + 12, mn, ZPX, 4)
    _op(base + 4, mn, ABS, 4); _op(base + 20, mn, ABX, 4, True)
    _op(base + 16, mn, ABY, 4, True)
    _op(base - 8, mn, IZX, 6); _op(base + 8, mn, IZY, 5, True)
_op(0xE0, "CPX", IMM, 2); _op(0xE4, "CPX", ZP, 3); _op(0xEC, "CPX", ABS, 4)
_op(0xC0, "CPY", IMM, 2); _op(0xC4, "CPY", ZP, 3); _op(0xCC, "CPY", ABS, 4)
_op(0x24, "BIT", ZP, 3); _op(0x2C, "BIT", ABS, 4)
# ---- inc/dec ----
_op(0xE6, "INC", ZP, 5); _op(0xF6, "INC", ZPX, 6)
_op(0xEE, "INC", ABS, 6); _op(0xFE, "INC", ABX, 7)
_op(0xC6, "DEC", ZP, 5); _op(0xD6, "DEC", ZPX, 6)
_op(0xCE, "DEC", ABS, 6); _op(0xDE, "DEC", ABX, 7)
_op(0xE8, "INX", IMP, 2); _op(0xC8, "INY", IMP, 2)
_op(0xCA, "DEX", IMP, 2); _op(0x88, "DEY", IMP, 2)
# ---- shifts ----
for base, mn in ((0x0A, "ASL"), (0x4A, "LSR"), (0x2A, "ROL"), (0x6A, "ROR")):
    _op(base, mn, ACC, 2)
    _op(base - 4, mn, ZP, 5); _op(base + 12, mn, ZPX, 6)
    _op(base + 4, mn, ABS, 6); _op(base + 20, mn, ABX, 7)
# ---- control flow ----
_op(0x4C, "JMP", ABS, 3); _op(0x6C, "JMP", IND, 5)
_op(0x20, "JSR", ABS, 6); _op(0x60, "RTS", IMP, 6)
_op(0x40, "RTI", IMP, 6); _op(0x00, "BRK", IMP, 7)
for code, mn in ((0x10, "BPL"), (0x30, "BMI"), (0x50, "BVC"), (0x70, "BVS"),
                 (0x90, "BCC"), (0xB0, "BCS"), (0xD0, "BNE"), (0xF0, "BEQ")):
    _op(code, mn, REL, 2)
# ---- flags ----
_op(0x18, "CLC", IMP, 2); _op(0x38, "SEC", IMP, 2)
_op(0x58, "CLI", IMP, 2); _op(0x78, "SEI", IMP, 2)
_op(0xB8, "CLV", IMP, 2); _op(0xD8, "CLD", IMP, 2); _op(0xF8, "SED", IMP, 2)
_op(0xEA, "NOP", IMP, 2)
# ---- stable illegal opcodes ----
for code, mode, cyc in ((0x07, ZP, 5), (0x17, ZPX, 6), (0x0F, ABS, 6),
                        (0x1F, ABX, 7), (0x1B, ABY, 7), (0x03, IZX, 8), (0x13, IZY, 8)):
    _op(code, "SLO", mode, cyc)
for code, mode, cyc in ((0x27, ZP, 5), (0x37, ZPX, 6), (0x2F, ABS, 6),
                        (0x3F, ABX, 7), (0x3B, ABY, 7), (0x23, IZX, 8), (0x33, IZY, 8)):
    _op(code, "RLA", mode, cyc)
for code, mode, cyc in ((0x47, ZP, 5), (0x57, ZPX, 6), (0x4F, ABS, 6),
                        (0x5F, ABX, 7), (0x5B, ABY, 7), (0x43, IZX, 8), (0x53, IZY, 8)):
    _op(code, "SRE", mode, cyc)
for code, mode, cyc in ((0x67, ZP, 5), (0x77, ZPX, 6), (0x6F, ABS, 6),
                        (0x7F, ABX, 7), (0x7B, ABY, 7), (0x63, IZX, 8), (0x73, IZY, 8)):
    _op(code, "RRA", mode, cyc)
for code, mode, cyc in ((0xC7, ZP, 5), (0xD7, ZPX, 6), (0xCF, ABS, 6),
                        (0xDF, ABX, 7), (0xDB, ABY, 7), (0xC3, IZX, 8), (0xD3, IZY, 8)):
    _op(code, "DCP", mode, cyc)
for code, mode, cyc in ((0xE7, ZP, 5), (0xF7, ZPX, 6), (0xEF, ABS, 6),
                        (0xFF, ABX, 7), (0xFB, ABY, 7), (0xE3, IZX, 8), (0xF3, IZY, 8)):
    _op(code, "ISC", mode, cyc)
for code, mode, cyc, pagep in ((0xA7, ZP, 3, False), (0xB7, ZPY, 4, False),
                               (0xAF, ABS, 4, False), (0xBF, ABY, 4, True),
                               (0xA3, IZX, 6, False), (0xB3, IZY, 5, True)):
    _op(code, "LAX", mode, cyc, pagep)
for code, mode, cyc in ((0x87, ZP, 3), (0x97, ZPY, 4), (0x8F, ABS, 4), (0x83, IZX, 6)):
    _op(code, "SAX", mode, cyc)
_op(0x0B, "ANC", IMM, 2); _op(0x2B, "ANC", IMM, 2)
_op(0x4B, "ALR", IMM, 2); _op(0x6B, "ARR", IMM, 2)
_op(0xCB, "SBX", IMM, 2); _op(0xEB, "SBC", IMM, 2)  # USBC == SBC #imm
# illegal NOPs (multi-byte / multi-cycle)
for code in (0x1A, 0x3A, 0x5A, 0x7A, 0xDA, 0xFA):
    _op(code, "NOP", IMP, 2)
for code in (0x80, 0x82, 0x89, 0xC2, 0xE2):
    _op(code, "NOP", IMM, 2)
for code in (0x04, 0x44, 0x64):
    _op(code, "NOP", ZP, 3)
for code in (0x14, 0x34, 0x54, 0x74, 0xD4, 0xF4):
    _op(code, "NOP", ZPX, 4)
_op(0x0C, "NOP", ABS, 4)
for code in (0x1C, 0x3C, 0x5C, 0x7C, 0xDC, 0xFC):
    _op(code, "NOP", ABX, 4, True)

JAM_OPCODES = frozenset(
    (0x02, 0x12, 0x22, 0x32, 0x42, 0x52, 0x62, 0x72, 0x92, 0xB2, 0xD2, 0xF2)
)
# Unstable illegals: fail loud until a real program exercises one with an
# observed contract (dos_re rule: no datasheet-driven completeness).
UNSTABLE_OPCODES = frozenset((0x8B, 0xAB, 0xBB, 0x9B, 0x93, 0x9F, 0x9E, 0x9C))

NMI_VECTOR = 0xFFFA
RESET_VECTOR = 0xFFFC
IRQ_VECTOR = 0xFFFE

Hook = Callable[["CPU6502"], None]


class CPU6502:
    """The interpreter.  ``mem`` must expose ``rb(addr)`` / ``wb(addr, val)``
    (banked CPU view) — see :mod:`c64_re.memory`.

    Hook seams (mirroring dos_re's CPU8086):

    - ``replacement_hooks[pc]`` — game-recovery replacements, installed by
      the adapter's :class:`c64_re.hooks.HookRegistry`; routed through
      ``hook_verifier`` when live verification is active.
    - ``service_hooks[pc]`` — framework HLE seams (the KERNAL shim bodies);
      never verified, never counted as recovered code.
    - ``tick`` — called with the cycle cost of every executed instruction;
      the machine advances VIC raster / CIA timers here.  Interrupt *lines*
      are sampled between instructions: ``irq_line()`` (level-triggered) and
      the edge-triggered ``nmi_pending`` flag.
    """

    def __init__(self, mem, state: CPUState | None = None) -> None:
        self.mem = mem
        self.s = state or CPUState()
        self.replacement_hooks: dict[int, Hook] = {}
        self.hook_names: dict[int, str] = {}
        self.service_hooks: dict[int, Hook] = {}
        self.tick: Callable[[int], None] | None = None
        self.irq_line: Callable[[], bool] | None = None
        self.nmi_pending = False
        self.trace_fn: Callable[[CPU6502, int, int], None] | None = None
        # When True, pending IRQ/NMI stay latched but are not delivered —
        # the hook verifier uses this to run the oracle over a routine body
        # without interrupt interleaving (which a hook cannot reproduce and
        # which the frame/tick oracles judge instead).
        self.inhibit_interrupts = False
        self.instr_count = 0
        self.cycle_count = 0
        # live hook verification routing (see verification.py)
        self.hook_verifier = None
        self.hook_verifier_passthrough: set[int] = set()

    # ---- reset / interrupts -------------------------------------------------
    def reset(self) -> None:
        self.s.pc = self.read_word(RESET_VECTOR)
        self.s.sp = 0xFD
        self.s.i = 1

    def read_word(self, addr: int) -> int:
        return self.mem.rb(addr) | (self.mem.rb((addr + 1) & 0xFFFF) << 8)

    def push(self, val: int) -> None:
        self.mem.wb(0x0100 | self.s.sp, val & 0xFF)
        self.s.sp = (self.s.sp - 1) & 0xFF

    def pull(self) -> int:
        self.s.sp = (self.s.sp + 1) & 0xFF
        return self.mem.rb(0x0100 | self.s.sp)

    def _interrupt(self, vector: int, *, brk: bool) -> None:
        s = self.s
        self.push((s.pc >> 8) & 0xFF)
        self.push(s.pc & 0xFF)
        self.push(s.get_p(b=1 if brk else 0))
        s.i = 1  # NMOS: D is NOT cleared
        s.pc = self.read_word(vector)
        self.cycle_count += 7
        if self.tick is not None:
            self.tick(7)

    def addr(self) -> int:
        """The current instruction address — dos_re's ``cpu.addr()`` analogue."""
        return self.s.pc & 0xFFFF

    # ---- main loop -----------------------------------------------------------
    def step(self) -> None:
        # Interrupt lines are sampled between instructions (the one-instruction
        # CLI/SEI delay of the real chip is not modeled; deterministic either way).
        if not self.inhibit_interrupts:
            if self.nmi_pending:
                self.nmi_pending = False
                self._interrupt(NMI_VECTOR, brk=False)
            elif self.s.i == 0 and self.irq_line is not None and self.irq_line():
                self._interrupt(IRQ_VECTOR, brk=False)

        pc = self.s.pc & 0xFFFF

        fn = self.service_hooks.get(pc)
        if fn is not None:
            fn(self)
            return

        fn = self.replacement_hooks.get(pc)
        if fn is not None:
            verifier = self.hook_verifier
            if verifier is not None and pc not in self.hook_verifier_passthrough:
                verifier(self, pc, fn, self.hook_names.get(pc, getattr(fn, "__name__", "replacement")))
            else:
                fn(self)
            return

        if self.trace_fn is not None:
            self.trace_fn(self, pc, self.mem.rb(pc))

        self._execute(pc)

    def run(self, max_instructions: int) -> int:
        """Step up to ``max_instructions``; returns how many actually ran."""
        for n in range(max_instructions):
            self.step()
        return max_instructions

    # ---- flag helpers ----------------------------------------------------------
    def _nz(self, val: int) -> int:
        val &= 0xFF
        self.s.n = val >> 7
        self.s.z = 1 if val == 0 else 0
        return val

    # ---- execution ---------------------------------------------------------------
    def _execute(self, pc: int) -> None:
        s = self.s
        mem = self.mem
        opcode = mem.rb(pc)
        entry = OPCODES.get(opcode)
        if entry is None:
            if opcode in JAM_OPCODES:
                raise CPUJam(f"JAM opcode {opcode:02X} at ${pc:04X}")
            if opcode in UNSTABLE_OPCODES:
                raise UnimplementedOpcode(
                    f"unstable illegal opcode {opcode:02X} at ${pc:04X} — "
                    "implement with an observed contract when a real program needs it"
                )
            raise UnimplementedOpcode(f"opcode {opcode:02X} at ${pc:04X}")
        mn, mode, cycles, pagep = entry

        # ---- effective address / operand ----
        addr = -1
        if mode == IMP or mode == ACC:
            s.pc = (pc + 1) & 0xFFFF
        elif mode == IMM:
            addr = (pc + 1) & 0xFFFF
            s.pc = (pc + 2) & 0xFFFF
        elif mode == ZP:
            addr = mem.rb((pc + 1) & 0xFFFF)
            s.pc = (pc + 2) & 0xFFFF
        elif mode == ZPX:
            addr = (mem.rb((pc + 1) & 0xFFFF) + s.x) & 0xFF
            s.pc = (pc + 2) & 0xFFFF
        elif mode == ZPY:
            addr = (mem.rb((pc + 1) & 0xFFFF) + s.y) & 0xFF
            s.pc = (pc + 2) & 0xFFFF
        elif mode == ABS:
            addr = mem.rb((pc + 1) & 0xFFFF) | (mem.rb((pc + 2) & 0xFFFF) << 8)
            s.pc = (pc + 3) & 0xFFFF
        elif mode == ABX:
            base = mem.rb((pc + 1) & 0xFFFF) | (mem.rb((pc + 2) & 0xFFFF) << 8)
            addr = (base + s.x) & 0xFFFF
            if pagep and (base & 0xFF00) != (addr & 0xFF00):
                cycles += 1
            s.pc = (pc + 3) & 0xFFFF
        elif mode == ABY:
            base = mem.rb((pc + 1) & 0xFFFF) | (mem.rb((pc + 2) & 0xFFFF) << 8)
            addr = (base + s.y) & 0xFFFF
            if pagep and (base & 0xFF00) != (addr & 0xFF00):
                cycles += 1
            s.pc = (pc + 3) & 0xFFFF
        elif mode == IND:
            ptr = mem.rb((pc + 1) & 0xFFFF) | (mem.rb((pc + 2) & 0xFFFF) << 8)
            # NMOS page-wrap bug: ($xxFF) reads the high byte from $xx00.
            lo = mem.rb(ptr)
            hi = mem.rb((ptr & 0xFF00) | ((ptr + 1) & 0xFF))
            addr = lo | (hi << 8)
            s.pc = (pc + 3) & 0xFFFF
        elif mode == IZX:
            zp = (mem.rb((pc + 1) & 0xFFFF) + s.x) & 0xFF
            addr = mem.rb(zp) | (mem.rb((zp + 1) & 0xFF) << 8)
            s.pc = (pc + 2) & 0xFFFF
        elif mode == IZY:
            zp = mem.rb((pc + 1) & 0xFFFF)
            base = mem.rb(zp) | (mem.rb((zp + 1) & 0xFF) << 8)
            addr = (base + s.y) & 0xFFFF
            if pagep and (base & 0xFF00) != (addr & 0xFF00):
                cycles += 1
            s.pc = (pc + 2) & 0xFFFF
        elif mode == REL:
            off = mem.rb((pc + 1) & 0xFFFF)
            if off >= 0x80:
                off -= 0x100
            addr = (pc + 2 + off) & 0xFFFF
            s.pc = (pc + 2) & 0xFFFF

        # ---- semantics ----
        if mn == "LDA":
            s.a = self._nz(mem.rb(addr))
        elif mn == "LDX":
            s.x = self._nz(mem.rb(addr))
        elif mn == "LDY":
            s.y = self._nz(mem.rb(addr))
        elif mn == "STA":
            mem.wb(addr, s.a)
        elif mn == "STX":
            mem.wb(addr, s.x)
        elif mn == "STY":
            mem.wb(addr, s.y)
        elif mn == "LAX":
            s.a = s.x = self._nz(mem.rb(addr))
        elif mn == "SAX":
            mem.wb(addr, s.a & s.x)
        elif mn == "TAX":
            s.x = self._nz(s.a)
        elif mn == "TAY":
            s.y = self._nz(s.a)
        elif mn == "TXA":
            s.a = self._nz(s.x)
        elif mn == "TYA":
            s.a = self._nz(s.y)
        elif mn == "TSX":
            s.x = self._nz(s.sp)
        elif mn == "TXS":
            s.sp = s.x
        elif mn == "PHA":
            self.push(s.a)
        elif mn == "PLA":
            s.a = self._nz(self.pull())
        elif mn == "PHP":
            self.push(s.get_p(b=1))
        elif mn == "PLP":
            s.set_p(self.pull())
        elif mn == "AND":
            s.a = self._nz(s.a & mem.rb(addr))
        elif mn == "ORA":
            s.a = self._nz(s.a | mem.rb(addr))
        elif mn == "EOR":
            s.a = self._nz(s.a ^ mem.rb(addr))
        elif mn == "ADC":
            self._adc(mem.rb(addr))
        elif mn == "SBC":
            self._sbc(mem.rb(addr))
        elif mn == "CMP":
            self._compare(s.a, mem.rb(addr))
        elif mn == "CPX":
            self._compare(s.x, mem.rb(addr))
        elif mn == "CPY":
            self._compare(s.y, mem.rb(addr))
        elif mn == "BIT":
            v = mem.rb(addr)
            s.n = v >> 7
            s.v = (v >> 6) & 1
            s.z = 1 if (s.a & v) == 0 else 0
        elif mn == "INC":
            old = mem.rb(addr)
            mem.wb(addr, old)  # NMOS RMW double write
            v = (old + 1) & 0xFF
            mem.wb(addr, v)
            self._nz(v)
        elif mn == "DEC":
            old = mem.rb(addr)
            mem.wb(addr, old)
            v = (old - 1) & 0xFF
            mem.wb(addr, v)
            self._nz(v)
        elif mn == "INX":
            s.x = self._nz(s.x + 1)
        elif mn == "INY":
            s.y = self._nz(s.y + 1)
        elif mn == "DEX":
            s.x = self._nz(s.x - 1)
        elif mn == "DEY":
            s.y = self._nz(s.y - 1)
        elif mn == "ASL":
            if mode == ACC:
                s.c = s.a >> 7
                s.a = self._nz((s.a << 1) & 0xFF)
            else:
                old = mem.rb(addr)
                mem.wb(addr, old)
                s.c = old >> 7
                v = (old << 1) & 0xFF
                mem.wb(addr, v)
                self._nz(v)
        elif mn == "LSR":
            if mode == ACC:
                s.c = s.a & 1
                s.a = self._nz(s.a >> 1)
            else:
                old = mem.rb(addr)
                mem.wb(addr, old)
                s.c = old & 1
                v = old >> 1
                mem.wb(addr, v)
                self._nz(v)
        elif mn == "ROL":
            if mode == ACC:
                v = ((s.a << 1) | s.c) & 0x1FF
                s.c = v >> 8
                s.a = self._nz(v & 0xFF)
            else:
                old = mem.rb(addr)
                mem.wb(addr, old)
                v = ((old << 1) | s.c) & 0x1FF
                s.c = v >> 8
                mem.wb(addr, v & 0xFF)
                self._nz(v & 0xFF)
        elif mn == "ROR":
            if mode == ACC:
                v = s.a | (s.c << 8)
                s.c = v & 1
                s.a = self._nz(v >> 1)
            else:
                old = mem.rb(addr)
                mem.wb(addr, old)
                v = old | (s.c << 8)
                s.c = v & 1
                mem.wb(addr, v >> 1)
                self._nz(v >> 1)
        elif mn == "JMP":
            s.pc = addr
        elif mn == "JSR":
            ret = (pc + 2) & 0xFFFF  # address of JSR's last byte
            self.push((ret >> 8) & 0xFF)
            self.push(ret & 0xFF)
            s.pc = addr
        elif mn == "RTS":
            lo = self.pull()
            hi = self.pull()
            s.pc = ((lo | (hi << 8)) + 1) & 0xFFFF
        elif mn == "RTI":
            s.set_p(self.pull())
            lo = self.pull()
            hi = self.pull()
            s.pc = lo | (hi << 8)
        elif mn == "BRK":
            # pushes PC+2 (opcode + padding byte), B set in the pushed P
            s.pc = (pc + 2) & 0xFFFF
            self.push((s.pc >> 8) & 0xFF)
            self.push(s.pc & 0xFF)
            self.push(s.get_p(b=1))
            s.i = 1
            s.pc = self.read_word(IRQ_VECTOR)
        elif mn in ("BPL", "BMI", "BVC", "BVS", "BCC", "BCS", "BNE", "BEQ"):
            taken = {
                "BPL": s.n == 0, "BMI": s.n == 1,
                "BVC": s.v == 0, "BVS": s.v == 1,
                "BCC": s.c == 0, "BCS": s.c == 1,
                "BNE": s.z == 0, "BEQ": s.z == 1,
            }[mn]
            if taken:
                cycles += 1
                if (s.pc & 0xFF00) != (addr & 0xFF00):
                    cycles += 1
                s.pc = addr
        elif mn == "CLC":
            s.c = 0
        elif mn == "SEC":
            s.c = 1
        elif mn == "CLI":
            s.i = 0
        elif mn == "SEI":
            s.i = 1
        elif mn == "CLV":
            s.v = 0
        elif mn == "CLD":
            s.d = 0
        elif mn == "SED":
            s.d = 1
        elif mn == "NOP":
            if mode not in (IMP, ACC) and addr >= 0:
                mem.rb(addr)  # dummy read (can matter for IO-mapped addresses)
        # ---- stable illegals with RMW+ALU composition ----
        elif mn == "SLO":
            old = mem.rb(addr)
            mem.wb(addr, old)
            s.c = old >> 7
            v = (old << 1) & 0xFF
            mem.wb(addr, v)
            s.a = self._nz(s.a | v)
        elif mn == "RLA":
            old = mem.rb(addr)
            mem.wb(addr, old)
            v = ((old << 1) | s.c) & 0x1FF
            s.c = v >> 8
            v &= 0xFF
            mem.wb(addr, v)
            s.a = self._nz(s.a & v)
        elif mn == "SRE":
            old = mem.rb(addr)
            mem.wb(addr, old)
            s.c = old & 1
            v = old >> 1
            mem.wb(addr, v)
            s.a = self._nz(s.a ^ v)
        elif mn == "RRA":
            old = mem.rb(addr)
            mem.wb(addr, old)
            v = old | (s.c << 8)
            s.c = v & 1
            v >>= 1
            mem.wb(addr, v)
            self._adc(v)
        elif mn == "DCP":
            old = mem.rb(addr)
            mem.wb(addr, old)
            v = (old - 1) & 0xFF
            mem.wb(addr, v)
            self._compare(s.a, v)
        elif mn == "ISC":
            old = mem.rb(addr)
            mem.wb(addr, old)
            v = (old + 1) & 0xFF
            mem.wb(addr, v)
            self._sbc(v)
        elif mn == "ANC":
            s.a = self._nz(s.a & mem.rb(addr))
            s.c = s.n
        elif mn == "ALR":
            v = s.a & mem.rb(addr)
            s.c = v & 1
            s.a = self._nz(v >> 1)
        elif mn == "ARR":
            self._arr(mem.rb(addr))
        elif mn == "SBX":
            v = mem.rb(addr)
            t = (s.a & s.x) - v
            s.c = 1 if t >= 0 else 0
            s.x = self._nz(t & 0xFF)
        else:  # pragma: no cover - table and dispatch must stay in sync
            raise UnimplementedOpcode(f"decoded {mn} without semantics at ${pc:04X}")

        self.instr_count += 1
        self.cycle_count += cycles
        if self.tick is not None:
            self.tick(cycles)

    # ---- arithmetic helpers (NMOS semantics incl. decimal mode) ----
    def _compare(self, reg: int, v: int) -> None:
        t = reg - v
        self.s.c = 1 if t >= 0 else 0
        self._nz(t & 0xFF)

    def _adc(self, v: int) -> None:
        s = self.s
        if s.d:
            # NMOS decimal ADC: Z from the binary result; N/V from the
            # intermediate high nibble before the +6 carry fix.
            lo = (s.a & 0x0F) + (v & 0x0F) + s.c
            half = 0
            if lo > 9:
                lo += 6
                half = 1
            hi = (s.a >> 4) + (v >> 4) + half
            binary = (s.a + v + s.c) & 0xFF
            s.z = 1 if binary == 0 else 0
            s.n = (hi >> 3) & 1
            s.v = 1 if (~(s.a ^ v) & (s.a ^ (hi << 4)) & 0x80) else 0
            carry = 0
            if hi > 9:
                hi += 6
                carry = 1
            s.c = carry
            s.a = ((hi << 4) | (lo & 0x0F)) & 0xFF
        else:
            t = s.a + v + s.c
            s.v = 1 if (~(s.a ^ v) & (s.a ^ t) & 0x80) else 0
            s.c = 1 if t > 0xFF else 0
            s.a = self._nz(t)

    def _sbc(self, v: int) -> None:
        s = self.s
        borrow = 1 - s.c
        t = s.a - v - borrow
        if s.d:
            # NMOS decimal SBC: all flags from the binary result; only A adjusts.
            lo = (s.a & 0x0F) - (v & 0x0F) - borrow
            if lo & 0x10:
                a_dec = ((lo - 6) & 0x0F) | (((s.a & 0xF0) - (v & 0xF0) - 0x10) & 0x1F0)
            else:
                a_dec = (lo & 0x0F) | (((s.a & 0xF0) - (v & 0xF0)) & 0x1F0)
            if a_dec & 0x100:
                a_dec -= 0x60
            s.v = 1 if ((s.a ^ v) & (s.a ^ t) & 0x80) else 0
            s.c = 1 if t >= 0 else 0
            self._nz(t & 0xFF)
            s.a = a_dec & 0xFF
        else:
            s.v = 1 if ((s.a ^ v) & (s.a ^ t) & 0x80) else 0
            s.c = 1 if t >= 0 else 0
            s.a = self._nz(t & 0xFF)

    def _arr(self, v: int) -> None:
        # AND #imm then ROR A, with the documented NMOS flag quirks
        # (VICE semantics, both binary and decimal modes).
        s = self.s
        t = s.a & v
        rored = (t >> 1) | (s.c << 7)
        if s.d:
            s.n = s.c  # old carry
            s.z = 1 if rored == 0 else 0
            s.v = 1 if ((t ^ rored) & 0x40) else 0
            if (t & 0x0F) + (t & 0x01) > 5:
                rored = (rored & 0xF0) | ((rored + 6) & 0x0F)
            if (t & 0xF0) + (t & 0x10) > 0x50:
                rored = (rored + 0x60) & 0xFF
                s.c = 1
            else:
                s.c = 0
            s.a = rored & 0xFF
        else:
            s.a = self._nz(rored)
            s.c = (rored >> 6) & 1
            s.v = ((rored >> 6) ^ (rored >> 5)) & 1
