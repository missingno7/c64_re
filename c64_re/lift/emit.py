"""The emitter (M1): FunctionScan -> a literal, self-contained Python hook.

Faithful by reuse, never by re-derivation:

- flag/ALU subtleties call the interpreter's own helpers (``cpu._adc``,
  ``cpu._sbc``, ``cpu._compare``, ``cpu._nz``);
- anything not in the deliberately-small inline set is emitted as an exact
  single-instruction interpreter fallback (``interp_one``);
- JSR becomes ``emulate_call`` — callees run through the VM, so hooks
  compose and lifting order is irrelevant;
- memory access order, RMW double writes, and per-instruction cycle ticks
  (including page-cross and taken-branch penalties) mirror the interpreter
  exactly, so chip time and I/O side effects are indistinguishable;
- instruction operands inside the region are constant-folded — sound
  because the fail-loud SMC entry guard proves the region bytes unchanged
  on every call (an unknown runtime-patched variant refuses; it never runs
  wrong).

The artifact is meant to be read: every instruction carries its
disassembly, and blocks dispatch on their original addresses.
"""
from __future__ import annotations

from ..cpu import ABS, ABX, ABY, ACC, IMM, IMP, IZX, IZY, OPCODES, ZP, ZPX, ZPY
from ..dis6502 import disassemble_one
from .cfg import FunctionScan
from .decode import BRANCH, CALL, JMP_ABS, RET, SEQ

_BRANCH_COND = {
    "BPL": "s.n == 0", "BMI": "s.n == 1", "BVC": "s.v == 0", "BVS": "s.v == 1",
    "BCC": "s.c == 0", "BCS": "s.c == 1", "BNE": "s.z == 0", "BEQ": "s.z == 1",
}

# mnemonics inlined by the emitter; everything else non-control-flow falls
# back to interp_one (exact by definition)
_INLINE = frozenset((
    "LDA", "LDX", "LDY", "STA", "STX", "STY", "TAX", "TAY", "TXA", "TYA",
    "TSX", "TXS", "PHA", "PLA", "PHP", "PLP", "AND", "ORA", "EOR", "ADC",
    "SBC", "CMP", "CPX", "CPY", "BIT", "INC", "DEC", "INX", "INY", "DEX",
    "DEY", "ASL", "LSR", "ROL", "ROR", "CLC", "SEC", "CLI", "SEI", "CLV",
    "CLD", "SED", "NOP",
))


class _W:
    def __init__(self) -> None:
        self.lines: list[str] = []

    def put(self, indent: int, text: str) -> None:
        self.lines.append("    " * indent + text)

    def text(self) -> str:
        return "\n".join(self.lines) + "\n"


def _addressing(insn, read):
    """(setup_lines, addr_expr, cycles_expr) for the operand of ``insn``.

    ``cycles_expr`` is an int for fixed-cost forms or the name ``cyc`` set
    by the setup lines when a page-cross penalty applies at runtime.
    """
    mn, mode, base_cycles, pagep = OPCODES[insn.opcode]
    pc = insn.pc
    if mode in (IMP, ACC):
        return [], None, base_cycles
    if mode == IMM:
        return [], None, base_cycles  # value constant-folded by caller
    op1 = read((pc + 1) & 0xFFFF)
    if mode == ZP:
        return [], f"0x{op1:02X}", base_cycles
    if mode == ZPX:
        return [f"addr = (0x{op1:02X} + s.x) & 0xFF"], "addr", base_cycles
    if mode == ZPY:
        return [f"addr = (0x{op1:02X} + s.y) & 0xFF"], "addr", base_cycles
    op16 = op1 | (read((pc + 2) & 0xFFFF) << 8)
    if mode == ABS:
        return [], f"0x{op16:04X}", base_cycles
    if mode in (ABX, ABY):
        reg = "x" if mode == ABX else "y"
        setup = [f"addr = (0x{op16:04X} + s.{reg}) & 0xFFFF"]
        if pagep:
            setup.append(
                f"cyc = {base_cycles} + (1 if (0x{op16:04X} ^ addr) & 0xFF00 else 0)"
            )
            return setup, "addr", "cyc"
        return setup, "addr", base_cycles
    if mode == IZX:
        setup = [
            f"zp = (0x{op1:02X} + s.x) & 0xFF",
            "addr = mem.rb(zp) | (mem.rb((zp + 1) & 0xFF) << 8)",
        ]
        return setup, "addr", base_cycles
    if mode == IZY:
        setup = [
            f"base = mem.rb(0x{op1:02X}) | (mem.rb(0x{(op1 + 1) & 0xFF:02X}) << 8)",
            "addr = (base + s.y) & 0xFFFF",
        ]
        if pagep:
            setup.append(
                f"cyc = {base_cycles} + (1 if (base ^ addr) & 0xFF00 else 0)"
            )
            return setup, "addr", "cyc"
        return setup, "addr", base_cycles
    raise AssertionError(f"unhandled mode for {mn} at ${pc:04X}")


def _emit_fin(w, ind, cycles) -> None:
    w.put(ind, "cpu.instr_count += 1")
    if isinstance(cycles, int):
        w.put(ind, f"cpu.cycle_count += {cycles}")
        w.put(ind, "if cpu.tick is not None:")
        w.put(ind + 1, f"cpu.tick({cycles})")
    else:
        w.put(ind, f"cpu.cycle_count += {cycles}")
        w.put(ind, "if cpu.tick is not None:")
        w.put(ind + 1, f"cpu.tick({cycles})")


def _rmw(w, ind, addr, transform_lines) -> None:
    w.put(ind, f"old = mem.rb({addr})")
    w.put(ind, f"mem.wb({addr}, old)  # NMOS RMW double write")
    for line in transform_lines:
        w.put(ind, line)
    w.put(ind, f"mem.wb({addr}, v)")


def _emit_inline(w, ind, insn, read) -> None:
    """One non-control-flow instruction from the inline set."""
    mn = insn.mnemonic
    mode = insn.mode
    setup, addr, cycles = _addressing(insn, read)
    for line in setup:
        w.put(ind, line)

    def rd():
        if mode == IMM:
            return f"0x{read((insn.pc + 1) & 0xFFFF):02X}"
        return f"mem.rb({addr})"

    if mn in ("LDA", "LDX", "LDY"):
        reg = mn[2].lower()
        w.put(ind, f"s.{reg} = cpu._nz({rd()})")
    elif mn in ("STA", "STX", "STY"):
        reg = mn[2].lower()
        w.put(ind, f"mem.wb({addr}, s.{reg})")
    elif mn in ("TAX", "TAY", "TXA", "TYA", "TSX"):
        src = {"TAX": "a", "TAY": "a", "TXA": "x", "TYA": "y", "TSX": "sp"}[mn]
        dst = {"TAX": "x", "TAY": "y", "TXA": "a", "TYA": "a", "TSX": "x"}[mn]
        w.put(ind, f"s.{dst} = cpu._nz(s.{src})")
    elif mn == "TXS":
        w.put(ind, "s.sp = s.x")
    elif mn == "PHA":
        w.put(ind, "cpu.push(s.a)")
    elif mn == "PLA":
        w.put(ind, "s.a = cpu._nz(cpu.pull())")
    elif mn == "PHP":
        w.put(ind, "cpu.push(s.get_p(b=1))")
    elif mn == "PLP":
        w.put(ind, "s.set_p(cpu.pull())")
    elif mn in ("AND", "ORA", "EOR"):
        op = {"AND": "&", "ORA": "|", "EOR": "^"}[mn]
        w.put(ind, f"s.a = cpu._nz(s.a {op} {rd()})")
    elif mn == "ADC":
        w.put(ind, f"cpu._adc({rd()})")
    elif mn == "SBC":
        w.put(ind, f"cpu._sbc({rd()})")
    elif mn in ("CMP", "CPX", "CPY"):
        reg = {"CMP": "a", "CPX": "x", "CPY": "y"}[mn]
        w.put(ind, f"cpu._compare(s.{reg}, {rd()})")
    elif mn == "BIT":
        w.put(ind, f"v = {rd()}")
        w.put(ind, "s.n = v >> 7")
        w.put(ind, "s.v = (v >> 6) & 1")
        w.put(ind, "s.z = 1 if (s.a & v) == 0 else 0")
    elif mn in ("INC", "DEC"):
        sign = "+" if mn == "INC" else "-"
        _rmw(w, ind, addr, [f"v = (old {sign} 1) & 0xFF"])
        w.put(ind, "cpu._nz(v)")
    elif mn in ("INX", "INY", "DEX", "DEY"):
        reg = mn[2].lower()
        sign = "+" if mn[0] == "I" else "-"
        w.put(ind, f"s.{reg} = cpu._nz(s.{reg} {sign} 1)")
    elif mn in ("ASL", "LSR", "ROL", "ROR"):
        if mode == ACC:
            if mn == "ASL":
                w.put(ind, "s.c = s.a >> 7")
                w.put(ind, "s.a = cpu._nz((s.a << 1) & 0xFF)")
            elif mn == "LSR":
                w.put(ind, "s.c = s.a & 1")
                w.put(ind, "s.a = cpu._nz(s.a >> 1)")
            elif mn == "ROL":
                w.put(ind, "v = ((s.a << 1) | s.c) & 0x1FF")
                w.put(ind, "s.c = v >> 8")
                w.put(ind, "s.a = cpu._nz(v & 0xFF)")
            else:  # ROR
                w.put(ind, "v = s.a | (s.c << 8)")
                w.put(ind, "s.c = v & 1")
                w.put(ind, "s.a = cpu._nz(v >> 1)")
        else:
            if mn == "ASL":
                _rmw(w, ind, addr, ["s.c = old >> 7", "v = (old << 1) & 0xFF"])
            elif mn == "LSR":
                _rmw(w, ind, addr, ["s.c = old & 1", "v = old >> 1"])
            elif mn == "ROL":
                _rmw(w, ind, addr, ["v = ((old << 1) | s.c) & 0x1FF",
                                    "s.c = v >> 8", "v &= 0xFF"])
            else:  # ROR
                _rmw(w, ind, addr, ["v = old | (s.c << 8)", "s.c = v & 1",
                                    "v >>= 1"])
            w.put(ind, "cpu._nz(v)")
    elif mn in ("CLC", "SEC", "CLI", "SEI", "CLV", "CLD", "SED"):
        flag = {"CLC": ("c", 0), "SEC": ("c", 1), "CLI": ("i", 0),
                "SEI": ("i", 1), "CLV": ("v", 0), "CLD": ("d", 0),
                "SED": ("d", 1)}[mn]
        w.put(ind, f"s.{flag[0]} = {flag[1]}")
    elif mn == "NOP":
        if mode not in (IMP, ACC) and addr is not None:
            w.put(ind, f"mem.rb({addr})  # dummy read")
    else:  # pragma: no cover - guarded by _INLINE membership
        raise AssertionError(mn)
    _emit_fin(w, ind, cycles)


def emit_hook(scan: FunctionScan, read, *, name: str | None = None) -> str:
    """Generate the hook module source for a scanned function."""
    entry = scan.entry
    name = name or f"lifted_{entry:04X}"
    w = _W()
    w.put(0, f'"""Lifted hook for ${entry:04X} — generated by c64_re.lift.emit.')
    w.put(0, "")
    w.put(0, "Literal per-instruction artifact; refactor from it, do not hand-tune")
    w.put(0, 'semantics.  Proven per-call by the differential hook oracle."""')
    w.put(0, "from c64_re.lift.runtime import emulate_call, interp_one")
    w.put(0, "")
    w.put(0, f"ENTRY = 0x{entry:04X}")
    w.put(0, f"NAME = {name!r}")
    guards = []
    for lo, hi in scan.byte_ranges:
        blob = bytes(read(a & 0xFFFF) for a in range(lo, hi))
        guards.append(f"    (0x{lo:04X}, 0x{hi:04X}, bytes.fromhex({blob.hex()!r})),")
    w.put(0, "GUARD_RANGES = [")
    for g in guards:
        w.put(0, g)
    w.put(0, "]")
    w.put(0, "")
    w.put(0, "")
    w.put(0, "def hook(cpu):")
    w.put(1, "s = cpu.s")
    w.put(1, "mem = cpu.mem")
    w.put(1, "for lo, hi, want in GUARD_RANGES:")
    w.put(2, "if hi <= 0xA000:")
    w.put(3, "live = bytes(mem.ram[lo:hi])")
    w.put(2, "else:")
    w.put(3, "live = bytes(mem.rb(a) for a in range(lo, hi))")
    w.put(2, "if live != want:")
    w.put(3, "raise RuntimeError(")
    w.put(4, f"f\"lifted hook {name} at ${{ENTRY:04X}} saw runtime-patched code \"")
    w.put(4, "f\"in ${lo:04X}-${hi - 1:04X} — unknown variant, refusing to run\")")
    w.put(1, f"b = 0x{entry:04X}")
    w.put(1, "while True:")

    # ---- block construction ----
    starts = sorted(pc for pc in scan.block_starts if pc in scan.insns)
    start_set = set(starts)
    first = True
    for start in starts:
        w.put(2, f"{'if' if first else 'elif'} b == 0x{start:04X}:")
        first = False
        pc = start
        ind = 3
        while True:
            insn = scan.insns[pc]
            w.put(ind, f"# {disassemble_one(read, pc)[0]}")
            nxt = (pc + insn.size) & 0xFFFF
            if insn.flow == RET:
                if insn.mnemonic == "RTS":
                    w.put(ind, "lo = cpu.pull()")
                    w.put(ind, "hi = cpu.pull()")
                    w.put(ind, "s.pc = ((lo | (hi << 8)) + 1) & 0xFFFF")
                else:  # RTI
                    w.put(ind, "s.set_p(cpu.pull())")
                    w.put(ind, "lo = cpu.pull()")
                    w.put(ind, "hi = cpu.pull()")
                    w.put(ind, "s.pc = lo | (hi << 8)")
                _emit_fin(w, ind, 6)
                w.put(ind, "return")
                break
            if insn.flow == JMP_ABS:
                _emit_fin(w, ind, 3)
                w.put(ind, f"b = 0x{insn.target:04X}")
                w.put(ind, "continue")
                break
            if insn.flow == BRANCH:
                taken = 3 + (1 if ((nxt ^ insn.target) & 0xFF00) else 0)
                w.put(ind, f"if {_BRANCH_COND[insn.mnemonic]}:")
                _emit_fin(w, ind + 1, taken)
                w.put(ind + 1, f"b = 0x{insn.target:04X}")
                w.put(ind + 1, "continue")
                _emit_fin(w, ind, 2)
                w.put(ind, f"b = 0x{nxt:04X}")
                w.put(ind, "continue")
                break
            if insn.flow == CALL:
                _emit_fin(w, ind, 6)
                w.put(ind, f"emulate_call(cpu, 0x{insn.target:04X}, 0x{nxt:04X})")
            elif insn.flow == SEQ:
                if insn.mnemonic in _INLINE:
                    _emit_inline(w, ind, insn, read)
                else:
                    w.put(ind, f"interp_one(cpu, 0x{pc:04X})  # exact fallback")
            else:  # pragma: no cover - cfg refuses other flows
                raise AssertionError(insn.flow)
            if nxt in start_set:
                w.put(ind, f"b = 0x{nxt:04X}")
                w.put(ind, "continue")
                break
            pc = nxt
    w.put(2, "else:")
    w.put(3, "raise AssertionError(f\"lifted dispatch reached unknown block ${b:04X}\")")
    return w.text()


def lift_and_compile(read, entry: int, *, name: str | None = None,
                     max_instructions: int = 768):
    """Scan + emit + compile in one step (for drivers and tests).

    Returns ``(hook_callable, source_text, scan)`` — or raises
    ``LiftRefused`` carrying the structured refusal.
    """
    from .cfg import scan_function

    scan = scan_function(read, entry, max_instructions=max_instructions)
    if not scan:
        raise LiftRefused(scan)
    source = emit_hook(scan, read, name=name)
    namespace: dict = {}
    exec(compile(source, f"<lifted ${entry:04X}>", "exec"), namespace)  # noqa: S102
    return namespace["hook"], source, scan


class LiftRefused(RuntimeError):
    def __init__(self, refusal) -> None:
        super().__init__(f"lift refused at ${refusal.entry:04X}: "
                         f"{refusal.reason} ({refusal.detail})")
        self.refusal = refusal
