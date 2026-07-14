"""KERNAL HLE: a synthesized shim ROM + Python service traps.

This is c64_re's analogue of dos_re's DOS/BIOS layer.  No original
Commodore ROM bytes are included; instead :func:`build_shim_kernal`
synthesizes an 8 KB image that is *shaped* like the real KERNAL where
programs depend on its shape:

- the hardware vectors ($FFFA/$FFFC/$FFFE) and the real IRQ/NMI dispatch
  stubs at their authentic addresses ($FF48 tests the BRK bit and routes
  through $0314/$0316; $FE43 routes NMI through $0318),
- the KERNAL API jump table ($FF81-$FFF3) with the authentic
  ``JMP ($03xx)`` indirections for the vectored calls, so programs that
  hook or restore the RAM vector table at $0314-$0333 work,
- the default vector *bodies* at their authentic addresses ($F1CA CHROUT,
  $F4A5 LOAD, $EA31 the default IRQ handler, $EA81 the register-restore
  RTI tail, ...), where Python service traps implement the documented
  register contracts,
- everything else is filled with a JAM opcode: executing unimplemented
  KERNAL code fails loud with the exact address, and becomes the next
  work item — never a silent fake.

Programs that *read* ROM as data (or need the real BASIC interpreter) need
real ROM images — drop them in a ``roms/`` directory (``kernal``, ``basic``,
``chargen``); the runtime picks them up and keeps the HLE traps installed
on top (see runtime.create_runtime).

The synthesized character ROM carries a readable clean-room 8x8 font for
screen codes 0-63 (letters, digits, punctuation) and deterministic
placeholder glyphs for the graphics characters; frame oracles compare a
candidate against a reference using the *same* chargen, so recovery
verification is unaffected — only cosmetics differ from real hardware.
"""
from __future__ import annotations

from pathlib import Path

from .d64 import prg_load_address, prg_payload

JAM = 0x02  # executing this raises CPUJam with the address — the fail-loud filler


class ProgramExit(Exception):
    """The booted program returned to its top-level caller (RTS to the boot
    stack frame) — the C64 analogue of a DOS program terminating."""


class KernalError(RuntimeError):
    """An unimplemented or contract-violating KERNAL service was reached."""


# ---- authentic addresses (shape the shim mirrors) --------------------------------
IRQ_ENTRY = 0xFF48
NMI_ENTRY = 0xFE43
NMI_DEFAULT_BODY = 0xFE47
BRK_DEFAULT_BODY = 0xFE66
IRQ_DEFAULT_BODY = 0xEA31
IRQ_RESTORE_TAIL = 0xEA81
RESET_BODY = 0xFCE2
EXIT_TRAP = 0xE05A          # shim-only: boot pushes this-1; RTS here = program exit
LOAD_PREAMBLE = 0xF49E      # STX $C3 / STY $C4 / JMP ($0330), as on the real ROM

# RAM vector table ($0314-$0333) power-on defaults — authentic values.
DEFAULT_VECTORS: dict[int, int] = {
    0x0314: IRQ_DEFAULT_BODY, 0x0316: BRK_DEFAULT_BODY, 0x0318: NMI_DEFAULT_BODY,
    0x031A: 0xF34A, 0x031C: 0xF291, 0x031E: 0xF20E, 0x0320: 0xF250,
    0x0322: 0xF333, 0x0324: 0xF157, 0x0326: 0xF1CA, 0x0328: 0xF6ED,
    0x032A: 0xF13E, 0x032C: 0xF32F, 0x032E: BRK_DEFAULT_BODY,
    0x0330: 0xF4A5, 0x0332: 0xF5ED,
}

# Jump-table entries that go through a RAM vector: table addr -> vector addr.
VECTORED_API = {
    0xFFC0: 0x031A, 0xFFC3: 0x031C, 0xFFC6: 0x031E, 0xFFC9: 0x0320,
    0xFFCC: 0x0322, 0xFFCF: 0x0324, 0xFFD2: 0x0326, 0xFFE1: 0x0328,
    0xFFE4: 0x032A, 0xFFE7: 0x032C,
}

# Direct jump-table entries (trap installed at the table address itself).
DIRECT_API = {
    0xFF81: "CINT", 0xFF84: "IOINIT", 0xFF87: "RAMTAS", 0xFF8A: "RESTOR",
    0xFF8D: "VECTOR", 0xFF90: "SETMSG", 0xFF93: "SECOND", 0xFF96: "TKSA",
    0xFF99: "MEMTOP", 0xFF9C: "MEMBOT", 0xFF9F: "SCNKEY", 0xFFA2: "SETTMO",
    0xFFA5: "ACPTR", 0xFFA8: "CIOUT", 0xFFAB: "UNTLK", 0xFFAE: "UNLSN",
    0xFFB1: "LISTEN", 0xFFB4: "TALK", 0xFFB7: "READST", 0xFFBA: "SETLFS",
    0xFFBD: "SETNAM", 0xFFDB: "SETTIM", 0xFFDE: "RDTIM", 0xFFEA: "UDTIM",
    0xFFED: "SCREEN", 0xFFF0: "PLOT", 0xFFF3: "IOBASE",
}

# Vector bodies implemented as traps: authentic body addr -> service name.
BODY_API = {
    0xF34A: "OPEN", 0xF291: "CLOSE", 0xF20E: "CHKIN", 0xF250: "CHKOUT",
    0xF333: "CLRCHN", 0xF157: "CHRIN", 0xF1CA: "CHROUT", 0xF6ED: "STOP",
    0xF13E: "GETIN", 0xF32F: "CLALL", 0xF4A5: "LOAD", 0xF5ED: "SAVE",
    # KERNAL internals games JSR into directly (observed need, not datasheet —
    # these are the canonical direct-call entries of KERNAL init routines):
    0xE518: "CINT",     # screen editor init (body of CINT's second half)
    0xE536: "CLRSCR",   # screen editor reset tail: falls into clear screen (Stix init)
    0xE544: "CLRSCR",   # screen editor: clear screen
    0xE566: "HOMECRS",  # screen editor: cursor home
    0xE5A0: "VICDEFAULTS",  # set VIC register defaults + screen editor base
    0xFD15: "RESTOR",   # restore RAM vector table (body of RESTOR)
    0xFD50: "RAMTAS",   # RAM test/init (body of RAMTAS)
    0xFDA3: "IOINIT",   # CIA/IRQ init (body of IOINIT)
    0xFF5B: "CINT",     # CINT tail (PAL/NTSC detect + editor init)
}


def build_shim_kernal() -> bytes:
    rom = bytearray([JAM]) * 0x2000

    def put(addr: int, data: bytes) -> None:
        rom[addr - 0xE000: addr - 0xE000 + len(data)] = data

    # hardware IRQ entry: save regs, dispatch BRK vs IRQ through the RAM vectors
    put(IRQ_ENTRY, bytes((
        0x48,              # PHA
        0x8A, 0x48,        # TXA / PHA
        0x98, 0x48,        # TYA / PHA
        0xBA,              # TSX
        0xBD, 0x04, 0x01,  # LDA $0104,X
        0x29, 0x10,        # AND #$10
        0xF0, 0x03,        # BEQ +3
        0x6C, 0x16, 0x03,  # JMP ($0316)   (BRK)
        0x6C, 0x14, 0x03,  # JMP ($0314)   (IRQ)
    )))
    # NMI entry
    put(NMI_ENTRY, bytes((0x78, 0x6C, 0x18, 0x03)))  # SEI / JMP ($0318)
    put(NMI_DEFAULT_BODY, bytes((0x40,)))            # RTI
    put(0xFEC1, bytes((0x40,)))  # RTI inside the real NMI body — cracks point
    #                              $0318 here to neutralize the RESTORE key
    # The RESTOR source table at its authentic address: programs block-copy
    # ROM $FD30 over $0314-$0333 to restore the standard vectors themselves
    # (observed in Stix's init) — the table image must therefore be real.
    table = bytearray()
    for vaddr in sorted(DEFAULT_VECTORS):
        target = DEFAULT_VECTORS[vaddr]
        table += bytes((target & 0xFF, target >> 8))
    put(0xFD30, bytes(table))
    # default IRQ body: jiffy clock + keyboard scan, CIA1 acknowledge (the
    # real handler's LDA $DC0D — without it the timer IRQ storms), restore+RTI
    put(IRQ_DEFAULT_BODY, bytes((
        0x20, 0xEA, 0xFF,  # JSR $FFEA (UDTIM)
        0x20, 0x9F, 0xFF,  # JSR $FF9F (SCNKEY)
        0xAD, 0x0D, 0xDC,  # LDA $DC0D (acknowledge CIA1)
        0x4C, 0x81, 0xEA,  # JMP $EA81
    )))
    put(IRQ_RESTORE_TAIL, bytes((0x68, 0xA8, 0x68, 0xAA, 0x68, 0x40)))  # PLA/TAY/PLA/TAX/PLA/RTI
    # LOAD preamble (games rely on X/Y landing in $C3/$C4 before the vector)
    put(LOAD_PREAMBLE, bytes((0x86, 0xC3, 0x84, 0xC4, 0x6C, 0x30, 0x03)))
    # jump table: vectored entries
    for table, vec in VECTORED_API.items():
        put(table, bytes((0x6C, vec & 0xFF, vec >> 8)))
    put(0xFFD5, bytes((0x4C, LOAD_PREAMBLE & 0xFF, LOAD_PREAMBLE >> 8)))  # JMP $F49E
    put(0xFFD8, bytes((0x4C, 0xED, 0xF5)))  # SAVE -> body trap at $F5ED
    # hardware vectors
    put(0xFFFA, bytes((
        NMI_ENTRY & 0xFF, NMI_ENTRY >> 8,
        RESET_BODY & 0xFF, RESET_BODY >> 8,
        IRQ_ENTRY & 0xFF, IRQ_ENTRY >> 8,
    )))
    return bytes(rom)


def build_shim_basic() -> bytes:
    # Programs that bank BASIC in and execute/read it need the real ROM;
    # the JAM filler makes that need fail loud instead of running garbage.
    return bytes([JAM]) * 0x2000


# ---- synthesized character ROM ------------------------------------------------------

_FONT = {
    0x00: "3C666E6E60623C00",  # @
    0x01: "183C667E66666600", 0x02: "7C66667C66667C00", 0x03: "3C66606060663C00",
    0x04: "786C6666666C7800", 0x05: "7E60607860607E00", 0x06: "7E60607860606000",
    0x07: "3C66606E66663C00", 0x08: "6666667E66666600", 0x09: "3C18181818183C00",
    0x0A: "1E0C0C0C0C6C3800", 0x0B: "666C7870786C6600", 0x0C: "6060606060607E00",
    0x0D: "63777F6B63636300", 0x0E: "66767E7E6E666600", 0x0F: "3C66666666663C00",
    0x10: "7C66667C60606000", 0x11: "3C666666663C0E00", 0x12: "7C66667C786C6600",
    0x13: "3C66603C06663C00", 0x14: "7E18181818181800", 0x15: "6666666666663C00",
    0x16: "66666666663C1800", 0x17: "6363636B7F776300", 0x18: "66663C183C666600",
    0x19: "6666663C18181800", 0x1A: "7E060C1830607E00",
    0x1B: "3C30303030303C00",  # [
    0x1C: "0E19307C3062FC00",  # pound
    0x1D: "3C0C0C0C0C0C3C00",  # ]
    0x1E: "183C7E1818181800",  # up arrow
    0x1F: "0010307F30100000",  # left arrow
    0x20: "0000000000000000",  # space
    0x21: "1818181800001800", 0x22: "6666660000000000", 0x23: "6666FF66FF666600",
    0x24: "183E603C067C1800", 0x25: "62660C1830664600", 0x26: "3C663C3867663F00",
    0x27: "060C180000000000", 0x28: "0C18303030180C00", 0x29: "30180C0C0C183000",
    0x2A: "00663CFF3C660000", 0x2B: "0018187E18180000", 0x2C: "0000000000181830",
    0x2D: "0000007E00000000", 0x2E: "0000000000181800", 0x2F: "0003060C18306000",
    0x30: "3C666E7666663C00", 0x31: "1818381818187E00", 0x32: "3C66060C30607E00",
    0x33: "3C66061C06663C00", 0x34: "060E1E667F060600", 0x35: "7E607C0606663C00",
    0x36: "3C66607C66663C00", 0x37: "7E660C1818181800", 0x38: "3C66663C66663C00",
    0x39: "3C66663E06663C00", 0x3A: "0000180000180000", 0x3B: "0000180000181830",
    0x3C: "0E18306030180E00", 0x3D: "00007E007E000000", 0x3E: "70180C060C187000",
    0x3F: "3C66060C18001800",
}


def build_shim_chargen() -> bytes:
    """A clean-room chargen: readable glyphs for screen codes 0-63,
    deterministic placeholder patterns for the graphics codes, and the
    standard inverse-video second half.  Both character sets (upper /
    lower halves of the 4K ROM) carry the same glyphs."""
    half = bytearray(0x800)
    for code in range(128):
        if code in _FONT:
            glyph = bytes.fromhex(_FONT[code])
        else:
            # placeholder graphics glyph: deterministic, code-distinct
            glyph = bytes(((code * 37 + row * 41) | 0x81) & 0xFF if row in (0, 7)
                          else (code * 29 + row * 53) & 0xFF for row in range(8))
        half[code * 8: code * 8 + 8] = glyph
    for code in range(128, 256):  # inverse video
        base = half[(code - 128) * 8: (code - 128) * 8 + 8]
        half[code * 8: code * 8 + 8] = bytes(b ^ 0xFF for b in base)
    return bytes(half) + bytes(half)


def load_real_roms(roms_dir: str | Path) -> dict[str, bytes]:
    """Pick up real ROM images if the user supplied them (never bundled).
    Recognized stems: kernal, basic, chargen (any extension)."""
    out: dict[str, bytes] = {}
    d = Path(roms_dir)
    if not d.is_dir():
        return out
    sizes = {"kernal": 0x2000, "basic": 0x2000, "chargen": 0x1000}
    for f in sorted(d.iterdir()):
        stem = f.stem.lower()
        for name, size in sizes.items():
            if name in stem and f.is_file():
                data = f.read_bytes()
                if len(data) == size:
                    out[name] = data
    return out


# ---- PETSCII keyboard decode (used by the SCNKEY trap) -------------------------------
# matrix code (col*8+row) -> (unshifted, shifted) PETSCII; None = modifier/ignored
_KEYDECODE: dict[int, tuple[int, int]] = {
    0: (0x14, 0x94), 1: (0x0D, 0x8D), 2: (0x1D, 0x9D), 3: (0x88, 0x8C),
    4: (0x85, 0x89), 5: (0x86, 0x8A), 6: (0x87, 0x8B), 7: (0x11, 0x91),
    8: (0x33, 0x23), 9: (0x57, 0xD7), 10: (0x41, 0xC1), 11: (0x34, 0x24),
    12: (0x5A, 0xDA), 13: (0x53, 0xD3), 14: (0x45, 0xC5),
    16: (0x35, 0x25), 17: (0x52, 0xD2), 18: (0x44, 0xC4), 19: (0x36, 0x26),
    20: (0x43, 0xC3), 21: (0x46, 0xC6), 22: (0x54, 0xD4), 23: (0x58, 0xD8),
    24: (0x37, 0x27), 25: (0x59, 0xD9), 26: (0x47, 0xC7), 27: (0x38, 0x28),
    28: (0x42, 0xC2), 29: (0x48, 0xC8), 30: (0x55, 0xD5), 31: (0x56, 0xD6),
    32: (0x39, 0x29), 33: (0x49, 0xC9), 34: (0x4A, 0xCA), 35: (0x30, 0x30),
    36: (0x4D, 0xCD), 37: (0x4B, 0xCB), 38: (0x4F, 0xCF), 39: (0x4E, 0xCE),
    40: (0x2B, 0xDB), 41: (0x50, 0xD0), 42: (0x4C, 0xCC), 43: (0x2D, 0xDD),
    44: (0x2E, 0x3E), 45: (0x3A, 0x5B), 46: (0x40, 0xBA), 47: (0x2C, 0x3C),
    48: (0x5C, 0xA9), 49: (0x2A, 0xC0), 50: (0x3B, 0x5D), 51: (0x13, 0x93),
    53: (0x3D, 0x3D), 54: (0x5E, 0xDE), 55: (0x2F, 0x3F),
    56: (0x31, 0x21), 57: (0x5F, 0x5F), 59: (0x32, 0x22),
    60: (0x20, 0xA0), 62: (0x51, 0xD1), 63: (0x03, 0x83),
}

# PETSCII -> screen code (for the CHROUT screen editor)
def petscii_to_screen(c: int) -> int | None:
    if 0x20 <= c <= 0x3F:
        return c
    if 0x40 <= c <= 0x5F:
        return c - 0x40
    if 0x60 <= c <= 0x7F:
        return c - 0x20
    if 0xA0 <= c <= 0xBF:
        return c - 0x40
    if 0xC0 <= c <= 0xFE:
        return c - 0x80
    if c == 0xFF:
        return 0x5E
    return None  # control code


_COLOR_CODES = {
    0x90: 0, 0x05: 1, 0x1C: 2, 0x9F: 3, 0x9C: 4, 0x1E: 5, 0x1F: 6, 0x9E: 7,
    0x81: 8, 0x95: 9, 0x96: 10, 0x97: 11, 0x98: 12, 0x99: 13, 0x9A: 14, 0x9B: 15,
}


class KernalHLE:
    """Implements the KERNAL service contracts against the machine state.

    Every trap sets the documented registers/flags/zero-page effects and then
    performs RTS semantics.  State that real KERNAL keeps in page 0/2/3
    (jiffy clock, cursor position, keyboard buffer, file name/LFS latches,
    status byte) is kept in the *real* RAM locations so programs that peek
    them see what they expect.
    """

    def __init__(self, machine) -> None:
        self.m = machine
        self._last_key_code = 64  # 64 = no key, as on real hardware
        self._open_files: dict[int, dict] = {}
        self._input_channel: dict | None = None

    # ---- install -------------------------------------------------------------
    def install(self, cpu) -> None:
        for addr, name in DIRECT_API.items():
            cpu.service_hooks[addr] = self._make(name)
        for addr, name in BODY_API.items():
            cpu.service_hooks[addr] = self._make(name)
        cpu.service_hooks[EXIT_TRAP] = self._exit_trap
        cpu.service_hooks[RESET_BODY] = self._reset_trap
        cpu.service_hooks[BRK_DEFAULT_BODY] = self._brk_trap

    def _make(self, name: str):
        fn = getattr(self, "svc_" + name.lower(), None)
        if fn is None:
            def missing(cpu, _name=name):
                raise KernalError(
                    f"KERNAL {_name} (${cpu.s.pc:04X}) reached but not implemented — "
                    "implement its documented contract in c64_re/kernal.py"
                )
            return missing

        def trap(cpu, _fn=fn):
            _fn(cpu)
        trap.__name__ = f"kernal_{name.lower()}"
        return trap

    # ---- helpers ---------------------------------------------------------------
    @staticmethod
    def _rts(cpu) -> None:
        lo = cpu.pull()
        hi = cpu.pull()
        cpu.s.pc = ((lo | (hi << 8)) + 1) & 0xFFFF

    def _ram(self):
        return self.m.mem.ram

    def _set_status(self, bits: int) -> None:
        self._ram()[0x90] |= bits

    # ---- boot/exit traps ----------------------------------------------------------
    def _exit_trap(self, cpu) -> None:
        raise ProgramExit(
            f"program returned to the boot frame (instr={cpu.instr_count})"
        )

    def _reset_trap(self, cpu) -> None:
        raise KernalError("CPU reset vector executed — unexpected in a booted program")

    def _brk_trap(self, cpu) -> None:
        raise KernalError(
            f"BRK reached the default vector at ${BRK_DEFAULT_BODY:04X} "
            f"(PC pushed near ${cpu.mem.rb(0x0100 | ((cpu.s.sp + 2) & 0xFF)) | (cpu.mem.rb(0x0100 | ((cpu.s.sp + 3) & 0xFF)) << 8):04X}) — "
            "either a crash or an unimplemented software-interrupt convention"
        )

    # ---- time -----------------------------------------------------------------------
    def svc_udtim(self, cpu) -> None:
        ram = self._ram()
        t = (ram[0xA0] << 16) | (ram[0xA1] << 8) | ram[0xA2]
        t = (t + 1) % 5184000  # 24h wrap, as on real hardware
        ram[0xA0], ram[0xA1], ram[0xA2] = (t >> 16) & 0xFF, (t >> 8) & 0xFF, t & 0xFF
        # STOP-key column latch ($91): $7F when the RUN/STOP row scan sees it
        ram[0x91] = 0x7F if "RUN/STOP" in self.m.pressed else 0xFF
        self._rts(cpu)

    def svc_rdtim(self, cpu) -> None:
        ram = self._ram()
        cpu.s.a, cpu.s.x, cpu.s.y = ram[0xA2], ram[0xA1], ram[0xA0]
        self._rts(cpu)

    def svc_settim(self, cpu) -> None:
        ram = self._ram()
        ram[0xA2], ram[0xA1], ram[0xA0] = cpu.s.a, cpu.s.x, cpu.s.y
        self._rts(cpu)

    def svc_stop(self, cpu) -> None:
        # Z=1 (and A=$7F pattern) when RUN/STOP is down
        if "RUN/STOP" in self.m.pressed:
            cpu.s.a = 0x7F
            cpu.s.z = 1
        else:
            cpu.s.a = 0xFF
            cpu.s.z = 0
        cpu.s.c = 0
        self._rts(cpu)

    # ---- keyboard ------------------------------------------------------------------
    def svc_scnkey(self, cpu) -> None:
        ram = self._ram()
        shift = ("LSHIFT" in self.m.pressed) or ("RSHIFT" in self.m.pressed)
        ram[0x28D] = 1 if shift else 0
        code = 64
        for name in self.m.pressed:
            if name in ("LSHIFT", "RSHIFT", "CTRL", "CBM"):
                continue
            from .cia import MATRIX
            if name in MATRIX:
                col, row = MATRIX[name]
                code = col * 8 + row
                break
        ram[0xCB] = code
        if code != 64 and code != self._last_key_code:
            decode = _KEYDECODE.get(code)
            if decode is not None:
                pet = decode[1] if shift else decode[0]
                n = ram[0xC6]
                maxlen = ram[0x289] or 10
                if n < maxlen:
                    ram[0x277 + n] = pet
                    ram[0xC6] = n + 1
        self._last_key_code = code
        self._rts(cpu)

    def svc_getin(self, cpu) -> None:
        if self._input_channel is not None:
            self._chrin_from_channel(cpu)
            return
        ram = self._ram()
        n = ram[0xC6]
        if n == 0:
            cpu.s.a = 0
        else:
            cpu.s.a = ram[0x277]
            for i in range(1, n):
                ram[0x277 + i - 1] = ram[0x277 + i]
            ram[0xC6] = n - 1
        cpu._nz(cpu.s.a)
        cpu.s.c = 0
        self._rts(cpu)

    # ---- screen editor (CHROUT subset) --------------------------------------------
    def svc_chrout(self, cpu) -> None:
        if getattr(self.m, "output_channel", 0) not in (0, 3):
            raise KernalError(
                f"CHROUT to device channel {self.m.output_channel} not implemented"
            )
        self._screen_put(cpu.s.a)
        cpu.s.c = 0
        self._rts(cpu)

    def _screen_put(self, c: int) -> None:
        ram = self._ram()
        base = ram[0x288] << 8
        row, col = ram[0xD6], ram[0xD3]
        color = ram[0x286] & 0x0F

        def sync(r, cl):
            ram[0xD6], ram[0xD3] = r, cl
            ram[0xD1], ram[0xD2] = (base + r * 40) & 0xFF, ((base + r * 40) >> 8) & 0xFF

        if c == 0x0D or c == 0x8D:
            row += 1
            col = 0
            ram[0xC7] = 0  # reverse off at CR
        elif c == 0x93:  # clear
            ram[base + 0:base + 1000] = b"\x20" * 1000  # noqa: E203
            for i in range(1000):
                self.m.mem.color_ram[i] = color
            row = col = 0
        elif c == 0x13:  # home
            row = col = 0
        elif c == 0x11:
            row += 1
        elif c == 0x91:
            row = max(0, row - 1)
        elif c == 0x1D:
            col += 1
            if col > 39:
                col = 0
                row += 1
        elif c == 0x9D:
            col = max(0, col - 1)
        elif c == 0x12:
            ram[0xC7] = 0x12
        elif c == 0x92:
            ram[0xC7] = 0
        elif c == 0x14:  # delete
            col = max(0, col - 1)
            ram[base + row * 40 + col] = 0x20
        elif c in _COLOR_CODES:
            ram[0x286] = _COLOR_CODES[c]
        else:
            sc = petscii_to_screen(c)
            if sc is None:
                return  # unhandled control code: inert, as most are here
            if ram[0xC7]:
                sc |= 0x80
            ram[base + row * 40 + col] = sc
            self.m.mem.color_ram[row * 40 + col] = color
            col += 1
            if col > 39:
                col = 0
                row += 1
        if row > 24:  # scroll
            ram[base:base + 960] = ram[base + 40: base + 1000]
            ram[base + 960: base + 1000] = b"\x20" * 40
            cr = self.m.mem.color_ram
            cr[0:960] = cr[40:1000]
            for i in range(960, 1000):
                cr[i] = color
            row = 24
        sync(row, col)

    def svc_clrscr(self, cpu) -> None:
        # KERNAL internal $E544 (games JSR it directly): clear screen + home
        self._screen_put(0x93)
        self._rts(cpu)

    def svc_homecrs(self, cpu) -> None:
        # KERNAL internal $E566: cursor home
        self._screen_put(0x13)
        self._rts(cpu)

    def svc_plot(self, cpu) -> None:
        ram = self._ram()
        if cpu.s.c:  # read
            cpu.s.x, cpu.s.y = ram[0xD6], ram[0xD3]
        else:
            ram[0xD6], ram[0xD3] = cpu.s.x, cpu.s.y
        self._rts(cpu)

    def svc_screen(self, cpu) -> None:
        cpu.s.x, cpu.s.y = 40, 25
        self._rts(cpu)

    def svc_iobase(self, cpu) -> None:
        cpu.s.x, cpu.s.y = 0x00, 0xDC
        self._rts(cpu)

    # ---- init group (idempotent; power-on already did the work) ----------------------
    def svc_cint(self, cpu) -> None:
        self.m.power_on_screen()
        self._rts(cpu)

    def svc_ioinit(self, cpu) -> None:
        self.m.cia_power_on()
        self._rts(cpu)

    def svc_vicdefaults(self, cpu) -> None:
        # KERNAL internal $E5A0: VIC register defaults + editor screen base
        self.m.vic_power_on()
        self._rts(cpu)

    def svc_ramtas(self, cpu) -> None:
        self._rts(cpu)

    def svc_restor(self, cpu) -> None:
        self.m.install_default_vectors()
        self._rts(cpu)

    def svc_vector(self, cpu) -> None:
        ram = self._ram()
        addr = cpu.s.x | (cpu.s.y << 8)
        if cpu.s.c:  # read vectors out
            for i in range(0x20):
                ram[(addr + i) & 0xFFFF] = ram[0x314 + i]
        else:
            for i in range(0x20):
                ram[0x314 + i] = ram[(addr + i) & 0xFFFF]
        self._rts(cpu)

    def svc_setmsg(self, cpu) -> None:
        self._ram()[0x9D] = cpu.s.a
        self._rts(cpu)

    def svc_settmo(self, cpu) -> None:
        self._ram()[0x285] = cpu.s.a
        self._rts(cpu)

    def svc_readst(self, cpu) -> None:
        cpu.s.a = self._ram()[0x90]
        cpu._nz(cpu.s.a)
        self._rts(cpu)

    def svc_memtop(self, cpu) -> None:
        ram = self._ram()
        if cpu.s.c:
            cpu.s.x, cpu.s.y = ram[0x283], ram[0x284]
        else:
            ram[0x283], ram[0x284] = cpu.s.x, cpu.s.y
        self._rts(cpu)

    def svc_membot(self, cpu) -> None:
        ram = self._ram()
        if cpu.s.c:
            cpu.s.x, cpu.s.y = ram[0x281], ram[0x282]
        else:
            ram[0x281], ram[0x282] = cpu.s.x, cpu.s.y
        self._rts(cpu)

    # ---- file name / logical file latches ----------------------------------------------
    def svc_setlfs(self, cpu) -> None:
        ram = self._ram()
        ram[0xB8], ram[0xBA], ram[0xB9] = cpu.s.a, cpu.s.x, cpu.s.y
        self._rts(cpu)

    def svc_setnam(self, cpu) -> None:
        ram = self._ram()
        ram[0xB7], ram[0xBB], ram[0xBC] = cpu.s.a, cpu.s.x, cpu.s.y
        self._rts(cpu)

    def _current_name(self) -> bytes:
        ram = self._ram()
        ptr = ram[0xBB] | (ram[0xBC] << 8)
        return bytes(self.m.mem.ram[ptr:ptr + ram[0xB7]])

    # ---- LOAD (drive HLE against the attached D64) -------------------------------------
    def svc_load(self, cpu) -> None:
        ram = self._ram()
        device = ram[0xBA]
        secondary = ram[0xB9]
        verify = cpu.s.a != 0
        name = self._current_name()
        ram[0x90] = 0
        if verify:
            raise KernalError("LOAD with verify flag (A!=0) not implemented")
        if device in (1, 0):
            raise KernalError(f"LOAD from device {device} (tape/keyboard) not modeled")
        if self.m.drive is None:
            raise KernalError(f"LOAD {name!r} from device {device} but no disk attached")
        try:
            prg = self.m.drive.read_file(name if name else b"*")
        except FileNotFoundError:
            cpu.s.c = 1
            cpu.s.a = 4  # FILE NOT FOUND
            self._set_status(0x02)
            self._rts(cpu)
            return
        if secondary == 0:
            dest = ram[0xC3] | (ram[0xC4] << 8)
        else:
            dest = prg_load_address(prg)
        payload = prg_payload(prg)
        end = dest + len(payload)
        if end > 0x10000:
            raise KernalError(
                f"LOAD {name!r} to ${dest:04X}+{len(payload)} overruns 64K"
            )
        self.m.mem.ram[dest:end] = payload  # LOAD writes RAM (under ROM if banked)
        ram[0xAE], ram[0xAF] = end & 0xFF, (end >> 8) & 0xFF
        ram[0x90] = 0x40  # EOF
        cpu.s.x, cpu.s.y = end & 0xFF, (end >> 8) & 0xFF
        cpu.s.c = 0
        cpu.s.a = 0
        self.m.log_event(f"LOAD {name!r} -> ${dest:04X}-${end - 1:04X}")
        self._rts(cpu)

    def svc_save(self, cpu) -> None:
        raise KernalError(
            "SAVE not implemented (writable disk images are a future evidence tier)"
        )

    # ---- OPEN/CHKIN/CHRIN channel reading (device 8 SEQ/PRG streams) --------------------
    def svc_open(self, cpu) -> None:
        ram = self._ram()
        lfn, device, secondary = ram[0xB8], ram[0xBA], ram[0xB9]
        name = self._current_name()
        if device != 8:
            raise KernalError(f"OPEN on device {device} not modeled (only disk 8)")
        if self.m.drive is None:
            raise KernalError(f"OPEN {name!r} but no disk attached")
        if secondary == 15:
            # command channel: accept and discard commands, report 00,OK
            self._open_files[lfn] = {"data": b"00, OK,00,00\r", "pos": 0, "cmd": True}
        else:
            data = self.m.drive.read_file(name if name else b"*")
            self._open_files[lfn] = {"data": data, "pos": 0, "cmd": False}
        ram[0x90] = 0
        cpu.s.c = 0
        self._rts(cpu)

    def svc_close(self, cpu) -> None:
        self._open_files.pop(cpu.s.a, None)
        cpu.s.c = 0
        self._rts(cpu)

    def svc_chkin(self, cpu) -> None:
        f = self._open_files.get(cpu.s.x)
        if f is None:
            raise KernalError(f"CHKIN on unopened logical file {cpu.s.x}")
        self._input_channel = f
        cpu.s.c = 0
        self._rts(cpu)

    def svc_chkout(self, cpu) -> None:
        raise KernalError("CHKOUT (output channels) not implemented")

    def svc_clrchn(self, cpu) -> None:
        self._input_channel = None
        cpu.s.c = 0
        self._rts(cpu)

    def svc_clall(self, cpu) -> None:
        self._open_files.clear()
        self._input_channel = None
        cpu.s.c = 0
        self._rts(cpu)

    def svc_chrin(self, cpu) -> None:
        if self._input_channel is None:
            raise KernalError("CHRIN from keyboard (line editor) not implemented")
        self._chrin_from_channel(cpu)

    def _chrin_from_channel(self, cpu) -> None:
        f = self._input_channel
        data, pos = f["data"], f["pos"]
        if pos >= len(data):
            cpu.s.a = 0x0D
            self._set_status(0x40)
        else:
            cpu.s.a = data[pos]
            f["pos"] = pos + 1
            if f["pos"] >= len(data):
                self._set_status(0x40)
        cpu._nz(cpu.s.a)
        cpu.s.c = 0
        self._rts(cpu)

    # ---- raw IEC serial: out of scope until a real fastloader forces the issue ---------
    def _iec(self, cpu, what: str) -> None:
        raise KernalError(
            f"raw IEC serial ({what}) reached — this program drives the drive "
            "directly; needs a 1541 model or a game-specific loader HLE"
        )

    def svc_second(self, cpu) -> None:
        self._iec(cpu, "SECOND")

    def svc_tksa(self, cpu) -> None:
        self._iec(cpu, "TKSA")

    def svc_acptr(self, cpu) -> None:
        self._iec(cpu, "ACPTR")

    def svc_ciout(self, cpu) -> None:
        self._iec(cpu, "CIOUT")

    def svc_untlk(self, cpu) -> None:
        self._iec(cpu, "UNTLK")

    def svc_unlsn(self, cpu) -> None:
        self._iec(cpu, "UNLSN")

    def svc_listen(self, cpu) -> None:
        self._iec(cpu, "LISTEN")

    def svc_talk(self, cpu) -> None:
        self._iec(cpu, "TALK")
