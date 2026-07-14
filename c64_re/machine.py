"""C64Machine: the chip set glued together behind the CPU's I/O window.

Owns the VIC-II, both CIAs, the SID register model, the keyboard matrix /
joystick state, the attached disk image, and the KERNAL HLE.  Routes
$D000-$DFFF reads/writes, advances chip time from CPU cycles, and derives
the IRQ/NMI lines.

Deterministic by construction: nothing here reads wall time; all time is
CPU-cycle telemetry.
"""
from __future__ import annotations

from .cia import CIA, MATRIX
from .kernal import DEFAULT_VECTORS, KernalHLE
from .sid import SID
from .vic import VIC


class OpenIOAccess(RuntimeError):
    """The program touched unmapped I/O ($DE00-$DFFF expansion area)."""


class C64Machine:
    def __init__(self, mem, *, pal: bool = True) -> None:
        self.mem = mem
        mem.io_read = self.io_read
        mem.io_write = self.io_write
        self.cia1 = CIA("CIA1")
        self.cia2 = CIA("CIA2")
        self.sid = SID()
        self.vic = VIC(self._vic_fetch, mem.color_ram)
        self.cia1.port_a_ext = self._kbd_pull_a
        self.cia1.port_b_ext = self._kbd_pull_b
        # ordered so "first pressed key" is deterministic for SCNKEY
        self.pressed: list[str] = []
        self.joy1 = 0  # active-high masks (JOY_* bits); inverted onto the port
        self.joy2 = 0
        self.drive = None            # DiskImage, attached by the runtime
        self.output_channel = 0
        self.kernal = KernalHLE(self)
        self.cpu = None              # wired by the runtime (for NMI edge delivery)
        self._nmi_level = False
        self.allow_open_io = False   # $DE00-$DFFF: fail loud unless opted in
        self.events: list[str] = []  # coarse machine event log (LOADs etc.)

    # ---- logging ---------------------------------------------------------------
    def log_event(self, text: str) -> None:
        self.events.append(text)

    # ---- VIC bus ----------------------------------------------------------------
    def vic_bank(self) -> int:
        return (~self.cia2.pins_a()) & 0x03

    def _vic_fetch(self, addr14: int) -> int:
        return self.mem.vic_read(addr14, self.vic_bank())

    # ---- keyboard / joystick resolution (CIA1 external pulls) ---------------------
    def _matrix_pulls(self, out_a: int, out_b: int) -> tuple[int, int]:
        pull_a = 0
        pull_b = 0
        for name in self.pressed:
            m = MATRIX.get(name)
            if m is None:
                continue
            col, row = m
            if not (out_a >> col) & 1:   # column driven low -> pulls row low
                pull_b |= 1 << row
            if not (out_b >> row) & 1:   # reverse scan
                pull_a |= 1 << col
        return pull_a, pull_b

    def _kbd_pull_a(self, out_a: int, out_b: int) -> int:
        pull_a, _ = self._matrix_pulls(out_a, out_b)
        return pull_a | (self.joy2 & 0x1F)

    def _kbd_pull_b(self, out_a: int, out_b: int) -> int:
        _, pull_b = self._matrix_pulls(out_a, out_b)
        return pull_b | (self.joy1 & 0x1F)

    # ---- input API (used by frontends and, later, input demos) ---------------------
    def key_down(self, name: str) -> None:
        if name not in MATRIX:
            raise KeyError(f"unknown C64 key {name!r}")
        if name not in self.pressed:
            self.pressed.append(name)

    def key_up(self, name: str) -> None:
        if name in self.pressed:
            self.pressed.remove(name)

    def set_joy1(self, mask: int) -> None:
        self.joy1 = mask & 0x1F

    def set_joy2(self, mask: int) -> None:
        self.joy2 = mask & 0x1F

    def press_restore(self) -> None:
        if self.cpu is not None:
            self.cpu.nmi_pending = True

    # ---- I/O window ------------------------------------------------------------------
    def io_read(self, addr: int) -> int:
        if addr < 0xD400:
            return self.vic.read(addr & 0x3F)
        if addr < 0xD800:
            return self.sid.read(addr & 0x1F)
        if addr < 0xDC00:
            # color RAM: low nibble is real; high nibble reads 0 here
            # (real hardware: open bus) — deterministic choice, documented.
            return self.mem.color_ram[addr - 0xD800] & 0x0F
        if addr < 0xDD00:
            return self.cia1.read(addr & 0x0F)
        if addr < 0xDE00:
            return self.cia2.read(addr & 0x0F)
        if self.allow_open_io:
            return 0xFF
        raise OpenIOAccess(
            f"read from unmapped I/O ${addr:04X} (expansion port area); "
            "set machine.allow_open_io=True only with an observed reason"
        )

    def io_write(self, addr: int, val: int) -> None:
        if addr < 0xD400:
            self.vic.write(addr & 0x3F, val)
            return
        if addr < 0xD800:
            cyc = self.cpu.cycle_count if self.cpu is not None else 0
            self.sid.write(addr & 0x1F, val, cyc)
            return
        if addr < 0xDC00:
            self.mem.color_ram[addr - 0xD800] = val & 0x0F
            return
        if addr < 0xDD00:
            self.cia1.write(addr & 0x0F, val)
            return
        if addr < 0xDE00:
            self.cia2.write(addr & 0x0F, val)
            return
        if self.allow_open_io:
            return
        raise OpenIOAccess(
            f"write ${val:02X} to unmapped I/O ${addr:04X} (expansion port area); "
            "set machine.allow_open_io=True only with an observed reason"
        )

    # ---- time -----------------------------------------------------------------------------
    def tick(self, cycles: int) -> None:
        self.vic.tick(cycles)
        self.cia1.tick(cycles)
        self.cia2.tick(cycles)
        nmi = self.cia2.irq_asserted()
        if nmi and not self._nmi_level and self.cpu is not None:
            self.cpu.nmi_pending = True  # edge-triggered
        self._nmi_level = nmi

    def irq_line(self) -> bool:
        return self.vic.irq_asserted() or self.cia1.irq_asserted()

    # ---- power-on environment -----------------------------------------------------------
    def install_default_vectors(self) -> None:
        ram = self.mem.ram
        for addr, target in DEFAULT_VECTORS.items():
            ram[addr] = target & 0xFF
            ram[addr + 1] = (target >> 8) & 0xFF

    def vic_power_on(self) -> None:
        """VIC register + editor-base defaults (KERNAL $E5A0's effect)."""
        ram = self.mem.ram
        ram[0x288] = 0x04  # screen page
        self.vic.write(0x11, 0x1B)
        self.vic.write(0x16, 0x08)
        self.vic.write(0x18, 0x14)
        self.vic.write(0x20, 14)  # light blue border
        self.vic.write(0x21, 6)   # blue background

    def cia_power_on(self) -> None:
        """CIA defaults (KERNAL IOINIT's effect): keyboard port directions,
        VIC bank 0 with IEC lines released, and the 1/60s jiffy timer."""
        self.cia1.ta_latch = self.cia1.ta = 0x4025  # PAL jiffy interval
        self.cia1.cra = 0x01
        self.cia1.icr_mask = 0x01
        self.cia1.ddra = 0xFF  # keyboard: columns out
        self.cia1.ddrb = 0x00  # rows in
        self.cia2.ddra = 0x3F  # VIC bank + IEC lines out
        self.cia2.pra = 0x03 | 0x04 | 0x30  # bank 0, IEC lines released

    def power_on_screen(self) -> None:
        ram = self.mem.ram
        ram[0x0400:0x07E8] = b"\x20" * 1000
        for i in range(1000):
            self.mem.color_ram[i] = 14
        ram[0x286] = 14   # current text color: light blue
        ram[0xD3] = ram[0xD6] = 0  # cursor col/row
        self.vic_power_on()

    def power_on(self) -> None:
        """The state a program loaded from BASIC's READY prompt actually sees.

        This is c64_re's analogue of dos_re's _init_bios_environment: not
        program-specific — the environment any LOAD"..",8,1 : RUN program
        expects.  Kept minimal and honest; extend only from observed needs.
        """
        ram = self.mem.ram
        self.install_default_vectors()
        self.power_on_screen()
        ram[0xC6] = 0      # keyboard buffer empty
        ram[0x289] = 10    # keyboard buffer size
        ram[0x91] = 0xFF   # STOP key not pressed
        ram[0x9D] = 0x80   # direct mode message flag (as after READY. + RUN)
        ram[0xA0] = ram[0xA1] = ram[0xA2] = 0  # jiffy clock
        ram[0x90] = 0      # I/O status
        ram[0xBA] = 8      # last-used device: disk
        ram[0x281], ram[0x282] = 0x00, 0x08  # MEMBOT $0800
        ram[0x283], ram[0x284] = 0x00, 0xA0  # MEMTOP $A000
        ram[0x2B], ram[0x2C] = 0x01, 0x08    # BASIC text start $0801
        ram[0x37], ram[0x38] = 0x00, 0xA0    # BASIC memory top
        self.cia_power_on()
