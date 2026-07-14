"""C64 memory: 64 KB RAM, PLA banking, 6510 port, ROMs, and the VIC's view.

The CPU sees memory through :meth:`Memory.rb` / :meth:`Memory.wb`, which
apply the PLA banking decided by the 6510 on-chip port ($0000 DDR / $0001
data): BASIC ROM at $A000, KERNAL ROM at $E000, and the $D000 window
(I/O registers, character ROM, or RAM).  Writes under ROM always land in
the RAM below — that is the hardware behaviour games rely on.

The VIC-II sees a different, un-banked view: 16 KB windows of RAM selected
by CIA2 port A, with the character ROM shadowed at $1000-$1FFF in banks 0
and 2 (:meth:`Memory.vic_read`).

No cartridge lines are modeled (GAME/EXROM high).  Add them when a real
cartridge program needs them, with its observed contract.
"""
from __future__ import annotations

from typing import Callable


class Memory:
    """Banked 64 KB address space + color RAM + ROM images.

    ``io_read`` / ``io_write`` are wired by the machine (see
    :class:`c64_re.machine.C64Machine`) and receive absolute addresses in
    $D000-$DFFF whenever the I/O window is banked in.
    """

    def __init__(self, *, basic_rom: bytes, kernal_rom: bytes, char_rom: bytes) -> None:
        if len(basic_rom) != 0x2000:
            raise ValueError(f"BASIC ROM must be 8192 bytes, got {len(basic_rom)}")
        if len(kernal_rom) != 0x2000:
            raise ValueError(f"KERNAL ROM must be 8192 bytes, got {len(kernal_rom)}")
        if len(char_rom) != 0x1000:
            raise ValueError(f"character ROM must be 4096 bytes, got {len(char_rom)}")
        self.ram = bytearray(0x10000)
        self.color_ram = bytearray(0x400)  # low nibbles only
        self.basic_rom = bytes(basic_rom)
        self.kernal_rom = bytes(kernal_rom)
        self.char_rom = bytes(char_rom)
        # 6510 on-chip port.  RAM below $00/$01 exists but is unreachable from
        # the CPU; the port registers live here, not in self.ram.
        self.cpu_port_ddr = 0x2F
        self.cpu_port_data = 0x37
        self.io_read: Callable[[int], int] | None = None
        self.io_write: Callable[[int, int], None] | None = None

    # ---- 6510 port / PLA ----------------------------------------------------
    def _port_lines(self) -> int:
        """Effective port pin values: driven bits from the data register,
        input bits pulled high (LORAM/HIRAM/CHAREN have pull-ups; the cassette
        sense line reads 1 = no button pressed)."""
        ddr = self.cpu_port_ddr
        return (self.cpu_port_data & ddr) | (0xFF & ~ddr)

    def banking(self) -> tuple[bool, bool, bool]:
        """(loram, hiram, charen) from the effective 6510 port lines."""
        lines = self._port_lines()
        return bool(lines & 1), bool(lines & 2), bool(lines & 4)

    # ---- CPU view -------------------------------------------------------------
    def rb(self, addr: int) -> int:
        addr &= 0xFFFF
        if addr >= 0xA000:
            loram, hiram, charen = self.banking()
            if addr >= 0xE000:
                if hiram:
                    return self.kernal_rom[addr - 0xE000]
                return self.ram[addr]
            if addr >= 0xD000:
                if not loram and not hiram:
                    return self.ram[addr]
                if charen:
                    if self.io_read is None:
                        raise RuntimeError(f"I/O read at ${addr:04X} with no machine attached")
                    return self.io_read(addr)
                return self.char_rom[addr - 0xD000]
            if addr < 0xC000:  # $A000-$BFFF
                if loram and hiram:
                    return self.basic_rom[addr - 0xA000]
                return self.ram[addr]
            return self.ram[addr]  # $C000-$CFFF is always RAM
        if addr >= 2:
            return self.ram[addr]
        if addr == 0:
            return self.cpu_port_ddr
        # Reading $01: output bits read back the register, input bits read the
        # pins — both of which _port_lines() already models.
        return self._port_lines()

    def wb(self, addr: int, val: int) -> None:
        addr &= 0xFFFF
        val &= 0xFF
        if addr >= 2:
            if 0xD000 <= addr <= 0xDFFF:
                loram, hiram, charen = self.banking()
                if (loram or hiram) and charen:
                    if self.io_write is None:
                        raise RuntimeError(f"I/O write at ${addr:04X} with no machine attached")
                    self.io_write(addr, val)
                    return
            self.ram[addr] = val  # writes under ROM land in RAM
            return
        if addr == 0:
            self.cpu_port_ddr = val
        else:
            self.cpu_port_data = val

    # ---- bulk helpers (RAM only, banking-free — for loaders and tests) ----
    def load_block(self, addr: int, data: bytes) -> None:
        end = addr + len(data)
        if end > 0x10000:
            raise ValueError(f"block ${addr:04X}+{len(data)} exceeds 64K")
        self.ram[addr:end] = data

    def block(self, addr: int, length: int) -> bytes:
        return bytes(self.ram[addr:addr + length])

    # ---- VIC view ----------------------------------------------------------------
    def vic_read(self, addr14: int, bank: int) -> int:
        """VIC bus fetch: 14-bit address within the CIA2-selected 16K bank.
        Character ROM is shadowed at $1000-$1FFF in banks 0 and 2."""
        addr14 &= 0x3FFF
        if (bank & 1) == 0 and 0x1000 <= addr14 < 0x2000:
            return self.char_rom[addr14 - 0x1000]
        return self.ram[(bank << 14) | addr14]
