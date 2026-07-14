"""SID 6581 register model — the observer seat, not a synthesizer.

Mirrors dos_re's AdLib approach: the VM records the register file the game
writes (the *command stream* is the recovery evidence; audio rendering is a
frontend concern layered on later).  The two readable registers games use
as entropy/envelope sources are modeled deterministically:

- OSC3 ($D41B): games commonly set voice 3 to noise and read this as a
  random-number source.  Real hardware value depends on the analog clock;
  here it is a deterministic 23-bit LFSR (the SID noise polynomial) advanced
  once per read.  Deterministic across replays; snapshot-carried.
- ENV3 ($D41C): returns 0 (no envelope model).  A game observed *depending*
  on ENV3 values gets a modeled contract then, not a guess.

Write-only registers read back 0 (real SID has decaying open-bus here;
0 is the deterministic choice — document per game if one ever cares).
"""
from __future__ import annotations


class SID:
    def __init__(self) -> None:
        self.regs = bytearray(0x20)
        self.noise_lfsr = 0x7FFFF8
        self.write_log: list[tuple[int, int, int]] | None = None  # (cycle, reg, val)

    def read(self, reg: int) -> int:
        reg &= 0x1F
        if reg in (0x19, 0x1A):
            return 0xFF  # POT X/Y: no paddles connected
        if reg == 0x1B:  # OSC3 — deterministic noise LFSR
            lfsr = self.noise_lfsr
            bit = ((lfsr >> 22) ^ (lfsr >> 17)) & 1
            self.noise_lfsr = ((lfsr << 1) | bit) & 0x7FFFFF
            l = self.noise_lfsr
            return (
                ((l >> 15) & 1) << 7 | ((l >> 12) & 1) << 6 | ((l >> 9) & 1) << 5
                | ((l >> 5) & 1) << 4 | ((l >> 2) & 1) << 3 | ((l >> 16) & 1) << 2
                | ((l >> 8) & 1) << 1 | (l & 1)
            )
        if reg == 0x1C:
            return 0  # ENV3
        return 0

    def write(self, reg: int, val: int, cycle: int = 0) -> None:
        reg &= 0x1F
        self.regs[reg] = val & 0xFF
        if self.write_log is not None:
            self.write_log.append((cycle, reg, val & 0xFF))

    def get_state(self) -> dict:
        return {"regs": bytes(self.regs), "noise_lfsr": self.noise_lfsr}

    def set_state(self, state: dict) -> None:
        self.regs[:] = state["regs"]
        self.noise_lfsr = state["noise_lfsr"]
