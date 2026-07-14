"""CIA 6526 model (timers, interrupts, ports) + the C64 keyboard matrix.

Deterministic and instruction-clocked: timers advance from the CPU's cycle
telemetry via :meth:`CIA.tick`, never from wall time.  Modeled because real
games exercise it: timer A/B one-shot and continuous modes, timer B counting
timer A underflows (the standard long-period music/IRQ chain), the ICR
mask/status protocol, and the port A/B keyboard+joystick resolution.

Not modeled (fail-loud or documented-inert until a real program needs it):
serial shift register (reads 0), RS-232, TOD alarm (TOD registers hold a
frozen 00:00:00.0 — games that *wait* on TOD would hang loudly rather than
run wrong; none encountered yet).
"""
from __future__ import annotations

from typing import Callable

# Keyboard matrix: name -> (column bit driven on port A, row bit read on port B).
# This is the physical keyboard wiring — hardware truth, not game knowledge.
MATRIX: dict[str, tuple[int, int]] = {
    "INS/DEL": (0, 0), "RETURN": (0, 1), "CRSR_LR": (0, 2), "F7": (0, 3),
    "F1": (0, 4), "F3": (0, 5), "F5": (0, 6), "CRSR_UD": (0, 7),
    "3": (1, 0), "W": (1, 1), "A": (1, 2), "4": (1, 3),
    "Z": (1, 4), "S": (1, 5), "E": (1, 6), "LSHIFT": (1, 7),
    "5": (2, 0), "R": (2, 1), "D": (2, 2), "6": (2, 3),
    "C": (2, 4), "F": (2, 5), "T": (2, 6), "X": (2, 7),
    "7": (3, 0), "Y": (3, 1), "G": (3, 2), "8": (3, 3),
    "B": (3, 4), "H": (3, 5), "U": (3, 6), "V": (3, 7),
    "9": (4, 0), "I": (4, 1), "J": (4, 2), "0": (4, 3),
    "M": (4, 4), "K": (4, 5), "O": (4, 6), "N": (4, 7),
    "+": (5, 0), "P": (5, 1), "L": (5, 2), "-": (5, 3),
    ".": (5, 4), ":": (5, 5), "@": (5, 6), ",": (5, 7),
    "POUND": (6, 0), "*": (6, 1), ";": (6, 2), "HOME": (6, 3),
    "RSHIFT": (6, 4), "=": (6, 5), "ARROW_UP": (6, 6), "/": (6, 7),
    "1": (7, 0), "ARROW_LEFT": (7, 1), "CTRL": (7, 2), "2": (7, 3),
    "SPACE": (7, 4), "CBM": (7, 5), "Q": (7, 6), "RUN/STOP": (7, 7),
}

# Joystick bit positions (active low on the CIA port): up/down/left/right/fire.
JOY_UP, JOY_DOWN, JOY_LEFT, JOY_RIGHT, JOY_FIRE = 1, 2, 4, 8, 16


class CIA:
    """One 6526.  ``port_a_ext`` / ``port_b_ext`` are callables returning the
    externally-pulled-low bits given the current output pin levels — the
    machine wires the keyboard matrix and joysticks through them."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.pra = 0
        self.prb = 0
        self.ddra = 0
        self.ddrb = 0
        self.ta_latch = 0xFFFF
        self.tb_latch = 0xFFFF
        self.ta = 0xFFFF
        self.tb = 0xFFFF
        self.cra = 0
        self.crb = 0
        self.icr_mask = 0
        self.icr_status = 0
        self.tod_latched = False
        # externally-driven input resolution (keyboard/joystick), set by machine
        self.port_a_ext: Callable[[int, int], int] | None = None
        self.port_b_ext: Callable[[int, int], int] | None = None

    # ---- pins -----------------------------------------------------------------
    def _out_a(self) -> int:
        return (self.pra & self.ddra) | (0xFF & ~self.ddra)

    def _out_b(self) -> int:
        return (self.prb & self.ddrb) | (0xFF & ~self.ddrb)

    def pins_a(self) -> int:
        v = self._out_a()
        if self.port_a_ext is not None:
            v &= ~self.port_a_ext(self._out_a(), self._out_b()) & 0xFF
        return v

    def pins_b(self) -> int:
        v = self._out_b()
        if self.port_b_ext is not None:
            v &= ~self.port_b_ext(self._out_a(), self._out_b()) & 0xFF
        return v

    # ---- register file -----------------------------------------------------------
    def read(self, reg: int) -> int:
        reg &= 0x0F
        if reg == 0x0:
            return self.pins_a()
        if reg == 0x1:
            return self.pins_b()
        if reg == 0x2:
            return self.ddra
        if reg == 0x3:
            return self.ddrb
        if reg == 0x4:
            return self.ta & 0xFF
        if reg == 0x5:
            return (self.ta >> 8) & 0xFF
        if reg == 0x6:
            return self.tb & 0xFF
        if reg == 0x7:
            return (self.tb >> 8) & 0xFF
        if reg in (0x8, 0x9, 0xA, 0xB):
            return 0  # TOD frozen at 00:00:00.0 (see module docstring)
        if reg == 0xC:
            return 0  # serial shift register not modeled
        if reg == 0xD:
            # reading ICR returns status (bit7 = any enabled source pending)
            # and clears it
            v = self.icr_status
            if v & self.icr_mask & 0x1F:
                v |= 0x80
            self.icr_status = 0
            return v
        if reg == 0xE:
            return self.cra
        return self.crb

    def write(self, reg: int, val: int) -> None:
        reg &= 0x0F
        val &= 0xFF
        if reg == 0x0:
            self.pra = val
        elif reg == 0x1:
            self.prb = val
        elif reg == 0x2:
            self.ddra = val
        elif reg == 0x3:
            self.ddrb = val
        elif reg == 0x4:
            self.ta_latch = (self.ta_latch & 0xFF00) | val
        elif reg == 0x5:
            self.ta_latch = (self.ta_latch & 0x00FF) | (val << 8)
            if not (self.cra & 0x01):  # writing high byte while stopped loads counter
                self.ta = self.ta_latch
        elif reg == 0x6:
            self.tb_latch = (self.tb_latch & 0xFF00) | val
        elif reg == 0x7:
            self.tb_latch = (self.tb_latch & 0x00FF) | (val << 8)
            if not (self.crb & 0x01):
                self.tb = self.tb_latch
        elif reg in (0x8, 0x9, 0xA, 0xB):
            pass  # TOD writes accepted and ignored (frozen clock)
        elif reg == 0xC:
            pass  # serial data register ignored
        elif reg == 0xD:
            if val & 0x80:
                self.icr_mask |= val & 0x1F
            else:
                self.icr_mask &= ~val & 0x1F
        elif reg == 0xE:
            self.cra = val
            if val & 0x10:  # force load
                self.ta = self.ta_latch
                self.cra &= ~0x10
        elif reg == 0xF:
            self.crb = val
            if val & 0x10:
                self.tb = self.tb_latch
                self.crb &= ~0x10

    # ---- time ------------------------------------------------------------------------
    def tick(self, cycles: int) -> None:
        """Advance by CPU cycles.  Timer A counts system clocks when running;
        timer B counts clocks or timer-A underflows per CRB bits 5-6."""
        ta_underflows = 0
        if (self.cra & 0x01) and not (self.cra & 0x20):  # running, counting φ2
            ta_underflows = self._run_timer("a", cycles)
        if self.crb & 0x01:
            mode = (self.crb >> 5) & 0x03
            if mode == 0:  # count φ2
                self._run_timer("b", cycles)
            elif mode == 2:  # count timer A underflows
                if ta_underflows:
                    self._run_timer("b", ta_underflows)
            # modes 1/3 (CNT pin) never tick: no CNT source is modeled

    def _run_timer(self, which: str, steps: int) -> int:
        counter = self.ta if which == "a" else self.tb
        latch = self.ta_latch if which == "a" else self.tb_latch
        ctrl = self.cra if which == "a" else self.crb
        underflows = 0
        remaining = steps
        while remaining > 0:
            if counter >= remaining:
                counter -= remaining
                remaining = 0
            else:
                remaining -= counter + 1
                underflows += 1
                self.icr_status |= 0x01 if which == "a" else 0x02
                if ctrl & 0x08:  # one-shot: stop
                    counter = latch
                    ctrl &= ~0x01
                    remaining = 0
                else:
                    counter = latch
        if which == "a":
            self.ta, self.cra = counter, ctrl
        else:
            self.tb, self.crb = counter, ctrl
        return underflows

    def irq_asserted(self) -> bool:
        return bool(self.icr_status & self.icr_mask & 0x1F)

    # ---- snapshot ------------------------------------------------------------------
    _STATE = ("pra", "prb", "ddra", "ddrb", "ta_latch", "tb_latch", "ta", "tb",
              "cra", "crb", "icr_mask", "icr_status")

    def get_state(self) -> dict:
        return {k: getattr(self, k) for k in self._STATE}

    def set_state(self, state: dict) -> None:
        for k in self._STATE:
            setattr(self, k, state[k])
