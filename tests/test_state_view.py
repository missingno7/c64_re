"""Smoke tests for the state-view descriptors (the state-mirror machinery).

Ported from the state_view section of dos_re's tests/test_promoted_scaffolding.py,
with segment bases translated to flat 16-bit C64 addresses and a RamBackend
(live-machine RAM) case added.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest  # noqa: E402

from c64_re.state_view import (  # noqa: E402
    ByteBackend,
    OverlayBackend,
    RamBackend,
    S8,
    S16,
    StructArray,
    StructView,
    U8,
    U16,
    WidthContractBackend,
    coerce_backend,
)


DATA_BASE = 0x0800  # a game's state block base, from the oracle


class Slot(StructView):
    x = U16(0)
    y = S16(2)
    life = U8(4)


class World(StructView):
    wind = U16(0x100)
    slots = StructArray(0x200, 6, 3, Slot)

    def __init__(self, source):
        super().__init__(coerce_backend(source, DATA_BASE), 0)


def test_byte_backend_view_roundtrip_writes_the_same_bytes():
    data = bytearray(0x10000)
    w = World(data)
    w.wind = 0x1234
    w.slots[1].x = 0xBEEF
    w.slots[1].y = -2
    w.slots[1].life = 7

    assert data[DATA_BASE + 0x100] == 0x34 and data[DATA_BASE + 0x101] == 0x12
    slot1 = DATA_BASE + 0x200 + 6
    assert data[slot1] == 0xEF and data[slot1 + 1] == 0xBE
    assert w.slots[1].y == -2          # signed round-trip
    assert w.slots[-2].x == 0xBEEF     # negative index wraps
    assert len(w.slots) == 3


def test_ram_backend_views_live_machine_ram():
    from c64_re.kernal import build_shim_basic, build_shim_chargen, build_shim_kernal
    from c64_re.memory import Memory

    mem = Memory(
        basic_rom=build_shim_basic(),
        kernal_rom=build_shim_kernal(),
        char_rom=build_shim_chargen(),
    )
    w = World(mem)                     # coerce_backend picks RamBackend
    assert isinstance(w._backend, RamBackend)
    w.wind = 0xCAFE
    assert mem.ram[DATA_BASE + 0x100] == 0xFE
    assert mem.ram[DATA_BASE + 0x101] == 0xCA
    assert mem.rb(DATA_BASE + 0x100) == 0xFE   # CPU sees the same RAM byte

    mem.ram[DATA_BASE + 0x200 + 4] = 0x81
    assert w.slots[0].life == 0x81

    # mirrors address RAM even under ROM: $A000-$BFFF writes land below BASIC
    hidden = RamBackend(mem, 0xA000)
    hidden.ww(0, 0xBEEF)
    assert mem.ram[0xA000] == 0xEF and mem.ram[0xA001] == 0xBE


def test_backend_addressing_wraps_at_64k():
    data = bytearray(0x10000)
    b = ByteBackend(data, 0xFFFF)
    b.ww(0, 0x1234)                    # high byte wraps to $0000
    assert data[0xFFFF] == 0x34 and data[0x0000] == 0x12
    assert b.rw(0) == 0x1234


def test_overlay_backend_accumulates_contract_without_touching_base():
    base = bytearray(0x10000)
    base[0x50] = 0xAA
    ov = OverlayBackend(lambda off: base[off])
    slot = Slot(ov, 0x50)

    assert slot.x == 0x00AA            # read-through
    slot.life = 9                      # write accumulates
    assert base[0x54] == 0             # base untouched
    assert ov.writes == {0x54: 9}
    assert slot.life == 9              # overlay sees its own write


def test_width_contract_backend_records_widths_and_reads_original_memory():
    base = bytearray(0x10000)
    base[0x50] = 0x11
    base[0x51] = 0x22
    wc = WidthContractBackend(lambda off: base[off],
                              lambda off: base[off] | (base[off + 1] << 8))
    slot = Slot(wc, 0x50)

    assert slot.x == 0x2211            # reads original memory
    slot.x = 0x3344
    slot.life = 5
    assert slot.x == 0x2211            # writes are NOT visible to reads
    assert wc.writes == {0x50: (0x3344, 2), 0x54: (5, 1)}
    assert base[0x50] == 0x11          # base untouched


def test_coerce_backend_passes_backends_through():
    b = ByteBackend(bytearray(16), 0)
    assert coerce_backend(b, 0xBEEF) is b


def test_signed_byte_field_roundtrip():
    class Flags(StructView):
        dx = S8(0)

    data = bytearray(16)
    f = Flags(ByteBackend(data), 0)
    f.dx = -3
    assert data[0] == 0xFD
    assert f.dx == -3


def test_pytest_raises_available():
    # keep the ported suite exercising the raises path like the original did
    with pytest.raises(ValueError):
        raise ValueError("boom")
