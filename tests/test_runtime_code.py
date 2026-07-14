"""Smoke tests for ``c64_re.runtime_code`` (polyvariant runtime-patched-code
support, ported from dos_re) -- especially relevant on the 6502, where
self-modifying code is idiomatic; the per-game slot table stays behind as
caller-supplied data."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c64_re.cpu import CPU6502  # noqa: E402
from c64_re.kernal import build_shim_basic, build_shim_chargen, build_shim_kernal  # noqa: E402
from c64_re.memory import Memory  # noqa: E402
from c64_re.runtime_code import (  # noqa: E402
    RuntimeCodeSlot,
    RuntimeCodeStaticization,
    RuntimeCodeStaticizationError,
    RuntimeCodeVariant,
    RuntimeCodeWriteTracer,
    UnknownRuntimeCodeVariant,
    assert_runtime_code_staticization_ready,
    default_runtime_code_regions,
    identify_runtime_code_variant,
    require_runtime_code_variant,
    runtime_code_staticization_report,
    variants_by_addr,
)

ADDR = 0x0900


def _cpu_with_bytes(data: bytes) -> CPU6502:
    mem = Memory(basic_rom=build_shim_basic(), kernal_rom=build_shim_kernal(),
                 char_rom=build_shim_chargen())
    cpu = CPU6502(mem)
    mem.load_block(ADDR, data)
    return cpu


def _slots(*variants: RuntimeCodeVariant, staticization=None, installer_status="observed") -> dict:
    slot = RuntimeCodeSlot(
        addr=ADDR, name="slot", island="test", owner=None, role="test slot",
        variants=variants, staticization=staticization, installer_status=installer_status,
    )
    return {ADDR: slot}


def test_identify_matches_known_variant_by_signature():
    accepted = RuntimeCodeVariant(addr=ADDR, name="accepted", signature=b"\xEA\xEA", island="x", status="hooked-verified")
    other = RuntimeCodeVariant(addr=ADDR, name="cold", signature=b"\x00\x00", island="x", status="known-not-this-hook")
    slots = _slots(accepted, other)
    cpu = _cpu_with_bytes(b"\xEA\xEA")

    variant = identify_runtime_code_variant(cpu, ADDR, slots)
    assert variant.name == "accepted"
    assert accepted.is_accepted_runtime_body
    assert not other.is_accepted_runtime_body


def test_identify_raises_on_unregistered_address():
    with pytest.raises(UnknownRuntimeCodeVariant):
        identify_runtime_code_variant(_cpu_with_bytes(b"\xEA"), ADDR, {})


def test_identify_raises_on_unknown_bytes():
    accepted = RuntimeCodeVariant(addr=ADDR, name="accepted", signature=b"\xEA\xEA", island="x", status="hooked-verified")
    cpu = _cpu_with_bytes(b"\x11\x22")
    with pytest.raises(UnknownRuntimeCodeVariant):
        identify_runtime_code_variant(cpu, ADDR, _slots(accepted))


def test_require_rejects_known_but_wrong_variant():
    accepted = RuntimeCodeVariant(addr=ADDR, name="accepted", signature=b"\xEA\xEA", island="x", status="hooked-verified")
    slots = _slots(accepted)
    cpu = _cpu_with_bytes(b"\xEA\xEA")
    assert require_runtime_code_variant(cpu, ADDR, "accepted", slots).name == "accepted"
    with pytest.raises(UnknownRuntimeCodeVariant):
        require_runtime_code_variant(cpu, ADDR, "some_other_hook", slots)


def test_staticization_report_and_gate():
    accepted = RuntimeCodeVariant(addr=ADDR, name="accepted", signature=b"\xEA\xEA", island="x", status="hooked-verified")
    slots = _slots(accepted)

    report = runtime_code_staticization_report(slots)
    assert report[0]["missing"] == ("static source target",)
    with pytest.raises(RuntimeCodeStaticizationError):
        assert_runtime_code_staticization_ready(slots)

    staticized = _slots(
        accepted,
        staticization=RuntimeCodeStaticization(
            source_module="game.recovered", source_function="run_slot", dispatch="variant_guarded_static_hook",
        ),
    )
    assert_runtime_code_staticization_ready(staticized)  # must not raise


def test_variants_by_addr_backwards_compatible_lookup():
    accepted = RuntimeCodeVariant(addr=ADDR, name="accepted", signature=b"\xEA\xEA", island="x", status="hooked-verified")
    slots = _slots(accepted)
    assert variants_by_addr(slots) == {ADDR: (accepted,)}


def test_write_tracer_fires_only_inside_registered_regions():
    accepted = RuntimeCodeVariant(addr=ADDR, name="accepted", signature=b"\xEA\xEA", island="x", status="hooked-verified")
    slots = _slots(accepted)
    cpu = _cpu_with_bytes(b"\xEA\xEA")
    regions = default_runtime_code_regions(slots)

    tracer = RuntimeCodeWriteTracer(cpu, regions).install()
    try:
        cpu.mem.wb(ADDR, 0xCC)             # inside the region -> traced
        cpu.mem.wb(ADDR + 0x1000, 0xCC)    # far outside -> not traced
        cpu.mem.wb(ADDR + 1, 0xEA)         # same value rewritten -> not traced
    finally:
        tracer.uninstall()

    assert len(tracer.events) == 1
    event = tracer.events[0]
    assert event.writer == cpu.s.pc
    assert event.target == ADDR
    assert (event.old, event.new) == (b"\xEA", b"\xCC")

    # uninstall restores the original wb: further writes are not traced
    cpu.mem.wb(ADDR, 0x33)
    assert len(tracer.events) == 1
    assert cpu.mem.ram[ADDR] == 0x33
