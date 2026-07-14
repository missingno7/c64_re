"""Snapshot cloning + differential hook oracle tests (no game assets)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c64_re.hooks import HookRegistry  # noqa: E402
from c64_re.runtime import create_runtime, run_frames  # noqa: E402
from c64_re.snapshot import capture, clone_runtime, load_snapshot, restore, write_snapshot  # noqa: E402
from c64_re.verification import HookDivergence, HookOracle, HookStop, install_live_verifier  # noqa: E402

from test_machine import basic_stub, make_d64_with_prg  # noqa: E402


def build_rt(tmp_path, code: bytes, name=b"T"):
    prg = basic_stub(2061, code)  # code lands at $080D
    disk = make_d64_with_prg(name, prg)
    p = tmp_path / "t.d64"
    p.write_bytes(disk.data)
    return create_runtime(p)


# A program with a JSR-entered leaf routine we can hook:
#   $080D: JSR $0900 ; JMP $080D          (main loop, IRQs live)
#   $0900: LDA $FB / CLC / ADC #$05 / STA $FB / INC $D020 / RTS
MAIN = bytes((0x20, 0x00, 0x09, 0x4C, 0x0D, 0x08))
# same, but SEI first — no interrupts, for the exact-transactional test
MAIN_SEI = bytes((0x78, 0x20, 0x00, 0x09, 0x4C, 0x0E, 0x08))
LEAF_ADDR = 0x0900
LEAF = bytes((0xA5, 0xFB, 0x18, 0x69, 0x05, 0x85, 0xFB, 0xEE, 0x20, 0xD0, 0x60))


def build_leaf_rt(tmp_path, main: bytes = MAIN):
    rt = build_rt(tmp_path, main)
    rt.mem.ram[LEAF_ADDR:LEAF_ADDR + len(LEAF)] = LEAF
    return rt


def make_correct_hook():
    def hook(cpu):
        s = cpu.s
        s.a = cpu._nz(cpu.mem.rb(0xFB))   # LDA $FB
        s.c = 0                            # CLC
        cpu._adc(0x05)                     # ADC #$05
        cpu.mem.wb(0xFB, s.a)              # STA $FB
        old = cpu.mem.rb(0xD020)           # INC $D020 (RMW double write)
        cpu.mem.wb(0xD020, old)
        cpu.mem.wb(0xD020, (old + 1) & 0xFF)
        cpu._nz((old + 1) & 0xFF)
        # RTS
        lo = cpu.pull()
        hi = cpu.pull()
        s.pc = ((lo | (hi << 8)) + 1) & 0xFFFF
    return hook


def make_wrong_hook():
    def hook(cpu):
        s = cpu.s
        s.a = cpu._nz((cpu.mem.rb(0xFB) + 6) & 0xFF)  # off by one: +6
        cpu.mem.wb(0xFB, s.a)
        lo = cpu.pull()
        hi = cpu.pull()
        s.pc = ((lo | (hi << 8)) + 1) & 0xFFFF
    return hook


def test_snapshot_roundtrip_in_memory(tmp_path):
    rt = build_leaf_rt(tmp_path)
    run_frames(rt, 3)
    state = capture(rt)
    digest0 = (bytes(rt.mem.ram), rt.cpu.s.as_dict(), rt.machine.vic.raster)
    run_frames(rt, 2)  # mutate
    restore(rt, state)
    assert (bytes(rt.mem.ram), rt.cpu.s.as_dict(), rt.machine.vic.raster) == digest0
    # and the restored runtime keeps running identically to an untouched one
    rt2 = clone_runtime(rt)
    run_frames(rt, 2)
    run_frames(rt2, 2)
    assert bytes(rt.mem.ram) == bytes(rt2.mem.ram)
    assert rt.cpu.s.as_dict() == rt2.cpu.s.as_dict()


def test_snapshot_file_roundtrip(tmp_path):
    rt = build_leaf_rt(tmp_path)
    run_frames(rt, 3)
    snap = tmp_path / "s.c64snap"
    write_snapshot(rt, snap)
    rt2 = load_snapshot(snap)
    assert bytes(rt2.mem.ram) == bytes(rt.mem.ram)
    run_frames(rt, 2)
    run_frames(rt2, 2)
    assert bytes(rt2.mem.ram) == bytes(rt.mem.ram)
    assert rt2.cpu.s.as_dict() == rt.cpu.s.as_dict()


def _install(rt, handler, name):
    reg = HookRegistry()
    reg.replace(LEAF_ADDR, name)(handler)
    reg.install(rt.cpu)


def test_correct_hook_verifies(tmp_path):
    rt = build_leaf_rt(tmp_path)
    _install(rt, make_correct_hook(), "leaf_add5")
    oracle = install_live_verifier(rt)
    run_frames(rt, 3)
    assert oracle.stats.verified > 0
    assert not oracle.stats.diverged


def test_wrong_hook_diverges_with_precise_diff(tmp_path):
    rt = build_leaf_rt(tmp_path)
    _install(rt, make_wrong_hook(), "leaf_wrong")
    install_live_verifier(rt)
    with pytest.raises(HookDivergence) as ei:
        run_frames(rt, 3)
    msg = str(ei.value)
    assert "$00FB" in msg or "cpu.a" in msg  # names the diverging byte/register
    assert "$D020" in msg or "vic" in msg    # catches the missing INC $D020 too


def test_verified_run_matches_pure_asm_run(tmp_path):
    """The transactional property: a verified hooked run is bit-identical to
    a never-hooked run when both are sampled at the same PROGRAM point (the
    loop head after K leaf iterations).  Sampling by wall frames instead can
    legally catch the pure run mid-routine — hook atomicity shifts
    observation points, never state.  Interrupt-free program; with IRQs
    live, delivery-point skew is the frame/tick oracles' jurisdiction."""
    from c64_re.runtime import run_until

    loop_head = 0x080E  # MAIN_SEI: JSR at $080E
    def at_boundary(rt):
        return rt.cpu.s.pc == loop_head and rt.mem.ram[0xFB] >= 0x32  # 10 calls

    rt_pure = build_leaf_rt(tmp_path, MAIN_SEI)
    run_until(rt_pure, at_boundary)

    rt_hooked = build_leaf_rt(tmp_path, MAIN_SEI)
    _install(rt_hooked, make_correct_hook(), "leaf_add5")
    install_live_verifier(rt_hooked)
    run_until(rt_hooked, at_boundary)

    assert bytes(rt_hooked.mem.ram) == bytes(rt_pure.mem.ram)
    assert rt_hooked.cpu.s.as_dict() == rt_pure.cpu.s.as_dict()
    assert rt_hooked.machine.vic.raster == rt_pure.machine.vic.raster
    assert rt_hooked.cpu.cycle_count == rt_pure.cpu.cycle_count


def test_metadata_mode_continuation(tmp_path):
    """HookStop with explicit continuation for a non-JSR boundary."""
    rt = build_leaf_rt(tmp_path)
    _install(rt, make_correct_hook(), "leaf_add5")
    # strict mode would also work here; force metadata mode to exercise it
    oracle = HookOracle(rt, metadata={LEAF_ADDR: HookStop(continuations=(0x0810,))})
    rt.cpu.hook_verifier = oracle
    run_frames(rt, 2)
    assert oracle.stats.verified > 0


def test_divergent_hook_leaves_live_runtime_unpoisoned(tmp_path):
    """When the hook raises, the live runtime is restored to pre-hook state."""
    def exploding(cpu):
        cpu.mem.wb(0xFB, 0x99)
        raise RuntimeError("boom")

    rt = build_leaf_rt(tmp_path)
    before_fb = rt.mem.ram[0xFB]
    _install(rt, exploding, "leaf_boom")
    install_live_verifier(rt)
    with pytest.raises(HookDivergence):
        run_frames(rt, 2)
    assert rt.mem.ram[0xFB] == before_fb
