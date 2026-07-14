"""Lifter tests: decode/cfg refusals, emit fidelity, oracle-verified installs."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c64_re.hooks import HookRegistry  # noqa: E402
from c64_re.lift.cfg import refuse_unsafe_callers, scan_function  # noqa: E402
from c64_re.lift.emit import LiftRefused, lift_and_compile  # noqa: E402
from c64_re.lift.manifest import LiftManifest, LiftRecord  # noqa: E402
from c64_re.runtime import run_until  # noqa: E402
from c64_re.verification import install_live_verifier  # noqa: E402

from test_machine import basic_stub, make_d64_with_prg  # noqa: E402
from test_verification import build_rt  # noqa: E402


def reader(data: bytes, base: int):
    def read(addr):
        a = (addr - base) & 0xFFFF
        return data[a] if 0 <= a < len(data) else 0x02  # JAM outside
    return read


# ---- cfg / refusals -----------------------------------------------------------
def test_scan_simple_leaf():
    # LDA #$01 / CLC / ADC $FB / STA $FB / RTS
    code = bytes((0xA9, 0x01, 0x18, 0x65, 0xFB, 0x85, 0xFB, 0x60))
    scan = scan_function(reader(code, 0x4000), 0x4000)
    assert scan
    assert len(scan.insns) == 5
    assert scan.exits == {0x4007}
    assert scan.byte_ranges == [(0x4000, 0x4008)]


def test_scan_branch_and_loop():
    # LDX #$05 / loop: DEX / BNE loop / RTS
    code = bytes((0xA2, 0x05, 0xCA, 0xD0, 0xFD, 0x60))
    scan = scan_function(reader(code, 0x4000), 0x4000)
    assert scan
    assert 0x4002 in scan.block_starts  # branch target
    assert 0x4005 in scan.block_starts  # fall-through


def test_scan_allows_bit_skip_overlap():
    # The 6502 BIT-skip idiom (as in Stix $653C):
    #   LDX $10 / BNE +3 / LDA #$14 / .byte $2C (BIT abs) / LDA #$02 / STA $11 / RTS
    # BNE targets the LDA #$02 that the $2C swallows on fall-through.
    #   4000 A6 10     LDX $10
    #   4002 D0 03     BNE $4007
    #   4004 A9 14     LDA #$14
    #   4006 2C A9 02  BIT $02A9   <- $4007 (A9 02 = LDA #$02) is inside it
    #   4009 85 11     STA $11
    #   400B 60        RTS
    code = bytes((0xA6, 0x10, 0xD0, 0x03, 0xA9, 0x14, 0x2C, 0xA9, 0x02,
                  0x85, 0x11, 0x60))
    scan = scan_function(reader(code, 0x4000), 0x4000)
    assert scan, "BIT-skip overlap must lift, not refuse"
    assert scan.overlaps >= 1
    assert 0x4007 in scan.insns and 0x4006 in scan.insns  # both readings kept


def test_bit_skip_overlap_lifts_and_matches_interpreter(tmp_path):
    """Both entry paths of a BIT-skip idiom reproduce the interpreter exactly."""
    from c64_re.cpu import CPU6502, CPUState
    from c64_re.kernal import build_shim_basic, build_shim_chargen, build_shim_kernal
    from c64_re.memory import Memory

    code = bytes((0xA6, 0x10, 0xD0, 0x03, 0xA9, 0x14, 0x2C, 0xA9, 0x02,
                  0x85, 0x11, 0x60))
    for x10 in (0x00, 0x05):  # 0 => fall-through (STA #$14); nonzero => skip (STA #$02)
        mem = Memory(basic_rom=build_shim_basic(), kernal_rom=build_shim_kernal(),
                     char_rom=build_shim_chargen())
        mem.ram[0x4000:0x4000 + len(code)] = code
        mem.ram[0x10] = x10
        interp = CPU6502(mem, CPUState(pc=0x4000, sp=0xFD))
        interp.push(0x12)  # RTS returns to $1233+1 = $1234
        interp.push(0x33)
        for _ in range(8):
            if interp.s.pc == 0x1234:
                break
            interp.step()
        expected = mem.ram[0x11]

        hook, source, scan = lift_and_compile(reader(code, 0x4000), 0x4000)
        assert "BIT" in source
        mem2 = Memory(basic_rom=build_shim_basic(), kernal_rom=build_shim_kernal(),
                      char_rom=build_shim_chargen())
        mem2.ram[0x4000:0x4000 + len(code)] = code
        mem2.ram[0x10] = x10
        cpu2 = CPU6502(mem2, CPUState(pc=0x4000, sp=0xFD))
        cpu2.push(0x12)
        cpu2.push(0x33)
        hook(cpu2)
        assert mem2.ram[0x11] == expected, f"x10={x10}: hook {mem2.ram[0x11]:#x} != {expected:#x}"


def test_scan_refuses_indirect_jmp():
    code = bytes((0x6C, 0x00, 0x30))  # JMP ($3000)
    scan = scan_function(reader(code, 0x4000), 0x4000)
    assert not scan and scan.reason == "jmp_ind"


def test_scan_refuses_jam_and_brk():
    assert scan_function(reader(bytes((0x02,)), 0x4000), 0x4000).reason == "bad_opcode"
    assert scan_function(reader(bytes((0x00,)), 0x4000), 0x4000).reason == "brk"


def test_scan_refuses_endless_loop():
    code = bytes((0x4C, 0x00, 0x40))  # JMP $4000 (self)
    assert scan_function(reader(code, 0x4000), 0x4000).reason == "no_exit"


# ---- non-local return (cascading-return) idiom ------------------------------------
# As found in Stix's $7166 (collision-abort): PLA PLA discards the return
# address ITS OWN caller's JSR pushed, then the eventual RTS lands in the
# caller's CALLER, not the immediate caller.  Legal on real hardware; unsafe
# to wrap in the lifter's emulate_call (which waits for a specific PC+SP).
#   4000 A9 00     LDA #$00
#   4002 68        PLA
#   4003 68        PLA
#   4004 A9 05     LDA #$05
#   4006 60        RTS
NONLOCAL_RETURN_CODE = bytes((0xA9, 0x00, 0x68, 0x68, 0xA9, 0x05, 0x60))


def test_scan_refuses_nonlocal_return():
    scan = scan_function(reader(NONLOCAL_RETURN_CODE, 0x4000), 0x4000)
    assert not scan and scan.reason == "nonlocal_return"


def test_scan_allows_balanced_push_pop():
    # PHA then PLA before RTS: net-zero, perfectly ordinary — must lift fine.
    code = bytes((0x48, 0x68, 0x60))  # PHA / PLA / RTS
    scan = scan_function(reader(code, 0x4000), 0x4000)
    assert scan


def test_refuse_unsafe_callers_flags_direct_caller_only():
    # $4000 = the nonlocal_return leaf; $4010 JSRs it (unsafe, tier 2);
    # $4020 JSRs $4010 (SAFE — one level up, per the module's design: once
    # $4010 is left uninstalled, plain interpretation handles the skip).
    leaf = NONLOCAL_RETURN_CODE
    mid = bytes((0x20, 0x00, 0x40, 0x60))          # JSR $4000 / RTS
    top = bytes((0x20, 0x10, 0x40, 0x60))          # JSR $4010 / RTS
    blob = {0x4000: leaf, 0x4010: mid, 0x4020: top}

    def read(addr):
        for base, code in blob.items():
            if base <= addr < base + len(code):
                return code[addr - base]
        return 0x02
    scans = {addr: scan_function(read, addr) for addr in blob}
    assert not scans[0x4000] and scans[0x4000].reason == "nonlocal_return"
    assert scans[0x4010]  # tier-1 alone doesn't catch the direct caller
    assert scans[0x4020]

    extra = refuse_unsafe_callers(scans)
    assert set(extra) == {0x4010}
    assert extra[0x4010].reason == "calls_nonlocal_return"


# ---- emit fidelity: lifted vs interpreted, full-state, via the oracle ------------
# A deliberately gnarly routine exercising branches, RMW on I/O, page-cross
# indexing, decimal mode, stack ops, and a JSR to a sub-leaf:
#   $0900: LDX #$04
#   $0902: loop: LDA $C0FE,X   (page-crossing for X>=2)
#   $0905: SED / CLC / ADC #$25 / CLD
#   $090A: STA $C100,X
#   $090D: JSR $0920
#   $0910: DEX / BPL loop
#   $0912: INC $D020
#   $0915: PHP / PLA / STA $FB
#   $0919: RTS
GNARLY = bytes((
    0xA2, 0x04,
    0xBD, 0xFE, 0xC0,
    0xF8, 0x18, 0x69, 0x25, 0xD8,
    0x9D, 0x00, 0xC1,
    0x20, 0x20, 0x09,
    0xCA, 0x10, 0xEF,
    0xEE, 0x20, 0xD0,
    0x08, 0x68, 0x85, 0xFB,
    0x60,
))
SUB = bytes((0xE6, 0xFC, 0x60))  # INC $FC / RTS
GNARLY_ADDR = 0x0900
SUB_ADDR = 0x0920
# main loop calls the gnarly routine forever:  JSR $0900 / JMP $080D
MAIN = bytes((0x20, 0x00, 0x09, 0x4C, 0x0D, 0x08))


def build_gnarly_rt(tmp_path):
    rt = build_rt(tmp_path, MAIN)
    rt.mem.ram[GNARLY_ADDR:GNARLY_ADDR + len(GNARLY)] = GNARLY
    rt.mem.ram[SUB_ADDR:SUB_ADDR + len(SUB)] = SUB
    for i in range(8):
        rt.mem.ram[0xC0FE + i] = 0x11 * (i + 1)
    return rt


def test_lifted_hook_passes_oracle(tmp_path):
    rt = build_gnarly_rt(tmp_path)
    hook, source, scan = lift_and_compile(rt.mem.rb, GNARLY_ADDR)
    assert "SED" in source and "emulate_call" in source  # literal artifact
    reg = HookRegistry()
    reg.replace(GNARLY_ADDR, "lifted_gnarly")(hook)
    reg.install(rt.cpu)
    # strict_cycles: a lifted hook must reproduce the interpreter's exact
    # cycle model, not just its end state
    oracle = install_live_verifier(rt, strict_cycles=True)
    # each gnarly call runs the sub-leaf 5 times (X = 4..0)
    run_until(rt, lambda r: r.mem.ram[0xFC] >= 25)
    assert oracle.stats.verified >= 5
    assert not oracle.stats.diverged


def test_lifted_hook_matches_pure_run_exactly(tmp_path):
    """Lifted+verified run vs never-hooked run: identical at a program point,
    including cycle counts (the emitter's cycle model is the interpreter's)."""
    def at_boundary(r):
        return r.cpu.s.pc == 0x080D and r.mem.ram[0xFC] >= 6

    pure = build_gnarly_rt(tmp_path)
    run_until(pure, at_boundary)

    lifted = build_gnarly_rt(tmp_path)
    hook, _, _ = lift_and_compile(lifted.mem.rb, GNARLY_ADDR)
    reg = HookRegistry()
    reg.replace(GNARLY_ADDR, "lifted_gnarly")(hook)
    reg.install(lifted.cpu)
    run_until(lifted, at_boundary)

    assert bytes(lifted.mem.ram) == bytes(pure.mem.ram)
    assert lifted.cpu.s.as_dict() == pure.cpu.s.as_dict()
    assert lifted.cpu.cycle_count == pure.cpu.cycle_count
    assert lifted.machine.vic.raster == pure.machine.vic.raster
    assert bytes(lifted.machine.vic.regs) == bytes(pure.machine.vic.regs)


def test_smc_guard_refuses_patched_code(tmp_path):
    rt = build_gnarly_rt(tmp_path)
    hook, _, _ = lift_and_compile(rt.mem.rb, GNARLY_ADDR)
    rt.mem.ram[GNARLY_ADDR + 7] = 0x26  # patch the ADC operand ($25 -> $26)
    with pytest.raises(RuntimeError) as ei:
        hook(rt.cpu)
    assert "runtime-patched" in str(ei.value)


def test_lift_refusal_is_structured(tmp_path):
    rt = build_gnarly_rt(tmp_path)
    rt.mem.ram[0x0930:0x0933] = bytes((0x6C, 0x00, 0x30))  # JMP ($3000)
    with pytest.raises(LiftRefused) as ei:
        lift_and_compile(rt.mem.rb, 0x0930)
    assert ei.value.refusal.reason == "jmp_ind"


def test_manifest_roundtrip(tmp_path):
    m = LiftManifest()
    m.update(LiftRecord(entry=0x0900, name="lifted_0900", status="ORACLE_PASSING",
                        size_bytes=26, instructions=15, calls_seen=6,
                        verified_calls=6))
    m.update(LiftRecord(entry=0x0930, name="refused_0930", status="REFUSED",
                        refusal_reason="jmp_ind"))
    p = tmp_path / "lift_manifest.json"
    m.save(p)
    m2 = LiftManifest.load(p)
    assert m2.get(0x0900).verified_calls == 6
    assert m2.get(0x0930).status == "REFUSED"
    assert "ORACLE_PASSING=1" in m2.summary()
