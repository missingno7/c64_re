"""6502 interpreter semantics tests (no game assets needed)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c64_re.cpu import CPU6502, CPUJam, CPUState  # noqa: E402
from c64_re.kernal import build_shim_basic, build_shim_chargen, build_shim_kernal  # noqa: E402
from c64_re.memory import Memory  # noqa: E402


def make_cpu(code: bytes, org: int = 0x1000) -> CPU6502:
    mem = Memory(basic_rom=build_shim_basic(), kernal_rom=build_shim_kernal(),
                 char_rom=build_shim_chargen())
    mem.ram[org:org + len(code)] = code
    cpu = CPU6502(mem, CPUState(pc=org, i=1))
    return cpu


def run(cpu: CPU6502, n: int) -> CPU6502:
    for _ in range(n):
        cpu.step()
    return cpu


def test_lda_flags():
    cpu = make_cpu(bytes((0xA9, 0x00, 0xA9, 0x80, 0xA9, 0x7F)))  # LDA #0/#$80/#$7F
    cpu.step()
    assert cpu.s.z == 1 and cpu.s.n == 0
    cpu.step()
    assert cpu.s.z == 0 and cpu.s.n == 1
    cpu.step()
    assert cpu.s.z == 0 and cpu.s.n == 0


def test_adc_binary_overflow_carry():
    # 0x50 + 0x50 = 0xA0: V=1 (pos+pos=neg), C=0
    cpu = make_cpu(bytes((0xA9, 0x50, 0x69, 0x50)))
    run(cpu, 2)
    assert cpu.s.a == 0xA0 and cpu.s.v == 1 and cpu.s.c == 0 and cpu.s.n == 1
    # 0xFF + 1 = 0x00: C=1, Z=1, V=0
    cpu = make_cpu(bytes((0xA9, 0xFF, 0x69, 0x01)))
    run(cpu, 2)
    assert cpu.s.a == 0x00 and cpu.s.c == 1 and cpu.s.z == 1 and cpu.s.v == 0


def test_sbc_borrow():
    # 0x00 - 1 (C=1): A=0xFF, C=0 (borrow)
    cpu = make_cpu(bytes((0x38, 0xA9, 0x00, 0xE9, 0x01)))
    run(cpu, 3)
    assert cpu.s.a == 0xFF and cpu.s.c == 0 and cpu.s.n == 1


def test_decimal_adc():
    # SED; LDA #$19; CLC; ADC #$01 -> $20 (BCD), C=0
    cpu = make_cpu(bytes((0xF8, 0xA9, 0x19, 0x18, 0x69, 0x01)))
    run(cpu, 4)
    assert cpu.s.a == 0x20 and cpu.s.c == 0
    # SED; LDA #$99; CLC; ADC #$01 -> $00, C=1
    cpu = make_cpu(bytes((0xF8, 0xA9, 0x99, 0x18, 0x69, 0x01)))
    run(cpu, 4)
    assert cpu.s.a == 0x00 and cpu.s.c == 1


def test_decimal_sbc():
    # SED; SEC; LDA #$20; SBC #$01 -> $19, C=1
    cpu = make_cpu(bytes((0xF8, 0x38, 0xA9, 0x20, 0xE9, 0x01)))
    run(cpu, 4)
    assert cpu.s.a == 0x19 and cpu.s.c == 1
    # SED; SEC; LDA #$00; SBC #$01 -> $99, C=0
    cpu = make_cpu(bytes((0xF8, 0x38, 0xA9, 0x00, 0xE9, 0x01)))
    run(cpu, 4)
    assert cpu.s.a == 0x99 and cpu.s.c == 0


def test_jmp_indirect_page_wrap_bug():
    cpu = make_cpu(bytes((0x6C, 0xFF, 0x20)))  # JMP ($20FF)
    cpu.mem.ram[0x20FF] = 0x34
    cpu.mem.ram[0x2000] = 0x12  # high byte from $2000, NOT $2100
    cpu.mem.ram[0x2100] = 0x99
    cpu.step()
    assert cpu.s.pc == 0x1234


def test_jsr_rts_roundtrip():
    cpu = make_cpu(bytes((0x20, 0x00, 0x20)))  # JSR $2000
    cpu.mem.ram[0x2000] = 0x60  # RTS
    run(cpu, 2)
    assert cpu.s.pc == 0x1003


def test_brk_dispatches_through_0316():
    cpu = make_cpu(bytes((0x00,)))
    cpu.s.i = 0
    # vectors: $0314 -> $4000 (IRQ), $0316 -> $5000 (BRK)
    cpu.mem.ram[0x314:0x318] = bytes((0x00, 0x40, 0x00, 0x50))
    for _ in range(12):
        cpu.step()
        if cpu.s.pc in (0x4000, 0x5000):
            break
    assert cpu.s.pc == 0x5000


def test_irq_respects_i_flag_and_vector():
    cpu = make_cpu(bytes((0xEA, 0xEA, 0xEA)))
    cpu.mem.ram[0x314:0x316] = bytes((0x00, 0x40))
    asserted = {"on": True}
    cpu.irq_line = lambda: asserted["on"]
    cpu.s.i = 1
    cpu.step()  # masked
    assert cpu.s.pc == 0x1001
    cpu.s.i = 0
    for _ in range(10):
        cpu.step()
        if cpu.s.pc == 0x4000:
            break
    assert cpu.s.pc == 0x4000
    # the pushed status must have B clear for a hardware IRQ
    flags = cpu.mem.ram[0x0100 | ((cpu.s.sp + 1) & 0xFF)]
    assert flags & 0x10 == 0


def test_illegal_lax_dcp():
    cpu = make_cpu(bytes((0xA7, 0x10, 0xC7, 0x10)))  # LAX $10; DCP $10
    cpu.mem.ram[0x10] = 0x41
    cpu.step()
    assert cpu.s.a == 0x41 and cpu.s.x == 0x41
    cpu.step()
    assert cpu.mem.ram[0x10] == 0x40
    assert cpu.s.c == 1  # A(0x41) >= 0x40


def test_jam_fails_loud():
    cpu = make_cpu(bytes((0x02,)))
    try:
        cpu.step()
    except CPUJam as e:
        assert "1000" in str(e)
    else:
        raise AssertionError("JAM did not raise")


def test_rmw_double_write_hits_io():
    """INC on a VIC register must write old-then-new (IRQ ack depends on it)."""
    from c64_re.machine import C64Machine
    mem = Memory(basic_rom=build_shim_basic(), kernal_rom=build_shim_kernal(),
                 char_rom=build_shim_chargen())
    machine = C64Machine(mem)
    writes = []
    orig = machine.vic.write
    machine.vic.write = lambda r, v: (writes.append((r, v)), orig(r, v))
    mem.ram[0x1000:0x1003] = bytes((0xEE, 0x20, 0xD0))  # INC $D020
    cpu = CPU6502(mem, CPUState(pc=0x1000))
    machine.cpu = cpu
    cpu.step()
    assert len(writes) == 2  # old value, then new
