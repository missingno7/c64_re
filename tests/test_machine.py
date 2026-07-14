"""Machine-level tests: D64, banking, CIA timers, KERNAL HLE, determinism."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c64_re.cia import CIA  # noqa: E402
from c64_re.cpu import CPU6502, CPUState  # noqa: E402
from c64_re.d64 import SECTORS_PER_TRACK, DiskImage, parse_basic_sys  # noqa: E402
from c64_re.kernal import build_shim_basic, build_shim_chargen, build_shim_kernal  # noqa: E402
from c64_re.machine import C64Machine  # noqa: E402
from c64_re.memory import Memory  # noqa: E402


# ---- synthetic D64 ------------------------------------------------------------------
def make_d64_with_prg(name: bytes, prg: bytes) -> DiskImage:
    data = bytearray(683 * 256)

    def off(t, s):
        o = 0
        for tt in range(1, t):
            o += SECTORS_PER_TRACK[tt]
        return (o + s) * 256

    # BAM
    data[off(18, 0) + 0x90:off(18, 0) + 0xA0] = b"TEST DISK".ljust(16, b"\xA0")
    # directory sector 18/1, one PRG entry starting at 17/0
    d = off(18, 1)
    data[d + 0] = 0
    data[d + 1] = 0xFF
    data[d + 2] = 0x82  # closed PRG
    data[d + 3] = 17
    data[d + 4] = 0
    data[d + 5:d + 21] = name.ljust(16, b"\xA0")
    blocks = (len(prg) + 253) // 254
    data[d + 30] = blocks & 0xFF
    # file chain on track 17
    pos = 0
    sector = 0
    while pos < len(prg):
        chunk = prg[pos:pos + 254]
        o = off(17, sector)
        pos += len(chunk)
        if pos < len(prg):
            data[o] = 17
            data[o + 1] = sector + 1
        else:
            data[o] = 0
            data[o + 1] = len(chunk) + 1
        data[o + 2:o + 2 + len(chunk)] = chunk
        sector += 1
    return DiskImage(bytes(data))


def basic_stub(sys_target: int, code: bytes) -> bytes:
    digits = str(sys_target).encode()
    line = b"\x9e" + digits + b"\x00"
    next_addr = 0x0801 + 4 + len(line)
    stub = bytes((next_addr & 0xFF, next_addr >> 8, 10, 0)) + line + b"\x00\x00"
    load = 0x0801
    body = stub + code
    return bytes((load & 0xFF, load >> 8)) + body


def test_d64_directory_and_read():
    prg = basic_stub(2061, bytes((0x60,)))
    disk = make_d64_with_prg(b"GAME", prg)
    assert disk.disk_name == b"TEST DISK"
    entries = disk.directory()
    assert len(entries) == 1 and entries[0].display_name == "GAME"
    assert disk.read_file(b"*") == prg
    assert disk.read_file(b"G*") == prg
    assert parse_basic_sys(prg) == 2061


def test_runtime_boot_and_exit(tmp_path):
    """A PRG that pokes the border color and RTSes back to the boot frame."""
    from c64_re.kernal import ProgramExit
    from c64_re.runtime import create_runtime
    code = bytes((0xA9, 0x02, 0x8D, 0x20, 0xD0, 0x60))  # LDA#2 STA $D020 RTS
    prg = basic_stub(2061, code)
    disk = make_d64_with_prg(b"TINY", prg)
    p = tmp_path / "tiny.d64"
    p.write_bytes(disk.data)
    rt = create_runtime(p)
    try:
        for _ in range(100):
            rt.cpu.step()
    except ProgramExit:
        pass
    else:
        raise AssertionError("program did not exit through the boot frame")
    assert rt.machine.vic.regs[0x20] == 2


def test_banking_ram_under_rom():
    mem = Memory(basic_rom=build_shim_basic(), kernal_rom=build_shim_kernal(),
                 char_rom=build_shim_chargen())
    C64Machine(mem)
    assert mem.rb(0xE000) == 0x02  # shim KERNAL filler visible
    mem.wb(0xE000, 0xAB)           # write lands under ROM
    assert mem.rb(0xE000) == 0x02
    mem.wb(1, 0x35)                # bank KERNAL out (hiram=0... 0x35: loram=1,hiram=0)
    assert mem.rb(0xE000) == 0xAB
    mem.wb(1, 0x37)
    assert mem.rb(0xE000) == 0x02


def test_cia_timer_underflow_counts():
    cia = CIA("t")
    cia.write(0x4, 9)     # latch A = 9
    cia.write(0x5, 0)
    cia.write(0xD, 0x81)  # enable timer A IRQ
    cia.write(0xE, 0x01)  # start, continuous
    fired = 0
    for _ in range(10):
        cia.tick(10)      # each 10 cycles = one underflow
        if cia.irq_asserted():
            fired += 1
            cia.read(0xD)  # acknowledge
    assert fired == 10


def test_kernal_chrout_writes_screen():
    from c64_re.runtime import create_runtime
    # SYS stub places code at $080D: JSR CHROUT with 'A', then loop forever
    code = bytes((0xA9, 0x41, 0x20, 0xD2, 0xFF, 0x4C, 0x12, 0x08))
    prg = basic_stub(2061, code)
    disk = make_d64_with_prg(b"HELLO", prg)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "hello.d64"
        p.write_bytes(disk.data)
        rt = create_runtime(p)
        for _ in range(200):
            rt.cpu.step()
    assert rt.mem.ram[0x0400] == 0x01  # screen code for 'A'


def test_determinism_same_run_same_state():
    from c64_re.runtime import create_runtime, run_frames
    code = bytes((  # code at $080D: increment $D020 and a RAM counter forever
        0xEE, 0x20, 0xD0, 0xE6, 0xFB, 0x4C, 0x0D, 0x08,
    ))
    prg = basic_stub(2061, code)
    disk = make_d64_with_prg(b"DET", prg)
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "det.d64"
        p.write_bytes(disk.data)
        states = []
        for _ in range(2):
            rt = create_runtime(p)
            run_frames(rt, 20)
            states.append((
                bytes(rt.mem.ram), rt.cpu.instr_count, rt.cpu.cycle_count,
                bytes(rt.machine.vic.regs), rt.machine.vic.raster,
            ))
    assert states[0] == states[1]
