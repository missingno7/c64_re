"""Runtime assembly: boot a PRG (from a D64 or a bare .prg) into a powered-on C64.

The boot path mirrors what LOAD"file",8,1 : RUN does — without executing any
BASIC ROM code: the PRG is placed at its load address, the standard BASIC
loader stub is parsed *statically* for its SYS target, and the CPU starts
there with a return address pointing at the exit trap (so a program that
RTSes back to BASIC raises :class:`c64_re.kernal.ProgramExit` instead of
running interpreter code we don't model).

A raw machine-code PRG without a BASIC stub needs an explicit ``entry=`` —
that is game knowledge and belongs to the adapter.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .cpu import CPU6502
from .d64 import DiskImage, parse_basic_sys, prg_load_address, prg_payload
from .hooks import registry
from .kernal import (
    EXIT_TRAP,
    build_shim_basic,
    build_shim_chargen,
    build_shim_kernal,
    load_real_roms,
)
from .machine import C64Machine
from .memory import Memory


@dataclass
class ProgramInfo:
    source: str
    file_name: str
    load_addr: int
    end_addr: int
    entry: int


@dataclass
class Runtime:
    program: ProgramInfo
    cpu: CPU6502
    mem: Memory
    machine: C64Machine
    # construction inputs, kept so verification can clone a fresh runtime
    boot_args: dict = None


def create_runtime(
    image_path: str | Path,
    *,
    file: bytes | str = "*",
    entry: int | None = None,
    roms_dir: str | Path | None = None,
    install_hooks: bool = True,
) -> Runtime:
    image_path = Path(image_path)
    roms = load_real_roms(roms_dir) if roms_dir else {}
    mem = Memory(
        basic_rom=roms.get("basic", build_shim_basic()),
        kernal_rom=roms.get("kernal", build_shim_kernal()),
        char_rom=roms.get("chargen", build_shim_chargen()),
    )
    machine = C64Machine(mem)
    cpu = CPU6502(mem)
    machine.cpu = cpu
    cpu.tick = machine.tick
    cpu.irq_line = machine.irq_line
    machine.kernal.install(cpu)
    machine.power_on()

    # ---- attach media & load the program ----
    if image_path.suffix.lower() == ".d64":
        disk = DiskImage.load(image_path)
        machine.drive = disk
        dir_entry = disk.find(file)
        prg = disk.read_chain(dir_entry.track, dir_entry.sector)
        file_name = dir_entry.display_name
    elif image_path.suffix.lower() == ".prg":
        prg = image_path.read_bytes()
        file_name = image_path.name
    else:
        raise ValueError(f"unsupported program image {image_path.name!r} (.d64/.prg)")

    load_addr = prg_load_address(prg)
    payload = prg_payload(prg)
    end = load_addr + len(payload)
    if end > 0x10000:
        raise ValueError(f"PRG ${load_addr:04X}+{len(payload)} overruns 64K")
    mem.ram[load_addr:end] = payload
    # pointers LOAD leaves behind (programs read them to find their own end)
    ram = mem.ram
    ram[0xAE], ram[0xAF] = end & 0xFF, (end >> 8) & 0xFF
    ram[0x2D], ram[0x2E] = end & 0xFF, (end >> 8) & 0xFF  # BASIC variables start

    if entry is None:
        entry = parse_basic_sys(prg)
        if entry is None:
            raise ValueError(
                f"{file_name!r} loads at ${load_addr:04X} with no recognizable "
                "BASIC SYS stub — pass entry= (adapter knowledge) to boot it"
            )

    # ---- start state: as if BASIC just executed SYS <entry> ----
    s = cpu.s
    s.sp = 0xF6
    ret = (EXIT_TRAP - 1) & 0xFFFF
    cpu.push((ret >> 8) & 0xFF)
    cpu.push(ret & 0xFF)
    s.pc = entry & 0xFFFF
    s.a = s.x = s.y = 0
    s.i = 0
    s.d = 0

    if install_hooks:
        registry.install(cpu)

    program = ProgramInfo(
        source=str(image_path),
        file_name=file_name,
        load_addr=load_addr,
        end_addr=end,
        entry=entry & 0xFFFF,
    )
    return Runtime(
        program=program, cpu=cpu, mem=mem, machine=machine,
        boot_args={
            # resolved so snapshots/clones re-open the media from any cwd
            "image_path": str(image_path.resolve()), "file": file, "entry": entry,
            "roms_dir": str(Path(roms_dir).resolve()) if roms_dir else None,
        },
    )


def run_frames(rt: Runtime, frames: int, *, max_instructions: int = 50_000_000) -> None:
    """Step the VM until the VIC has completed ``frames`` more frames."""
    target = rt.machine.vic.frame + frames
    step = rt.cpu.step
    budget = max_instructions
    vic = rt.machine.vic
    while vic.frame < target:
        step()
        budget -= 1
        if budget <= 0:
            raise RuntimeError(
                f"run_frames exceeded {max_instructions} instructions "
                f"({frames} frames requested, at frame {vic.frame}, "
                f"PC=${rt.cpu.s.pc:04X})"
            )


def run_until(rt: Runtime, predicate, *, max_instructions: int = 50_000_000) -> int:
    """Step until ``predicate(rt)`` is true; returns instructions executed."""
    step = rt.cpu.step
    for n in range(max_instructions):
        if predicate(rt):
            return n
        step()
    raise RuntimeError(f"run_until exceeded {max_instructions} instructions")
