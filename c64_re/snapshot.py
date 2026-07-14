"""Full machine freeze/thaw — the determinism substrate.

Mirrors dos_re's snapshot role: pin reproducible starting points, skip slow
bootstraps (Stix's decrunch is ~400 frames), and — critically — give the
differential hook verifier a way to clone a live runtime so the original
ASM and a replacement can be run side by side from an identical state.

Two layers:

- :func:`capture` / :func:`restore` — in-memory state dicts (cheap; used by
  the verifier's runtime cloning on every verified call).
- :func:`write_snapshot` / :func:`load_snapshot` — the same state persisted
  to a file (versioned, zlib-compressed, stdlib-only format).

The captured set is the FULL machine: RAM, color RAM, 6510 port, CPU
architectural state, VIC (registers + raster clock + latched lines are
re-derived), both CIAs, SID (including the OSC3 LFSR — deterministic
entropy is state), KERNAL HLE bookkeeping, pressed keys/joysticks, and the
instruction/cycle counters.  Partial snapshots hide divergence; there is no
narrow mode (dos_re pitfall: narrowing the diff).
"""
from __future__ import annotations

import pickle
import zlib
from pathlib import Path

MAGIC = b"C64RESNAP1"


def capture(rt) -> dict:
    cpu, mem, m = rt.cpu, rt.mem, rt.machine
    k = m.kernal
    return {
        "version": 1,
        "program": {
            "source": rt.program.source,
            "file_name": rt.program.file_name,
            "load_addr": rt.program.load_addr,
            "end_addr": rt.program.end_addr,
            "entry": rt.program.entry,
        },
        "boot_args": dict(rt.boot_args),
        "ram": bytes(mem.ram),
        "color_ram": bytes(mem.color_ram),
        "cpu_port_ddr": mem.cpu_port_ddr,
        "cpu_port_data": mem.cpu_port_data,
        "cpu": rt.cpu.s.as_dict(),
        "nmi_pending": cpu.nmi_pending,
        "instr_count": cpu.instr_count,
        "cycle_count": cpu.cycle_count,
        "vic": m.vic.get_state(),
        "cia1": m.cia1.get_state(),
        "cia2": m.cia2.get_state(),
        "sid": m.sid.get_state(),
        "machine": {
            "pressed": list(m.pressed),
            "joy1": m.joy1,
            "joy2": m.joy2,
            "nmi_level": m._nmi_level,
            "output_channel": m.output_channel,
        },
        "kernal": {
            "last_key_code": k._last_key_code,
            "open_files": {lfn: dict(f) for lfn, f in k._open_files.items()},
            "input_lfn": _input_lfn(k),
        },
    }


def _input_lfn(k):
    if k._input_channel is None:
        return None
    for lfn, f in k._open_files.items():
        if f is k._input_channel:
            return lfn
    return None


def restore(rt, state: dict) -> None:
    if state.get("version") != 1:
        raise ValueError(f"unsupported snapshot version {state.get('version')!r}")
    cpu, mem, m = rt.cpu, rt.mem, rt.machine
    mem.ram[:] = state["ram"]
    mem.color_ram[:] = state["color_ram"]
    mem.cpu_port_ddr = state["cpu_port_ddr"]
    mem.cpu_port_data = state["cpu_port_data"]
    for f, v in state["cpu"].items():
        setattr(cpu.s, f, v)
    cpu.nmi_pending = state["nmi_pending"]
    cpu.instr_count = state["instr_count"]
    cpu.cycle_count = state["cycle_count"]
    m.vic.set_state(state["vic"])
    m.cia1.set_state(state["cia1"])
    m.cia2.set_state(state["cia2"])
    m.sid.set_state(state["sid"])
    ms = state["machine"]
    m.pressed[:] = ms["pressed"]
    m.joy1, m.joy2 = ms["joy1"], ms["joy2"]
    m._nmi_level = ms["nmi_level"]
    m.output_channel = ms["output_channel"]
    k = m.kernal
    ks = state["kernal"]
    k._last_key_code = ks["last_key_code"]
    k._open_files = {lfn: dict(f) for lfn, f in ks["open_files"].items()}
    k._input_channel = (
        k._open_files[ks["input_lfn"]] if ks["input_lfn"] is not None else None
    )


def clone_runtime(rt, *, install_hooks: bool = False):
    """A fresh runtime in the exact state of ``rt``.

    ``install_hooks=False`` (default) yields a pure-ASM oracle clone — the
    verifier's reference side.  The clone shares the immutable disk image
    object but nothing mutable.
    """
    from .runtime import create_runtime

    args = rt.boot_args
    clone = create_runtime(
        args["image_path"],
        file=args["file"],
        entry=args["entry"],
        roms_dir=args["roms_dir"],
        install_hooks=install_hooks,
    )
    restore(clone, capture(rt))
    return clone


def write_snapshot(rt, path: str | Path) -> None:
    blob = MAGIC + zlib.compress(pickle.dumps(capture(rt), protocol=4), 6)
    Path(path).write_bytes(blob)


def load_snapshot(path: str | Path, *, install_hooks: bool = True):
    """Boot a fresh runtime from a snapshot file (media is re-read from the
    original image path recorded at capture time)."""
    from .runtime import create_runtime

    blob = Path(path).read_bytes()
    if not blob.startswith(MAGIC):
        raise ValueError(f"{path} is not a c64_re snapshot")
    state = pickle.loads(zlib.decompress(blob[len(MAGIC):]))
    args = state["boot_args"]
    rt = create_runtime(
        args["image_path"],
        file=args["file"],
        entry=args["entry"],
        roms_dir=args["roms_dir"],
        install_hooks=install_hooks,
    )
    restore(rt, state)
    return rt
