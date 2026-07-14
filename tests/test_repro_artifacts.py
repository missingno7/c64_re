"""Repro-artifact tests: snapshot-directory capture + detached clone (no game assets)."""
from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c64_re.repro_artifacts import (  # noqa: E402
    clone_runtime_state,
    safe_artifact_part,
    write_runtime_repro_snapshot,
)
from c64_re.runtime import create_runtime  # noqa: E402
from c64_re.snapshot import load_snapshot  # noqa: E402

from test_machine import basic_stub, make_d64_with_prg  # noqa: E402


def build_rt(tmp_path):
    # code at $080D: INC $D020 / JMP $080D (runs forever, keeps mutating state)
    code = bytes((0xEE, 0x20, 0xD0, 0x4C, 0x0D, 0x08))
    prg = basic_stub(2061, code)
    disk = make_d64_with_prg(b"T", prg)
    p = tmp_path / "t.d64"
    p.write_bytes(disk.data)
    return create_runtime(p)


def test_safe_artifact_part_normalizes_paths_and_addresses():
    assert safe_artifact_part("crash stix $080D/ValueError") == "crash_stix_080D_ValueError"


def test_write_runtime_repro_snapshot_writes_loadable_snapshot_and_manifest(tmp_path):
    rt = build_rt(tmp_path)
    for _ in range(25):
        rt.cpu.step()
    out = write_runtime_repro_snapshot(
        rt,
        root=tmp_path / "artifacts",
        name="crash stix ValueError",
        status="unit-test crash",
        metadata={"exception_type": "ValueError",
                  "replay_hint": "c64_re.snapshot.load_snapshot(<dir>/snapshot.c64snap)"},
        trace_tail=["INC $D020"],
        timestamp=datetime(2026, 6, 16, 13, 9, 0),
    )

    assert out.name == "crash_stix_ValueError_20260616_130900"
    assert (out / "snapshot.c64snap").exists()
    manifest = json.loads((out / "repro.json").read_text(encoding="utf-8"))
    assert manifest["kind"] == "runtime_snapshot"
    assert manifest["snapshot"] == "snapshot.c64snap"
    assert manifest["cpu_addr"] == f"${rt.cpu.s.pc:04X}"
    assert manifest["steps"] == rt.cpu.instr_count
    assert manifest["trace_tail"] == ["INC $D020"]
    assert manifest["metadata"]["exception_type"] == "ValueError"

    # the snapshot file boots back to the exact captured machine state
    rt2 = load_snapshot(out / "snapshot.c64snap")
    assert rt2.cpu.s.pc == rt.cpu.s.pc
    assert rt2.cpu.instr_count == rt.cpu.instr_count
    assert bytes(rt2.mem.ram) == bytes(rt.mem.ram)


def test_clone_runtime_state_is_detached(tmp_path):
    rt = build_rt(tmp_path)
    for _ in range(10):
        rt.cpu.step()
    clone = clone_runtime_state(rt)
    assert clone.cpu.s.pc == rt.cpu.s.pc
    assert bytes(clone.mem.ram) == bytes(rt.mem.ram)
    assert not clone.cpu.replacement_hooks  # pure-ASM oracle clone by default
    # mutating the live runtime leaves the clone untouched
    rt.cpu.step()
    assert clone.cpu.instr_count != rt.cpu.instr_count
