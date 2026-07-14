"""Tests for c64_re.frame_verify — the semantic frame oracle (no game assets)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c64_re.frame_verify import (  # noqa: E402
    FrameSample,
    FrameVerifyConfig,
    FrameVerifyDivergence,
    compare_samples,
    dump_divergence,
    run_frame_verifier,
)
from c64_re.runtime import create_runtime  # noqa: E402

from test_machine import basic_stub, make_d64_with_prg  # noqa: E402


# ---- helpers -----------------------------------------------------------------------
# SEI; JMP $080E — an interrupt-free busy loop; VIC frames advance on cycle telemetry.
LOOP = bytes((0x78, 0x4C, 0x0E, 0x08))
LOOP_PC = 0x080E


def build_rt(tmp_path, code: bytes = LOOP, name: str = "t.d64"):
    prg = basic_stub(2061, code)  # code lands at $080D
    disk = make_d64_with_prg(b"T", prg)
    p = tmp_path / name
    if not p.exists():
        p.write_bytes(disk.data)
    return create_runtime(p)


def _sample(*, side: str, frame_no: int = 1, pixels: bytes, width: int = 4,
            height: int = 2, values: dict | None = None, kind: str = "vic_frame",
            pc: int = 0x080E) -> FrameSample:
    return FrameSample(
        side=side,
        frame_no=frame_no,
        kind=kind,
        pc=pc,
        vic_frame=frame_no,
        steps_since_start=123,
        width=width,
        height=height,
        pixels=pixels,
        values=dict(values or {}),
    )


# ---- compare_samples ----------------------------------------------------------------
def test_compare_samples_equal_is_none():
    px = bytes(range(8))
    ref = _sample(side="reference", pixels=px, values={"border": 14, "vic_regs": b"\x00" * 47})
    cand = _sample(side="candidate", pixels=px, values={"border": 14, "vic_regs": b"\x00" * 47})
    config = FrameVerifyConfig()
    assert compare_samples(ref, cand, config) is None


def test_compare_samples_reports_first_pixels_with_coordinates():
    ref_px = bytearray(8)
    cand_px = bytearray(8)
    cand_px[5] = 7          # width 4 -> pixel 5 is x=1 y=1
    cand_px[6] = 2
    ref = _sample(side="reference", pixels=bytes(ref_px))
    cand = _sample(side="candidate", pixels=bytes(cand_px))
    report = compare_samples(ref, cand, FrameVerifyConfig(max_pixel_diffs=1))
    assert report is not None
    assert "x=1 y=1 ref_color=0 cand_color=7" in report
    assert "differing pixels: 2" in report
    assert "first 1 shown" in report          # max_pixel_diffs honored
    assert "x=2" not in report


def test_compare_samples_diffs_every_value_key():
    px = bytes(8)
    ref = _sample(side="reference", pixels=px,
                  values={"vic_regs": bytes(47), "border": 14, "only_ref": 1})
    cand_regs = bytearray(47)
    cand_regs[0x21] = 0x05
    cand = _sample(side="candidate", pixels=px,
                   values={"vic_regs": bytes(cand_regs), "border": 2})
    report = compare_samples(ref, cand, FrameVerifyConfig())
    assert report is not None
    assert "'vic_regs'" in report and "first differing byte: 33" in report
    assert "'border'" in report and "REF:  14" in report and "HOOK: 2" in report
    assert "'only_ref'" in report and "missing on HOOK side" in report


def test_compare_samples_boundary_pc_must_match_in_pc_mode():
    px = bytes(8)
    ref = _sample(side="reference", pixels=px, kind="boundary_pc", pc=0x1234)
    cand = _sample(side="candidate", pixels=px, kind="boundary_pc", pc=0x1301)
    report = compare_samples(ref, cand, FrameVerifyConfig())
    assert report is not None and "$1234" in report and "$1301" in report
    # ...but differing continuation PCs are legitimate in vic_frame mode
    ref2 = _sample(side="reference", pixels=px, kind="vic_frame", pc=0x1234)
    cand2 = _sample(side="candidate", pixels=px, kind="vic_frame", pc=0x1301)
    assert compare_samples(ref2, cand2, FrameVerifyConfig()) is None


# ---- dump_divergence ----------------------------------------------------------------
def test_dump_divergence_writes_pngs_and_report(tmp_path):
    ref = _sample(side="reference", frame_no=7, pixels=bytes([1] * 8))
    cand = _sample(side="candidate", frame_no=7, pixels=bytes([2] * 8))
    config = FrameVerifyConfig(dump_dir=tmp_path)
    dump_divergence(ref, cand, "FRAME VERIFY DIVERGENCE\nframe: 7", config)
    for suffix in ("report.txt", "report.json", "ref.png", "cand.png", "diff.png"):
        f = tmp_path / f"frame_00007_{suffix}"
        assert f.exists() and f.stat().st_size > 0, f


# ---- run_frame_verifier on real runtimes ---------------------------------------------
def test_identical_runtimes_verify_ok_and_inputs_pump_at_pair_boundaries(tmp_path):
    """Two clones of the same PRG must verify green — and live input must be
    sampled only at pair boundaries, before BOTH runtimes advance.  Pumping
    between the reference and candidate passes would hand the candidate an
    event one frame earlier than the oracle (dos_re's one-frame-skew bug)."""
    ref = build_rt(tmp_path)
    cand = build_rt(tmp_path)
    pump_calls: list[int] = []

    def pump_inputs(ref_rt, cand_rt):
        # pair-boundary property: neither side has advanced past the other
        assert ref_rt.machine.vic.frame == cand_rt.machine.vic.frame
        mask = len(pump_calls) + 1
        pump_calls.append(mask)
        ref_rt.machine.set_joy2(mask)
        cand_rt.machine.set_joy2(mask)

    result = run_frame_verifier(
        reference=ref,
        candidate=cand,
        config=FrameVerifyConfig(max_frames=3, dump_dir=tmp_path, log_every=0),
        pump_inputs=pump_inputs,
    )
    assert result == 0
    assert pump_calls == [1, 2, 3]
    assert ref.machine.joy2 == cand.machine.joy2 == 3
    assert ref.machine.vic.frame == cand.machine.vic.frame == 3


def test_divergence_detected_dumped_and_pre_frame_state_captured(tmp_path):
    """A candidate whose background color register differs must diverge on
    frame 1, dump artifacts, and hand on_divergence the PRE-frame states."""
    ref = build_rt(tmp_path)
    cand = build_rt(tmp_path)
    cand.machine.vic.write(0x21, 0x00)   # background black instead of blue
    dump_dir = tmp_path / "evidence"
    seen: list[tuple] = []

    def on_divergence(pre_ref, pre_cand, ref_sample, cand_sample, report):
        seen.append((pre_ref, pre_cand, ref_sample, cand_sample, report))

    result = run_frame_verifier(
        reference=ref,
        candidate=cand,
        config=FrameVerifyConfig(max_frames=5, dump_dir=dump_dir, log_every=0),
        on_divergence=on_divergence,
    )
    assert result == 1
    assert len(seen) == 1
    pre_ref, pre_cand, ref_sample, cand_sample, report = seen[0]
    assert "DIVERGENCE" in report and "ref_color=" in report
    assert ref_sample.frame_no == cand_sample.frame_no == 1
    assert ref_sample.values["bg0"] == 6 and cand_sample.values["bg0"] == 0
    # the pre-frame snapshots already show the planted difference (register
    # file), captured BEFORE the diverging frame was drawn
    assert pre_ref["vic"]["regs"][0x21] != pre_cand["vic"]["regs"][0x21]
    for suffix in ("report.txt", "ref.png", "cand.png", "diff.png"):
        assert (dump_dir / f"frame_00001_{suffix}").exists()


def test_stop_on_diff_false_still_ends_run_with_exit_zero(tmp_path):
    ref = build_rt(tmp_path)
    cand = build_rt(tmp_path)
    cand.machine.vic.write(0x20, 0x00)   # border differs (vic_regs + 'border' value)
    result = run_frame_verifier(
        reference=ref,
        candidate=cand,
        config=FrameVerifyConfig(max_frames=5, dump_dir=tmp_path / "e2",
                                 stop_on_diff=False, log_every=0),
    )
    assert result == 0
    assert (tmp_path / "e2" / "frame_00001_report.txt").exists()


def test_boundary_pcs_mode_stops_at_the_handshake_pc(tmp_path):
    """With adapter-declared boundary PCs the frame is the PC handshake, not
    the VIC counter: three 'frames' of a two-instruction loop complete long
    before a single VIC frame elapses."""
    ref = build_rt(tmp_path)
    cand = build_rt(tmp_path)
    result = run_frame_verifier(
        reference=ref,
        candidate=cand,
        config=FrameVerifyConfig(max_frames=3, dump_dir=tmp_path, log_every=0),
        boundary_pcs={LOOP_PC},
    )
    assert result == 0
    assert ref.machine.vic.frame == 0 and cand.machine.vic.frame == 0
    assert ref.cpu.s.pc == cand.cpu.s.pc == LOOP_PC


def test_boundary_timeout_fails_loud(tmp_path):
    ref = build_rt(tmp_path)
    cand = build_rt(tmp_path)
    with pytest.raises(FrameVerifyDivergence, match="TIMEOUT"):
        run_frame_verifier(
            reference=ref,
            candidate=cand,
            config=FrameVerifyConfig(max_frames=1, frame_budget=50,
                                     dump_dir=tmp_path, log_every=0),
            boundary_pcs={0x4000},   # never reached by the loop
        )


def test_sample_builder_widens_the_diff(tmp_path):
    """An adapter sample_builder adds raw memory ranges to the per-frame diff;
    a difference invisible in the rendered frame is still caught."""
    ref = build_rt(tmp_path)
    cand = build_rt(tmp_path)
    cand.mem.ram[0xC000] = 0x5A          # touches nothing the VIC renders

    def widen(rt):
        return {"scratch": bytes(rt.mem.ram[0xC000:0xC100])}

    result = run_frame_verifier(
        reference=ref,
        candidate=cand,
        config=FrameVerifyConfig(max_frames=2, dump_dir=tmp_path / "e3", log_every=0),
        sample_builder=widen,
    )
    assert result == 1
    report = (tmp_path / "e3" / "frame_00001_report.txt").read_text(encoding="utf-8")
    assert "'scratch'" in report and "first differing byte: 0" in report
