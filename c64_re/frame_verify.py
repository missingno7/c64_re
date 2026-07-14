"""Reusable frame-level differential verification primitives — the semantic
frame oracle.

The dos_re design, VIC flavor: run two C64 runtimes side-by-side — a
pure-ASM *reference* and a hook-carrying *candidate* — stop both at
caller-provided semantic frame boundaries, compare frame samples, and write
diff artifacts.  This module deliberately does not know any game-specific
address, screen mode, or asset path; those belong in the game adapter.

Boundary model:

- **Default** — the VIC frame counter advancing: each side runs until its
  ``machine.vic.frame`` increments by one.  The VIC clock is driven from
  CPU cycle telemetry, so both sides cross the boundary at the same machine
  time even when the candidate's hooks execute far fewer instructions
  (the hook verifier resyncs cycle counts; see :mod:`c64_re.verification`).
- **``boundary_pcs``** — an adapter-supplied set of program counters, for
  games with an explicit frame handshake (a raster-wait loop head, a
  "frame done" routine).  The boundary is then "PC in the set", checked
  after every step, so both sides stop at the identical instruction.

Sample model: a :class:`FrameSample` carries the rendered indexed frame
(``machine.vic.render_frame()``) plus a named value dict that defaults to
the VIC register file and the border/background colors.  An adapter widens
it with ``sample_builder(rt) -> dict[str, bytes | int]`` (e.g. raw screen /
bitmap / color RAM ranges).  Every key is diffed; the rendered frame diff
reports the first N differing pixels as (x, y, ref_color, cand_color) —
the oracle compares indexed color numbers, never RGB.

Verdict model mirrors dos_re's: a sample divergence is *reported and
dumped* (report.txt + ref/cand/diff PNGs into ``dump_dir``) and ends the
run — ``stop_on_diff`` selects the exit code, not whether the run stops.
Boundary timeouts and internal errors raise :class:`FrameVerifyDivergence`
(fail loud; a side that never reaches its boundary is itself a finding).
"""
from __future__ import annotations

import json
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Mapping

from .pngout import save_frame_png

# Adapter-supplied sample widener: extra named values diffed every frame.
SampleBuilder = Callable[[object], Mapping[str, "bytes | int"]]
# pump_inputs(reference, candidate): deliver live/demo input to BOTH sides.
RuntimePairCallback = Callable[[object, object], None]
StopCallback = Callable[[], bool]
StatusCallback = Callable[[str], None]
# on_divergence(pre_ref_state, pre_cand_state, ref_sample, cand_sample, report)
# — the pre-frame snapshot.capture() dicts, taken BEFORE the diverging frame.
DivergenceCallback = Callable[[dict, dict, "FrameSample", "FrameSample", str], None]


class FrameVerifyDivergence(RuntimeError):
    """Raised when frame verification cannot continue deterministically
    (boundary timeout, internal error).  Ordinary sample differences are
    not exceptions — they are reported, dumped, and end the run."""


@dataclass(frozen=True)
class FrameVerifyConfig:
    """Game-independent frame verifier controls."""

    max_frames: int = 60
    frame_budget: int = 2_000_000      # instructions per boundary, per side
    dump_dir: Path = Path("artifacts/evidence/frame_verify")
    stop_on_diff: bool = True          # exit code on divergence (run ends either way)
    log_every: int = 10
    max_pixel_diffs: int = 8           # first N differing pixels in the report
    render_border: bool = False        # include the PAL-visible border in samples


@dataclass
class FrameSample:
    """One side's state at a semantic frame boundary."""

    side: str                # "reference" | "candidate"
    frame_no: int            # verifier pair counter (1-based)
    kind: str                # "vic_frame" | "boundary_pc"
    pc: int                  # continuation PC at the boundary
    vic_frame: int           # machine.vic.frame at the boundary
    steps_since_start: int   # instructions this side ran for this frame
    width: int
    height: int
    pixels: bytes            # rendered indexed C64 colors (0-15), row-major
    values: dict = field(default_factory=dict)   # name -> bytes | int

    @property
    def pixel_crc(self) -> int:
        return zlib.crc32(self.pixels) & 0xFFFFFFFF


def default_values(rt) -> dict:
    """The game-independent sample values: the VIC register file and the
    border/background colors (convenience duplicates of regs $20/$21)."""
    regs = bytes(rt.machine.vic.regs)
    return {
        "vic_regs": regs,
        "border": regs[0x20] & 0x0F,
        "bg0": regs[0x21] & 0x0F,
    }


def make_frame_sample(rt, *, side: str, frame_no: int, kind: str,
                      start_count: int, config: FrameVerifyConfig,
                      sample_builder: SampleBuilder | None = None) -> FrameSample:
    width, height, indexed = rt.machine.vic.render_frame(border=config.render_border)
    values = default_values(rt)
    if sample_builder is not None:
        values.update(sample_builder(rt))
    return FrameSample(
        side=side,
        frame_no=frame_no,
        kind=kind,
        pc=rt.cpu.s.pc & 0xFFFF,
        vic_frame=rt.machine.vic.frame,
        steps_since_start=rt.cpu.instr_count - start_count,
        width=width,
        height=height,
        pixels=bytes(indexed),
        values=values,
    )


class _BoundaryRunner:
    """Advances one runtime to its next semantic frame boundary."""

    def __init__(self, rt, *, config: FrameVerifyConfig, side: str,
                 boundary_pcs: "set[int] | frozenset[int] | None",
                 sample_builder: SampleBuilder | None) -> None:
        self.rt = rt
        self.config = config
        self.side = side
        self.boundary_pcs = (
            frozenset(pc & 0xFFFF for pc in boundary_pcs) if boundary_pcs else None
        )
        self.sample_builder = sample_builder

    def run_to_boundary(self, frame_no: int) -> FrameSample:
        rt = self.rt
        cpu = rt.cpu
        start = cpu.instr_count
        step = cpu.step
        if self.boundary_pcs is None:
            vic = rt.machine.vic
            target = vic.frame + 1
            for _ in range(self.config.frame_budget):
                step()
                if vic.frame >= target:
                    return make_frame_sample(
                        rt, side=self.side, frame_no=frame_no, kind="vic_frame",
                        start_count=start, config=self.config,
                        sample_builder=self.sample_builder,
                    )
        else:
            pcs = self.boundary_pcs
            s = cpu.s
            for _ in range(self.config.frame_budget):
                step()
                if (s.pc & 0xFFFF) in pcs:
                    return make_frame_sample(
                        rt, side=self.side, frame_no=frame_no, kind="boundary_pc",
                        start_count=start, config=self.config,
                        sample_builder=self.sample_builder,
                    )
        raise FrameVerifyDivergence(
            f"FRAME VERIFY TIMEOUT side={self.side} frame={frame_no} "
            f"budget={self.config.frame_budget} at PC=${cpu.s.pc:04X}"
        )


def run_frame_verifier(
    *,
    reference,
    candidate,
    config: FrameVerifyConfig,
    boundary_pcs: "set[int] | frozenset[int] | None" = None,
    sample_builder: SampleBuilder | None = None,
    pump_inputs: RuntimePairCallback | None = None,
    on_divergence: DivergenceCallback | None = None,
    stop_requested: StopCallback | None = None,
    status_callback: StatusCallback | None = None,
    label: str = "FRAME VERIFY",
) -> int:
    """Run a generic headless frame-boundary verifier.

    The caller supplies two already-initialized runtimes (typically a
    ``snapshot.clone_runtime(rt, install_hooks=False)`` oracle and the live
    hooked runtime), optional boundary PCs, and an optional sample widener.
    Returns 0 (all frames matched, or a stop was requested, or divergence
    with ``stop_on_diff=False``) or 1 (divergence with ``stop_on_diff``).
    """
    ref_runner = _BoundaryRunner(
        reference, config=config, side="reference",
        boundary_pcs=boundary_pcs, sample_builder=sample_builder,
    )
    cand_runner = _BoundaryRunner(
        candidate, config=config, side="candidate",
        boundary_pcs=boundary_pcs, sample_builder=sample_builder,
    )

    frame_no = 1
    while config.max_frames <= 0 or frame_no <= config.max_frames:
        if stop_requested is not None and stop_requested():
            return 0
        # Inputs are sampled only at pair boundaries, before BOTH runtimes
        # advance.  Pumping between the reference and candidate passes would
        # give the candidate a key/joystick event one frame earlier than the
        # oracle — a one-frame input skew producing false divergences
        # (dos_re's hard-won rule, unchanged).
        if pump_inputs is not None:
            pump_inputs(reference, candidate)
        # Capture the pair-start state only when a caller asked for
        # divergence repros: a snapshot from BEFORE the frame that first
        # diverged, not after the candidate already drew the differing frame.
        pre_ref = pre_cand = None
        if on_divergence is not None:
            from .snapshot import capture
            pre_ref = capture(reference)
            pre_cand = capture(candidate)

        ref_sample = ref_runner.run_to_boundary(frame_no)
        cand_sample = cand_runner.run_to_boundary(frame_no)

        report = compare_samples(ref_sample, cand_sample, config, label=label)
        if report is not None:
            dump_divergence(ref_sample, cand_sample, report, config, label=label)
            print(report, flush=True)
            if status_callback is not None:
                status_callback(f"{label} divergence at frame {frame_no}")
            if on_divergence is not None:
                on_divergence(pre_ref, pre_cand, ref_sample, cand_sample, report)
            return 1 if config.stop_on_diff else 0

        if config.log_every and (frame_no == 1 or frame_no % config.log_every == 0):
            msg = (
                f"{label} ok frame={frame_no} boundary={ref_sample.kind} "
                f"pixels={ref_sample.pixel_crc:08X}"
            )
            print(msg, flush=True)
            if status_callback is not None:
                status_callback(msg)
        frame_no += 1

    print(f"{label} OK frames={config.max_frames}", flush=True)
    if status_callback is not None:
        status_callback(f"{label} OK frames={config.max_frames}")
    return 0


# ---- comparison ---------------------------------------------------------------
def compare_samples(ref: FrameSample, cand: FrameSample, config: FrameVerifyConfig,
                    *, label: str = "FRAME VERIFY") -> "str | None":
    """Diff two boundary samples; returns a human-readable report or None."""
    sections: list[str] = []
    if (ref.width, ref.height) != (cand.width, cand.height):
        sections.append(
            "Frame geometry differences:\n"
            f"  REF:  {ref.width}x{ref.height}\n"
            f"  HOOK: {cand.width}x{cand.height}"
        )
    if ref.kind != cand.kind:
        sections.append(
            "Boundary differences:\n"
            f"  REF:  {ref.kind} at ${ref.pc:04X}\n"
            f"  HOOK: {cand.kind} at ${cand.pc:04X}"
        )
    elif ref.kind == "boundary_pc" and ref.pc != cand.pc:
        # In vic_frame mode the sides legitimately stop at different PCs
        # (hooks compress instruction counts); at an explicit handshake PC
        # they must stop at the identical instruction.
        sections.append(
            f"Boundary PC differences:\n  REF:  ${ref.pc:04X}\n  HOOK: ${cand.pc:04X}"
        )
    if ref.pixels != cand.pixels and (ref.width, ref.height) == (cand.width, cand.height):
        diffs = pixel_diffs(ref.pixels, cand.pixels, ref.width,
                            limit=config.max_pixel_diffs)
        total = byte_diff_count(ref.pixels, cand.pixels)
        lines = [
            "Rendered frame differences:",
            f"  REF crc:  {ref.pixel_crc:08X}",
            f"  HOOK crc: {cand.pixel_crc:08X}",
            f"  differing pixels: {total}",
        ]
        for x, y, rc, cc in diffs:
            lines.append(f"  x={x} y={y} ref_color={rc} cand_color={cc}")
        if total > len(diffs):
            lines.append(f"  ... first {len(diffs)} shown")
        sections.append("\n".join(lines))
    elif ref.pixels != cand.pixels:
        sections.append("Rendered frame differences: (geometry differs, no pixel map)")

    for name in sorted(set(ref.values) | set(cand.values)):
        if name not in ref.values or name not in cand.values:
            missing = "REF" if name not in ref.values else "HOOK"
            sections.append(f"Sample value {name!r}: missing on {missing} side")
            continue
        rv, cv = ref.values[name], cand.values[name]
        if rv == cv:
            continue
        if isinstance(rv, (bytes, bytearray)) and isinstance(cv, (bytes, bytearray)):
            idx = first_diff(bytes(rv), bytes(cv))
            sections.append(
                f"Sample value {name!r} differences:\n"
                f"  REF crc:  {zlib.crc32(bytes(rv)) & 0xFFFFFFFF:08X} len={len(rv)}\n"
                f"  HOOK crc: {zlib.crc32(bytes(cv)) & 0xFFFFFFFF:08X} len={len(cv)}\n"
                f"  first differing byte: {idx}"
            )
        else:
            sections.append(
                f"Sample value {name!r} differences:\n  REF:  {rv!r}\n  HOOK: {cv!r}"
            )

    if not sections:
        return None
    return (
        f"{label} DIVERGENCE\n"
        f"frame: {ref.frame_no}\n"
        f"boundary: {ref.kind}\n"
        f"REF continuation:  ${ref.pc:04X} steps={ref.steps_since_start} "
        f"vic_frame={ref.vic_frame}\n"
        f"HOOK continuation: ${cand.pc:04X} steps={cand.steps_since_start} "
        f"vic_frame={cand.vic_frame}\n"
        + "\n\n".join(sections)
    )


def pixel_diffs(a: bytes, b: bytes, width: int, *,
                limit: int = 8) -> list[tuple[int, int, int, int]]:
    """The first ``limit`` differing pixels as (x, y, ref_color, cand_color)."""
    out: list[tuple[int, int, int, int]] = []
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            y, x = divmod(i, width) if width else (-1, -1)
            out.append((x, y, a[i], b[i]))
            if len(out) >= limit:
                break
    return out


def first_diff(a: bytes, b: bytes) -> int:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return -1


def byte_diff_count(a: bytes, b: bytes) -> int:
    n = min(len(a), len(b))
    count = sum(1 for i in range(n) if a[i] != b[i])
    return count + abs(len(a) - len(b))


# ---- artifacts ----------------------------------------------------------------
def dump_divergence(ref: FrameSample, cand: FrameSample, report: str,
                    config: FrameVerifyConfig, *, label: str = "FRAME VERIFY") -> None:
    """Write the divergence evidence: report.txt + json meta + ref/cand/diff
    PNGs (via :func:`c64_re.pngout.save_frame_png`) into ``dump_dir``."""
    out = config.dump_dir
    out.mkdir(parents=True, exist_ok=True)
    stem = f"frame_{ref.frame_no:05d}"
    (out / f"{stem}_report.txt").write_text(report + "\n", encoding="utf-8")
    meta = {
        "frame": ref.frame_no,
        "reference": sample_meta(ref),
        "candidate": sample_meta(cand),
    }
    (out / f"{stem}_report.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    save_frame_png(out / f"{stem}_ref.png", (ref.width, ref.height, ref.pixels))
    save_frame_png(out / f"{stem}_cand.png", (cand.width, cand.height, cand.pixels))
    if (ref.width, ref.height) == (cand.width, cand.height):
        diff = bytearray(len(ref.pixels))
        for i in range(min(len(ref.pixels), len(cand.pixels))):
            if ref.pixels[i] != cand.pixels[i]:
                diff[i] = 1  # white on black
        save_frame_png(out / f"{stem}_diff.png", (ref.width, ref.height, bytes(diff)))
    print(f"{label} artifacts written to {out / stem}_*", flush=True)


def sample_meta(sample: FrameSample) -> dict:
    """A JSON-safe summary of one sample (bytes values become crc + length)."""
    values: dict = {}
    for name, v in sample.values.items():
        if isinstance(v, (bytes, bytearray)):
            values[name] = {"crc": f"{zlib.crc32(bytes(v)) & 0xFFFFFFFF:08X}",
                            "len": len(v)}
        else:
            values[name] = v
    return {
        "side": sample.side,
        "frame_no": sample.frame_no,
        "kind": sample.kind,
        "pc": f"${sample.pc:04X}",
        "vic_frame": sample.vic_frame,
        "steps_since_start": sample.steps_since_start,
        "width": sample.width,
        "height": sample.height,
        "pixel_crc": f"{sample.pixel_crc:08X}",
        "values": values,
    }
