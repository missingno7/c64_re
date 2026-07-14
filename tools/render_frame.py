#!/usr/bin/env python3
"""Render the C64_RE emulator's VIC frame to a dependency-free PNG dump.

The day-0 "see output" tool: point it at a snapshot (or a fresh program
image) and get a PNG of what the emulated screen shows.  It stays
standard-library-only for headless evidence inspection — the VIC does the
decoding (:meth:`c64_re.vic.VIC.render_frame`), :mod:`c64_re.pngout` writes
the file.

Usage:
    python tools/render_frame.py game.d64 [--file NAME] [--frames 400] [--out frame.png]
    python tools/render_frame.py state.c64snap [--border] [--scale 2]

A ``.d64``/``.prg`` is booted fresh and run ``--frames`` VIC frames first; a
``.c64snap`` snapshot is resumed as-is (and one frame is rendered from its
restored state).

Origin: ported from dos_re's ``tools/render_frame.py`` — the VGA/EGA decode
paths are replaced by the VIC's own renderer (the C64 has one video chip and
the VM already models it), ``--seg``/``--steps`` become ``--border``/
``--frames``, and snapshots are single ``.c64snap`` files, not directories.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from c64_re.pngout import save_frame_png  # noqa: E402


def boot_image(path: Path, file: str, frames: int):
    """Boot a .d64/.prg fresh (running ``frames`` VIC frames), or resume a
    .c64snap snapshot as-is."""
    from c64_re.runtime import create_runtime, run_frames
    from c64_re.snapshot import load_snapshot

    if path.suffix.lower() == ".c64snap":
        return load_snapshot(path)
    rt = create_runtime(path, file=file)
    if frames:
        run_frames(rt, frames)
    return rt


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Render the emulated VIC frame to PNG")
    p.add_argument("image", help="program image (.d64/.prg) or snapshot (.c64snap)")
    p.add_argument("--file", default="*", help="file name on the D64 (default: first PRG)")
    p.add_argument("--frames", type=int, default=0,
                   help="run a fresh boot this many VIC frames before rendering")
    p.add_argument("--border", action="store_true", help="include the PAL-visible border")
    p.add_argument("--scale", type=int, default=2, help="integer upscale (default 2)")
    p.add_argument("--out", default=None, help="output PNG path (default: <image>_frame.png in cwd)")
    args = p.parse_args(argv)

    image = Path(args.image)
    rt = boot_image(image, args.file, args.frames)
    frame = rt.machine.vic.render_frame(border=args.border)
    out = Path(args.out) if args.out else Path(f"{image.stem}_frame.png")
    save_frame_png(out, frame, scale=args.scale)
    width, height, _ = frame
    print(
        f"wrote {out} ({width * args.scale}x{height * args.scale}, "
        f"frame {rt.machine.vic.frame}, PC=${rt.cpu.s.pc:04X})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
