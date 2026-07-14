#!/usr/bin/env python3
"""Watch the oracle run — run any program image in the interactive viewer.

A thin wrapper over :class:`c64_re.player.Viewer`: boot any .d64/.prg with
the generic runtime (or resume a .c64snap) and open the pygame window.  A
real port's play script subclasses/extends the player instead; this tool
needs no game adapter at all.

Usage:
    python tools/view.py game.d64 [--file NAME] [--frames N]
                         [--snapshot state.c64snap] [--scale 3] [--joy-port 2]

--frames N runs the boot headless for N VIC frames before the window opens
(skip a slow decruncher).  --snapshot resumes a saved .c64snap instead of a
fresh boot.  The viewer needs numpy + pygame (the framework core does not).

Origin: ported from dos_re's ``tools/view.py`` (there a shim over
``dos_re.player.GameFrontend``; here a direct ``Viewer`` construction —
c64_re's standard frontend CLI does not exist yet).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Run any program image in the interactive viewer")
    p.add_argument("image", nargs="?", help="program image (.d64/.prg)")
    p.add_argument("--file", default="*", help="file name on the D64 (default: first PRG)")
    p.add_argument("--frames", type=int, default=0,
                   help="run the boot headless this many VIC frames before opening the window")
    p.add_argument("--snapshot", default=None, help="resume a .c64snap instead of a fresh boot")
    p.add_argument("--scale", type=int, default=3, help="window pixel scale (default 3)")
    p.add_argument("--joy-port", type=int, default=2, choices=(1, 2),
                   help="joystick port the keyboard drives (default 2)")
    args = p.parse_args(argv)

    from c64_re.player import Viewer
    from c64_re.runtime import create_runtime, run_frames
    from c64_re.snapshot import load_snapshot

    if args.snapshot:
        rt = load_snapshot(args.snapshot)
        title = Path(args.snapshot).name
    elif args.image:
        rt = create_runtime(args.image, file=args.file)
        if args.frames:
            run_frames(rt, args.frames)
        title = rt.program.file_name
    else:
        p.error("provide a program image or --snapshot")

    Viewer(rt, scale=args.scale, joy_port=args.joy_port, title=title).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
