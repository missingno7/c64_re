"""The standard play runner — dos_re's ``player.py`` role for C64 ports.

The game-agnostic core of every port's ``scripts/play.py``: a
:class:`GameFrontend` the adapter subclasses, the STANDARD unified CLI
(identical across every port — the human's muscle memory and the prompts'
instructions depend on it), the live pygame viewer, and headless demo
replay.

CLI surface:
  --headless                 no window; requires --frames or --play-demo
  --frames N                 run/boot this many frames
  --snapshot PATH            resume a .c64snap instead of booting
  --save-snapshot PATH       write a snapshot at exit (or hotkey F12)
  --record-demo NAME         arm demo recording (F11 toggles start/stop)
  --play-demo PATH           replay a recorded demo (viewer or headless)
  --demo-continue            keep running after a replayed demo ends
  --no-replacements          ORACLE mode: original ASM only, no hooks
  --verify-hooks             route every hook through the differential oracle
  --trace-hooks              count hook fires, print a summary each second
  --joy-port {1,2}, --scale, --fps, --file, --start-args ...

Viewer hotkeys (the template convention): F10 screenshot, F11 demo-record
toggle, F12 snapshot, F9 pause.  Arrows + Right-Ctrl = joystick; the C64
keyboard is mapped 1:1 for letters/digits and common specials.

Determinism contract: host input is applied only at frame boundaries,
through the same machine API demos use, and recording captures exactly
what was applied — so a recorded demo replays bit-identically.  The wall
clock paces presentation only.  numpy/pygame imports stay lazy: importing
this module and running headless replay needs neither.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

from .runtime import Runtime, create_runtime, run_frames
from .snapshot import load_snapshot, write_snapshot


class GameFrontend:
    """Subclass in the port's scripts/play.py; override what the game needs."""

    name = "c64_re"
    default_image: str | None = None   # e.g. "assets/Game.d64"
    default_file: str = "*"
    default_joy_port = 2               # most C64 games use port 2; Stix uses 1

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def add_arguments(self, ap: argparse.ArgumentParser) -> None:
        """Extra game-specific flags (never rename the standard set)."""

    def create_runtime(self, args) -> Runtime:
        image = args.image or self.default_image
        if not image:
            raise SystemExit("no image: pass a .d64/.prg or set default_image")
        return create_runtime(self.root / image if not Path(image).is_absolute()
                              else image,
                              file=args.file,
                              install_hooks=not args.no_replacements)

    def boot_to_start(self, rt: Runtime, args) -> None:
        """Navigate menus/trainers to the interesting state (fresh boots
        only; snapshot/demo resumes skip this)."""

    def demo_metadata(self, rt: Runtime, args) -> dict:
        return {"frontend": self.name}

    def is_input_wait(self, rt: Runtime) -> bool:
        """True when the game is parked in a fine-grained input poll —
        demos then deliver one event per frame (adapter wires its
        input-wait registry here)."""
        return False


# ---- CLI --------------------------------------------------------------------------
def build_parser(frontend: GameFrontend, description: str | None) -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=description or frontend.name)
    ap.add_argument("image", nargs="?", default=None,
                    help=".d64/.prg (default: the frontend's)")
    ap.add_argument("--file", default=frontend.default_file)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--snapshot", default="", help="resume a .c64snap")
    ap.add_argument("--save-snapshot", default="", help="write snapshot at exit")
    ap.add_argument("--record-demo", action="store_true",
                    help="record a cold-start input demo: the whole session "
                         "from power-on (you drive the menus; starts "
                         "immediately, F11 stops/restarts, close to save). "
                         "With --snapshot it records snapshot-anchored instead.")
    ap.add_argument("--demo-name", default="", metavar="NAME",
                    help="name for the recorded demo (default: the game name)")
    ap.add_argument("--play-demo", default="", metavar="PATH")
    ap.add_argument("--demo-continue", action="store_true")
    ap.add_argument("--no-replacements", action="store_true",
                    help="ORACLE mode: pure original ASM, no hooks")
    ap.add_argument("--verify-hooks", action="store_true")
    ap.add_argument("--trace-hooks", action="store_true")
    ap.add_argument("--joy-port", type=int, choices=(1, 2),
                    default=frontend.default_joy_port)
    ap.add_argument("--scale", type=int, default=3)
    ap.add_argument("--fps", type=float, default=50.125)
    frontend.add_arguments(ap)
    return ap


def main(frontend: GameFrontend, argv=None, *, description: str | None = None) -> int:
    args = build_parser(frontend, description).parse_args(argv)

    playback = None
    if args.play_demo:
        from .input_demo import InputDemoPlayback
        playback = InputDemoPlayback.load(args.play_demo)
        rt = playback.make_runtime(install_hooks=not args.no_replacements)
    elif args.snapshot:
        rt = load_snapshot(args.snapshot, install_hooks=not args.no_replacements)
    else:
        rt = frontend.create_runtime(args)
        # --record-demo captures the whole session from power-on, so the
        # frontend's scripted menu navigation is skipped — the human drives it
        # and it lands in the demo.  Otherwise navigate to the play state.
        if not args.record_demo:
            frontend.boot_to_start(rt, args)

    if args.verify_hooks:
        from .verification import install_live_verifier
        install_live_verifier(rt)
    trace_stats = None
    if args.trace_hooks:
        trace_stats = _install_hook_tracing(rt)

    recorder = None
    if args.record_demo and args.headless:
        raise SystemExit("--record-demo needs the viewer (drop --headless): "
                         "there is no interactive input source in headless mode")
    if args.record_demo:
        from .input_demo import InputDemoRecorder
        name = args.demo_name or frontend.name.lower().replace(" ", "_")
        recorder = InputDemoRecorder(
            root=frontend.root / "artifacts" / "demos",
            name=name,
            metadata=frontend.demo_metadata(rt, args),
        )
        # Auto-start: recording begins right away, no hidden F11 step.  Plain
        # --record-demo is a fresh power-on runtime (frame 0) -> a cold-start
        # demo; --record-demo --snapshot resumes mid-session -> anchored.
        # F11 stops (and can start a fresh take).
        fresh = rt.machine.vic.frame == 0 and rt.cpu.instr_count == 0
        recorder.start(rt, write_start_snapshot=not fresh)
        kind = "cold-start" if fresh else "snapshot-anchored"
        print(f"[REC] recording {kind} demo {name!r} -> {recorder.demo_dir.name}\n"
              f"      drive the game; F11 stops (or just close the window to save)")

    if args.headless:
        rc = _run_headless(frontend, rt, args, playback, trace_stats)
    else:
        rc = _run_viewer(frontend, rt, args, playback, recorder, trace_stats)

    if args.save_snapshot:
        write_snapshot(rt, args.save_snapshot)
        print(f"snapshot -> {args.save_snapshot}")
    return rc


def _install_hook_tracing(rt):
    from .gaps import HookTraceStats
    stats = HookTraceStats()
    for pc, fn in list(rt.cpu.replacement_hooks.items()):
        name = rt.cpu.hook_names.get(pc, f"hook_{pc:04X}")

        def traced(cpu, _fn=fn, _name=name):
            stats.bump(_name)
            _fn(cpu)
        rt.cpu.replacement_hooks[pc] = traced
    return stats


def _run_headless(frontend, rt, args, playback, trace_stats) -> int:
    if playback is None and args.frames <= 0:
        raise SystemExit("--headless needs --frames and/or --play-demo")
    frames_left = args.frames if args.frames > 0 else None
    vic = rt.machine.vic
    while True:
        boundary = vic.frame
        if playback is not None:
            playback.apply_to_runtime(boundary, rt,
                                      single=frontend.is_input_wait(rt))
            if playback.finished(boundary):
                print(f"demo finished at frame {boundary} "
                      f"({playback.next_event_index} events applied)")
                if not args.demo_continue:
                    break
                playback = None
        run_frames(rt, 1)
        if frames_left is not None:
            frames_left -= 1
            if frames_left <= 0:
                break
    if trace_stats is not None:
        print("hook fires:", trace_stats.summary())
    return 0


def _build_keymap():
    import pygame

    keymap = {
        pygame.K_RETURN: "RETURN", pygame.K_BACKSPACE: "INS/DEL",
        pygame.K_SPACE: "SPACE", pygame.K_ESCAPE: "RUN/STOP",
        pygame.K_LSHIFT: "LSHIFT", pygame.K_RSHIFT: "RSHIFT",
        pygame.K_LCTRL: "CTRL", pygame.K_TAB: "CBM",
        pygame.K_F1: "F1", pygame.K_F3: "F3", pygame.K_F5: "F5", pygame.K_F7: "F7",
        pygame.K_HOME: "HOME",
        pygame.K_COMMA: ",", pygame.K_PERIOD: ".", pygame.K_SLASH: "/",
        pygame.K_SEMICOLON: ";", pygame.K_MINUS: "-", pygame.K_EQUALS: "=",
    }
    for i in range(26):
        keymap[pygame.K_a + i] = chr(ord("A") + i)
    for i in range(10):
        keymap[pygame.K_0 + i] = str(i)
    return keymap


def _run_viewer(frontend, rt, args, playback, recorder, trace_stats) -> int:
    import numpy as np
    import pygame

    from .cia import JOY_DOWN, JOY_FIRE, JOY_LEFT, JOY_RIGHT, JOY_UP
    from .vic import DISPLAY_LINES, DISPLAY_WIDTH, PALETTE

    pygame.init()
    keymap = _build_keymap()
    joykeys = {
        pygame.K_UP: JOY_UP, pygame.K_DOWN: JOY_DOWN,
        pygame.K_LEFT: JOY_LEFT, pygame.K_RIGHT: JOY_RIGHT,
        pygame.K_RCTRL: JOY_FIRE,
    }
    vic = rt.machine.vic
    machine = rt.machine
    w, h = DISPLAY_WIDTH + 64, DISPLAY_LINES + 72
    screen = pygame.display.set_mode((w * args.scale, h * args.scale))
    pygame.display.set_caption(f"{frontend.name} — c64_re oracle VM")
    lut = np.array(PALETTE, dtype=np.uint8)
    clock = pygame.time.Clock()
    artifacts = frontend.root / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)

    set_joy = machine.set_joy1 if args.joy_port == 1 else machine.set_joy2
    joy_bits = 0
    paused = False
    shots = snaps = 0
    behind = 0.0
    replay_active = playback is not None
    running = True

    def record_key(kind: str, key: str) -> None:
        if recorder is not None and recorder.active:
            fn = (recorder.record_key_down if kind == "key_down"
                  else recorder.record_key_up)
            fn(boundary=vic.frame, key=key)

    while running:
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                running = False
            elif ev.type in (pygame.KEYDOWN, pygame.KEYUP):
                down = ev.type == pygame.KEYDOWN
                if ev.key == pygame.K_F10 and down:
                    from .pngout import save_frame_png
                    shots += 1
                    path = artifacts / f"screenshot_{shots:03d}.png"
                    save_frame_png(path, vic.render_frame(border=True))
                    print(f"screenshot -> {path}")
                elif ev.key == pygame.K_F11 and down:
                    if recorder is None:
                        print("no --record-demo NAME armed")
                    elif not recorder.active:
                        out = recorder.start(rt)
                        print(f"demo recording -> {out}")
                    else:
                        out = recorder.stop(boundary=vic.frame)
                        print(f"demo saved -> {out} ({recorder.event_count} events)")
                elif ev.key == pygame.K_F12 and down:
                    snaps += 1
                    path = artifacts / f"snapshot_{snaps:03d}.c64snap"
                    write_snapshot(rt, path)
                    print(f"snapshot -> {path}")
                elif ev.key == pygame.K_F9 and down:
                    paused = not paused
                elif replay_active:
                    pass  # host input is ignored while a demo drives the run
                elif ev.key in joykeys:
                    if down:
                        joy_bits |= joykeys[ev.key]
                    else:
                        joy_bits &= ~joykeys[ev.key]
                    set_joy(joy_bits)
                    if recorder is not None and recorder.active:
                        recorder.record_joy(boundary=vic.frame,
                                            port=args.joy_port, mask=joy_bits)
                elif ev.key in keymap:
                    key = keymap[ev.key]
                    if down:
                        machine.key_down(key)
                        record_key("key_down", key)
                    else:
                        machine.key_up(key)
                        record_key("key_up", key)

        if not paused:
            boundary = vic.frame
            if replay_active:
                playback.apply_to_runtime(boundary, rt,
                                          single=frontend.is_input_wait(rt))
                if playback.finished(boundary):
                    print(f"demo finished at frame {boundary}")
                    replay_active = False
                    if not args.demo_continue:
                        running = False
            t0 = time.perf_counter()
            run_frames(rt, 1)
            emu_ms = (time.perf_counter() - t0) * 1000
            behind = max(0.0, behind + emu_ms - 1000.0 / args.fps)
            if behind < 1000.0 / args.fps:
                width, height, indexed = vic.render_frame(border=True)
                arr = np.frombuffer(bytes(indexed), dtype=np.uint8)
                rgb = lut[arr.reshape(height, width)]
                surf = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
                pygame.transform.scale(surf, screen.get_size(), screen)
                pygame.display.flip()
            else:
                behind = 0.0  # drop this render, never emulation frames
        clock.tick(args.fps)

    if recorder is not None and recorder.active:
        out = recorder.stop(boundary=vic.frame)
        print(f"demo saved -> {out} ({recorder.event_count} events)")
    if trace_stats is not None:
        print("hook fires:", trace_stats.summary())
    pygame.quit()
    return 0


# ---- back-compat: the simple embeddable viewer used by early scripts --------------
class Viewer:
    """Minimal viewer wrapper (pre-GameFrontend API, kept for tools/view.py
    and early port scripts).  New ports should use :func:`main` with a
    :class:`GameFrontend` instead."""

    def __init__(self, rt: Runtime, *, scale: int = 3, joy_port: int = 1,
                 fps: float = 50.125, title: str = "c64_re",
                 screenshot_dir: str | Path = ".") -> None:
        self.rt = rt
        self.scale = scale
        self.joy_port = joy_port
        self.fps = fps
        self.title = title
        self.screenshot_dir = Path(screenshot_dir)

    def run(self) -> None:
        fe = GameFrontend(self.screenshot_dir)
        fe.name = self.title
        fe.default_joy_port = self.joy_port
        args = argparse.Namespace(
            joy_port=self.joy_port, scale=self.scale, fps=self.fps,
            demo_continue=True,
        )
        _run_viewer(fe, self.rt, args, None, None, None)
