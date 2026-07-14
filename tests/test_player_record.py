"""Auto-start demo recording through the standard player (viewer path).

Regression guard for the '--record-demo did nothing without F11' bug: the
flag must begin recording immediately and save on window close, for both
snapshot-anchored and cold-start demos.  Uses SDL's dummy video driver so
it runs headless; skips where numpy/pygame are unavailable.
"""
import json
import os
import sys
import threading
import time
from pathlib import Path

import pytest

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

pytest.importorskip("numpy")
pygame = pytest.importorskip("pygame")

from c64_re import player  # noqa: E402
from c64_re.input_demo import InputDemoPlayback  # noqa: E402

from test_machine import basic_stub, make_d64_with_prg  # noqa: E402

# a synthetic PRG that just spins so the viewer has something to run
SPIN = bytes((0xEA, 0xEA, 0x4C, 0x0D, 0x08))  # NOP NOP JMP $080D


def make_frontend(tmp_path, *, boot_frames: int):
    prg = basic_stub(2061, SPIN)
    disk = make_d64_with_prg(b"SPIN", prg)
    image = tmp_path / "spin.d64"
    image.write_bytes(disk.data)

    class Frontend(player.GameFrontend):
        name = "spin"
        default_image = str(image)
        default_joy_port = 1

        def boot_to_start(self, rt, args):
            from c64_re.runtime import run_frames
            if boot_frames:
                run_frames(rt, boot_frames)

    return Frontend(tmp_path)


def _feed_after(delay, events):
    def run():
        time.sleep(delay)
        for pause, ev in events:
            time.sleep(pause)
            pygame.event.post(ev)
    threading.Thread(target=run, daemon=True).start()


def _latest_demo(root: Path, name: str) -> Path:
    demos = sorted((root / "artifacts" / "demos").glob(f"demo_{name}_*"))
    assert demos, f"no demo recorded for {name!r}"
    return demos[-1]


def test_record_demo_is_cold_start_from_power_on(tmp_path):
    # boot_to_start would run 999 frames — --record-demo must SKIP it and
    # record a cold-start demo from a fresh power-on runtime.
    fe = make_frontend(tmp_path, boot_frames=999)
    _feed_after(0.5, [
        (0.2, pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP)),
        (0.2, pygame.event.Event(pygame.KEYUP, key=pygame.K_UP)),
        (0.2, pygame.event.Event(pygame.QUIT)),
    ])
    rc = player.main(fe, ["--record-demo", "--demo-name", "session"])
    assert rc == 0
    demo = _latest_demo(tmp_path, "session")
    m = json.loads((demo / "input_demo.json").read_text())
    assert m["status"] == "complete"       # saved on close, no F11 needed
    assert m["snapshot"] is None           # cold-start: no start snapshot
    assert m["event_count"] >= 2           # joystick down+up captured
    assert InputDemoPlayback.load(demo).is_cold_start


def test_demo_name_defaults_to_game_name(tmp_path):
    fe = make_frontend(tmp_path, boot_frames=0)  # name defaults to frontend.name
    _feed_after(0.5, [(0.2, pygame.event.Event(pygame.QUIT))])
    rc = player.main(fe, ["--record-demo"])
    assert rc == 0
    _latest_demo(tmp_path, "spin")  # frontend.name == "spin"


def test_record_demo_from_snapshot_is_anchored(tmp_path):
    from c64_re.runtime import run_frames
    from c64_re.snapshot import write_snapshot

    # a mid-session snapshot: resuming it + --record-demo records anchored
    seed = fe = make_frontend(tmp_path, boot_frames=0)
    rt = seed.create_runtime(player.build_parser(seed, "t").parse_args([]))
    run_frames(rt, 5)
    snap = tmp_path / "mid.c64snap"
    write_snapshot(rt, snap)

    _feed_after(0.5, [
        (0.2, pygame.event.Event(pygame.KEYDOWN, key=pygame.K_UP)),
        (0.2, pygame.event.Event(pygame.KEYUP, key=pygame.K_UP)),
        (0.2, pygame.event.Event(pygame.QUIT)),
    ])
    rc = player.main(fe, ["--snapshot", str(snap), "--record-demo",
                          "--demo-name", "anchored"])
    assert rc == 0
    demo = _latest_demo(tmp_path, "anchored")
    m = json.loads((demo / "input_demo.json").read_text())
    assert m["snapshot"] == "snapshot.c64snap"   # anchored, not cold
    assert not InputDemoPlayback.load(demo).is_cold_start


def test_record_demo_headless_is_rejected(tmp_path):
    fe = make_frontend(tmp_path, boot_frames=0)
    with pytest.raises(SystemExit):
        player.main(fe, ["--record-demo", "--headless", "--frames", "2"])
