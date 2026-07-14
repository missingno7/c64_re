"""Checkpoint stepping + headless player smoke tests (no game assets)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c64_re.checkpoints import checkpoints_for, run_to_next_checkpoint  # noqa: E402
from c64_re import player  # noqa: E402
from c64_re.runtime import create_runtime, run_frames  # noqa: E402

from test_machine import basic_stub, make_d64_with_prg  # noqa: E402

# $080D: INC $FB / JSR $0900 / JMP $080D ; $0900: INC $FC / RTS
# The subroutine is INSIDE the PRG payload (the player boots its own runtime,
# so nothing can be poked in afterwards).
CODE = bytes((0xE6, 0xFB, 0x20, 0x00, 0x09, 0x4C, 0x0D, 0x08))
SUB = bytes((0xE6, 0xFC, 0x60))
PAYLOAD = CODE + b"\x00" * (0x0900 - 0x080D - len(CODE)) + SUB

CHECKPOINTS = {
    0x080D: "frame: main loop head",
    0x0900: "object-update: leaf entry",
}


def build(tmp_path):
    prg = basic_stub(2061, PAYLOAD)
    disk = make_d64_with_prg(b"CHK", prg)
    p = tmp_path / "chk.d64"
    p.write_bytes(disk.data)
    rt = create_runtime(p)
    return rt, p


def test_checkpoint_kinds_and_stepping(tmp_path):
    rt, _ = build(tmp_path)
    assert checkpoints_for(CHECKPOINTS, "frame") == frozenset({0x080D})
    with pytest.raises(KeyError):
        checkpoints_for(CHECKPOINTS, "nope")
    hit = run_to_next_checkpoint(rt.cpu, CHECKPOINTS)
    assert hit in CHECKPOINTS
    nxt = run_to_next_checkpoint(rt.cpu, CHECKPOINTS, kinds="object-update")
    assert nxt == 0x0900
    again = run_to_next_checkpoint(rt.cpu, CHECKPOINTS, kinds="object-update")
    assert again == 0x0900  # skip_current stepped off and came around


def test_player_headless_frames_and_demo(tmp_path, capsys):
    rt, image = build(tmp_path)

    class Frontend(player.GameFrontend):
        name = "chk"
        default_image = str(image)

    fe = Frontend(tmp_path)
    rc = player.main(fe, ["--headless", "--frames", "5"])
    assert rc == 0

    # record a demo programmatically, then replay it through the player CLI
    from c64_re.input_demo import InputDemoRecorder
    rt2 = create_runtime(image)
    run_frames(rt2, 3)
    rec = InputDemoRecorder(root=tmp_path / "demos", name="smoke")
    demo_dir = rec.start(rt2)
    rt2.machine.set_joy1(1)
    rec.record_joy(boundary=rt2.machine.vic.frame, port=1, mask=1)
    run_frames(rt2, 4)
    rec.stop(boundary=rt2.machine.vic.frame)

    rc = player.main(fe, ["--headless", "--play-demo", str(demo_dir)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "demo finished" in out


def test_player_oracle_mode_flag(tmp_path):
    rt, image = build(tmp_path)

    class Frontend(player.GameFrontend):
        name = "chk"
        default_image = str(image)

        def create_runtime(self, args):
            rt = super().create_runtime(args)
            assert not rt.cpu.replacement_hooks or not args.no_replacements
            return rt

    rc = player.main(Frontend(tmp_path),
                     ["--headless", "--frames", "2", "--no-replacements"])
    assert rc == 0
