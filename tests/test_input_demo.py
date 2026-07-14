"""Input demo record/replay determinism tests (no game assets)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c64_re.input_demo import InputDemoPlayback, InputDemoRecorder  # noqa: E402
from c64_re.cia import JOY_FIRE, JOY_UP  # noqa: E402
from c64_re.runtime import create_runtime, run_frames  # noqa: E402

from test_machine import basic_stub, make_d64_with_prg  # noqa: E402

# A program whose state depends on input: every loop iteration samples the
# CIA1 ports (joystick 2 on $DC00, joystick 1 + keyboard rows on $DC01) and
# stores both into a RAM history ring — so replayed input must land on the
# exact same frames to reproduce RAM.
#   $080D: LDA $DC00 / STA $0340,X / LDA $DC01 / STA $0440,X
#          INX / JMP $080D
CODE = bytes((
    0xAD, 0x00, 0xDC, 0x9D, 0x40, 0x03,
    0xAD, 0x01, 0xDC, 0x9D, 0x40, 0x04,
    0xE8, 0x4C, 0x0D, 0x08,
))


def build_input_rt(tmp_path, fname="in.d64"):
    prg = basic_stub(2061, CODE)
    disk = make_d64_with_prg(b"INPUT", prg)
    p = tmp_path / fname
    p.write_bytes(disk.data)
    return create_runtime(p)


SCRIPT = [
    # (frame, action, arg)
    (3, "joy2", JOY_FIRE),
    (6, "joy2", 0),
    (8, "key_down", "SPACE"),
    (12, "key_up", "SPACE"),
    (13, "joy1", JOY_UP | JOY_FIRE),
    (17, "joy1", 0),
]
TOTAL_FRAMES = 22


def drive_scripted(rt, recorder=None) -> None:
    """Apply the script at exact frame boundaries, recording if asked."""
    m = rt.machine
    for frame in range(TOTAL_FRAMES):
        boundary = m.vic.frame
        for f, kind, arg in SCRIPT:
            if f != frame:
                continue
            if kind == "joy1":
                m.set_joy1(arg)
                if recorder:
                    recorder.record_joy(boundary=boundary, port=1, mask=arg)
            elif kind == "joy2":
                m.set_joy2(arg)
                if recorder:
                    recorder.record_joy(boundary=boundary, port=2, mask=arg)
            elif kind == "key_down":
                m.key_down(arg)
                if recorder:
                    recorder.record_key_down(boundary=boundary, key=arg)
            else:
                m.key_up(arg)
                if recorder:
                    recorder.record_key_up(boundary=boundary, key=arg)
        run_frames(rt, 1)


def replay_all(playback, rt) -> None:
    while True:
        boundary = rt.machine.vic.frame
        playback.apply_to_runtime(boundary, rt)
        if playback.finished(boundary):
            break
        run_frames(rt, 1)


def state_digest(rt):
    return (bytes(rt.mem.ram), rt.cpu.s.as_dict(), rt.cpu.cycle_count,
            rt.machine.vic.frame)


def test_snapshot_anchored_record_replay(tmp_path):
    rt = build_input_rt(tmp_path)
    run_frames(rt, 5)  # anchor mid-run: demo starts from a snapshot
    recorder = InputDemoRecorder(root=tmp_path / "demos", name="anchored")
    demo_dir = recorder.start(rt)
    drive_scripted(rt, recorder)
    recorder.stop(boundary=rt.machine.vic.frame)
    reference = state_digest(rt)

    playback = InputDemoPlayback.load(demo_dir)
    assert not playback.is_cold_start
    rt2 = playback.make_runtime()
    replay_all(playback, rt2)
    assert state_digest(rt2) == reference


def test_cold_start_record_replay(tmp_path):
    rt = build_input_rt(tmp_path)
    recorder = InputDemoRecorder(root=tmp_path / "demos", name="cold")
    demo_dir = recorder.start(rt, write_start_snapshot=False)
    drive_scripted(rt, recorder)
    recorder.stop(boundary=rt.machine.vic.frame)
    reference = state_digest(rt)

    playback = InputDemoPlayback.load(demo_dir)
    assert playback.is_cold_start
    rt2 = playback.make_runtime()  # boots fresh from recorded boot_args
    replay_all(playback, rt2)
    assert state_digest(rt2) == reference


def test_cold_start_requires_fresh_boot(tmp_path):
    rt = build_input_rt(tmp_path)
    run_frames(rt, 1)
    recorder = InputDemoRecorder(root=tmp_path / "demos", name="late")
    try:
        recorder.start(rt, write_start_snapshot=False)
    except RuntimeError as e:
        assert "cold-start" in str(e)
    else:
        raise AssertionError("late cold-start recording did not fail")


def test_single_event_delivery_for_poll_waits(tmp_path):
    """Two same-frame events must be deliverable one per call (poll waits)."""
    rt = build_input_rt(tmp_path)
    recorder = InputDemoRecorder(root=tmp_path / "demos", name="pair")
    demo_dir = recorder.start(rt)
    b = rt.machine.vic.frame
    recorder.record_key_up(boundary=b, key="N")     # release...
    recorder.record_key_down(boundary=b, key="N")   # ...and re-press, same frame
    recorder.stop(boundary=b + 1)

    playback = InputDemoPlayback.load(demo_dir)
    rt2 = playback.make_runtime()
    assert playback.apply_to_runtime(10_000, rt2, single=True) == 1
    assert playback.apply_to_runtime(10_000, rt2, single=True) == 1
    assert playback.exhausted


def test_suffix_rebases_remaining_events(tmp_path):
    rt = build_input_rt(tmp_path)
    recorder = InputDemoRecorder(root=tmp_path / "demos", name="long")
    demo_dir = recorder.start(rt)
    drive_scripted(rt, recorder)
    recorder.stop(boundary=rt.machine.vic.frame)

    playback = InputDemoPlayback.load(demo_dir)
    rt2 = playback.make_runtime()
    # replay only the first 10 frames, then cut a suffix
    for _ in range(10):
        playback.apply_to_runtime(rt2.machine.vic.frame, rt2)
        run_frames(rt2, 1)
    cut = rt2.machine.vic.frame
    suffix_dir = playback.write_suffix(rt2, root=tmp_path / "demos",
                                       name="tail", boundary=cut)
    suffix = InputDemoPlayback.load(suffix_dir)
    rt3 = suffix.make_runtime()
    replay_all(suffix, rt3)
    # finishing the original replay from the cursor gives the same state
    replay_all(playback, rt2)
    assert state_digest(rt3) == state_digest(rt2)
