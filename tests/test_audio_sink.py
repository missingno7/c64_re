"""SID audio sink: renders the register stream, stays observer-only."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

np = pytest.importorskip("numpy")

from c64_re.audio_sink import SidAudioSink  # noqa: E402
from c64_re.sid import SID  # noqa: E402


def make_sink(sid):
    sink = SidAudioSink(sid, fps=50.0)
    sink._np = np  # render without opening an audio device
    return sink


def test_silent_when_no_voice_gated():
    sid = SID()
    sink = make_sink(sid)
    buf = sink.render_samples()
    assert int(np.abs(buf).max()) == 0


def test_gated_voice_produces_audio():
    sid = SID()
    # voice 1: sawtooth, gate on, mid frequency, full sustain, master volume max
    sid.regs[0], sid.regs[1] = 0x00, 0x20   # freq
    sid.regs[4] = 0x21                       # sawtooth + gate
    sid.regs[5] = 0x00                       # fast attack/decay
    sid.regs[6] = 0xF0                       # full sustain
    sid.regs[0x18] = 0x0F                    # master volume
    sink = make_sink(sid)
    # let the envelope open over a few frames
    peak = 0
    for _ in range(5):
        peak = max(peak, int(np.abs(sink.render_samples()).max()))
    assert peak > 1000, f"gated voice should be audible (peak {peak})"


def test_master_volume_scales_output():
    def peak_at_volume(vol):
        sid = SID()
        sid.regs[0], sid.regs[1] = 0x00, 0x20
        sid.regs[4] = 0x21
        sid.regs[6] = 0xF0
        sid.regs[0x18] = vol
        sink = make_sink(sid)
        p = 0
        for _ in range(5):
            p = max(p, int(np.abs(sink.render_samples()).max()))
        return p

    assert peak_at_volume(0x00) == 0          # muted
    assert peak_at_volume(0x0F) > peak_at_volume(0x07) > 0


def test_render_does_not_touch_sid_state():
    sid = SID()
    sid.regs[0], sid.regs[1], sid.regs[4], sid.regs[0x18] = 0x00, 0x20, 0x21, 0x0F
    before = bytes(sid.regs)
    sink = make_sink(sid)
    for _ in range(10):
        sink.render_samples()
    assert bytes(sid.regs) == before  # observer-only: never writes game state
