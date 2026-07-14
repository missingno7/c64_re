"""Observer-only SID audio for the viewer — the dos_re ``audio_sink`` role.

The VM's :class:`c64_re.sid.SID` records the register writes the game makes
but does not make sound.  This module turns that register file into audible
audio for the interactive viewer, exactly the way dos_re's ``AdlibSpeakerSink``
turned the AdLib register stream into sound: it only ever *reads* SID state,
never writes game state, so a demo replays bit-identically with audio on or
off.  It lives in the FRONTEND RING (numpy/pygame allowed here and nowhere
else in the package), and imports lazily.

It is a **playable approximation**, not a cycle-exact 6581/8580 (that is
reSID's job): three voices (triangle / sawtooth / pulse / noise), a per-voice
ADSR envelope driven by the gate bit, and the master volume — sampled once
per emulated frame (like the VIC's per-line latch).  The filter is not
modeled.  Good enough to hear the game; honest about what it is.
"""
from __future__ import annotations

PAL_PHI2 = 985248            # PAL system clock (Hz)
SID_FREQ_SCALE = PAL_PHI2 / 16_777_216  # reg value -> Hz

# SID envelope rates: attack times in ms per 4-bit rate nibble (decay/release
# use 3x these).  From the 6581 datasheet.
_ATTACK_MS = (2, 8, 16, 24, 38, 56, 68, 80, 100, 250, 500, 800, 1000, 3000, 5000, 8000)


class _Voice:
    __slots__ = ("phase", "env", "stage")

    def __init__(self) -> None:
        self.phase = 0.0     # 0..1 oscillator phase, continuous across frames
        self.env = 0.0       # 0..1 envelope level
        self.stage = "off"   # off / attack / decay-sustain / release


class SidAudioSink:
    """Synthesizes the SID register stream and plays it through pygame.

    Call :meth:`open` once, then :meth:`pump` once per emulated frame from the
    viewer loop.  :meth:`close` on exit.
    """

    def __init__(self, sid, *, sample_rate: int = 44100, fps: float = 50.125,
                 gain: float = 0.28) -> None:
        self.sid = sid
        self.sample_rate = sample_rate
        self.fps = fps
        self.gain = gain
        self.samples_per_frame = max(1, round(sample_rate / fps))
        self.voices = [_Voice(), _Voice(), _Voice()]
        self._np = None
        self._pygame = None
        self._channel = None
        self._channels = 1  # actual mixer channel count (mono/stereo)

    # ---- lifecycle -----------------------------------------------------------
    def open(self) -> bool:
        try:
            import numpy as np
            import pygame
        except Exception:  # noqa: BLE001 - audio is optional
            return False
        self._np = np
        self._pygame = pygame
        if not pygame.mixer.get_init():
            try:
                pygame.mixer.init(frequency=self.sample_rate, size=-16,
                                  channels=1, buffer=1024)
            except Exception:  # noqa: BLE001 - no audio device
                self._np = None
                return False
        init = pygame.mixer.get_init()  # (freq, size, channels) — pygame.init()
        if init:                        # often pre-inits the mixer as stereo
            self.sample_rate = init[0]
            self._channels = init[2]
            self.samples_per_frame = max(1, round(self.sample_rate / self.fps))
        pygame.mixer.set_num_channels(8)
        self._channel = pygame.mixer.Channel(7)
        return True

    def close(self) -> None:
        if self._channel is not None:
            self._channel.stop()

    # ---- per-frame synthesis --------------------------------------------------
    def render_samples(self):
        """One emulated frame of audio as an int16 numpy array (advances the
        voice envelopes/phases).  Separated from playback so it is testable
        without an audio device."""
        np = self._np
        regs = self.sid.regs
        n = self.samples_per_frame
        dt = 1.0 / self.fps
        mix = np.zeros(n, dtype=np.float64)

        for vi in range(3):
            base = vi * 7
            freq = regs[base] | (regs[base + 1] << 8)
            pw = ((regs[base + 2] | (regs[base + 3] << 8)) & 0x0FFF) / 4096.0
            control = regs[base + 4]
            ad = regs[base + 5]
            sr = regs[base + 6]
            v = self.voices[vi]

            self._advance_envelope(v, control, ad, sr, dt)
            if v.env <= 0.0005 and v.stage in ("off", "release"):
                continue

            mix += self._waveform(v, control, freq, pw, n) * v.env

        master = (regs[0x18] & 0x0F) / 15.0
        mix *= self.gain * master
        np.clip(mix, -1.0, 1.0, out=mix)
        return (mix * 32767).astype(np.int16)

    def pump(self) -> None:
        if self._np is None:
            return
        buf = self.render_samples()
        if self._channels == 2:  # match a stereo mixer (pygame's default)
            buf = self._np.repeat(buf[:, None], 2, axis=1)
        snd = self._pygame.sndarray.make_sound(buf)
        ch = self._channel
        # keep at most one frame queued so latency stays ~1 frame and gaps
        # self-heal if the viewer briefly falls behind real time
        if not ch.get_busy():
            ch.play(snd)
        elif ch.get_queue() is None:
            ch.queue(snd)

    # ---- oscillators ----------------------------------------------------------
    def _waveform(self, v, control, freq, pw, n):
        np = self._np
        inc = (freq * SID_FREQ_SCALE) / self.sample_rate
        phase = (v.phase + np.arange(1, n + 1) * inc) % 1.0
        v.phase = float(phase[-1]) if n else v.phase

        use_tri = control & 0x10
        use_saw = control & 0x20
        use_pulse = control & 0x40
        use_noise = control & 0x80

        if use_noise:
            return np.random.uniform(-1.0, 1.0, n)
        if use_pulse and not (use_tri or use_saw):
            return np.where(phase < pw, 1.0, -1.0)
        if use_saw and not (use_tri or use_pulse):
            return 2.0 * phase - 1.0
        if use_tri and not (use_saw or use_pulse):
            return 2.0 * np.abs(2.0 * phase - 1.0) - 1.0
        if use_tri or use_saw or use_pulse:
            # combined waveforms: SID ANDs them; approximate by product of the
            # enabled shapes (keeps the timbre in the right family)
            out = np.ones(n)
            if use_tri:
                out *= 2.0 * np.abs(2.0 * phase - 1.0) - 1.0
            if use_saw:
                out *= 2.0 * phase - 1.0
            if use_pulse:
                out *= np.where(phase < pw, 1.0, -1.0)
            return out
        return np.zeros(n)

    # ---- envelope -------------------------------------------------------------
    def _advance_envelope(self, v, control, ad, sr, dt) -> None:
        gate = control & 0x01
        attack = _ATTACK_MS[(ad >> 4) & 0x0F] / 1000.0
        decay = 3 * _ATTACK_MS[ad & 0x0F] / 1000.0
        sustain = ((sr >> 4) & 0x0F) / 15.0
        release = 3 * _ATTACK_MS[sr & 0x0F] / 1000.0

        if gate:
            if v.stage in ("off", "release"):
                v.stage = "attack"
            if v.stage == "attack":
                v.env += dt / max(attack, 1e-4)
                if v.env >= 1.0:
                    v.env = 1.0
                    v.stage = "decay-sustain"
            elif v.stage == "decay-sustain":
                if v.env > sustain:
                    v.env -= dt / max(decay, 1e-4)
                    if v.env < sustain:
                        v.env = sustain
                else:
                    v.env = sustain
        elif v.stage != "off":
            v.stage = "release"
            v.env -= dt / max(release, 1e-4)
            if v.env <= 0.0:
                v.env = 0.0
                v.stage = "off"
