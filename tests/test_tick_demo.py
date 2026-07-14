"""Tests for c64_re.tick_demo — the generic game-tick equivalence engine.

Game-free by construction: a synthetic 'game' whose state is a small bytearray, a fake CPU for the
recorder plumbing, hand-built demos for the verify loop, and one end-to-end run on a real synthetic
PRG.  The game-specific halves these tests stand in for (seam PCs, ownership masks, transition
outcomes) are adapter territory.
"""
import hashlib
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c64_re.tick_demo import TickDemo, masked_digest, record_ticks, replay_to, verify_ticks  # noqa: E402

from test_machine import basic_stub, make_d64_with_prg  # noqa: E402


# --- masked_digest --------------------------------------------------------------------------------
def test_masked_digest_zero_mask_post_and_range():
    base = bytes(range(16))
    plain = masked_digest(base)
    assert plain == hashlib.sha1(base).hexdigest()
    # zeroing offset 3 makes two buffers differing only there agree
    a = bytearray(base); a[3] = 0xAA
    b = bytearray(base); b[3] = 0xBB
    assert masked_digest(a, zero=[3]) == masked_digest(b, zero=[3])
    assert masked_digest(a) != masked_digest(b)
    # mask keeps the owned low bits significant
    c = bytearray(base); c[5] = 0x21          # low bit 1, render bit 0x20
    d = bytearray(base); d[5] = 0x01
    assert masked_digest(c, mask=[(5, 0x9F)]) == masked_digest(d, mask=[(5, 0x9F)])
    e = bytearray(base); e[5] = 0x00          # low bit differs -> still detected
    assert masked_digest(e, mask=[(5, 0x9F)]) != masked_digest(d, mask=[(5, 0x9F)])
    # post callback: adapter rule mutating the working buffer
    f = bytearray(base); f[7] = 0x99
    def clear7(buf): buf[7] = 0
    assert masked_digest(f, post=clear7) == masked_digest(base, post=clear7)
    # out-of-range offsets are ignored (one mask serves differently-sized captures)
    assert masked_digest(base, zero=[1000], mask=[(2000, 0x0F)]) == plain


# --- TickDemo container + format ------------------------------------------------------------------
def test_tick_demo_round_trip(tmp_path):
    demo = TickDemo(seed=bytes(range(256)) * 4)
    for i in range(5):
        demo.keys.append(bytes([i, i + 1, i + 2]))
        demo.digests.append(hashlib.sha1(bytes([i])).hexdigest())
    demo.sidebands["idle"] = [10, 20, 30, 40, 50]
    demo.sidebands["skew"] = [0, 1, 0, 1, 0]
    p = tmp_path / "demo.bin"
    demo.save(p)
    back = TickDemo.load(p)
    assert back.seed == demo.seed
    assert back.keys == demo.keys
    assert back.digests == demo.digests
    assert back.sidebands == demo.sidebands
    assert back.n_ticks == 5
    assert back.sideband_at(2) == {"idle": 30, "skew": 0}


def test_tick_demo_bad_magic_and_shape(tmp_path):
    p = tmp_path / "bad.bin"
    p.write_bytes(b"NOTADEMO" + bytes(64))
    with pytest.raises(ValueError, match="bad magic"):
        TickDemo.load(p)
    d = TickDemo(seed=b"s", keys=[b"ab", b"abc"], digests=["0" * 40, "1" * 40])
    with pytest.raises(ValueError, match="one length"):
        d.save(tmp_path / "x.bin")
    d2 = TickDemo(seed=b"s", keys=[b"ab"], digests=["0" * 40], sidebands={"idle": [1, 2]})
    with pytest.raises(ValueError, match="idle"):
        d2.save(tmp_path / "y.bin")


# --- verify_ticks ----------------------------------------------------------------------------------
# The synthetic 'native core': state[0] accumulates key[0] each tick; state[1] mirrors the injected
# 'idle' sideband; state[2] is 'render residue' excluded by the digest mask.
def _digest(state):
    return masked_digest(state, zero=[2])


def _inject(state, keys, sb):
    state[3] = keys[0]
    state[1] = sb.get("idle", 0) & 0xFF


def _tick(state):
    state[0] = (state[0] + state[3]) & 0xFF
    state[2] = (state[2] + 17) & 0xFF          # render residue: drifts, must not matter
    return None


def _make_demo(n=6):
    demo = TickDemo(seed=bytes(4))
    ref = bytearray(4)
    for i in range(n):
        keys = bytes([i + 1])
        _inject(ref, keys, {"idle": 7 * i})
        _tick(ref)
        demo.keys.append(keys)
        demo.sidebands.setdefault("idle", []).append(7 * i)
        demo.digests.append(_digest(ref))
    return demo


def test_verify_ticks_all_match():
    demo = _make_demo()
    n, div = verify_ticks(demo, bytearray(demo.seed), inject=_inject, tick=_tick, digest=_digest)
    assert (n, div) == (demo.n_ticks, None)


def test_verify_ticks_detects_first_divergence():
    demo = _make_demo()
    demo.digests[3] = "f" * 40                     # corrupt the recording at tick 3
    n, div = verify_ticks(demo, bytearray(demo.seed), inject=_inject, tick=_tick, digest=_digest)
    assert n == 3 and "digest mismatch" in div and "tick 3" in div


def test_verify_ticks_sideband_matters():
    demo = _make_demo()
    demo.sidebands["idle"][2] = 999                # native injects a different idle -> state[1] differs
    n, div = verify_ticks(demo, bytearray(demo.seed), inject=_inject, tick=_tick, digest=_digest)
    assert n == 2 and "digest mismatch" in div


def test_verify_ticks_terminal_and_raise():
    demo = _make_demo()

    def tick_terminal(state):
        return "LEVEL-END at this tick" if state[0] >= 3 else _tick(state)

    n, div = verify_ticks(demo, bytearray(demo.seed), inject=_inject, tick=tick_terminal, digest=_digest)
    assert div == "LEVEL-END at this tick" and 0 < n < demo.n_ticks

    def tick_raises(state):
        raise RuntimeError("unrecovered gap")

    n, div = verify_ticks(demo, bytearray(demo.seed), inject=_inject, tick=tick_raises, digest=_digest)
    assert n == 0 and "RuntimeError" in div and "unrecovered gap" in div


def test_suffix_and_replay_to_reproduce_divergence_at_tick_zero():
    # The divergence-repro workflow: verify reports tick i -> replay_to repositions a fresh state to just
    # BEFORE tick i -> suffix(i, captured seed) reproduces the divergence at its OWN tick 0.
    demo = _make_demo(6)
    demo.digests[4] = "f" * 40                               # planted divergence at tick 4
    n, div = verify_ticks(demo, bytearray(demo.seed), inject=_inject, tick=_tick, digest=_digest)
    assert n == 4 and "tick 4" in div
    st = bytearray(demo.seed)
    replay_to(demo, st, n, inject=_inject, tick=_tick)       # fast reposition (no digest checks)
    repro = demo.suffix(n, bytes(st))
    assert repro.n_ticks == 2                                # ticks 4..5 remain
    assert repro.sidebands["idle"] == demo.sidebands["idle"][4:]
    n2, div2 = verify_ticks(repro, bytearray(repro.seed), inject=_inject, tick=_tick, digest=_digest)
    assert n2 == 0 and "tick 0" in div2                      # the same divergence, now instant
    # and a suffix carved at a CLEAN point verifies green to the end
    clean = _make_demo(6)
    st2 = bytearray(clean.seed)
    replay_to(clean, st2, 3, inject=_inject, tick=_tick)
    tail = clean.suffix(3, bytes(st2))
    assert verify_ticks(tail, bytearray(tail.seed), inject=_inject, tick=_tick, digest=_digest) == (3, None)
    # suffix round-trips through the file format too
    import os, tempfile
    fd, p = tempfile.mkstemp(); os.close(fd)
    try:
        repro.save(p)
        back = TickDemo.load(p)
        assert back.keys == repro.keys and back.seed == repro.seed
    finally:
        os.unlink(p)


def test_suffix_bounds_and_replay_to_terminal_raises():
    demo = _make_demo(3)
    with pytest.raises(ValueError, match="out of range"):
        demo.suffix(7, b"")

    def tick_terminal(state):
        return "LEVEL-END"

    with pytest.raises(RuntimeError, match="terminal outcome at tick 0"):
        replay_to(demo, bytearray(demo.seed), 2, inject=_inject, tick=tick_terminal)


# --- record_ticks (recorder plumbing on a fake CPU) --------------------------------------------------
class _S:
    def __init__(self):
        self.pc = 0


class _FakeMem:
    def __init__(self, n=64):
        self.ram = bytearray(n)


class _FakeCPU:
    """Steps through a scripted (pc, mutate) timeline; record_ticks installs
    .trace_fn like the real CPU calls it (before executing, at the pc)."""

    def __init__(self, script):
        self.s = _S()
        self.mem = _FakeMem()
        self.trace_fn = None
        self._script = list(script)
        self._i = 0

    def pending(self):
        return self._i < len(self._script)

    def step(self):
        pc, mutate = self._script[self._i]
        self.s.pc = pc
        if mutate:
            mutate(self.mem.ram)
        if self.trace_fn is not None:
            self.trace_fn(self, pc, 0)
        self._i += 1


class _FakeRT:
    def __init__(self, cpu):
        self.cpu = cpu
        self.mem = cpu.mem


SEED_PC, KEYS_PC, REFINE_PC, COMMIT_PC = 0x0810, 0x0900, 0x0950, 0x0A00


def test_record_ticks_seed_observe_refine_commit():
    # Two ticks. Tick 1 exercises the REFINE pattern: an early capture at KEYS_PC is overwritten at
    # REFINE_PC (the consumption point) — the recording must keep the refined value. ram[0]=key cell,
    # ram[1]=sideband cell, ram[2]=gameplay state the digest covers.
    def setmem(**kv):
        def m(data):
            for k, v in kv.items():
                data[int(k[1:])] = v
        return m

    script = [
        (0x0500, None),                    # unrelated pc before seeding
        (SEED_PC, setmem(_0=0, _2=5)),     # seed captured HERE (ram[2]==5 in the seed)
        (KEYS_PC, setmem(_0=0x11, _1=3)),  # base capture: key 0x11
        (REFINE_PC, setmem(_0=0x22)),      # IRQ-style overwrite before consumption: the truth is 0x22
        (COMMIT_PC, setmem(_2=6)),         # end of tick 1
        (KEYS_PC, setmem(_0=0x33, _1=4)),  # tick 2: no refine
        (COMMIT_PC, setmem(_2=7)),         # end of tick 2
    ]
    cpu = _FakeCPU(script)
    rt = _FakeRT(cpu)

    def obs_keys(pending, rt):
        pending["keys"] = bytes([rt.mem.ram[0]])
        pending["idle"] = rt.mem.ram[1]

    def obs_refine(pending, rt):
        pending["keys"] = bytes([rt.mem.ram[0]])      # overwrite = the intended idiom

    def commit(pending, rt):
        if "keys" not in pending:
            return None
        return pending["keys"], {"idle": pending["idle"]}

    def digest(rt):
        return masked_digest(rt.mem.ram, zero=[0, 1])   # key/sideband cells excluded

    def advance():
        if not cpu.pending():
            return False
        cpu.step()                          # the trace hook observes s.pc as the script plays
        return True

    demo = record_ticks(rt, seed_pc=SEED_PC, commit_pc=COMMIT_PC,
                        observe={KEYS_PC: obs_keys, REFINE_PC: obs_refine},
                        commit=commit, digest=digest, advance_one_frame=advance)
    assert demo.n_ticks == 2
    assert demo.seed[2] == 5                                  # captured at SEED_PC
    assert demo.keys == [b"\x22", b"\x33"]                    # refined value won on tick 1
    assert demo.sidebands["idle"] == [3, 4]
    assert cpu.trace_fn is None                               # recorder uninstalled
    # a commit-pc hit BEFORE the seed must NOT commit (nothing recorded pre-seed)
    cpu2 = _FakeCPU([(COMMIT_PC, None), (0x0500, None)])
    rt2 = _FakeRT(cpu2)
    demo2 = record_ticks(rt2, seed_pc=SEED_PC, commit_pc=COMMIT_PC,
                         observe={}, commit=lambda p, r: (b"k", {}), digest=lambda r: "0" * 40,
                         advance_one_frame=lambda: (cpu2.step() or True) if cpu2.pending() else False)
    assert demo2.n_ticks == 0 and demo2.seed == b""


def test_record_ticks_chains_and_restores_existing_trace_fn():
    seen = []
    cpu = _FakeCPU([(SEED_PC, None), (COMMIT_PC, None)])
    prior = lambda c, pc, op: seen.append(pc)  # noqa: E731
    cpu.trace_fn = prior
    rt = _FakeRT(cpu)
    record_ticks(rt, seed_pc=SEED_PC, commit_pc=COMMIT_PC, observe={},
                 commit=lambda p, r: None, digest=lambda r: "0" * 40,
                 advance_one_frame=lambda: (cpu.step() or True) if cpu.pending() else False)
    assert seen == [SEED_PC, COMMIT_PC]        # the prior trace still saw every step
    assert cpu.trace_fn is prior               # and was restored


def test_record_and_verify_round_trip_fake_cpu():
    # End to end on the synthetic game: record from a scripted 'VM', verify with the native tick.
    ticks = 5
    script = [(SEED_PC, None)]
    vm_state = {"acc": 0}

    def make_tick(i):
        def pre(data):
            data[0] = i + 1                    # the key cell the game will consume
            data[1] = (7 * i) & 0xFF           # the sideband cell
        def post(data):
            vm_state["acc"] = (vm_state["acc"] + data[0]) & 0xFF
            data[2] = vm_state["acc"]          # gameplay state after the tick
            data[3] = (data[3] + 17) & 0xFF    # render residue (excluded)
        return [(KEYS_PC, pre), (COMMIT_PC, post)]

    for i in range(ticks):
        script += make_tick(i)
    cpu = _FakeCPU(script)
    rt = _FakeRT(cpu)
    dig = lambda r: masked_digest(r.mem.ram[:4], zero=[0, 1, 3])  # noqa: E731
    demo = record_ticks(
        rt, seed_pc=SEED_PC, commit_pc=COMMIT_PC,
        observe={KEYS_PC: lambda p, r: p.update(keys=bytes([r.mem.ram[0]]), idle=r.mem.ram[1])},
        commit=lambda p, r: (p["keys"], {"idle": p["idle"]}) if "keys" in p else None,
        digest=dig,
        advance_one_frame=lambda: (cpu.step() or True) if cpu.pending() else False)
    assert demo.n_ticks == ticks

    # native replay: state = [key, idle, acc, residue]; inject writes 0/1, tick accumulates into 2
    def n_inject(state, keys, sb):
        state[0] = keys[0]
        state[1] = sb["idle"] & 0xFF

    def n_tick(state):
        state[2] = (state[2] + state[0]) & 0xFF
        state[3] = (state[3] + 99) & 0xFF      # different residue drift than the VM: must not matter
        return None

    n, div = verify_ticks(demo, bytearray(demo.seed[:4]), inject=n_inject, tick=n_tick,
                          digest=lambda s: masked_digest(s, zero=[0, 1, 3]))
    assert (n, div) == (ticks, None)


# --- record_ticks on a REAL runtime (synthetic PRG, no game assets) -----------------------------------
# The 'game': an interrupt-free main loop consuming a key cell into a gameplay
# accumulator, with a drifting render-residue byte the digest must exclude.
#   $080D  SEI
#   $080E  LDA $C000     <- seed pc (main-loop top) & the key read
#   $0811  CLC           <- observe pc: the consumed key is now committed to A
#   $0812  ADC $FB
#   $0814  STA $FB       gameplay accumulator
#   $0816  INC $FC       render residue (excluded by the ownership mask)
#   $0818  JMP $080E     <- commit pc (end of tick)
GAME = bytes((0x78, 0xAD, 0x00, 0xC0, 0x18, 0x65, 0xFB, 0x85, 0xFB,
              0xE6, 0xFC, 0x4C, 0x0E, 0x08))
G_SEED, G_OBS, G_COMMIT = 0x080E, 0x0811, 0x0818


def test_record_ticks_real_runtime_and_native_verify(tmp_path):
    from c64_re.runtime import create_runtime, run_frames

    prg = basic_stub(2061, GAME)
    disk = make_d64_with_prg(b"TICK", prg)
    p = tmp_path / "tick.d64"
    p.write_bytes(disk.data)
    rt = create_runtime(p)

    frames = [0]

    def advance():
        if frames[0] >= 2:
            return False
        rt.mem.ram[0xC000] = frames[0] + 1     # the drive owns pacing/input
        run_frames(rt, 1)
        frames[0] += 1
        return True

    demo = record_ticks(
        rt, seed_pc=G_SEED, commit_pc=G_COMMIT,
        observe={G_OBS: lambda pend, r: pend.update(keys=bytes([r.mem.ram[0xC000]]))},
        commit=lambda pend, r: (pend["keys"], {}) if "keys" in pend else None,
        digest=lambda r: masked_digest(r.mem.ram[0xFB:0xFD], zero=[1]),
        advance_one_frame=advance,
    )
    assert rt.cpu.trace_fn is None
    assert demo.n_ticks > 100                  # ~983 ticks per PAL frame at 20 cycles/loop
    assert len(demo.seed) == 0x10000           # the full RAM image
    assert demo.seed[0xFB] == 0                # seeded at the FIRST loop top

    # the VM-less native core: state = [fb, residue, key]
    state = bytearray((demo.seed[0xFB], demo.seed[0xFC], 0))

    def inject(st, keys, sb):
        st[2] = keys[0]

    def tick(st):
        st[0] = (st[0] + st[2]) & 0xFF
        st[1] = (st[1] + 99) & 0xFF            # residue drifts differently: must not matter
        return None

    n, div = verify_ticks(demo, state, inject=inject, tick=tick,
                          digest=lambda st: masked_digest(st[0:2], zero=[1]))
    assert (n, div) == (demo.n_ticks, None)

    # and the demo round-trips through the on-disk format
    f = tmp_path / "tick.demo"
    demo.save(f)
    back = TickDemo.load(f)
    assert back.keys == demo.keys and back.digests == demo.digests and back.seed == demo.seed
