"""c64_re — an oracle-driven Commodore 64 game recovery framework.

The C64 sibling of ``dos_re``: a deterministic 6502/6510 VM (CPU + PLA
banking + VIC-II + CIA pair + SID register model + D64 disk services) plus
the recovery machinery — replacement hooks, differential hook verification,
full-machine snapshots, and deterministic input demos.

The original program running in this VM is the *oracle* — the single source
of truth.  Game-specific knowledge (addresses, filenames, formats) never
lives in this package; it lives in a per-game adapter next to it.
"""

__all__ = [
    # the machine
    "cpu", "memory", "machine", "vic", "cia", "sid", "kernal", "d64",
    "runtime",
    # the proof engines + determinism substrate
    "hooks", "gaps", "snapshot", "input_demo", "verification",
    "frame_verify", "tick_demo", "checkpoints",
    # recovery bookkeeping
    "islands", "coverage", "hook_taxonomy", "frontier", "repro_artifacts",
    "runtime_code", "state_view",
    # the lifter
    "lift",
    # frontend ring + tools support
    "player", "audio_sink", "pngout", "dis6502", "testing",
]
