# AGENTS.md — c64_re framework repository

`c64_re` is the game-agnostic core of an oracle-driven C64 recovery method.
It follows the same rulebook as `dos_re` (read
`D:\Games\DOS\dos_recosystem\dos_re\AGENTS.md` — those working principles
apply verbatim here); this file records only what is C64-specific.

## The rules that bind every change

- **Game-agnostic core.** No game addresses, filenames, or data formats in
  `c64_re/`. Game knowledge lives in the port's adapter (e.g.
  `stix_port/stix/`). If a change needs a concrete address, it belongs in
  the adapter — with one exception: *authentic KERNAL/hardware addresses*
  (jump table entries, vector bodies, `$EA31`, `$FD30`) are hardware truth,
  not game knowledge, and belong in `kernal.py`.
- **Stdlib-only core; numpy/pygame only in the frontend ring**
  (`player.py`). Measured performance choice, not purity: the interpreter
  is scalar per-instruction work, and PyPy is the speed path.
- **Do not model more than a real program exercised.** New opcodes, KERNAL
  services, or chip behaviour are added only when a concrete program hits
  them, with the observed contract in a comment and a focused test.
  The JAM-filled shim ROM is the enforcement mechanism: unimplemented
  KERNAL code fails loud with the exact address.
- **Fail loud, never fall back silently.** Unstable illegal opcodes, raw
  IEC serial, tape, SAVE, unmapped I/O — all raise with precise context.
  Never replace a fail-fast path with a plausible default.
- **Determinism is the product.** No wall time anywhere in the core; all
  chip time derives from CPU cycle telemetry. The viewer may pace with the
  wall clock but must feed input through the same machine API demos use.
  Anything time-driven or random (SID OSC3 reads) is a deterministic,
  snapshot-carried model.
- **Behaviour changes need tests.** `python -m pytest tests -q` must pass
  without game assets.

## No Commodore ROM bytes — ever

The shim ROM in `kernal.py` is clean-room: authentic *addresses and
shapes* (which are interface facts), synthesized *bytes*. Real ROM images
are only ever loaded from a user-supplied `roms/` directory at runtime.
The synthesized chargen font is original work.

## Known model limits (honest list)

See `docs/hardware_support.md`. Headlines: instruction-level timing (no
badline/sprite DMA cycle stealing, no in-line register-change granularity),
lazy per-frame sprite collisions (collision *IRQs* fail loud), TOD frozen,
no IEC bit-banging / 1541 model, no tape, SAVE unimplemented, color-RAM
upper nibble reads 0. Each is a documented decision waiting for a real
program to force the issue — when one does, model it from observed
behaviour and add the test.
