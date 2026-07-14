# c64_re — an oracle-driven C64 game recovery framework

The Commodore 64 sibling of [`dos_re`](../../dos_recosystem/dos_re/README.md):
a recovery laboratory, not an emulator. The original program runs as the
*oracle* inside a deterministic VM; individual routines get replaced with
recovered source and verified against the original execution. The original
binary is the single source of truth — never guess, trace what it did.

**This is infrastructure for AI agents, not a library for end users.** The
expected operator is an autonomous agent handed a porting repo plus a game's
disk image. The human supplies the game, plays it, and records demos.

## What exists today (bring-up milestone)

- **The machine** — a deterministic 6510 interpreter (`cpu.py`: all documented
  opcodes incl. decimal mode + the stable illegals; unstable ones fail loud),
  PLA banking with RAM-under-ROM and the VIC's bank view (`memory.py`),
  VIC-II with per-raster-line register latching, all screen modes, sprites,
  lazy per-frame collision latching (`vic.py`), both CIAs with timers and the
  keyboard matrix / joysticks (`cia.py`), a SID register model with a
  deterministic OSC3 noise LFSR (`sid.py`).
- **The KERNAL seam** — no Commodore ROM bytes ship here. `kernal.py`
  synthesizes a shim ROM shaped like the real one where programs depend on
  its shape (vectors, IRQ/NMI dispatch stubs, the API jump table with real
  `JMP ($03xx)` indirections, the $FD30 vector-table image, `$EA31`/`$EA81`)
  and implements the service contracts as Python traps at the authentic body
  addresses — LOAD is served from the attached D64. Everything unimplemented
  is JAM-filled: it fails loud with the exact address and becomes the next
  work item. Real ROMs (`roms/` directory) are picked up if the user supplies
  them.
- **Media** — `d64.py`: D64 directory/chain parsing, CBM-DOS name matching,
  and the static BASIC-stub `SYS` parser used to boot without any BASIC ROM.
- **Recovery scaffolding** — `hooks.py` (the `@registry.replace(pc, name)`
  pattern, duplicate fail-fast, env-var disabling, live-code signature
  guards), CPU-level replacement/service hook dispatch with verifier routing,
  and `gaps.py` (`HybridGap` + hook stats, pure).
- **The determinism substrate** — `snapshot.py`: full machine freeze/thaw
  (in-memory `capture`/`restore` + `.c64snap` files), `clone_runtime` for
  side-by-side oracle execution. Proven on Stix: a resumed gameplay
  snapshot runs bit-identical to the original runtime.
- **The differential hook oracle** — `verification.py`: on every hook call,
  clone the machine, run the *original* instructions to the continuation
  (strict JSR-return mode, or `HookStop` metadata), run the hook, diff
  registers + flags + all 64K RAM + color RAM + VIC/CIA/SID state.
  Interrupts stay latched-but-undelivered inside the verified window (hook
  atomicity contract); machine time is resynced or, for cycle-faithful
  lifted hooks, held to the exact interpreter cycle model
  (`strict_cycles`). `C64_RE_TRACE_HOOK=$XXXX` dumps the oracle trace on
  divergence.
- **The automatic lifter** — `lift/`: static decode over the interpreter's
  own opcode table (never a second semantic model), function-region
  discovery with structured refusals (indirect jump, BRK, mid-instruction
  target, budget, no-exit), and an emitter producing literal
  per-instruction Python hooks — interpreter-helper reuse, exact cycle
  ticks, fail-loud SMC entry guard, `emulate_call` composition for JSR
  dependencies. `tools/liftgen.py` (census + emit) and
  `tools/liftverify.py` (in-situ install + per-call oracle verification +
  the `LiftManifest` proof ledger). Proven on Stix: 41/76 census
  candidates liftable; 6 gameplay-hot routines ORACLE_PASSING with 781
  verified calls, 0 divergences, under `strict_cycles`.
- **The frame oracle** — `frame_verify.py`: lockstep-step a pure-ASM
  reference and a hooked/native candidate to frame boundaries (VIC frame
  counter, or adapter-declared boundary PCs), diff rendered frames +
  sampled state, dump ref/cand/diff PNG artifacts on divergence.
- **The endgame equivalence engine** — `tick_demo.py`: game-tick-keyed
  demos (seed + per-tick inputs + ownership-masked digests + sidebands)
  that replay identically in pure-ASM, hybrid, and VM-less native modes.
- **Determinism substrate, demos** — `input_demo.py`: snapshot-anchored and
  cold-start input demos keyed to the VIC frame counter, single-event
  delivery for poll waits, suffix extraction; proven bit-identical on real
  Stix gameplay. `checkpoints.py`: VM-until-checkpoint stepping over the
  adapter's phase map.
- **Recovery bookkeeping** — `islands.py` (`@oracle_link` + the
  GUESS→CANONICAL status ladder + manifest generation), `coverage.py` (the
  measured native-% collector; unmeasured stays outside the %),
  `hook_taxonomy.py`, `frontier.py` (cold-start triage),
  `repro_artifacts.py` (crash/divergence repro capture), `runtime_code.py`
  (polyvariant self-modified-code support — idiomatic 6502 territory),
  `state_view.py` (typed views over swappable memory backends).
- **Frontend ring** — `player.py`: the standard play runner — GameFrontend
  + the unified CLI every port shares (`--headless`, `--snapshot`,
  `--save-snapshot`, `--record-demo`, `--play-demo`, `--demo-continue`,
  `--no-replacements`, `--verify-hooks`, `--trace-hooks`) with the
  canonical hotkeys (F10 screenshot, F11 demo record, F12 snapshot).
  numpy/pygame stay lazy and are allowed ONLY here (lint-enforced).
  `pngout.py` (stdlib PNG frame evidence), `dis6502.py` (disassembler over
  the interpreter's own opcode table — never a second semantic model),
  `testing.py` (stdlib test runner for constrained sandboxes).
- **Guardrail + evidence tools** — `tools/`: `lint.py` (stdlib-only core,
  frontend-ring boundary, adapter layer rules), `audit_layers.py` (pure
  layers never import the VM), `check_undefined_names.py`,
  `gen_island_manifest.py`, `liftgen.py`/`liftverify.py`,
  `render_frame.py` (image/snapshot → PNG), `view.py` (run any image in
  the viewer), `run_tests.py`.

Proven end-to-end on the first target: Stix (1983, Supersoft) boots from its
original cracked D64 through decruncher → trainer menu → gameplay → game
over, deterministically (byte-identical machine state across runs, including
under scripted input).

## Deliberately not built yet

The dos_re machinery is now fully mirrored except for, in rough order of
expected need:

1. **Viewer audio** (`audio_sink.py` analogue) — an observer-only SID
   synthesis backend for the viewer (the register stream is already
   captured; synthesis is presentation and never touches game state).
2. `overlay_menu.py` — the native product's settings widget
   (POST-ENDGAME-gated; building it earlier violates the method).
3. Misc tools as need arises: hotspot profiler, the static hook-oracle
   audit tool (`audit_hook_oracle.py` analogue), a VICE-snapshot importer
   (the `dosbox_savestate.py` analogue), the overnight-loop harness
   (template-side).

## The hard boundary

`c64_re/` never learns a game's addresses, filenames, or formats — that
lives in the game's adapter package (see `../stix_port/stix/`). The core is
stdlib-only; numpy/pygame only in the frontend ring.

## Sanity check

```bash
python -m pytest tests -q          # framework suite, no game assets needed
```

## Requirements

Python 3.11+. Core: zero dependencies. Viewer: `numpy` + `pygame`.
Headless runs work unchanged (and much faster) under PyPy.

No game ROMs, disks, or KERNAL/BASIC images are included. Bring your own
legally owned originals.
