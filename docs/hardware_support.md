# Hardware support — honest status

What the VM models, what it approximates, and what fails loud. The rule
(inherited from dos_re): model what real programs exercised, from observed
behaviour; everything else fails loud rather than guessing.

## CPU (6510) — `cpu.py`

| Area | Status |
|---|---|
| Documented opcodes | full, including decimal-mode ADC/SBC (NMOS flag semantics) |
| Stable illegal opcodes | LAX SAX DCP ISC SLO RLA SRE RRA ANC ALR ARR SBX USBC + NOP variants |
| Unstable illegals (ANE/LXA/LAS/TAS/SHA/SHX/SHY) | fail loud until a program needs one |
| JAM opcodes | raise `CPUJam` (also the shim ROM's fail-loud filler) |
| Cycle counting | per-instruction base + page-cross + branch penalties; **not** bus-cycle-exact |
| RMW double write | modeled (old value then new — `INC $D019` acknowledge depends on it) |
| Interrupts | level IRQ + edge NMI sampled between instructions; the one-instruction CLI/SEI delay is **not** modeled |
| 6510 port ($00/$01) | banking bits with input pull-ups; cassette lines inert |

## Memory / PLA — `memory.py`

RAM-under-ROM writes, BASIC/KERNAL/CHAR/IO banking from the 6510 port,
VIC bank view with char-ROM shadow at $1000 in banks 0/2. No cartridge
(GAME/EXROM) lines, no C128/REU — fail-loud territory.

## VIC-II — `vic.py`

- PAL 6569 geometry: 312 lines x 63 cycles, driven from CPU cycles.
- Registers latched **per raster line**: line-granular splits (colors,
  modes, scroll, sprite multiplexing per line) render correctly;
  within-line effects (mid-line $D020 stripes, FLI) do not.
- No badline / sprite-DMA cycle stealing: raster-counting code lands on
  slightly different lines than real hardware (deterministically so).
- All display modes: text, MC text, ECM, hires bitmap, MC bitmap;
  38/40-column and 24/25-row borders approximated at 8px/4px.
- Sprites: position, expand, multicolor, bg priority, correct layering.
- Collisions ($D01E/$D01F): computed lazily once per frame from latched
  line state; read-clear semantics. Collision **interrupts** fail loud.
- Light pen reads a fixed neutral value. $D030-$D03F (C128/DTV) read $FF.

## CIA 6526 — `cia.py`

Timers A/B (φ2 and B-counts-A-underflows modes, one-shot/continuous,
force-load), ICR mask/status protocol, full keyboard matrix + both
joysticks with reverse-scan support. TOD frozen at 00:00:00 (reads 0,
writes ignored) — a program *waiting* on TOD hangs visibly instead of
running wrong. Serial shift register reads 0. No CNT pin.

## SID — `sid.py`

Write-through register file (the command stream is the recovery evidence)
plus deterministic reads: OSC3 = SID noise-polynomial LFSR stepped per
read (snapshot-carried), ENV3 = 0, paddles = $FF. No audio synthesis yet —
that is a frontend-ring feature to layer on later (dos_re's AdLib pattern).

## KERNAL — `kernal.py`

HLE traps at authentic body addresses over a clean-room shim ROM; see the
module docstring. Implemented: CHROUT screen editor subset (+scroll,
colors, reverse), GETIN/keyboard buffer/SCNKEY/STOP, LOAD from the attached
D64 (with $C3/$C4 preamble and $AE/$AF end pointers), OPEN/CHKIN/CHRIN
byte streams from disk, CLOSE/CLRCHN/CLALL, SETLFS/SETNAM/READST/SETMSG,
RDTIM/SETTIM/UDTIM, SCREEN/PLOT/IOBASE, MEMTOP/MEMBOT, VECTOR/RESTOR,
CINT/IOINIT/RAMTAS and the direct-call internals $E518/$E536/$E544/$E566/
$E5A0/$FD15/$FD50/$FDA3/$FF5B. Fail loud: SAVE, tape, RS-232, raw IEC
(SECOND/TKSA/ACPTR/CIOUT/TALK/LISTEN/...), CHRIN from keyboard, CHKOUT.

## Drive

No 1541 CPU/GCR model. Disk access is KERNAL-level HLE against the D64
image. A game that bit-bangs IEC or ships a fastloader will fail loud at
the IEC traps — that is the signal to build the game-specific loader HLE
(or, if it ever becomes unavoidable, a real drive model).
