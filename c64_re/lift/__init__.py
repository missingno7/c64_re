"""The automatic lifter: 6502 function entry -> literal, verified Python hook.

dos_re's M0/M1 design, 6502 flavor:

- ``decode``  — static decode over the interpreter's OWN opcode table
  (never a second semantic model), plus control-flow classification.
- ``cfg``     — function-region discovery from an entry PC, with the
  structured refusal taxonomy (indirect jump, JAM, mid-instruction jump
  target, region budget).
- ``emit``    — the emitter: a FunctionScan becomes a self-contained hook
  reproducing the routine per-instruction — same memory access order, same
  flag helpers (the interpreter's own), same cycle ticks — behind a
  fail-loud SMC entry guard.
- ``runtime`` — support imported by generated hooks (``emulate_call`` runs
  callees through the VM so hooks compose and lifting order is irrelevant;
  ``interp_one`` is the exact single-instruction fallback tier).
- ``manifest``— the lift proof ledger: LIFTED -> ORACLE_PASSING ->
  INSTALLED.  Deliberately disjoint from the recovery status ladder:
  lifted is NOT recovered (the metrics-honesty rule).

Lifting produces a *verified artifact to refactor from*, not understanding.
Every lifted hook is proven per-call by the differential oracle
(:mod:`c64_re.verification`) before it counts as anything.
"""
