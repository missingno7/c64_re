"""In-situ lift verification driver.

    python tools/liftverify.py IMAGE --entries $A,$B,... [--file NAME]
                               [--frames N] [--verify-frames M]
                               [--manifest PATH] [--no-strict-cycles]

Boots IMAGE (or resumes a .c64snap), lifts each entry in memory, installs
the lifted hooks, routes every call through the differential hook oracle
(strict cycle model by default — a lifted hook must reproduce the
interpreter's exact machine time), runs M frames, and reports per-hook
verified/diverged tallies.  A hook with calls, zero divergences becomes
ORACLE_PASSING in the manifest; divergences and refusals are recorded as
what they are.  Nothing is claimed for a hook that never fired.
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c64_re.hooks import HookRegistry  # noqa: E402
from c64_re.lift.emit import LiftRefused, lift_and_compile  # noqa: E402
from c64_re.lift.manifest import LiftManifest, LiftRecord  # noqa: E402
from c64_re.verification import install_live_verifier  # noqa: E402

from liftgen import boot_image, parse_entries  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image")
    ap.add_argument("--file", default="*")
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--entries", required=True)
    ap.add_argument("--verify-frames", type=int, default=100)
    ap.add_argument("--manifest", default="")
    ap.add_argument("--no-strict-cycles", action="store_true")
    ap.add_argument("--max-instructions", type=int, default=768)
    args = ap.parse_args()

    rt = boot_image(args.image, args.file, args.frames)
    manifest = LiftManifest.load(args.manifest) if args.manifest else LiftManifest()

    registry = HookRegistry()
    installed: dict[str, int] = {}
    scans: dict[int, object] = {}
    for entry in parse_entries(args.entries):
        try:
            hook, _, scan = lift_and_compile(rt.mem.rb, entry,
                                             max_instructions=args.max_instructions)
        except LiftRefused as exc:
            print(f"${entry:04X}  REFUSED  {exc.refusal.reason}: {exc.refusal.detail}")
            manifest.update(LiftRecord(entry=entry, name=f"refused_{entry:04X}",
                                       status="REFUSED",
                                       refusal_reason=exc.refusal.reason))
            continue
        name = f"lifted_{entry:04X}"
        registry.replace(entry, name)(hook)
        installed[name] = entry
        scans[entry] = scan
        print(f"${entry:04X}  installed ({len(scan.insns)} insns, "
              f"{len(scan.calls)} call dep(s))")

    if not installed:
        print("nothing installed; done")
        if args.manifest:
            manifest.save(args.manifest)
        return 1

    registry.install(rt.cpu)
    verified: Counter[str] = Counter()
    diverged: Counter[str] = Counter()
    reasons: dict[str, str] = {}

    def on_result(name, ok, reason):
        if ok:
            verified[name] += 1
        else:
            diverged[name] += 1
            reasons.setdefault(name, reason)

    install_live_verifier(rt, on_result=on_result, raise_on_divergence=False,
                          strict_cycles=not args.no_strict_cycles)

    from c64_re.runtime import run_frames
    run_frames(rt, args.verify_frames)

    print(f"\nafter {args.verify_frames} frames:")
    exit_code = 0
    for name, entry in sorted(installed.items(), key=lambda kv: kv[1]):
        ok, bad = verified[name], diverged[name]
        scan = scans[entry]
        if bad:
            status, exit_code = "DIVERGED", 2
            print(f"  {name}: {ok} verified, {bad} DIVERGED — first: {reasons[name]}")
        elif ok:
            status = "ORACLE_PASSING"
            print(f"  {name}: {ok} calls verified, 0 diverged")
        else:
            status = "LIFTED"
            print(f"  {name}: never fired (no claim)")
        manifest.update(LiftRecord(
            entry=entry, name=name, status=status,
            size_bytes=scan.size_bytes, instructions=len(scan.insns),
            calls_seen=ok + bad, verified_calls=ok, diverged_calls=bad,
            notes=reasons.get(name, ""),
        ))
    if args.manifest:
        manifest.save(args.manifest)
        print(f"manifest -> {args.manifest} ({manifest.summary()})")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
