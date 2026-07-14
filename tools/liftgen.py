"""Lift census + emitter CLI.

    python tools/liftgen.py IMAGE [--file NAME] [--frames N]
                            [--entries $A,$B,...] [--scan-jsr LO:HI]
                            [--emit DIR] [--manifest PATH]

IMAGE is a .d64/.prg (booted fresh, run N frames first) or a .c64snap
snapshot (resumed as-is).  Entries come from --entries and/or a --scan-jsr
census (every JSR target found in the given RAM range — a heuristic seed
list, not evidence).  Each entry is scanned; the table says LIFTED (with
size) or REFUSED (with the structured reason).  --emit writes one literal
hook module per liftable entry; --manifest records the census.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c64_re.lift.cfg import scan_function  # noqa: E402
from c64_re.lift.emit import emit_hook  # noqa: E402
from c64_re.lift.manifest import LiftManifest, LiftRecord  # noqa: E402


def boot_image(path: str, file: str, frames: int):
    from c64_re.runtime import create_runtime, run_frames
    from c64_re.snapshot import load_snapshot

    p = Path(path)
    if p.suffix.lower() == ".c64snap":
        return load_snapshot(p, install_hooks=False)
    rt = create_runtime(p, file=file, install_hooks=False)
    if frames:
        run_frames(rt, frames)
    return rt


def parse_entries(text: str) -> list[int]:
    out = []
    for token in text.replace(";", ",").split(","):
        token = token.strip().lstrip("$")
        if token:
            out.append(int(token, 16) & 0xFFFF)
    return out


def scan_jsr_targets(mem, lo: int, hi: int) -> list[int]:
    """Census heuristic: collect JSR operands in [lo, hi).  Code/data are not
    distinguished statically — targets are candidates, nothing more."""
    targets = set()
    a = lo
    while a < hi - 2:
        if mem.rb(a) == 0x20:
            targets.add(mem.rb(a + 1) | (mem.rb(a + 2) << 8))
        a += 1
    return sorted(targets)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("image")
    ap.add_argument("--file", default="*")
    ap.add_argument("--frames", type=int, default=0)
    ap.add_argument("--entries", default="")
    ap.add_argument("--scan-jsr", default="", metavar="LO:HI")
    ap.add_argument("--emit", default="", metavar="DIR")
    ap.add_argument("--manifest", default="")
    ap.add_argument("--max-instructions", type=int, default=768)
    args = ap.parse_args()

    rt = boot_image(args.image, args.file, args.frames)
    entries = parse_entries(args.entries)
    if args.scan_jsr:
        lo, _, hi = args.scan_jsr.partition(":")
        entries += scan_jsr_targets(rt.mem, int(lo.lstrip("$"), 16),
                                    int(hi.lstrip("$"), 16))
    entries = sorted(set(entries))
    if not entries:
        ap.error("no entries: pass --entries and/or --scan-jsr")

    manifest = LiftManifest.load(args.manifest) if args.manifest else LiftManifest()
    emit_dir = Path(args.emit) if args.emit else None
    if emit_dir:
        emit_dir.mkdir(parents=True, exist_ok=True)

    lifted = refused = 0
    for entry in entries:
        scan = scan_function(rt.mem.rb, entry, max_instructions=args.max_instructions)
        if scan:
            lifted += 1
            print(f"${entry:04X}  LIFTED   {len(scan.insns):4d} insns  "
                  f"{scan.size_bytes:4d} bytes  {len(scan.calls)} call dep(s)")
            record = LiftRecord(entry=entry, name=f"lifted_{entry:04X}",
                                status="LIFTED", size_bytes=scan.size_bytes,
                                instructions=len(scan.insns))
            if emit_dir:
                source = emit_hook(scan, rt.mem.rb)
                (emit_dir / f"lifted_{entry:04X}.py").write_text(source,
                                                                 encoding="utf-8")
        else:
            refused += 1
            print(f"${entry:04X}  REFUSED  {scan.reason}: {scan.detail}")
            record = LiftRecord(entry=entry, name=f"refused_{entry:04X}",
                                status="REFUSED", refusal_reason=scan.reason)
        manifest.update(record)

    print(f"\ncensus: {lifted} liftable, {refused} refused "
          f"({100.0 * lifted / max(1, lifted + refused):.1f}% liftable)")
    if args.manifest:
        manifest.save(args.manifest)
        print(f"manifest -> {args.manifest} ({manifest.summary()})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
