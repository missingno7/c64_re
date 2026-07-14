"""HookRegistry: original addresses -> Python replacements (dos_re pattern).

The migration path is identical to dos_re's: run the original in the VM,
understand a routine, register a replacement at its address, let the rest
of the original keep running.  Keys are 16-bit PCs (no segmentation on the
6502); one address gets exactly one replacement — duplicates fail fast.

``C64_RE_DISABLE_HOOKS=a1b2,c3d4`` disables individual hooks without code
changes (A/B checks, bisecting a suspected-wrong hook).
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable

from .cpu import CPU6502

Hook = Callable[[CPU6502], None]


@dataclass(frozen=True)
class Replacement:
    pc: int
    name: str
    handler: Hook


class HookRegistry:
    def __init__(self) -> None:
        self.replacements: dict[int, Replacement] = {}

    def replace(self, pc: int, name: str):
        key = pc & 0xFFFF

        def deco(fn: Hook) -> Hook:
            existing = self.replacements.get(key)
            if existing is not None:
                raise ValueError(
                    f"duplicate replacement at ${key:04X} "
                    f"({existing.name!r} then {name!r})"
                )
            self.replacements[key] = Replacement(key, name, fn)
            return fn
        return deco

    def install(self, cpu: CPU6502) -> None:
        disabled = _parse_disabled(os.environ.get("C64_RE_DISABLE_HOOKS", ""))
        for key, repl in self.replacements.items():
            if key in disabled:
                continue
            cpu.replacement_hooks[key] = repl.handler
            cpu.hook_names[key] = repl.name


def _parse_disabled(text: str) -> set[int]:
    out: set[int] = set()
    for token in text.replace(";", ",").split(","):
        token = token.strip().lstrip("$")
        if token:
            out.add(int(token, 16) & 0xFFFF)
    return out


registry = HookRegistry()


# ---- live-code signature guards (ported dos_re pattern) -----------------------------
def code_matches(cpu: CPU6502, pc: int, expected: bytes | tuple[bytes, ...]) -> bool:
    variants = expected if isinstance(expected, tuple) else (expected,)
    return any(
        all(cpu.mem.rb((pc + i) & 0xFFFF) == b for i, b in enumerate(sig))
        for sig in variants
    )


def self_disable_if_patched(cpu: CPU6502, pc: int,
                            expected: bytes | tuple[bytes, ...], name: str) -> bool:
    """Fail fast when a hook entry no longer matches the original bytes
    (an unknown runtime-patched variant).  Returns False when fine."""
    if code_matches(cpu, pc, expected):
        return False
    variants = expected if isinstance(expected, tuple) else (expected,)
    max_len = max(len(sig) for sig in variants)
    live = bytes(cpu.mem.rb((pc + i) & 0xFFFF) for i in range(max_len))
    if all(b == 0 for b in live):  # synthetic-fixture case: no live signature
        return False
    expected_text = " or ".join(sig.hex(" ") for sig in variants)
    raise RuntimeError(
        f"hook {name} at ${pc:04X} saw runtime-patched code; "
        f"live bytes {live.hex(' ')} != expected {expected_text}"
    )


def call_installed_hook_like_jsr(cpu: CPU6502, pc: int,
                                 default_handler: Hook, return_pc: int) -> None:
    """Run a child hook with original JSR stack semantics (dos_re's
    call_installed_hook_like_near_call, 6502 flavor): pushes return_pc-1,
    jumps to the child's real address, and routes through the live
    verifier when one is installed."""
    handler = cpu.replacement_hooks.get(pc, default_handler)
    name = cpu.hook_names.get(pc, getattr(handler, "__name__", "replacement"))
    ret = (return_pc - 1) & 0xFFFF
    cpu.push((ret >> 8) & 0xFF)
    cpu.push(ret & 0xFF)
    cpu.s.pc = pc & 0xFFFF
    verifier = cpu.hook_verifier
    if verifier is not None and pc not in cpu.hook_verifier_passthrough:
        verifier(cpu, pc, handler, name)
    else:
        handler(cpu)
