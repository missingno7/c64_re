#!/usr/bin/env python3
"""Structural lint for the c64_re framework repo.

Two checks:

1. Every Python file parses (syntax).
2. The framework core stays game-agnostic and dependency-free: ``c64_re/``
   may import only the Python stdlib and other ``c64_re`` modules — never a
   game adapter, never a third-party package.  Anything that knows a specific
   game's addresses, filenames, or formats belongs in a game adapter built
   *on top of* this repo, never inside ``c64_re/``.

The single exception is the FRONTEND RING (``player.py``): the viewer-facing
module may import numpy/pygame — lazily, so ``import c64_re`` itself never
pulls them in.

Game adapters that vendor this framework should extend PACKAGE_ROOTS with
their own package and add a rule that ``c64_re`` does not import it (see the
dos_re original: tools/lint.py, and its pre2_port ancestor scripts/lint.py).

Origin: ported from dos_re's ``tools/lint.py`` (frontend ring trimmed to the
modules that exist here; adapter roots pre-seeded as a commented example).
"""
from __future__ import annotations

import ast
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_ROOTS = [
    ROOT / "c64_re",
    ROOT / "tools",
    ROOT / "tests",
    # Adapter packages add themselves here so their files are syntax-checked
    # and covered by layer rules, e.g.:
    # ROOT.parent / "stix_port" / "stix",
]

# Modules the framework core is allowed to import besides the stdlib.
CORE_ALLOWED_PREFIXES = ("c64_re",)

# Optional third-party backends the *non-core* layers may use.
KNOWN_OPTIONAL = ("numpy", "pygame", "pytest", "cffi")

# The FRONTEND RING: the viewer-facing modules inside the package that may use
# the optional viewer dependencies (numpy + pygame).  ``import c64_re`` itself
# must never pull them in — player.py keeps its imports lazy.
FRONTEND_RING = {"player.py", "audio_sink.py"}
FRONTEND_ALLOWED = ("numpy", "pygame")


def _stdlib_names() -> set[str]:
    return set(sys.stdlib_module_names)


def iter_py_files():
    for root in PACKAGE_ROOTS:
        if not root.exists():
            continue
        for p in root.rglob("*.py"):
            if "__pycache__" not in p.parts:
                yield p


def collect_errors() -> list[str]:
    """Run both checks over PACKAGE_ROOTS; returns human-readable errors."""
    stdlib = _stdlib_names()
    errors: list[str] = []
    for path in iter_py_files():
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as exc:
            errors.append(f"{path.relative_to(ROOT)}: syntax error: {exc}")
            continue
        if not path.is_relative_to(ROOT / "c64_re"):
            continue
        for node in ast.walk(tree):
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                if node.level:  # relative import stays inside c64_re
                    continue
                if node.module:
                    names = [node.module]
            for name in names:
                top = name.split(".")[0]
                if top in stdlib or any(name == p or name.startswith(p + ".") for p in CORE_ALLOWED_PREFIXES):
                    continue
                if path.name in FRONTEND_RING and top in FRONTEND_ALLOWED:
                    continue
                errors.append(
                    f"{path.relative_to(ROOT)}:{node.lineno}: c64_re core must stay "
                    f"stdlib-only and game-agnostic; imports {name!r}"
                )
    return errors


def main() -> int:
    errors = collect_errors()
    if errors:
        print("lint failed:")
        for err in errors:
            print("  " + err)
        return 1
    print("lint passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
