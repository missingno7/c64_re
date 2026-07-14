"""Guard against the NameError class of latent bugs: a name used in a never-exercised branch that was never
imported or defined. A lightweight AST check — no external linter dependency — over the framework package.

It over-approximates "defined" (module imports + every def/class + every assigned name anywhere + all arg
names + comprehension/except/with targets + builtins), so it only flags names defined *nowhere* in the
module — which is virtually always a real NameError waiting in an unexercised path.

Game adapters built on this framework should point the same check at their own recovered layers
(see the dos_re original: tests/test_no_undefined_names.py, and its pre2_port ancestor scanning
recovered/checkpoints/bridge/codecs).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import ast  # noqa: E402
import builtins  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
# the shipped framework package (tools are scanned too — they must stay importable)
DIRS = ("c64_re", "tools")
_BUILTINS = set(dir(builtins)) | {"__file__", "__name__", "__doc__", "__class__", "__all__", "self", "cls"}


def _modules():
    files: list[Path] = []
    for d in DIRS:
        root = ROOT / d
        if root.exists():
            files += sorted(root.rglob("*.py"))
    return [f for f in files if "__pycache__" not in f.parts]


def _undefined(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"), str(path))
    defined = set(_BUILTINS)
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                defined.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.ImportFrom):
            for a in n.names:
                defined.add(a.asname or a.name)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(n.name)
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            defined.add(n.id)
        elif isinstance(n, ast.arg):
            defined.add(n.arg)
        elif isinstance(n, ast.Global):
            defined.update(n.names)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            defined.add(n.name)
        elif isinstance(n, ast.withitem) and isinstance(n.optional_vars, ast.Name):
            defined.add(n.optional_vars.id)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
    return sorted(used - defined)


def test_no_undefined_names():
    # a plain loop (not pytest.mark.parametrize as in the dos_re original) so
    # the stdlib runner (c64_re.testing) can run this test too
    failures = []
    for path in _modules():
        undefined = _undefined(path)
        if undefined:
            failures.append(f"{path.relative_to(ROOT)}: undefined name(s) {undefined}")
    assert not failures, (
        "unimported/undefined names — a NameError in some branch:\n" + "\n".join(failures)
    )
