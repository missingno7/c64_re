"""The structural lint (tools/lint.py) must pass on the real tree.

Runs the lint's check function in-process against the repository so a
layering break (a third-party import in the core, a game adapter leaking
into c64_re/) fails the test suite, not just the standalone tool.
"""
import importlib.util
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]


def _load_lint():
    spec = importlib.util.spec_from_file_location("c64_re_tools_lint", ROOT / "tools" / "lint.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_lint_passes_on_the_real_tree():
    lint = _load_lint()
    errors = lint.collect_errors()
    assert errors == [], "tools/lint.py failed:\n" + "\n".join(errors)


def test_lint_frontend_ring_is_only_player():
    lint = _load_lint()
    assert lint.FRONTEND_RING == {"player.py"}
    assert set(lint.FRONTEND_ALLOWED) <= set(lint.KNOWN_OPTIONAL)


def test_lint_flags_a_third_party_import_in_the_core(tmp_path):
    """The check itself must bite: a synthetic core file importing numpy fails."""
    lint = _load_lint()
    core = tmp_path / "c64_re"
    core.mkdir()
    (core / "bad.py").write_text("import numpy\n", encoding="utf-8")
    old_root, old_pkgs = lint.ROOT, lint.PACKAGE_ROOTS
    try:
        lint.ROOT = tmp_path
        lint.PACKAGE_ROOTS = [core]
        errors = lint.collect_errors()
    finally:
        lint.ROOT, lint.PACKAGE_ROOTS = old_root, old_pkgs
    assert len(errors) == 1 and "numpy" in errors[0]
