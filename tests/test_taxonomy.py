"""Smoke tests for ``c64_re.hook_taxonomy`` (role-based hook classification,
ported from dos_re — flat 16-bit PCs instead of CS:IP tuples)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from c64_re.hook_taxonomy import CATEGORIES, HookTaxonomy  # noqa: E402


def test_hook_taxonomy_defaults_to_glue_and_groups_registry():
    tax = HookTaxonomy(
        checkpoints={0x080D: "frame top"},
        env_waits={0x0913: "raster-line wait (frame pacing)"},
    )
    assert tax.classify(0x080D) == "checkpoint"
    assert tax.classify(0x0913) == "env_wait"
    assert tax.classify(0xBEEF) == "glue"

    grouped = tax.classify_registry([0xBEEF, 0x080D])
    assert grouped["checkpoint"] == [0x080D]
    assert grouped["glue"] == [0xBEEF]
    assert grouped["debug_probe"] == []


def test_hook_taxonomy_debug_probes_and_categories():
    tax = HookTaxonomy(debug_probes={0x0C00: "frame digest probe"})
    assert tax.classify(0x0C00) == "debug_probe"
    grouped = tax.classify_registry([0x0C00])
    assert set(grouped) == set(CATEGORIES)
    assert grouped["debug_probe"] == [0x0C00]
