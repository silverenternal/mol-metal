"""Wf-A61 (2026-10-17) W2 — JSON-readback regression tests for the
CFM round 9 audit artifacts.

* §6.55g `configs_method_audit.json` — cross-YAML rk4 config audit.
  Asserts ``n_configs_total >= 15`` and every entry is a real path
  to a file on disk.

* §6.52d `vina_block_audit.json` — Vina panel BLOCKED-on-every-precious-metal.
  Asserts ``n_rows == 6`` and every row's ``status == "BLOCKED"``.

These tests are the Wf-A61 closeout ledger: any future wave that
mutates the audit JSONs has to update these tests, which means the
change is explicit rather than silent.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


AUDIT_DIR = (
    Path(__file__).resolve().parent.parent
    / "reports"
    / "wfa61_cfm_round9"
)


def _load(name: str) -> dict:
    path = AUDIT_DIR / name
    if not path.exists():
        pytest.skip(f"{name} not on disk yet (audit artifact pending)")
    return json.loads(path.read_text())


def test_configs_method_audit_rk4_count() -> None:
    """§6.55g — at least 15 rk4 cohorts MEASURED on disk."""
    data = _load("configs_method_audit.json")
    assert data["method"] == "rk4"
    assert data["n_configs_total"] >= 15
    assert len(data["configs"]) == data["n_configs_total"]
    # Every entry must point to a real file on disk.
    for cfg in data["configs"]:
        # The audit lists repo-relative paths; resolve from repo root.
        path = Path(__file__).resolve().parent.parent.parent / cfg
        assert path.exists(), f"missing audit-listed config: {cfg}"


def test_vina_block_audit_n_rows() -> None:
    """§6.52d — 6 precious metals × BLOCKED status."""
    data = _load("vina_block_audit.json")
    assert data["n_rows"] == 6
    assert data["all_blocked"] is True
    for row in data["rows"]:
        assert row["status"] == "BLOCKED"


def test_vina_block_audit_expected_metals() -> None:
    """§6.52d — Pt / Pd / Ir / Ru / Os / Au all BLOCKED."""
    data = _load("vina_block_audit.json")
    metals = {row["metal"] for row in data["rows"]}
    expected = {"Pt", "Pd", "Ir", "Ru", "Os", "Au"}
    assert expected.issubset(metals), (
        f"missing expected metals: {expected - metals}"
    )


def test_hn_resolution_index_exists() -> None:
    """Wf-A61 closeout ledger exists."""
    path = AUDIT_DIR / "HN_RESOLUTION_INDEX.md"
    assert path.exists()
    text = path.read_text()
    # Sanity-check the file references the wave + audit artifacts.
    assert "Wf-A61" in text
    assert "configs_method_audit.json" in text
    assert "vina_block_audit.json" in text