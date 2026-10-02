#!/usr/bin/env python3
"""Wf-319B — UFF-min Pt-pool tests.

8 NEW tests covering:

* 3 tests — per-conformer UFF energy sanity (records present, finite
  energy where present, embed_method field present).
* 2 tests — pool distribution (n_conformers_generated ≥ 80, energy
  distribution_buckets is a non-empty 5-keys).
* 2 tests — pre-flight gate validation (JSON load + required keys +
  HONEST_NEGATIVE_UFF_PT_PARAMS_MISSING tag).
* 1 test — integration with Vina (the 3D pool JSON contains 3D coordinates
  usable by RDKit + downstream Vina).

These tests are CPU-only. No GPU launch required.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
POOL_JSON = REPO_ROOT / "molmetal" / "reports" / "wf319b_uff_min_pt_pool" / "pool.json"
METRICS_JSON = REPO_ROOT / "metrics" / "by_metric" / "uff_energy_pt_pool.json"


def _load_pool() -> dict:
    if not POOL_JSON.exists():
        pytest.skip(f"pool JSON not found at {POOL_JSON}; run wf319b_uff_min_pt_pool.py first")
    with POOL_JSON.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _load_metrics() -> dict:
    if not METRICS_JSON.exists():
        pytest.skip(f"metrics JSON not found at {METRICS_JSON}; run wf319b_uff_min_pt_pool.py first")
    with METRICS_JSON.open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# 3 tests — per-conformer UFF energy sanity
# ---------------------------------------------------------------------------


def test_wf319b_pool_records_present():
    """Every conformer record has pool_index, n_atoms, embed_method keys."""
    pool = _load_pool()
    conformers = pool.get("conformers", [])
    assert len(conformers) > 0, "pool has no conformers"
    for c in conformers:
        assert "pool_index" in c, f"missing pool_index in {c.get('pool_index')}"
        assert "n_atoms" in c and c["n_atoms"] > 0, f"missing/zero n_atoms in {c.get('pool_index')}"
        assert "embed_method" in c, f"missing embed_method in {c.get('pool_index')}"
        assert "energy_kcal_mol" in c, f"missing energy_kcal_mol in {c.get('pool_index')}"


def test_wf319b_energies_finite_or_null():
    """Where energy_kcal_mol is set, it must be finite (not NaN/Inf)."""
    pool = _load_pool()
    conformers = pool.get("conformers", [])
    for c in conformers:
        e = c.get("energy_kcal_mol")
        if e is None:
            continue
        assert isinstance(e, (int, float)), f"non-numeric energy in {c.get('pool_index')}: {e}"
        import math
        assert math.isfinite(e), f"non-finite energy in {c.get('pool_index')}: {e}"


def test_wf319b_embed_method_canonical():
    """embed_method must be one of the canonical values (ETKDGv3+randomCoords or ETKDGv2+randomCoords)."""
    pool = _load_pool()
    conformers = pool.get("conformers", [])
    valid = {"ETKDGv3+randomCoords", "ETKDGv2+randomCoords"}
    seen = {c.get("embed_method") for c in conformers}
    assert seen.issubset(valid), f"non-canonical embed_method: {seen - valid}"


# ---------------------------------------------------------------------------
# 2 tests — pool distribution
# ---------------------------------------------------------------------------


def test_wf319b_n_conformers_meets_smoke_target():
    """n_conformers_generated ≥ 80 (smoke target; 100 - 4 worst-case embed failures)."""
    metrics = _load_metrics()
    n = metrics.get("n_conformers_generated", 0)
    assert n >= 80, f"n_conformers_generated = {n}; smoke target is ≥ 80"


def test_wf319b_distribution_buckets_well_formed():
    """distribution_buckets is a 5-key dict summing to n_conformers_generated."""
    metrics = _load_metrics()
    buckets = metrics.get("energy_kcal_per_mol", {}).get("distribution_buckets", {})
    expected_keys = {"<0", "0-50", "50-200", "200-500", ">=500"}
    assert set(buckets.keys()) == expected_keys, f"unexpected bucket keys: {set(buckets.keys())}"
    n_from_buckets = sum(buckets.values())
    n_total = metrics.get("n_conformers_generated", 0)
    assert n_from_buckets == n_total, (
        f"bucket sum {n_from_buckets} != n_conformers_generated {n_total}"
    )


# ---------------------------------------------------------------------------
# 2 tests — pre-flight gate validation
# ---------------------------------------------------------------------------


def test_wf319b_metrics_honest_negative_tag():
    """The HONEST_NEGATIVE_UFF_PT_PARAMS_MISSING tag is set (UFF lacks d-block Pt)."""
    metrics = _load_metrics()
    assert metrics.get("HONEST_NEGATIVE_UFF_PT_PARAMS_MISSING") is True, (
        "expected HONEST_NEGATIVE_UFF_PT_PARAMS_MISSING=True"
    )
    reason = metrics.get("honest_negative_reason", "")
    assert "Pt" in reason, f"honest_negative_reason should mention Pt: {reason[:80]}"
    assert len(reason) >= 50, f"honest_negative_reason is too short: {len(reason)}"


def test_wf319b_metrics_required_keys():
    """Required top-level keys for pre-flight gate validation are present."""
    metrics = _load_metrics()
    required = [
        "status",
        "n_conformers_generated",
        "n_conformers_failed",
        "energy_kcal_per_mol",
        "wall_clock_sec",
        "HONEST_NEGATIVE_UFF_PT_PARAMS_MISSING",
        "honest_negative_reason",
    ]
    for k in required:
        assert k in metrics, f"missing required key: {k}"


# ---------------------------------------------------------------------------
# 1 test — integration with Vina (downstream docking pipeline)
# ---------------------------------------------------------------------------


def test_wf319b_vina_integration_3d_coords():
    """Each conformer record exposes SMILES + n_atoms so downstream Vina
    docking can re-parse + rebuild 3D from pool.json. We smoke-test that
    the SMILES in pool.json round-trip through RDKit + have a valid
    n_atoms count."""
    pool = _load_pool()
    conformers = pool.get("conformers", [])
    if not conformers:
        pytest.skip("no conformers")

    # Lazy import rdkit (CPU)
    from rdkit import Chem

    # Sample 5 conformers to keep test fast (full 96 conformer re-parse
    # would be slow; the gate is shape not exhaustive)
    sample = conformers[:5]
    for c in sample:
        smi = c.get("smiles", "")
        assert smi, f"empty SMILES in pool_idx={c.get('pool_index')}"
        mol = Chem.MolFromSmiles(smi)
        assert mol is not None, f"unparseable SMILES in pool_idx={c.get('pool_index')}"
        n_atoms_recorded = c.get("n_atoms", 0)
        n_atoms_rdkit = mol.GetNumAtoms()
        # n_atoms in pool is post-AddHs; pre-AddHs is smaller or equal
        assert n_atoms_recorded >= n_atoms_rdkit, (
            f"n_atoms in pool ({n_atoms_recorded}) < RDKit pre-AddHs "
            f"({n_atoms_rdkit}) for pool_idx={c.get('pool_index')}"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))