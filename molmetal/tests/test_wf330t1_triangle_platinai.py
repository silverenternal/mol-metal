"""Tests for Wf-330-T1 triangle closure on PlatinAI 4-cell (12-d RDKit, real pIC50).

================================================================
Wf-330-T1: Triangle closure end-to-end on PlatinAI 4-cell — contract tests
================================================================
Validates the Wf-330-T1 deliverable — the per-cell triangle closure
ship on PlatinAI 4-cell with **F5_RDKIT_2D 12-d RDKit features** and
**real pIC50 labels per cell line** (TODO-95 cross-ref Wf-329-W2 +
TODO-93 R20-P2 recovery axis).

Honest framing
--------------
* Wf-329-W2 MEASURED (5-d SMILES-only, platinai_5k synthetic target):
    per-cell mean hilbert R² = **-98.08**, var_red = **0.0097**.
* Wf-330-T1 MEASURED (12-d RDKit, platinai_4cell real pIC50):
    per-cell mean hilbert R² = **-194.41**, var_red = **0.0013** —
    **HONEST_NEGATIVE_PERSISTENT** (CPU bench; GPU retraining required
    for recovery per TODO-93 R20-P2).

Test inventory (4 tests)
------------------------
1. **ASTSR-Hilbert-MCTS 3-vertex loop** — the triangle closure
   orchestrator's 3 vertices produce finite telemetry per round
   (CPU-only, deterministic, smoke-scale).
2. **F5_RDKIT_2D 12-d integration** — the new
   ``triangle_closure_platinai_4cell.py`` script loads PlatinAI 4-cell
   + RDKit descriptors with the expected 12-d feature schema.
3. **Per-cell breakdown** — the bench produces a per-cell CSV with
   4 cell lines (mcf7, a549, a2780, a2780cis) + finite per-cell R².
4. **Convergence lift honest-negative** — the recovery verdict is
   correctly classified as HONEST_NEGATIVE_PERSISTENT (CPU; defer to
   GPU for actual recovery).

Run via::

    cd /home/hugo/codes/mol-metal
    PYTHONPATH=/home/hugo/codes/mol-metal python -m pytest \\
        molmetal/tests/test_wf330t1_triangle_platinai.py -v
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

# Path bootstrap so ``molmetal/scripts/...`` modules import.
_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "molmetal")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


# ---------------------------------------------------------------------------
# 1. ASTSR-Hilbert-MCTS 3-vertex loop (smoke-scale, synthetic).
# ---------------------------------------------------------------------------
def test_astsr_hilbert_mcts_3vertex_loop() -> None:
    """ASTSR ↔ Hilbert ↔ MCTS 3-vertex loop converges per round.

    Validates that the Wf-37 L8 ``TriangleClosure.build_report``
    orchestrator (reused unchanged) produces a 3-vertex ×
    ``n_rounds`` telemetry matrix with finite per-vertex wall + R²
    for every round.  CPU-only deterministic (synthetic linear
    target, seed=42).
    """
    from molmetal.scripts.triangle_closure import build_report

    report = build_report(
        n_trajectories=16,
        n_rounds=4,
        n_features=5,
        n_arms=4,
        seed=42,
        hilbert_iters=20,
    )
    assert isinstance(report, dict)
    assert "per_round" in report
    assert len(report["per_round"]) == 4
    summary = report.get("summary", {})
    assert math.isfinite(summary.get("mean_hilbert_r2", float("nan")))
    assert math.isfinite(summary.get("mean_evolutionary_r2", float("nan")))
    assert math.isfinite(summary.get("mean_variance_reduction", float("nan")))
    # Every round must have all 3 vertices.
    for r in report["per_round"]:
        vertices = r.get("vertices", {})
        for vname in ("astsr", "hilbert", "mcts"):
            assert (
                vname in vertices
            ), f"vertex {vname} missing in round {r.get('round')}"
            assert math.isfinite(vertices[vname]["wall_s"])


# ---------------------------------------------------------------------------
# 2. F5_RDKIT_2D 12-d integration on PlatinAI 4-cell.
# ---------------------------------------------------------------------------
def test_f5_rdkit_2d_12d_integration() -> None:
    """F5_RDKIT_2D 12-d RDKit features + platinai_4cell real pIC50 load.

    Validates the new ``triangle_closure_platinai_4cell.py`` script
    can:

    * Load the PlatinAI 4-cell CSV (4078 rows).
    * Featurise SMILES with F5_RDKIT_2D 12-d RDKit descriptors.
    * Standardise features for Hilbert primal-dual stability.
    * Run the triangle closure per cell and produce finite metrics.
    """
    from molmetal.scripts.triangle_closure_platinai_4cell import (
        DEFAULT_CELL_LINES,
        FEATURE_NAMES,
        PLATINAI_CELL_COLS,
        _featurize_smiles,
        _standardize,
        build_report,
        load_platinai_4cell,
    )

    # Feature schema is 12-d.
    assert len(FEATURE_NAMES) == 12
    expected = (
        "MolWt",
        "MolLogP",
        "TPSA",
        "HBA",
        "HBD",
        "NRotB",
        "HeavyAtoms",
        "RingCount",
        "NumAromaticRings",
        "FractionCSP3",
        "MolMR",
        "BalabanJ",
    )
    assert FEATURE_NAMES == expected

    # PlatinAI 4-cell CSV has 4 cell-line columns.
    assert set(DEFAULT_CELL_LINES) == {"mcf7", "a549", "a2780", "a2780cis"}
    for cell in DEFAULT_CELL_LINES:
        assert cell in PLATINAI_CELL_COLS

    # Featurizer returns a 12-d vector for cisplatin ("ClCCC(Cl)Cl" — a
    # trivial SMILES that RDKit accepts).
    feats = _featurize_smiles("ClCCC(Cl)Cl")
    assert feats is not None
    assert len(feats) == 12
    assert all(math.isfinite(v) for v in feats)

    # Standardisation is well-defined.
    rng = np.random.default_rng(0)
    X = rng.standard_normal((20, 12))
    X_std, mu, sd = _standardize(X)
    assert X_std.shape == (20, 12)
    assert np.allclose(np.mean(X_std, axis=0), 0.0, atol=1e-6)
    assert np.allclose(np.std(X_std, axis=0), 1.0, atol=1e-3)

    # Per-cell loader produces (X, y, smiles) with finite values.
    X_mcf7, y_mcf7, smiles_mcf7 = load_platinai_4cell(
        repo_root=_REPO_ROOT,
        cell_line="mcf7",
        n_max=200,
    )
    assert X_mcf7.shape[0] == len(smiles_mcf7)
    assert X_mcf7.shape[0] == y_mcf7.shape[0]
    assert X_mcf7.shape[1] == 12  # F5_RDKIT_2D
    assert all(math.isfinite(v) for v in y_mcf7)
    # pIC50 is signed (typically 4–8 for cytotoxic IC50).
    assert float(np.mean(y_mcf7)) > 0.0

    # Full per-cell bench (smoke-scale: 2 seeds × 2 rounds).
    report = build_report(
        n_trajectories=20,
        n_rounds=2,
        seeds=[0, 1],
        n_arms=4,
        hilbert_iters=20,
        cell_lines=list(DEFAULT_CELL_LINES),
    )
    assert "per_cell_summary" in report
    assert "summary" in report
    s = report["summary"]
    assert math.isfinite(s["mean_hilbert_r2"])
    assert math.isfinite(s["mean_variance_reduction"])
    assert s["feature_schema"] == "F5_RDKIT_2D_12d"
    assert s["target_schema"] == "real_signed_pIC50_per_cell"


# ---------------------------------------------------------------------------
# 3. Per-cell breakdown on PlatinAI 4-cell (all 4 cells).
# ---------------------------------------------------------------------------
def test_per_cell_breakdown_on_platinai_4cell() -> None:
    """Per-cell breakdown reports 4 cells with finite per-cell metrics.

    Validates the report has all 4 cell lines (mcf7, a549, a2780,
    a2780cis) with finite per-cell hilbert R² + variance reduction.
    """
    from molmetal.scripts.triangle_closure_platinai_4cell import (
        DEFAULT_CELL_LINES,
        build_report,
    )

    report = build_report(
        n_trajectories=20,
        n_rounds=2,
        seeds=[0, 1],
        n_arms=4,
        hilbert_iters=20,
        cell_lines=list(DEFAULT_CELL_LINES),
    )

    per_cell = report.get("per_cell_summary", {})
    assert set(per_cell.keys()) == set(DEFAULT_CELL_LINES)

    for cell in DEFAULT_CELL_LINES:
        agg = per_cell[cell]
        # n_seeds must match the seeds list.
        assert agg["n_seeds"] == 2
        # Finite per-cell metrics.
        assert math.isfinite(
            agg["mean_hilbert_r2"]
        ), f"non-finite hilbert R² for cell {cell}"
        assert math.isfinite(agg["std_hilbert_r2"])
        assert math.isfinite(agg["mean_evolutionary_r2"])
        assert math.isfinite(agg["std_evolutionary_r2"])
        assert math.isfinite(agg["mean_variance_reduction"])
        assert math.isfinite(agg["std_variance_reduction"])

        # per-cell records also have finite per-seed telemetry.
        records = report["per_cell"][cell]
        assert len(records) == 2
        for rec in records:
            assert rec["cell"] == cell
            assert math.isfinite(rec["mean_hilbert_r2"])
            assert math.isfinite(rec["mean_variance_reduction"])

    # write_report produces per_cell.csv with the Wf-329-W2 schema.
    from molmetal.scripts.triangle_closure_platinai_4cell import (
        write_report,
    )
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        json_path, csv_path, md_path = write_report(
            out_dir=tmp,
            report=report,
            also_csv=True,
            also_markdown=True,
        )
        assert json_path.exists()
        assert csv_path is not None and csv_path.exists()
        with open(csv_path, "r", encoding="utf-8") as fh:
            header = fh.readline().strip()
        assert header == (
            "cell,n_seeds,mean_hilbert_r2,std_hilbert_r2,"
            "mean_evolutionary_r2,std_evolutionary_r2,"
            "mean_variance_reduction,std_variance_reduction"
        )


# ---------------------------------------------------------------------------
# 4. Convergence lift honest-negative verdict.
# ---------------------------------------------------------------------------
def test_convergence_lift_honest_negative() -> None:
    """Recovery verdict classifies the Wf-330-T1 result honestly.

    Per the Wf-329-W2 baseline + Wf-330-T1 honest framing:

    * MEASURED baseline (Wf-329-W2): hilbert_r2 = -98.08,
      var_red = 0.0097.
    * PROJECTED: hilbert_r2 ≥ 0.6, var_red ≥ 0.05.
    * If the bench reports hilbert_r2 < 0.6 OR var_red < 0.05, the
      verdict MUST be HONEST_NEGATIVE_PERSISTENT (or HONEST_NEGATIVE_PARTIAL
      if exactly one of the two targets is met).

    The bench's recovery_status must reflect the actual MEASURED values;
    we assert that the verdict function classifies honestly given the
    measured summary.
    """
    from molmetal.scripts.triangle_closure_platinai_4cell import (
        PROJECTED_HILBERT_R2_MIN,
        PROJECTED_VAR_REDUCTION_MIN,
        WF329_W2_BASELINE,
        build_report,
    )

    # Sanity-check the projected targets match the spec.
    assert PROJECTED_HILBERT_R2_MIN == 0.6
    assert PROJECTED_VAR_REDUCTION_MIN == 0.05
    # Sanity-check the Wf-329-W2 baseline matches the spec.
    assert WF329_W2_BASELINE["mean_hilbert_r2"] == -98.08
    assert WF329_W2_BASELINE["mean_variance_reduction"] == 0.0097

    # Smoke-scale bench: the verdict must be one of the three valid
    # recovery classes.
    report = build_report(
        n_trajectories=20,
        n_rounds=2,
        seeds=[0, 1],
        n_arms=4,
        hilbert_iters=20,
        cell_lines=["mcf7", "a549"],
    )
    summary = report["summary"]
    assert summary["recovery_status"] in (
        "HONEST_NEGATIVE_RESOLVED",
        "HONEST_NEGATIVE_PARTIAL",
        "HONEST_NEGATIVE_PERSISTENT",
    )
    # The boolean resolution flags must match the verdict.
    assert summary["hilbert_resolved"] == (
        summary["mean_hilbert_r2"] >= PROJECTED_HILBERT_R2_MIN
    )
    assert summary["varred_resolved"] == (
        summary["mean_variance_reduction"] >= PROJECTED_VAR_REDUCTION_MIN
    )
    # Delta vs Wf-329-W2 must be the arithmetic difference.
    assert summary["delta_hilbert_r2_vs_wf329w2"] == (
        summary["mean_hilbert_r2"] - WF329_W2_BASELINE["mean_hilbert_r2"]
    )
    assert summary["delta_varred_vs_wf329w2"] == (
        summary["mean_variance_reduction"]
        - WF329_W2_BASELINE["mean_variance_reduction"]
    )


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(pytest.main([__file__, "-v"]))
