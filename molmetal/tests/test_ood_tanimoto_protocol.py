"""Tests for the Tanimoto<0.4 OOD protocol on tmQM d-block (Wf-57 Phase 1).

================================================================
Wf-57 Phase 1 — Tanimoto<0.4 OOD protocol contract.

This file ships the **15 pytest contracts** that pin the behaviour
of :mod:`molmetal.scripts.bench_ood_tanimoto_protocol`:

Defaults / contract surface
 1. test_module_imports                  — script imports cleanly
 2. test_default_threshold_is_0_4       — Tanimoto threshold = 0.4
 3. test_default_train_ratio_is_0_8     — Train ratio = 0.8
 4. test_default_output_dir             — Default output dir is wf57_tanimoto
 5. test_default_tmqm_csv_exists        — tmQM CSV is present
 6. test_cluster_column_is_metal_center — Cluster col = metal_center
 7. test_smiles_column_is_smiles        — SMILES col = smiles
 8. test_verification_target_is_0_8     — Verification target >= 80%

Mechanics
 9. test_load_tmqm_pool_loads           — Pool loader returns per-cluster dict
10. test_split_train_test_preserves_total — Split preserves total
11. test_split_train_test_uses_train_ratio — Train fraction honoured
12. test_benchmark_cluster_computes_matrices — Per-cluster matrices
13. test_run_protocol_smoke             — Full protocol smoke run (small caps)
14. test_run_protocol_verification_passes — Verification PASS at default
15. test_save_protocol_writes_artifacts — JSON + CSV + Markdown written

All tests are CPU-only, deterministic, and complete in < 90 s
combined.  Smoke runs use ``--max-samples-per-cluster 8`` for
sub-second per-cluster CI runs.

Run with::

    uv run --no-sync pytest -q \\
        molmetal/tests/test_ood_tanimoto_protocol.py --tb=short
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest


# ---------------------------------------------------------------------------
# Path bootstrap
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_DIR = PROJECT_ROOT / "molmetal" / "scripts"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


from molmetal.scripts import bench_ood_tanimoto_protocol  # noqa: E402
from molmetal.scripts.bench_ood_tanimoto_protocol import (  # noqa: E402
    CLUSTER_COLUMN,
    CODE_COLUMN,
    DEFAULT_MAX_SAMPLES_PER_CLUSTER,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TANIMOTO_THRESHOLD,
    DEFAULT_TMQM_CSV,
    DEFAULT_TRAIN_RATIO,
    FP_NBITS,
    FP_RADIUS,
    OOD_VERIFICATION_TARGET,
    SMILES_COLUMN,
    _frac_below,
    _have_rdkit,
    _safe_div,
    _utcnow_iso,
    benchmark_cluster,
    compute_cluster_tanimoto_matrix,
    load_tmqm_pool,
    pairs_from_matrix,
    run_ood_tanimoto_protocol,
    save_ood_tanimoto_protocol,
    split_train_test,
)


# ---------------------------------------------------------------------------
# RDKit availability probe
# ---------------------------------------------------------------------------
requires_rdkit = pytest.mark.skipif(
    not _have_rdkit(),
    reason="RDKit not installed; Tanimoto OOD protocol tests require Morgan FP",
)


# ---------------------------------------------------------------------------
# 1. Module imports
# ---------------------------------------------------------------------------
def test_module_imports() -> None:
    """The OOD protocol benchmark module imports cleanly."""
    assert bench_ood_tanimoto_protocol is not None
    assert hasattr(bench_ood_tanimoto_protocol, "run_ood_tanimoto_protocol")
    assert hasattr(bench_ood_tanimoto_protocol, "save_ood_tanimoto_protocol")
    assert hasattr(bench_ood_tanimoto_protocol, "load_tmqm_pool")
    assert hasattr(bench_ood_tanimoto_protocol, "split_train_test")
    assert hasattr(bench_ood_tanimoto_protocol, "benchmark_cluster")
    assert hasattr(bench_ood_tanimoto_protocol, "compute_cluster_tanimoto_matrix")


# ---------------------------------------------------------------------------
# 2. Default Tanimoto threshold = 0.4
# ---------------------------------------------------------------------------
def test_default_threshold_is_0_4() -> None:
    """The default Tanimoto OOD threshold is 0.4 (Pocket2Mol / TargetDiff)."""
    assert DEFAULT_TANIMOTO_THRESHOLD == 0.4


# ---------------------------------------------------------------------------
# 3. Default train ratio = 0.8
# ---------------------------------------------------------------------------
def test_default_train_ratio_is_0_8() -> None:
    """The default train fraction per cluster is 0.8 (80/20 split)."""
    assert DEFAULT_TRAIN_RATIO == 0.8


# ---------------------------------------------------------------------------
# 4. Default output dir is wf57_tanimoto
# ---------------------------------------------------------------------------
def test_default_output_dir() -> None:
    """The default output dir is reports/wf57_tanimoto/."""
    assert DEFAULT_OUTPUT_DIR.name == "wf57_tanimoto"
    assert DEFAULT_OUTPUT_DIR.parent.name == "reports"


# ---------------------------------------------------------------------------
# 5. tmQM CSV is present
# ---------------------------------------------------------------------------
def test_default_tmqm_csv_exists() -> None:
    """The default tmQM CSV (Wf-28 ship) is present on disk."""
    assert Path(DEFAULT_TMQM_CSV).is_file(), f"tmQM CSV missing at {DEFAULT_TMQM_CSV}"


# ---------------------------------------------------------------------------
# 6. Cluster column = metal_center
# ---------------------------------------------------------------------------
def test_cluster_column_is_metal_center() -> None:
    """The cluster (split) column is metal_center."""
    assert CLUSTER_COLUMN == "metal_center"


# ---------------------------------------------------------------------------
# 7. SMILES column = smiles
# ---------------------------------------------------------------------------
def test_smiles_column_is_smiles() -> None:
    """The SMILES column is smiles (matches Wf-28 CSV header)."""
    assert SMILES_COLUMN == "smiles"


# ---------------------------------------------------------------------------
# 8. Verification target >= 80%
# ---------------------------------------------------------------------------
def test_verification_target_is_0_8() -> None:
    """The verification target is >= 80% of cross-pairs below threshold."""
    assert OOD_VERIFICATION_TARGET >= 0.80
    assert OOD_VERIFICATION_TARGET <= 1.0


# ---------------------------------------------------------------------------
# 9. Pool loader returns per-cluster dict
# ---------------------------------------------------------------------------
@requires_rdkit
def test_load_tmqm_pool_loads() -> None:
    """Pool loader returns a non-empty per-cluster dict."""
    pool = load_tmqm_pool(Path(DEFAULT_TMQM_CSV), max_per_cluster=10)
    assert isinstance(pool, dict)
    assert len(pool) >= 5, (
        f"tmQM pool must have >= 5 clusters (d-block 30 metals); " f"got {len(pool)}"
    )
    # All values are lists of rows.
    for cluster, rows in pool.items():
        assert isinstance(rows, list)
        assert all(
            "smiles" in r and "metal_center" in r for r in rows
        ), f"cluster {cluster} rows must have smiles + metal_center"
        assert all(
            r["metal_center"] == cluster for r in rows
        ), f"cluster {cluster} contains foreign rows"


# ---------------------------------------------------------------------------
# 10. Split preserves total
# ---------------------------------------------------------------------------
def test_split_train_test_preserves_total() -> None:
    """Train + test = total (no rows dropped by the split)."""
    rows = [
        {"csd_code": f"CODE_{i:04d}", "smiles": "c1ccccc1", "metal_center": "Fe"}
        for i in range(10)
    ]
    train, test = split_train_test(rows, train_ratio=0.8)
    assert len(train) + len(test) == len(rows)
    assert len(train) == 8
    assert len(test) == 2


# ---------------------------------------------------------------------------
# 11. Split uses train_ratio
# ---------------------------------------------------------------------------
def test_split_train_test_uses_train_ratio() -> None:
    """The split honours the train_ratio argument (within rounding)."""
    rows = [
        {"csd_code": f"CODE_{i:04d}", "smiles": "c1ccccc1", "metal_center": "Fe"}
        for i in range(20)
    ]
    # 50/50 split — train_ratio = 0.5.
    train, test = split_train_test(rows, train_ratio=0.5)
    assert len(train) == 10
    assert len(test) == 10
    # 70/30 split.
    train, test = split_train_test(rows, train_ratio=0.7)
    assert len(train) == 14
    assert len(test) == 6


# ---------------------------------------------------------------------------
# 12. Per-cluster matrices
# ---------------------------------------------------------------------------
@requires_rdkit
def test_benchmark_cluster_computes_matrices() -> None:
    """Per-cluster benchmark yields three Tanimoto matrices + ood_audit."""
    rows = [
        {
            "csd_code": f"CODE_{i:04d}",
            "smiles": "c1ccccc1" if i < 5 else "C#CCN",
            "metal_center": "Fe",
            "coord_number": "4",
            "n_atoms": "6",
        }
        for i in range(10)
    ]
    # Lazy-load the protocol to avoid pulling RDKit at import time.
    from molmetal_lam.lam_chem.ood_protocol import TanimotoOODProtocol

    protocol = TanimotoOODProtocol(threshold=0.4, radius=2, n_bits=2048)

    result = benchmark_cluster(
        rows,
        train_ratio=0.8,
        tanimoto_threshold=0.4,
        protocol=protocol,
    )
    assert result["n_total"] == 10
    assert result["n_train"] == 8
    assert result["n_test"] == 2
    assert "matrices" in result
    assert "train_train" in result["matrices"]
    assert "train_test" in result["matrices"]
    assert "test_test" in result["matrices"]
    assert "ood_audit" in result
    # Each matrix summary has the expected keys.
    for side, summary in result["matrices"].items():
        for key in (
            "n_pairs",
            "min",
            "max",
            "mean",
            "median",
            "std",
            "fraction_below_threshold",
            "threshold",
            "matrix",
        ):
            assert key in summary, f"matrix {side} missing key {key!r}"


# ---------------------------------------------------------------------------
# 13. Full protocol smoke run (small caps)
# ---------------------------------------------------------------------------
@requires_rdkit
def test_run_protocol_smoke(tmp_path: Path) -> None:
    """Full protocol smoke run (5 samples / cluster, 2 clusters)."""
    payload = run_ood_tanimoto_protocol(
        train_ratio=0.8,
        tanimoto_threshold=0.4,
        tmqm_csv=Path(DEFAULT_TMQM_CSV),
        output_dir=tmp_path,
        max_samples_per_cluster=5,
        clusters=["Pt", "Pd"],
    )
    assert payload["n_clusters"] == 2
    assert "Pt" in payload["clusters"]
    assert "Pd" in payload["clusters"]
    assert payload["per_cluster_summary"]["Pt"]["n_total"] >= 2
    agg = payload["aggregate"]
    assert agg["n_pairs_cross"] >= 1
    assert agg["fraction_below_threshold"] >= 0.0


# ---------------------------------------------------------------------------
# 14. Verification PASS at default on a small subset
# ---------------------------------------------------------------------------
@requires_rdkit
def test_run_protocol_verification_passes(tmp_path: Path) -> None:
    """With a small subset the verification claim passes (>= 80% OOD)."""
    payload = run_ood_tanimoto_protocol(
        train_ratio=0.8,
        tanimoto_threshold=0.4,
        tmqm_csv=Path(DEFAULT_TMQM_CSV),
        output_dir=tmp_path,
        max_samples_per_cluster=20,
    )
    agg = payload["aggregate"]
    # The fraction_below_threshold should be >= verification_target.
    # On a per-cluster 80/20 split the cross-side pairs are
    # predominantly different SMILES (different CSD codes), so
    # Tanimoto<0.4 should hold for nearly every cross-pair.
    assert agg["fraction_below_threshold"] >= float(OOD_VERIFICATION_TARGET), (
        f"verification failed: fraction_below_threshold="
        f"{agg['fraction_below_threshold']:.4f} < target "
        f"{OOD_VERIFICATION_TARGET:.2f}"
    )
    assert agg["verification_passed"] is True


# ---------------------------------------------------------------------------
# 15. Save protocol writes all artefacts
# ---------------------------------------------------------------------------
@requires_rdkit
def test_save_protocol_writes_artifacts(tmp_path: Path) -> None:
    """save_ood_tanimoto_protocol writes JSON + CSV + Markdown."""
    payload = run_ood_tanimoto_protocol(
        train_ratio=0.8,
        tanimoto_threshold=0.4,
        tmqm_csv=Path(DEFAULT_TMQM_CSV),
        output_dir=tmp_path,
        max_samples_per_cluster=4,
        clusters=["Pt"],
    )
    paths = save_ood_tanimoto_protocol(payload, output_dir=tmp_path)
    assert paths["json"].is_file()
    assert paths["csv"].is_file()
    assert paths["markdown"].is_file()

    # JSON is parseable + has the expected top-level keys.
    parsed = json.loads(paths["json"].read_text())
    assert parsed["tool"] == "bench_ood_tanimoto_protocol.py"
    assert parsed["pilot_id"] == "wf57_tanimoto_ood"
    assert "aggregate" in parsed
    assert "per_cluster_summary" in parsed
    assert "Pt" in parsed["per_cluster_summary"]

    # CSV has the header + the Pt row.
    csv_text = paths["csv"].read_text()
    assert "cluster" in csv_text
    assert "Pt," in csv_text

    # Markdown mentions the verification result.
    md_text = paths["markdown"].read_text()
    assert "Wf-57 Phase 1" in md_text
    assert "verification" in md_text.lower()


# ---------------------------------------------------------------------------
# Helper coverage (small units, not part of the 15 above)
# ---------------------------------------------------------------------------
def test_safe_div_zero_denominator() -> None:
    """_safe_div returns 0.0 when the denominator is zero."""
    assert _safe_div(1.0, 0.0) == 0.0
    assert _safe_div(5.0, 2.0) == 2.5


def test_frac_below_empty() -> None:
    """_frac_below returns 1.0 for an empty list (vacuous OOD)."""
    assert _frac_below([], 0.4) == 1.0


def test_frac_below_basic() -> None:
    """_frac_below counts pairs with Tanimoto strictly below threshold."""
    pairs = [0.1, 0.2, 0.5, 0.6, 0.39, 0.40, 0.41]
    # Below 0.4 (strict): 0.1, 0.2, 0.39 → 3 / 7
    assert _frac_below(pairs, 0.4) == pytest.approx(3.0 / 7.0, abs=1e-6)


def test_utcnow_iso_format() -> None:
    """_utcnow_iso returns ISO 8601 UTC format with Z suffix."""
    ts = _utcnow_iso()
    assert ts.endswith("Z")
    assert "T" in ts
    assert len(ts) == 20


def test_pairs_from_matrix_upper_triangle() -> None:
    """pairs_from_matrix returns only upper-triangle off-diagonal entries."""
    m = [[1.0, 0.2, 0.3], [0.2, 1.0, 0.4], [0.3, 0.4, 1.0]]
    pairs = pairs_from_matrix(m, upper_triangle_only=True)
    # 3 pairs: (0,1), (0,2), (1,2).
    assert len(pairs) == 3
    assert sorted(pairs) == [0.2, 0.3, 0.4]


def test_fp_defaults_match_ood_protocol() -> None:
    """FP_RADIUS / FP_NBITS match the ood_protocol defaults."""
    # The OOD protocol uses Morgan-ECFP4 (radius=2) + 2048-bit FP.
    assert FP_RADIUS == 2
    assert FP_NBITS == 2048
