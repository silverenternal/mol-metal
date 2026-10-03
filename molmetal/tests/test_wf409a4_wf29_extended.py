"""Pytest: Wf-409-A4 — Wf-29 ``train_se3sb.py`` AttributeError fix EXTENDED smoke.

=========================================================================
The Wf-29 fix at ``molmetal/scripts/train_se3sb.py:620`` (post-fix)
changed ``use_sinkhorn=args.se3_sinkhorn`` →
``use_sinkhorn=args.se3_use_sinkhorn`` to match the argparse flag
``--se3-use-sinkhorn``.

Wf-409-A4 ships an EXTENDED smoke that anchors three Wf-243..Wf-252
ship-wave integrations alongside the Wf-29 attr-fix surface:

1. **inv4 — m18 per-residue metal coordination verification**
   (Wf-247-B / TODO-130 m18 REDESIGN ship-wave) — verifies the
   :class:`PerResidueProteinMetalAUC` adapter runs end-to-end on the
   protein-metal CSV slice and emits the canonical result contract
   keys.

2. **inv5 — F5_RDKIT_2D 32-d feature extraction verification**
   (Wf-248 ship-wave) — verifies the 32-d RDKit descriptor extractor
   produces a 32-d vector with first-12 entries equal to the canonical
   12-d baseline (the Wf-248 backward-compat contract).

3. **inv6 — click-aug wire verification** (Wf-221 ship-wave) — verifies
   :class:`ClickAugmentedChempropGenerator` G4 backward-compat
   (``use_click_prob=0.0`` returns exact passthrough SMILES) + G2
   acceptance (``use_click_prob=0.5`` produces a non-empty wrapper
   output of size ``n_samples``).

These tests pin the contract via the Wf-409-A4 smoke harness:

* **test_m18_per_residue** — :func:`verify_m18_per_residue` returns
  ``verdict == "PASS"``; per-complex / per-residue / per-residue-pr
  AUCs in ``[0.0, 1.0]``; reconciliation table non-empty.

* **test_f5_32d_extraction** — :func:`verify_f5_32d_extraction` returns
  ``verdict == "PASS"``; F5 32-d dim matches
  :data:`RDKIT_2D_DIM == 32`; first-12 entries match the canonical
  12-d vector.

The companion :func:`verify_click_aug_wire` is covered by the
:file:`molmetal/tests/test_chemprop_v2_click_augmentation.py` suite
already (Wf-221 G1 + G2 + G4); Wf-409-A4 includes it in the smoke
JSON for ship-wave integration cross-reference but does not duplicate
the test (would be redundant).

Honest framing
--------------
* These tests are CPU-only and use the live RDKit / sklearn install.
* We test the *adapter mechanics* (synthetic probe + small CSV slice +
  passthrough base_generator), not a deep retrain — actual 5-seed ×
  50-epoch W2.1 SE(3) SB retrain remains BLOCKED on Wf-299 per
  TODO-207 + AGENTS.md §5.1.
* The m18 per-residue AUC is a numerator-level upper bound (per-residue
  labels do NOT ship with MetalCytoToxDB); see
  :mod:`molmetal.adapters.per_residue_protein_metal_auc` §Honest framing.

Run with::

    source env.sh
    pytest -xvs molmetal/tests/test_wf409a4_wf29_extended.py
"""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR: Path = PROJECT_ROOT / "molmetal" / "reports" / "wf409a4_wf29_extended"
SMOKE_REPORT_PATH: Path = REPORTS_DIR / "smoke_result.json"


def _build_smoke() -> dict:
    """Run the Wf-409-A4 smoke end-to-end + assert report shape."""
    from molmetal.scripts.wf409a4_wf29_extended_smoke import (
        EXPECTED_F5_DIM,
        run_wf409a4_smoke,
    )

    report = run_wf409a4_smoke(out_dir=REPORTS_DIR)
    assert report["recipe"] == "wf409a4_wf29_extended_smoke"
    assert report["wf_id"] == "Wf-409-A4"
    assert report["provenance"] == "MEASURED"
    assert "invariants" in report
    for key in (
        "inv4_m18_per_residue",
        "inv5_f5_32d_extraction",
        "inv6_click_aug_wire",
    ):
        assert key in report["invariants"]
    assert report["expected_f5_dim"] == EXPECTED_F5_DIM
    # Aggregate verdict must be PASS or SKIP (never FAIL — the host may
    # be RDKit/torch-missing per the Wf-404-A5 SKIP pattern).
    assert report["aggregate_verdict"] in {
        "PASS",
        "SKIP",
    }, f"aggregate_verdict FAIL: {report.get('honest_framing', '?')}"
    # Smoke report must be on disk per AGENTS.md §5.5.
    assert SMOKE_REPORT_PATH.exists()
    on_disk = json.loads(SMOKE_REPORT_PATH.read_text())
    assert on_disk["aggregate_verdict"] == report["aggregate_verdict"]
    return report


def test_m18_per_residue() -> None:
    """m18 per-residue metal coordination verification PASS + legal AUCs.

    FAIL is a code regression. SKIP is impossible here (no torch/RDKit
    dependency) — only FAIL is a regression signal.
    """
    report = _build_smoke()
    inv4 = report["invariants"]["inv4_m18_per_residue"]
    assert (
        inv4["verdict"] == "PASS"
    ), f"inv4 m18 per-residue FAIL: {inv4.get('failure_reason', '?')}"
    # All three AUCs must be in [0.0, 1.0].
    for key in ("per_complex_auc", "per_residue_auc", "per_residue_pr_auc"):
        v = float(inv4[key])
        assert 0.0 <= v <= 1.0, f"{key}={v} out of [0.0, 1.0]"
    # Reconciliation table must be non-empty (the adapter caps at 25 rows).
    assert inv4["n_reconciliation_rows"] >= 1
    # n_residue_samples must be > n_complex (per-residue expansion).
    assert inv4["n_residue_samples"] >= inv4["n_complex"]
    # n_complex must be exactly MAX_PROTEIN_METAL_ROWS = 50.
    assert inv4["n_complex"] == 50


def test_f5_32d_extraction() -> None:
    """F5_RDKIT_2D 32-d extraction PASS or honest-negative SKIP on host.

    SKIP is an environment gap (RDKit not installed on this venv) —
    the Wf-404-A5 SKIP pattern applies. PASS requires the canonical
    32-d shape + first-12 backward-compat.
    """
    report = _build_smoke()
    inv5 = report["invariants"]["inv5_f5_32d_extraction"]
    assert inv5["verdict"] in {
        "PASS",
        "SKIP",
    }, f"inv5 F5 32-d FAIL: {inv5.get('failure_reason', '?')}"
    # Probe SMILES must be the canonical Wf-248 toluene SMILES.
    assert inv5["probe_smiles"] == "Cc1ccccc1"
    if inv5["verdict"] == "PASS":
        # Schema drift check — 32-d shape is the canonical contract.
        assert inv5["f5_dim"] == 32
        assert inv5["expected_f5_dim"] == 32
        # First-12 entries MUST match the canonical 12-d baseline exactly.
        assert inv5["first_12_match_12d"] is True
    # SKIP case: failure_reason must be present and explanatory.
    else:
        assert inv5.get("failure_reason"), "SKIP requires failure_reason"
        assert (
            "rdkit" in inv5["failure_reason"].lower()
            or "import" in inv5["failure_reason"].lower()
        )


def test_click_aug_wire() -> None:
    """Click-aug wire verification: G4 passthrough + G2 active, or SKIP.

    Sister contract :file:`molmetal/tests/test_chemprop_v2_click_augmentation.py`
    covers the same surface; this test is the Wf-409-A4 smoke
    cross-reference anchor. SKIP is an environment gap (torch missing
    on this venv triggers the package ``__init__`` DMPNN import).
    """
    report = _build_smoke()
    inv6 = report["invariants"]["inv6_click_aug_wire"]
    assert inv6["verdict"] in {
        "PASS",
        "SKIP",
    }, f"inv6 click-aug FAIL: {inv6.get('failure_reason', '?')}"
    if inv6["verdict"] == "PASS":
        # G4 backward-compat: passthrough MUST return n_samples entries
        # with the seed SMILES preserved exactly.
        assert inv6["passthrough_match_seed"] is True
        assert inv6["passthrough_n"] == inv6["n_samples"] == 20
        # G2 acceptance: active wrapper MUST emit ``n_samples`` entries
        # (monotone-on-n_samples contract).
        assert inv6["active_n"] == inv6["n_samples"] == 20
    else:
        # SKIP: failure_reason must be present and mention torch/import.
        assert inv6.get("failure_reason"), "SKIP requires failure_reason"
        assert (
            "torch" in inv6["failure_reason"].lower()
            or "import" in inv6["failure_reason"].lower()
        )
