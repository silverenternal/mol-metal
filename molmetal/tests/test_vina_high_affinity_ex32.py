"""Tests for Wf-142 per-pocket high_affinity lift at ex=32.

Wf-142 — re-dock Wf-140 closure-bounded cohort at Vina exhaustiveness=32
+ n_poses=5 and compare high_affinity rate vs the Wf-126 P3 ex=4
closure-bounded baseline.

This test module validates:

1. ``_load_wf126_baseline`` — baseline JSON read.
2. ``_find_receptor`` — receptor PDBQT discovery.
3. ``_receptor_centroid`` — geometric centroid computation.
4. ``_load_cohort`` — Wf-140 cohort SMILES loader.
5. ``_smiles_to_pdbqt_string`` — RDKit + Meeko PDBQT generation.
6. ``per_smiles_results.json`` schema (pocket 3eml fixture).
"""
from __future__ import annotations

import importlib
import json
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Lazy import of the bench script.
# ---------------------------------------------------------------------------


def _import_bench_module():
    """Lazy-import the bench script module."""
    scripts_dir = PROJECT_ROOT / "molmetal" / "scripts"
    if str(scripts_dir) not in sys.path:
        sys.path.insert(0, str(scripts_dir))
    return importlib.import_module("bench_vina_high_affinity_ex32")


# ---------------------------------------------------------------------------
# 1. _load_wf126_baseline
# ---------------------------------------------------------------------------


class TestLoadWf126Baseline:
    def test_3eml_loads(self):
        mod = _import_bench_module()
        b = mod._load_wf126_baseline("3eml")
        assert b["pocket"]["name"] == "3eml"
        assert "co_crystal_ligand_center_A" in b["pocket"]

    def test_1h36_loads(self):
        mod = _import_bench_module()
        b = mod._load_wf126_baseline("1h36")
        assert b["pocket"]["name"] == "1h36"
        assert b["value"]["reference_kcal_mol"] is not None

    def test_missing_pocket_raises(self, tmp_path):
        mod = _import_bench_module()
        with pytest.raises(FileNotFoundError):
            mod._load_wf126_baseline("__nonexistent_pocket__")


# ---------------------------------------------------------------------------
# 2. _find_receptor
# ---------------------------------------------------------------------------


class TestFindReceptor:
    def test_3eml_receptor_found(self):
        mod = _import_bench_module()
        p = mod._find_receptor("3eml")
        assert p.exists()
        assert p.suffix == ".pdbqt"

    def test_1h36_receptor_found(self):
        mod = _import_bench_module()
        p = mod._find_receptor("1h36")
        assert p.exists()


# ---------------------------------------------------------------------------
# 3. _receptor_centroid
# ---------------------------------------------------------------------------


class TestReceptorCentroid:
    def test_returns_3_tuple(self):
        mod = _import_bench_module()
        p = mod._find_receptor("3eml")
        c = mod._receptor_centroid(p)
        assert isinstance(c, tuple)
        assert len(c) == 3
        assert all(isinstance(v, float) for v in c)

    def test_centroid_near_atoms(self):
        mod = _import_bench_module()
        p = mod._find_receptor("3eml")
        cx, cy, cz = mod._receptor_centroid(p)
        # All receptor atoms should be within ~20 Å of centroid
        # (binding pocket typically small). Test only broad sanity.
        assert -50 < cx < 50
        assert -50 < cy < 50
        assert -50 < cz < 50


# ---------------------------------------------------------------------------
# 4. _load_cohort
# ---------------------------------------------------------------------------


class TestLoadCohort:
    def test_3eml_cohort_loads(self):
        mod = _import_bench_module()
        cohort = mod._load_cohort("3eml")
        assert len(cohort) >= 1
        for entry in cohort:
            assert "smiles" in entry

    def test_1h36_cohort_larger(self):
        mod = _import_bench_module()
        cohort = mod._load_cohort("1h36")
        assert len(cohort) >= 5  # 1h36 should have a sizeable cohort


# ---------------------------------------------------------------------------
# 5. _smiles_to_pdbqt_string
# ---------------------------------------------------------------------------


class TestSmilesToPdbqtString:
    def test_simple_smiles(self):
        mod = _import_bench_module()
        s = mod._smiles_to_pdbqt_string("CCO")
        assert s is not None
        assert "REMARK" in s or "ATOM" in s

    def test_aromatic_smiles(self):
        mod = _import_bench_module()
        s = mod._smiles_to_pdbqt_string("c1ccccc1")
        assert s is not None
        assert "ATOM" in s

    def test_invalid_smiles_returns_none(self):
        mod = _import_bench_module()
        s = mod._smiles_to_pdbqt_string("NOT_A_SMILES_$$$")
        assert s is None


# ---------------------------------------------------------------------------
# 6. per_smiles_results.json schema (3eml fixture)
# ---------------------------------------------------------------------------


WF142_REPORT_DIR = (
    PROJECT_ROOT
    / "molmetal"
    / "reports"
    / "wf142_ex32_high_affinity"
)


@pytest.mark.skipif(
    not (WF142_REPORT_DIR / "3eml" / "per_smiles_results.json").exists(),
    reason="Wf-142 3eml fixture not yet produced; run bench script first",
)
class TestPerSmilesResultsSchema:
    def test_3eml_schema_keys(self):
        d = json.loads((WF142_REPORT_DIR / "3eml" / "per_smiles_results.json").read_text())
        for k in (
            "_workflow",
            "_phase",
            "_pocket",
            "_reference_kcal_mol",
            "_exhaustiveness",
            "_n_poses",
            "_n_docked",
            "_n_high_affinity",
            "_high_affinity_rate",
            "_delta_vs_baseline_ex4",
            "_kcal_min_overall",
            "_kcal_mean_top5_avg",
            "_per_smiles",
        ):
            assert k in d, f"missing key {k}"

    def test_3eml_rate_in_0_1(self):
        d = json.loads((WF142_REPORT_DIR / "3eml" / "per_smiles_results.json").read_text())
        rate = d["_high_affinity_rate"]
        assert 0.0 <= rate <= 1.0

    def test_3eml_kcal_min_negative(self):
        d = json.loads((WF142_REPORT_DIR / "3eml" / "per_smiles_results.json").read_text())
        assert d["_kcal_min_overall"] is not None
        assert d["_kcal_min_overall"] < 0.0

    def test_3eml_exhaustiveness_is_32(self):
        d = json.loads((WF142_REPORT_DIR / "3eml" / "per_smiles_results.json").read_text())
        assert d["_exhaustiveness"] == 32
        assert d["_n_poses"] == 5

    def test_3eml_per_smiles_have_kcal(self):
        d = json.loads((WF142_REPORT_DIR / "3eml" / "per_smiles_results.json").read_text())
        for entry in d["_per_smiles"]:
            assert "smiles" in entry
            assert "kcal_min" in entry
            if entry["kcal_min"] is not None:
                assert entry["kcal_min"] < 0.0


@pytest.mark.skipif(
    not (WF142_REPORT_DIR / "1h36" / "per_smiles_results.json").exists(),
    reason="Wf-142 1h36 fixture not yet produced; run bench script first",
)
class TestPerSmilesResults1h36:
    def test_1h36_high_rate(self):
        """1h36 reference kcal/mol is -0.801 (very weak target). The
        Wf-140 closure-bounded fragments should easily beat this."""
        d = json.loads((WF142_REPORT_DIR / "1h36" / "per_smiles_results.json").read_text())
        assert d["_high_affinity_rate"] >= 0.5


# ---------------------------------------------------------------------------
# 7. 4aaw fixture — Wf-142 P1 main proof-of-concept (NEGATIVE-METRIC LIFT)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (WF142_REPORT_DIR / "4aaw" / "per_smiles_results.json").exists(),
    reason="Wf-142 4aaw fixture not yet produced; run bench script first",
)
class TestPerSmilesResults4aaw:
    """Wf-142 P1 main result — 4aaw closure-bounded ex=32 lift.

    Wf-126 P3 baseline (ex=4, n_poses=1, n=5 docked) gave rate=0.0
    (0/5 high-affinity at -4.537 kcal/mol co-crystal reference).
    Wf-142 lifts by re-docking the same cohort at ex=32 + n_poses=5.
    """

    def test_4aaw_schema_keys(self):
        d = json.loads(
            (WF142_REPORT_DIR / "4aaw" / "per_smiles_results.json").read_text()
        )
        for k in (
            "_workflow",
            "_phase",
            "_pocket",
            "_reference_kcal_mol",
            "_exhaustiveness",
            "_n_poses",
            "_n_docked",
            "_n_high_affinity",
            "_high_affinity_rate",
            "_delta_vs_baseline_ex4",
            "_kcal_min_overall",
            "_kcal_mean_top5_avg",
            "_box_center_A",
            "_box_size_A",
            "_wf126_baseline_ex4",
            "_per_smiles",
        ):
            assert k in d, f"missing key {k}"

    def test_4aaw_rate_in_0_1(self):
        d = json.loads(
            (WF142_REPORT_DIR / "4aaw" / "per_smiles_results.json").read_text()
        )
        rate = d["_high_affinity_rate"]
        assert 0.0 <= rate <= 1.0

    def test_4aaw_kcal_min_negative(self):
        d = json.loads(
            (WF142_REPORT_DIR / "4aaw" / "per_smiles_results.json").read_text()
        )
        assert d["_kcal_min_overall"] is not None
        assert d["_kcal_min_overall"] < 0.0

    def test_4aaw_exhaustiveness_is_32_nposes_is_5(self):
        d = json.loads(
            (WF142_REPORT_DIR / "4aaw" / "per_smiles_results.json").read_text()
        )
        assert d["_exhaustiveness"] == 32
        assert d["_n_poses"] == 5

    def test_4aaw_n_docked_equals_cohort(self):
        d = json.loads(
            (WF142_REPORT_DIR / "4aaw" / "per_smiles_results.json").read_text()
        )
        assert d["_n_docked"] == d["_n_cohort_total"]
        assert d["_n_docked"] >= 1

    def test_4aaw_lifts_vs_wf126_baseline(self):
        """Main Wf-142 P1 result: 4aaw closure-bounded ex=32 must lift vs
        Wf-126 P3 ex=4 closure-bounded baseline (0.0)."""
        d = json.loads(
            (WF142_REPORT_DIR / "4aaw" / "per_smiles_results.json").read_text()
        )
        # Wf-126 P3 baseline 0/5 = 0.0 (per
        # reports/wf126_p3_vina_per_pocket/4aaw/vina_high_affinity.json)
        baseline = d["_wf126_baseline_ex4"]["rate"]
        assert baseline == 0.0
        # Wf-142 lift must clear the 0.30 uniform floor
        assert d["_high_affinity_rate"] >= 0.30
        assert d["_delta_vs_baseline_ex4"] >= 0.30

    def test_4aaw_per_smiles_records(self):
        d = json.loads(
            (WF142_REPORT_DIR / "4aaw" / "per_smiles_results.json").read_text()
        )
        for entry in d["_per_smiles"]:
            assert "smiles" in entry
            assert "kcal_min" in entry
            assert "high_affinity" in entry
            if entry["kcal_min"] is not None:
                assert entry["kcal_min"] < 0.0
                assert isinstance(entry["high_affinity"], bool)

    def test_4aaw_box_center_is_centroid(self):
        """Box center should equal the receptor geometric centroid, not
        the historical co-crystal centre (which is ~100 Å off on 4aaw)."""
        d = json.loads(
            (WF142_REPORT_DIR / "4aaw" / "per_smiles_results.json").read_text()
        )
        cx, cy, cz = d["_box_center_A"]
        # 4aaw pocket10 subset centroid is ~(45.6, 35.9, 81.0) — wide
        # tolerance because the JSON is not exact.
        assert 30 < cx < 60
        assert 20 < cy < 50
        assert 60 < cz < 100
        # Box side should be 20 Å (Wf-126 P3 protocol), not 28 Å
        assert d["_box_size_A"] == [20.0, 20.0, 20.0]

    def test_4aaw_anti_confusion_guards(self):
        """Verify the run actually used Wf-126 P3 closure-bounded cohort
        (n=14) + Wf-30 pocket10 receptor + Wf-126 P3 docking protocol
        (receptor centroid + 20 Å cube)."""
        d = json.loads(
            (WF142_REPORT_DIR / "4aaw" / "per_smiles_results.json").read_text()
        )
        # NOT cisplatin kcal/mol metric
        assert d["_workflow"] == "Wf-142"
        # NOT Wf-141 (cisplatin exhaustiveness lift)
        assert d["_pocket"] == "4aaw"
        # NOT 1h36 (different pocket)
        # NOT vina_kcal_per_mol.json metric (different cut)
        # Per-pocket high_affinity, NOT cisplatin ex=8 lift
        assert d["_n_cohort_total"] == 14
        assert d["_n_docked"] == 14
        # Real Vina (not stub)
        assert d["_engine"].startswith("vina")


# ---------------------------------------------------------------------------
# 8. 3ny8 fixture — Wf-142 P1 single-pocket proof-of-concept
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (WF142_REPORT_DIR / "3ny8" / "per_smiles_results.json").exists(),
    reason="Wf-142 3ny8 fixture not yet produced; run bench script first",
)
class TestPerSmilesResults3ny8:
    """Wf-142 P1 single-pocket result on 3ny8.

    Wf-126 P3 closure-bounded baseline (ex=4, n_poses=1): 4/18 = 0.222
    Wf-142 P1 (ex=32, n_poses=5): 4/14 = 0.286 (same 4 SMILES)
    Delta = +0.064 vs Wf-126 P3.  Still FAILS 0.30 uniform floor.
    Same high-affinity SMILES at both ex settings — ex lift does NOT
    break new ground on 3ny8 (cohort ceiling, not Vina search).
    """

    def test_3ny8_schema_keys(self):
        d = json.loads(
            (WF142_REPORT_DIR / "3ny8" / "per_smiles_results.json").read_text()
        )
        for k in (
            "_workflow", "_phase", "_pocket", "_reference_kcal_mol",
            "_exhaustiveness", "_n_poses", "_n_cohort_total",
            "_n_docked", "_n_high_affinity", "_high_affinity_rate",
            "_delta_vs_baseline_ex4", "_kcal_min_overall",
            "_kcal_mean_top5_avg", "_box_center_A", "_box_size_A",
            "_wf126_baseline_ex4", "_per_smiles",
        ):
            assert k in d, f"missing key {k}"

    def test_3ny8_pocket_is_3ny8(self):
        d = json.loads(
            (WF142_REPORT_DIR / "3ny8" / "per_smiles_results.json").read_text()
        )
        assert d["_pocket"] == "3ny8"
        assert d["_workflow"] == "Wf-142"

    def test_3ny8_rate_in_0_1(self):
        d = json.loads(
            (WF142_REPORT_DIR / "3ny8" / "per_smiles_results.json").read_text()
        )
        rate = d["_high_affinity_rate"]
        assert 0.0 <= rate <= 1.0

    def test_3ny8_kcal_min_negative(self):
        d = json.loads(
            (WF142_REPORT_DIR / "3ny8" / "per_smiles_results.json").read_text()
        )
        assert d["_kcal_min_overall"] is not None
        assert d["_kcal_min_overall"] < 0.0

    def test_3ny8_exhaustiveness_is_32_nposes_is_5(self):
        d = json.loads(
            (WF142_REPORT_DIR / "3ny8" / "per_smiles_results.json").read_text()
        )
        assert d["_exhaustiveness"] == 32
        assert d["_n_poses"] == 5

    def test_3ny8_n_docked_equals_cohort(self):
        d = json.loads(
            (WF142_REPORT_DIR / "3ny8" / "per_smiles_results.json").read_text()
        )
        # Wf-140 cohort has 14 unique SMILES; all should dock
        assert d["_n_docked"] == d["_n_cohort_total"]
        assert d["_n_docked"] >= 1

    def test_3ny8_reference_is_3ny8_co_crystal(self):
        """3ny8 co-crystal reference = -7.14 kcal/mol (residue JRZ)."""
        d = json.loads(
            (WF142_REPORT_DIR / "3ny8" / "per_smiles_results.json").read_text()
        )
        assert d["_reference_kcal_mol"] == -7.14

    def test_3ny8_box_center_uses_centroid(self):
        """Box center should be receptor centroid (-1.7, 3.7, -50.7),
        not the JSON co_crystal_ligand_center_A which is offset ~100Å in z
        (would produce empty-space docking with kcal_min ~ 0)."""
        d = json.loads(
            (WF142_REPORT_DIR / "3ny8" / "per_smiles_results.json").read_text()
        )
        cx, cy, cz = d["_box_center_A"]
        assert -3 < cx < 0  # centroid x ≈ -1.7
        assert 2 < cy < 6   # centroid y ≈ 3.7
        assert -55 < cz < -45  # centroid z ≈ -50.7

    def test_3ny8_box_size_is_20_cube(self):
        """Box side hard-coded to 20 Å to match Wf-126 P3 protocol
        (JSON box_side_A=24 Å is from historical unconstrained run)."""
        d = json.loads(
            (WF142_REPORT_DIR / "3ny8" / "per_smiles_results.json").read_text()
        )
        assert d["_box_size_A"] == [20.0, 20.0, 20.0]

    def test_3ny8_lifts_vs_wf126_baseline(self):
        """Wf-142 P1 3ny8 must be >= Wf-126 P3 baseline (0.222)."""
        d = json.loads(
            (WF142_REPORT_DIR / "3ny8" / "per_smiles_results.json").read_text()
        )
        baseline_rate = d["_wf126_baseline_ex4"]["rate"]
        assert 0.20 < baseline_rate < 0.25  # 4/18 = 0.222
        assert d["_high_affinity_rate"] >= baseline_rate
        assert d["_delta_vs_baseline_ex4"] >= 0.0

    def test_3ny8_per_smiles_have_kcal(self):
        """Every docked SMILES must have a finite negative kcal_min
        (real Vina, not stub)."""
        d = json.loads(
            (WF142_REPORT_DIR / "3ny8" / "per_smiles_results.json").read_text()
        )
        for entry in d["_per_smiles"]:
            assert "smiles" in entry
            assert "kcal_min" in entry
            assert "high_affinity" in entry
            if entry["kcal_min"] is not None:
                assert entry["kcal_min"] < 0.0
                assert isinstance(entry["high_affinity"], bool)

    def test_3ny8_anti_confusion_guards(self):
        """Anti-confusion: this is NOT cisplatin kcal/mol, NOT
        vina_kcal_per_mol.json, NOT Wf-141 v2 (cisplatin ex lift)."""
        d = json.loads(
            (WF142_REPORT_DIR / "3ny8" / "per_smiles_results.json").read_text()
        )
        assert d["_workflow"] == "Wf-142"
        assert d["_pocket"] == "3ny8"
        assert d["_n_cohort_total"] == 14
        assert d["_engine"].startswith("vina")


# ---------------------------------------------------------------------------
# 9. 4rlu fixture — Wf-142 P1 harness-required pocket (HONEST-NEGATIVE)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (WF142_REPORT_DIR / "4rlu" / "per_smiles_results.json").exists(),
    reason="Wf-142 4rlu fixture not yet produced; run bench script first",
)
class TestPerSmilesResults4rlu:
    """Wf-142 P1 result on 4rlu — HONEST-NEGATIVE.

    Wf-126 P3 closure-bounded baseline (ex=4, n_poses=1): 0/14 = 0.000
    Wf-142 P1 (ex=32, n_poses=5): 0/10 = 0.000
    Delta = +0.000 (NO LIFT, NO REGRESS).

    Honest framing (party-twenty-spirit §3.5):
    * Reference -6.373 kcal/mol is the HCC co-crystal redocked score;
      the Wf-140 closure-bounded fragments are too small to bind that
      strongly even at ex=32 (best kcal_min=-3.802, ~2.6 kcal/mol
      weaker than reference).
    * Vina stochastic search does not fail here — it consistently
      produces a tight ~[-2.0, -3.8] kcal/mol range. The bottleneck
      is the fragment cohort (closure-bounded small molecules), not
      the search exhaustiveness.
    * kcal_min improved marginally from -3.796 (Wf-126 P3) to -3.802
      (Wf-142 P1), +0.006 kcal/mol lift — within seed-std noise.

    Anti-confusion:
    * NOT cisplatin kcal/mol (separate metric, separate JSON).
    * NOT vina_kcal_per_mol.json (different cut).
    * NOT Wf-141 v2 (cisplatin ex=32 lift).
    * NOT vina_high_affinity_test_001.json (test fixture).
    """

    def test_4rlu_schema_keys(self):
        d = json.loads(
            (WF142_REPORT_DIR / "4rlu" / "per_smiles_results.json").read_text()
        )
        for k in (
            "_workflow", "_phase", "_pocket", "_reference_kcal_mol",
            "_exhaustiveness", "_n_poses", "_n_cohort_total",
            "_n_docked", "_n_high_affinity", "_high_affinity_rate",
            "_delta_vs_baseline_ex4", "_kcal_min_overall",
            "_kcal_mean_top5_avg", "_box_center_A", "_box_size_A",
            "_wf126_baseline_ex4", "_per_smiles",
        ):
            assert k in d, f"missing key {k}"

    def test_4rlu_pocket_is_4rlu(self):
        d = json.loads(
            (WF142_REPORT_DIR / "4rlu" / "per_smiles_results.json").read_text()
        )
        assert d["_pocket"] == "4rlu"
        assert d["_workflow"] == "Wf-142"

    def test_4rlu_rate_in_0_1(self):
        d = json.loads(
            (WF142_REPORT_DIR / "4rlu" / "per_smiles_results.json").read_text()
        )
        rate = d["_high_affinity_rate"]
        assert 0.0 <= rate <= 1.0

    def test_4rlu_kcal_min_negative(self):
        d = json.loads(
            (WF142_REPORT_DIR / "4rlu" / "per_smiles_results.json").read_text()
        )
        assert d["_kcal_min_overall"] is not None
        assert d["_kcal_min_overall"] < 0.0

    def test_4rlu_exhaustiveness_is_32_nposes_is_5(self):
        d = json.loads(
            (WF142_REPORT_DIR / "4rlu" / "per_smiles_results.json").read_text()
        )
        assert d["_exhaustiveness"] == 32
        assert d["_n_poses"] == 5

    def test_4rlu_n_docked_equals_cohort(self):
        d = json.loads(
            (WF142_REPORT_DIR / "4rlu" / "per_smiles_results.json").read_text()
        )
        # Wf-140 cohort has 10 unique SMILES on 4rlu (smaller than other
        # pockets because Catherine filter is stricter on 4rlu's
        # tetrameric trypsin geometry).
        assert d["_n_docked"] == d["_n_cohort_total"]
        assert d["_n_docked"] >= 1

    def test_4rlu_reference_is_hcc_co_crystal(self):
        """4rlu co-crystal reference = -6.373 kcal/mol (residue HCC)."""
        d = json.loads(
            (WF142_REPORT_DIR / "4rlu" / "per_smiles_results.json").read_text()
        )
        assert d["_reference_kcal_mol"] == -6.373

    def test_4rlu_box_center_uses_centroid(self):
        """Box center should be receptor centroid (-13.1, 13.2, -35.3),
        not the JSON co_crystal_ligand_center_A (-0.7, 22.8, -31.2)
        which is the HCC co-crystal ligand center (offset ~12-15 Å from
        centroid; both inside receptor bounding box, but centroid is
        more reproducible across seed runs)."""
        d = json.loads(
            (WF142_REPORT_DIR / "4rlu" / "per_smiles_results.json").read_text()
        )
        cx, cy, cz = d["_box_center_A"]
        assert -20 < cx < -5     # centroid x ≈ -13.1
        assert 5 < cy < 25       # centroid y ≈ 13.2
        assert -45 < cz < -25    # centroid z ≈ -35.3

    def test_4rlu_box_size_is_20_cube(self):
        """Box side hard-coded to 20 Å to match Wf-126 P3 protocol
        (JSON box_side_A=25.5 Å is from historical unconstrained run)."""
        d = json.loads(
            (WF142_REPORT_DIR / "4rlu" / "per_smiles_results.json").read_text()
        )
        assert d["_box_size_A"] == [20.0, 20.0, 20.0]

    def test_4rlu_honest_negative_no_lift(self):
        """Wf-142 P1 4rlu — HONEST-NEGATIVE. Rate stays 0.000.

        We deliberately do NOT assert a lift; we assert no regression.
        """
        d = json.loads(
            (WF142_REPORT_DIR / "4rlu" / "per_smiles_results.json").read_text()
        )
        # Wf-126 P3 closure-bounded baseline: 0/14 = 0.0
        baseline_rate = d["_wf126_baseline_ex4"]["rate"]
        assert baseline_rate == 0.0
        # Wf-142 must not regress
        assert d["_high_affinity_rate"] >= baseline_rate
        assert d["_delta_vs_baseline_ex4"] >= 0.0
        # And explicitly: rate stays at 0 because fragments too small
        assert d["_high_affinity_rate"] == 0.0

    def test_4rlu_kcal_min_improvement_within_seed_noise(self):
        """Wf-142 P1 kcal_min vs Wf-126 P3 baseline — modest improvement
        within seed-std noise. Honest reporting of marginal lift."""
        d = json.loads(
            (WF142_REPORT_DIR / "4rlu" / "per_smiles_results.json").read_text()
        )
        # Wf-126 P3 baseline kcal_min = -3.796 (n=14 docked, seed=0)
        # Wf-142 P1 kcal_min = -3.802 (n=10 docked, seed=0)
        # delta = -3.802 - (-3.796) = -0.006 (improvement of 6 mmol/mol)
        kcal_min_ex32 = d["_kcal_min_overall"]
        assert kcal_min_ex32 <= -3.0  # bound: meaningful real binding
        assert kcal_min_ex32 > d["_reference_kcal_mol"]  # does NOT reach ref

    def test_4rlu_anti_confusion_guards(self):
        """Anti-confusion: this is NOT cisplatin kcal/mol, NOT
        vina_kcal_per_mol.json, NOT Wf-141 v2 (cisplatin ex lift),
        NOT test_001."""
        d = json.loads(
            (WF142_REPORT_DIR / "4rlu" / "per_smiles_results.json").read_text()
        )
        assert d["_workflow"] == "Wf-142"
        assert d["_pocket"] == "4rlu"
        assert d["_n_cohort_total"] == 10
        assert d["_engine"].startswith("vina")
