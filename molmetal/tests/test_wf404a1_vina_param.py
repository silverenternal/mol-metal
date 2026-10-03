"""Tests for Wf-404-A1 — m3 Vina docking parameter tuning SCAFFOLD.

3 NEW tests:
1. test_param_grid_build — verifies the 4-level exhaustiveness {8, 16, 32, 64}
   AND n_poses {5, 10, 20, 50} grid matches the Wf-404-A1 spec.
2. test_cohort_loader_4pockets — verifies the cohort loader returns the
   canonical Wf-333-V3 4-pocket selection (1h36, 3eml, 3ny8, 4aaw) and
   REJECTS the Wf-381-A1 5th pocket 4rlu.
3. test_smoke_vina_50step — runs the SCAFFOLD end-to-end smoke + verifies
   the smoke report JSON conforms to the Wf-404-A1 schema.

All 3 tests are CPU-only deterministic per AGENTS.md §5.1 + kill_then_adapt
§0.  No real Vina binary is invoked (the smoke run is dry_run=True).

party-twenty-spirit §3.5 MEASURED-vs-PROJECTED discipline preserved
throughout: the synthetic per-level kcal/mol values are PROJECTED
placeholders, never MEASURED.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from molmetal.scripts.wf404a1_vina_param_tuning_scaffold import (
    BASE_KCAL_PER_LEVEL_WF404A1,
    COHORT_POCKETS_WF333V3,
    PARAM_GRID_4LEVEL,
    aggregate_per_level,
    build_param_grid,
    invoke_vina_cli,
    load_cohort_4pockets,
    run_smoke,
    tune_docking,
    validate_param_grid,
    which_vina,
)


# ---------------------------------------------------------------------------
# Test 1 — param grid builds the canonical 4-level exhaustiveness + n_poses
# ---------------------------------------------------------------------------
class TestParamGridBuild:
    """Verify the 4-level grid covers exhaustiveness {8, 16, 32, 64}
    AND n_poses {5, 10, 20, 50}.

    Both axes must be 4-element lists with no overlap (the Wf-404-A1
    expansion over the Wf-380-A4 3-level grid is exactly the addition of
    the (ex=16, n_poses=10) mid-tier).
    """

    EXPECTED_LEVELS = {"ex8_nposes5", "ex16_nposes10", "ex32_nposes20", "ex64_nposes50"}
    EXPECTED_EXHAUSTIVENESS = (8, 16, 32, 64)
    EXPECTED_N_POSES = (5, 10, 20, 50)

    def test_levels_match_canonical_spec(self):
        grid = build_param_grid()
        assert set(grid) == self.EXPECTED_LEVELS, (
            f"Wf-404-A1 grid levels mismatch: "
            f"expected {sorted(self.EXPECTED_LEVELS)}, got {sorted(grid)}"
        )

    def test_exhaustiveness_axis_4_tiers(self):
        grid = build_param_grid()
        actual = tuple(grid[lvl]["exhaustiveness"] for lvl in grid)
        assert sorted(actual) == sorted(self.EXPECTED_EXHAUSTIVENESS), (
            f"exhaustiveness mismatch: expected {self.EXPECTED_EXHAUSTIVENESS}, "
            f"got {actual}"
        )

    def test_n_poses_axis_4_tiers(self):
        grid = build_param_grid()
        actual = tuple(grid[lvl]["n_poses"] for lvl in grid)
        assert sorted(actual) == sorted(
            self.EXPECTED_N_POSES
        ), f"n_poses mismatch: expected {self.EXPECTED_N_POSES}, got {actual}"

    def test_validate_accepts_canonical_grid(self):
        grid = build_param_grid()
        # Should not raise
        validate_param_grid(grid)

    def test_validate_rejects_unknown_level(self):
        bad_grid = dict(PARAM_GRID_4LEVEL)
        bad_grid["ex128_nposes100"] = {"exhaustiveness": 128, "n_poses": 100}
        with pytest.raises(ValueError, match="param grid mismatch"):
            validate_param_grid(bad_grid)

    def test_validate_rejects_out_of_bracket_exhaustiveness(self):
        bad_grid = {
            "ex8_nposes5": {"exhaustiveness": 8, "n_poses": 5},
            "ex16_nposes10": {"exhaustiveness": 16, "n_poses": 10},
            "ex32_nposes20": {"exhaustiveness": 32, "n_poses": 20},
            "ex64_nposes50": {"exhaustiveness": 999, "n_poses": 50},
        }
        with pytest.raises(ValueError, match="exhaustiveness=999 out of canonical"):
            validate_param_grid(bad_grid)


# ---------------------------------------------------------------------------
# Test 2 — cohort loader returns the canonical Wf-333-V3 4-pocket selection
# ---------------------------------------------------------------------------
class TestCohortLoader4Pockets:
    """Verify the cohort loader returns the canonical Wf-333-V3 4-pocket
    selection (1h36, 3eml, 3ny8, 4aaw) and explicitly REJECTS the
    Wf-381-A1 5th pocket 4rlu.
    """

    EXPECTED_POCKETS = ("1h36", "3eml", "3ny8", "4aaw")
    FORBIDDEN_POCKETS = ("4rlu",)
    EXPECTED_SEEDS = (0, 1, 2)

    def test_default_cohort_is_4_pockets_wf333v3(self):
        cohort = load_cohort_4pockets()
        assert tuple(cohort["pockets"]) == self.EXPECTED_POCKETS, (
            f"Wf-404-A1 default cohort mismatch: expected {self.EXPECTED_POCKETS}, "
            f"got {cohort['pockets']}"
        )

    def test_cohort_excludes_4rlu(self):
        """The Wf-381-A1 5-pocket cohort included 4rlu; Wf-404-A1 drops it."""
        cohort = load_cohort_4pockets()
        for forbidden in self.FORBIDDEN_POCKETS:
            assert (
                forbidden not in cohort["pockets"]
            ), f"Wf-404-A1 cohort must NOT include {forbidden!r} (drops vs Wf-381-A1)"

    def test_default_cohort_has_3_seeds(self):
        cohort = load_cohort_4pockets()
        assert tuple(cohort["seeds"]) == self.EXPECTED_SEEDS

    def test_cohort_has_cisplatin_smiles(self):
        cohort = load_cohort_4pockets()
        assert cohort["smiles"] == "Cl[Pt](Cl)(N)N"

    def test_cohort_has_industry_cutoff(self):
        cohort = load_cohort_4pockets()
        assert cohort["industry_cutoff_kcal_mol"] == -7.0

    def test_cohort_module_constant_matches_wf333v3(self):
        """Cross-check: COHORT_POCKETS_WF333V3 module constant is the
        canonical Wf-333-V3 selection (1h36, 3eml, 3ny8, 4aaw)."""
        assert tuple(COHORT_POCKETS_WF333V3) == self.EXPECTED_POCKETS

    def test_cohort_override_accepts_4pockets(self):
        cohort = load_cohort_4pockets(pockets=self.EXPECTED_POCKETS, seeds=(5,))
        assert cohort["pockets"] == list(self.EXPECTED_POCKETS)
        assert cohort["seeds"] == [5]

    def test_per_pocket_offsets_cover_all_4_pockets(self):
        """Each pocket in the 4-pocket cohort must have a per-pocket offset
        for the synthetic kcal/mol generator."""
        from molmetal.scripts.wf404a1_vina_param_tuning_scaffold import (
            PER_POCKET_OFFSET_KCAL_4POCKETS,
        )

        for pocket in self.EXPECTED_POCKETS:
            assert pocket in PER_POCKET_OFFSET_KCAL_4POCKETS, (
                f"Missing per-pocket offset for {pocket!r} in "
                f"PER_POCKET_OFFSET_KCAL_4POCKETS"
            )


# ---------------------------------------------------------------------------
# Test 3 — SCAFFOLD smoke runner: 4-level end-to-end + report JSON schema
# ---------------------------------------------------------------------------
class TestSmokeVina50Step:
    """Runs the SCAFFOLD end-to-end smoke + verifies the report JSON.

    This test deliberately runs in dry_run=True mode so it NEVER invokes a
    real Vina binary (CPU-only deterministic per AGENTS.md §5.1 +
    kill_then_adapt §0).  The 50-step refers to the cohort cardinality
    (4 pockets × 4 levels × 3 seeds ≈ 48 per-(pocket, level, seed) tuples
    rounded to 50).
    """

    def test_run_smoke_writes_report(self, tmp_path):
        out = tmp_path / "wf404a1_smoke.json"
        aggregate = run_smoke(out_path=out)
        assert out.exists()
        assert aggregate["verdict"] == "SYNTHETIC_PLACEHOLDER"

    def test_smoke_report_schema(self, tmp_path):
        out = tmp_path / "wf404a1_smoke.json"
        run_smoke(out_path=out)
        payload = json.loads(out.read_text())

        # Wf-404-A1 schema checks
        assert payload["_workflow"] == "Wf-404-A1"
        assert payload["_phase"] == "SCAFFOLD"
        assert payload["_n_pockets"] == 4
        assert payload["_n_seeds"] == 3
        assert payload["_industry_cutoff_kcal_mol"] == -7.0
        assert payload["_smiles"] == "Cl[Pt](Cl)(N)N"

    def test_smoke_report_per_level_count(self, tmp_path):
        out = tmp_path / "wf404a1_smoke.json"
        run_smoke(out_path=out)
        payload = json.loads(out.read_text())
        # Exactly 4 levels per the Wf-404-A1 spec (NOT 3 per Wf-380-A4)
        assert len(payload["_per_level"]) == 4, (
            f"Wf-404-A1 SCAFFOLD must emit exactly 4 per-level entries, "
            f"got {len(payload['_per_level'])}"
        )

    def test_smoke_report_param_grid_4_tiers(self, tmp_path):
        out = tmp_path / "wf404a1_smoke.json"
        run_smoke(out_path=out)
        payload = json.loads(out.read_text())
        grid = payload["_param_grid"]
        assert len(grid) == 4
        assert set(grid) == {
            "ex8_nposes5",
            "ex16_nposes10",
            "ex32_nposes20",
            "ex64_nposes50",
        }

    def test_smoke_per_level_kcal_monotone_lift(self, tmp_path):
        """Per Wf-141 §0 PROJECTED lift model: ex16 < ex8, ex32 < ex16, ex64 < ex32.

        Higher exhaustiveness should produce more negative (lower) kcal/mol
        cohort median — the canonical PROJECTED Pareto-curve.
        """
        out = tmp_path / "wf404a1_smoke.json"
        run_smoke(out_path=out)
        payload = json.loads(out.read_text())
        per_level = payload["_aggregate"]["per_level_cohort_median"]
        assert (
            per_level["ex16_nposes10"] < per_level["ex8_nposes5"]
        ), "ex16_nposes10 should beat ex8_nposes5 per Wf-141 §0 PROJECTED lift"
        assert per_level["ex32_nposes20"] < per_level["ex16_nposes10"]
        assert per_level["ex64_nposes50"] < per_level["ex32_nposes20"]

    def test_smoke_lift_vs_baseline(self, tmp_path):
        """Lift vs ex8_nposes5 baseline should be strictly negative for the
        3 upper levels (PROJECTED — Wf-141 §0)."""
        out = tmp_path / "wf404a1_smoke.json"
        run_smoke(out_path=out)
        payload = json.loads(out.read_text())
        lift = payload["_aggregate"]["lift_vs_ex8_nposes5_baseline"]
        assert lift["ex8_nposes5"] == 0.0
        for lvl in ("ex16_nposes10", "ex32_nposes20", "ex64_nposes50"):
            assert lift[lvl] < 0.0, f"{lvl} lift vs baseline should be negative"

    def test_invoke_vina_cli_dry_run_no_binary(self, tmp_path):
        """CPU-only dry-run: invoke_vina_cli must return a PROJECTED placeholder
        when the binary is not on PATH."""
        out_pdbqt = tmp_path / "out.pdbqt"
        result = invoke_vina_cli(
            receptor_pdbqt=tmp_path / "receptor.pdbqt",
            ligand_pdbqt=tmp_path / "ligand.pdbqt",
            exhaustiveness=16,
            n_poses=10,
            out_pdbqt=out_pdbqt,
            binary="definitely-not-a-real-binary-vina",
            dry_run=True,
        )
        assert result["dry_run"] is True
        assert result["tag"] == "PROJECTED"
        assert "--exhaustiveness 16" in " ".join(result["cmd"])
        assert "--num_modes 10" in " ".join(result["cmd"])

    def test_invoke_vina_cli_dry_run_with_binary_check(self, tmp_path):
        """Same as above but exercises the binary-check code path (binary may
        or may not exist on the host; either way the function returns the
        dry-run placeholder)."""
        result = invoke_vina_cli(
            receptor_pdbqt=tmp_path / "receptor.pdbqt",
            ligand_pdbqt=tmp_path / "ligand.pdbqt",
            exhaustiveness=32,
            n_poses=20,
            out_pdbqt=tmp_path / "out.pdbqt",
            binary="vina",
            dry_run=True,
        )
        assert result["dry_run"] is True
        assert result["tag"] == "PROJECTED"

    def test_which_vina_returns_str_or_none(self):
        """which_vina is O(path length) and never raises; returns Optional[str]."""
        result = which_vina("vina")
        assert result is None or isinstance(result, str)

    def test_aggregate_per_level_lift_keys(self):
        """aggregate_per_level must include lift_vs_ex8_nposes5_baseline
        AND expected_pass_at_industry_cutoff_-7_0 (canonical Wf-404-A1
        verdict keys)."""
        cohort = load_cohort_4pockets()
        param_grid = build_param_grid()
        cohort_results = {
            lvl: tune_docking(cohort, lvl, param_grid) for lvl in param_grid
        }
        agg = aggregate_per_level(cohort_results)
        assert "lift_vs_ex8_nposes5_baseline" in agg
        assert "expected_pass_at_industry_cutoff_-7_0" in agg
        assert "per_level_cohort_median" in agg
        assert agg["verdict"] == "SYNTHETIC_PLACEHOLDER"
