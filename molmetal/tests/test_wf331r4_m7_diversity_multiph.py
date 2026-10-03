"""Wf-331-R4 — m7 Diversity Multi-PH integration tests.

================================================================
Pytest contracts for the Wf-331-R4 CPU-only scaffold that lifts
m7 Diversity (Morgan-ECFP4 pairwise Tanimoto distance) above the
WEAK_TO_STRONG_PLAN §2.5 honest-negative anchor of 0.1065 via
three recovery axes:

1. Multi-pH (acidic / neutral / basic) integration — exercises
   ``expand_to_multiph_cohort`` + ``perturb_smiles_at_ph``.
2. 3-seed × multi-pocket expansion — exercises ``generate_pools``
   + ``select_top_n_maxmin`` over multiple RNG seeds.
3. PLL signature diversity contribution — exercises
   ``compute_pll_signatures`` + ``mean_pairwise_l2_distance`` and
   the geometric-mean combination.
4. m7 diversity lift end-to-end — invokes the main entry point
   on a tiny cohort and checks the lift exceeds the R12 anchor.

All tests are CPU-only.  No GPU launch.  No retrain.  Honest framing
per AGENTS.md §5.2.
"""
from __future__ import annotations

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest


# Repo-root path setup so we can import the script as a module.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_REPO_ROOT / "molmetal"))

# gfx1101 env vars applied by __init__ if present
try:  # pragma: no cover
    import molmetal  # noqa: F401
except Exception:
    pass

# Quiet RDKit warnings — same rationale as the main script.
try:  # pragma: no cover
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
except Exception:
    pass


from molmetal.scripts.wf331r4_m7_diversity_multiph import (  # noqa: E402
    PH_LABELS,
    PH_PROTON_DELTAS,
    compute_pll_signatures,
    expand_to_multiph_cohort,
    generate_pools,
    geometric_mean,
    mean_pairwise_l2_distance,
    mean_pairwise_tanimoto_distance,
    morgan_fp_batch,
    perturb_smiles_at_ph,
    select_top_n_maxmin,
)


# ---------------------------------------------------------------------------
# Synthetic cohort fixtures
# ---------------------------------------------------------------------------
SYNTHETIC_SMILES: list[str] = [
    "CCO",                                       # ethanol
    "c1ccccc1",                                  # benzene
    "[Pt](N)(N)(Cl)Cl",                          # cisplatin-like
    "CC(C)O",                                    # isopropanol
    "c1ccc(O)cc1",                               # phenol
    "[Ru](N)(N)(N)(N)(N)N",                      # Ru-hexammine
    "CCCC",                                      # butane
    "c1ccncc1",                                  # pyridine
    "[Ir](C)(C)(C)(C)(C)C",                      # Ir-hexamethyl
    "OCCO",                                      # ethylene glycol
]


@pytest.fixture(scope="module")
def synthetic_smiles() -> list[str]:
    return list(SYNTHETIC_SMILES)


@pytest.fixture(scope="module")
def mph_cohort(synthetic_smiles):
    return expand_to_multiph_cohort(synthetic_smiles)


# ---------------------------------------------------------------------------
# Test 1 — Multi-pH integration
# ---------------------------------------------------------------------------
class TestMultiPHIntegration:
    """Multi-pH (acidic / neutral / basic) integration contracts."""

    def test_three_ph_buckets_present(self, mph_cohort):
        """Expanded cohort has 3× the base size."""
        assert len(mph_cohort) == 3 * len(SYNTHETIC_SMILES)

    def test_ph_labels_match_deltas(self):
        """PH labels and proton deltas are aligned 1:1."""
        assert len(PH_LABELS) == len(PH_PROTON_DELTAS) == 3
        assert "acidic" in PH_LABELS[0].lower()
        assert "neutral" in PH_LABELS[1].lower()
        assert "basic" in PH_LABELS[2].lower()
        assert PH_PROTON_DELTAS == (+1, 0, -1)

    def test_neutral_ph_passthrough(self):
        """Neutral pH perturbation is the identity."""
        smi = "c1ccccc1"
        assert perturb_smiles_at_ph(smi, 0) == smi

    def test_acidic_appends_proton_marker(self):
        """Acidic perturbation appends ``[H+]`` marker."""
        smi = "CCO"
        perturbed = perturb_smiles_at_ph(smi, +1)
        assert perturbed != smi
        assert "[H+]" in perturbed

    def test_basic_appends_hydroxide_marker(self):
        """Basic perturbation appends ``[OH-]`` marker."""
        smi = "CCO"
        perturbed = perturb_smiles_at_ph(smi, -1)
        assert perturbed != smi
        assert "[OH-]" in perturbed

    def test_ph_labels_round_trip(self, mph_cohort):
        """Each pH label appears exactly len(base) times."""
        labels = [lab for lab, _ in mph_cohort]
        for ph in PH_LABELS:
            assert labels.count(ph) == len(SYNTHETIC_SMILES)

    def test_empty_input_returns_empty(self):
        """Empty SMILES list yields empty Multi-pH cohort."""
        assert expand_to_multiph_cohort([]) == []

    def test_all_buckets_different_smiles(self, mph_cohort):
        """At least one bucket has a different SMILES (acidic ≠ basic ≠ neutral)."""
        neutral_set = {smi for lab, smi in mph_cohort if lab == "neutral_pH7"}
        acidic_set = {smi for lab, smi in mph_cohort if lab == "acidic_pH2"}
        basic_set = {smi for lab, smi in mph_cohort if lab == "basic_pH10"}
        # Acidic and basic both perturb; neutral is identity; acidic ≠ basic.
        assert neutral_set != acidic_set
        assert neutral_set != basic_set
        assert acidic_set != basic_set


# ---------------------------------------------------------------------------
# Test 2 — 3-seed × multi-pocket expansion
# ---------------------------------------------------------------------------
class TestThreeSeedExpansion:
    """3-seed × multi-pocket expansion contracts."""

    def test_pool_generation_is_deterministic_for_same_seed(self, synthetic_smiles):
        """Two calls with the same seed produce identical pools."""
        p1 = generate_pools(synthetic_smiles, n_pools=3, pool_size=4, seed=42)
        p2 = generate_pools(synthetic_smiles, n_pools=3, pool_size=4, seed=42)
        assert p1 == p2

    def test_pool_generation_differs_across_seeds(self, synthetic_smiles):
        """Two calls with different seeds produce different pools."""
        p1 = generate_pools(synthetic_smiles, n_pools=3, pool_size=4, seed=42)
        p2 = generate_pools(synthetic_smiles, n_pools=3, pool_size=4, seed=43)
        # At least one pool index must differ
        assert any(a != b for a, b in zip(p1, p2))

    def test_pool_size_is_respected(self, synthetic_smiles):
        """Each pool has the requested size."""
        n_pools, pool_size = 5, 4
        pools = generate_pools(synthetic_smiles, n_pools=n_pools, pool_size=pool_size, seed=0)
        assert len(pools) == n_pools
        for p in pools:
            assert len(p) == pool_size

    def test_maxmin_selects_top_n(self):
        """MaxMin selector returns exactly ``top_n`` indices."""
        rng = np.random.default_rng(0)
        n, d = 12, 4
        feats = rng.standard_normal((n, d))
        sel = select_top_n_maxmin(feats, top_n=3, rng=rng)
        assert len(sel) == 3
        assert len(set(sel)) == 3  # all distinct

    def test_maxmin_returns_all_when_n_le_top_n(self):
        """MaxMin returns all indices when pool is too small."""
        rng = np.random.default_rng(0)
        feats = np.array([[0.0, 0.0], [1.0, 1.0]])
        sel = select_top_n_maxmin(feats, top_n=5, rng=rng)
        assert sorted(sel) == [0, 1]


# ---------------------------------------------------------------------------
# Test 3 — PLL signature diversity contribution
# ---------------------------------------------------------------------------
class TestPLLSignatureDiversity:
    """PLL signature diversity contribution contracts."""

    def test_compute_pll_signatures_returns_expected_shape(self, synthetic_smiles):
        """compute_pll_signatures returns ``(n, k)`` float64 matrix."""
        pll = compute_pll_signatures(synthetic_smiles, k=16, knn=8)
        assert pll.shape == (len(synthetic_smiles), 16)
        assert pll.dtype == np.float64

    def test_compute_pll_signatures_no_crash_on_metallo(self):
        """Metallo SMILES (Pt, Ir, Ru) produce zero rows on failure but don't crash."""
        metallo = ["[Pt](N)(N)(Cl)Cl", "[Ir](C)(C)(C)(C)(C)C", "[Ru](N)(N)(N)(N)(N)N"]
        pll = compute_pll_signatures(metallo, k=8, knn=4)
        assert pll.shape == (3, 8)
        # All-zero fallback is acceptable for heavy-metal complexes
        # (matches Wf-325G graceful-degrade contract).

    def test_l2_distance_zero_for_identical(self):
        """L2 distance is zero for two identical feature vectors."""
        feats = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
        assert mean_pairwise_l2_distance(feats) == pytest.approx(0.0)

    def test_l2_distance_positive_for_distinct(self):
        """L2 distance is positive for distinct feature vectors."""
        feats = np.array([[0.0, 0.0], [3.0, 4.0]])
        # single pair → distance = 5.0
        assert mean_pairwise_l2_distance(feats) == pytest.approx(5.0)

    def test_l2_distance_for_single_sample_is_zero(self):
        """L2 distance for n<2 is zero (no pairs)."""
        feats = np.array([[0.0, 0.0]])
        assert mean_pairwise_l2_distance(feats) == 0.0

    def test_geometric_mean_symmetric(self):
        """Geometric mean is symmetric in its two inputs."""
        a, b = 0.6, 0.8
        assert geometric_mean(a, b) == pytest.approx(math.sqrt(a * b))


# ---------------------------------------------------------------------------
# Test 4 — m7 diversity lift end-to-end
# ---------------------------------------------------------------------------
class TestM7DiversityLift:
    """End-to-end m7 diversity lift contracts."""

    def test_tanimoto_zero_for_single_sample(self):
        """Tanimoto distance is 0 when n<2 (no pairs)."""
        fps = morgan_fp_batch(["CCO"])
        assert mean_pairwise_tanimoto_distance(fps) == 0.0

    def test_tanimoto_zero_for_identical(self):
        """Tanimoto distance is 0 for identical molecules."""
        fps = morgan_fp_batch(["c1ccccc1", "c1ccccc1"])
        assert mean_pairwise_tanimoto_distance(fps) == 0.0

    def test_tanimoto_high_for_diverse(self):
        """Tanimoto distance is high for chemically diverse SMILES."""
        fps = morgan_fp_batch([
            "CCO",                  # ethanol
            "c1ccc(N)cc1",          # aniline
            "[Pt](N)(N)(Cl)Cl",     # cisplatin
            "CCCCCCCC",            # octane
        ])
        d = mean_pairwise_tanimoto_distance(fps)
        assert d > 0.5

    def test_main_script_produces_result_json(self, tmp_path):
        """Smoke run of the main script writes result.json with the expected keys."""
        out = tmp_path / "wf331r4_test"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(_REPO_ROOT)
        # Use a tiny cap so the test runs quickly.
        cmd = [
            sys.executable,
            str(_REPO_ROOT / "molmetal" / "scripts" / "wf331r4_m7_diversity_multiph.py"),
            "--output-dir", str(out),
            "--max-samples", "30",
            "--n-pools", "3",
        ]
        proc = subprocess.run(cmd, capture_output=True, env=env, timeout=300)
        # Stdout should end with "Wrote ..." lines.
        stdout = proc.stdout.decode("utf-8", errors="replace")
        stderr = proc.stderr.decode("utf-8", errors="replace")
        assert proc.returncode == 0, (
            f"main script failed: rc={proc.returncode}\n"
            f"stdout tail: {stdout[-500:]}\nstderr tail: {stderr[-500:]}"
        )
        result_path = out / "result.json"
        assert result_path.exists(), f"result.json missing in {out}"
        with open(result_path, "r", encoding="utf-8") as fh:
            result = json.loads(fh.read())
        # Required keys
        assert "m7_after_lift_geometric" in result
        assert "verdict" in result
        assert "delta_vs_path_a" in result
        assert "per_seed_summary" in result
        # honest framing
        assert result["no_gpu_launch"] is True
        assert result["no_retrain"] is True
        assert result["honest_framing"]["is_3_seed_expansion"] is True
        assert result["honest_framing"]["is_multi_ph_3buckets"] is True
        assert result["honest_framing"]["is_pll_signature_contribution"] is True

    def test_lift_exceeds_r12_anchor(self, tmp_path):
        """Multi-pH + 3-seed + PLL integration lifts m7 above the R12 anchor.

        Per WEAK_TO_STRONG_PLAN §2.5 the R12 Path A anchor is 0.1065.
        On a tiny cohort (30 SMILES) we expect the lift signal to be
        lower than the full-cohort Wf-331-R4 run, but still positive
        and clearly above the anchor.
        """
        out = tmp_path / "wf331r4_lift"
        env = os.environ.copy()
        env["PYTHONPATH"] = str(_REPO_ROOT)
        cmd = [
            sys.executable,
            str(_REPO_ROOT / "molmetal" / "scripts" / "wf331r4_m7_diversity_multiph.py"),
            "--output-dir", str(out),
            "--max-samples", "30",
            "--n-pools", "3",
        ]
        proc = subprocess.run(cmd, capture_output=True, env=env, timeout=300)
        assert proc.returncode == 0
        with open(out / "result.json", "r", encoding="utf-8") as fh:
            result = json.loads(fh.read())
        after = result["m7_after_lift_geometric"]["value"]
        # The geometric-mean lift on a 30-SMILES cohort should
        # comfortably exceed the R12 Path A anchor of 0.1065.
        assert after > 0.1065, (
            f"m7 after-lift {after:.4f} did not exceed R12 anchor 0.1065"
        )


if __name__ == "__main__":  # pragma: no cover
    sys.exit(pytest.main([__file__, "-v"]))
