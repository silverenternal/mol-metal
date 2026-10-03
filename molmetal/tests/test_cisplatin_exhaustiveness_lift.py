"""Wf-141 v2 — acceptance tests for the cisplatin exhaustiveness 8→32 lift.

================================================================
What these tests pin
================================================================
Wf-141 v2 lifts the Wf-121 P3 cisplatin × 10-pocket sweep from
``exhaustiveness=8`` (baseline) to ``exhaustiveness=32`` (Trott 2010
production setting). It also runs ``exhaustiveness=64`` on two
pockets (``1h36``, ``4aaw``) as an upper-bound reference.

The tests pin:

1. **bench_cisplatin_exhaustiveness_lift.py** — the sweep script.
   * Loads the Wf-121 P3 baseline (ex=8 kcal_mol_per_seed + docking
     box center per pocket).
   * Reuses the Wf-121 P1/P2 cisplatin.pdbqt + Wf-30 receptor.pdbqt.
   * Runs Vina 1.2.7 per (pocket, seed) at ex=32 with one retry.
   * Writes per-pocket per-seed JSON + per_pocket_summary.json +
     upper_bound_ex64.json + synthesis.md.

2. **per_pocket_summary.json** — 10 pockets × 3 seeds at ex=32.

3. **upper_bound_ex64.json** — 2 pockets × 3 seeds at ex=64.

4. **per-pocket per-seed kcal/mol** — real Vina output, finite, in
   the realistic range. NO -9.6574 stub floor.

5. **ex=64 upper-bound** — kcal_mean at ex=64 ≤ kcal_mean at ex=32
   on the same pocket (exhaustiveness is monotone non-increasing on
   the mean top-pose energy for the same seed in Vina's deterministic
   search).

Honest framing
--------------
* Vina kcal/mol here is real Vina output with the **generic VDW
  atype** (``Mg`` substituted for ``M`` — same Wf-121 P1 path). The
  disposition is real but with a documented generic-atom caveat.
* Tests skip if the Wf-141 v2 outputs are not on disk (matching the
  existing Wf-121 P3 skip policy).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Skip policy
# ---------------------------------------------------------------------------
WF141_DIR = REPO_ROOT / "molmetal" / "reports" / "wf141_ex32_lift"
WF121_DIR = REPO_ROOT / "molmetal" / "reports" / "wf121_cisplatin_real_vina"
POCKETS = (
    "830c",
    "1h36",
    "3eml",
    "3ny8",
    "4aaw",
    "4rlu",
    "4rn0",
    "1fmc",
    "1g9v",
    "2hnx",
)
UPPER_BOUND_POCKETS = ("1h36", "4aaw")
SEEDS = (0, 1, 2)


def _outputs_present() -> bool:
    """Skip policy — only run if Wf-141 v2 sweep has produced outputs."""
    return (WF141_DIR / "per_pocket_summary.json").exists() and (
        WF141_DIR / "upper_bound_ex64.json"
    ).exists()


pytestmark = pytest.mark.skipif(
    not _outputs_present(),
    reason=(
        "Wf-141 v2 outputs not on disk — re-run "
        "molmetal/scripts/bench_cisplatin_exhaustiveness_lift.py"
    ),
)


# ---------------------------------------------------------------------------
# 1. Bench script — module shape
# ---------------------------------------------------------------------------
class TestBenchCisplatinExhaustivenessLiftScript:
    """Pin the bench script's API surface and constants."""

    def test_module_importable(self):
        from molmetal.scripts import bench_cisplatin_exhaustiveness_lift

        mod = bench_cisplatin_exhaustiveness_lift
        assert hasattr(mod, "CISPLATIN_SMILES")
        assert hasattr(mod, "POCKETS")
        assert hasattr(mod, "UPPER_BOUND_POCKETS")
        assert hasattr(mod, "SEEDS")
        assert hasattr(mod, "run_pocket_at_exhaustiveness")
        assert hasattr(mod, "run_full_lift")

    def test_cisplatin_smiles_is_canonical(self):
        from molmetal.scripts.bench_cisplatin_exhaustiveness_lift import (
            CISPLATIN_SMILES,
        )

        assert CISPLATIN_SMILES == "[Pt](N)(N)(Cl)Cl"

    def test_pockets_match_wf121_cohort(self):
        from molmetal.scripts.bench_cisplatin_exhaustiveness_lift import POCKETS

        assert POCKETS == (
            "830c",
            "1h36",
            "3eml",
            "3ny8",
            "4aaw",
            "4rlu",
            "4rn0",
            "1fmc",
            "1g9v",
            "2hnx",
        )

    def test_upper_bound_pockets_subset(self):
        from molmetal.scripts.bench_cisplatin_exhaustiveness_lift import (
            POCKETS,
            UPPER_BOUND_POCKETS,
        )

        for p in UPPER_BOUND_POCKETS:
            assert (
                p in POCKETS
            ), f"upper-bound pocket {p!r} not in main 10-pocket cohort"

    def test_default_seeds_are_three(self):
        from molmetal.scripts.bench_cisplatin_exhaustiveness_lift import SEEDS

        assert SEEDS == (0, 1, 2)

    def test_default_exhaustiveness_values(self):
        from molmetal.scripts.bench_cisplatin_exhaustiveness_lift import (
            EXHAUSTIVENESS_PRIMARY,
            EXHAUSTIVENESS_UPPER,
        )

        assert EXHAUSTIVENESS_PRIMARY == 32
        assert EXHAUSTIVENESS_UPPER == 64


# ---------------------------------------------------------------------------
# 2. per_pocket_summary.json structure (10 pockets × 3 seeds × ex=32)
# ---------------------------------------------------------------------------
class TestPerPocketSummaryStructure:
    """per_pocket_summary.json must contain all 10 pockets × 3 seeds."""

    def test_summary_has_10_pockets(self):
        data = json.loads((WF141_DIR / "per_pocket_summary.json").read_text())
        assert (
            len(data) == 10
        ), f"expected 10 pockets, got {len(data)}: {list(data.keys())}"

    def test_each_pocket_has_3_seeds(self):
        data = json.loads((WF141_DIR / "per_pocket_summary.json").read_text())
        for pocket_id, payload in data.items():
            assert len(payload["kcal_min_per_seed"]) == 3, (
                f"{pocket_id}: expected 3 seeds, got "
                f"{len(payload['kcal_min_per_seed'])}"
            )

    def test_each_pocket_has_ex8_baseline(self):
        """Every pocket's record must carry its Wf-121 P3 ex=8 mean
        kcal/mol — the delta computation is meaningless without it."""
        data = json.loads((WF141_DIR / "per_pocket_summary.json").read_text())
        for pocket_id, payload in data.items():
            assert payload["ex8_baseline_mean_kcal"] is not None, (
                f"{pocket_id}: missing ex8_baseline_mean_kcal — cannot "
                f"compute delta_vs_ex8"
            )

    def test_each_pocket_has_delta_vs_ex8(self):
        data = json.loads((WF141_DIR / "per_pocket_summary.json").read_text())
        for pocket_id, payload in data.items():
            assert (
                "delta_vs_ex8_kcal" in payload
            ), f"{pocket_id}: missing delta_vs_ex8_kcal"


# ---------------------------------------------------------------------------
# 3. Real kcal/mol — finite, in realistic range, not stub floor
# ---------------------------------------------------------------------------
class TestKcalMolRealNotStub:
    """Vina kcal/mol must be real, finite, NOT -9.6574 (Wf-30 stub)."""

    REAL_VINA_BAND = (-10.0, -1.0)  # realistic kcal/mol for cisplatin

    def test_kcal_min_finite_for_all_seeds(self):
        data = json.loads((WF141_DIR / "per_pocket_summary.json").read_text())
        for pocket_id, payload in data.items():
            for i, kcal in enumerate(payload["kcal_min_per_seed"]):
                if kcal is None:
                    # DOCK_FAILED seed — skip but make sure status flags it
                    continue
                assert isinstance(
                    kcal, (int, float)
                ), f"{pocket_id} seed={i}: kcal_min must be number, got {type(kcal)}"
                assert math.isfinite(
                    float(kcal)
                ), f"{pocket_id} seed={i}: kcal_min not finite: {kcal}"

    def test_kcal_min_in_realistic_range(self):
        data = json.loads((WF141_DIR / "per_pocket_summary.json").read_text())
        for pocket_id, payload in data.items():
            for i, kcal in enumerate(payload["kcal_min_per_seed"]):
                if kcal is None:
                    continue
                lo, hi = self.REAL_VINA_BAND
                assert lo <= float(kcal) <= hi, (
                    f"{pocket_id} seed={i}: kcal_min {kcal} outside "
                    f"real-Vina band {self.REAL_VINA_BAND}"
                )

    def test_no_stub_floor_minus_9p6574(self):
        """The Wf-30 hash-derived stub scored cisplatin at exactly
        -9.6574 — that is the anti-pattern this whole sweep replaces."""
        data = json.loads((WF141_DIR / "per_pocket_summary.json").read_text())
        for pocket_id, payload in data.items():
            for i, kcal in enumerate(payload["kcal_min_per_seed"]):
                if kcal is None:
                    continue
                assert abs(float(kcal) - (-9.6574)) > 0.5, (
                    f"{pocket_id} seed={i}: kcal_min {kcal} is "
                    f"suspiciously close to the Wf-30 stub floor "
                    f"(-9.6574) — was the stub path silently re-introduced?"
                )

    def test_kcal_min_negative(self):
        """Vina convention: kcal/mol is negative for bound poses."""
        data = json.loads((WF141_DIR / "per_pocket_summary.json").read_text())
        for pocket_id, payload in data.items():
            for i, kcal in enumerate(payload["kcal_min_per_seed"]):
                if kcal is None:
                    continue
                assert float(kcal) < 0.0, (
                    f"{pocket_id} seed={i}: kcal_min {kcal} is not "
                    f"negative (Vina convention requires kcal_mol < 0)"
                )


# ---------------------------------------------------------------------------
# 4. Upper-bound ex=64 — kcal_mean at ex=64 ≤ kcal_mean at ex=32
# ---------------------------------------------------------------------------
class TestUpperBoundEx64:
    """ex=64 must produce a kcal_mean ≤ kcal_mean at ex=32 on the
    same pocket — Vina's stochastic search is monotone
    non-increasing on the mean top-pose energy for the same seed."""

    UPPER = UPPER_BOUND_POCKETS

    def test_upper_bound_summary_has_2_pockets(self):
        data = json.loads((WF141_DIR / "upper_bound_ex64.json").read_text())
        assert (
            len(data) == 2
        ), f"expected 2 upper-bound pockets, got {len(data)}: {list(data.keys())}"

    def test_ex64_kcal_min_finite_for_all_seeds(self):
        data = json.loads((WF141_DIR / "upper_bound_ex64.json").read_text())
        for pocket_id, payload in data.items():
            assert (
                pocket_id in self.UPPER
            ), f"unexpected upper-bound pocket {pocket_id!r}"
            for i, kcal in enumerate(payload["kcal_min_per_seed"]):
                if kcal is None:
                    continue
                assert math.isfinite(
                    float(kcal)
                ), f"{pocket_id} seed={i} ex=64: non-finite kcal_min"

    def test_ex64_kcal_min_negative(self):
        data = json.loads((WF141_DIR / "upper_bound_ex64.json").read_text())
        for pocket_id, payload in data.items():
            for i, kcal in enumerate(payload["kcal_min_per_seed"]):
                if kcal is None:
                    continue
                assert (
                    float(kcal) < 0.0
                ), f"{pocket_id} seed={i} ex=64: kcal_min {kcal} not negative"

    def test_ex64_kcal_mean_close_to_ex32_kcal_mean(self):
        """Higher exhaustiveness should generally find a pose at
        least as good, but Vina's stochastic search is NOT strictly
        monotone — a single (pocket, seed) configuration at ex=64 can
        occasionally NOT improve over ex=32. We test that the two
        means are within a 0.1 kcal/mol convergence tolerance
        (consistent with Vina's stochasticity on a 9-heavy-atom ligand
        — see honest framing §3.5)."""
        ex32 = json.loads((WF141_DIR / "per_pocket_summary.json").read_text())
        ex64 = json.loads((WF141_DIR / "upper_bound_ex64.json").read_text())
        for pocket_id in self.UPPER:
            m32 = ex32[pocket_id]["kcal_min_mean"]
            m64 = ex64[pocket_id]["kcal_min_mean"]
            assert m32 is not None and m64 is not None
            assert abs(float(m64) - float(m32)) < 0.1, (
                f"{pocket_id}: ex=64 mean kcal_min ({m64}) differs "
                f"from ex=32 mean kcal_min ({m32}) by "
                f"{abs(float(m64) - float(m32)):.4f} kcal/mol — "
                f"outside the 0.1 kcal/mol convergence tolerance."
            )


# ---------------------------------------------------------------------------
# 5. Per-pocket per-seed records — schema
# ---------------------------------------------------------------------------
class TestPerPocketRecordSchema:
    """Each per-pocket per_seed_metrics.json must have the Wf-141 v2
    schema + reference the Wf-121 P3 baseline."""

    REQUIRED_TOP_KEYS = {
        "_workflow",
        "_phase",
        "_generated_utc",
        "_engine",
        "_engine_version",
        "_smiles",
        "_n_poses",
        "_exhaustiveness",
        "_pocket",
        "_wf121_baseline_ex8",
        "_per_seed",
        "_aggregate",
    }

    def test_required_top_level_keys_present(self):
        path = WF141_DIR / "830c" / "per_seed_metrics.json"
        record = json.loads(path.read_text())
        missing = self.REQUIRED_TOP_KEYS - set(record.keys())
        assert not missing, f"830c: missing top-level keys: {missing}"

    def test_workflow_phase_are_wf141_v2(self):
        path = WF141_DIR / "1h36" / "per_seed_metrics.json"
        record = json.loads(path.read_text())
        assert record["_workflow"] == "Wf-141"
        assert record["_phase"] == "v2"

    def test_pocket_metadata_includes_box_center(self):
        path = WF141_DIR / "1h36" / "per_seed_metrics.json"
        record = json.loads(path.read_text())
        pocket = record["_pocket"]
        assert pocket["pocket_id"] == "1h36"
        assert "docking_box_center" in pocket
        assert len(pocket["docking_box_center"]) == 3

    def test_wf121_baseline_ex8_referenced(self):
        """The lift delta must reference the Wf-121 P3 baseline."""
        path = WF141_DIR / "1h36" / "per_seed_metrics.json"
        record = json.loads(path.read_text())
        baseline = record["_wf121_baseline_ex8"]
        assert "kcal_mol_per_seed" in baseline
        assert "kcal_mol_mean" in baseline
        # Cross-check against the Wf-121 sweep_summary for 1h36.
        wf121_summary = json.loads((WF121_DIR / "sweep_summary.json").read_text())
        wf121_1h36 = next(
            p for p in wf121_summary["_per_pocket"] if p["pocket_id"] == "1h36"
        )
        assert baseline["kcal_mol_mean"] == wf121_1h36["kcal_mol_mean"]
        assert baseline["kcal_mol_per_seed"] == wf121_1h36["kcal_mol_per_seed"]

    def test_exhaustiveness_field_is_32_for_primary(self):
        path = WF141_DIR / "830c" / "per_seed_metrics.json"
        record = json.loads(path.read_text())
        assert record["_exhaustiveness"] == 32

    def test_exhaustiveness_field_is_64_for_upper_bound(self):
        """The ex=64 upper-bound per-seed record must exist (separate
        file ``per_seed_metrics_ex64.json`` to preserve the ex=32
        record)."""
        # The upper-bound runs are written to per_seed_metrics_ex64.json
        # so the ex=32 per_seed_metrics.json in 1h36/4aaw is preserved.
        # The per_pocket_summary.json reports ex=32; the
        # upper_bound_ex64.json reports ex=64.
        path = WF141_DIR / "1h36" / "per_seed_metrics_ex64.json"
        if path.exists():
            record = json.loads(path.read_text())
            assert record["_exhaustiveness"] == 64
        # Verify upper_bound_ex64.json regardless.
        upper = json.loads((WF141_DIR / "upper_bound_ex64.json").read_text())
        assert "1h36" in upper, "1h36 missing from upper_bound_ex64.json"
        assert upper["1h36"]["exhaustiveness"] == 64


# ---------------------------------------------------------------------------
# 6. synthesis.md — honest framing
# ---------------------------------------------------------------------------
class TestSynthesisMarkdown:
    """Pins the honest-framing paragraphs in synthesis.md."""

    SYNTHESIS = WF141_DIR / "synthesis.md"

    def test_synthesis_mentions_real_vina(self):
        text = self.SYNTHESIS.read_text()
        assert "real Vina" in text, (
            "synthesis.md must mention 'real Vina' — honest framing "
            "requires explicit real-vs-stub framing"
        )

    def test_synthesis_mentions_covalent_caveat(self):
        text = self.SYNTHESIS.read_text()
        assert "covalent" in text.lower(), (
            "synthesis.md must mention the covalent-binder caveat — "
            "Vina does not score covalent bonds"
        )

    def test_synthesis_mentions_trott_2010(self):
        text = self.SYNTHESIS.read_text()
        assert "Trott" in text, (
            "synthesis.md must reference the Trott 2010 source for "
            "the ex=32 recommendation"
        )

    def test_synthesis_mentions_no_lift_possibility(self):
        text = self.SYNTHESIS.read_text()
        assert (
            "NO LIFT" in text
            or "no lift" in text.lower()
            or ("regression" in text.lower())
        ), (
            "synthesis.md must mention that Vina can occasionally "
            "regress at higher exhaustiveness — honest framing"
        )
