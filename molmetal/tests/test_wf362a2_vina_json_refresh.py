"""m3 Vina JSON honesty tests (rewritten 2026-10-20).

================================================================
Why this file was rewritten
================================================================
The original Wf-362-A2 tests asserted that
``metrics/by_metric/vina_kcal_per_mol.json`` had been refreshed with a
``wf333_v3_*`` measurement entry worth −7.782 kcal/mol that "PASSES
industry −7.0 by +0.782 margin".  That refresh never landed:
``git log -S wf333_v3 -- metrics/by_metric/vina_kcal_per_mol.json`` is
empty, so the file has never contained such an entry, and the tests
have been red since they were written.

More importantly, the claim they encoded was not measurable.  In
``molmetal/reports/wf333v3_vina_p2p3/smoke_per_pocket.json`` every
pocket's "V3 combined" value is the REAL Vina score plus a fixed
−4.5 kcal/mol constant (per-centre −1.5 + per-bond −3.0, with the
distance term folded into the same ladder)::

    1h36: vina −3.589 → v3 −8.089   (Δ = −4.5)
    4aaw: vina −3.282 → v3 −7.782   (Δ = −4.5)

The "lift" is therefore arithmetic on a hand-set constant, not a
docking run, and ``TODO/completed/333_final_synthesis.md`` §0 tags the
V3 −1.5 delta PROJECTED.  The same JSON's own
``aggregate.n_pockets_pass_cutoff_minus7`` is 0 — the file contradicted
the test that read it.

What this file asserts now
--------------------------
1. The V3 ladder is provably a constant offset (per pocket AND in the
   smoke headline), so it cannot be reported as a measurement.
2. The JSON records no ``wf333_v3_*`` measurement entry.
3. The only REAL number is the cisplatin positive control, and it is
   flagged as a control rather than passed off as a model score.
4. The old mislabelled key ``v3_cohort_median_REAL`` is gone.
5. The measured values do not pass the −7.0 industry cutoff.
6. The tuning block that quotes −7.782 is tagged PROJECTED.

Per AGENTS.md §5.2 every assertion here is either arithmetic over
on-disk files or a structural presence/absence check.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
JSON_PATH = REPO_ROOT / "metrics" / "by_metric" / "vina_kcal_per_mol.json"
SMOKE_PATH = (
    REPO_ROOT / "molmetal" / "reports" / "wf333v3_vina_p2p3" / "smoke_per_pocket.json"
)

# The constant the V3 ladder adds to every pocket's real Vina score.
V3_CONSTANT_OFFSET_KCAL_MOL = -4.5
INDUSTRY_CUTOFF_KCAL_MOL = -7.0


def _load_json(path: Path) -> Dict[str, Any]:
    assert path.exists(), f"required JSON does not exist: {path}"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _v3_measurement_entries(data: Dict[str, Any]) -> list:
    return [
        m
        for m in data.get("measurements", [])
        if isinstance(m, dict)
        and isinstance(m.get("run", ""), str)
        and m["run"].startswith("wf333_v3_")
    ]


# ---------------------------------------------------------------------------
# The ladder is arithmetic, not a docking run
# ---------------------------------------------------------------------------
class TestV3LadderIsArithmetic:
    def test_inputs_exist(self) -> None:
        assert JSON_PATH.exists(), f"vina_kcal_per_mol.json missing at {JSON_PATH}"
        assert SMOKE_PATH.exists(), f"smoke_per_pocket.json missing at {SMOKE_PATH}"

    def test_json_parses(self) -> None:
        data = _load_json(JSON_PATH)
        assert isinstance(data, dict)
        for key in ("measurements", "aggregate", "verdict"):
            assert key in data, f"vina_kcal_per_mol.json missing {key!r}"

    def test_every_pocket_offset_is_the_same_constant(self) -> None:
        """v3_combined − vina must equal −4.5 for every pocket.

        A real re-bench would not produce a constant difference; a
        constant difference is proof the V3 number is derived.
        """
        smoke = _load_json(SMOKE_PATH)
        per_pocket = smoke["per_pocket"]
        assert per_pocket, "smoke_per_pocket.json has no per_pocket entries"
        for pocket, row in per_pocket.items():
            offset = row["v3_combined_kcal_mol"] - row["vina_kcal_mol"]
            assert offset == pytest.approx(
                V3_CONSTANT_OFFSET_KCAL_MOL, abs=1e-6
            ), (
                f"pocket {pocket}: v3 − vina = {offset}, expected the constant "
                f"{V3_CONSTANT_OFFSET_KCAL_MOL}"
            )

    def test_headline_median_offset_is_the_same_constant(self) -> None:
        smoke = _load_json(SMOKE_PATH)
        headline = smoke["headline"]
        offset = (
            headline["cohort_v3_combined_median_kcal_mol"]
            - headline["cohort_v0_vina_median_kcal_mol"]
        )
        assert offset == pytest.approx(V3_CONSTANT_OFFSET_KCAL_MOL, abs=1e-6), (
            f"headline V3 − V0 = {offset}, expected {V3_CONSTANT_OFFSET_KCAL_MOL}"
        )

    def test_the_real_median_does_not_pass_the_cutoff(self) -> None:
        """The measured V0 median sits well above −7.0; only the offset does."""
        smoke = _load_json(SMOKE_PATH)
        v0 = smoke["headline"]["cohort_v0_vina_median_kcal_mol"]
        assert v0 > INDUSTRY_CUTOFF_KCAL_MOL, (
            f"measured V0 median {v0} unexpectedly reaches the {INDUSTRY_CUTOFF_KCAL_MOL} cutoff"
        )


# ---------------------------------------------------------------------------
# The JSON must not dress the ladder up as a measurement
# ---------------------------------------------------------------------------
class TestJsonDoesNotClaimAMeasurement:
    def test_no_v3_measurement_entry(self) -> None:
        """A wf333_v3_* entry would record arithmetic as a run."""
        data = _load_json(JSON_PATH)
        entries = _v3_measurement_entries(data)
        assert not entries, (
            "vina_kcal_per_mol.json records wf333_v3_* as a measurement: "
            f"{[m.get('run') for m in entries]}. The V3 value is the real Vina "
            "score plus a fixed −4.5 kcal/mol constant and belongs in a "
            "PROJECTED block, not in `measurements`."
        )

    def test_old_mislabelled_key_is_gone(self) -> None:
        """`v3_cohort_median_REAL` held the cisplatin control, not a cohort median."""
        data = _load_json(JSON_PATH)
        assert "v3_cohort_median_REAL" not in data, (
            "v3_cohort_median_REAL is back: that name asserts a generated-molecule "
            "cohort median while its value is aggregate.cisplatin_mean_kcal_per_mol"
        )

    def test_control_is_recorded_as_a_control(self) -> None:
        data = _load_json(JSON_PATH)
        value = data.get("cisplatin_reference_mean_kcal_per_mol")
        assert value is not None, "cisplatin_reference_mean_kcal_per_mol missing"
        assert value < 0.0, f"a Vina score must be negative kcal/mol, got {value}"
        assert data.get("cisplatin_reference_mean_is_control") is True, (
            "the cisplatin mean must be flagged as a positive control so it is "
            "not read as a generated-molecule score"
        )

    def test_measured_values_do_not_pass_the_cutoff(self) -> None:
        data = _load_json(JSON_PATH)
        agg = data["aggregate"]
        assert agg["n_pockets_pass_cutoff_minus7"] == 0, (
            "n_pockets_pass_cutoff_minus7 must stay 0 — no measured pocket "
            f"reaches −7.0, got {agg['n_pockets_pass_cutoff_minus7']}"
        )
        assert agg["cisplatin_mean_kcal_per_mol"] > INDUSTRY_CUTOFF_KCAL_MOL, (
            "the measured cisplatin mean must not be reported as passing −7.0"
        )

    def test_projected_block_is_tagged(self) -> None:
        data = _load_json(JSON_PATH)
        block = data["wf432a1_vina_param_tuning_block"]
        assert block.get("MEASURED_vs_PROJECTED") == "PROJECTED", (
            "the block that quotes −7.782 must be tagged PROJECTED"
        )
        status = block.get("current_status", "")
        assert not status.startswith("MEASURED"), (
            f"current_status still opens with a MEASURED claim: {status[:80]!r}"
        )
        assert "PROJECTED" in status, (
            "current_status must say the −7.782 is PROJECTED"
        )
