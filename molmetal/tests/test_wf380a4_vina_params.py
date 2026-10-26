"""Wf-380-A4 — m3 Vina docking parameter tuning tests.

================================================================
Why this test file exists
================================================================
TODO-203 (m3 Vina docking tuning: exhaustiveness + n_poses) calls
for a 3-level Vina exhaustiveness ladder. Wf-380-A4 ships:

1. ``molmetal/configs/vina_docking_params.yaml`` — 3-level config
2. ``molmetal/scripts/wf380a4_vina_param_tuning.py`` — synthetic
   per-level kcal/mol generator + report writer
3. ``molmetal/reports/wf380a4_vina_tuning_smoke/result.json`` —
   smoke report with per-level cohort median kcal/mol

This test file covers four contracts:

  (i)   The 3-level config loads successfully and the per-level
        (exhaustiveness, n_poses) pairs match the documented spec:
            low  -> (8, 9)   baseline
            med  -> (32, 20) lift target
            high -> (64, 50) PROJECTED ceiling

  (ii)  ``tune_docking(cohort, level)`` returns deterministic
        per-pocket seed kcal/mol + per-pocket median kcal/mol +
        cohort median kcal/mol for each of the 3 levels.

  (iii) The 3-level cohort median kcal/mol follows the Wf-141 §0
        PROJECTED lift curve: each higher level is more negative
        than the previous one (more binding = lower kcal/mol).
        Specifically:
            medium < low  (delta <= -0.3 kcal/mol, PROJECTED -0.5)
            high   < med  (delta <= -0.2 kcal/mol, PROJECTED -0.5)

  (iv)  ``write_smoke_report`` emits a JSON file with the canonical
        schema (workflow / phase / per_level / aggregate sections).

All tests are CPU-only and do NOT launch the Vina binary.
"""

from __future__ import annotations

import json
import logging
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "molmetal" / "scripts" / "wf380a4_vina_param_tuning.py"
CONFIG_PATH = ROOT / "molmetal" / "configs" / "vina_docking_params.yaml"
REPORT_PATH = (
    ROOT / "molmetal" / "reports" / "wf380a4_vina_tuning_smoke" / "result.json"
)
_SCRIPT_DIR = ROOT / "molmetal" / "scripts"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

LOG = logging.getLogger("test_wf380a4_vina_params")


# ---------------------------------------------------------------------------
# Module-scope fixture: run the script once, load both report and config
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def _script_run() -> dict:
    """Run the tuning script once per module and return parsed report."""
    proc = subprocess.run(
        [sys.executable, str(SCRIPT)],
        check=True,
        capture_output=True,
        text=True,
    )
    LOG.debug("script stdout: %s", proc.stdout)
    LOG.debug("script stderr: %s", proc.stderr)
    return json.loads(REPORT_PATH.read_text())


@pytest.fixture(scope="module")
def _config() -> dict:
    """Load the 3-level config via the script's load_vina_config()."""
    from molmetal.scripts.wf380a4_vina_param_tuning import load_vina_config

    return load_vina_config(CONFIG_PATH)


@pytest.fixture(scope="module")
def _tune_docking():
    """Import the tune_docking function for direct unit testing."""
    from molmetal.scripts.wf380a4_vina_param_tuning import tune_docking

    return tune_docking


# ---------------------------------------------------------------------------
# (i) 3-level config loads + per-level (exhaustiveness, n_poses) pairs
# ---------------------------------------------------------------------------
def test_01_config_levels_present(_config: dict) -> None:
    """All 3 levels (low, medium, high) are present in the config."""
    levels = _config["levels"]
    assert "low" in levels, "missing level 'low'"
    assert "medium" in levels, "missing level 'medium'"
    assert "high" in levels, "missing level 'high'"


def test_02_config_exhaustiveness_values(_config: dict) -> None:
    """Per-level exhaustiveness matches the documented spec."""
    levels = _config["levels"]
    assert int(levels["low"]["exhaustiveness"]) == 8
    assert int(levels["medium"]["exhaustiveness"]) == 32
    assert int(levels["high"]["exhaustiveness"]) == 64


def test_03_config_n_poses_values(_config: dict) -> None:
    """Per-level n_poses matches the documented spec."""
    levels = _config["levels"]
    assert int(levels["low"]["n_poses"]) == 9
    assert int(levels["medium"]["n_poses"]) == 20
    assert int(levels["high"]["n_poses"]) == 50


def test_04_config_use_cases_present(_config: dict) -> None:
    """Each level carries a non-empty use_case string."""
    levels = _config["levels"]
    for lvl in ("low", "medium", "high"):
        use_case = str(levels[lvl]["use_case"])
        assert use_case, f"level {lvl} missing use_case string"
        assert len(use_case) >= 10, f"level {lvl} use_case too short: {use_case!r}"


# ---------------------------------------------------------------------------
# (ii) tune_docking returns deterministic per-(pocket, seed) kcal/mol
# ---------------------------------------------------------------------------
def test_05_tune_docking_returns_3_levels(_tune_docking) -> None:
    """tune_docking at each level returns the canonical 5-pocket cohort."""
    cohort = {"pockets": ["1h36", "3eml", "3ny8", "4aaw", "4rlu"], "seeds": [0, 1, 2]}
    for lvl in ("low", "medium", "high"):
        out = _tune_docking(cohort, lvl)
        assert out["level"] == lvl
        assert out["pockets"] == cohort["pockets"]
        assert out["seeds"] == cohort["seeds"]
        assert set(out["per_pocket_seed_kcal"]) == set(cohort["pockets"])
        for pocket in cohort["pockets"]:
            assert len(out["per_pocket_seed_kcal"][pocket]) == len(cohort["seeds"])
        assert set(out["per_pocket_median_kcal"]) == set(cohort["pockets"])


def test_06_tune_docking_deterministic(_tune_docking) -> None:
    """Same (cohort, level) call returns the same per-pocket median."""
    cohort = {"pockets": ["1h36", "3eml", "3ny8", "4aaw", "4rlu"], "seeds": [0, 1, 2]}
    a = _tune_docking(cohort, "low")
    b = _tune_docking(cohort, "low")
    assert a["per_pocket_seed_kcal"] == b["per_pocket_seed_kcal"]
    assert a["per_pocket_median_kcal"] == b["per_pocket_median_kcal"]
    assert a["cohort_median_kcal"] == b["cohort_median_kcal"]


# ---------------------------------------------------------------------------
# (iii) 3-level cohort median kcal/mol follows Wf-141 §0 lift curve
# ---------------------------------------------------------------------------
def test_07_cohort_median_lift_curve(_tune_docking) -> None:
    """Each higher level is more negative than the previous (lift curve)."""
    cohort = {"pockets": ["1h36", "3eml", "3ny8", "4aaw", "4rlu"], "seeds": [0, 1, 2]}
    low = _tune_docking(cohort, "low")["cohort_median_kcal"]
    med = _tune_docking(cohort, "medium")["cohort_median_kcal"]
    high = _tune_docking(cohort, "high")["cohort_median_kcal"]
    # More negative = more binding.
    assert med < low, f"medium ({med}) should be < low ({low})"
    assert high < med, f"high ({high}) should be < medium ({med})"


def test_08_lift_bounds_per_wf141(_tune_docking) -> None:
    """Lift bounds per Wf-141 §0: medium-vs-low <= -0.3; high-vs-medium <= -0.2."""
    cohort = {"pockets": ["1h36", "3eml", "3ny8", "4aaw", "4rlu"], "seeds": [0, 1, 2]}
    low = _tune_docking(cohort, "low")["cohort_median_kcal"]
    med = _tune_docking(cohort, "medium")["cohort_median_kcal"]
    high = _tune_docking(cohort, "high")["cohort_median_kcal"]
    med_lift = med - low
    high_lift = high - med
    assert med_lift <= -0.3, f"medium-vs-low lift {med_lift} not <= -0.3"
    assert high_lift <= -0.2, f"high-vs-medium lift {high_lift} not <= -0.2"


def test_09_high_below_industry_cutoff(_tune_docking) -> None:
    """`high` level cohort median is <= -7.0 kcal/mol industry cutoff."""
    cohort = {"pockets": ["1h36", "3eml", "3ny8", "4aaw", "4rlu"], "seeds": [0, 1, 2]}
    high = _tune_docking(cohort, "high")["cohort_median_kcal"]
    assert high <= -7.0, f"high cohort median {high} not <= industry cutoff -7.0"


# ---------------------------------------------------------------------------
# (iv) write_smoke_report emits a JSON with the canonical schema
# ---------------------------------------------------------------------------
def test_10_report_schema_present(_script_run: dict) -> None:
    """Smoke report contains all canonical top-level keys."""
    expected = {
        "_workflow",
        "_phase",
        "_title",
        "_generated_utc",
        "_hardware",
        "_config_path",
        "_n_pockets",
        "_n_seeds",
        "_smiles",
        "_industry_cutoff_kcal_mol",
        "_cohort",
        "_per_level",
        "_aggregate",
        "_honest_framing",
        "_cross_references",
    }
    assert expected.issubset(_script_run.keys())


def test_11_report_per_level_keys(_script_run: dict) -> None:
    """Smoke report `_per_level` contains all 3 levels."""
    per_level = _script_run["_per_level"]
    assert "low" in per_level
    assert "medium" in per_level
    assert "high" in per_level


def test_12_report_aggregate_lift(_script_run: dict) -> None:
    """Smoke report `_aggregate.lift_vs_low` shows medium and high lifts."""
    lift = _script_run["_aggregate"]["lift_vs_low"]
    assert "low" in lift
    assert "medium" in lift
    assert "high" in lift
    # Lift vs low: low=0, medium<0, high<0
    assert lift["low"] == pytest.approx(0.0, abs=1e-3)
    assert lift["medium"] < 0.0
    assert lift["high"] < 0.0
    # high lift should be more negative than medium lift
    assert lift["high"] < lift["medium"]


def test_13_report_honest_framing(_script_run: dict) -> None:
    """Smoke report honest framing declares SYNTHETIC_PLACEHOLDER status."""
    hf = _script_run["_honest_framing"]
    assert hf["status"] == "SYNTHETIC_PLACEHOLDER"
    assert "real_validation_blocker" in hf
    assert "Vina" in hf["real_validation_blocker"]
    assert "USER_OK" in hf["real_validation_blocker"]


def test_14_report_well_formed_json() -> None:
    """The result.json on disk parses cleanly (no truncation)."""
    text = REPORT_PATH.read_text()
    parsed = json.loads(text)
    assert isinstance(parsed, dict)
    assert parsed["_workflow"] == "Wf-380-A4"


# ---------------------------------------------------------------------------
# The loader (2026-10-27) — a block sequence was being silently DROPPED
# ---------------------------------------------------------------------------
# `cohort.pockets` is written as
#
#     pockets:
#     - 1h36
#     - 3eml
#
# which is a block sequence at the SAME indent as its key. The old loader
# created a dict for any `key:` with no inline value and then tested
# `isinstance(parent, list)` against that dict, so every `- item` line was
# discarded without a word. `pockets` came back as `{}`, `list({})` as `[]`,
# and `statistics.median([])` killed the script — which is why the four report
# tests above used to ERROR at setup rather than run.


def test_15_block_sequences_load_as_lists() -> None:
    from molmetal.scripts.wf380a4_vina_param_tuning import load_vina_config

    cfg = load_vina_config(CONFIG_PATH)
    cohort = cfg["cohort"]
    assert cohort["pockets"] == ["1h36", "3eml", "3ny8", "4aaw", "4rlu"], (
        "cohort.pockets must be the five-item list the YAML declares, not the "
        f"empty mapping the old loader returned: {cohort['pockets']!r}"
    )
    assert cohort["seeds"] == [0, 1, 2]
    # Every other block sequence in the same file was dropped the same way.
    assert len(cfg["cross_references"]) == 12
    assert len(cfg["honest_framing"]["claims"]) == 4
    assert len(cfg["directives"]["pre_flight_gates_required"]) == 4


def test_16_a_pending_key_becomes_a_list_only_when_an_item_follows() -> None:
    """`key:` + `- x` is a list; `key:` + `k: v` is still a mapping."""
    from molmetal.scripts.wf380a4_vina_param_tuning import _load_yaml_simple

    import tempfile

    body = (
        "a:\n"
        "  list_key:\n"
        "  - one\n"
        "  - two\n"
        "  map_key:\n"
        "    inner: 1\n"
        "  scalar_key: 7\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write(body)
        path = Path(fh.name)
    try:
        cfg = _load_yaml_simple(path)
    finally:
        path.unlink()
    assert cfg["a"]["list_key"] == ["one", "two"]
    assert cfg["a"]["map_key"] == {"inner": 1}
    assert cfg["a"]["scalar_key"] == 7


def test_17_an_unsupported_construct_raises_instead_of_being_dropped() -> None:
    """Silent loss is the bug; refusing to parse is the fix."""
    from molmetal.scripts.wf380a4_vina_param_tuning import _load_yaml_simple

    import tempfile

    body = "a:\n  - key: value\n"
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write(body)
        path = Path(fh.name)
    try:
        with pytest.raises(ValueError, match="mapping items inside a block"):
            _load_yaml_simple(path)
    finally:
        path.unlink()


def test_18_an_empty_cohort_is_named_at_the_boundary() -> None:
    """An empty cohort must not surface as `StatisticsError: no median`."""
    from molmetal.scripts.wf380a4_vina_param_tuning import load_vina_config

    import tempfile

    body = (
        "levels:\n"
        "  low:\n    exhaustiveness: 8\n"
        "  medium:\n    exhaustiveness: 32\n"
        "  high:\n    exhaustiveness: 64\n"
        "cohort:\n"
        "  pockets: []\n"
        "  seeds:\n"
        "  - 0\n"
    )
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as fh:
        fh.write(body)
        path = Path(fh.name)
    try:
        with pytest.raises(RuntimeError, match=r"cohort\.pockets"):
            load_vina_config(path)
    finally:
        path.unlink()
