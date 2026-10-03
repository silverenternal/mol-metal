"""Wf-390-A1 — m2 R² Path A 5-metal subset smoke tests.

=================================================================
Wf-390-A1 — CPU-only tests for the Path A 5-metal subset smoke
``molmetal/scripts/wf390a1_path_a_smoke.py``. Verifies that the
existing trainer ``train_dmpnn_multitask_wasserstein.py`` accepts
the 5-metal subset natively (no CHOICES constraint violation per
``FIVE_METALS = ('Ru', 'Ir', 'Rh', 'Os', 'Re')`` line 141) and
emits a per-cohort config JSON for Path A retrain (per Wf-380-A2 +
Wf-388-A1 SCAFFOLD/APPLIED pattern).

Per the Wf-390-A1 task brief, 1 NEW test verifies the smoke:

1. test_wf390a1_path_a_smoke_passes
   - smoke_path_a_5metal() returns smoke_passed=True
   - FIVE_METALS tuple matches PATH_A_METALS exactly
   - Wf-379-A2 11-metal set is REJECTED with SystemExit 2 (negative test
     validates the CHOICES constraint is still active)
   - build_path_a_aggregate() emits valid aggregate.json with 7 cohort
     configs + total_wall_projected_hr=12.0 + max_vram_projected_gb=12.0

All tests are CPU-only (per AGENTS.md §1.2 + §5.1 + §5.4 — no GPU
launches). Per ``party-twenty-spirit`` §3.5: every claim traces to a
file/line or live-run output. No ``paper/*`` touches (TODO-84 honored).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# Load the smoke module via importlib (matches Wf-376-A2 test pattern).
_smoke_spec = importlib.util.spec_from_file_location(
    "wf390a1_smoke_module",
    str(PROJECT_ROOT / "molmetal" / "scripts" / "wf390a1_path_a_smoke.py"),
)
assert _smoke_spec is not None and _smoke_spec.loader is not None
_wf390a1_mod = importlib.util.module_from_spec(_smoke_spec)
_smoke_spec.loader.exec_module(_wf390a1_mod)

smoke_mod = _wf390a1_mod


# ---------------------------------------------------------------------------
# Test
# ---------------------------------------------------------------------------
def test_wf390a1_path_a_smoke_passes() -> None:
    """Wf-390-A1 smoke: trainer accepts 5-metal subset + emits config JSON."""
    # Run the smoke
    smoke_result = smoke_mod.smoke_path_a_5metal()

    # Assert FIVE_METALS tuple matches PATH_A_METALS exactly
    assert smoke_result["path_a_metals_match"] is True, (
        f"Path A 5-metal must match trainer FIVE_METALS exactly; "
        f"got {smoke_result['path_a_metals_parsed']} vs expected "
        f"{list(smoke_mod.PATH_A_METALS)}"
    )

    # Assert argparse accepts the 5-metal subset (no CHOICES violation)
    assert smoke_result["path_a_metals_parsed"] == list(smoke_mod.PATH_A_METALS), (
        f"parse_args must accept Path A 5-metal subset; got "
        f"{smoke_result['path_a_metals_parsed']}"
    )

    # Assert Wf-379-A2 11-metal set is REJECTED with SystemExit 2
    # (negative test validates the CHOICES constraint is still active)
    assert smoke_result["wf379a2_11metal_rejected"] is True, (
        f"Wf-379-A2 11-metal set MUST be rejected (argparse choices "
        f"constraint); got rejected={smoke_result['wf379a2_11metal_rejected']}"
    )
    assert smoke_result["wf379a2_11metal_systemexit_code"] == 2, (
        f"Wf-379-A2 11-metal rejection must be argparse SystemExit 2; "
        f"got {smoke_result['wf379a2_11metal_systemexit_code']}"
    )

    # Overall smoke passed
    assert smoke_result["smoke_passed"] is True

    # Build the aggregate and validate the schema
    aggregate = smoke_mod.build_path_a_aggregate(smoke_result)
    assert aggregate["wf"] == "Wf-390-A1"
    assert aggregate["user_ok_required"] is True
    assert "Wf-355-B1" in aggregate["user_ok_gate_pattern"]

    # 7 cohort configs (3 multi-seed + 4 per-metal singlets; Re counts too)
    assert aggregate["n_cohort_configs"] == 7
    assert len(aggregate["cohort_configs"]) == 7

    # Aggregate wall + VRAM match the 3-path plan
    assert aggregate["total_wall_projected_hr"] == 12.0
    assert aggregate["total_vram_projected_gb"] == 12.0

    # Honest framing must reflect SCAFFOLD + no GPU launch
    hf = aggregate["honest_framing"]
    assert hf["scope"].startswith("CPU-only")
    assert hf["smoke_passes_argparse_check"] is True
    assert hf["lift_values_are_projected_not_measured"] is True
    assert hf["user_ok_required_for_actual_launch"] is True
    # kill_then_adapt_directive_honored is a paraphrase of the directive,
    # not a literal substring match — assert on key markers instead
    kill_then_adapt_text = hf["kill_then_adapt_directive_honored"].lower()
    assert "user_ok" in kill_then_adapt_text
    assert "gpu" in kill_then_adapt_text
    assert "pre-flight" in kill_then_adapt_text or "preflight" in kill_then_adapt_text

    # Verify aggregate.json round-trips through JSON serialization
    serialized = json.dumps(aggregate, sort_keys=False)
    deserialized = json.loads(serialized)
    assert deserialized["smoke_result"]["smoke_passed"] is True
    assert deserialized["n_cohort_configs"] == 7

    # Verify all cohort configs have the required keys
    required_keys = {
        "cohort_id",
        "n_smiles",
        "csv_path",
        "metals",
        "seeds",
        "epochs",
        "hidden",
        "depth",
        "w2_weight",
        "alpha",
        "ablation_config",
        "single_task",
        "baseline_r2_measured",
        "baseline_source",
        "projected_r2_low",
        "projected_r2_high",
        "wall_projected_hr",
        "vram_projected_gb",
    }
    for cfg in aggregate["cohort_configs"]:
        missing = required_keys - set(cfg.keys())
        assert not missing, f"cohort config missing keys: {missing}"

    # Verify each cohort's metals is a subset of PATH_A_METALS
    for cfg in aggregate["cohort_configs"]:
        for m in cfg["metals"]:
            assert m in smoke_mod.PATH_A_METALS, (
                f"cohort {cfg['cohort_id']} has metal {m} not in "
                f"FIVE_METALS {list(smoke_mod.PATH_A_METALS)}"
            )
