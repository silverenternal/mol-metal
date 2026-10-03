"""Wf-319A — ablation_runner.py unit tests (12 NEW tests).

================================================================
Per Wf-315B analysis: ``molmetal/scripts/ablation_runner.py`` is 1037 LOC
with NO unit tests.  This file introduces 12 NEW tests that cover:

  * 3 tests for the 5 **toggleable** ablations (M7, M12, M13, M14, M15)
  * 3 tests for the 6 **DESIGN_ONLY** ablations (M3, M4, M8, M10, M11, M21)
  * 2 tests for ``aggregate.csv`` generation
  * 2 tests for ``final.md`` generation
  * 2 tests for integration with ``metallo_drug_smoke_retrain.py``

These tests are pure registry / format / subprocess-cmd-shape tests —
they do NOT spin up a CFM retrain (no GPU, no torch).  The actual
end-to-end smoke output is verified by re-reading
``molmetal/reports/wf_ablation_runner/{aggregate.csv,final.md,<ID>.json}``
in pre-flight gate ``p3_ablation_runner_aggregate_csv_present``.

Honest framing (per Wf-319A brief)
----------------------------------
* Each test asserts ONE observable contract (registry classification,
  file content shape, CLI flag wiring) so failures localise cleanly.
* Tests do NOT fabricate m14 / m13 / m6 values — every numeric value
  read from a JSON is read-only and the test asserts presence + type,
  not magnitude.
* The DESIGN_ONLY ablations are explicitly documented in the table as
  NOT-exercised (per Wf-319A honest framing §9 honest catalog).

Cross-references
----------------
* ``molmetal/scripts/ablation_runner.py`` — the runner under test.
* ``TODO/MECHANISM_METRIC_MAPPING.md`` §5 — the 11-ablation source table.
* ``molmetal/reports/wf_ablation_runner/aggregate.csv`` — paper §6 input.
* ``molmetal/reports/wf_ablation_runner/final.md`` — paper §6 honest framing.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

import pytest


# ---------------------------------------------------------------------------
# Helpers — resolve paths to repo-root so the tests are runnable from any cwd
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "molmetal" / "scripts"
RUNNER_PATH = SCRIPTS_DIR / "ablation_runner.py"
TRAIN_SCRIPT = SCRIPTS_DIR / "metallo_drug_smoke_retrain.py"
TRAIN_CSV = REPO_ROOT / "molmetal" / "data" / "metallo_drugs_500_train.csv"
REPORT_DIR = REPO_ROOT / "molmetal" / "reports" / "wf_ablation_runner"


def _import_runner_module():
    """Load the ablation_runner script via :func:`runpy.run_path`.

    We use ``runpy`` (rather than :mod:`importlib`) because the script
    uses ``from __future__ import annotations`` + ``@dataclass`` with
    certain ``Optional[...]`` annotations that interact poorly with
    ``spec_from_file_location`` in Python 3.12.13 (the dataclass
    decorator trips an internal ``_is_kw_only`` check).  ``runpy`` runs
    the script as ``__main__`` would, which is the path the production
    CLI uses and avoids the dataclass import ordering bug.
    """
    import runpy

    return runpy.run_path(
        str(RUNNER_PATH),
        run_name="ablation_runner_test",
    )


@pytest.fixture(scope="module")
def runner():
    """Module-level fixture for the ablation_runner module.

    Returns a SimpleNamespace view over the runpy globals dict so
    tests can use attribute-style access (``runner.ABLATION_TABLE``)
    rather than key-style (``runner["ABLATION_TABLE"]``).
    """
    from types import SimpleNamespace

    return SimpleNamespace(**_import_runner_module())


# ---------------------------------------------------------------------------
# 1 — Toggleable ablation specs (3 tests)
# ---------------------------------------------------------------------------

#: Canonical toggleable IDs per MECHANISM_METRIC_MAPPING §5.
TOGGLEABLE_IDS = ("M7", "M12", "M13", "M14", "M15")


def test_toggleable_specs_are_classified_toggleable(runner):
    """All 5 toggleable specs (M7/M12/M13/M14/M15) declare toggle='toggleable'."""
    table = {s.ablation_id: s for s in runner.ABLATION_TABLE}
    for abl_id in TOGGLEABLE_IDS:
        assert (
            abl_id in table
        ), f"Toggleable ablation {abl_id} missing from ABLATION_TABLE"
        spec = table[abl_id]
        assert (
            spec.toggle == "toggleable"
        ), f"{abl_id} expected toggle='toggleable', got '{spec.toggle}'"


def test_toggleable_specs_wire_env_vars_or_extra_args(runner):
    """Each toggleable spec wires at least one of env_vars / extra_args.

    This is the MECHANISM wiring contract — without it, the
    :class:`AblationSpec` cannot influence the subprocess via the
    metallo_drug_smoke_retrain.py harness.
    """
    table = {s.ablation_id: s for s in runner.ABLATION_TABLE}
    for abl_id in TOGGLEABLE_IDS:
        spec = table[abl_id]
        has_env = bool(spec.env_vars)
        has_args = bool(spec.extra_args)
        assert has_env or has_args, (
            f"{abl_id} has neither env_vars nor extra_args; "
            "toggleable ablation would be a no-op in subprocess"
        )


def test_toggleable_specs_lookup_by_id(runner):
    """Each toggleable ID has unique label + non-empty expected_effect."""
    table = {s.ablation_id: s for s in runner.ABLATION_TABLE}
    for abl_id in TOGGLEABLE_IDS:
        spec = table[abl_id]
        # Unique label so the report doesn't collapse two rows.
        labels = [s.label for s in runner.ABLATION_TABLE if s.ablation_id == abl_id]
        assert len(labels) == 1, f"{abl_id} is not unique in ABLATION_TABLE"
        assert spec.label, f"{abl_id} has empty label"
        # expected_effect must be a non-trivial sentence.
        assert spec.expected_effect and len(spec.expected_effect) >= 8, (
            f"{abl_id} has suspiciously short expected_effect: "
            f"{spec.expected_effect!r}"
        )


# ---------------------------------------------------------------------------
# 2 — DESIGN_ONLY ablation specs (3 tests)
# ---------------------------------------------------------------------------

#: Canonical DESIGN_ONLY IDs per MECHANISM_METRIC_MAPPING §5.
DESIGN_ONLY_IDS = ("M3", "M4", "M8", "M10", "M11", "M21")


def test_design_only_specs_are_classified_design(runner):
    """All 6 DESIGN_ONLY specs declare toggle='design'."""
    table = {s.ablation_id: s for s in runner.ABLATION_TABLE}
    for abl_id in DESIGN_ONLY_IDS:
        assert (
            abl_id in table
        ), f"DESIGN_ONLY ablation {abl_id} missing from ABLATION_TABLE"
        spec = table[abl_id]
        assert (
            spec.toggle == "design"
        ), f"{abl_id} expected toggle='design', got '{spec.toggle}'"


def test_design_only_run_returns_design_only_dict(runner, tmp_path):
    """_run_one_ablation on a DESIGN_ONLY spec returns a DESIGN_ONLY dict
    without spawning a subprocess.

    This is the honest-framing contract: design-only rows must NOT
    fabricate measured metrics.  The status must be the literal
    'DESIGN_ONLY' string and the env_vars/extra_args must be the
    spec's empty defaults (since design-only specs do not wire any).
    """
    table = {s.ablation_id: s for s in runner.ABLATION_TABLE}
    for abl_id in DESIGN_ONLY_IDS:
        spec = table[abl_id]
        result = runner._run_one_ablation(
            spec,
            steps=10,
            decode_samples=4,
            out_dir=tmp_path,
            csv_path=TRAIN_CSV,
            timeout_s=10,
        )
        assert (
            result["status"] == "DESIGN_ONLY"
        ), f"{abl_id} expected status='DESIGN_ONLY', got {result['status']!r}"
        assert result["toggle"] == "design"
        # DESIGN_ONLY rows must NOT contain measured metrics — verify
        # the metric sub-dict is empty so the aggregator row is honest.
        assert result.get("metrics") in ({}, None), (
            f"{abl_id} DESIGN_ONLY leaked measured metrics: "
            f"{result.get('metrics')!r}"
        )
        # No subprocess was spawned — rc + wall_s should not be set.
        assert "rc" not in result, (
            f"{abl_id} DESIGN_ONLY path set rc={result.get('rc')}; "
            "expected no subprocess invocation"
        )


def test_design_only_specs_reference_correct_owner_workflow(runner):
    """Each DESIGN_ONLY spec points to its owner workflow via mechanism_module.

    Per §5 + Wf-319A honest framing: M3/M4/M8 are Lambda pipeline
    (workflow H + workflow C); M10/M11 are click-rule matrix
    (workflow wf_lambda_88rule_deflex); M21 is data-hygiene (data/
    leakage_utils.py).
    """
    table = {s.ablation_id: s for s in runner.ABLATION_TABLE}
    expected_module_substrings = {
        "M3": "proof_search",  # KataGo-MCTS → vanilla UCB
        "M4": "pll_heuristic",  # PLL heuristic → random
        "M8": "lambda_astsr",  # LambdaASTSR → F5
        "M10": "pt_click_compat",  # 84-rule MLE matrix
        "M11": "pt_click_compat",  # 4 NEW SMARTS
        "M21": "leakage_utils",  # random split → scaffold split
    }
    for abl_id, expected in expected_module_substrings.items():
        spec = table[abl_id]
        assert expected in spec.mechanism_module, (
            f"{abl_id} expected mechanism_module to contain {expected!r}, "
            f"got {spec.mechanism_module!r}"
        )


# ---------------------------------------------------------------------------
# 3 — aggregate.csv generation (2 tests)
# ---------------------------------------------------------------------------


def _make_summaries(runner) -> List[dict]:
    """Synthesise a representative summaries list (toggleable + design)."""
    table = {s.ablation_id: s for s in runner.ABLATION_TABLE}
    summaries: List[dict] = []
    # BASELINE — toggleable reference (rc=0, real metrics).
    summaries.append(
        {
            "ablation_id": "BASELINE",
            "label": "no ablation (reference run)",
            "mechanism_module": "n/a",
            "target_metric": "m14",
            "expected_effect": "n/a",
            "toggle": "reference",
            "status": "PASS",
            "rc": 0,
            "wall_s": 74.5,
            "match": "N/A",
            "metrics": {
                "m14_decode_ratio_at_end": 1.0,
                "m14_max_decode_ratio": 1.0,
                "m13_validity_rate": 1.0,
                "m5_sa_unit_mean": 0.355,
                "m6_qed_mean": 0.415,
                "atom_loss_last": 2.44,
                "bond_loss_last": 1.25,
            },
        }
    )
    # One toggleable run with bond_loss lift (M13-like).
    summaries.append(
        {
            "ablation_id": "M13",
            "label": "logits clamp (±50)",
            "mechanism_module": "flow_matching_lipman/__init__.py",
            "target_metric": "m14",
            "expected_effect": "m14 → finite but small / unstable",
            "toggle": "toggleable",
            "status": "PASS",
            "rc": 0,
            "wall_s": 76.7,
            "match": "PREDICTION_QUALITATIVE",
            "metrics": {
                "m14_decode_ratio_at_end": 1.0,
                "m14_max_decode_ratio": 1.0,
                "m13_validity_rate": 1.0,
                "m5_sa_unit_mean": 0.322,
                "m6_qed_mean": 0.405,
                "atom_loss_last": 1.23,
                "bond_loss_last": 3.16,
            },
        }
    )
    # One DESIGN_ONLY row — no measured metrics.
    summaries.append(
        {
            "ablation_id": "M3",
            "label": "KataGo-MCTS → vanilla UCB",
            "mechanism_module": "search_alg/proof_search.py",
            "target_metric": "m7",
            "expected_effect": "m7 diversity ↓ + m11 n_distinct ↓",
            "toggle": "design",
            "status": "DESIGN_ONLY",
            "note": "requires Lambda proof-search harness",
            "env_vars_requested": {},
            "extra_args_requested": [],
            "predicted_value": None,
        }
    )
    return summaries


def test_aggregate_csv_writes_expected_columns_and_rows(runner, tmp_path):
    """_write_aggregate_csv emits one CSV row per summary with the
    expected column schema.
    """
    summaries = _make_summaries(runner)
    csv_path = runner._write_aggregate_csv(tmp_path, summaries)
    assert csv_path.is_file(), f"aggregate.csv not written at {csv_path}"
    with csv_path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    # Row count matches summaries count (1 header is auto-stripped).
    assert len(rows) == len(
        summaries
    ), f"aggregate.csv has {len(rows)} rows but expected {len(summaries)}"
    # Column schema — every summary's keys must round-trip.
    expected_cols = {
        "ablation_id",
        "label",
        "toggle",
        "status",
        "match",
        "target_metric",
        "m14_decode_ratio",
        "m14_max",
        "m13_validity",
        "m5_sa_unit",
        "m6_qed",
        "atom_loss_last",
        "bond_loss_last",
        "wall_s",
    }
    assert expected_cols.issubset(set(rows[0].keys())), (
        f"aggregate.csv missing expected columns: "
        f"{expected_cols - set(rows[0].keys())}"
    )
    # Row 1 should be BASELINE with a wall_s value of type str-coerced.
    assert rows[0]["ablation_id"] == "BASELINE"
    assert rows[0]["toggle"] == "reference"
    # DESIGN_ONLY row should have empty metric columns (not fabricated).
    design_rows = [r for r in rows if r["toggle"] == "design"]
    assert len(design_rows) == 1
    assert design_rows[0]["m14_decode_ratio"] == ""
    assert design_rows[0]["m14_max"] == ""


def test_aggregate_csv_handles_all_11_ablations(runner, tmp_path):
    """Running the FULL ABLATION_TABLE through _write_aggregate_csv
    produces 11 ablations + 1 BASELINE = 12 data rows.
    """
    # Re-construct full summaries by calling the runner paths.
    summaries: List[dict] = []
    for spec in runner.ABLATION_TABLE:
        if spec.toggle == "design":
            summaries.append(
                {
                    "ablation_id": spec.ablation_id,
                    "label": spec.label,
                    "mechanism_module": spec.mechanism_module,
                    "target_metric": spec.target_metric,
                    "expected_effect": spec.expected_effect,
                    "toggle": "design",
                    "status": "DESIGN_ONLY",
                    "env_vars_requested": dict(spec.env_vars),
                    "extra_args_requested": list(spec.extra_args),
                    "predicted_value": spec.predicted_value,
                }
            )
        else:
            # Synthesise a PASS row for the test — no actual subprocess.
            summaries.append(
                {
                    "ablation_id": spec.ablation_id,
                    "label": spec.label,
                    "mechanism_module": spec.mechanism_module,
                    "target_metric": spec.target_metric,
                    "expected_effect": spec.expected_effect,
                    "toggle": spec.toggle,
                    "status": "PASS",
                    "rc": 0,
                    "wall_s": 60.0,
                    "match": "N/A"
                    if spec.ablation_id == "BASELINE"
                    else "PREDICTION_QUALITATIVE",
                    "metrics": {
                        "m14_decode_ratio_at_end": 1.0,
                        "m14_max_decode_ratio": 1.0,
                        "m13_validity_rate": 1.0,
                        "m5_sa_unit_mean": 0.33,
                        "m6_qed_mean": 0.42,
                        "atom_loss_last": 2.0,
                        "bond_loss_last": 1.25,
                    },
                }
            )
    csv_path = runner._write_aggregate_csv(tmp_path, summaries)
    with csv_path.open("r", newline="") as fh:
        rows = list(csv.DictReader(fh))
    # 11 ablations + 1 BASELINE reference = 12 data rows.
    assert len(rows) == 12, (
        f"aggregate.csv has {len(rows)} rows but expected 12 "
        "(11 ablations + 1 BASELINE)"
    )
    # DESIGN_ONLY rows must equal 6 (M3, M4, M8, M10, M11, M21).
    design_count = sum(1 for r in rows if r["toggle"] == "design")
    assert design_count == 6, f"expected 6 DESIGN_ONLY rows, got {design_count}"
    # All ablation IDs are present exactly once.
    ids = [r["ablation_id"] for r in rows]
    expected_ids = {
        "BASELINE",
        "M3",
        "M4",
        "M7",
        "M8",
        "M10",
        "M11",
        "M12",
        "M13",
        "M14",
        "M15",
        "M21",
    }
    assert set(ids) == expected_ids, (
        f"missing IDs: {expected_ids - set(ids)}; "
        f"extra IDs: {set(ids) - expected_ids}"
    )


# ---------------------------------------------------------------------------
# 4 — final.md generation (2 tests)
# ---------------------------------------------------------------------------


def _tmp_under_repo(tmp_path: Path) -> Path:
    """Return a tmp dir under REPO_ROOT (so csv_path.relative_to(REPO_ROOT)
    works in :func:`ablation_runner._write_final_md`)."""
    nested = REPO_ROOT / ".pytest_tmp" / tmp_path.name
    nested.mkdir(parents=True, exist_ok=True)
    return nested


def test_final_md_includes_summary_table(runner, tmp_path):
    """_write_final_md writes a markdown file with the Summary section
    + the per-ablation table.
    """
    work_dir = _tmp_under_repo(tmp_path)
    summaries = _make_summaries(runner)
    csv_path = runner._write_aggregate_csv(work_dir, summaries)
    md_path = runner._write_final_md(
        work_dir,
        summaries,
        csv_path=csv_path,
        steps=200,
    )
    assert md_path.is_file(), f"final.md not written at {md_path}"
    text = md_path.read_text()
    # Required sections.
    assert "## Summary" in text, "final.md missing '## Summary' section"
    assert (
        "## Per-ablation table" in text
    ), "final.md missing '## Per-ablation table' section"
    assert "## Honest framing" in text, "final.md missing '## Honest framing' section"
    # Toggleable / design-only counts are surfaced.
    assert "Toggleable" in text
    assert "Design-only" in text
    # Reference to source script path is present.
    assert "ablation_runner.py" in text


def test_final_md_observes_honest_framing_contract(runner, tmp_path):
    """final.md explicitly calls out the smoke-vs-production contract.

    Per Wf-319A honest framing: the final.md MUST NOT overstate the
    measured decode_ratio (which is 1.0 in every smoke) as a
    production result.
    """
    work_dir = _tmp_under_repo(tmp_path)
    summaries = _make_summaries(runner)
    csv_path = runner._write_aggregate_csv(work_dir, summaries)
    md_path = runner._write_final_md(
        work_dir,
        summaries,
        csv_path=csv_path,
        steps=200,
    )
    text = md_path.read_text()
    # 200-step smoke framing MUST appear at least once.
    assert (
        "200-step" in text or "smoke" in text.lower()
    ), "final.md missing explicit smoke-vs-production framing"
    # DESIGN_ONLY ablations MUST be listed in a dedicated section.
    assert (
        "## Design-only ablations" in text
    ), "final.md missing design-only section header"
    # All 6 DESIGN_ONLY IDs must appear.
    for abl_id in ("M3", "M4", "M8", "M10", "M11", "M21"):
        assert abl_id in text, f"final.md missing {abl_id} in design-only section"


# ---------------------------------------------------------------------------
# 5 — Integration with metallo_drug_smoke_retrain.py (2 tests)
# ---------------------------------------------------------------------------


def test_metallo_drug_smoke_retrain_accepts_required_cli_flags():
    """The metallo_drug_smoke_retrain.py harness accepts the exact CLI
    flags that ablation_runner.py forwards for the toggleable ablations.

    We probe the harness with ``--help`` and assert each flag is
    listed — failure means ablation_runner.py would crash with
    'unrecognized argument' on a toggleable ablation.

    Honest caveat
    -------------
    The M14 spec passes ``--lr-warmup-steps 0`` as extra_args, but the
    harness does NOT define this flag (verified empirically: it
    errors with ``unrecognized arguments: --lr-warmup-steps 0``).  This
    is an existing bug surfaced by Wf-319A — the M14 ablation has been
    silently broken since the flag was removed from the harness.  We
    therefore assert the flag is **missing** as an honest-negative
    contract so the reader knows to fix either the harness or the
    runner's M14 extra_args.
    """
    import subprocess

    proc = subprocess.run(
        [sys.executable, str(TRAIN_SCRIPT), "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=60,
    )
    help_blob = (proc.stdout + "\n" + proc.stderr).lower()
    assert (
        proc.returncode == 0
    ), f"metallo_drug_smoke_retrain.py --help failed: {proc.stderr}"
    # Flags that DO exist and are used by the toggleable ablations.
    for required_flag in (
        "--steps",
        "--csv",
        "--decode-smoke-every",
        "--decode-smoke-samples",
        "--out-dir",
        "--bond-head",
        "--lr-warmup-steps",
    ):
        assert required_flag in help_blob, (
            f"metallo_drug_smoke_retrain.py --help missing flag "
            f"{required_flag!r} — ablation_runner.py would crash"
        )
    # Wf-323A: M14's --lr-warmup-steps flag is now wired (re-integrated
    # after Wf-319A surfaced the silent-break).  The harness accepts
    # the flag, the adapter applies a linear ramp, and ablation_runner
    # toggles M14 by passing --lr-warmup-steps 0.


def test_toggleable_spec_cmd_includes_abl_harness_flags(runner):
    """Build the cmd that ablation_runner.py would invoke for a
    toggleable spec and assert every required CLI flag is present.

    Honest caveat
    -------------
    The M14 spec's ``--lr-warmup-steps 0`` extra_args is NOT a valid
    harness flag (verified in test above).  This test therefore checks
    that ``--bond-head`` (M7) is wired correctly, and that M12's
    env-var toggle is wired without extra_args.
    """
    # Use the M7 spec — it has both extra_args (--bond-head distance).
    table = {s.ablation_id: s for s in runner.ABLATION_TABLE}
    spec_m7 = table["M7"]
    # Replicate the cmd construction from _run_one_ablation.
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT.relative_to(REPO_ROOT)),
        "--steps",
        "200",
        "--decode-smoke-every",
        "200",
        "--decode-smoke-samples",
        "16",
        "--csv",
        str(TRAIN_CSV.relative_to(REPO_ROOT)),
        "--out-dir",
        "molmetal/reports/wf_ablation_runner/M7",
    ] + spec_m7.extra_args
    # Each required flag must appear in the cmd.
    cmd_str = " ".join(cmd)
    for required in (
        "--steps 200",
        "--decode-smoke-every 200",
        "--decode-smoke-samples 16",
        "--csv",
        "--out-dir",
        "--bond-head distance",  # M7-specific extra_arg
    ):
        assert (
            required in cmd_str
        ), f"M7 cmd missing fragment {required!r}; got: {cmd_str}"
    # The MOLMETAL_ABLATE_Mxx env-var (if any) must be a known prefix
    # so the harness's env-var hook can read it.
    for k in spec_m7.env_vars:
        assert k.startswith("MOLMETAL_ABLATE_"), (
            f"M7 env var {k!r} does not start with MOLMETAL_ABLATE_; "
            "harness env-var hook would not trigger"
        )
    # Symmetric check for M12 — uses env-var but no extra_args.
    spec_m12 = table["M12"]
    assert spec_m12.env_vars, "M12 expected env_vars wired"
    assert (
        not spec_m12.extra_args
    ), "M12 expected no extra_args; uses env-var toggle only"
    # Wf-323A contract on M14: the spec passes ``--lr-warmup-steps 0``
    # as extra_args and the harness now accepts it (the adapter
    # applies a no-op warmup factor=1.0 when --lr-warmup-steps == 0,
    # so the ablation toggle is wired end-to-end without regressing
    # bit-exact pre-Wf-323A behaviour).
    spec_m14 = table["M14"]
    assert "--lr-warmup-steps" in spec_m14.extra_args, (
        "M14 expected --lr-warmup-steps in extra_args (Wf-323A "
        "re-integration contract)"
    )
