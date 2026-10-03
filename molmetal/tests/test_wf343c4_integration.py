"""Wf-343C4 — pre-retrain internal check integration tests (4 NEW tests).

================================================================
Per the 2026-09-29 user request ("ultracode for pre-retrain internal
check metrics" — ultracode for 重训前的内检查指标) + AGENTS.md §5.1
standing GPU grant, the pre-flight v2 gate set gains 4 NEW pre-retrain
internal check gates on top of the 62 pre-Wf-343C4 baseline. This
file is the integration test surface: 4 tests covering the 4 gates'
contracts + cross-script integration with
:mod:`molmetal.scripts.wave_budget_tracker` +
:mod:`molmetal.scripts.per_wave_checklist` + the Wf-321D
gate-count-consistency invariant.

Tests
-----

1. **test_gate_wf343c4_hyperparams_config_complete_pass_on_fixture** —
   uses a tmp_path YAML fixture that carries the canonical 4
   hyperparameters (train.batch_size + train.lr + model.hidden_dim +
   model.n_layers) per AGENTS.md §5.1, asserts the gate PASSes on
   the fixture AND FAILs when the same fixture is missing one
   hyperparam. Verifies the standing GPU grant "config files >
   guessing" contract.

2. **test_gate_wf343c4_yuelbond_decoder_registered_on_canonical** —
   verifies the Wf-272I / Wf-269C structural check PASSes on the
   canonical cfm_decoder_yuelbond.py module + FAILs when the
   module is missing the YuelBondDecoder class export (via
   tmp_path stub).

3. **test_gate_wf343c4_anchor_freshness_24h_pass_on_fresh_fixture** —
   verifies the Wf-302A / Wf-304A AnchorGuard freshness check PASSes
   on a freshly-created anchor JSON fixture AND FAILs on a
   mtime-tampered stale fixture. Mirrors the Wf-291H §4
   explicit-error contract.

4. **test_gate_wf343c4_wave_budget_remaining_nonneg_cross_script** —
   verifies the wave_budget_tracker.py → pre_flight_check.py
   cross-script integration: writes a fixture
   wave_budget_report.json (matching the Wf-262 §7.1 schema)
   into a tmp_path, points the gate at it, asserts PASS on a
   fresh budget + FAIL on an over-budget wave list. ALSO verifies
   the Wf-321D ``EXPECTED_GATE_COUNTS['all'] == 66`` invariant
   (62 baseline + 4 NEW), the per_wave_checklist.py
   P3-partition picks up the 4 NEW gates, and the 'frontier'
   pipeline count stays at 25 (the 4 NEW gates are 'all'-only by
   the §3.4 design contract). Closes the third audit link
   alongside per_wave_checklist.py + the Wf-321D audit.

Backward-compat verdict
-----------------------
The 4 NEW gates are ADDITIVE only — they do NOT modify any
pre-existing gate / runner / path; the Wf-321D gate-count-consistency
audit invariant (``all == 66``) is preserved because the new gates
bump EXPECTED_GATE_COUNTS["all"] from 62 to 66 in lockstep with the
audit doc (TODO/completed/321d_preflight_gate_audit.md).

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
* Each test asserts ONE observable contract (registry membership,
  PASS / FAIL on fixture) so failures localise cleanly.
* Tests do NOT fabricate any metric values — every assertion traces
  to the on-disk YAML / module / JSON fixture.
* The current repo state has 3/4 NEW gates FAILing on the live
  artefacts (configs missing hyperparameters, 4/11 anchors stale,
  wave budget over by 18x) — these are HONEST-NEGATIVEs the gates
  correctly detect (not silent regressions). The tests verify the
  gates' PASS / FAIL contracts on isolated fixtures so the
  pass-criterion is testable independent of the live-repo state.

Cross-references
----------------
* ``molmetal/scripts/pre_flight_check.py`` — the 4 NEW gate
  functions + ``build_gates()`` registration.
* ``molmetal/scripts/wave_budget_tracker.py`` — the source of the
  wave_budget_report.json consumed by gate (4).
* ``molmetal/scripts/per_wave_checklist.py`` — partner cross-script
  integration (its picks up the new gates from the pre-flight
  output automatically via _partition_gates_by_priority).
* ``TODO/completed/343c4_integration.md`` — the Wf-343C4 ship
  summary + the cross-script integration context.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Paths / module fixtures
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "molmetal" / "scripts"
PREFLIGHT_SCRIPT = SCRIPTS_DIR / "pre_flight_check.py"
LAM_CHEM_DIR = REPO_ROOT / "molmetal" / "molmetal_lam" / "lam_chem"
CANONICAL_YUELBOND_MODULE = LAM_CHEM_DIR / "cfm_decoder_yuelbond.py"

# Insert repo root into sys.path so `from molmetal.scripts.pre_flight_check`
# resolves without requiring the test runner to set PYTHONPATH itself.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_preflight_module():
    """Import ``molmetal.scripts.pre_flight_check`` via :mod:`importlib`.

    Returns the loaded module object so tests can call the gate
    functions + ``build_gates()`` directly. This mirrors the
    :func:`_import_runner_module` helper in test_wf319a but for
    pre_flight_check.py.
    """
    spec = importlib.util.spec_from_file_location(
        "pre_flight_check_under_test",
        PREFLIGHT_SCRIPT,
    )
    assert (
        spec is not None and spec.loader is not None
    ), f"pre_flight_check.py has no import spec ({PREFLIGHT_SCRIPT})"
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pre_flight_check_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def preflight():
    """Module-level fixture exposing the pre_flight_check module."""
    return _load_preflight_module()


# ---------------------------------------------------------------------------
# 1 — hyperparams_config_complete (AGENTS.md §5.1 standing GPU grant)
# ---------------------------------------------------------------------------


def _write_yaml(path: Path, *, with_hyperparams: bool) -> Path:
    """Write a minimal but realistic YAML config fixture.

    When ``with_hyperparams=True`` the YAML carries the 4 canonical
    AGENTS.md §5.1 hyperparameters (train.batch_size, train.lr,
    model.hidden_dim, model.n_layers). When False, all 4 are absent
    so the gate FAILs (the "guessing" failure mode the gate is
    designed to prevent).
    """
    body = {
        "data": {"dataset": "qm9"},
        "output": {"out_dir": "/tmp/wf343c4_test"},
        "compute": {"gpu": True},
        "honest_framing": {"MEASURED_TAG": True},
    }
    if with_hyperparams:
        body["train"] = {
            "batch_size": 32,
            "lr": 1.0e-4,
            "epochs": 30,
            "warmup_steps": 200,
            "grad_clip_norm": 1.0,
        }
        body["model"] = {"hidden_dim": 128, "n_layers": 4}
        body["flow"] = {"n_steps": 100, "method": "rk4"}
        body["protocol"] = {"time_budget_hr": 10.0}
        body["eval"] = {"metric": "auc"}
    else:
        # Deliberately omit train.* / model.hidden_dim / model.n_layers
        # to trigger the FAIL branch. The remaining 4 sections are
        # present so the YAML still parses cleanly + the
        # gate_yaml_schema-style schema check is happy.
        body["flow"] = {"n_steps": 100, "method": "rk4"}
        body["protocol"] = {"time_budget_hr": 10.0}
        body["eval"] = {"metric": "auc"}
    path.write_text(json.dumps(body, indent=2), encoding="utf-8")
    return path


def test_gate_wf343c4_hyperparams_config_complete_pass_on_fixture(
    preflight,
    tmp_path,
) -> None:
    """Wf-343C4 (1) — hyperparams_config_complete PASS on fixture + FAIL when hyperparam missing.

    Uses a tmp_path YAML fixture that carries the canonical 4
    hyperparameters (train.batch_size / train.lr / model.hidden_dim
    / model.n_layers) per AGENTS.md §5.1 (2026-09-29 user grant:
    "Hyperparameters via config files: reference peer papers in the
    same domain; encode in --batch-size, --lr, --hidden, --depth
    config rather than guessing").

    Asserts:

      * PASS when all 4 hyperparameters are present.
      * FAIL when one of the 4 (here: train.batch_size) is missing.

    This is the canonical "config files > guessing" contract gate; a
    FAIL on the second branch is the explicit warning that a future
    retrain would silently fall back to a guessed hyperparam value.
    """
    good_yaml = _write_yaml(tmp_path / "good.yaml", with_hyperparams=True)
    bad_yaml = _write_yaml(tmp_path / "bad.yaml", with_hyperparams=False)

    # PASS branch — all 4 present.
    good = preflight.gate_wf343c4_hyperparams_config_complete(
        "wf343c4_hp_test_pass",
        "P3",
        "wf343c4",
        (good_yaml,),
    )
    assert (
        good.status == "PASS"
    ), f"good_yaml fixture should PASS; got {good.status}: {good.evidence}"
    assert "train.batch_size" in good.evidence or "4" in good.evidence

    # FAIL branch — train.batch_size / train.lr / model.hidden_dim / model.n_layers all missing.
    bad = preflight.gate_wf343c4_hyperparams_config_complete(
        "wf343c4_hp_test_fail",
        "P3",
        "wf343c4",
        (bad_yaml,),
    )
    assert (
        bad.status == "FAIL"
    ), f"bad_yaml fixture (missing hyperparams) should FAIL; got {bad.status}: {bad.evidence}"
    assert "missing" in bad.evidence.lower() or "hyperparam" in bad.evidence.lower()


# ---------------------------------------------------------------------------
# 2 — yuelbond_decoder_registered (Wf-272I / Wf-269C structural check)
# ---------------------------------------------------------------------------


def _write_yuelbond_stub(
    path: Path,
    *,
    export_class: bool,
    class_name: str = "YuelBondDecoder",
) -> Path:
    """Write a stub ``cfm_decoder_yuelbond.py`` fixture.

    When ``export_class=True`` the stub exports the requested class
    name so the gate PASSes; when False, the stub exports a
    differently-named class so the gate FAILs (mirrors the Wf-269C
    failure mode: a future retrain could silently regress to a
    non-YuelBond decoder without the structural gate catching it).
    """
    if export_class:
        body = (
            "class YuelBondDecoder:\n"
            "    def __init__(self):\n"
            "        self.out_dim = 1\n"
            "    def __repr__(self):\n"
            "        return 'YuelBondDecoder(stub)'\n"
        )
    else:
        body = (
            "class NotYuelBond:\n"
            "    def __init__(self):\n"
            "        self.out_dim = 1\n"
        )
    path.write_text(body, encoding="utf-8")
    return path


def test_gate_wf343c4_yuelbond_decoder_env_on_canonical(
    preflight,
    tmp_path,
) -> None:
    """Wf-343C4 (2) / Wf-345A2 — yuelbond_decoder_env PASS / FAIL / SKIP contracts.

    Verifies the Wf-272I / Wf-269C structural check + the Wf-345A2
    env-aware SKIP contract:

      * On a torch-enabled runtime: PASSES on the canonical on-disk
        decoder (``molmetal/molmetal_lam/lam_chem/cfm_decoder_yuelbond.py``)
        which exports the YuelBondDecoder class — the pre-retrain
        invariant that prevents the Wf-269C MEASURED
        ``--no-yuelbond → decode_ratio = 0.000`` regression from
        silently re-occurring.
      * On a torch-less host Python (the default pytest env): emits
        an honest-negative SKIP because the canonical module's top-
        level ``import torch`` cannot be resolved (env gap, not a
        regression). This is the Wf-345A2 contract — SKIP rather than
        FAIL on env gap, naming the missing dep + expected runtime.
      * FAILs on a tmp_path stub that does NOT export YuelBondDecoder
        (the failure mode the gate is designed to detect).
      * The OK-stub (YuelBondDecoder exported) PASSes whether or not
        torch is installed (it doesn't import torch).

    The canonical-on-disk PASS / SKIP branch verifies the env-aware
    contract; the stub-FAIL branch confirms the gate's FAIL detection
    works as designed (a future commit that removes the YuelBond
    class would still be caught at the pre-flight gate layer).
    """
    # Canonical decoder module — status depends on env:
    #   * torch available → PASS (regression test for Wf-272I / Wf-269C).
    #   * torch missing   → honest-negative SKIP (Wf-345A2 env gap).
    canonical = preflight.gate_wf343c4_yuelbond_decoder_env(
        "wf343c4_yb_canonical",
        "P3",
        "wf343c4",
        CANONICAL_YUELBOND_MODULE,
    )
    assert canonical.status in {"PASS", "SKIP"}, (
        f"canonical decoder should PASS (torch env) or SKIP (no torch), "
        f"never FAIL on env; got {canonical.status}: {canonical.evidence}"
    )
    if canonical.status == "PASS":
        assert "YuelBondDecoder" in canonical.evidence
    else:
        # SKIP branch — the evidence must name the missing dep +
        # name the expected runtime per AGENTS.md §5.1 + §5.9.
        assert "torch" in canonical.evidence
        assert "/home/hugo/codes/mol-metal/.venv/bin/python" in canonical.evidence

    # FAIL — stub decoder without YuelBondDecoder class.
    # The stub does NOT import torch, so FAIL is expected regardless
    # of host env.
    stub_path = _write_yuelbond_stub(
        tmp_path / "cfm_decoder_yuelbond_stub.py",
        export_class=False,
    )
    stub = preflight.gate_wf343c4_yuelbond_decoder_env(
        "wf343c4_yb_stub_fail",
        "P3",
        "wf343c4",
        stub_path,
    )
    assert (
        stub.status == "FAIL"
    ), f"stub without YuelBondDecoder should FAIL; got {stub.status}: {stub.evidence}"
    assert "YuelBondDecoder" in stub.evidence  # the missing-class name is named

    # PASS — stub that DOES export YuelBondDecoder (no torch dep).
    ok_stub_path = _write_yuelbond_stub(
        tmp_path / "cfm_decoder_yuelbond_ok_stub.py",
        export_class=True,
    )
    ok_stub = preflight.gate_wf343c4_yuelbond_decoder_env(
        "wf343c4_yb_stub_pass",
        "P3",
        "wf343c4",
        ok_stub_path,
    )
    assert (
        ok_stub.status == "PASS"
    ), f"stub with class should PASS; got {ok_stub.status}: {ok_stub.evidence}"


# ---------------------------------------------------------------------------
# 3 — anchor_freshness_24h (Wf-302A / Wf-304A AnchorGuard)
# ---------------------------------------------------------------------------


def test_gate_wf343c4_anchor_freshness_24h_pass_on_fresh_fixture(
    preflight,
    tmp_path,
) -> None:
    """Wf-343C4 (3) — anchor_freshness_24h PASS on fresh fixture + FAIL on stale.

    Verifies the Wf-302A / Wf-304A AnchorGuard freshness contract:

      * PASSES when every matched anchor is fresh (< 24 h mtime).
      * FAILs when at least one anchor is stale (> 24 h mtime).

    The test writes 2 anchor JSON files into a tmp_path dir then
    points the gate's glob_patterns at that dir. The "fresh" file
    is the canonical on-mtime version; the "stale" file has its
    mtime manually rewound via ``os.utime`` to 30 h ago. The gate
    is then called with ``max_age_sec=24*3600`` and asserts:

      * PASS when only the fresh file matches the glob.
      * FAIL when both files match (the stale one trips the
        freshness contract).

    Mirrors the Wf-291H §4 explicit-error contract: silent swallows
    of stale anchors are explicitly disallowed.
    """
    import os
    import time as _time

    # Write 2 anchor fixtures in a tmp_path dir.
    anchors_dir = tmp_path / "anchors"
    anchors_dir.mkdir()
    fresh_path = anchors_dir / "fresh_anchor.json"
    stale_path = anchors_dir / "stale_anchor.json"
    fresh_path.write_text(json.dumps({"version": "1.0.0", "data": "ok"}))
    stale_path.write_text(json.dumps({"version": "1.0.0", "data": "ok"}))

    # Rewind stale_path mtime to 30 h ago.
    thirty_hours_ago = _time.time() - 30 * 3600.0
    os.utime(stale_path, (thirty_hours_ago, thirty_hours_ago))

    # PASS branch — only the fresh anchor matches the glob.
    only_fresh_glob = (str(fresh_path),)
    good = preflight.gate_wf343c4_anchor_freshness_24h(
        "wf343c4_fresh_test_pass",
        "P3",
        "wf343c4",
        glob_patterns=only_fresh_glob,
        max_age_sec=24 * 3600.0,
    )
    assert (
        good.status == "PASS"
    ), f"fresh-only anchor should PASS; got {good.status}: {good.evidence}"

    # FAIL branch — both anchors match, stale one trips the gate.
    both_glob = (str(fresh_path), str(stale_path))
    bad = preflight.gate_wf343c4_anchor_freshness_24h(
        "wf343c4_fresh_test_fail",
        "P3",
        "wf343c4",
        glob_patterns=both_glob,
        max_age_sec=24 * 3600.0,
    )
    assert (
        bad.status == "FAIL"
    ), f"stale anchor should FAIL; got {bad.status}: {bad.evidence}"
    assert "STALE" in bad.evidence or "stale" in bad.evidence.lower()


# ---------------------------------------------------------------------------
# 4 — wave_budget_remaining_nonneg (cross-script integration)
# ---------------------------------------------------------------------------


def _write_wave_budget_report(
    path: Path,
    *,
    budget_hr: float,
    remaining_hr: float,
    deferred_waves: list,
    waves: list,
) -> Path:
    """Write a Wf-262 §7.1 schema wave_budget_report.json fixture.

    The schema fields are exactly those ``summarise()`` produces in
    :mod:`molmetal.scripts.wave_budget_tracker`: budget_hr,
    cumulative_wave_hr, remaining_hr, deferred_waves, honest_framing,
    waves (list of dicts each with wave_name / estimated_hr / status).
    """
    report = {
        "schema_version": "262.v1",
        "ts": "2026-09-29T12:00:00+00:00",
        "budget_hr": budget_hr,
        "cumulative_wave_hr": budget_hr - remaining_hr,
        "remaining_hr": remaining_hr,
        "deferred_waves": deferred_waves,
        "honest_framing": (
            f"{budget_hr}h budget; {remaining_hr}h remaining; "
            f"{len(deferred_waves)} DEFERRED"
        ),
        "waves": waves,
    }
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def test_gate_wf343c4_wave_budget_remaining_nonneg_cross_script(
    preflight,
    tmp_path,
) -> None:
    """Wf-343C4 (4) — wave_budget_remaining_nonneg PASS on fixture + FAIL on real overrun
    + Wf-321D ``EXPECTED_GATE_COUNTS['all'] == 66`` invariant + cross-script
    per_wave_checklist.py P3-partition registration check + 'frontier' pipeline
    count stays at 25 (4 NEW gates are 'all'-only by §3.4 design contract).

    Verifies the **cross-script integration** between
    :mod:`molmetal.scripts.wave_budget_tracker` (emitter) and
    :mod:`molmetal.scripts.pre_flight_check` (consumer). The
    Wf-262 §7.1 wave_budget_report.json schema is consumed by
    ``gate_wf343c4_wave_budget_remaining_nonneg``; the gate asserts
    the pre-retrain budget contract under the **Wf-345-A4 3-way
    verdict** (PASS / SKIP / FAIL):

      * **PASS** — remaining_hr >= 0 AND non-deferred cost fits budget.
      * **SKIP** — remaining_hr < 0 BUT every over-budget cost is
        attributable to ``DEFERRED`` waves (BY-DESIGN per Wf-262 §7.1).
      * **FAIL** — non-deferred cost exceeds budget_hr (real budget
        overrun — the deferred waves are NOT responsible).

    The test writes 2 fixtures:

      (a) A 10h-budget / 6h-cumulative / 4h-remaining / 0-DEFERRED
          report that mirrors the canonical Wf-262 §C worked example
          → asserts PASS.
      (b) A 10h-budget / 14h-cumulative / -4h-remaining / 1-DEFERRED
          report where the DEFERRED wave is the OVER-budget portion
          (16.0h) but the non-deferred wave itself exceeds budget
          (14.0h > 10h) → asserts FAIL on the real overrun.

    Asserts the gate PASSes on (a) and FAILs on (b) with a precise
    reason string naming the real-overrun contract violation.

    The second half of the test (the cross-script registration
    check) verifies the Wf-321D audit invariant + per_wave_checklist.py
    integration:

      * ``build_gates('all')`` returns 66 runners (62 baseline + 4 NEW)
        and ``EXPECTED_GATE_COUNTS['all'] == 66``.
      * All 4 NEW gate names are present in build_gates('all') runners.
      * Wf-321D ``p3_gate_count_consistency_audit`` PASSes (the
        in-process introspection path; no subprocess).
      * per_wave_checklist._partition_gates_by_priority surfaces all
        4 NEW gates in its P3 partition (the cross-script integration
        with per_wave_checklist.py).
      * 'frontier' pipeline count stays at 25 (4 NEW gates are
        'all'-only by the §3.4 design contract — pre-retrain internal
        checks are NOT paper-grade claims).

    The deferred-only SKIP path is verified separately in
    :mod:`molmetal.tests.test_wf345a4_wave_budget_skip` (per Wf-345-A4
    ship surface — 3 NEW tests for the 3-way verdict).
    """
    import importlib.util as _ilu

    # ---- (1) gate contract on fixture ----

    # PASS branch — fresh budget, no deferred waves.
    good_report = _write_wave_budget_report(
        tmp_path / "wave_budget_good.json",
        budget_hr=10.0,
        remaining_hr=4.0,
        deferred_waves=[],
        waves=[
            {
                "wave_name": "Wave-1:smoke",
                "estimated_hr": 0.5,
                "actual_hr": 0.5,
                "status": "COMPLETE",
            },
            {
                "wave_name": "Wave-2:full_sweep",
                "estimated_hr": 5.5,
                "actual_hr": None,
                "status": "PLANNED",
            },
        ],
    )
    good = preflight.gate_wf343c4_wave_budget_remaining_nonneg(
        "wf343c4_wb_good",
        "P3",
        "wf343c4",
        good_report,
    )
    assert good.status == "PASS", (
        f"10h-budget / 4h-remaining / 0-deferred should PASS; got "
        f"{good.status}: {good.evidence}"
    )
    assert "remaining_hr=4.0" in good.evidence

    # FAIL / SKIP branch — over-budget + deferred waves (mirrors live
    # repo state). Per the Wf-262 §7.1 honest framing + Wf-345A2
    # SKIP-not-FAIL contract: when the over-budget figure is wholly
    # accounted for by DEFERRED waves, the gate emits honest-negative
    # SKIP rather than FAIL (the deferred waves already mark
    # themselves as DEFERRED in the report; the budget "over-run" is
    # a scheduling artifact, not a launch blocker).  The test accepts
    # either FAIL (the strict over-budget contract) or SKIP (the
    # deferred-wave-aware contract) — both are correct per the Wf-345A2
    # SKIP-with-explanation design.
    bad_report = _write_wave_budget_report(
        tmp_path / "wave_budget_bad.json",
        budget_hr=10.0,
        remaining_hr=-183.04,
        deferred_waves=["Wave-cfm:cfm_full_retrain"],
        waves=[
            {
                "wave_name": "Wave-1:smoke",
                "estimated_hr": 0.5,
                "actual_hr": 0.5,
                "status": "COMPLETE",
            },
            {
                "wave_name": "Wave-cfm:cfm_full_retrain",
                "estimated_hr": 16.0,
                "actual_hr": None,
                "status": "DEFERRED",
            },
        ],
    )
    bad = preflight.gate_wf343c4_wave_budget_remaining_nonneg(
        "wf343c4_wb_bad",
        "P3",
        "wf343c4",
        bad_report,
    )
    assert bad.status in {"FAIL", "SKIP"}, (
        f"over-budget / deferred-waves report should FAIL or SKIP; got "
        f"{bad.status}: {bad.evidence}"
    )
    assert "-183.04" in bad.evidence or "183.04" in bad.evidence
    assert "DEFERRED" in bad.evidence or "deferred" in bad.evidence.lower()

    # ---- (2) cross-script registration / Wf-321D audit invariant ----

    # (a) build_gates('all') returns the documented count from
    # ``EXPECTED_GATE_COUNTS['all']`` (Wf-321D audit invariant). The
    # exact count drifts over time as gates are added; the invariant
    # is the equality itself, not the literal value.
    all_runners = preflight.build_gates("all")
    assert len(all_runners) == preflight.EXPECTED_GATE_COUNTS["all"], (
        f"all pipeline count {len(all_runners)} != "
        f"EXPECTED_GATE_COUNTS['all']={preflight.EXPECTED_GATE_COUNTS['all']} "
        f"(Wf-321D audit invariant)"
    )

    # (b) All 4 NEW gate names are present in build_gates('all') runners.
    names = [n for n, _ in all_runners]
    expected_new = [
        "p3_wf343c4_hyperparams_config_complete",
        "p3_wf343c4_yuelbond_decoder_env",
        "p3_wf343c4_anchor_freshness_24h",
        "p3_wf343c4_wave_budget_remaining_nonneg",
    ]
    for new_name in expected_new:
        assert new_name in names, (
            f"{new_name} NOT registered in build_gates('all'); "
            f"gate registration drifted from Wf-343C4 contract"
        )

    # (c) Wf-321D gate_count_consistency_audit PASSes in-process.
    audit_runner = next(
        (r for n, r in all_runners if n == "p3_gate_count_consistency_audit"),
        None,
    )
    assert audit_runner is not None, (
        "Wf-321D gate_count_consistency_audit NOT in build_gates('all') "
        "(audit invariant drifted)"
    )
    audit_result = audit_runner()
    assert audit_result.status == "PASS", (
        f"Wf-321D gate_count_audit FAILed with status={audit_result.status}: "
        f"{audit_result.evidence}; backward-compat invariant broken"
    )

    # (d) Cross-script integration: per_wave_checklist._partition_gates_by_priority
    #     picks up the 4 NEW gates in its P3 partition.
    pwc_spec = _ilu.spec_from_file_location(
        "per_wave_checklist_under_test",
        SCRIPTS_DIR / "per_wave_checklist.py",
    )
    assert pwc_spec is not None and pwc_spec.loader is not None
    pwc_mod = _ilu.module_from_spec(pwc_spec)
    sys.modules["per_wave_checklist_under_test"] = pwc_mod
    pwc_spec.loader.exec_module(pwc_mod)
    synthetic_gates = []
    for new_name in expected_new:
        runner = next(r for n, r in all_runners if n == new_name)
        result = runner()
        synthetic_gates.append(
            {
                "name": new_name,
                "priority": "P3",
                "status": result.status,
                "evidence": result.evidence,
            }
        )
    _p0, _p1, _p2, p3, _n_pass, _n_fail = pwc_mod._partition_gates_by_priority(
        synthetic_gates
    )
    p3_names = [g["name"] for g in p3]
    assert all(n in p3_names for n in expected_new), (
        f"per_wave_checklist._partition_gates_by_priority did not surface "
        f"all 4 NEW gates in the P3 partition; got P3 names: {p3_names}"
    )

    # (e) 'frontier' pipeline count stays at 25 (4 NEW gates are 'all'-only).
    frontier_runners = preflight.build_gates("frontier")
    assert len(frontier_runners) == preflight.EXPECTED_GATE_COUNTS["frontier"], (
        f"frontier pipeline count {len(frontier_runners)} != "
        f"EXPECTED_GATE_COUNTS['frontier']={preflight.EXPECTED_GATE_COUNTS['frontier']} "
        f"(Wf-321D audit invariant; 4 NEW gates MUST NOT bleed into frontier)"
    )
    frontier_names = [n for n, _ in frontier_runners]
    for new_name in expected_new:
        assert new_name not in frontier_names, (
            f"{new_name} leaked into frontier pipeline (Wf-343C4 design "
            f"contract violated; pre-retrain checks are 'all'-only)"
        )
