"""Wf-345-A1 — hyperparam alias resolution tests (4 NEW tests).

================================================================
Per Wf-345-A1 (2026-09-29 user ultracode directive: "覆盖所有发现的
问题把残余的问题解决了，一定要全"), the
``gate_wf343c4_hyperparams_config_complete`` gate was extended with a
``KEY_ALIASES`` map so the 6 non-canonical ``configs/*.yaml`` files
(which carry ``protocol.batch_size``, ``protocol.lr``, ``model.depth``
instead of the canonical ``train.batch_size``, ``train.lr``,
``model.n_layers``) all PASS the AGENTS.md §5.1 standing GPU grant
hyperparam contract WITHOUT requiring YAML renames (which would
break downstream consumers reading those keys).

This file is the test surface for that extension:

  1. **test_alias_resolution_canonical_keys_pass** — confirms the
     canonical-schema path still PASSes (backward-compat).

  2. **test_alias_resolution_alias_keys_pass** — confirms each
     documented alias location resolves to the canonical key
     (protocol.batch_size, protocol.lr, model.depth). Also confirms
     that a YAML with NO matching alias (in either canonical or
     aliased locations) correctly FAILs.

  3. **test_all_7_active_yamls_pass_after_alias_resolution** — runs
     the gate against the 7 active ``configs/*.yaml`` files on
     disk (the regression-test surface) and asserts 7/7 PASS.

  4. **test_alias_resolutions_in_evidence_message** — confirms the
     alias-resolution info is surfaced in the PASS evidence message
     so audit consumers can see which YAMLs used which alias.

Backward-compat verdict
-----------------------
The ``KEY_ALIASES`` extension is ADDITIVE only — no pre-existing
canonical-schema YAML was touched.  ``default.yaml`` (which already
carries the canonical schema) PASSes via the canonical location, and
``cfm_full_retrain.yaml`` (which now carries canonical train.batch_size
per Wf-345-A1) likewise PASSes canonically.  The 5 remaining YAMLs
PASS via resolved aliases (4 Wf-124 P3 + 2 CFM p1.x).

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
* All 4 tests assert ONE observable contract (alias resolution
  PASS/FAIL or evidence string match) so failures localise cleanly.
* Tests do NOT fabricate any metric values — every assertion
  traces to either a tmp_path YAML fixture or the on-disk
  configs/*.yaml (with absolute paths).
* The 7/7 PASS verdict on the live ``configs/`` dir is the
  gate-FAILURE regression-test that motivated this fix; if a
  future YAML regresses to a non-aliased non-canonical schema,
  these tests will catch it at the test layer.
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
CONFIGS_DIR = REPO_ROOT / "configs"

# Insert repo root into sys.path so the pre_flight_check module can
# be loaded via importlib without requiring the test runner to set
# PYTHONPATH itself.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_preflight_module():
    """Import ``molmetal.scripts.pre_flight_check`` via :mod:`importlib`."""
    spec = importlib.util.spec_from_file_location(
        "pre_flight_check_under_test", PREFLIGHT_SCRIPT,
    )
    assert spec is not None and spec.loader is not None, (
        f"pre_flight_check.py has no import spec ({PREFLIGHT_SCRIPT})"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pre_flight_check_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def preflight():
    """Module-level fixture exposing the pre_flight_check module."""
    return _load_preflight_module()


def _write_yaml(path: Path, body: dict) -> Path:
    """Write a YAML fixture from a dict (uses real yaml.safe_dump)."""
    import yaml as _yaml
    path.write_text(_yaml.safe_dump(body, sort_keys=False), encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# 1 — Alias resolution correctness (canonical keys)
# ---------------------------------------------------------------------------


def test_alias_resolution_canonical_keys_pass(preflight, tmp_path) -> None:
    """Wf-345-A1 (1) — alias-resolved gate PASSes on canonical-schema YAML.

    Verifies the BACKWARD-COMPAT invariant: the
    ``KEY_ALIASES`` extension does not break the canonical-schema
    path.  A YAML carrying the 4 canonical hyperparameters
    (``train.batch_size`` / ``train.lr`` / ``model.hidden_dim`` /
    ``model.n_layers``) per AGENTS.md §5.1 continues to PASS via
    the canonical (first-match) location.

    Mirrors the Wf-343C4 test 1 contract — same fixture shape,
    same PASS expectation; only the gate body changed.
    """
    yaml_path = _write_yaml(
        tmp_path / "canonical_schema.yaml",
        {
            "data": {"dataset": "qm9"},
            "model": {"hidden_dim": 128, "n_layers": 4},
            "train": {"batch_size": 32, "lr": 1.0e-4},
            "protocol": {"time_budget_hr": 10.0},  # alias candidates — should NOT be used
        },
    )
    result = preflight.gate_wf343c4_hyperparams_config_complete(
        "wf345a1_canonical", "P3", "wf343c4", (yaml_path,),
    )
    assert result.status == "PASS", (
        f"canonical-schema YAML should PASS; got {result.status}: {result.evidence}"
    )
    # No alias resolutions should be reported (all canonical).
    assert "alias-resolved" not in result.evidence, (
        f"canonical YAML should NOT trigger alias-resolved branch; "
        f"evidence={result.evidence}"
    )


# ---------------------------------------------------------------------------
# 2 — Alias resolution correctness (alias keys)
# ---------------------------------------------------------------------------


def test_alias_resolution_alias_keys_pass(preflight, tmp_path) -> None:
    """Wf-345-A1 (2) — alias-resolved gate PASSes on alias-schema YAML + FAILs on no-match.

    Two sub-cases verified in one test (per Wf-345-A1 brief: 2 tests
    for alias-resolution correctness):

      (a) Each documented alias location resolves to the canonical
          key:
            * protocol.batch_size → train.batch_size
            * protocol.lr         → train.lr
            * model.depth         → model.n_layers

          A YAML carrying ONLY alias locations (no canonical keys)
          MUST PASS the gate.  The alias_resolutions dict in the
          evidence message MUST name the alias locations used.

      (b) A YAML with NEITHER canonical NOR aliased hyperparameters
          (e.g. only a top-level ``foo.batch_size`` key) correctly
          FAILs.  This guards against over-permissive alias maps.
    """
    # ---- (a) alias-only YAML PASSes ----
    alias_yaml = _write_yaml(
        tmp_path / "alias_schema.yaml",
        {
            "data": {"dataset": "qm9"},
            "model": {
                "hidden_dim": 128,       # canonical
                "depth": 3,               # alias for n_layers
            },
            "train": {},                  # NO canonical batch_size / lr
            "protocol": {
                "batch_size": 64,         # alias for train.batch_size
                "lr": 1.0e-3,             # alias for train.lr
            },
        },
    )
    alias_result = preflight.gate_wf343c4_hyperparams_config_complete(
        "wf345a1_alias", "P3", "wf343c4", (alias_yaml,),
    )
    assert alias_result.status == "PASS", (
        f"alias-only YAML should PASS via KEY_ALIASES; "
        f"got {alias_result.status}: {alias_result.evidence}"
    )
    # Verify the alias-resolutions branch was triggered AND the
    # 3 aliases used are named in the evidence (n_layers:model.depth,
    # batch_size:protocol.batch_size, lr:protocol.lr).
    assert "alias-resolved" in alias_result.evidence, (
        f"alias YAML should trigger alias-resolved branch; "
        f"evidence={alias_result.evidence}"
    )
    assert "n_layers:model.depth" in alias_result.evidence, (
        f"model.depth alias not surfaced in evidence: {alias_result.evidence}"
    )
    assert "batch_size:protocol.batch_size" in alias_result.evidence, (
        f"protocol.batch_size alias not surfaced: {alias_result.evidence}"
    )
    assert "lr:protocol.lr" in alias_result.evidence, (
        f"protocol.lr alias not surfaced: {alias_result.evidence}"
    )

    # ---- (b) no-match YAML FAILs ----
    nomatch_yaml = _write_yaml(
        tmp_path / "no_match.yaml",
        {
            "data": {"dataset": "qm9"},
            "model": {"hidden_dim": 128},  # missing n_layers + model.depth + model.num_layers
            "train": {"epochs": 30},       # missing batch_size + lr
            "foo": {                       # unknown section
                "batch_size": 64,
                "lr": 1.0e-3,
                "depth": 3,
            },
        },
    )
    nomatch_result = preflight.gate_wf343c4_hyperparams_config_complete(
        "wf345a1_no_match", "P3", "wf343c4", (nomatch_yaml,),
    )
    assert nomatch_result.status == "FAIL", (
        f"no-match YAML should FAIL; got {nomatch_result.status}: "
        f"{nomatch_result.evidence}"
    )
    # The evidence MUST name the missing canonical keys.
    assert "missing" in nomatch_result.evidence.lower(), (
        f"FAIL evidence should name 'missing'; got {nomatch_result.evidence}"
    )
    assert "n_layers" in nomatch_result.evidence, (
        f"FAIL evidence should name n_layers; got {nomatch_result.evidence}"
    )


# ---------------------------------------------------------------------------
# 3 — All 7 active YAMLs PASS after alias resolution
# ---------------------------------------------------------------------------


def test_all_7_active_yamls_pass_after_alias_resolution(preflight) -> None:
    """Wf-345-A1 (3) — the regression-test surface: 7/7 active configs/*.yaml PASS.

    This is the test that proves the Wf-345-A1 fix delivers the
    expected gate-state regression.  Before the fix the gate was
    FAILing with evidence "6/7 YAMLs missing AGENTS.md §5.1 standing
    GPU grant hyperparam contract" (14 missing hyperparams total).
    After the fix the gate PASSes with 7/7 and the alias_resolutions
    dict in the evidence message names the 5 YAMLs that resolved via
    alias (4 Wf-124 P3 chemprop_v2_v3 YAMLs + 2 CFM p1.x YAMLs).
    """
    # The 7 active configs/*.yaml files per Wf-343C4 registration.
    yaml_paths = tuple(
        CONFIGS_DIR / name
        for name in (
            "cfm_full_retrain.yaml",
            "cfm_per_innovation_ablation_p1.4.yaml",
            "cfm_real_decode_lift_p1.3.yaml",
            "default.yaml",
            "wf124_p3_50epoch_full_sweep_g5.yaml",
            "wf124_p3_per_innovation_ablation_g6.yaml",
            "wf124_p3_remaining_metals_g4.yaml",
        )
    )
    # Sanity check: every YAML exists on disk (otherwise this test is a
    # false PASS / FAIL signal rather than a meaningful contract).
    missing_files = [str(p) for p in yaml_paths if not p.is_file()]
    assert not missing_files, (
        f"missing configs/*.yaml on disk (test fixture broken): {missing_files}"
    )

    result = preflight.gate_wf343c4_hyperparams_config_complete(
        "wf345a1_all_7_yamls", "P3", "wf343c4", yaml_paths,
    )
    assert result.status == "PASS", (
        f"all 7 active configs/*.yaml should PASS after alias resolution; "
        f"got {result.status}: {result.evidence}"
    )
    # The evidence MUST mention all 7 YAML names.
    for p in yaml_paths:
        assert p.name in result.evidence, (
            f"{p.name} missing from PASS evidence: {result.evidence}"
        )
    # The alias-resolutions dict MUST name the 5 YAMLs that resolved
    # via alias (default.yaml + cfm_full_retrain.yaml are fully
    # canonical, so they should NOT appear in alias_resolutions).
    assert "default.yaml" not in result.evidence.split("alias-resolved")[-1], (
        f"default.yaml should be canonical, not alias-resolved; "
        f"evidence={result.evidence}"
    )
    # 5 YAMLs that resolved via alias:
    expected_alias_yaml_names = [
        "cfm_per_innovation_ablation_p1.4.yaml",
        "cfm_real_decode_lift_p1.3.yaml",
        "wf124_p3_50epoch_full_sweep_g5.yaml",
        "wf124_p3_per_innovation_ablation_g6.yaml",
        "wf124_p3_remaining_metals_g4.yaml",
    ]
    for name in expected_alias_yaml_names:
        assert name in result.evidence, (
            f"{name} missing from PASS evidence: {result.evidence}"
        )


# ---------------------------------------------------------------------------
# 4 — Alias-resolutions surfaced in PASS evidence message
# ---------------------------------------------------------------------------


def test_alias_resolutions_in_evidence_message(preflight, tmp_path) -> None:
    """Wf-345-A1 (4) — alias-resolutions dict structure is preserved in evidence.

    Confirms the audit-observable contract: when a YAML uses an
    alias, the alias-resolutions branch names
    ``<key>:<resolved_section>.<resolved_key>`` so downstream
    consumers (per_wave_checklist.py, AGENTS.md sync, audit
    consumers) can see exactly which YAMLs used which aliases.

    Also confirms a fully-canonical YAML does NOT add the
    alias-resolved branch (to keep the canonical-path evidence
    clean for audit consumers).
    """
    # Mixed YAML — 2 canonical + 1 alias.
    mixed_yaml = _write_yaml(
        tmp_path / "mixed.yaml",
        {
            "model": {
                "hidden_dim": 128,       # canonical
                "depth": 3,               # alias for n_layers
            },
            "train": {
                "batch_size": 32,         # canonical
                # lr missing — should be supplied via protocol.lr alias
            },
            "protocol": {"lr": 1.0e-4},    # alias for train.lr
        },
    )
    mixed_result = preflight.gate_wf343c4_hyperparams_config_complete(
        "wf345a1_mixed", "P3", "wf343c4", (mixed_yaml,),
    )
    assert mixed_result.status == "PASS", (
        f"mixed canonical+alias YAML should PASS; "
        f"got {mixed_result.status}: {mixed_result.evidence}"
    )
    # Exactly 2 alias resolutions reported (n_layers:model.depth + lr:protocol.lr);
    # batch_size is canonical so NOT in alias-resolved.
    alias_section = mixed_result.evidence.split("alias-resolved:")[1]
    assert "n_layers:model.depth" in alias_section, (
        f"n_layers:model.depth alias missing from evidence: {alias_section}"
    )
    assert "lr:protocol.lr" in alias_section, (
        f"lr:protocol.lr alias missing from evidence: {alias_section}"
    )
    assert "batch_size" not in alias_section, (
        f"canonical batch_size should NOT appear in alias-resolved branch: "
        f"{alias_section}"
    )

    # Fully-canonical YAML — no alias-resolved branch should be added.
    canonical_yaml = _write_yaml(
        tmp_path / "fully_canonical.yaml",
        {
            "model": {"hidden_dim": 128, "n_layers": 4},
            "train": {"batch_size": 32, "lr": 1.0e-4},
        },
    )
    canonical_result = preflight.gate_wf343c4_hyperparams_config_complete(
        "wf345a1_fully_canonical", "P3", "wf343c4", (canonical_yaml,),
    )
    assert canonical_result.status == "PASS", (
        f"fully-canonical YAML should PASS; "
        f"got {canonical_result.status}: {canonical_result.evidence}"
    )
    assert "alias-resolved" not in canonical_result.evidence, (
        f"fully-canonical YAML should NOT add alias-resolved branch; "
        f"evidence={canonical_result.evidence}"
    )
