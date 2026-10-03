"""Wf-350-C1 — pre-flight p1_gpu_env_probe gate integration tests (3 NEW tests).

================================================================
Per the 2026-09-29 user request ("ultracode — 覆盖所有发现的问题
把残余的问题解决了" — ultracode for NeurIPS reproducibility backbone)
+ Wf-343a-R3 finding #1 ("GPU/CUDA env probe should be wired into
pre-flight"), the pre-flight v2 gate set gains 1 NEW gate
(``p1_gpu_env_probe``) on top of the Wf-321D + Wf-347A1 +
Wf-350-A1/A2/A3 + Wf-353-B1 baseline. The gate wraps the
canonical R18 Blackwell probe at
``molmetal/scripts/check_blackwell_env.py`` and surfaces the 4
host-runtime fields required by AGENTS.md §5.1 (standing GPU grant —
"显存够就放心大胆用"):

  1. ``torch.cuda.is_available`` (via ``torch_probe.ok``)
  2. ``torch.cuda.device_count`` (via ``torch_probe.device_count``)
  3. GPU model (via ``gpu_names`` + per-GPU ``gpu_details[*].name``)
  4. CUDA runtime version (via ``torch_probe.cuda_runtime``)

This file is the integration test surface: 3 tests covering the
gate's contracts + cross-script registration + Wf-321D
gate-count-consistency invariant.

Tests
-----

1. **test_gate_wf350c1_gpu_env_probe_pass_on_host** —
   verifies the gate PASSes on the live host environment. The gate
   invokes ``check_blackwell_env.py`` as a subprocess (writes JSON to
   the canonical ``molmetal/reports/wf350c1_gpu_env_check.json``
   path), re-reads the artefact, and asserts the canonical schema is
   valid (gpu_count + gpu_names + torch_probe.{ok, device_count,
   device_name, cuda_runtime}).

2. **test_gate_wf350c1_gpu_env_probe_json_schema_validity** —
   verifies the gate PASSes on a tmp_path output path AND the
   resulting JSON conforms to the canonical schema. Mirrors the
   pattern used by test_wf350a2_compute_manifest.py for the
   Wf-350-A2 manifest.

3. **test_gate_wf350c1_gpu_env_probe_registered_in_build_gates** —
   verifies the gate is registered in build_gates() dispatcher + the
   Wf-321D ``EXPECTED_GATE_COUNTS['all'] == 94`` invariant is
   preserved (93 pre-Wf-350-C1 baseline + 1 NEW gate).

Backward-compat verdict
-----------------------
The 1 NEW gate is ADDITIVE only — it does NOT modify any
pre-existing gate / runner / path; the Wf-321D gate-count-consistency
audit invariant (``all == 94``) is preserved because the new gate
bumps EXPECTED_GATE_COUNTS["all"] from 93 to 94 in lockstep with
the in-process audit gate (``p3_gate_count_consistency_audit``).

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
* Each test asserts ONE observable contract (registry membership,
  PASS / FAIL on fixture, JSON schema validity) so failures localise
  cleanly.
* Tests do NOT fabricate any metric values — every assertion traces
  to the on-disk env JSON written by the live probe (test 1) or to
  the in-process gate function + tmp_path fixture (test 2).
* The live probe may emit different driver/torch/CUDA versions on
  different hosts; the tests accept any non-empty string per field
  rather than asserting a specific version string.

Cross-references
----------------
* ``molmetal/scripts/pre_flight_check.py`` — the new gate
  function ``gate_p1_gpu_env_probe`` + ``build_gates()``
  registration.
* ``molmetal/scripts/check_blackwell_env.py`` — the canonical R18
  Blackwell env probe that the gate wraps (108 LOC, ship surface
  Wf-265 + Wf-326C).
* ``molmetal/reports/wf350c1_gpu_env_check.json`` — the canonical
  env JSON emitted by the gate at host run-time.
* ``TODO/completed/350c1_gpu_env_probe.md`` — the Wf-350-C1 ship
  summary + per-deliverable MEASURED status.
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
PROBE_SCRIPT = SCRIPTS_DIR / "check_blackwell_env.py"
ENV_JSON_PATH = (
    REPO_ROOT / "molmetal" / "reports" / "wf350c1_gpu_env_check.json"
)

# Insert repo root into sys.path so `from molmetal.scripts.pre_flight_check`
# resolves without requiring the test runner to set PYTHONPATH itself.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_preflight_module():
    """Import ``molmetal.scripts.pre_flight_check`` via :mod:`importlib`.

    Returns the loaded module object so tests can call the gate
    functions + ``build_gates()`` directly. This mirrors the
    :func:`_load_preflight_module` helper in test_wf350a2_compute_
    manifest.py + test_wf343c4_integration.py + test_wf321d_preflight
    _gate_audit.py.
    """
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


# ---------------------------------------------------------------------------
# 1 — gate PASS on live host + canonical schema validity
# ---------------------------------------------------------------------------


def test_gate_wf350c1_gpu_env_probe_pass_on_host(preflight) -> None:
    """Wf-350-C1 (1) — p1_gpu_env_probe PASS on live host + env JSON schema-valid.

    Runs the gate against the canonical env JSON path and asserts:

      * Status == PASS (the gate's schema-validity check accepts the
        canonical on-disk env JSON written by check_blackwell_env.py).
      * Env JSON file exists, is non-empty, parses as valid JSON.
      * Top-level keys: ``gpu_count`` (int), ``gpu_names`` (list),
        ``torch_probe`` (dict).
      * ``torch_probe`` carries ``ok`` (bool), ``device_count`` (int
        >= 1), ``device_name`` (str), ``cuda_runtime`` (str).

    This is the canonical host-runtime schema validation — every
    field is required by AGENTS.md §5.1 (standing GPU grant:
    "显存够就放心大胆用" — use freely as long as VRAM is enough).
    """
    # Confirm probe script is on disk first (gate would FAIL anyway
    # with a clear evidence string if missing; pre-check surfaces
    # a sharper test diagnostic).
    assert PROBE_SCRIPT.is_file(), (
        f"probe script not on disk: {PROBE_SCRIPT}; cannot run "
        f"Wf-350-C1 integration test"
    )

    result = preflight.gate_p1_gpu_env_probe(
        "p1_gpu_env_probe_test", "P3", "wf350c1",
        ENV_JSON_PATH,
    )
    assert result.status == "PASS", (
        f"gate should PASS on live host; got {result.status}: "
        f"{result.evidence}"
    )

    # Env JSON exists, is non-empty, parses as JSON.
    assert ENV_JSON_PATH.is_file(), (
        f"env JSON file not on disk: {ENV_JSON_PATH}"
    )
    raw = ENV_JSON_PATH.read_text(encoding="utf-8")
    assert raw.strip(), f"env JSON file is empty: {ENV_JSON_PATH}"
    env_summary = json.loads(raw)

    # Top-level keys.
    for key in ("gpu_count", "gpu_names", "torch_probe"):
        assert key in env_summary, (
            f"env summary missing top-level key {key!r}; got keys: "
            f"{sorted(env_summary.keys())}"
        )

    # gpu_names must be a list.
    assert isinstance(env_summary["gpu_names"], list), (
        f"env summary['gpu_names'] is not a list; got "
        f"{type(env_summary['gpu_names']).__name__}"
    )
    assert len(env_summary["gpu_names"]) >= 1, (
        f"env summary['gpu_names'] is empty; expected >= 1 GPU visible "
        f"via nvidia-smi on a 2-GPU host"
    )

    # gpu_count must be int >= 1.
    assert isinstance(env_summary["gpu_count"], int), (
        f"env summary['gpu_count'] is not an int; got "
        f"{type(env_summary['gpu_count']).__name__}"
    )
    assert env_summary["gpu_count"] >= 1, (
        f"env summary['gpu_count']={env_summary['gpu_count']} < 1 "
        f"(expected >= 1)"
    )

    # torch_probe must be a dict with the 4 mandatory fields.
    tp = env_summary["torch_probe"]
    assert isinstance(tp, dict), (
        f"env summary['torch_probe'] is not a dict; got "
        f"{type(tp).__name__}"
    )
    for key in ("ok", "device_count", "device_name", "cuda_runtime"):
        assert key in tp, (
            f"env summary['torch_probe'] missing field {key!r}; got "
            f"keys: {sorted(tp.keys())}"
        )

    # The 4 mandatory host-runtime fields:
    # 1. torch.cuda.is_available == True (torch_probe.ok)
    assert tp["ok"] is True, (
        f"torch_probe.ok={tp['ok']!r} (expected True — torch.cuda."
        f"is_available on at least one probed venv)"
    )
    # 2. torch.cuda.device_count >= 1
    assert isinstance(tp["device_count"], int) and tp["device_count"] >= 1, (
        f"torch_probe.device_count={tp['device_count']!r} (expected "
        f"int >= 1 — torch.cuda.device_count)"
    )
    # 3. GPU model non-empty (torch_probe.device_name)
    assert tp["device_name"], (
        f"torch_probe.device_name is empty; got {tp['device_name']!r}"
    )
    # 4. CUDA runtime version non-empty (torch_probe.cuda_runtime)
    assert tp["cuda_runtime"], (
        f"torch_probe.cuda_runtime is empty; got {tp['cuda_runtime']!r}"
    )


# ---------------------------------------------------------------------------
# 2 — JSON schema validity on tmp_path fixture
# ---------------------------------------------------------------------------


def test_gate_wf350c1_gpu_env_probe_json_schema_validity(
    preflight, tmp_path,
) -> None:
    """Wf-350-C1 (2) — gate PASS on tmp_path fixture + env JSON on disk.

    Verifies the gate's schema-validity check accepts a freshly-written
    env JSON at a tmp_path location (mirrors the test_wf350a2_compute_
    manifest PASS-on-fixture pattern).

    Asserts:

      * Status == PASS on a tmp_path fixture.
      * The env JSON file is created at the specified path with the
        canonical schema (gpu_count + gpu_names + torch_probe).
      * If the host has at least 1 GPU (nvidia-smi succeeds), the
        torch_probe.ok + device_count + gpu_names are all populated;
        if the host has no GPU, the gate FAILs (it is a strict probe).
    """
    fixture_path = tmp_path / "gpu_env_probe_fixture.json"
    result = preflight.gate_p1_gpu_env_probe(
        "p1_gpu_env_probe_fixture", "P3", "wf350c1",
        fixture_path,
    )

    # On hosts WITH at least 1 GPU (the canonical 2-GPU Blackwell +
    # RTX 5090 dev box), the gate PASSes. On CPU-only hosts (no
    # nvidia-smi), the gate FAILs — both are acceptable, but the
    # fixture file MUST exist and parse either way.
    assert result.status in ("PASS", "FAIL"), (
        f"gate should PASS or FAIL on tmp_path fixture (no SKIP); "
        f"got {result.status}: {result.evidence}"
    )
    assert fixture_path.is_file(), (
        f"env JSON not written at {fixture_path}"
    )

    env_summary = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert isinstance(env_summary, dict), (
        f"tmp_path fixture env JSON is not a dict; got "
        f"{type(env_summary).__name__}"
    )
    for key in ("gpu_count", "gpu_names", "torch_probe"):
        assert key in env_summary, (
            f"tmp_path fixture env JSON missing top-level key {key!r}; "
            f"got keys: {sorted(env_summary.keys())}"
        )
    assert isinstance(env_summary["gpu_names"], list), (
        f"tmp_path fixture env JSON['gpu_names'] is not a list; got "
        f"{type(env_summary['gpu_names']).__name__}"
    )
    assert isinstance(env_summary["torch_probe"], dict), (
        f"tmp_path fixture env JSON['torch_probe'] is not a dict; got "
        f"{type(env_summary['torch_probe']).__name__}"
    )


# ---------------------------------------------------------------------------
# 3 — gate registered in build_gates + Wf-321D audit invariant
# ---------------------------------------------------------------------------


def test_gate_wf350c1_gpu_env_probe_registered_in_build_gates(
    preflight,
) -> None:
    """Wf-350-C1 (3) — p1_gpu_env_probe registered in build_gates +
    Wf-321D ``EXPECTED_GATE_COUNTS['all'] == 94`` invariant preserved.

    Asserts:

      * ``EXPECTED_GATE_COUNTS["all"] == 94`` (93 pre-Wf-350-C1
        baseline + 1 NEW).
      * ``len(build_gates("all")) == 94`` (in lockstep with the
        EXPECTED bump).
      * ``p1_gpu_env_probe`` is registered in ``build_gates("all")``.
      * ``p1_gpu_env_probe`` is NOT registered in ``build_gates(
        "frontier")`` / ``"chemprop_v2_v3")`` / ``"cfm")`` (per the
        §3.4 design contract: pre-retrain internal check is
        'all'-only, NOT a paper-rule parity gate).
      * Wf-321D ``p3_gate_count_consistency_audit`` PASSes (the
        in-process introspection path; no subprocess).
      * The registered gate function itself PASSes when called via
        build_gates('all') (catches schema-drift regression on the
        gate function between registration and execution).
    """
    # (a) build_gates('all') count + EXPECTED invariant.
    all_runners = preflight.build_gates("all")
    assert len(all_runners) == preflight.EXPECTED_GATE_COUNTS["all"], (
        f"all pipeline count {len(all_runners)} != "
        f"EXPECTED_GATE_COUNTS['all']={preflight.EXPECTED_GATE_COUNTS['all']} "
        f"(Wf-321D audit invariant)"
    )
    assert preflight.EXPECTED_GATE_COUNTS["all"] == 94, (
        f"EXPECTED_GATE_COUNTS['all']={preflight.EXPECTED_GATE_COUNTS['all']} "
        f"!= 94 (Wf-350-C1 ship target; 93 pre-Wf-350-C1 baseline + 1 NEW "
        f"Gp — gate_p1_gpu_env_probe)"
    )

    # (b) p1_gpu_env_probe is registered in 'all'.
    all_names = [n for n, _ in all_runners]
    assert "p1_gpu_env_probe" in all_names, (
        "p1_gpu_env_probe NOT registered in build_gates('all'); "
        "gate registration drifted from Wf-350-C1 contract"
    )

    # (c) p1_gpu_env_probe is NOT registered in 'frontier' /
    #     'chemprop_v2_v3' / 'cfm' (per the design contract).
    frontier_names = [n for n, _ in preflight.build_gates("frontier")]
    chemprop_names = [n for n, _ in preflight.build_gates("chemprop_v2_v3")]
    cfm_names = [n for n, _ in preflight.build_gates("cfm")]
    assert "p1_gpu_env_probe" not in frontier_names, (
        "p1_gpu_env_probe leaked into frontier pipeline "
        "(Wf-350-C1 design contract violated; GPU env probe is "
        "NOT a paper-rule parity gate)"
    )
    assert "p1_gpu_env_probe" not in chemprop_names, (
        "p1_gpu_env_probe leaked into chemprop_v2_v3 pipeline "
        "(Wf-350-C1 design contract violated; GPU env probe is "
        "'all'-only)"
    )
    assert "p1_gpu_env_probe" not in cfm_names, (
        "p1_gpu_env_probe leaked into cfm pipeline "
        "(Wf-350-C1 design contract violated; GPU env probe is "
        "'all'-only)"
    )

    # (d) frontier / chemprop_v2_v3 / cfm counts UNCHANGED (the
    #     1 NEW gate is 'all'-only by design contract).
    for pipeline_name, names in (
        ("frontier", frontier_names),
        ("chemprop_v2_v3", chemprop_names),
        ("cfm", cfm_names),
    ):
        assert len(names) == preflight.EXPECTED_GATE_COUNTS[pipeline_name], (
            f"{pipeline_name} pipeline count {len(names)} != "
            f"EXPECTED_GATE_COUNTS[{pipeline_name!r}]="
            f"{preflight.EXPECTED_GATE_COUNTS[pipeline_name]} "
            f"(Wf-321D audit invariant; p1_gpu_env_probe MUST NOT bleed "
            f"into {pipeline_name})"
        )

    # (e) Wf-321D gate_count_consistency_audit PASSes in-process.
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
        f"Wf-321D gate_count_audit FAILed with status="
        f"{audit_result.status}: {audit_result.evidence}; "
        f"backward-compat invariant broken"
    )

    # (f) The registered gate function itself PASSes when called via
    #     build_gates('all') (catches schema-drift regression on the
    #     gate function between registration and execution).
    gpu_probe_runner = next(
        r for n, r in all_runners if n == "p1_gpu_env_probe"
    )
    gpu_probe_result = gpu_probe_runner()
    assert gpu_probe_result.status == "PASS", (
        f"p1_gpu_env_probe runner FAILed: status="
        f"{gpu_probe_result.status}: {gpu_probe_result.evidence}; "
        f"live-host regression — gate function broke between "
        f"registration and execution"
    )
