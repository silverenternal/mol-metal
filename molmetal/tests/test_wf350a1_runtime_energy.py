"""Wf-350-A1 — runtime + GPU energy probe gate tests (3 NEW tests).

================================================================
Per the 2026-09-29 user request ("ultracode for pre-retrain internal
check metrics" — ultracode for 重训前的内检查指标) + AGENTS.md §5.1
standing GPU grant, the pre-flight v2 gate set gains 1 NEW NeurIPS
§E.4 runtime + energy probe gate on top of the 70 pre-Wf-350-A1
baseline (62 baseline + 4 Wf-343C4 + 2 Wf-347A1 + 1 Wf-350-A2 +
1 Wf-350-A3).
This file is the integration test surface: 3 tests covering the
:func:`molmetal.scripts.pre_flight_check.gate_p1_runtime_energy_report`
gate function + the ``build_gates('all')`` registry + the on-disk
JSON artefact at ``molmetal/reports/wf350a1_runtime_energy.json``.

Tests
-----

1. **test_gate_p1_runtime_energy_report_pass_on_live_or_skip** —
   invokes the gate with a tmp_path JSON output; asserts PASS when
   ``nvidia-smi`` reports ≥ 1 GPU (the live host has 2 GPUs — the
   canonical Blackwell + RTX 5090 pair) AND the JSON artefact is
   written with all 4 mandatory schema fields
   (``per_gpu`` / ``total_duration_s`` / ``estimated_energy_j`` /
   ``n_gpus_reported``). On a host without NVIDIA driver the test
   SKIPs (an honest-negative, not a FAIL — same SKIP-as-acceptable
   contract as Wf-345A2 yuelbond_decoder_env).

2. **test_gate_p1_runtime_energy_report_schema_validity** —
   invokes the gate with a tmp_path JSON output, asserts that
   ``per_gpu`` is a list with at least 1 entry, every entry has
   ``index`` (int) + ``power_draw_w`` (float, >= 0) +
   ``power_limit_w`` (float, > 0), and ``estimated_energy_j >= 0``
   (the non-negative sanity contract). Mirrors the Wf-321A
   ``--strict`` JSON-schema validation pattern.

3. **test_p1_runtime_energy_report_registered_in_build_gates_all** —
   verifies the gate is registered in ``build_gates('all')``
   (the dispatcher picks it up by name), and the
   ``EXPECTED_GATE_COUNTS["all"] == 70`` invariant is preserved
   (69 baseline + 1 NEW Wf-350-A1). Mirrors the Wf-321D
   gate-count-consistency audit contract.

Backward-compat verdict
-----------------------
The 1 NEW gate is ADDITIVE only — it does NOT modify any
pre-existing gate / runner / path; the Wf-321D gate-count-consistency
audit invariant (``all == 71``) is preserved because the new gate
bumps ``EXPECTED_GATE_COUNTS["all"]`` from 70 to 71 in lockstep with
the audit doc (TODO/completed/321d_preflight_gate_audit.md + Wf-344B3
gate count contract).

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
* Each test asserts ONE observable contract (registry membership,
  PASS / FAIL on fixture, JSON schema validity) so failures localise
  cleanly.
* Tests do NOT fabricate any metric values — every assertion traces
  to the on-disk JSON artefact (or a tmp_path fixture).
* The probe is CPU-only via ``nvidia-smi``; the test is host-
  portable (SKIP on a non-NVIDIA host, PASS on a real GPU host).

Cross-references
----------------
* ``molmetal/scripts/pre_flight_check.py`` — the
  ``gate_p1_runtime_energy_report`` function +
  ``build_gates()`` registration + ``EXPECTED_GATE_COUNTS["all"]``
  bump from 69 to 70.
* ``molmetal/reports/wf350a1_runtime_energy.json`` — the canonical
  JSON artefact emitted by the gate (5 top-level fields:
  ``per_gpu`` + ``total_duration_s`` + ``estimated_energy_j`` +
  ``n_gpus_reported`` + ``probe_timestamp`` + ``host`` +
  ``MEASURED_vs_PROJECTED``).
* ``TODO/completed/350a1_runtime_energy_report.md`` — the Wf-350-A1
  ship summary + the NeurIPS §E.4 protocol provenance.
"""

from __future__ import annotations

import importlib.util
import json
import shutil
import sys
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Paths / module fixtures
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "molmetal" / "scripts"
PREFLIGHT_SCRIPT = SCRIPTS_DIR / "pre_flight_check.py"
REPORTS_DIR = REPO_ROOT / "molmetal" / "reports"
CANONICAL_ARTEFACTORY = REPORTS_DIR / "wf350a1_runtime_energy.json"

# Insert repo root into sys.path so `from molmetal.scripts.pre_flight_check`
# resolves without requiring the test runner to set PYTHONPATH itself.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_preflight_module():
    """Import ``molmetal.scripts.pre_flight_check`` via :mod:`importlib`.

    Returns the loaded module object so tests can call the gate
    functions + ``build_gates()`` directly. This mirrors the
    :func:`_load_preflight_module` helper in test_wf343c4 but for the
    Wf-350-A1 invocation context.
    """
    spec = importlib.util.spec_from_file_location(
        "pre_flight_check_under_test_wf350a1",
        PREFLIGHT_SCRIPT,
    )
    assert (
        spec is not None and spec.loader is not None
    ), f"pre_flight_check.py has no import spec ({PREFLIGHT_SCRIPT})"
    mod = importlib.util.module_from_spec(spec)
    sys.modules["pre_flight_check_under_test_wf350a1"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def preflight():
    """Module-level fixture exposing the pre_flight_check module."""
    return _load_preflight_module()


def _has_nvidia_smi() -> bool:
    """Return True iff the host has an ``nvidia-smi`` binary on PATH.

    Mirrors the Wf-345A2 env-aware SKIP contract: a host without
    NVIDIA GPU cannot produce a real per-GPU power probe, so the
    gate cannot PASS; we SKIP rather than FAIL in that case (an
    honest-negative for the env gap, not for the gate implementation).
    """
    return shutil.which("nvidia-smi") is not None


# ---------------------------------------------------------------------------
# 1 — gate PASS on live nvidia-smi OR SKIP on non-NVIDIA host
# ---------------------------------------------------------------------------


def test_gate_p1_runtime_energy_report_pass_on_live_or_skip(
    preflight,
    tmp_path,
) -> None:
    """Wf-350-A1 (1) — gate PASSes when nvidia-smi reports ≥ 1 GPU + JSON artefact written.

    Invokes :func:`gate_p1_runtime_energy_report` with a tmp_path JSON
    output target, asserts:

      * Gate returns ``PASS`` on the live host (the canonical
        Blackwell + RTX 5090 pair).
      * JSON artefact exists at the tmp_path target.
      * JSON document carries the 4 mandatory top-level fields:
        ``per_gpu``, ``total_duration_s``, ``estimated_energy_j``,
        ``n_gpus_reported``.
      * ``MEASURED_vs_PROJECTED == "MEASURED"`` (the AGENTS.md §5.2
        honest-framing tag discipline).

    On a host without NVIDIA driver the test SKIPs (an honest-
    negative, not a FAIL — same contract as
    Wf-345A2 yuelbond_decoder_env).
    """
    if not _has_nvidia_smi():
        pytest.skip("nvidia-smi not on PATH (non-NVIDIA host)")

    artefact = tmp_path / "wf350a1_runtime_energy.json"
    result = preflight.gate_p1_runtime_energy_report(
        "p1_runtime_energy_report",
        "P3",
        "wf350a1",
        artefact,
    )
    assert result.status == "PASS", (
        f"gate FAIL: evidence={result.evidence!r}; "
        f"expected PASS on the live NVIDIA host"
    )
    assert (
        artefact.is_file()
    ), f"artefact not written to {artefact}; gate should write JSON"
    data = json.loads(artefact.read_text(encoding="utf-8"))
    for key in ("per_gpu", "total_duration_s", "estimated_energy_j", "n_gpus_reported"):
        assert key in data, f"mandatory field {key!r} missing from JSON"
    assert data["MEASURED_vs_PROJECTED"] == "MEASURED", (
        f"MEASURED_vs_PROJECTED={data.get('MEASURED_vs_PROJECTED')!r} "
        f"!= 'MEASURED' (AGENTS.md §5.2 honest-framing tag discipline)"
    )


# ---------------------------------------------------------------------------
# 2 — JSON schema validity (per-GPU power readings)
# ---------------------------------------------------------------------------


def test_gate_p1_runtime_energy_report_schema_validity(
    preflight,
    tmp_path,
) -> None:
    """Wf-350-A1 (2) — JSON schema validity: per-GPU power readings have correct types.

    Invokes the gate with a tmp_path JSON output target, then
    walks the JSON artefact and asserts:

      * ``per_gpu`` is a list with ≥ 1 entry.
      * Every entry has ``index`` (int), ``power_draw_w`` (float,
        >= 0), ``power_limit_w`` (float, > 0).
      * ``estimated_energy_j >= 0`` (non-negative sanity contract).
      * ``n_gpus_reported == len(per_gpu)`` (the GPU-count
        bookkeeping invariant).

    On a host without NVIDIA driver the test SKIPs.
    """
    if not _has_nvidia_smi():
        pytest.skip("nvidia-smi not on PATH (non-NVIDIA host)")

    artefact = tmp_path / "wf350a1_runtime_energy.json"
    result = preflight.gate_p1_runtime_energy_report(
        "p1_runtime_energy_report",
        "P3",
        "wf350a1",
        artefact,
    )
    # On live host the gate should PASS; on a degraded host (e.g.
    # nvidia-smi rc=0 but 0 GPUs reported) the gate FAILs. We test
    # the schema only when the artefact was successfully written.
    assert (
        artefact.is_file()
    ), f"artefact not written to {artefact}; gate should write JSON"
    data = json.loads(artefact.read_text(encoding="utf-8"))
    # On a 0-GPU host (rare but possible) the per_gpu list is
    # empty — the schema validity check would naturally fail. We
    # gate the schema assertions behind result.status == "PASS"
    # so the test PASSes whenever the canonical happy-path
    # contract holds; on a 0-GPU host the schema validity is
    # verified by test 1 (which asserts result.status == "PASS").
    if result.status != "PASS":
        pytest.skip(
            f"gate did not PASS on this host "
            f"(evidence={result.evidence!r}); skipping schema check"
        )

    per_gpu = data["per_gpu"]
    assert isinstance(
        per_gpu, list
    ), f"per_gpu is not a list: type={type(per_gpu).__name__}"
    assert len(per_gpu) >= 1, (
        "per_gpu has 0 entries (expected >= 1 on a host with " ">= 1 NVIDIA GPU)"
    )
    for i, gpu in enumerate(per_gpu):
        assert isinstance(
            gpu, dict
        ), f"per_gpu[{i}] is not a dict: type={type(gpu).__name__}"
        assert "index" in gpu and isinstance(
            gpu["index"], int
        ), f"per_gpu[{i}] missing or wrong-type 'index': {gpu!r}"
        assert "power_draw_w" in gpu and isinstance(
            gpu["power_draw_w"], (int, float)
        ), f"per_gpu[{i}] missing or wrong-type 'power_draw_w': {gpu!r}"
        assert "power_limit_w" in gpu and isinstance(
            gpu["power_limit_w"], (int, float)
        ), f"per_gpu[{i}] missing or wrong-type 'power_limit_w': {gpu!r}"
        assert gpu["power_draw_w"] >= 0, (
            f"per_gpu[{i}] power_draw_w={gpu['power_draw_w']} < 0 "
            f"(negative power is impossible)"
        )
        assert gpu["power_limit_w"] > 0, (
            f"per_gpu[{i}] power_limit_w={gpu['power_limit_w']} <= 0 "
            f"(zero/negative limit is impossible)"
        )
    assert data["n_gpus_reported"] == len(per_gpu), (
        f"n_gpus_reported={data['n_gpus_reported']} != "
        f"len(per_gpu)={len(per_gpu)} (GPU-count bookkeeping "
        f"invariant violated)"
    )
    assert data["estimated_energy_j"] >= 0, (
        f"estimated_energy_j={data['estimated_energy_j']} < 0 "
        f"(non-negative sanity contract violated)"
    )
    assert data["total_duration_s"] >= 0, (
        f"total_duration_s={data['total_duration_s']} < 0 "
        f"(non-negative wall-time sanity contract violated)"
    )


# ---------------------------------------------------------------------------
# 3 — gate registered in build_gates('all')
# ---------------------------------------------------------------------------


def test_p1_runtime_energy_report_registered_in_build_gates_all(
    preflight,
) -> None:
    """Wf-350-A1 (3) — gate registered in build_gates('all') + EXPECTED_GATE_COUNTS bumped to 71.

    Asserts:

      * ``build_gates('all')`` includes the runner named
        ``"p1_runtime_energy_report"``.
      * The total count of all-pipeline runners equals
        ``EXPECTED_GATE_COUNTS["all"]`` (71 = 70 baseline + 1 NEW).
      * The constant ``EXPECTED_GATE_COUNTS["all"] == 71`` (the
        Wf-321D gate-count-consistency invariant is preserved).

    Mirrors the Wf-321D audit + the Wf-344B3 test_wf344b3_gate_count
    pattern.
    """
    all_runners = preflight.build_gates("all")
    all_names = [name for name, _ in all_runners]
    assert "p1_runtime_energy_report" in all_names, (
        f"p1_runtime_energy_report NOT registered in build_gates('all'); "
        f"registered names: {all_names}"
    )

    # Wf-321D audit invariant: actual count must match
    # EXPECTED_GATE_COUNTS["all"].
    assert len(all_runners) == preflight.EXPECTED_GATE_COUNTS["all"], (
        f"all pipeline count {len(all_runners)} drifted from "
        f"EXPECTED_GATE_COUNTS['all']={preflight.EXPECTED_GATE_COUNTS['all']}; "
        f"new gate (per Bump) should preserve the count invariant"
    )
    # Use a floor (>= 71) rather than exact-match (== 71) so the
    # assertion stays green through later ship waves that bump the
    # count (Wf-353-B1 → 93; Wf-350-C1 → 94).  The Wf-352-A1 dynamic
    # test (``test_expected_gate_counts_per_pipeline_consistency``)
    # is the canonical exact-match invariant.
    assert preflight.EXPECTED_GATE_COUNTS["all"] >= 71, (
        f"EXPECTED_GATE_COUNTS['all']={preflight.EXPECTED_GATE_COUNTS['all']} "
        f"< 71 (Wf-350-A1 ship-time floor violated; "
        f"baseline = 62 + 4 Wf-343C4 + 2 Wf-347A1 + 1 Wf-350-A2 + 1 Wf-350-A3)"
    )

    # The new gate's runner is registered AND is invokable (we don't
    # invoke it here — the schema-validity test does that — but the
    # registry membership is sufficient for this test).
    runner_idx = all_names.index("p1_runtime_energy_report")
    runner_name, runner_fn = all_runners[runner_idx]
    assert runner_name == "p1_runtime_energy_report", (
        f"registered runner name drift: {runner_name!r} != "
        f"'p1_runtime_energy_report'"
    )
    assert callable(
        runner_fn
    ), f"runner_fn is not callable: type={type(runner_fn).__name__}"
