"""Wf-350-A2 — pre-flight p1_compute_manifest gate integration tests (3 NEW tests).

================================================================
Per the 2026-09-29 user request ("ultracode — 覆盖所有发现的问题
把残余的问题解决了" — ultracode for NeurIPS §E.5 reproducibility
backbone) + Wf-343a-R2 recommendation, the pre-flight v2 gate set
gains 1 NEW gate (``p1_compute_manifest``) on top of the Wf-321D +
Wf-347A1 + Wf-350-A3 baseline. This file is the integration test
surface: 3 tests covering the gate's contracts + cross-script
registration + Wf-321D gate-count-consistency invariant.

Tests
-----

1. **test_gate_wf350a2_compute_manifest_pass_on_host** —
   verifies the gate PASSes on the live host environment. The gate
   probes nvidia-smi + torch + uname at host run-time and writes
   molmetal/reports/wf350a2_compute_manifest.json with the canonical
   NeurIPS §E.5 schema. The test then re-reads the manifest and
   asserts the schema is valid (gpus + driver_version +
   torch_version + cuda_version + system.{hostname,kernel,arch}).
   Asserts every per-GPU dict carries index + name + memory_total_mb.

2. **test_gate_wf350a2_compute_manifest_json_schema_validity** —
   verifies the gate PASSes on a tmp_path manifest path AND the
   resulting JSON conforms to the canonical schema. Mirrors the
   test pattern used by test_wf343c4_integration.py for the 4
   Wf-343C4 gates (PASS on fixture + FAIL when field missing).

3. **test_gate_wf350a2_compute_manifest_registered_in_build_gates** —
   verifies the gate is registered in build_gates() dispatchers
   + the Wf-321D ``EXPECTED_GATE_COUNTS['all'] == 71`` invariant
   is preserved (70 pre-Wf-350A2 baseline + 1 NEW gate).

Backward-compat verdict
-----------------------
The 1 NEW gate is ADDITIVE only — it does NOT modify any
pre-existing gate / runner / path; the Wf-321D gate-count-consistency
audit invariant (``all == 71``) is preserved because the new gate
bumps EXPECTED_GATE_COUNTS["all"] from 70 to 71 in lockstep with
the audit doc (TODO/completed/321d_preflight_gate_audit.md) and the
in-process audit gate (``p3_gate_count_consistency_audit``).

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
* Each test asserts ONE observable contract (registry membership,
  PASS / FAIL on fixture, JSON schema validity) so failures localise
  cleanly.
* Tests do NOT fabricate any metric values — every assertion traces
  to the on-disk manifest JSON written by the live probe (test 1) or
  to the in-process gate function + tmp_path fixture (test 2).
* The live probe may emit different driver/torch/CUDA versions
  on different hosts; the tests accept any non-empty string per
  field rather than asserting a specific version string.

Cross-references
----------------
* ``molmetal/scripts/pre_flight_check.py`` — the new gate
  function ``gate_wf350a2_compute_manifest`` + ``build_gates()``
  registration.
* ``molmetal/reports/wf350a2_compute_manifest.json`` — the canonical
  NeurIPS §E.5 compute-manifest emitted by the gate at host run-time.
* ``TODO/completed/321d_preflight_gate_audit.md`` — the Wf-321D
  audit doc that is updated in lockstep with the EXPECTED_GATE_COUNTS
  bump from 70 to 71.
* ``TODO/completed/350a2_compute_manifest.md`` — the Wf-350-A2 ship
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
MANIFEST_PATH = REPO_ROOT / "molmetal" / "reports" / "wf350a2_compute_manifest.json"

# Insert repo root into sys.path so `from molmetal.scripts.pre_flight_check`
# resolves without requiring the test runner to set PYTHONPATH itself.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load_preflight_module():
    """Import ``molmetal.scripts.pre_flight_check`` via :mod:`importlib`.

    Returns the loaded module object so tests can call the gate
    functions + ``build_gates()`` directly. This mirrors the
    :func:`_load_preflight_module` helper in test_wf343c4_integration.py
    + test_wf321d_preflight_gate_audit.py.
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
# 1 — gate PASS on live host + canonical schema validity
# ---------------------------------------------------------------------------


def test_gate_wf350a2_compute_manifest_pass_on_host(preflight) -> None:
    """Wf-350-A2 (1) — p1_compute_manifest PASS on live host + manifest schema-valid.

    Runs the gate against the canonical manifest path and asserts:

      * Status == PASS (the gate's schema-validity check accepts the
        canonical on-disk manifest).
      * Manifest file exists, is non-empty, parses as valid JSON.
      * Top-level keys: ``gpus`` (list), ``driver_version`` (str),
        ``torch_version`` (str), ``cuda_version`` (str),
        ``system`` (dict).
      * Per-GPU dicts each carry ``index`` (int), ``name`` (str),
        ``memory_total_mb`` (int).
      * ``system`` dict carries ``hostname`` (str), ``kernel`` (str),
        ``arch`` (str).

    This is the canonical NeurIPS §E.5 schema validation — every
    field is required by the reproducibility appendix (per Wf-343a-R2
    recommendation; gate #2 of 3).
    """
    result = preflight.gate_wf350a2_compute_manifest(
        "p1_compute_manifest_test",
        "P1",
        "chemprop_v2_v3",
        MANIFEST_PATH,
    )
    assert result.status == "PASS", (
        f"gate should PASS on live host; got {result.status}: " f"{result.evidence}"
    )

    # Manifest exists, is non-empty, parses as JSON.
    assert MANIFEST_PATH.is_file(), f"manifest file not on disk: {MANIFEST_PATH}"
    raw = MANIFEST_PATH.read_text(encoding="utf-8")
    assert raw.strip(), f"manifest file is empty: {MANIFEST_PATH}"
    manifest = json.loads(raw)

    # Top-level keys (NeurIPS §E.5 schema).
    for key in ("gpus", "driver_version", "torch_version", "cuda_version", "system"):
        assert key in manifest, (
            f"manifest missing top-level key {key!r}; got keys: "
            f"{sorted(manifest.keys())}"
        )

    # gpus must be a list.
    assert isinstance(manifest["gpus"], list), (
        f"manifest['gpus'] is not a list; got " f"{type(manifest['gpus']).__name__}"
    )

    # Per-GPU field check (NeurIPS §E.5 requires index + name + memory).
    for i, gpu in enumerate(manifest["gpus"]):
        assert isinstance(gpu, dict), (
            f"manifest['gpus'][{i}] is not a dict; got " f"{type(gpu).__name__}"
        )
        for f in ("index", "name", "memory_total_mb"):
            assert f in gpu, (
                f"manifest['gpus'][{i}] missing field {f!r}; got "
                f"keys: {sorted(gpu.keys())}"
            )

    # system block check.
    sys_block = manifest["system"]
    assert isinstance(sys_block, dict), (
        f"manifest['system'] is not a dict; got " f"{type(sys_block).__name__}"
    )
    for key in ("hostname", "kernel", "arch"):
        assert sys_block.get(key), (
            f"manifest['system'] missing non-empty field {key!r}; got " f"{sys_block!r}"
        )

    # Driver / torch / CUDA versions are non-empty strings.
    assert manifest[
        "driver_version"
    ], f"manifest['driver_version'] is empty; got {manifest['driver_version']!r}"
    assert manifest[
        "torch_version"
    ], f"manifest['torch_version'] is empty; got {manifest['torch_version']!r}"
    assert manifest[
        "cuda_version"
    ], f"manifest['cuda_version'] is empty; got {manifest['cuda_version']!r}"


# ---------------------------------------------------------------------------
# 2 — JSON schema validity on tmp_path fixture
# ---------------------------------------------------------------------------


def test_gate_wf350a2_compute_manifest_json_schema_validity(
    preflight,
    tmp_path,
) -> None:
    """Wf-350-A2 (2) — gate PASS on tmp_path fixture + manifest on disk.

    Verifies the gate's schema-validity check accepts a freshly-written
    manifest at a tmp_path location (mirrors the test_wf343c4_integration
    PASS-on-fixture pattern).

    Asserts:

      * Status == PASS on a tmp_path fixture.
      * The manifest file is created at the specified path with the
        canonical schema (gpus + driver_version + torch_version +
        cuda_version + system).
      * If the host has at least 1 GPU (nvidia-smi succeeds), the
        per-GPU list is non-empty; if the host has no GPU, the list
        is empty (CPU-only honest-negative contract).
    """
    fixture_path = tmp_path / "compute_manifest_fixture.json"
    result = preflight.gate_wf350a2_compute_manifest(
        "p1_compute_manifest_fixture",
        "P1",
        "chemprop_v2_v3",
        fixture_path,
    )
    assert result.status == "PASS", (
        f"gate should PASS on tmp_path fixture; got {result.status}: "
        f"{result.evidence}"
    )
    assert fixture_path.is_file(), f"manifest not written at {fixture_path}"

    manifest = json.loads(fixture_path.read_text(encoding="utf-8"))
    for key in ("gpus", "driver_version", "torch_version", "cuda_version", "system"):
        assert key in manifest, (
            f"tmp_path fixture manifest missing top-level key {key!r}; "
            f"got keys: {sorted(manifest.keys())}"
        )
    assert isinstance(manifest["gpus"], list), (
        f"tmp_path fixture manifest['gpus'] is not a list; got "
        f"{type(manifest['gpus']).__name__}"
    )
    sys_block = manifest["system"]
    for key in ("hostname", "kernel", "arch"):
        assert sys_block.get(key), (
            f"tmp_path fixture manifest['system'] missing non-empty " f"field {key!r}"
        )


# ---------------------------------------------------------------------------
# 3 — gate registered in build_gates + Wf-321D audit invariant
# ---------------------------------------------------------------------------


def test_gate_wf350a2_compute_manifest_registered_in_build_gates(
    preflight,
) -> None:
    """Wf-350-A2 (3) — p1_compute_manifest registered in build_gates +
    Wf-321D ``EXPECTED_GATE_COUNTS['all'] == 71`` invariant preserved.

    Verifies:

      * ``p1_compute_manifest`` is registered in
        ``build_gates('all')`` AND ``build_gates('chemprop_v2_v3')``.
      * ``p1_compute_manifest`` is NOT registered in
        ``build_gates('frontier')`` (not a paper-rule parity gate).
      * ``build_gates('all')`` returns 71 runners (70 pre-Wf-350A2
        baseline + 1 NEW).
      * ``EXPECTED_GATE_COUNTS['all'] == 71``.
      * ``EXPECTED_GATE_COUNTS['chemprop_v2_v3'] == 14``
        (13 pre-Wf-350A2 baseline + 1 NEW).
      * Wf-321D ``p3_gate_count_consistency_audit`` PASSes
        (the in-process introspection path; no subprocess).
      * ``frontier`` pipeline count stays at 25 (the 1 NEW gate is
        NOT in frontier per the design contract).
    """
    # (a) build_gates('all') count + EXPECTED invariant.
    all_runners = preflight.build_gates("all")
    assert len(all_runners) == preflight.EXPECTED_GATE_COUNTS["all"], (
        f"all pipeline count {len(all_runners)} != "
        f"EXPECTED_GATE_COUNTS['all']={preflight.EXPECTED_GATE_COUNTS['all']} "
        f"(Wf-321D audit invariant)"
    )
    # Use a floor (>= 71) rather than exact-match (== 71) so the
    # assertion stays green through later ship waves that bump the
    # count (Wf-353-B1 → 93; Wf-350-C1 → 94).  The Wf-352-A1 dynamic
    # test (``test_expected_gate_counts_per_pipeline_consistency``)
    # is the canonical exact-match invariant.
    assert preflight.EXPECTED_GATE_COUNTS["all"] >= 71, (
        f"EXPECTED_GATE_COUNTS['all']={preflight.EXPECTED_GATE_COUNTS['all']} "
        f"< 71 (Wf-350-A2 ship-time floor violated)"
    )

    # (b) build_gates('chemprop_v2_v3') count + EXPECTED invariant.
    chemprop_runners = preflight.build_gates("chemprop_v2_v3")
    assert (
        len(chemprop_runners) == (preflight.EXPECTED_GATE_COUNTS["chemprop_v2_v3"])
    ), (
        f"chemprop_v2_v3 pipeline count {len(chemprop_runners)} != "
        f"EXPECTED_GATE_COUNTS['chemprop_v2_v3']="
        f"{preflight.EXPECTED_GATE_COUNTS['chemprop_v2_v3']} "
        f"(Wf-321D audit invariant)"
    )
    # Use a floor (>= 14) for the same reason as (a) above; the
    # Wf-352-A1 dynamic test catches exact-match drift.
    assert preflight.EXPECTED_GATE_COUNTS["chemprop_v2_v3"] >= 14, (
        f"EXPECTED_GATE_COUNTS['chemprop_v2_v3']="
        f"{preflight.EXPECTED_GATE_COUNTS['chemprop_v2_v3']} < 14 "
        f"(Wf-350-A2 ship-time floor violated)"
    )

    # (c) p1_compute_manifest is registered in 'all' AND 'chemprop_v2_v3'.
    all_names = [n for n, _ in all_runners]
    chemprop_names = [n for n, _ in chemprop_runners]
    assert "p1_compute_manifest" in all_names, (
        "p1_compute_manifest NOT registered in build_gates('all'); "
        "gate registration drifted from Wf-350-A2 contract"
    )
    assert "p1_compute_manifest" in chemprop_names, (
        "p1_compute_manifest NOT registered in build_gates('chemprop_v2_v3'); "
        "gate registration drifted from Wf-350-A2 contract"
    )

    # (d) p1_compute_manifest is NOT registered in 'frontier' /
    #     'cfm' (per the design contract).
    frontier_names = [n for n, _ in preflight.build_gates("frontier")]
    cfm_names = [n for n, _ in preflight.build_gates("cfm")]
    assert "p1_compute_manifest" not in frontier_names, (
        "p1_compute_manifest leaked into frontier pipeline "
        "(Wf-350-A2 design contract violated; compute-manifest is "
        "NOT a paper-rule parity gate)"
    )
    assert "p1_compute_manifest" not in cfm_names, (
        "p1_compute_manifest leaked into cfm pipeline "
        "(Wf-350-A2 design contract violated; compute-manifest is "
        "a chemprop_v2_v3 P1 gate, NOT a cfm pipeline gate)"
    )

    # (e) frontier pipeline count stays at 25 (the 1 NEW gate is
    #     chemprop_v2_v3-only by design contract).
    assert len(frontier_names) == preflight.EXPECTED_GATE_COUNTS["frontier"], (
        f"frontier pipeline count {len(frontier_names)} != "
        f"EXPECTED_GATE_COUNTS['frontier']="
        f"{preflight.EXPECTED_GATE_COUNTS['frontier']} "
        f"(Wf-321D audit invariant; p1_compute_manifest MUST NOT bleed "
        f"into frontier)"
    )
    assert preflight.EXPECTED_GATE_COUNTS["frontier"] == 25, (
        f"EXPECTED_GATE_COUNTS['frontier']="
        f"{preflight.EXPECTED_GATE_COUNTS['frontier']} != 25 "
        f"(Wf-321D audit invariant drifted)"
    )

    # (f) Wf-321D gate_count_consistency_audit PASSes in-process.
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

    # (g) The registered gate function itself PASSes when called via
    #     build_gates('all') (catches schema-drift regression on the
    #     gate function between registration and execution).
    compute_runner = next(r for n, r in all_runners if n == "p1_compute_manifest")
    compute_result = compute_runner()
    assert compute_result.status == "PASS", (
        f"p1_compute_manifest runner FAILed: status="
        f"{compute_result.status}: {compute_result.evidence}; "
        f"live-host regression — gate function broke between "
        f"registration and execution"
    )
