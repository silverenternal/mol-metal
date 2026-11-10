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

   [Wf-A170, 2026-11-01 — the ``== 94`` rendering above is the
   Wf-350-C1 ship-time text and is PRESERVED VERBATIM per AGENTS.md
   §5.5 rule 6 (append-only + dated, never a silent edit, never a
   deletion).  The LIVE count is 129; the executable pin in the test
   body has been re-synced to 129, dated, with the superseded 94 kept
   visible in the dated comment and in the assertion message.]

================================================================
Wf-A170 (2026-11-01) — THE TWO-SIDED SESSION CONTRACT
================================================================
MEASURED DEFECT (CPU-only session, ``CUDA_VISIBLE_DEVICES=""``):
all 3 tests in this file were RED, and NONE of the 3 failures had
anything to do with what they guard:

  * test 1 failed on ``assert result.status == "PASS"`` — the gate
    returns **SKIP**, not PASS.
  * test 2 failed on ``assert result.status in ("PASS", "FAIL")`` —
    the gate returns **SKIP**, which the assertion did not allow.
  * test 3 failed on the gate-count PIN (``== 94`` vs the live 129).

The gate's SKIP branch (pre_flight_check.py, Wf-A128, 2026-10-31)
fires when — and ONLY when — the session hides the GPUs from torch
(``CUDA_VISIBLE_DEVICES`` present and blank: the repo's standing CPU
lock, applied at IMPORT time by BOTH ``/home/hugo/codes/mol-metal/
conftest.py`` and ``molmetal/tests/conftest.py``).  In such a session
the probe's torch half CANNOT see a GPU by construction, so the gate
SKIPs and reports the host facts that WERE measured (nvidia-smi is
unaffected by the lock).  A SKIP is NOT a FAIL and NOT a false PASS.

THE REPAIR (this file) keeps the assertion honest in BOTH sessions:

  * **GPU visible** (``CUDA_VISIBLE_DEVICES`` non-empty AND torch sees
    a GPU): the ORIGINAL assertions hold unchanged — the gate PASSes,
    the artefact exists and parses, the registration invariants hold.
  * **GPU masked** (``CUDA_VISIBLE_DEVICES`` blank — the CPU-locked
    session): the gate MUST return **SKIP** (asserting a PASS here
    would be the false green this wave exists to prevent), the SKIP
    evidence MUST be non-empty and MUST name BOTH the session fact
    and the measured host facts, and everything that IS checkable on
    CPU still passes (the artefact exists and parses, the schema
    holds, the registration invariants hold, the Wf-321D count-consistency
    audit PASSes).

A blanket ``pytest.mark.skipif`` over this module was REJECTED: it
would discard the CPU-checkable coverage that IS available, and it
would make the tests vacuous.  The branch is a runtime check on the
session state, so both branches stay live and neither is skipped.

Backward-compat verdict
-----------------------
The 1 NEW gate is ADDITIVE only — it does NOT modify any
pre-existing gate / runner / path; the Wf-321D gate-count-consistency
audit invariant (``all == 94``) is preserved because the new gate
bumps EXPECTED_GATE_COUNTS["all"] from 93 to 94 in lockstep with
the in-process audit gate (``p3_gate_count_consistency_audit``).

   [2026-11-01 CORRECTION (Wf-A170) — the paragraph above is the
   Wf-350-C1 ship-time rendering and is PRESERVED VERBATIM per
   AGENTS.md §5.5 rule 6 (append-only + dated).  The LIVE gate count
   is **129** on the ``all`` pipeline (131 on ``launch``) — AGENTS.md
   §1.4 — and was MEASURED on 2026-11-01 by live import of
   ``pre_flight_check.EXPECTED_GATE_COUNTS``:
   ``all``=129, ``frontier``=25, ``chemprop_v2_v3``=18, ``cfm``=8.
   The ``== 94`` figure is the Wf-350-C1 ship target; it was correct
   when written and went stale as later waves added gates.  The exact
   intermediate history of the count is NOT re-derived here.  The
   executable pin in test 3 has been re-synced to 129 with a dated
   comment that keeps the superseded 94 visible.]

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
* The SKIP verdict is asserted as SKIP — it is never rewritten as a
  PASS, and never paraphrased as "achievable" (AGENTS.md §5.2).

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
* ``molmetal/tests/test_wfa170_gpu_required_test_hygiene.py`` — the
  Wf-A170 negative control that proves this repair is NOT vacuous.
"""

from __future__ import annotations

import importlib.util
import json
import os
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
ENV_JSON_PATH = REPO_ROOT / "molmetal" / "reports" / "wf350c1_gpu_env_check.json"

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
# Wf-A170 — the session-state helpers the two-sided contract is built on
# ---------------------------------------------------------------------------


def _session_gpu_masked() -> bool:
    """True when THIS session hides the GPUs from torch.

    Mirrors ``pre_flight_check._session_cuda_is_locked()``
    (``pre_flight_check.py:13118``): ``CUDA_VISIBLE_DEVICES`` is present
    and blank.  The repo's two conftests (repo-root ``conftest.py`` and
    ``molmetal/tests/conftest.py``) blank it at IMPORT time unless
    ``MOLMETAL_TEST_ALLOW_GPU=1``.

    MEASURED 2026-11-01: the gate's SKIP branch fires on exactly this
    predicate, so this test's branch and the gate's verdict can never
    disagree.  This helper reads the ambient environment ONLY — it never
    imports torch, never touches a GPU and never spawns a subprocess.
    """
    return os.environ.get("CUDA_VISIBLE_DEVICES", None) == ""


#: The tokens the SKIP evidence MUST carry.  A SKIP that does not name
#: the session fact AND the measured host facts is a SILENT skip, which
#: is exactly the failure mode the Wf-A128 SKIP contract exists to
#: prevent (a SKIP that reports nothing looks like a green).
_SKIP_EVIDENCE_TOKENS = (
    "CUDA_VISIBLE_DEVICES",
    "SESSION",
    "nvidia_smi_rc=",
    "gpu_count=",
    "gpu_names=",
    "vram_free_gb=",
    "torch_probe=",
)


def _assert_skip_evidence(evidence: str, *, gpu_count=None) -> None:
    """Assert the SKIP is LOUD — it names the session fact + host facts.

    ``gpu_count`` is the value read back from the probe artefact; when
    given, the evidence must carry THAT measured number, so the SKIP
    reports the host facts it measured instead of re-deriving them.
    """
    assert evidence and evidence.strip(), (
        "SKIP evidence is empty; a silent SKIP is indistinguishable from a "
        "false green"
    )
    for token in _SKIP_EVIDENCE_TOKENS:
        assert token in evidence, (
            f"SKIP evidence does not name {token!r}; a SKIP that omits the "
            f"session fact or the measured host facts is not auditable. "
            f"Evidence: {evidence}"
        )
    if gpu_count is not None:
        assert f"gpu_count={gpu_count}" in evidence, (
            f"SKIP evidence does not carry the measured gpu_count="
            f"{gpu_count} read back from the probe artefact; the SKIP must "
            f"report the host facts it measured. Evidence: {evidence}"
        )


# ---------------------------------------------------------------------------
# 1 — gate PASS on live host + canonical schema validity
# ---------------------------------------------------------------------------


def test_gate_wf350c1_gpu_env_probe_pass_on_host(preflight) -> None:
    """Wf-350-C1 (1) — p1_gpu_env_probe verdict + env JSON schema-valid.

    TWO-SIDED contract (Wf-A170, 2026-11-01) — the assertion is honest
    in BOTH sessions:

      * **GPU visible** (``CUDA_VISIBLE_DEVICES`` non-empty AND torch
        sees a GPU): the ORIGINAL contract holds — the gate PASSes and
        the 4 mandatory host-runtime fields are populated.
      * **GPU masked** (``CUDA_VISIBLE_DEVICES`` blank — the repo's
        standing CPU lock): the gate MUST return **SKIP**, never a
        false PASS, and the SKIP evidence MUST be non-empty and MUST
        name both the session fact and the measured host facts.

    Unconditional in BOTH sessions (CPU-checkable): the probe script is
    on disk; the env JSON exists, is non-empty and parses; it carries
    the canonical top-level keys (``gpu_count`` int, ``gpu_names``
    list, ``torch_probe`` dict) and ``torch_probe`` carries the 4
    mandatory fields (``ok`` / ``device_count`` / ``device_name`` /
    ``cuda_runtime``).
    """
    # Confirm probe script is on disk first (gate would FAIL anyway
    # with a clear evidence string if missing; pre-check surfaces
    # a sharper test diagnostic).
    assert PROBE_SCRIPT.is_file(), (
        f"probe script not on disk: {PROBE_SCRIPT}; cannot run "
        f"Wf-350-C1 integration test"
    )

    result = preflight.gate_p1_gpu_env_probe(
        "p1_gpu_env_probe_test",
        "P3",
        "wf350c1",
        ENV_JSON_PATH,
    )

    # ---- CPU-checkable, UNCONDITIONAL: the artefact contract ----
    # The gate writes the artefact BEFORE it decides PASS vs SKIP, so
    # the schema below is checkable in a CPU-locked session too.
    assert ENV_JSON_PATH.is_file(), (
        f"env JSON file not on disk: {ENV_JSON_PATH}; gate status="
        f"{result.status}: {result.evidence}"
    )
    raw = ENV_JSON_PATH.read_text(encoding="utf-8")
    assert raw.strip(), f"env JSON file is empty: {ENV_JSON_PATH}"
    env_summary = json.loads(raw)
    assert isinstance(env_summary, dict), (
        f"env summary is not a JSON object; got {type(env_summary).__name__}"
    )

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

    # gpu_count must be an int (its VALUE is a host fact: 0 on a host
    # with no NVIDIA GPU, >= 1 otherwise — the value is only asserted
    # in the GPU-visible branch below).
    assert isinstance(env_summary["gpu_count"], int), (
        f"env summary['gpu_count'] is not an int; got "
        f"{type(env_summary['gpu_count']).__name__}"
    )

    # torch_probe must be a dict with the 4 mandatory fields.
    tp = env_summary["torch_probe"]
    assert isinstance(tp, dict), (
        f"env summary['torch_probe'] is not a dict; got " f"{type(tp).__name__}"
    )
    for key in ("ok", "device_count", "device_name", "cuda_runtime"):
        assert key in tp, (
            f"env summary['torch_probe'] missing field {key!r}; got "
            f"keys: {sorted(tp.keys())}"
        )

    # ---- the session-dependent verdict ----
    if _session_gpu_masked():
        # CPU-locked session: the gate is GPU-REQUIRED and MUST SKIP.
        # Asserting PASS here would be the false green; asserting FAIL
        # would misreport a SESSION artefact as a host defect.
        assert result.status == "SKIP", (
            f"gate must SKIP in a CPU-locked session (CUDA_VISIBLE_DEVICES "
            f"is blank), never a false PASS; got {result.status}: "
            f"{result.evidence}"
        )
        _assert_skip_evidence(result.evidence, gpu_count=env_summary["gpu_count"])
    else:
        # GPU-visible session: the ORIGINAL contract, unchanged.
        assert result.status == "PASS", (
            f"gate should PASS on live host; got {result.status}: " f"{result.evidence}"
        )

        assert len(env_summary["gpu_names"]) >= 1, (
            "env summary['gpu_names'] is empty; expected >= 1 GPU visible "
            "via nvidia-smi on a 2-GPU host"
        )

        assert env_summary["gpu_count"] >= 1, (
            f"env summary['gpu_count']={env_summary['gpu_count']} < 1 " f"(expected >= 1)"
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
        assert tp[
            "device_name"
        ], f"torch_probe.device_name is empty; got {tp['device_name']!r}"
        # 4. CUDA runtime version non-empty (torch_probe.cuda_runtime)
        assert tp[
            "cuda_runtime"
        ], f"torch_probe.cuda_runtime is empty; got {tp['cuda_runtime']!r}"


# ---------------------------------------------------------------------------
# 2 — JSON schema validity on tmp_path fixture
# ---------------------------------------------------------------------------


def test_gate_wf350c1_gpu_env_probe_json_schema_validity(
    preflight,
    tmp_path,
) -> None:
    """Wf-350-C1 (2) — gate verdict on tmp_path fixture + env JSON on disk.

    Verifies the gate's schema-validity check accepts a freshly-written
    env JSON at a tmp_path location (mirrors the test_wf350a2_compute_
    manifest PASS-on-fixture pattern).

    TWO-SIDED contract (Wf-A170, 2026-11-01):

      * **GPU visible**: the ORIGINAL assertion holds — the gate's
        verdict on the fixture is PASS or FAIL (a host with no
        GPU-backed torch is a legitimate FAIL for a strict probe).
      * **GPU masked** (the CPU-locked session): the gate MUST return
        **SKIP** and the SKIP evidence MUST be non-empty and name the
        session/host facts.  The Wf-350-C1 text said "no SKIP" — that
        rendering predates the Wf-A128 SKIP contract (2026-10-31) and is
        corrected here, dated, with the SKIP now asserted explicitly.

    Unconditional in BOTH sessions: the fixture file exists, parses to
    a dict, carries the 3 canonical top-level keys, ``gpu_names`` is a
    list and ``torch_probe`` is a dict.
    """
    fixture_path = tmp_path / "gpu_env_probe_fixture.json"
    result = preflight.gate_p1_gpu_env_probe(
        "p1_gpu_env_probe_fixture",
        "P3",
        "wf350c1",
        fixture_path,
    )

    # ---- CPU-checkable, UNCONDITIONAL: the fixture contract ----
    assert fixture_path.is_file(), (
        f"env JSON not written at {fixture_path}; gate status="
        f"{result.status}: {result.evidence}"
    )

    env_summary = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert isinstance(env_summary, dict), (
        f"tmp_path fixture env JSON is not a dict; got " f"{type(env_summary).__name__}"
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

    # ---- the session-dependent verdict ----
    if _session_gpu_masked():
        # CPU-locked session: the gate is GPU-REQUIRED and MUST SKIP.
        assert result.status == "SKIP", (
            f"gate must SKIP in a CPU-locked session (CUDA_VISIBLE_DEVICES "
            f"is blank), never a false PASS; got {result.status}: "
            f"{result.evidence}"
        )
        _assert_skip_evidence(result.evidence, gpu_count=env_summary["gpu_count"])
    else:
        # On hosts WITH at least 1 GPU (the canonical 2-GPU Blackwell +
        # RTX 5090 dev box), the gate PASSes. On CPU-only hosts (no
        # nvidia-smi), the gate FAILs — both are acceptable, but the
        # fixture file MUST exist and parse either way.
        assert result.status in ("PASS", "FAIL"), (
            f"gate should PASS or FAIL on tmp_path fixture; got "
            f"{result.status}: {result.evidence}"
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

    [Wf-A170, 2026-11-01 — the ``== 94`` renderings above are the
    Wf-350-C1 ship-time text and are PRESERVED VERBATIM per AGENTS.md
    §5.5 rule 6 (append-only + dated).  The LIVE count is 129; the
    executable pin in the body has been re-synced to 129, dated.]

    The GPU-dependent LAST assertion (the registered runner's verdict)
    is now TWO-SIDED: a GPU-visible session asserts PASS; a CPU-locked
    session asserts SKIP + a loud SKIP evidence.  Every OTHER assertion
    in this test is CPU-checkable and runs UNCONDITIONALLY.
    """
    # (a) build_gates('all') count + EXPECTED invariant.
    all_runners = preflight.build_gates("all")
    assert len(all_runners) == preflight.EXPECTED_GATE_COUNTS["all"], (
        f"all pipeline count {len(all_runners)} != "
        f"EXPECTED_GATE_COUNTS['all']={preflight.EXPECTED_GATE_COUNTS['all']} "
        f"(Wf-321D audit invariant)"
    )

    # Wf-A170 (2026-11-01) — the executable gate-count PIN, re-synced to
    # the LIVE count.  The superseded rendering is PRESERVED VERBATIM
    # below (AGENTS.md §5.5 rule 6: append-only + dated, never a silent
    # edit, never a deletion):
    #
    #     assert preflight.EXPECTED_GATE_COUNTS["all"] == 94, (
    #         f"EXPECTED_GATE_COUNTS['all']={preflight.EXPECTED_GATE_COUNTS['all']} "
    #         f"!= 94 (Wf-350-C1 ship target; 93 pre-Wf-350-C1 baseline + 1 NEW "
    #         f"Gp — gate_p1_gpu_env_probe)"
    #     )
    #
    # MEASURED 2026-11-01: the live count is 129 (AGENTS.md §1.4 — "129
    # gates on the `all` pipeline (131 on `launch`)"; live import of
    # ``pre_flight_check.EXPECTED_GATE_COUNTS`` gives all=129,
    # frontier=25, chemprop_v2_v3=18, cfm=8).  The pin said 94 — the
    # Wf-350-C1 ship target, correct when written — so the PIN itself
    # was STALE and this test was RED in EVERY session for a reason
    # unrelated to what it guards: the same class of defect this wave
    # repairs for the SKIP branch.  Re-synced here, dated, with the
    # superseded value still visible above and in the message below.
    # 2026-11-04 (Wf-A269 wave-5/6) — re-synced, dated; the superseded
    # value 129 stays visible above and in the message below, per
    # AGENTS.md §5.5 rule #6.
    # 2026-11-04 (Wf-A269 wave-22) — re-synced, dated; the superseded
    # value 132 stays visible above and in the message below, per
    # AGENTS.md §5.5 rule #6.  WHY: wave-19 registered
    # `p3_script_contract_check` and moved the live count to 133, but did
    # NOT re-sync this pin — so this TRACKED test was RED for a reason
    # unrelated to what it guards.  MEASURED 2026-11-04 by live import;
    # the stale value was reported by the Wf-A169 gate-count-pin census
    # (`molmetal/scripts/wfa169_gate_count_pins.py`).
    assert preflight.EXPECTED_GATE_COUNTS["all"] == 133, (
        f"EXPECTED_GATE_COUNTS['all']={preflight.EXPECTED_GATE_COUNTS['all']} "
        f"!= 133 (the live 'all'-pipeline gate count — AGENTS.md §1.4, "
        f"MEASURED 2026-11-04 by live import; the superseded Wf-350-C1 "
        f"ship-target pin was 94 — see the dated Wf-A170 comment above)"
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
    #     [Wf-A170, 2026-11-01 — this is the ONE GPU-dependent
    #     assertion in this test; it is now TWO-SIDED: SKIP + a loud
    #     SKIP evidence in a CPU-locked session, PASS in a GPU-visible
    #     session.  (a)-(e) above stay UNCONDITIONAL.]
    gpu_probe_runner = next(r for n, r in all_runners if n == "p1_gpu_env_probe")
    gpu_probe_result = gpu_probe_runner()
    if _session_gpu_masked():
        assert gpu_probe_result.status == "SKIP", (
            f"p1_gpu_env_probe runner must SKIP in a CPU-locked session, "
            f"never a false PASS; got status={gpu_probe_result.status}: "
            f"{gpu_probe_result.evidence}"
        )
        _assert_skip_evidence(gpu_probe_result.evidence)
    else:
        assert gpu_probe_result.status == "PASS", (
            f"p1_gpu_env_probe runner FAILed: status="
            f"{gpu_probe_result.status}: {gpu_probe_result.evidence}; "
            f"live-host regression — gate function broke between "
            f"registration and execution"
        )
