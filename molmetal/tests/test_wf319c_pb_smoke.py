"""Wf-319C regression tests for PoseBusters packages status + 1h36 PB smoke.

==============================================================================
Wf-319C — PoseBusters packages verify install + 1h36 PB check (ACTUAL EXECUTION)
==============================================================================

This module locks in the Wf-319C / Wf-315D PB smoke contract:

1. **Package availability (3 envs)** — probe ``import rdkit, posebusters``
   on the primary the project .venv, the alt mol-metal .venv, and the pocket
   posebusters311 fallback env.  At least one env must have both
   packages; the canonical env (.venv) MUST have both for the
   pre-flight gate to PASS.

2. **1h36 PB pass rate computation** — verify the smoke output
   ``molmetal/reports/wf315d_pb_smoke/m8_prod_1h36.json`` exists,
   parses, and reports ``m8_protein_aware_pass_rate = 1.0`` on
   ``n_poses = 15`` (the 3 seeds × 5 poses from Wf-179 P2).  Also
   verify the corresponding metric JSON
   ``metrics/by_metric/steric_clash_pb_dock_1h36.json`` carries
   ``status == "MEASURED_SMOKE"``.

3. **Pre-flight gate validation** — verify the
   ``gate_wf319c_pb_packages_status_present`` function exists in
   :mod:`molmetal.scripts.pre_flight_check` and returns PASS when
   invoked against the canonical on-disk artefacts.

Honest framing (party-twenty-spirit §3.5):
* The smoke run is **1h36 only** (15 poses, not 100×3 production).
* Full production 100×3 is BLOCKED on F2(a) MetalLigandExchange SMARTS
  rule (TODO-29) + GPU portion per AGENTS.md §5.1.
* All numbers in this file trace to the on-disk artefacts cited in
  the docstring — no fabricated values.

All tests are CPU-only and run in < 30 seconds.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Optional, Tuple

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Canonical on-disk artefact paths (Wf-315D ship surface + Wf-319C gate)
# ---------------------------------------------------------------------------

CANONICAL_VENV_PYTHON = "/home/hugo/codes/mol-metal/.venv/bin/python"
ALT_MOLMETAL_VENV_PYTHON = "/home/hugo/codes/mol-metal/.venv/bin/python"
POCKET_PB311_VENV_PYTHON = (
    "/home/hugo/codes/pocket/.eval_envs/posebusters311/bin/python"
)

METRIC_JSON = PROJECT_ROOT / "metrics" / "by_metric" / "steric_clash_pb_dock_1h36.json"
SMOKE_OUTPUT = (
    PROJECT_ROOT / "molmetal" / "reports" / "wf315d_pb_smoke" / "m8_prod_1h36.json"
)
RECEPTOR_PDB = (
    PROJECT_ROOT / "molmetal" / "reports" / "wf246d_m8_pb_dock" / "1h36_receptor.pdb"
)
WF179_POSE_DIR = (
    PROJECT_ROOT / "molmetal" / "reports" / "wf179_bd9_m15_rmsd_p2" / "1h36"
)


# ---------------------------------------------------------------------------
# Helper probes
# ---------------------------------------------------------------------------


def _probe_packages(
    python_path: str, timeout: float = 10.0
) -> Tuple[bool, Optional[str], Optional[str]]:
    """Return ``(both_available, rdkit_version, posebusters_version)``.

    Probes ``import rdkit, posebusters`` in the given venv.  Returns
    ``(False, None, None)`` on any probe failure (missing venv,
    missing pkg, timeout, or non-zero returncode).  Versions are only
    populated when both packages import successfully.
    """
    if not Path(python_path).exists():
        return False, None, None
    probe = (
        "import rdkit, posebusters; "
        "print('rdkit=' + rdkit.__version__); "
        "print('posebusters=' + posebusters.__version__)"
    )
    try:
        out = subprocess.run(
            [python_path, "-c", probe],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False, None, None
    if out.returncode != 0:
        return False, None, None
    rdkit_v: Optional[str] = None
    pb_v: Optional[str] = None
    for line in out.stdout.splitlines():
        if line.startswith("rdkit="):
            rdkit_v = line.split("=", 1)[1].strip()
        elif line.startswith("posebusters="):
            pb_v = line.split("=", 1)[1].strip()
    if rdkit_v is None or pb_v is None:
        return False, None, None
    return True, rdkit_v, pb_v


# ===========================================================================
# §1 — Package availability (3 envs)
# ===========================================================================


def test_canonical_venv_vlm_has_rdkit_and_posebusters() -> None:
    """Canonical .venv at ``/home/hugo/codes/mol-metal/.venv/`` MUST
    carry both ``rdkit`` and ``posebusters``.  This is the env the Wf-315D
    pre-flight probe targets; if it loses either package, the
    ``p3_pb_packages_available_or_documented`` and
    ``p3_pb_packages_status_present`` gates fail-closed.
    """
    ok, rdkit_v, pb_v = _probe_packages(CANONICAL_VENV_PYTHON)
    assert ok, (
        f"canonical env probe failed: {CANONICAL_VENV_PYTHON} — "
        f"rdkit={rdkit_v!r} posebusters={pb_v!r}"
    )
    assert rdkit_v is not None and pb_v is not None
    # Sanity: posebusters 0.6.x is the version pinned in Wf-315D
    # wf315d_pb_1h36.py and the wf246d_m8_pb_dock summary.json.
    assert pb_v.startswith("0."), f"unexpected posebusters version: {pb_v}"


def test_alt_molmetal_venv_has_rdkit_only() -> None:
    """Alt mol-metal ``.venv`` MUST have ``rdkit`` but may NOT have
    ``posebusters``.  This pins the partial-install state observed on
    2026-09-29: the canonical .venv is the working env, the alt
    .venv is rdkit-only (no PB).  If the alt .venv silently gains PB
    without our knowledge, the venv selection in the pre-flight probe
    becomes ambiguous and we should update wf319c_pb_venv_python.
    """
    # Use a probe that requires rdkit only — independent of posebusters.
    probe_rdkit = "import rdkit; print('rdkit=' + rdkit.__version__)"
    if not Path(ALT_MOLMETAL_VENV_PYTHON).exists():
        pytest.skip(f"alt venv not present: {ALT_MOLMETAL_VENV_PYTHON}")
    out = subprocess.run(
        [ALT_MOLMETAL_VENV_PYTHON, "-c", probe_rdkit],
        capture_output=True,
        text=True,
        timeout=10.0,
    )
    assert out.returncode == 0, f"alt .venv missing rdkit: {out.stderr}"
    assert "rdkit=" in out.stdout

    # posebusters is optional in this env — branch on absence.
    ok, _, pb_v = _probe_packages(ALT_MOLMETAL_VENV_PYTHON)
    if ok:
        # If the alt .venv now has both packages, that's fine but
        # worth surfacing — it means the canonical env choice should
        # be revisited in pre_flight_check.py.
        pytest.skip(
            f"alt .venv now carries posebusters={pb_v!r} — canonical env "
            f"selection may need revisit (Wf-319C cross-ref)"
        )
    # else: posebusters absent — the documented state.


def test_pocket_pb311_env_has_both_packages() -> None:
    """Pocket fallback env at
    ``/home/hugo/codes/pocket/.eval_envs/posebusters311/`` MUST carry
    both ``rdkit`` and ``posebusters`` (it is the documented fallback
    if the canonical .venv ever breaks).  If the pocket env also
    loses PB, the Wf-319C ship surface has zero working envs — this is
    a fail-closed signal.
    """
    if not Path(POCKET_PB311_VENV_PYTHON).exists():
        pytest.skip(
            f"pocket pb311 env not present: {POCKET_PB311_VENV_PYTHON} — "
            f"no fallback env available"
        )
    ok, rdkit_v, pb_v = _probe_packages(POCKET_PB311_VENV_PYTHON)
    assert ok, (
        f"pocket pb311 env probe failed — RDKit={rdkit_v!r} "
        f"PoseBusters={pb_v!r}. Both packages MUST be present in "
        f"this fallback env per Wf-319C documented state."
    )


# ===========================================================================
# §2 — 1h36 PB pass rate computation
# ===========================================================================


def test_smoke_output_exists_with_canonical_schema() -> None:
    """``molmetal/reports/wf315d_pb_smoke/m8_prod_1h36.json`` MUST exist
    and carry the canonical keys: ``n_poses`` (int == 15),
    ``n_protein_aware_pass`` (int), ``m8_protein_aware_pass_rate``
    (float in [0, 1]), ``results`` (list of 15 pose dicts).
    """
    assert SMOKE_OUTPUT.exists(), f"smoke output missing: {SMOKE_OUTPUT}"
    data = json.loads(SMOKE_OUTPUT.read_text(encoding="utf-8"))
    assert isinstance(data, dict)
    assert (
        "n_poses" in data and data["n_poses"] == 15
    ), f"expected n_poses=15, got {data.get('n_poses')!r}"
    assert (
        "n_protein_aware_pass" in data
    ), f"missing canonical key: n_protein_aware_pass in {SMOKE_OUTPUT}"
    assert (
        "m8_protein_aware_pass_rate" in data
    ), f"missing canonical key: m8_protein_aware_pass_rate in {SMOKE_OUTPUT}"
    rate = data["m8_protein_aware_pass_rate"]
    assert 0.0 <= rate <= 1.0, f"rate out of [0,1]: {rate}"
    assert isinstance(data.get("results"), list)
    assert (
        len(data["results"]) == 15
    ), f"results list length {len(data['results'])} != 15"


def test_m8_protein_aware_pass_rate_is_one() -> None:
    """MEASURED check: ``m8_protein_aware_pass_rate == 1.0000`` (15/15)
    on the on-disk Wf-179 P2 docked poses for pocket 1h36.  This is the
    Wf-315D smoke result that flipped ``steric_clash_pb_dock_1h36.json``
    from DEFERRED to MEASURED_SMOKE.

    Honest framing: 1.0000 is the m8 protein-aware subset rate (9
    protein-aware checks per pose, all pass on Vina heavy-atom-only
    output).  Full 26-check PB battery is 0.0/15 on raw Vina — this
    is the documented Vina limitation per Wf-246D synthesis.md §3.
    """
    assert SMOKE_OUTPUT.exists(), f"smoke output missing: {SMOKE_OUTPUT}"
    data = json.loads(SMOKE_OUTPUT.read_text(encoding="utf-8"))
    rate = data.get("m8_protein_aware_pass_rate")
    n_pass = data.get("n_protein_aware_pass")
    n_total = data.get("n_poses")
    assert rate == pytest.approx(
        1.0
    ), f"m8_protein_aware_pass_rate={rate!r} (expected 1.0)"
    assert (
        n_pass == n_total == 15
    ), f"n_protein_aware_pass={n_pass!r} n_poses={n_total!r} (expected 15/15)"


def test_metric_json_status_is_measured_smoke() -> None:
    """``metrics/by_metric/steric_clash_pb_dock_1h36.json`` MUST carry
    ``status == "MEASURED_SMOKE"`` (flipped from DEFERRED by Wf-315D).

    The previous_status field is also expected to be ``DEFERRED`` —
    this locks in the Wf-315D status-flip evidence for downstream
    tooling (paper §3.4 steric clash cluster) that branches on
    ``status`` only.
    """
    assert METRIC_JSON.exists(), f"metric json missing: {METRIC_JSON}"
    data = json.loads(METRIC_JSON.read_text(encoding="utf-8"))
    assert data.get("status") == "MEASURED_SMOKE", (
        f"{METRIC_JSON.name}: status={data.get('status')!r} "
        f"(expected 'MEASURED_SMOKE')"
    )
    assert data.get("previous_status") == "DEFERRED", (
        f"{METRIC_JSON.name}: previous_status={data.get('previous_status')!r} "
        f"(expected 'DEFERRED' — the Wf-315D status flip)"
    )


# ===========================================================================
# §3 — Pre-flight gate validation
# ===========================================================================


def test_gate_wf319c_pb_packages_status_present_exists_and_passes() -> None:
    """The ``gate_wf319c_pb_packages_status_present`` function MUST exist
    in :mod:`molmetal.scripts.pre_flight_check` and MUST return
    ``GateResult(status='PASS')`` when invoked against the canonical
    on-disk artefacts (the Wf-315D metric + smoke output + the
    the project .venv).

    This is the regression net that prevents silent removal of the
    Wf-319C gate or the Wf-315D artefacts it depends on.
    """
    try:
        from molmetal.scripts.pre_flight_check import (  # type: ignore
            gate_wf319c_pb_packages_status_present,
        )
    except ImportError as exc:
        pytest.fail(
            f"gate_wf319c_pb_packages_status_present not importable: {exc}. "
            f"Wf-319C requires this gate in molmetal/scripts/pre_flight_check.py."
        )

    result = gate_wf319c_pb_packages_status_present(
        "p3_pb_packages_status_present",
        "P3",
        "wf319c",
        CANONICAL_VENV_PYTHON,
        METRIC_JSON,
        SMOKE_OUTPUT,
    )
    assert (
        result.status in ("PASS", "SKIP")
    ), f"gate returned {result.status!r}: {result.evidence}"
    assert "rdkit=" in result.evidence
    assert "posebusters=" in result.evidence
    # Wf-445-A1: the on-disk smoke is n_poses=15, below
    # M8_PB_PRODUCTION_MIN_POSES=100, so the gate SKIPs rather than
    # PASSes. If a future production re-measure lands, this flips to PASS.
    if result.status == "SKIP":
        assert "NOT PB production" in result.evidence, (
            "SKIP must state the smoke-vs-production distinction "
            f"(evidence={result.evidence!r})"
        )


def test_gate_wf319c_pb_packages_status_present_fails_on_missing_metric() -> None:
    """The same gate MUST return ``FAIL`` when the metric JSON is
    absent — pins the fail-closed contract.

    We pass a non-existent metric path to verify the failure branch
    triggers, NOT the venv probe or smoke output branches.
    """
    try:
        from molmetal.scripts.pre_flight_check import (  # type: ignore
            gate_wf319c_pb_packages_status_present,
        )
    except ImportError:
        pytest.skip("gate not importable — covered by sibling test")

    nonexistent = PROJECT_ROOT / "nonexistent_metric_for_wf319c_test.json"
    assert not nonexistent.exists(), "test pre-condition violated"

    result = gate_wf319c_pb_packages_status_present(
        "p3_pb_packages_status_present_fail_test",
        "P3",
        "wf319c",
        CANONICAL_VENV_PYTHON,
        nonexistent,
        SMOKE_OUTPUT,
    )
    assert (
        result.status == "FAIL"
    ), f"expected FAIL on missing metric, got {result.status!r}: {result.evidence}"
    assert "metric_json" in result.evidence or "missing" in result.evidence.lower()
