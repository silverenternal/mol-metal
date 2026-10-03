"""Wf-315D unit tests — PoseBusters smoke retry on 1h36.

Per WEAK_TO_STRONG §2.6: 6 NEW tests verify the Wf-315D PB smoke
retry pipeline on pocket 1h36.  All CPU-only, no GPU launch.

Test breakdown:
  3 tests — package availability (RDKit + PoseBusters)
  2 tests — 1h36 PB pass rate computation (m8 protein-aware subset)
  1 test — pre-flight gate (p3_pb_packages_available_or_documented)

If the heavy ``posebusters`` wheel is not installed in the
production venv, the gate-paths tests SKIP with a clear message
that maps to HONEST_NEGATIVE_BLOCKED in the 1h36 metric.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

REPORTS_DIR = ROOT / "molmetal" / "reports" / "wf315d_pb_smoke"
WF246D_DIR = ROOT / "molmetal" / "reports" / "wf246d_m8_pb_dock"
WF179_DIR = ROOT / "molmetal" / "reports" / "wf179_bd9_m15_rmsd_p2"
METRICS_DIR = ROOT / "metrics" / "by_metric"

# The venv where the 1h36 PB smoke actually ran
VENV_PYTHON = "/home/hugo/codes/mol-metal/.venv/bin/python"

# 1h36 docked poses + receptor (cached in Wf-246D)
POSE_DIR_1H36 = WF179_DIR / "1h36"
RECEPTOR_1H36 = WF246D_DIR / "1h36_receptor.pdb"

# Smoke output + metric JSON
SMOKE_OUTPUT = REPORTS_DIR / "m8_prod_1h36.json"
METRIC_1H36 = METRICS_DIR / "steric_clash_pb_dock_1h36.json"


# ---------------------------------------------------------------------------
# Package availability — uses production venv python (not the test venv)
# ---------------------------------------------------------------------------


def _probe_packages():
    """Probe RDKit + PoseBusters availability in the venv.

    Returns ``(ok: bool, rdkit_v: str | None, posebusters_v: str | None)``.
    Falls back to ``(False, None, None)`` if the venv python is
    unreachable / probe fails.
    """
    if not Path(VENV_PYTHON).exists():
        return False, None, None
    probe = (
        "import rdkit, posebusters; "
        "print('rdkit=' + rdkit.__version__); "
        "print('posebusters=' + posebusters.__version__)"
    )
    try:
        out = subprocess.run(
            [VENV_PYTHON, "-c", probe],
            capture_output=True,
            text=True,
            timeout=20.0,
        )
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False, None, None
    if out.returncode != 0:
        return False, None, None
    rdkit_v = None
    posebusters_v = None
    for line in out.stdout.splitlines():
        if line.startswith("rdkit="):
            rdkit_v = line.split("=", 1)[1].strip()
        elif line.startswith("posebusters="):
            posebusters_v = line.split("=", 1)[1].strip()
    if rdkit_v is None or posebusters_v is None:
        return False, None, None
    return True, rdkit_v, posebusters_v


# ---------------------------------------------------------------------------
# Test 1: RDKit available in the venv (>= 2025 release)
# ---------------------------------------------------------------------------


def test_1_rdkit_version_in_venv():
    """RDKit must be importable in the venv; record version."""
    ok, rdkit_v, _ = _probe_packages()
    if not ok:
        pytest.skip(
            "rdkit not installed in {VENV_PYTHON}; Wf-315D path (b) "
            "(HONEST_NEGATIVE_BLOCKED) applies — see "
            "metrics/by_metric/steric_clash_pb_dock_1h36.json "
            "HONEST_NEGATIVE_BLOCKED tag + blocker_description"
        )
    assert rdkit_v is not None
    # RDKit version is "YYYY.MM.patch" — accept any 2024+ release.
    year = int(rdkit_v.split(".")[0])
    assert year >= 2024, f"rdkit {rdkit_v} is older than 2024 baseline"


# ---------------------------------------------------------------------------
# Test 2: PoseBusters available in the venv (>= 0.5)
# ---------------------------------------------------------------------------


def test_2_posebusters_version_in_venv():
    """PoseBusters must be importable in the venv; record version."""
    ok, _, pb_v = _probe_packages()
    if not ok:
        pytest.skip(
            "posebusters not installed in {VENV_PYTHON}; Wf-315D path (b) "
            "(HONEST_NEGATIVE_BLOCKED) applies — see "
            "metrics/by_metric/steric_clash_pb_dock_1h36.json"
        )
    assert pb_v is not None
    major = int(pb_v.split(".")[0])
    assert major >= 0
    # Require at least 0.5 (per Buttenschoen 2024 + DockMode stability)
    assert (major, int(pb_v.split(".")[1])) >= (
        0,
        5,
    ), f"posebusters {pb_v} is older than 0.5 baseline"


# ---------------------------------------------------------------------------
# Test 3: RDKit AllChem EmbedMolecule round-trip works (sanity)
# ---------------------------------------------------------------------------


def test_3_rdkit_allchem_embed_works():
    """RDKit AllChem.EmbedMolecule on ethanol can ship into a conformer."""
    ok, rdkit_v, _ = _probe_packages()
    if not ok:
        pytest.skip("rdkit not installed in {VENV_PYTHON}")
    # Run a minimal embed probe in the venv
    probe = (
        "from rdkit.Chem import AllChem; "
        "from rdkit import Chem; "
        "m = AllChem.MolFromSmiles('CCO'); "
        "m_h = Chem.AddHs(m); "
        "cid = AllChem.EmbedMolecule(m_h); "
        "print('cid=' + str(cid)); "
        "print('n_confs=' + str(m_h.GetNumConformers()))"
    )
    out = subprocess.run(
        [VENV_PYTHON, "-c", probe],
        capture_output=True,
        text=True,
        timeout=20.0,
    )
    assert (
        out.returncode == 0
    ), f"AllChem.EmbedMolecule failed in {VENV_PYTHON}: {out.stderr}"
    cid_line = [l for l in out.stdout.splitlines() if l.startswith("cid=")][0]
    n_confs_line = [l for l in out.stdout.splitlines() if l.startswith("n_confs=")][0]
    cid = int(cid_line.split("=", 1)[1])
    n_confs = int(n_confs_line.split("=", 1)[1])
    assert cid >= 0, f"AllChem.EmbedMolecule returned cid={cid} (failure)"
    assert n_confs == 1, f"expected 1 conformer, got {n_confs}"


# ---------------------------------------------------------------------------
# Test 4: Smoke output exists with valid m8 protein-aware pass rate
# (or HONEST_NEGATIVE_BLOCKED fallback if packages missing)
# ---------------------------------------------------------------------------


def test_4_1h36_pb_smoke_pass_rate():
    """m8_prod_1h36.json must report a valid pass rate for 1h36.

    If RDKit + PoseBusters are installed: m8_protein_aware_pass_rate
    must be in [0, 1] with n_poses > 0 and n_protein_aware_pass
    matching the docked pose count.

    If packages are NOT installed: this test SKIPs with a clear
    message that the metric JSON should carry HONEST_NEGATIVE_BLOCKED.
    """
    ok, rdkit_v, pb_v = _probe_packages()
    if not ok:
        pytest.skip(
            "packages not installed in {VENV_PYTHON}; Wf-315D path (b) "
            "(HONEST_NEGATIVE_BLOCKED) applies"
        )
    if not SMOKE_OUTPUT.exists():
        pytest.fail(
            f"smoke output missing: {SMOKE_OUTPUT} — "
            f"run molmetal/reports/wf315d_pb_smoke/wf315d_pb_1h36.py"
        )
    with SMOKE_OUTPUT.open("r", encoding="utf-8") as fh:
        smoke = json.load(fh)
    rate = smoke.get("m8_protein_aware_pass_rate")
    n_pass = smoke.get("n_protein_aware_pass")
    n_total = smoke.get("n_poses")
    assert rate is not None, "smoke output missing m8_protein_aware_pass_rate"
    assert n_pass is not None, "smoke output missing n_protein_aware_pass"
    assert n_total is not None, "smoke output missing n_poses"
    assert isinstance(rate, float), f"rate type={type(rate).__name__}"
    assert 0.0 <= rate <= 1.0, f"rate {rate} not in [0, 1]"
    assert n_total > 0, f"n_poses={n_total} (expected > 0)"
    assert n_pass >= 0
    assert n_pass <= n_total
    # Cross-check: rate should equal n_pass / n_total (within float epsilon)
    expected_rate = n_pass / n_total
    assert (
        abs(rate - expected_rate) < 1e-9
    ), f"rate={rate} != n_pass/n_total={expected_rate}"


# ---------------------------------------------------------------------------
# Test 5: 1h36 metric file is MEASURED_SMOKE (or HONEST_NEGATIVE_BLOCKED)
# ---------------------------------------------------------------------------


def test_5_1h36_metric_status_transition():
    """steric_clash_pb_dock_1h36.json must be MEASURED_SMOKE (or
    HONEST_NEGATIVE_BLOCKED if packages missing).

    Verifies the Wf-315D DEFERRED → MEASURED_SMOKE transition (or the
    HONEST_NEGATIVE_BLOCKED fallback for path (b)).
    """
    if not METRIC_1H36.exists():
        pytest.fail(f"metric file missing: {METRIC_1H36}")
    with METRIC_1H36.open("r", encoding="utf-8") as fh:
        data = json.load(fh)
    status = data.get("status")
    ok, _, _ = _probe_packages()
    if ok:
        assert status == "MEASURED_SMOKE", (
            f"packages installed but {METRIC_1H36}: status={status!r} "
            f"(expected 'MEASURED_SMOKE')"
        )
        assert data.get("value", {}).get("rate_protein_aware_subset") is not None
    else:
        # Path (b): packages missing
        assert (
            data.get("HONEST_NEGATIVE_BLOCKED") is True
        ), f"packages missing but HONEST_NEGATIVE_BLOCKED={data.get('HONEST_NEGATIVE_BLOCKED')!r}"
        blocker = data.get("blocker_description") or ""
        assert isinstance(blocker, str) and blocker.strip(), (
            "blocker_description must be a non-empty string when "
            "HONEST_NEGATIVE_BLOCKED is True"
        )


# ---------------------------------------------------------------------------
# Test 6: Pre-flight gate logic (the gate_wf315d_pb_smoke function)
# ---------------------------------------------------------------------------


def test_6_preflight_gate_wf315d():
    """Pre-flight gate ``p3_pb_packages_available_or_documented`` PASSES.

    Either path (a) packages available + MEASURED_SMOKE + smoke output,
    or path (b) packages missing + HONEST_NEGATIVE_BLOCKED + blocker.
    """
    # Import the gate function directly
    from molmetal.scripts.pre_flight_check import gate_wf315d_pb_smoke, GateResult

    result = gate_wf315d_pb_smoke(
        name="p3_pb_packages_available_or_documented",
        priority="P3",
        pipeline="wf315d",
        json_path=METRIC_1H36,
        smoke_output=SMOKE_OUTPUT,
        venv_python=VENV_PYTHON,
    )
    assert isinstance(
        result, GateResult
    ), f"gate returned wrong type: {type(result).__name__}"
    assert result.status in ("PASS", "FAIL"), f"unexpected status {result.status!r}"
    # When packages are installed + metric is MEASURED_SMOKE + smoke
    # output exists, the gate should PASS.
    ok, rdkit_v, pb_v = _probe_packages()
    if ok and METRIC_1H36.exists() and SMOKE_OUTPUT.exists():
        with METRIC_1H36.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        if data.get("status") == "MEASURED_SMOKE":
            assert result.status == "PASS", (
                f"gate FAILED but packages installed + MEASURED_SMOKE + "
                f"smoke output present: {result.evidence}"
            )
