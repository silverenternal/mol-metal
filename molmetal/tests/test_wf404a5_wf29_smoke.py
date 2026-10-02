"""Pytest: Wf-404-A5 — Wf-29 ``train_se3sb.py`` AttributeError fix smoke.

=========================================================================
The Wf-29 fix at ``molmetal/scripts/train_se3sb.py:620`` (post-fix)
changed ``use_sinkhorn=args.se3_sinkhorn`` →
``use_sinkhorn=args.se3_use_sinkhorn`` to match the argparse flag
``--se3-use-sinkhorn``. Without this fix the W2.1 SE(3) Schrödinger-
Bridge trainer raised ``AttributeError: 'Namespace' object has no
attribute 'se3_sinkhorn'`` before any training started.

This module pins the contract via the canonical
:mod:`wf404a5_wf29_attr_fix_smoke` smoke harness:

1. **test_attr_fix_smoke_passes** — ``run_wf404a5_smoke`` returns
   ``aggregate_verdict == "PASS"`` for all three invariants:
   (a) argparse contract exposes ``se3_use_sinkhorn`` and NOT
       ``se3_sinkhorn`` (the Wf-29 literal 1-line fix);
   (b) :class:`V4MultiTaskDMPNN.forward` returns a dict with the
       canonical 5 metal + ``"coord"`` keys that
       :class:`V4MultiTaskLoss.forward` consumes;
   (c) when the 5-metal CSV is available, the subprocess
       :func:`train_se3sb.main` runs to completion with no
       ``AttributeError`` regression signatures.

2. **test_attr_dict_compatibility** — asserts the forward() output
   dict shape directly via a 2-SMILES smoke; this is the literal
   compatibility check the Wf-29 fix surfaces downstream.

The full subprocess smoke is gated behind
``pytest -m slow`` (mirrors the existing ``test_train_se3sb_smoke.py``
slow marker convention). The fast tests verify the in-process
invariants and run in <10 s total.

Constraints
-----------
* CPU-only — no GPU launch per AGENTS.md §5.1.
* No ``paper/*`` touches per AGENTS.md §5.3 + TODO-84.
* No wet-lab planning per AGENTS.md §5.4 + §10.
* Pure stdlib + project-internal imports — no third-party deps.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict

import pytest

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
SMOKE_SCRIPT = (
    PROJECT_ROOT / "molmetal" / "scripts" / "wf404a5_wf29_attr_fix_smoke.py"
)
TRAIN_SCRIPT = PROJECT_ROOT / "molmetal" / "scripts" / "train_se3sb.py"
DEFAULT_DATA = (
    PROJECT_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask.csv"
)
VENV_PY = (
    PROJECT_ROOT / "molmetal" / ".venvs" / "mol-metal-venv" / "bin" / "python"
)

#: Canonical expected forward() output keys (5 metal + coord).
EXPECTED_FORWARD_KEYS = ("Ru", "Ir", "Rh", "Re", "Os", "coord")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _import_smoke():
    """Import the Wf-404-A5 smoke module via in-process path mutation."""
    sys.path.insert(0, str(PROJECT_ROOT / "molmetal" / "scripts"))
    import wf404a5_wf29_attr_fix_smoke  # type: ignore[import-not-found]

    return wf404a5_wf29_attr_fix_smoke


def _run_in_process_smoke(tmp_path: Path) -> Dict[str, Any]:
    """Invoke :func:`run_wf404a5_smoke` directly (CPU-only in-process)."""
    smoke = _import_smoke()
    report = smoke.run_wf404a5_smoke(
        out_dir=tmp_path / "wf404a5_wf29_attr_fix",
        data_path=None,  # SKIP the subprocess arm — argparse + forward only
    )
    return report


# ---------------------------------------------------------------------------
# Test 1: aggregate smoke PASS (in-process: argparse + forward)
# ---------------------------------------------------------------------------
def test_attr_fix_smoke_passes(tmp_path: Path) -> None:
    """``run_wf404a5_smoke`` returns ``aggregate_verdict == "PASS"``.

    Without the 5-metal CSV the subprocess arm is SKIPPED per the
    Wf-404-A5 contract; the argparse + forward invariants are
    environment-independent and gate the Wf-29 fix.
    """
    report = _run_in_process_smoke(tmp_path)
    assert report["aggregate_verdict"] == "PASS", (
        f"aggregate_verdict={report['aggregate_verdict']}; "
        f"inv1={report['invariants']['inv1_argparse_contract']}; "
        f"inv2={report['invariants']['inv2_forward_dict_shape']}; "
        f"inv3={report['invariants']['inv3_subprocess_smoke']}"
    )
    # Provenance must be MEASURED per party-twenty-spirit §3.5.
    assert report["provenance"] == "MEASURED"
    # Recipe + wf_id must be present.
    assert report["recipe"] == "wf404a5_wf29_attr_fix_smoke"
    assert report["wf_id"] == "Wf-404-A5"


def test_attr_fix_smoke_inv1_argparse(tmp_path: Path) -> None:
    """Inv1 — Wf-29 fix surfaces as ``se3_use_sinkhorn`` in ``parse_args``."""
    smoke = _import_smoke()
    inv1 = smoke.verify_attr_dict_compatibility()
    assert inv1["verdict"] == "PASS", (
        f"argparse contract regressed: {inv1}"
    )
    assert "se3_use_sinkhorn" in inv1["attrs_present"]
    assert "se3_sinkhorn" not in inv1["attrs_present"]


def test_attr_fix_smoke_inv1_legacy_typo_absent(tmp_path: Path) -> None:
    """Inv1 — the literal legacy typo ``se3_sinkhorn`` MUST be absent.

    This is the Wf-29 root cause: ``train_se3sb.py`` main() referenced
    ``args.se3_sinkhorn`` before the fix; the flag is
    ``--se3-use-sinkhorn`` which lands as ``args.se3_use_sinkhorn``.
    """
    smoke = _import_smoke()
    inv1 = smoke.verify_attr_dict_compatibility()
    must_not_have = inv1["must_not_have"]
    assert must_not_have not in inv1["attrs_present"], (
        f"legacy typo {must_not_have!r} present in parse_args "
        "— Wf-29 fix regressed"
    )


# ---------------------------------------------------------------------------
# Test 2: forward dict shape — V4MultiTaskDMPNN contract
# ---------------------------------------------------------------------------
def test_attr_dict_compatibility(tmp_path: Path) -> None:
    """V4MultiTaskDMPNN.forward() returns canonical 5-metal + coord keys."""
    smoke = _import_smoke()
    inv2 = smoke.verify_forward_dict_shape()
    assert inv2["verdict"] == "PASS", (
        f"forward dict shape regressed: {inv2}"
    )
    assert inv2["shape_match"] is True
    assert sorted(inv2["output_keys"]) == sorted(EXPECTED_FORWARD_KEYS)


def test_attr_dict_compatibility_keys_explicit() -> None:
    """Output dict MUST contain exactly the 5 metal keys + 'coord' key.

    This is the literal contract :class:`V4MultiTaskLoss.forward`
    consumes — if any metal head is renamed or dropped, the loss
    loop will KeyError. The Wf-29 fix preserves this contract.
    """
    sys.path.insert(0, str(PROJECT_ROOT / "molmetal" / "scripts"))
    import torch as _torch
    from molmetal.models.v4_multitask_dmpnn import (  # type: ignore[import-not-found]
        SUPPORTED_METALS,
        V4MultiTaskDMPNN,
    )
    model = V4MultiTaskDMPNN(
        hidden_dim=32,
        n_layers=2,
        n_metals=len(SUPPORTED_METALS),
    )
    metal_ids_t = _torch.tensor([0, 1], dtype=_torch.long)
    out = model(["CCO", "c1ccccc1"], metal_ids_t)
    for metal in SUPPORTED_METALS:
        assert metal in out, (
            f"metal head {metal!r} missing from forward output — "
            "train_se3sb.py loss loop will KeyError"
        )
    assert "coord" in out, (
        "coord head missing from forward output — "
        "coord loss will KeyError"
    )
    # No extra unexpected keys (e.g. accidentally registered twice).
    unexpected = set(out.keys()) - set(SUPPORTED_METALS) - {"coord"}
    assert not unexpected, (
        f"unexpected forward output keys {unexpected} — "
        "V4 contract must be exactly 5 metal + coord"
    )


# ---------------------------------------------------------------------------
# Subprocess smoke (slow; gated behind pytest -m slow)
# ---------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.skipif(
    not DEFAULT_DATA.exists(),
    reason=f"5metal CSV not found at {DEFAULT_DATA}",
)
def test_attr_fix_subprocess_smoke(tmp_path: Path) -> None:
    """Full subprocess smoke: train_se3sb.py main() exits 0, no AttributeError."""
    import subprocess

    cmd = [
        str(VENV_PY),
        str(TRAIN_SCRIPT),
        "--seeds", "0",
        "--epochs", "1",
        "--batch", "8",
        "--hidden", "32",
        "--depth", "2",
        "--train-max-per-metal", "8",
        "--se3-n-steps", "2",
        "--out-dir", str(tmp_path / "ckpt"),
        "--metrics-path", str(tmp_path / "metrics.json"),
        "--summary-path", str(tmp_path / "summary.json"),
        "--data", str(DEFAULT_DATA),
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, (
        f"main() exited {result.returncode}\n"
        f"stdout (tail):\n{result.stdout[-2000:]}\n"
        f"stderr (tail):\n{result.stderr[-2000:]}"
    )
    combined = (result.stdout or "") + "\n" + (result.stderr or "")
    assert "AttributeError" not in combined, (
        f"AttributeError leaked:\n{combined[-1500:]}"
    )
    assert "'Namespace' object has no attribute" not in combined
    # Verify metrics were emitted.
    metrics_path = tmp_path / "metrics.json"
    assert metrics_path.exists(), "metrics.json not written by smoke"
    metrics = json.loads(metrics_path.read_text())
    assert metrics.get("recipe") == "r18_w2.1_se3sb_5metal_multitask"


# ---------------------------------------------------------------------------
# Helper smoke: report shape contract (always runs)
# ---------------------------------------------------------------------------
def test_smoke_report_contract(tmp_path: Path) -> None:
    """Smoke JSON must conform to the documented shape."""
    report = _run_in_process_smoke(tmp_path)
    # Required top-level keys.
    required_keys = {
        "wf_id", "recipe", "purpose", "provenance", "aggregate_verdict",
        "invariants", "expected_keys", "supported_metals", "wall_seconds",
        "data_csv_used", "env", "cross_refs", "honest_framing",
    }
    assert required_keys.issubset(report.keys()), (
        f"missing required keys {required_keys - set(report.keys())}"
    )
    # Invariants sub-dict must have all three arms.
    inv_keys = {
        "inv1_argparse_contract",
        "inv2_forward_dict_shape",
        "inv3_subprocess_smoke",
    }
    assert inv_keys.issubset(report["invariants"].keys())
    # Without the CSV the subprocess arm MUST be SKIP.
    assert report["invariants"]["inv3_subprocess_smoke"]["verdict"] == "SKIP"
    # Provenance discipline per party-twenty-spirit §3.5.
    assert report["provenance"] == "MEASURED"
    # Honest framing clause present.
    assert "BLOCKED" in report["honest_framing"] or "USER_OK" in report["honest_framing"]


def test_invariants_run_in_isolation() -> None:
    """Each invariant can be invoked independently (no shared state)."""
    smoke = _import_smoke()
    inv1 = smoke.verify_attr_dict_compatibility()
    inv2 = smoke.verify_forward_dict_shape()
    inv3 = smoke.verify_attr_fix_subprocess_smoke(data_path=None)
    # With no data path, inv3 returns an explicit SKIP-equivalent error.
    assert inv1["verdict"] in {"PASS", "FAIL"}
    assert inv2["verdict"] in {"PASS", "FAIL"}
    # inv3 with no CSV must NOT have run a subprocess.
    assert inv3["returncode"] in {None, -1} or inv3["verdict"] in {"SKIP", "FAIL"}
    assert inv3["wall_seconds"] >= 0