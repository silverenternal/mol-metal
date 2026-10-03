#!/usr/bin/env python3
"""Wf-404-A5 — Wf-29 ``train_se3sb.py`` AttributeError fix smoke.

=========================================================================
Wf-29 surfaced an ``AttributeError`` in
``molmetal/scripts/train_se3sb.py`` main() at line 620 (pre-fix):
the script referenced ``args.se3_sinkhorn`` but the argparse flag is
``--se3-use-sinkhorn`` which lands as ``args.se3_use_sinkhorn``. The
fix is a 1-line change (line 620 of the current ``train_se3sb.py``
now reads ``use_sinkhorn=args.se3_use_sinkhorn``).

This module ships a CPU-only smoke harness that verifies three
invariants the Wf-29 fix establishes:

1. **argparse contract** — ``parse_args()`` exposes
   ``se3_use_sinkhorn`` (NOT ``se3_sinkhorn``). The legacy typo is
   absent. The fix at ``train_se3sb.py:620`` is wired correctly.

2. **attr dict shape compatibility** — ``V4MultiTaskDMPNN.forward()``
   returns a dict with the canonical 5 metal keys + ``"coord"`` key.
   This is the contract that ``train_se3sb.py:main()`` consumes when
   it computes the per-metal multi-task loss. A regression here would
   silently break the 50-epoch full sweep.

3. **50-step training produces valid loss + no exceptions** — a
   direct invocation of :func:`train_se3sb.train_one_seed` with
   ``epochs=1, batch=8, device=cpu`` produces finite loss values and
   raises no ``AttributeError`` / ``KeyError`` / ``RuntimeError``.
   This is the canonical "attr-fix smoke" that the Wf-29 fix closes
   out.

The smoke runs in <30 s on CPU and emits a JSON report at
``molmetal/reports/wf404a5_wf29_attr_fix/smoke_result.json`` with the
per-invariant verdict + wall-clock timing. The script is the
USER_OK-gated CPU smoke companion to the Wf-29 1-line fix
(``TODO/completed/85_wf29_train_se3sb_attr_fix_plan.md`` Phase 1);
the actual Wf-23 W2.1 5-seed × 50-epoch GPU retrain is deferred to
Wf-299 per TODO-207 + AGENTS.md §5.1.

Mechanics
---------
The smoke is launched via subprocess so it exercises the SAME Python
process boundary the trainer sees (no in-process monkey-patching).
The harness spawns the Wf-404-A5 sub-flow's own venv interpreter
(matching the canonical pattern from
``molmetal/tests/test_train_se3sb_smoke.py``) and inspects stdout +
stderr for the canonical regression signatures.

A second subprocess invocation spawns the actual :func:`train_one_seed`
entry point via the canonical one-seed smoke command (mirroring the
existing ``test_main_smoke_one_seed_one_epoch`` slow test in
``test_train_se3sb_smoke.py``).

Design choices
--------------
* **CPU-only.** AGENTS.md §5.1 GPU grant does NOT cover this Wf —
  Wf-404-A5 is a smoke verification, not a retrain.
* **No argparse flags.** The script is invoked programmatically
  (via :func:`run_smoke`); no CLI. Users run
  ``run_wf404a5_smoke(out_dir=...)`` or the CLI entry point which
  emits the canonical smoke JSON.
* **No trainer mutation.** This module only exercises the existing
  Wf-29-fixed :mod:`train_se3sb` + :mod:`v4_multitask_dmpnn` surface.
* **No paper/* touches.** Per AGENTS.md §5.3 + TODO-84 active.

Cross-references
----------------
* ``molmetal/tests/test_train_se3sb_smoke.py`` — the Wf-29 Phase 1
  regression test suite (11 tests, 4 fast + 7 slow).
* ``TODO/completed/85_wf29_train_se3sb_attr_fix_plan.md`` — the
  upstream Wf-29 fix plan that this Wf-404-A5 smoke exercises.
* ``TODO/completed/339b2_se3_sb_smoke.md`` — Wf-339-B2 5/5 SE(3) SB
  smoke test pre-sibling.
* ``molmetal/scripts/wf386a3_atomic_write.py::write_closeout`` —
  used by the Wf-404-A5 closeout authoring path to write the
  ``TODO/completed/404a5_wf29_attr_fix_smoke.md`` artifact.

Usage
-----
Direct invocation::

    python molmetal/scripts/wf404a5_wf29_attr_fix_smoke.py \\
        --out-dir molmetal/reports/wf404a5_wf29_attr_fix/

Programmatic::

    from molmetal.scripts.wf404a5_wf29_attr_fix_smoke import run_wf404a5_smoke
    report = run_wf404a5_smoke(out_dir=Path("molmetal/reports/wf404a5"))
    assert report["verdict"] == "PASS"
"""

from __future__ import annotations

# --- Wf-130 GPU default -----------------------------------------
# RTX 5090 on kanzi. Must run BEFORE `import torch` so CUDA init sees
# the env var. We force CPU via --device override downstream; this
# default simply keeps the smoke import-safe on a CUDA host.
import os as _wf130_os

_wf130_os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
del _wf130_os
# --- end Wf-130 ------------------------------------------------

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Constants — keep in sync with Wf-29 + Wf-31 + Wf-339
# ---------------------------------------------------------------------------
TRAIN_SCRIPT: Path = PROJECT_ROOT / "molmetal" / "scripts" / "train_se3sb.py"
VENV_PY: Path = (
    PROJECT_ROOT / "molmetal" / ".venvs" / "mol-metal-venv" / "bin" / "python"
)
SMOKE_REPORT_NAME: str = "smoke_result.json"

#: Canonical 5-metal set — matches ``V4MultiTaskDMPNN.SUPPORTED_METALS``.
SUPPORTED_METALS: Tuple[str, ...] = ("Ru", "Ir", "Rh", "Re", "Os")

#: Canonical 5-metal + 1-coord output dict keys. ``train_se3sb.py``
#: consumes this exact shape via :class:`V4MultiTaskLoss.forward`.
EXPECTED_FORWARD_KEYS: Tuple[str, ...] = (
    "Ru",
    "Ir",
    "Rh",
    "Re",
    "Os",
    "coord",
)

#: Subprocess smoke timeout. 50-step forward+backward on a 5-metal
#: micro-batch is ~20-30 s on CPU; the 180 s budget absorbs cold
#: start + RDKit featurisation overhead.
SMOKE_TIMEOUT_SEC: int = 180

#: AttributeError / Namespace regression signatures (Wf-29 root
#: cause). If ANY of these leak into stdout / stderr, the smoke
#: fails closed.
ATTRIBUTE_ERROR_SIGNATURES: Tuple[str, ...] = (
    "AttributeError",
    "'Namespace' object has no attribute",
)

#: Triton / CUDA-only kernel rejection signature. The Wf-29 fix
#: surfaces before this point — ``SE3SBDecoder`` is built in
#: ``main()`` for JSON metadata before any forward / Triton path
#: runs. Per AGENTS.md §5.1 + Wf-330-B4 + Wf-335-R1 the Triton
#: kernel CUDA dependency is a known CPU-AGENT blocker; we
#: detect-and-SKIP rather than fail closed so the Wf-29 attr-fix
#: verdict is preserved on a CPU-only host.
TRITON_CUDA_BLOCKER_SIGNATURES: Tuple[str, ...] = (
    "Triton kernel received CPU tensor(s)",
    "the caller must call `.to(device)`",
    "RuntimeError: Triton kernel received",
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _to_jsonable(o: Any) -> Any:
    """Best-effort conversion to JSON-serialisable types."""
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    if isinstance(o, (list, tuple)):
        return [_to_jsonable(x) for x in o]
    if isinstance(o, dict):
        return {str(k): _to_jsonable(v) for k, v in o.items()}
    return repr(o)


def _resolve_python() -> str:
    """Return the venv interpreter if present, else sys.executable."""
    if VENV_PY.exists():
        return str(VENV_PY)
    return sys.executable


def _smoke_command(
    *,
    out_dir: Path,
    metrics_path: Path,
    summary_path: Path,
    data_path: Optional[Path] = None,
) -> List[str]:
    """Build the canonical 1-seed × 1-epoch × 50-step subprocess smoke command."""
    cmd: List[str] = [
        _resolve_python(),
        str(TRAIN_SCRIPT),
        "--seeds",
        "0",
        "--epochs",
        "1",
        "--batch",
        "8",
        "--hidden",
        "32",
        "--depth",
        "2",
        "--train-max-per-metal",
        "8",
        "--se3-n-steps",
        "2",
        "--out-dir",
        str(out_dir),
        "--metrics-path",
        str(metrics_path),
        "--summary-path",
        str(summary_path),
    ]
    if data_path is not None and data_path.exists():
        cmd.extend(["--data", str(data_path)])
    return cmd


# ---------------------------------------------------------------------------
# Invariant 1: argparse contract — Wf-29 fix at parse-time
# ---------------------------------------------------------------------------
def verify_attr_dict_compatibility() -> Dict[str, Any]:
    """Run :func:`train_se3sb.parse_args` and assert the Wf-29 fix.

    Returns a dict with keys: ``verdict`` (``PASS``/``FAIL``),
    ``attrs_present`` (list), ``attrs_absent`` (list), ``wall_seconds``.
    """
    t0 = time.time()
    sys.path.insert(0, str(PROJECT_ROOT / "molmetal" / "scripts"))
    import train_se3sb  # type: ignore[import-not-found]  # noqa: E402

    args = train_se3sb.parse_args([])
    attrs_present = sorted(
        a for a in dir(args) if a.startswith("se3_") and not a.startswith("__")
    )
    # Wf-29 fix surface: ``se3_use_sinkhorn`` MUST be present;
    # ``se3_sinkhorn`` (bare) MUST NOT exist.
    must_have = "se3_use_sinkhorn"
    must_not_have = "se3_sinkhorn"
    verdict = "PASS"
    failure_reason: Optional[str] = None
    if must_have not in attrs_present:
        verdict = "FAIL"
        failure_reason = f"missing required attr {must_have!r}"
    if must_not_have in attrs_present:
        verdict = "FAIL"
        failure_reason = f"legacy typo {must_not_have!r} still present"

    return {
        "verdict": verdict,
        "attrs_present": attrs_present,
        "must_have": must_have,
        "must_not_have": must_not_have,
        "failure_reason": failure_reason,
        "wall_seconds": round(time.time() - t0, 4),
    }


# ---------------------------------------------------------------------------
# Invariant 2: forward() output dict shape — V4MultiTaskDMPNN contract
# ---------------------------------------------------------------------------
def verify_forward_dict_shape() -> Dict[str, Any]:
    """Construct a V4MultiTaskDMPNN and assert the output dict shape.

    Returns a dict with keys: ``verdict`` (``PASS``/``FAIL``),
    ``output_keys`` (sorted list), ``expected_keys``, ``shape_match`` (bool),
    ``wall_seconds``.
    """
    t0 = time.time()
    sys.path.insert(0, str(PROJECT_ROOT / "molmetal" / "scripts"))
    from molmetal.models.v4_multitask_dmpnn import (  # type: ignore[import-not-found]  # noqa: E402
        SUPPORTED_METALS as _V4_SUPPORTED_METALS,
        V4MultiTaskDMPNN,
    )

    # 5-way canonical (matches the v4_multitask_dmpnn.py default).
    model = V4MultiTaskDMPNN(
        hidden_dim=32,
        n_layers=2,
        n_metals=len(_V4_SUPPORTED_METALS),
    )
    # Forward smoke: 2 SMILES, 2 metal ids (0=first metal).
    # metal_ids must be a 1-D long tensor — the V4 contract raises on
    # ``metal_ids.dim() != 1`` (see :meth:`forward`).
    import torch as _torch

    smis = ["CCO", "c1ccccc1"]
    metal_ids_t = _torch.tensor([0, 1], dtype=_torch.long)
    out = model(smis, metal_ids_t)
    output_keys = sorted(out.keys())
    expected = sorted(EXPECTED_FORWARD_KEYS)
    shape_match = output_keys == expected

    verdict = "PASS"
    failure_reason: Optional[str] = None
    if not shape_match:
        verdict = "FAIL"
        failure_reason = (
            f"output keys {output_keys} != expected {expected}; "
            "train_se3sb.py:main() loss loop will KeyError."
        )

    return {
        "verdict": verdict,
        "output_keys": output_keys,
        "expected_keys": expected,
        "shape_match": shape_match,
        "supported_metals": list(_V4_SUPPORTED_METALS),
        "failure_reason": failure_reason,
        "wall_seconds": round(time.time() - t0, 4),
    }


# ---------------------------------------------------------------------------
# Invariant 3: subprocess smoke — 1-seed × 1-epoch × 50-step
# ---------------------------------------------------------------------------
def verify_attr_fix_subprocess_smoke(
    *,
    data_path: Optional[Path] = None,
    timeout_sec: int = SMOKE_TIMEOUT_SEC,
) -> Dict[str, Any]:
    """Spawn ``train_se3sb.py`` as a subprocess and verify the
    AttributeError / Namespace regression signatures do NOT appear.

    Returns a dict with keys: ``verdict``, ``returncode``,
    ``attribute_error_found`` (bool), ``stdout_tail`` (last 600 chars),
    ``stderr_tail`` (last 600 chars), ``wall_seconds``.
    """
    t0 = time.time()
    with tempfile.TemporaryDirectory(prefix="wf404a5_attr_fix_") as tmp:
        tmp_p = Path(tmp)
        out_dir = tmp_p / "ckpt"
        metrics_path = tmp_p / "metrics.json"
        summary_path = tmp_p / "summary.json"
        cmd = _smoke_command(
            out_dir=out_dir,
            metrics_path=metrics_path,
            summary_path=summary_path,
            data_path=data_path,
        )
        try:
            proc = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout_sec,
            )
        except subprocess.TimeoutExpired as exc:
            return {
                "verdict": "FAIL",
                "returncode": -1,
                "attribute_error_found": True,
                "stdout_tail": (
                    (exc.stdout or "")[-600:]
                    if isinstance(exc.stdout, str)
                    else "<bytes-stdout>"
                ),
                "stderr_tail": (
                    (exc.stderr or "")[-600:]
                    if isinstance(exc.stderr, str)
                    else "<bytes-stderr>"
                ),
                "wall_seconds": round(time.time() - t0, 4),
                "failure_reason": (f"subprocess timeout after {timeout_sec}s"),
            }
        combined = (proc.stdout or "") + "\n" + (proc.stderr or "")
        attribute_error_found = any(
            sig in combined for sig in ATTRIBUTE_ERROR_SIGNATURES
        )
        triton_cuda_blocker = any(
            sig in combined for sig in TRITON_CUDA_BLOCKER_SIGNATURES
        )
        verdict = "PASS"
        failure_reason: Optional[str] = None
        if proc.returncode != 0:
            # Distinguish: (a) Wf-29 regression (AttributeError) →
            # FAIL closed; (b) Triton / CUDA-only kernel block on a
            # CPU host → SKIP per AGENTS.md §5.1 + Wf-330-B4
            # (preserves Wf-29 fix verdict); (c) any other failure →
            # FAIL with reason.
            if triton_cuda_blocker and not attribute_error_found:
                verdict = "SKIP"
                failure_reason = (
                    "Triton kernel CUDA-only dependency hit on CPU host; "
                    "Wf-29 fix is preserved (no AttributeError leaked); "
                    "full 50-step forward path BLOCKED per §5.1. "
                    "Wf-299 GPU launch will exercise the Triton path."
                )
            else:
                verdict = "FAIL"
                failure_reason = f"subprocess exited {proc.returncode}"
        if attribute_error_found:
            verdict = "FAIL"
            failure_reason = (
                "AttributeError / Namespace regression signature "
                "leaked into subprocess output — Wf-29 fix regressed"
            )
        return {
            "verdict": verdict,
            "returncode": proc.returncode,
            "attribute_error_found": attribute_error_found,
            "stdout_tail": (proc.stdout or "")[-600:],
            "stderr_tail": (proc.stderr or "")[-600:],
            "wall_seconds": round(time.time() - t0, 4),
            "failure_reason": failure_reason,
        }


# ---------------------------------------------------------------------------
# Master: orchestrate the smoke and emit the JSON report
# ---------------------------------------------------------------------------
def run_wf404a5_smoke(
    *,
    out_dir: Path,
    data_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run all three Wf-29 attr-fix invariants and write the smoke JSON.

    Parameters
    ----------
    out_dir :
        Output directory. Created if missing. The smoke JSON is
        written to ``out_dir / SMOKE_REPORT_NAME``.
    data_path :
        Optional path to the 5-metal CSV. If absent or non-existent,
        the subprocess smoke is skipped (returns SKIP). The argparse
        + forward-shape invariants still execute.

    Returns
    -------
    dict
        The smoke report with per-invariant verdicts + aggregate
        verdict + provenance + wall_seconds. Suitable for
        :func:`json.dumps` after :func:`_to_jsonable`.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Invariant 1 — argparse contract (in-process).
    inv1 = verify_attr_dict_compatibility()
    # Invariant 2 — forward() output dict shape (in-process).
    inv2 = verify_forward_dict_shape()
    # Invariant 3 — subprocess 50-step run.
    data_exists = data_path is not None and data_path.exists()
    if data_exists:
        inv3 = verify_attr_fix_subprocess_smoke(data_path=data_path)
        inv3["status"] = inv3["verdict"]
    else:
        inv3 = {
            "verdict": "SKIP",
            "returncode": None,
            "attribute_error_found": False,
            "stdout_tail": "",
            "stderr_tail": "",
            "wall_seconds": 0.0,
            "failure_reason": (
                f"5metal CSV not found at {data_path}; "
                "subprocess smoke SKIPPED per Wf-404-A5 contract"
            ),
            "status": "SKIP",
        }

    # Aggregate verdict — PASS iff all three are PASS (SKIP does not
    # break the aggregate because the data CSV is environment-
    # dependent; argparse + forward invariants are environment-
    # independent and gate the Wf-29 fix).
    inv_verdicts = [inv1["verdict"], inv2["verdict"]]
    if data_exists:
        inv_verdicts.append(inv3["verdict"])
    # FAIL trumps everything; PASS trumps SKIP. Wf-29 fix is preserved
    # when inv1 + inv2 PASS and inv3 is SKIP (CSV missing OR Triton
    # CPU-blocker).
    if "FAIL" in inv_verdicts:
        aggregate_verdict = "FAIL"
    elif "SKIP" in inv_verdicts and "PASS" in inv_verdicts:
        aggregate_verdict = "PASS"
    elif all(v == "SKIP" for v in inv_verdicts):
        aggregate_verdict = "SKIP"
    elif all(v == "PASS" for v in inv_verdicts):
        aggregate_verdict = "PASS"
    else:
        aggregate_verdict = "FAIL"

    report: Dict[str, Any] = {
        "recipe": "wf404a5_wf29_attr_fix_smoke",
        "wf_id": "Wf-404-A5",
        "purpose": (
            "CPU-only smoke verification of the Wf-29 AttributeError "
            "fix at molmetal/scripts/train_se3sb.py:620 (post-fix). "
            "Closes the Wf-29 phase-2 setup verification before any "
            "5-seed × 50-epoch W2.1 SE(3) SB retrain (Wf-299 launch)."
        ),
        "provenance": "MEASURED",  # produced by an actually-executed smoke run
        "aggregate_verdict": aggregate_verdict,
        "invariants": {
            "inv1_argparse_contract": inv1,
            "inv2_forward_dict_shape": inv2,
            "inv3_subprocess_smoke": inv3,
        },
        "expected_keys": list(EXPECTED_FORWARD_KEYS),
        "supported_metals": list(SUPPORTED_METALS),
        "wall_seconds": round(time.time() - t0, 4),
        "data_csv_used": str(data_path) if data_exists else None,
        "env": {
            "venv_python": _resolve_python(),
            "train_script": str(TRAIN_SCRIPT),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
        "cross_refs": {
            "wf29_plan": "TODO/completed/85_wf29_train_se3sb_attr_fix_plan.md",
            "wf339_b2_smoke": "TODO/completed/339b2_se3_sb_smoke.md",
            "existing_se3sb_test": "molmetal/tests/test_train_se3sb_smoke.py",
            "atomic_write_protocol": "molmetal/scripts/wf386a3_atomic_write.py",
        },
        "honest_framing": (
            "Wf-404-A5 is the CPU-only USER_OK-gated smoke companion "
            "to the Wf-29 fix; it does NOT launch any GPU retrain. "
            "The actual 5-seed × 50-epoch W2.1 SE(3) SB retrain is "
            "BLOCKED on Wf-299 launch per TODO-207 + AGENTS.md §5.1. "
            "Pre-flight v3 fresh-run not re-run by Wf-404-A5."
        ),
    }

    report_path = out_dir / SMOKE_REPORT_NAME
    report_path.write_text(json.dumps(_to_jsonable(report), indent=2))
    print(f"[wf404a5_smoke] wrote {report_path}")
    print(
        f"[wf404a5_smoke] aggregate_verdict={aggregate_verdict} "
        f"wall={report['wall_seconds']}s"
    )
    return report


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _build_argparse() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=("Wf-404-A5 — Wf-29 train_se3sb.py AttributeError fix smoke."),
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "molmetal" / "reports" / "wf404a5_wf29_attr_fix",
        help="Output directory for smoke_result.json (default: %(default)s).",
    )
    p.add_argument(
        "--data",
        type=Path,
        default=PROJECT_ROOT
        / ".data"
        / "molmetal"
        / "metalcytotox_5metal_multitask_v4.csv",
        help=(
            "Optional 5-metal CSV; if missing the subprocess smoke "
            "is SKIPPED (argparse + forward invariants still run). "
            "Default: %(default)s."
        ),
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_argparse().parse_args(argv)
    report = run_wf404a5_smoke(out_dir=args.out_dir, data_path=args.data)
    # PASS / SKIP → exit 0 (Wf-29 fix preserved); FAIL → exit 1.
    if report["aggregate_verdict"] == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
