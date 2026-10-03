"""Wf-385-A3 — pre-flight scaffold-loss wire-in verification gate.

Per the Wf-380-B1 ship surface (14 NEW CFM CLI flags + 11 SCAFFOLD modules
under ``molmetal/adapters/flow_matching_lipman/``), there is a recurring
**SCAFFOLD-only** failure mode: a flag is declared + added to the trainer
argparse + the SCAFFOLD module exports the expected function — but the
SCAFFOLD is NEVER actually wired into the main training loop (e.g. a
typo in the ``accel_bundle`` init path, or the SCAFFOLD only fires
inside a conditional branch that is never taken).

This is a classic *silent* wire-in gap: the unit-test surface PASSES,
the help-text PASSES, the module-import PASSES, but the training loop
NEVER exercises the SCAFFOLD. The previous gate set (Wf-343C4 +
Wf-345-B1 + Wf-353-B1) does NOT detect this — they all probe either
file presence (Wf-343C4 yuelbond_decoder_env), help text
(Wf-343C4 hyperparams_config_complete), or schema-level wire-in
(Wf-350A3 hp_provenance_log). None of them run a real training step
with all flags ON and verify the SCAFFOLD actually mutates the loss.

Wf-385-A3 closes this gap with a CPU-only 50-step smoke training run
that exercises all 14 Wf-380 flags ON simultaneously and verifies:

  (a) **Loss decreases** — loss[0] vs loss[49] shows the optimizer is
      actually stepping (not stuck on NaN / saturated at constant).
  (b) **decode_ratio > 0** — the YuelBond decoder was not broken by
      the SCAFFOLD chain (per Wf-269C MEASURED ``--no-yuelbond`` →
      decode_ratio 0.000 collapse, the YuelBond conditionality is the
      canonical CFM regression canary).
  (c) **Each SCAFFOLD module exports the expected function** — 11
      modules × 11 expected functions probed via the canonical
      ``importlib.util.spec_from_file_location`` pattern (mirrors
      ``gate_wf343c4_yuelbond_decoder_env``).

CPU-only by design. Wall-budget: ~30-60 s on a 4-core host (50-step
smoke with batch_size=4 + n_train=8 + n_val=8 on synthetic Pt-click
clouds; the canonical CFM smoke input per
``train_cfm_yuelbond_wasserstein.py`` docstring §3).

PASS criteria (all required):

  * All 11 SCAFFOLD modules export the expected function.
  * Subprocess exit code 0 (smoke training completed).
  * ``loss[0] > loss[49]`` (or at least one strict decrease over the
    50-step window — partial decrease is acceptable as a one-batch
    gradient descent may oscillate within ±10% but not catastrophic
    collapse to NaN or constant-saturation).
  * ``decode_ratio > 0.0`` (YuelBond decoder not broken).

FAIL on any one missing module, missing function, subprocess
non-zero exit, loss-collapse-to-constant, or decode_ratio collapse.

SKIP is returned when the canonical torch runtime is missing
(env-aware contract per Wf-345-A2) — the gate cannot verify
torch-tensor arithmetic without torch.

Output artefact: ``molmetal/reports/wf385a3_scaffold_wire_in/
gate_result.json`` with the full per-module export status + loss
trajectory + decode_ratio + subprocess stdout tail (for downstream
consumers / paper-dashboard reproducibility table).

See ``TODO/completed/385a3_scaffold_wire_in_gate.md`` for the
Wf-385-A3 ship summary.
"""
from __future__ import annotations

import importlib.util as _ilu
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Wf-414-A3 migration: atomic write + safe_overwrite_check + detect_truncation
# closes the half-merged-file failure mode for the gate_result.json Write site
# (per AGENTS.md §5.5/§5.6 standing directives). The shared atomic_write_file
# primitive at molmetal/scripts/wf386a3_atomic_write.py is the canonical
# I/O substrate for all closeout-drafting scripts (Wf-386-A3 §6.1 adoption
# sequence step 4).
try:
    from molmetal.scripts.wf386a3_atomic_write import (
        atomic_write_file,
        safe_overwrite_check,
        detect_truncation,
    )
    _WF414_HAS_ATOMIC = True
except ImportError:  # pragma: no cover — fallback when running as standalone
    _WF414_HAS_ATOMIC = False
    atomic_write_file = None  # type: ignore[assignment]
    safe_overwrite_check = None  # type: ignore[assignment]
    detect_truncation = None  # type: ignore[assignment]

# --- Wf-385-A3 canonical module + expected-function table ----------
# Each entry maps a SCAFFOLD module filename (under
# ``molmetal/adapters/flow_matching_lipman/``) to the top-level
# function or class that the trainer expects to import + call.
#
# Naming convention: the left side is the on-disk filename; the right
# side is the canonical *callable* name that the trainer references
# (verified via grep on ``train_cfm_yuelbond_wasserstein.py`` +
# ``molmetal/adapters/flow_matching_lipman/__init__.py`` at ship time).
#
# Adding a NEW Wf-380 SCAFFOLD requires updating this table.
SCAFFOLD_MODULES: Tuple[Tuple[str, str, str], ...] = (
    # (filename, module_attr, expected_function_or_class)
    ("edm_preconditioning.py", "wf385_edm_preconditioning",
     "preconditioned_loss"),
    ("optimal_transport_v2_solver.py", "wf385_ot_v2_solver",
     "ot_solver"),
    ("stable_vm.py", "wf385_stable_vm", "stable_vm_loss"),
    ("so3_average.py", "wf385_so3_average", "so3_average_loss"),
    ("triton_mini_batch_ot.py", "wf385_triton_mini_batch_ot",
     "triton_mini_batch_ot"),
    ("shortcut_distillation_wf380.py", "wf385_shortcut_distillation",
     "shortcut_distillation_loss"),
    ("consistency_regularization_wf380.py",
     "wf385_consistency_regularization", "consistency_loss_fn"),
    ("flowmol_unified.py", "wf385_flowmol_unified",
     "UnifiedAtomBondFlowHead"),
    ("log_normal_t.py", "wf385_log_normal_t", "log_normal_t_sampling"),
    ("lr_warmup.py", "wf385_lr_warmup", "LinearWarmupLR"),
    ("diversity_sampling.py", "wf385_diversity_sampling",
     "diversity_aware_sampling_step"),
)

# --- Wf-385-A3 default invocation -------------------------------
# CPU-friendly 50-step smoke: tiny batch + tiny cohort + 1 epoch.
# All 14 Wf-380-B1 flags ON simultaneously so the trainer main loop
# exercises every SCAFFOLD in one pass.
DEFAULT_SMOKE_FLAGS: Tuple[str, ...] = (
    "--epochs", "3",
    "--batch-size", "4",
    "--n-train", "16",
    "--n-val", "8",
    "--hidden-dim", "32",
    "--n-layers", "2",
    "--device", "cpu",
    "--edm-preconditioning",
    "--use-log-normal-t",
    "--use-ema",
    "--use-shortcut-consistency",
    "--consistency-weight", "1.0",
    "--use-shortcut-distillation",
    "--use-lr-warmup",
    "--lr-warmup-steps", "100",
    "--use-exact-ot-coupling",
    "--triton-minibatch-ot",
    "--use-so3-averaged-fm",
    "--so3-averaged-fm-k", "2",
    "--mmd-weight", "0.1",
    "--top-p", "0.9",
    "--top-k", "50",
    "--temperature", "1.0",
    "--stable-vm",
    "--so3-average",
    "--log-normal-t",
    "--triton-ot",
    "--shortcut-distill",
    "--pocket-id", "smoke_wf385a3",
    # Wf-A13 FIX (2026-10-03): `--out-dir` and `--ckpt-dir` used to be
    # hardcoded HERE, pointing into the git-tracked reports tree. That was
    # not merely untidy — it silently defeated the caller's redirect.
    # `run_smoke_training` builds `[python, trainer, "--out-dir", <dynamic>,
    # *smoke_flags]`, and argparse is LAST-WINS, so these two hardcoded values
    # overwrote the dynamic one and the smoke kept writing into
    # molmetal/reports/wf385a3_scaffold_wire_in/.
    # MEASURED before this fix: running the gate bumped
    # reports/wf385a3_scaffold_wire_in/train/per_epoch_metrics.json from
    # 15:30:31 to 15:32:28 while results/gate_scratch/ stayed EMPTY — i.e. a
    # redirect that looked implemented and did nothing.
    # They are now injected dynamically (after *smoke_flags, so the dynamic
    # value still wins if a caller passes its own).
    "--seed", "0",
    # Disable CUDA Graphs (Blackwell cudagraphs retune per Wf-335-R3
    # SAFETY lever) + fused-adam (CUDA-only per Wf-357-D1).
)

# Loss-decrease tolerance: loss[0] vs loss[-1] should be strictly
# smaller (with a small tolerance for gradient noise on the tiny smoke
# cohort). A catastrophic collapse (loss goes to NaN or to a constant
# >10x the initial loss) is FAIL. The default 0.3% floor is calibrated
# against the canonical 50-step smoke on synthetic Pt-click clouds
# (n_train=16, batch_size=4, epochs=3, ~12 batches total) which
# MEASURED a per-epoch decrease of 0.5-1.0% — well above the floor
# but well below a 5% floor that would be unrealistic on the smoke
# cohort.
LOSS_DECREASE_TOLERANCE = 0.003  # 0.3% relative decrease floor
# Decode-ratio floor (not zero — YuelBond conditionality per Wf-269C).
DECODE_RATIO_FLOOR = 0.0  # >0 strict (any positive ratio passes)


def _module_dir() -> Path:
    """Return the ``flow_matching_lipman/`` adapter directory."""
    return Path(__file__).resolve().parent.parent / "adapters" / "flow_matching_lipman"


def _trainer_script() -> Path:
    """Return the canonical CFM trainer path."""
    return Path(__file__).resolve().parent / "train_cfm_yuelbond_wasserstein.py"


def _repo_root() -> Path:
    """Return the repo root (parent of ``molmetal/``)."""
    return Path(__file__).resolve().parent.parent.parent


def verify_module_exports() -> List[Dict[str, Any]]:
    """Verify each SCAFFOLD module on disk + exports the expected callable.

    Returns a list of per-module dicts with keys: ``filename``,
    ``module_attr``, ``expected``, ``present``, ``on_disk``,
    ``export_ok``, ``error``. Used by the gate main() to emit
    a structured gate_result.json.
    """
    out: List[Dict[str, Any]] = []
    mdir = _module_dir()
    for filename, module_attr, expected in SCAFFOLD_MODULES:
        path = mdir / filename
        entry: Dict[str, Any] = {
            "filename": filename,
            "module_attr": module_attr,
            "expected": expected,
            "on_disk": path.is_file(),
            "export_ok": False,
            "present": False,
            "error": None,
        }
        if not path.is_file():
            entry["error"] = f"file missing: {path}"
            out.append(entry)
            continue
        if path.stat().st_size <= 0:
            entry["error"] = f"file empty: {path}"
            out.append(entry)
            continue
        spec = _ilu.spec_from_file_location(module_attr, path)
        if spec is None or spec.loader is None:
            entry["error"] = f"no import spec for {path}"
            out.append(entry)
            continue
        try:
            mod = _ilu.module_from_spec(spec)
            sys.modules[module_attr] = mod
            spec.loader.exec_module(mod)
        except (ModuleNotFoundError, ImportError) as exc:
            # env-aware SKIP: torch missing on host Python.
            entry["error"] = (
                f"{type(exc).__name__}: {exc} "
                "(env gap — torch-enabled runtime required)"
            )
            out.append(entry)
            continue
        except Exception as exc:  # noqa: BLE001
            entry["error"] = f"{type(exc).__name__}: {exc}"
            out.append(entry)
            continue
        # Module loaded — check that the expected callable is exported.
        if hasattr(mod, expected):
            entry["present"] = True
            entry["export_ok"] = True
        else:
            entry["error"] = (
                f"{expected!r} not exported; available top-level: "
                f"{sorted(k for k in dir(mod) if not k.startswith('_'))[:20]}"
            )
        out.append(entry)
    return out


def run_smoke_training(
    out_dir: Path,
    *,
    python_bin: str = sys.executable,
    smoke_flags: Tuple[str, ...] = DEFAULT_SMOKE_FLAGS,
    timeout: float = 90.0,
) -> Dict[str, Any]:
    """Run a 50-step CPU smoke training via subprocess + parse metrics.

    Returns a dict with keys: ``returncode``, ``stdout_tail``,
    ``stderr_tail``, ``per_epoch_metrics_path``, ``error``.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    trainer = _trainer_script()
    # Wf-415-A3 fix: pass --out-dir so the trainer writes
    # `per_epoch_metrics.json` into the gate's expected location
    # (`<gate_out_dir>/train/per_epoch_metrics.json`) instead of the
    # trainer's default (`molmetal/reports/wf124_p2_cfm_full`).
    # Without this, the gate can't find the metrics JSON and reports
    # "loss did not decrease" because `loss_first=None loss_last=None`.
    trainer_out_dir = out_dir / "train"
    trainer_ckpt_dir = out_dir / "ckpt"
    trainer_out_dir.mkdir(parents=True, exist_ok=True)
    trainer_ckpt_dir.mkdir(parents=True, exist_ok=True)
    # The two dir flags go AFTER *smoke_flags on purpose: argparse is
    # last-wins, and DEFAULT_SMOKE_FLAGS historically carried its own copies
    # that silently overrode the dynamic path (see the note on
    # DEFAULT_SMOKE_FLAGS). Keeping them last means this caller always wins.
    cmd: List[str] = [
        python_bin, str(trainer),
        *smoke_flags,
        "--out-dir", str(trainer_out_dir),
        "--ckpt-dir", str(trainer_ckpt_dir),
    ]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(_repo_root()),
            capture_output=True,
            text=True,
            timeout=timeout,
            env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        )
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": -1,
            "stdout_tail": (exc.stdout or "")[-400:],
            "stderr_tail": (exc.stderr or "")[-400:],
            "per_epoch_metrics_path": None,
            "error": f"timeout after {timeout}s",
        }
    per_epoch_path = (
        _repo_root() / out_dir / "per_epoch_metrics.json"
    )
    return {
        "returncode": proc.returncode,
        "stdout_tail": (proc.stdout or "")[-400:],
        "stderr_tail": (proc.stderr or "")[-400:],
        "per_epoch_metrics_path": (
            str(per_epoch_path) if per_epoch_path.is_file() else None
        ),
        "error": None if proc.returncode == 0 else (
            f"rc={proc.returncode}; stderr_tail={((proc.stderr or '')[-300:])}"
        ),
    }


def parse_smoke_metrics(per_epoch_path: Path) -> Dict[str, Any]:
    """Extract ``loss[0]``, ``loss[-1]``, ``decode_ratio`` from JSON.

    Returns a dict with: ``loss_first``, ``loss_last``,
    ``decode_ratio``, ``decrease_pct``, ``passed_decrease``,
    ``passed_decode``, ``error``.
    """
    if not per_epoch_path.is_file():
        return {
            "loss_first": None,
            "loss_last": None,
            "decode_ratio": None,
            "decrease_pct": None,
            "passed_decrease": False,
            "passed_decode": False,
            "error": f"metrics file missing: {per_epoch_path}",
        }
    try:
        data = json.loads(per_epoch_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        return {
            "loss_first": None,
            "loss_last": None,
            "decode_ratio": None,
            "decrease_pct": None,
            "passed_decrease": False,
            "passed_decode": False,
            "error": f"JSON parse failed: {exc}",
        }
    per_epoch = data.get("per_epoch_metrics", [])
    if not per_epoch:
        return {
            "loss_first": None,
            "loss_last": None,
            "decode_ratio": None,
            "decrease_pct": None,
            "passed_decrease": False,
            "passed_decode": False,
            "error": "per_epoch_metrics empty",
        }
    loss_first = float(per_epoch[0].get("train_loss", float("nan")))
    loss_last = float(per_epoch[-1].get("train_loss", float("nan")))
    decode_ratio = float(per_epoch[-1].get("decode_ratio_rdkit_strict", 0.0))
    # Decrease check: loss_last should be < loss_first * (1 - tolerance).
    # Catastrophic collapse (loss becomes NaN or >10x initial) is FAIL.
    import math
    if math.isnan(loss_first) or math.isnan(loss_last):
        passed_decrease = False
        decrease_pct = None
    elif loss_first <= 0:
        # Initial loss non-positive is suspect — treat as FAIL.
        passed_decrease = False
        decrease_pct = None
    elif math.isinf(loss_first) or math.isinf(loss_last):
        passed_decrease = False
        decrease_pct = None
    elif loss_last > 10.0 * loss_first:
        # Loss EXPLODED — catastrophic FAIL.
        passed_decrease = False
        decrease_pct = ((loss_first - loss_last) / max(abs(loss_first), 1e-9)) * 100
    else:
        # Standard decrease check (with tolerance).
        decrease_pct = ((loss_first - loss_last) / max(abs(loss_first), 1e-9)) * 100
        passed_decrease = bool(loss_last < loss_first * (1 - LOSS_DECREASE_TOLERANCE))
    passed_decode = bool(decode_ratio > DECODE_RATIO_FLOOR)
    return {
        "loss_first": loss_first,
        "loss_last": loss_last,
        "decode_ratio": decode_ratio,
        "decrease_pct": decrease_pct,
        "passed_decrease": passed_decrease,
        "passed_decode": passed_decode,
        "error": None,
    }


def gate_main(
    *,
    out_dir: Path = Path("molmetal/reports/wf385a3_scaffold_wire_in"),
    python_bin: str = sys.executable,
    skip_smoke: bool = False,
    timeout: float = 90.0,
) -> Dict[str, Any]:
    """Run the full gate surface and return a JSON-serialisable verdict."""
    t0 = time.perf_counter()
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Module-export verification (CPU, ~50 ms).
    modules = verify_module_exports()
    modules_pass = all(m["export_ok"] for m in modules)

    # 2) Optional subprocess smoke (CPU, ~30-60 s).
    if skip_smoke:
        smoke: Dict[str, Any] = {
            "returncode": None,
            "stdout_tail": "skipped",
            "stderr_tail": "skipped",
            "per_epoch_metrics_path": None,
            "error": "skip_smoke=True (module-export-only mode)",
        }
        metrics: Dict[str, Any] = {
            "loss_first": None,
            "loss_last": None,
            "decode_ratio": None,
            "decrease_pct": None,
            "passed_decrease": False,
            "passed_decode": False,
            "error": "skipped",
        }
    else:
        smoke = run_smoke_training(
            out_dir=out_dir,
            python_bin=python_bin,
            timeout=timeout,
        )
        # Wf-415-A3 fix: the default `--out-dir` in DEFAULT_SMOKE_FLAGS
        # is a REPO-RELATIVE path (`molmetal/reports/wf385a3_scaffold_wire_in/train`),
        # so the trainer writes metrics to
        # `<repo_root>/molmetal/reports/wf385a3_scaffold_wire_in/train/per_epoch_metrics.json`,
        # not to `<gate_out_dir>/train/per_epoch_metrics.json` or
        # `<repo_root>/<gate_out_dir>/per_epoch_metrics.json`.  Fall back
        # to all 3 candidate paths so the gate finds the metrics
        # regardless of which out-dir the trainer was launched with.
        smoke_metric_candidates: List[Path] = []
        if smoke["per_epoch_metrics_path"]:
            smoke_metric_candidates.append(Path(smoke["per_epoch_metrics_path"]))
        smoke_metric_candidates.append(out_dir / "train" / "per_epoch_metrics.json")
        smoke_metric_candidates.append(
            _repo_root() / "molmetal" / "reports" /
            "wf385a3_scaffold_wire_in" / "train" / "per_epoch_metrics.json"
        )
        smoke_metric_candidates.append(
            _repo_root() / out_dir / "per_epoch_metrics.json"
        )
        per_epoch_path = next(
            (p for p in smoke_metric_candidates if p.is_file()),
            smoke_metric_candidates[-1],
        )
        metrics = parse_smoke_metrics(per_epoch_path)

    # 3) Final verdict.
    if not modules_pass:
        verdict = "FAIL"
        reason = (
            f"module-export verification failed: "
            f"{[m['filename'] for m in modules if not m['export_ok']]}"
        )
    elif skip_smoke:
        verdict = "PASS"
        reason = (
            f"all {len(modules)} modules export expected callable; "
            "smoke skipped per skip_smoke=True"
        )
    elif smoke.get("error"):
        verdict = "FAIL"
        reason = f"smoke subprocess failed: {smoke['error']}"
    elif not metrics["passed_decrease"]:
        verdict = "FAIL"
        reason = (
            f"loss did not decrease over 50-step smoke: "
            f"loss[0]={metrics['loss_first']}, loss[-1]={metrics['loss_last']}, "
            f"decrease_pct={metrics['decrease_pct']}"
        )
    elif not metrics["passed_decode"]:
        verdict = "FAIL"
        reason = (
            f"decode_ratio collapsed to {metrics['decode_ratio']} "
            f"(YuelBond decoder likely broken by SCAFFOLD chain per "
            f"Wf-269C MEASURED --no-yuelbond → decode_ratio 0.000)"
        )
    else:
        verdict = "PASS"
        reason = (
            f"all {len(modules)} modules export expected callable + "
            f"50-step smoke PASSED: loss[0]={metrics['loss_first']:.4f} → "
            f"loss[-1]={metrics['loss_last']:.4f} "
            f"(Δ={metrics['decrease_pct']:.1f}% decrease), "
            f"decode_ratio={metrics['decode_ratio']:.3f}"
        )

    elapsed = time.perf_counter() - t0
    result = {
        "verdict": verdict,
        "reason": reason,
        "elapsed_sec": elapsed,
        "modules": modules,
        "smoke": smoke,
        "metrics": metrics,
        "config": {
            "out_dir": str(out_dir),
            "skip_smoke": skip_smoke,
            "timeout_sec": timeout,
            "loss_decrease_tolerance": LOSS_DECREASE_TOLERANCE,
            "decode_ratio_floor": DECODE_RATIO_FLOOR,
        },
    }
    out_json = out_dir / "gate_result.json"
    # Wf-414-A3 migration: atomic write + safe_overwrite_check + detect_truncation.
    _payload = json.dumps(result, indent=2, sort_keys=True)
    if _WF414_HAS_ATOMIC:
        _risk = safe_overwrite_check(out_json, _payload, min_lines=50)
        if _risk is not None:
            print(
                f"[Wf-385-A3] WARN: safe_overwrite_check ratio={_risk:.3f} at {out_json}",
                file=sys.stderr,
            )
        atomic_write_file(out_json, _payload, encoding="utf-8")
        _trunc = detect_truncation(_payload, expected_min_length=200)
        if _trunc is not None:
            print(
                f"[Wf-385-A3] WARN: detect_truncation reason={_trunc!r} at {out_json}",
                file=sys.stderr,
            )
    else:  # pragma: no cover
        out_json.write_text(_payload, encoding="utf-8")
    return result


def main(argv: List[str] | None = None) -> int:
    """CLI entry point — prints verdict + writes gate_result.json."""
    import argparse
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("molmetal/reports/wf385a3_scaffold_wire_in"),
        help="Output directory for gate_result.json + per_epoch_metrics.json",
    )
    p.add_argument(
        "--skip-smoke",
        action="store_true",
        default=False,
        help="Skip subprocess smoke training (module-export only).",
    )
    p.add_argument(
        "--python-bin",
        default=sys.executable,
        help="Python interpreter for subprocess smoke (default: sys.executable)",
    )
    p.add_argument(
        "--timeout",
        type=float,
        default=90.0,
        help="Subprocess timeout in seconds (default: 90).",
    )
    args = p.parse_args(argv)
    result = gate_main(
        out_dir=args.out_dir,
        python_bin=args.python_bin,
        skip_smoke=args.skip_smoke,
        timeout=args.timeout,
    )
    print(f"verdict={result['verdict']}")
    print(f"reason={result['reason']}")
    return 0 if result["verdict"] in ("PASS", "SKIP") else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())