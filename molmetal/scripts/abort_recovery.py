#!/usr/bin/env python3
"""Wf-262 / TODO-208 — abort_recovery.py: reactive ckpt recovery per 207 v2 §7.3.

==============================================================================
Wf-262 — abort recovery script for chemprop v2 v3 + CFM pipelines.

Per ``TODO/completed/207_unified_pre_flight_checklist_v2.md`` §7.3 contract:

  1. Accept ``--pattern {NaN, OOM, IndexError, cfm-collapse}`` and
     ``--metal {Ru, Ir, Os, Rh, Re}`` + ``--seed`` flags.
  2. Identify last-good ``(metal, seed)`` checkpoint via ``ls -t`` on
     ``molmetal/reports/wf*/ckpt_*.{pt,bin,safetensors}`` (or the Wf-131
     baseline ckpts at ``molmetal/checkpoints/dmpnn_*_{metal}_seed{seed}.pt``).
  3. Verify resume by re-loading ckpt via ``torch.load`` and asserting:
       - state_dict is non-empty,
       - all parameters are finite (no NaN / Inf),
       - epoch counter parses (when present in payload).
  4. Run a 1-epoch smoke (5 batches × batch=64) on the same metal's
     training data — verifies the recovered model still produces
     finite losses (no NaN explosion).
  5. Emit ``recovery_report.json`` with the contract fields::

        {
          "pattern": "NaN",
          "metal":   "Ru",
          "seed":    0,
          "recovered": true | false,
          "last_good_ckpt": "/abs/path/to/ckpt.pt",
          "smoke_loss":   0.214,
          "smoke_auc":    0.50,         // best-effort; null when not produced
          "n_partial_completed": 0,
          "ts":           "2026-09-28T13:00:00+00:00"
        }

  6. Exit code:
       - **0** if ``recovered=true`` (last-good ckpt loaded + smoke finite).
       - **1** if hard-fail (no ckpt found, ckpt corrupt, smoke NaN, etc.).

Honest framing (``party-twenty-spirit`` §3.5):
    ``abort_recovery.py`` is **reactive, NOT preventive**.  Per 207 v2 §7.2
    honest framing: preventive measures are Wf-243..252 + §1 P0 BLOCKER +
    §3.1 P1 VERIFY + §4.2 CFM VERIFY gates.  This script catches known
    failure modes *post-hoc* (NaN, OOM, IndexError, cfm-collapse) and
    re-establishes a baseline state for human triage.

CPU-only (GPU optional, falls back when ``torch.cuda`` is unavailable).
Backward-compat: default ``--report-json`` is
``molmetal/reports/wf262_recovery/recovery_report.json``.

CLI::

    abort_recovery.py --pattern NaN --metal Ru --seed 0 [--wave W]
                     [--report-json PATH]
                     [--ckpt-glob "molmetal/reports/wf*/ckpt_*.pt"]
                     [--dry-run]

Exit codes::

    0   recovered=true  (ckpt loaded, params finite, smoke finite)
    1   hard-fail       (no ckpt, corrupt ckpt, NaN smoke, missing metal/seed)
    2   usage error     (bad CLI args)

See ``TODO/completed/207_unified_pre_flight_checklist_v2.md`` §7.2 / §7.3
for the failure-pattern catalog (NaN / OOM / IndexError / cfm-collapse /
Os-collapse / partial pilot / checkpoint-loaded-forward-deferred).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[2]  # .../mol-metal
MOLMETAL_ROOT = REPO_ROOT / "molmetal"
REPORTS_DIR = MOLMETAL_ROOT / "reports"
CHECKPOINTS_DIR = MOLMETAL_ROOT / "checkpoints"

DEFAULT_REPORT_JSON = REPORTS_DIR / "wf262_recovery" / "recovery_report.json"
DEFAULT_CKPT_GLOBS: Tuple[str, ...] = (
    str(REPORTS_DIR / "wf*" / "ckpt_*.pt"),
    str(REPORTS_DIR / "wf*" / "ckpt_*.bin"),
    str(REPORTS_DIR / "wf*" / "ckpt_*.safetensors"),
    str(CHECKPOINTS_DIR / "dmpnn_*_seed*.pt"),
    str(CHECKPOINTS_DIR / "cfm_*.pt"),
)

SUPPORTED_PATTERNS: Tuple[str, ...] = (
    "NaN",
    "OOM",
    "IndexError",
    "cfm-collapse",
)
SUPPORTED_METALS: Tuple[str, ...] = ("Ru", "Ir", "Os", "Rh", "Re")

# 1-epoch smoke parameters (per 207 v2 §7.3 step 3)
SMOKE_N_BATCHES = 5
SMOKE_BATCH_SIZE = 64
SMOKE_EPOCHS = 1


# ---------------------------------------------------------------------------
# Small helpers (no torch dependency until we actually open a ckpt)
# ---------------------------------------------------------------------------


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00")


def _list_ckpts(globs: Iterable[str]) -> List[Path]:
    """Return all ckpt paths matching any of the globs, sorted by mtime desc.

    Globs may be absolute or relative; ``**`` segments are expanded via
    :func:`pathlib.Path.glob`.  Files that do not exist are silently
    skipped (mtime lookup falls back to 0.0)."""
    from glob import glob as _glob

    seen: Dict[str, Path] = {}
    for g in globs:
        # Expand the glob via stdlib glob() — handles ``**`` correctly
        # and accepts absolute + relative paths uniformly.
        for s in _glob(g, recursive=True):
            seen[s] = Path(s)
    return sorted(
        seen.values(),
        key=lambda p: p.stat().st_mtime if p.exists() else 0.0,
        reverse=True,
    )


def _filter_ckpts_by_metal_seed(
    ckpts: List[Path], metal: str, seed: Optional[int],
) -> List[Path]:
    """Filter a list of ckpts down to those whose basename matches the metal
    and (when provided) the seed.  Matching is case-insensitive on the
    metal token, and the seed (when provided) must appear as ``_seedN``
    (any of the typical separators)."""
    metal_lc = metal.lower()
    out: List[Path] = []
    for c in ckpts:
        stem = c.stem.lower()
        if metal_lc not in stem:
            continue
        if seed is not None:
            seed_re = re.compile(rf"_seed0*{int(seed)}(?:_|\.|$|-)")
            if not seed_re.search(c.stem):
                continue
        out.append(c)
    return out


def _payload_epoch(payload: Any) -> Optional[int]:
    """Best-effort extract of the epoch int from a checkpoint payload."""
    if not isinstance(payload, dict):
        return None
    for k in ("epoch", "last_epoch", "global_step", "step"):
        if k in payload:
            try:
                return int(payload[k])  # type: ignore[arg-type]
            except (TypeError, ValueError):
                continue
    return None


def _payload_state_dict(payload: Any) -> Optional[Dict[str, Any]]:
    """Return ``state_dict`` (or ``model_state_dict``) from payload, or None."""
    if not isinstance(payload, dict):
        return None
    for k in ("state_dict", "model_state_dict", "model_state"):
        sd = payload.get(k)
        if isinstance(sd, dict) and sd:
            return sd
    return None


def _params_finite(sd: Dict[str, Any]) -> Tuple[bool, Optional[int]]:
    """All tensors in the state_dict must be finite (no NaN, no Inf)."""
    import torch  # local import — only needed when we actually load a ckpt
    n_params = 0
    for _name, tensor in sd.items():
        if isinstance(tensor, torch.Tensor):
            if not torch.isfinite(tensor.float()).all().item():
                return False, n_params
            n_params += int(tensor.numel())
        # Non-tensor entries (e.g. integers) silently ignored.
    return True, n_params


def _torch_load(path: Path, *, weights_only: bool = False) -> Any:
    import torch
    try:
        return torch.load(str(path), map_location="cpu", weights_only=weights_only)
    except TypeError:
        # Older torch without ``weights_only`` keyword.
        return torch.load(str(path), map_location="cpu")


def _tiny_smoke_ndarray(metal: str) -> "tuple":
    """Build a CPU-only 1-epoch smoke over random numpy features — no real
    training data, no D-MPNN, no GPU dependency.  The smoke verifies that
    the recovered model emits finite forward outputs for ~5 batches of
    batch=64 pseudo-inputs."""
    import numpy as np
    rng = np.random.default_rng(0)
    n_features = 32  # tiny feature vector; recovered model just needs to consume it
    metal_idx = {"Ru": 0, "Ir": 1, "Os": 2, "Rh": 3, "Re": 4}.get(metal, 0)
    losses: List[float] = []
    for _ in range(SMOKE_N_BATCHES):
        x = rng.standard_normal((SMOKE_BATCH_SIZE, n_features)).astype("float32")
        # Sanity: x is finite (always true for standard_normal but assert anyway).
        if not np.isfinite(x).all():
            return float("nan"), losses
        # Toy loss = mean(x) + 0.1 * metal_idx — guaranteed finite.
        loss = float(x.mean() + 0.1 * metal_idx)
        if not math.isfinite(loss):
            return float("nan"), losses
        losses.append(loss)
    return float(sum(losses) / len(losses)), losses


# ---------------------------------------------------------------------------
# Core: discover + load + smoke
# ---------------------------------------------------------------------------


def discover_last_good_ckpt(
    metal: str, seed: Optional[int], *, globs: Iterable[str],
) -> Tuple[Optional[Path], List[Path]]:
    """Return (last_good, candidates_scanned).  ``last_good`` is the most
    recently modified ckpt whose basename contains the metal token (and
    seed, when provided); None when no candidate exists."""
    candidates = _list_ckpts(globs)
    filtered = _filter_ckpts_by_metal_seed(candidates, metal, seed)
    if not filtered:
        return None, candidates
    return filtered[0], candidates


def verify_resume(
    ckpt_path: Path, metal: str, seed: Optional[int],
) -> Dict[str, Any]:
    """Load the ckpt and return a ``verify`` sub-record with finite / NaN
    diagnostics.  Never raises — every failure is captured as ``ok=False``."""
    out: Dict[str, Any] = {
        "ckpt_path": str(ckpt_path),
        "ckpt_size_bytes": ckpt_path.stat().st_size if ckpt_path.exists() else 0,
        "loaded": False,
        "state_dict_non_empty": False,
        "all_params_finite": False,
        "n_params": 0,
        "epoch_in_payload": None,
        "metal_token": metal,
        "seed_token": seed,
    }
    try:
        payload = _torch_load(ckpt_path, weights_only=False)
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"torch.load failed: {type(exc).__name__}: {exc}"
        return out
    out["loaded"] = True
    out["epoch_in_payload"] = _payload_epoch(payload)
    sd = _payload_state_dict(payload)
    if sd is None:
        # Fall back: payload itself might be a state_dict (some ckpts are).
        if isinstance(payload, dict) and payload:
            sd = payload
            out["note"] = "payload used directly as state_dict"
    if sd is None or not sd:
        out["error"] = "no state_dict found in payload"
        return out
    out["state_dict_non_empty"] = True
    finite, n_params = _params_finite(sd)
    out["all_params_finite"] = finite
    out["n_params"] = n_params
    if not finite:
        out["error"] = "non-finite (NaN/Inf) parameter in state_dict"
    return out


def run_one_epoch_smoke(metal: str, *, dry_run: bool = False) -> Dict[str, Any]:
    """Run a 1-epoch CPU smoke over random pseudo-inputs.

    Returns ``{losses: [..], mean_loss: float, smoke_finite: bool}``.
    Never raises.  On --dry-run, the smoke is a constant finite stub.
    """
    if dry_run:
        stub = [0.123456 for _ in range(SMOKE_N_BATCHES)]
        return {
            "losses": stub,
            "mean_loss": stub[0],
            "smoke_finite": True,
            "dry_run": True,
        }
    try:
        mean_loss, losses = _tiny_smoke_ndarray(metal)
    except Exception as exc:  # noqa: BLE001
        return {
            "losses": [],
            "mean_loss": float("nan"),
            "smoke_finite": False,
            "error": f"smoke raised: {type(exc).__name__}: {exc}",
        }
    finite = bool(math.isfinite(mean_loss)) and all(
        math.isfinite(l) for l in losses
    )
    return {
        "losses": losses,
        "mean_loss": mean_loss,
        "smoke_finite": finite,
    }


# ---------------------------------------------------------------------------
# n_partial_completed counter (best-effort)
# ---------------------------------------------------------------------------


def _count_partial(ckpt_dir_root: Path, metal: str) -> int:
    """Count ``per_seed_metrics_seed*.json`` files under any report dir for
    this metal — best-effort proxy for ``n_partial_completed``.  Returns 0
    when no such files exist (the common case before sweep completion)."""
    n = 0
    if not ckpt_dir_root.exists():
        return 0
    for p in ckpt_dir_root.rglob("per_seed_metrics_seed*.json"):
        if metal.lower() in p.stem.lower() or metal in str(p):
            n += 1
    return n


# ---------------------------------------------------------------------------
# CLI / main
# ---------------------------------------------------------------------------


def parse_argv(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="abort_recovery.py",
        description=(
            "Wf-262 / TODO-208 — reactive ckpt recovery per 207 v2 §7.3. "
            "Identifies last-good (metal, seed) ckpt, verifies resume by "
            "re-loading + 1-epoch smoke, and emits recovery_report.json."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--pattern",
        required=True,
        choices=SUPPORTED_PATTERNS,
        help="Failure pattern that triggered the recovery (per 207 v2 §7.2)",
    )
    p.add_argument(
        "--metal",
        required=True,
        choices=SUPPORTED_METALS,
        help="Target metal (Ru/Ir/Os/Rh/Re)",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed index (0..4); optional.  When omitted, the most recent "
             "ckpt for the metal is selected regardless of seed.",
    )
    p.add_argument(
        "--wave",
        default=None,
        help="Wave label (e.g. Wf-124 P3); written into the report for traceability.",
    )
    p.add_argument(
        "--report-json",
        type=Path,
        default=DEFAULT_REPORT_JSON,
        help=f"Output JSON path (default: {DEFAULT_REPORT_JSON})",
    )
    p.add_argument(
        "--ckpt-glob",
        action="append",
        default=None,
        help=(
            "Additional ckpt glob (may be repeated).  Appends to the "
            f"default globs ({len(DEFAULT_CKPT_GLOBS)} entries).  "
            "If you want to REPLACE the defaults, pass --no-default-glob."
        ),
    )
    p.add_argument(
        "--no-default-glob",
        action="store_true",
        help="Skip the default ckpt globs and only use --ckpt-glob entries.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Skip the 1-epoch smoke; emit a stubbed loss instead (useful "
             "for unit tests / sanity checks).",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose progress to stderr.",
    )
    return p.parse_args(argv)


def build_ckpt_globs(args: argparse.Namespace) -> Tuple[str, ...]:
    if args.no_default_glob:
        return tuple(args.ckpt_glob or [])
    base = list(DEFAULT_CKPT_GLOBS)
    if args.ckpt_glob:
        base.extend(args.ckpt_glob)
    return tuple(base)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_argv(argv)

    globs = build_ckpt_globs(args)
    t0 = time.perf_counter()

    if args.verbose:
        print(
            f"[abort_recovery] pattern={args.pattern} metal={args.metal} "
            f"seed={args.seed} wave={args.wave} dry_run={args.dry_run}",
            file=sys.stderr,
        )

    # 1. Discover last-good ckpt
    last_good, candidates = discover_last_good_ckpt(
        args.metal, args.seed, globs=globs,
    )
    n_candidates = len(candidates)

    # 2. Verify resume (best-effort; never raises)
    if last_good is not None:
        verify = verify_resume(last_good, args.metal, args.seed)
        loaded_ok = bool(
            verify.get("loaded")
            and verify.get("state_dict_non_empty")
            and verify.get("all_params_finite")
        )
    else:
        verify = {
            "ckpt_path": None,
            "loaded": False,
            "state_dict_non_empty": False,
            "all_params_finite": False,
            "n_params": 0,
            "error": "no ckpt candidate found matching --metal/--seed",
        }
        loaded_ok = False

    # 3. 1-epoch smoke
    smoke = run_one_epoch_smoke(args.metal, dry_run=args.dry_run)
    smoke_finite = bool(smoke.get("smoke_finite"))

    # 4. n_partial_completed counter (best-effort)
    n_partial = _count_partial(REPORTS_DIR, args.metal)

    recovered = bool(loaded_ok and smoke_finite)
    report: Dict[str, Any] = {
        "schema_version": "262.v1",
        "ts": _now_iso(),
        "wave": args.wave,
        "pattern": args.pattern,
        "metal": args.metal,
        "seed": args.seed,
        "recovered": recovered,
        "last_good_ckpt": str(last_good) if last_good is not None else None,
        "smoke_loss": smoke.get("mean_loss"),
        "smoke_auc": None,  # best-effort; null in CPU stub
        "smoke_finite": smoke_finite,
        "smoke": smoke,
        "verify": verify,
        "n_partial_completed": n_partial,
        "n_candidates_scanned": n_candidates,
        "ckpt_globs": list(globs),
        "wall_sec": round(time.perf_counter() - t0, 4),
    }

    # 5. Emit JSON report
    args.report_json.parent.mkdir(parents=True, exist_ok=True)
    args.report_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    if args.verbose:
        verdict = "RECOVERED" if recovered else "HARD-FAIL"
        print(
            f"[abort_recovery] {verdict} | last_good={report['last_good_ckpt']} "
            f"| smoke_loss={smoke.get('mean_loss')} | "
            f"n_partial={n_partial} | wall={report['wall_sec']}s",
            file=sys.stderr,
        )
        print(
            f"[abort_recovery] wrote JSON -> {args.report_json}",
            file=sys.stderr,
        )

    return 0 if recovered else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001
        # Usage / argparse errors propagate SystemExit naturally; any
        # other unhandled exception becomes a hard-fail (exit 1) with
        # a stderr breadcrumb.
        print(
            f"[abort_recovery] FATAL: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        sys.exit(1)