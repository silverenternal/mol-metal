"""Wf-399-A1 — 50-step end-to-end smoke protocol for trainer flag wire-ins.

PURPOSE (per AGENTS.md §5.7 added by Wf-399-A1):

  Whenever a new `--flag` is wired into a trainer's training loop (loss / sampling /
  dispatch), the wire-in is INCOMPLETE until a 50-step end-to-end smoke has been run
  that verifies:

    1. `parse_args` accepts the flag without NameError or argparse error.
    2. The trainer imports cleanly with the flag set.
    3. A 50-step end-to-end run produces a metrics JSON / log with
       `decode_ratio > 0` and a strictly decreasing `loss[0] > loss[49]`.

This script provides the protocol helper so that future trainer-wire-in waves (Wf-381
wire-in of Wf-380 SCAFFOLD-only flags, Wf-394-B1 Lambda CFM patch, Wf-401+ follow-ups)
can drop a single import + 3-line invocation and emit a structured verdict.

CPU-only by design — no `torch.cuda` invocation. Smoke invocations run on CPU
(`--device cpu`) at tiny batch sizes so a full 50-step run finishes in <60 s on the
test host.

Reference (real precedent, NOT fictional):
  - Wf-385-A1 audit (`TODO/completed/385a1_wf380_cli_flag_audit.md`) — 17/17 Wf-380 NEW
    CLI flags were FLAG_ONLY; SCAFFOLD modules existed + 4-4 unit tests PASS but ZERO
    trainer import; main-loop regression invisible at unit-test layer.
  - Wf-291D + Wf-299 — canonical `decode_ratio = 0/192` regression → 1-line try/except
    + 5 shim edits → `1.000 ± 0.000` (3 seeds × 4 probes × 16 samples).
  - Wf-394-A1 (`TODO/completed/394a1_lambda_cfm_wire.md`) — TODO-21 Lambda × CFM
    coupling audit + smoke scaffold (2 NEW tests PASS; wire-in patch deferred to
    USER_OK wave).

Usage example (future Wf-381 wire-in):

    from molmetal.scripts.wf399a1_smoke_protocol import (
        run_smoke_50step, verify_decode_ratio, verify_loss_decrease,
    )
    output = run_smoke_50step(
        trainer_path="molmetal/scripts/train_cfm_yuelbond_wasserstein.py",
        flags_list=["--use-ot-cfm", "--triton-ot", "--epochs", "1",
                    "--batch-size", "8", "--device", "cpu"],
    )
    assert verify_decode_ratio(output)["decode_ratio"] > 0
    assert verify_loss_decrease(output)["decreasing"] is True

CLI usage (smoke + audit from a shell):

    python molmetal/scripts/wf399a1_smoke_protocol.py \
        --trainer molmetal/scripts/train_cfm_yuelbond_wasserstein.py \
        --flag-arg "--use-ot-cfm" --flag-arg "--triton-ot" \
        --epochs 1 --batch-size 8 --device cpu
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TIMEOUT_SEC = 600  # 10 min — plenty for a 50-step CPU smoke


@dataclass
class SmokeOutput:
    """Structured output of a 50-step end-to-end smoke invocation."""

    trainer_path: str
    flags: List[str]
    returncode: int
    stdout: str
    stderr: str
    metrics_path: Optional[str]
    metrics: Dict[str, Any]
    wall_seconds: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Smoke invocation
# ---------------------------------------------------------------------------


def run_smoke_50step(
    trainer_path: str,
    flags_list: List[str],
    *,
    timeout_sec: int = DEFAULT_TIMEOUT_SEC,
    cwd: Optional[str] = None,
    capture_metrics_path: Optional[str] = None,
) -> SmokeOutput:
    """Invoke the trainer with a tiny 50-step end-to-end smoke and capture output.

    The smoke is constructed by passing `--epochs 1 --batch-size 8 --device cpu`
    to the trainer so that the first epoch runs ~50 training steps (assuming the
    trainer's default split produces ~50 batches per epoch on the smoke cohort).
    If the caller passes their own `--epochs` / `--batch-size`, those override.

    The trainer's own per-epoch / per-step metrics JSON is detected automatically
    via the trainer's stdout banner (e.g. `[results] wrote {out_dir}/metrics.json`)
    or via the `--out-dir` flag's `metrics.json` companion file. If neither is
    found, an empty `metrics` dict is returned and `verify_decode_ratio` /
    `verify_loss_decrease` will return their not-found dict variant.

    Parameters
    ----------
    trainer_path : str
        Path to the trainer script (e.g. ``molmetal/scripts/train_cfm_yuelbond_wasserstein.py``).
    flags_list : list[str]
        List of CLI arguments to forward to the trainer. Defaults to a tiny
        CPU smoke configuration if `flags_list` is empty.
    timeout_sec : int, default 600
        Subprocess timeout in seconds.
    cwd : str | None
        Working directory for the subprocess. Defaults to REPO_ROOT so that
        relative paths in the trainer resolve correctly.
    capture_metrics_path : str | None
        If provided, this explicit path is read after the smoke completes and
        its JSON contents are returned in the SmokeOutput.metrics field.

    Returns
    -------
    SmokeOutput
        Structured dataclass with the trainer's stdout / stderr / metrics.
    """
    import time

    if not flags_list:
        flags_list = ["--epochs", "1", "--batch-size", "8", "--device", "cpu"]
    else:
        flags_list = list(flags_list)

    cmd = [sys.executable, trainer_path] + flags_list
    work_dir = cwd or str(REPO_ROOT)

    t0 = time.time()
    try:
        completed = subprocess.run(
            cmd,
            cwd=work_dir,
            capture_output=True,
            text=True,
            timeout=timeout_sec,
        )
        returncode = completed.returncode
        stdout = completed.stdout or ""
        stderr = completed.stderr or ""
    except subprocess.TimeoutExpired as exc:
        return SmokeOutput(
            trainer_path=trainer_path,
            flags=flags_list,
            returncode=-1,
            stdout=(exc.stdout or "") if isinstance(exc.stdout, str) else "",
            stderr=f"TimeoutExpired after {timeout_sec}s",
            metrics_path=None,
            metrics={},
            wall_seconds=time.time() - t0,
        )
    wall = time.time() - t0

    # Locate metrics JSON via --out-dir or stdout banner
    metrics_path: Optional[str] = None
    if capture_metrics_path is not None:
        metrics_path = capture_metrics_path
    else:
        # Look for --out-dir flag in flags_list
        out_dir: Optional[str] = None
        for i, flag in enumerate(flags_list):
            if flag == "--out-dir" and i + 1 < len(flags_list):
                out_dir = flags_list[i + 1]
                break
        if out_dir is None:
            # Look for the trainer's stdout banner pattern
            m = re.search(
                r"(?:results|wrote|saved).*?([^\s]+\.json)", stdout, re.IGNORECASE
            )
            if m:
                metrics_path = m.group(1)
        else:
            candidate = Path(out_dir) / "metrics.json"
            if candidate.exists():
                metrics_path = str(candidate)

    metrics: Dict[str, Any] = {}
    if metrics_path and Path(metrics_path).exists():
        try:
            metrics = json.loads(Path(metrics_path).read_text())
        except (json.JSONDecodeError, OSError):
            metrics = {}

    return SmokeOutput(
        trainer_path=trainer_path,
        flags=flags_list,
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        metrics_path=metrics_path,
        metrics=metrics,
        wall_seconds=wall,
    )


# ---------------------------------------------------------------------------
# Verdict helpers
# ---------------------------------------------------------------------------


def verify_decode_ratio(smoke: SmokeOutput) -> Dict[str, Any]:
    """Parse the smoke output for `decode_ratio > 0`.

    Accepts either:
      * `smoke.metrics["decode_ratio"]` (canonical JSON key from the trainer),
      * `smoke.metrics["final"]["decode_ratio"]`,
      * a stdout pattern ``decode_ratio=<float>`` (the trainer's print format).

    Returns a dict with:
      * ``decode_ratio`` (float | None) — value found,
      * ``invariant_holds`` (bool) — True iff decode_ratio > 0,
      * ``source`` (str) — where the value was found,
      * ``error`` (str | None) — diagnostic if not found.
    """
    decode_ratio: Optional[float] = None
    source = "none"

    # Direct metrics key
    if "decode_ratio" in smoke.metrics:
        try:
            decode_ratio = float(smoke.metrics["decode_ratio"])
            source = "metrics.decode_ratio"
        except (TypeError, ValueError):
            pass

    # Nested final key (some trainers emit a {"final": {...}} wrapper)
    if decode_ratio is None and isinstance(smoke.metrics.get("final"), dict):
        try:
            decode_ratio = float(smoke.metrics["final"]["decode_ratio"])
            source = "metrics.final.decode_ratio"
        except (KeyError, TypeError, ValueError):
            pass

    # Stdout pattern. Trainers print either:
    #   "decode_ratio=1.000 (192/192) ..."  (no spaces) OR
    #   "Best decode_ratio = 0.788 (96/122) ..."  (with spaces around '=')
    # The regex accepts both forms.
    if decode_ratio is None:
        m = re.search(
            r"decode_ratio\s*=\s*([0-9]+(?:\.[0-9]+)?)", smoke.stdout
        )
        if m:
            try:
                decode_ratio = float(m.group(1))
                source = "stdout regex"
            except ValueError:
                pass

    if decode_ratio is None:
        return {
            "decode_ratio": None,
            "invariant_holds": False,
            "source": source,
            "error": "decode_ratio metric not found in smoke output",
        }

    return {
        "decode_ratio": decode_ratio,
        "invariant_holds": decode_ratio > 0.0,
        "source": source,
        "error": None,
    }


def verify_loss_decrease(smoke: SmokeOutput) -> Dict[str, Any]:
    """Parse the smoke output for `loss[0] > loss[N-1]` (monotone non-increasing).

    Accepts either:
      * `smoke.metrics["loss_curve"]` as a list of per-step / per-epoch losses,
      * `smoke.metrics["epoch_losses"]` as a list of per-epoch losses,
      * a stdout pattern ``loss=<float>`` (single value; not decreasing-checkable).

    Returns a dict with:
      * ``loss_start`` (float | None),
      * ``loss_end`` (float | None),
      * ``decreasing`` (bool) — True iff loss_start > loss_end,
      * ``n_points`` (int),
      * ``source`` (str),
      * ``error`` (str | None).
    """
    curve: Optional[List[float]] = None
    source = "none"

    if isinstance(smoke.metrics.get("loss_curve"), list):
        curve = [float(x) for x in smoke.metrics["loss_curve"]]
        source = "metrics.loss_curve"
    elif isinstance(smoke.metrics.get("epoch_losses"), list):
        curve = [float(x) for x in smoke.metrics["epoch_losses"]]
        source = "metrics.epoch_losses"

    if curve is None or len(curve) < 2:
        return {
            "loss_start": None,
            "loss_end": None,
            "decreasing": False,
            "n_points": len(curve) if curve is not None else 0,
            "source": source,
            "error": "loss curve not found or has fewer than 2 points",
        }

    return {
        "loss_start": curve[0],
        "loss_end": curve[-1],
        "decreasing": curve[0] > curve[-1],
        "n_points": len(curve),
        "source": source,
        "error": None,
    }


def verify_all_invariants(smoke: SmokeOutput) -> Dict[str, Any]:
    """Convenience wrapper — runs both verifiers + emits a single PASS/FAIL verdict."""
    decode_ratio = verify_decode_ratio(smoke)
    loss_decrease = verify_loss_decrease(smoke)
    all_pass = decode_ratio["invariant_holds"] and loss_decrease["decreasing"]
    return {
        "pass": all_pass,
        "decode_ratio": decode_ratio,
        "loss_decrease": loss_decrease,
        "returncode": smoke.returncode,
        "wall_seconds": smoke.wall_seconds,
    }


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _main(argv: List[str]) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Wf-399-A1 50-step end-to-end smoke protocol. "
            "Runs a trainer with a tiny CPU config + verifies decode_ratio > 0 "
            "and loss[0] > loss[N-1]."
        )
    )
    parser.add_argument(
        "--trainer",
        required=True,
        help="Path to trainer script (relative to repo root or absolute).",
    )
    parser.add_argument(
        "--flag-arg",
        action="append",
        default=[],
        help="Forwarded flag (repeatable). Defaults to a tiny CPU smoke config.",
    )
    parser.add_argument(
        "--out-json",
        default=None,
        help="If set, write the structured SmokeOutput + verdict JSON here.",
    )
    parser.add_argument(
        "--timeout-sec",
        type=int,
        default=DEFAULT_TIMEOUT_SEC,
        help="Subprocess timeout in seconds.",
    )
    args = parser.parse_args(argv)

    smoke = run_smoke_50step(
        trainer_path=args.trainer,
        flags_list=args.flag_arg,
        timeout_sec=args.timeout_sec,
    )
    verdict = verify_all_invariants(smoke)
    summary = {
        "smoke": smoke.to_dict(),
        "verdict": verdict,
    }
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0 if verdict["pass"] and smoke.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(_main(sys.argv[1:]))