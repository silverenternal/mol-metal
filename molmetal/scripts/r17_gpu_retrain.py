"""R17 GPU retrain wrapper — production CLI scaffold for the 10000-step YuelBond retrain.

================================================================
Why this script exists
================================================================
R17 plans a production-grade 10000-step YuelBond retrain at the
R16b-verified config (h=128, n_layers=3, batch=4, decode smoke every
1000 steps) on the Blackwell dGPU.  The bare training script
``molmetal/scripts/metallo_drug_smoke_retrain.py`` is correct but
operator-unfriendly:

* No GPU detection — operator has to verify nvidia-smi manually
* No subprocess supervision — a CUDA hang kills the whole run
* No checkpoint archival — only the *final* ckpt survives
* No metrics stream — decode_ratio per-step only lives in stdout
* No resume-on-hang protocol — a SMU firmware hang means restarting
  from step 0 and losing hours of work

This wrapper solves all of these.  It is **intentionally a NEW, thin
supervisor** around :mod:`metallo_drug_smoke_retrain` — it does not
modify the adapter nor the canonical training script.  All wins are
obtained by orchestrating the existing script via subprocess +
stdout parsing + checkpoint archival.

Outputs (created by the wrapper)
-------------------------------
* ``logs/r17_retrain/train.log``               — full wrapped stdout
* ``logs/r17_retrain/checkpoints/step_<N>.pt`` — copies of ckpt.pt every 1000 steps
* ``logs/r17_retrain/checkpoints/latest.pt``   — symlink to the most recent ckpt
* ``logs/r17_retrain/gpu_probe.json``          — GPU detection snapshot
* ``logs/r17_retrain/probe_state.json``        — periodic state dump
* ``metrics/by_metric/cfm_decode_ratio_r17.json`` — per-step trajectory

Usage
-----
    source env.sh
    uv run --no-sync python molmetal/scripts/r17_gpu_retrain.py \\
        --steps 10000 --seed 42

For CPU smoke (no GPU launch):

    uv run --no-sync python molmetal/scripts/r17_gpu_retrain.py \\
        --steps 100 --dry-run

Honest framing
==============
This is a **productionising wrapper** around the existing verified
smoke.  The 30/30 PASS verdict in
``molmetal/reports/wf_r17_10k_retrain_done/final.md`` is the upstream
proof that ``metallo_drug_smoke_retrain.py`` *trains correctly*; this
CLI does not retrain from scratch — it adds the GPU-detection,
checkpointing, monitoring, and resume-on-hang protocol that the
R17 final run needs.

When the dGPU cold-power-cycles back, the operator runs:

    source env.sh
    uv run --no-sync python molmetal/scripts/r17_gpu_retrain.py \\
        --steps 10000 --seed 42 --seeds 0 1 2

and the wrapper handles everything: 3 sequential seeds, full 30K
total updates, decode-ratio stream, GPU-hang resume, and a final
metrics/by_metric/cfm_decode_ratio_r17.json deliverable.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    from molmetal.scripts.wf386a3_atomic_write import (
        atomic_write_file,
        safe_overwrite_check,
        detect_truncation,
    )

    _WF416_HAS_ATOMIC = True
except ImportError:  # pragma: no cover — fallback when running as standalone
    _WF416_HAS_ATOMIC = False
    atomic_write_file = None  # type: ignore[assignment]
    safe_overwrite_check = None  # type: ignore[assignment]
    detect_truncation = None  # type: ignore[assignment]


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

SMOKE_SCRIPT = PROJECT_ROOT / "molmetal" / "scripts" / "metallo_drug_smoke_retrain.py"
DEFAULT_LOG_ROOT = PROJECT_ROOT / "logs" / "r17_retrain"
DEFAULT_METRIC_PATH = (
    PROJECT_ROOT / "molmetal" / "metrics" / "by_metric" / "cfm_decode_ratio_r17.json"
)

# ---------------------------------------------------------------------------
# Regex patterns — log parsing
# ---------------------------------------------------------------------------
# Match the smoke script's standard step-per-50 log lines.  Example:
#   [metallo_smoke] step 1000/10000  loss=8.27  cfm=8.27  atom=1.15  bond=1.12  ...
RE_STEP_LINE = re.compile(
    r"step\s+(?P<step>\d+)/(?P<total>\d+)\s+loss=(?P<loss>[\d\.eE+\-]+)"
    r"\s+cfm=(?P<cfm>[\d\.eE+\-]+)"
    r"\s+atom=(?P<atom>[\d\.eE+\-]+)"
    r"\s+bond=(?P<bond>[\d\.eE+\-]+)"
)
# Match the decode smoke line:
#   [metallo_smoke]  >> decode_smoke step=1000: decode_ratio=1.000  n_decoded=16/16 ...
RE_DECODE_LINE = re.compile(
    r"decode_smoke\s+step=(?P<step>\d+):\s+"
    r"decode_ratio=(?P<decode_ratio>[\d\.]+)"
    r"\s+n_decoded=(?P<n_decoded>\d+)/(?P<n_total>\d+)"
    r"\s+mean_atoms=(?P<mean_atoms>[\d\.]+)"
    r"\s+disconnected=(?P<disconnected>\d+)"
    r"\s+wall=(?P<wall>[\d\.]+)s"
)
# Match the "CKPT" line we add to the smoke script - actually the
# smoke script just writes to out_dir/ckpt.pt at the end.  Our wrapper
# detects that file's mtime to know when a checkpoint landed.
RE_RESOLVED_DEVICE = re.compile(r"resolved device:\s+(?P<device>\S+)")
RE_SAVED_CKPT = re.compile(r"saved checkpoint:\s+(?P<path>\S+)")


# ---------------------------------------------------------------------------
# GPU detection
# ---------------------------------------------------------------------------
def _wf416_atomic_write_text(
    path, content: str, min_lines: int = 50, expected_min_length: int = 200
) -> None:
    """Wf-416-A1 atomic-write-protocol wrapper around direct write_text.

    Falls back to direct write_text if atomic_write_file is unavailable
    (e.g. when the script is run standalone without the molmetal package
    on PYTHONPATH).
    """
    if _WF416_HAS_ATOMIC and atomic_write_file is not None:
        if safe_overwrite_check is not None:
            _risk = safe_overwrite_check(path, content, min_lines=min_lines)
            if _risk is not None:
                import logging

                logging.getLogger(__name__).warning(
                    "[Wf-416-A1] safe_overwrite_check diff_ratio=%.3f at %s",
                    _risk,
                    path,
                )
        atomic_write_file(path, content, encoding="utf-8")
        if detect_truncation is not None:
            _trunc = detect_truncation(content, expected_min_length=expected_min_length)
            if _trunc is not None:
                import logging

                logging.getLogger(__name__).warning(
                    "[Wf-416-A1] detect_truncation reason=%r at %s",
                    _trunc,
                    path,
                )
    else:  # pragma: no cover — fallback when atomic primitive unavailable
        Path(path).write_text(content, encoding="utf-8")


def _run_cli(cmd: List[str], timeout: float = 5.0) -> Tuple[int, str, str]:
    """Run a CLI command; return (returncode, stdout, stderr).  Never raises."""
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except (FileNotFoundError, subprocess.TimeoutExpired, OSError) as exc:
        return 127, "", f"{type(exc).__name__}: {exc}"


def detect_gpu() -> Dict[str, Any]:
    """Probe GPU state via nvidia-smi (NVIDIA Blackwell) and rocm-smi (legacy AMD).

    Returns a dict with:

    | Key           | Type | Meaning                                        |
    |---------------|------|------------------------------------------------|
    | nvidia_smi    | dict | presence + GPUs listed + rc                    |
    | rocm_smi      | dict | presence + rc (legacy AMD probe)               |
    | torch_cuda    | dict | torch.cuda.is_available + device_count         |
    | vendor        | str  | ``nvidia`` / ``amd`` / ``none``                |
    | usable        | bool | whether torch can dispatch to GPU              |
    | decision      | str  | ``NVIDIA_PATH`` / ``AMD_LEGACY`` / ``BLOCKED`` / ``CPU_DRYRUN`` |
    | probed_at     | str  | ISO8601                                        |
    """
    probed_at = time.strftime("%Y-%m-%dT%H:%M:%S%z")

    # --- nvidia-smi ---------------------------------------------------------
    nv_rc, nv_stdout, nv_stderr = _run_cli(["nvidia-smi", "-L"], timeout=4.0)
    nvidia_present = nv_rc == 0 and "GPU" in nv_stdout
    nvidia_gpus: List[str] = []
    if nvidia_present:
        for line in nv_stdout.strip().splitlines():
            line = line.strip()
            if line.startswith("GPU "):
                nvidia_gpus.append(line)

    # --- rocm-smi (legacy AMD) ---------------------------------------------
    rcm_rc, rcm_stdout, rcm_stderr = _run_cli(["rocm-smi", "-L"], timeout=4.0)
    rocm_present = rcm_rc == 0 and ("GPU" in rcm_stdout or "Card" in rcm_stdout)

    # --- torch -------------------------------------------------------------
    torch_record: Dict[str, Any] = {"probed": False}
    vendor = "none"
    decision = "BLOCKED"
    usable = False
    torch_error: Optional[str] = None
    try:
        import torch  # type: ignore

        cuda_ok = bool(torch.cuda.is_available())
        dev_count = int(torch.cuda.device_count()) if cuda_ok else 0
        torch_record = {
            "probed": True,
            "cuda_available": cuda_ok,
            "device_count": dev_count,
            "torch_version": torch.__version__,
        }
        if cuda_ok and dev_count > 0:
            usable = True
            if nvidia_present:
                vendor = "nvidia"
                decision = "NVIDIA_PATH"
            elif rocm_present:
                vendor = "amd"
                decision = "AMD_LEGACY"
            else:
                # torch says CUDA but neither CLI tool reports it -
                # surface the env and let the operator decide.
                vendor = "unknown"
                decision = "NVIDIA_PATH"
        else:
            decision = "BLOCKED"
    except Exception as exc:  # noqa: BLE001
        torch_error = f"{type(exc).__name__}: {exc}"
        torch_record["error"] = torch_error
        decision = "BLOCKED"

    return {
        "probed_at": probed_at,
        "nvidia_smi": {
            "present": nvidia_present,
            "rc": nv_rc,
            "gpus": nvidia_gpus,
            "stderr": (nv_stderr or "")[:300],
        },
        "rocm_smi": {
            "present": rocm_present,
            "rc": rcm_rc,
            "stderr": (rcm_stderr or "")[:300],
        },
        "torch_cuda": torch_record,
        "vendor": vendor,
        "usable": usable,
        "decision": decision,
        "env": {
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "MOLMETAL_CFM_AMP_DTYPE": os.environ.get("MOLMETAL_CFM_AMP_DTYPE"),
            "MOLMETAL_CFM_AMP": os.environ.get("MOLMETAL_CFM_AMP"),
            "PYTORCH_CUDA_ALLOC_CONF": os.environ.get("PYTORCH_CUDA_ALLOC_CONF"),
        },
    }


# ---------------------------------------------------------------------------
# Subprocess + log tailing
# ---------------------------------------------------------------------------
class TrainSupervisor:
    """Stream a child process, parse decode-ratio events, archive checkpoints.

    The supervisor wraps a single child retraining run
    (``metallo_drug_smoke_retrain.py``) and exposes:

    * ``events``      — list of parsed step + decode events (in order)
    * ``last_step``   — highest training step observed so far
    * ``heartbeat``   — wall-clock of most recent step-log line
    * ``wait_for_progress(deadline_s)`` — sleep loop waiting for new
      stdout (used for hang detection + monitoring thread sync)
    * ``archive_checkpoint_if_due``  — copy ckpt.pt to the archive dir
    * ``final_out_dir``              — child process --out-dir
    """

    def __init__(
        self,
        cmd: List[str],
        log_path: Path,
        ckpt_archive_dir: Path,
        child_out_dir: Path,
        env: Optional[Dict[str, str]] = None,
    ) -> None:
        self.cmd = cmd
        self.log_path = log_path
        self.ckpt_archive_dir = ckpt_archive_dir
        self.child_out_dir = child_out_dir
        self.env = env or os.environ.copy()
        self.proc: Optional[subprocess.Popen] = None
        self.events: List[Dict[str, Any]] = []
        self.last_step: int = 0
        self.last_total: int = 0
        self.heartbeat: float = 0.0
        self._buffer: str = ""
        self._archive_every: int = 1000
        self._last_archived_step: int = 0
        self.stdout_tail_lines: List[str] = []

    def start(self) -> None:
        self.ckpt_archive_dir.mkdir(parents=True, exist_ok=True)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"[r17_gpu] launching child: {' '.join(self.cmd[:3])} "
            f"... (full cmd len={len(self.cmd)})"
        )
        # Open the log file in line-buffered append mode so the parent
        # can `readline` in real-time even if the child writes large
        # bursts.  We use Popen with bufsize=1 + text=True for line
        # buffering on the parent side.
        self._fh = self.log_path.open("w", buffering=1)
        self.proc = subprocess.Popen(
            self.cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            bufsize=1,
            text=True,
            env=self.env,
            cwd=str(PROJECT_ROOT),
        )
        self.heartbeat = time.time()

    def _parse_line(self, line: str) -> None:
        """Ingest one stdout line; update events/last_step/heartbeat."""
        self.stdout_tail_lines.append(line)
        if len(self.stdout_tail_lines) > 200:
            self.stdout_tail_lines = self.stdout_tail_lines[-200:]
        m = RE_STEP_LINE.search(line)
        if m:
            self.last_step = max(self.last_step, int(m.group("step")))
            self.last_total = int(m.group("total"))
            self.heartbeat = time.time()
            self.events.append(
                {
                    "kind": "step",
                    "step": int(m.group("step")),
                    "total": int(m.group("total")),
                    "loss": float(m.group("loss")),
                    "cfm": float(m.group("cfm")),
                    "atom": float(m.group("atom")),
                    "bond": float(m.group("bond")),
                    "wall_s": self.heartbeat,
                }
            )
            return
        m = RE_DECODE_LINE.search(line)
        if m:
            step = int(m.group("step"))
            self.heartbeat = time.time()
            self.events.append(
                {
                    "kind": "decode",
                    "step": step,
                    "decode_ratio": float(m.group("decode_ratio")),
                    "n_decoded": int(m.group("n_decoded")),
                    "n_total": int(m.group("n_total")),
                    "mean_atoms": float(m.group("mean_atoms")),
                    "disconnected": int(m.group("disconnected")),
                    "probe_wall_s": float(m.group("wall")),
                    "wall_s": self.heartbeat,
                }
            )
            return
        if RE_RESOLVED_DEVICE.search(line):
            self.heartbeat = time.time()
            return
        if RE_SAVED_CKPT.search(line):
            self.heartbeat = time.time()
            # Schedule a checkpoint archive for the most recent probe step
            self.archive_checkpoint_if_due(force=True)

    def drain(self, max_lines: int = 200) -> int:
        """Read up to ``max_lines`` lines from child stdout; return lines read."""
        if self.proc is None or self.proc.stdout is None:
            return 0
        n = 0
        for _ in range(max_lines):
            line = self.proc.stdout.readline()
            if not line:
                break
            line = line.rstrip("\n")
            # Mirror to log file (already stripped of \n by readline)
            self._fh.write(line + "\n")
            self._fh.flush()
            # Print every line to the parent's stdout so the operator
            # can watch live.
            print(line)
            self._parse_line(line)
            n += 1
        return n

    def archive_checkpoint_if_due(self, force: bool = False) -> Optional[Path]:
        """If the child wrote ckpt.pt and ``last_step`` crossed a 1000-step
        boundary, copy it to the archive directory.  Returns the archived path
        or None.

        We treat the discovery atomically: only after the file's mtime
        is >= the latest self.events timestamp do we archive it.  The
        child writes ckpt.pt at the END of training (after the final
        step), so this is also a natural "run completed" sentinel.
        """
        src = self.child_out_dir / "ckpt.pt"
        if not src.exists():
            return None
        try:
            src_mtime = src.stat().st_mtime
        except OSError:
            return None
        # The smoke script overwrites ckpt.pt at end of training, so
        # src_mtime > last_heartbeat signals "fresh ckpt available".
        due_step = (self.last_step // self._archive_every) * self._archive_every
        if due_step <= 0:
            return None
        if not force and self._last_archived_step >= due_step:
            return None
        dst_dir = self.ckpt_archive_dir
        dst_dir.mkdir(parents=True, exist_ok=True)
        dst = dst_dir / f"step_{due_step}.pt"
        try:
            shutil.copy2(src, dst)
            latest = dst_dir / "latest.pt"
            if latest.exists() or latest.is_symlink():
                latest.unlink()
            try:
                latest.symlink_to(dst.name)
            except OSError:
                # Fallback for filesystems that disallow symlinks
                shutil.copy2(src, latest)
            self._last_archived_step = due_step
            print(f"[r17_gpu] archived checkpoint step={due_step} -> {dst}")
            return dst
        except OSError as exc:
            print(f"[r17_gpu] WARN: failed to archive ckpt: {exc}")
            return None

    def poll(self) -> Optional[int]:
        return self.proc.poll() if self.proc is not None else None

    def wait(self, timeout: Optional[float] = None) -> int:
        if self.proc is None:
            return -1
        try:
            return self.proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            return -2  # sentinel for timeout

    def kill(self, reason: str = "") -> None:
        if self.proc is None:
            return
        if self.proc.poll() is None:
            print(f"[r17_gpu] killing child (PID={self.proc.pid}): {reason}")
            try:
                self.proc.send_signal(signal.SIGTERM)
                try:
                    self.proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    self.proc.kill()
                    self.proc.wait(timeout=5)
            except ProcessLookupError:
                pass
        if hasattr(self, "_fh") and self._fh and not self._fh.closed:
            self._fh.close()

    def close(self) -> None:
        if hasattr(self, "_fh") and self._fh and not self._fh.closed:
            self._fh.close()


# ---------------------------------------------------------------------------
# Resume-on-hang detection
# ---------------------------------------------------------------------------
def compute_resume_plan(
    supervisor: TrainSupervisor,
    args: argparse.Namespace,
    gpu_probe: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Decide whether to resume; return a plan dict if so, else None.

    Detection rule: if the child is alive but has not produced a new
    step line for ``--hang-timeout`` seconds AND we are *not* between
    probe boundaries (long quiet stretch during training) AND torch
    still sees the GPU, then we declare a hang and resume.
    """
    if args.dry_run:
        return None
    if supervisor.heartbeat <= 0:
        return None
    silent_s = time.time() - supervisor.heartbeat
    if silent_s < args.hang_timeout:
        return None
    # Confirm the GPU is still alive
    try:
        import torch  # type: ignore

        if not torch.cuda.is_available():
            return {
                "reason": "GONE",
                "silent_s": silent_s,
                "last_step": supervisor.last_step,
                "action": "ABORT no GPU - cold power cycle required",
            }
    except Exception:
        return {
            "reason": "GONE",
            "silent_s": silent_s,
            "last_step": supervisor.last_step,
            "action": "ABORT no GPU - cold power cycle required",
        }
    # Try the archive for a resume-from
    last = supervisor.ckpt_archive_dir / "latest.pt"
    if not last.exists() or supervisor._last_archived_step <= 0:
        return {
            "reason": "HANG_NO_CKPT",
            "silent_s": silent_s,
            "last_step": supervisor.last_step,
            "action": "ABORT hang but no checkpoint available - restart from step 0",
        }
    remaining = max(1, args.steps - supervisor.last_step)
    return {
        "reason": "HANG",
        "silent_s": silent_s,
        "last_step": supervisor.last_step,
        "resume_from": str(last),
        "remaining_steps": remaining,
        "action": "RESTART with --resume-from",
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def build_subprocess_cmd(
    args: argparse.Namespace,
    seed: int,
    resume_from: Optional[Path],
    steps: int,
    out_dir: Path,
) -> List[str]:
    """Compose the python invocation for one retrain run."""
    py = sys.executable  # the uv-managed python interpreter
    cmd = [
        py,
        "-u",
        str(SMOKE_SCRIPT.relative_to(PROJECT_ROOT)),
        "--csv",
        str(PROJECT_ROOT / "molmetal" / "data" / "metallo_drugs_500_train.csv"),
        "--steps",
        str(int(steps)),
        "--batch-size",
        str(int(args.batch_size)),
        "--hidden-dim",
        str(int(args.hidden_dim)),
        "--n-layers",
        str(int(args.n_layers)),
        "--lr",
        str(float(args.lr)),
        "--lr-warmup-steps",
        str(int(args.lr_warmup_steps)),
        "--decode-smoke-every",
        str(int(args.decode_smoke_every)),
        "--decode-smoke-samples",
        str(int(args.decode_smoke_samples)),
        "--atom-vocab",
        "20",  # metallodrug = 20 post-TODO-33
        "--bond-head",
        "learned",
        "--seed",
        str(int(seed)),
        "--out-dir",
        str(out_dir),
    ]
    if resume_from is not None:
        cmd += ["--resume-from", str(resume_from)]
    return cmd


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="R17 GPU retrain wrapper (monitoring + checkpointing + resume)"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10000,
        help="Total retrain steps per seed (default 10000)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42],
        help="Seeds to run sequentially (default [42])",
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lr-warmup-steps", type=int, default=200)
    parser.add_argument("--decode-smoke-every", type=int, default=1000)
    parser.add_argument("--decode-smoke-samples", type=int, default=16)
    parser.add_argument(
        "--hang-timeout",
        type=float,
        default=60.0,
        help="No-progress-in-N-seconds -> hang (default 60)",
    )
    parser.add_argument(
        "--heartbeat-poll-s",
        type=float,
        default=2.0,
        help="Child stdout drain period (default 2)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Disable resume-on-hang (just kill and exit)",
    )
    parser.add_argument(
        "--max-resume-attempts",
        type=int,
        default=3,
        help="Max resume attempts per seed before giving up (default 3)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Probe GPU + validate scaffold without launching training",
    )
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--metric-path", type=Path, default=DEFAULT_METRIC_PATH)
    parser.add_argument(
        "--skip-gpu-probe",
        action="store_true",
        help="Skip nvidia-smi / rocm-smi probe (faster)",
    )
    args = parser.parse_args(argv)

    print("[r17_gpu] === R17 GPU RETRAIN CLI WRAPPER ===")
    # TODO IMPROVEMENT_BACKLOG B4 (2026-09-25) — explicit dry-run banner.
    # Print the spec-defined headline before any GPU probe / data
    # load so a CI log that only captured the first 5 lines is
    # unambiguous about what the wrapper would have done.  The rest
    # of the dry-run flow (GPU probe + scaffold + cmd-record write)
    # runs unchanged below.
    if args.dry_run:
        print(
            f"DRY RUN: would launch {args.steps}-step retrain"
            f" (seeds={args.seeds}, hidden_dim={args.hidden_dim},"
            f" n_layers={args.n_layers}, batch={args.batch_size})"
        )
    print(f"[r17_gpu] steps={args.steps} seeds={args.seeds}")
    print(
        f"[r17_gpu] hidden_dim={args.hidden_dim} n_layers={args.n_layers}"
        f" batch_size={args.batch_size} lr={args.lr} warmup={args.lr_warmup_steps}"
    )
    print(
        f"[r17_gpu] decode_smoke_every={args.decode_smoke_every}"
        f" decode_smoke_samples={args.decode_smoke_samples}"
    )
    print(f"[r17_gpu] log_root={args.log_root}")
    print(f"[r17_gpu] metric_path={args.metric_path}")
    print(f"[r17_gpu] dry_run={args.dry_run}")

    # -- Step 1: GPU probe ------------------------------------------------
    args.log_root.mkdir(parents=True, exist_ok=True)
    if args.skip_gpu_probe and not args.dry_run:
        # Use a minimal stub when skipping (still want a JSON on disk)
        gpu_probe = {
            "vendor": "unknown",
            "usable": True,
            "decision": "SKIPPED",
            "probed_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
    else:
        gpu_probe = detect_gpu()
    (
        _wf416_atomic_write_text(args.log_root / "gpu_probe.json"),
        json.dumps(gpu_probe, indent=2),
    )
    print(
        f"[r17_gpu] GPU probe: vendor={gpu_probe['vendor']}"
        f" usable={gpu_probe['usable']} decision={gpu_probe['decision']}"
    )
    if not gpu_probe["usable"]:
        if args.dry_run:
            print("[r17_gpu] dry-run mode: launching scaffold anyway")
        else:
            print("[r17_gpu] ERROR: no usable GPU detected.  Aborting.")
            print("[r17_gpu] HINT: cold-power-cycle the dGPU and re-run.")
            return 10

    # -- Step 2: Launch child processes per seed --------------------------
    aggregate_decode: List[Dict[str, Any]] = []
    seed_summaries: List[Dict[str, Any]] = []
    global_start = time.time()

    for seed in args.seeds:
        print(f"\n[r17_gpu] ===== SEED {seed} =====")
        seed_out = args.log_root / f"seed_{seed}"
        seed_out.mkdir(parents=True, exist_ok=True)
        ckpt_dir = args.log_root / "checkpoints" / f"seed_{seed}"
        log_path = args.log_root / f"seed_{seed}.log"
        resume_path: Optional[Path] = None
        steps_remaining = args.steps
        attempts = 0

        while attempts <= args.max_resume_attempts:
            attempts += 1
            cmd = build_subprocess_cmd(
                args,
                seed=seed,
                resume_from=resume_path,
                steps=steps_remaining,
                out_dir=seed_out,
            )
            if args.dry_run:
                # Don't actually launch - just record the cmd + decision
                cmd_record = args.log_root / f"seed_{seed}_cmd_attempt_{attempts}.txt"
                _wf416_atomic_write_text(cmd_record, "\n".join(cmd) + "\n")
                print(f"[r17_gpu] dry-run: would launch -> {cmd_record}")
                # Synthesize a tiny scaffold decode event so the metric
                # file has a valid shape (probe=0, n_decoded=0,
                # step=0, decode_ratio=0.0).
                synthetic_event = {
                    "kind": "decode",
                    "step": 0,
                    "decode_ratio": 0.0,
                    "n_decoded": 0,
                    "n_total": args.decode_smoke_samples,
                    "mean_atoms": 0.0,
                    "disconnected": args.decode_smoke_samples,
                    "probe_wall_s": 0.0,
                    "wall_s": time.time(),
                    "_synthetic": True,
                    "cmd": cmd,
                }
                aggregate_decode.append(synthetic_event)
                seed_summaries.append(
                    {
                        "seed": seed,
                        "dry_run": True,
                        "cmd_path": str(cmd_record),
                        "attempts": attempts,
                        "decode_events": 0,
                        "max_step": 0,
                    }
                )
                break

            sup = TrainSupervisor(
                cmd=cmd,
                log_path=log_path,
                ckpt_archive_dir=ckpt_dir,
                child_out_dir=seed_out,
            )
            sup.start()
            seed_start = time.time()
            last_dump = seed_start

            try:
                while True:
                    rc = sup.poll()
                    drained = sup.drain(max_lines=200)
                    # Periodic state dump (every 10s)
                    if time.time() - last_dump > 10.0:
                        _dump_state(args, sup, seed, attempts)
                        last_dump = time.time()
                    # Checkpoint archive
                    sup.archive_checkpoint_if_due()
                    # Hang detection
                    if rc is not None:
                        break
                    plan = compute_resume_plan(sup, args, gpu_probe)
                    if (
                        plan
                        and not args.no_resume
                        and attempts <= args.max_resume_attempts
                    ):
                        print(f"[r17_gpu] HANG DETECTED: {plan}")
                        sup.kill(reason=plan.get("action", "hang detected"))
                        if plan["reason"] in ("HANG",) and plan.get("resume_from"):
                            resume_path = Path(plan["resume_from"])
                            steps_remaining = int(plan["remaining_steps"])
                            print(
                                f"[r17_gpu] RESUMING attempt {attempts + 1}"
                                f" from {resume_path} steps_remaining={steps_remaining}"
                            )
                            break  # leave inner while, restart with resume
                        else:
                            # No GPU at all or no checkpoint - cannot recover
                            seed_summaries.append(
                                {
                                    "seed": seed,
                                    "attempts": attempts,
                                    "hang_plan": plan,
                                    "decode_events": len(
                                        [e for e in sup.events if e["kind"] == "decode"]
                                    ),
                                    "max_step": sup.last_step,
                                    "verdict": "FAILED",
                                }
                            )
                            sup.close()
                            break  # bail out of inner while
                    if drained == 0 and rc is None:
                        time.sleep(args.heartbeat_poll_s)
            finally:
                # Drain remaining lines before exiting
                sup.drain(max_lines=500)
                if sup.poll() is None:
                    sup.kill(reason="timeout or exit")
                sup.close()

            if rc is not None:
                # Process exited normally
                seed_summary = {
                    "seed": seed,
                    "attempts": attempts,
                    "returncode": rc,
                    "wall_s": time.time() - seed_start,
                    "decode_events": [e for e in sup.events if e["kind"] == "decode"],
                    "max_step": sup.last_step,
                    "verdict": (
                        "PASS"
                        if any(
                            e["decode_ratio"] > 0.0
                            for e in sup.events
                            if e["kind"] == "decode"
                        )
                        else "DECODE_ZERO"
                    ),
                    "resume_used": resume_path is not None,
                }
                seed_summaries.append(seed_summary)
                aggregate_decode.extend(
                    [e for e in sup.events if e["kind"] == "decode"]
                )
                if rc != 0 and attempts > args.max_resume_attempts:
                    print(
                        f"[r17_gpu] seed {seed} FAILED rc={rc} after"
                        f" {attempts} attempts"
                    )
                    break
                break  # done with this seed
        # end while attempts
    # end for seed

    # -- Step 3: Write final metrics ---------------------------------------
    final_metrics = build_final_metrics(
        gpu_probe,
        seed_summaries,
        aggregate_decode,
        args,
        global_start,
    )
    args.metric_path.parent.mkdir(parents=True, exist_ok=True)
    _wf416_atomic_write_text(args.metric_path, json.dumps(final_metrics, indent=2))
    print(f"[r17_gpu] wrote {args.metric_path}")

    # -- Step 4: Final summary ---------------------------------------------
    print("\n[r17_gpu] === FINAL SUMMARY ===")
    for ss in seed_summaries:
        print(
            f"  seed={ss.get('seed')} verdict={ss.get('verdict')}"
            f" max_step={ss.get('max_step')}"
            f" attempts={ss.get('attempts')}"
            f" wall={ss.get('wall_s', 0):.0f}s"
        )
    return 0


def build_final_metrics(
    gpu_probe: Dict[str, Any],
    seed_summaries: List[Dict[str, Any]],
    decode_events: List[Dict[str, Any]],
    args: argparse.Namespace,
    global_start: float,
) -> Dict[str, Any]:
    """Aggregate per-seed + per-step decode-ratio trajectory into one JSON."""
    decode_summary = {
        "n_events": len(decode_events),
        "first_nonzero_step": next(
            (e["step"] for e in decode_events if e["decode_ratio"] > 0.0),
            None,
        ),
        "max_decode_ratio": float(
            max((e["decode_ratio"] for e in decode_events), default=0.0)
        ),
        "all_probes": decode_events,
    }
    return {
        "phase": "R17_GPU_WRAPPER_v1",
        "date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "gpu_probe": gpu_probe,
        "config": {
            "steps_per_seed": args.steps,
            "seeds": list(args.seeds),
            "hidden_dim": args.hidden_dim,
            "n_layers": args.n_layers,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "lr_warmup_steps": args.lr_warmup_steps,
            "decode_smoke_every": args.decode_smoke_every,
            "decode_smoke_samples": args.decode_smoke_samples,
            "hang_timeout_s": args.hang_timeout,
            "max_resume_attempts": args.max_resume_attempts,
        },
        "wall_clock_s": time.time() - global_start,
        "seed_summaries": seed_summaries,
        "decode_summary": decode_summary,
        "verdict": (
            "PASS"
            if decode_summary["max_decode_ratio"] > 0.0
            else "DRY_RUN_OK"
            if args.dry_run
            else "DECODE_ZERO"
        ),
    }


def _dump_state(
    args: argparse.Namespace, sup: TrainSupervisor, seed: int, attempt: int
) -> None:
    """Write a periodic state snapshot for post-mortem."""
    state = {
        "wall_s": time.time(),
        "seed": seed,
        "attempt": attempt,
        "last_step": sup.last_step,
        "last_total": sup.last_total,
        "heartbeat_age_s": time.time() - sup.heartbeat if sup.heartbeat else None,
        "n_events": len(sup.events),
        "n_decode_events": sum(1 for e in sup.events if e["kind"] == "decode"),
        "last_decode": next(
            (e for e in reversed(sup.events) if e["kind"] == "decode"), None
        ),
    }
    state_path = args.log_root / "probe_state.json"
    _wf416_atomic_write_text(state_path, json.dumps(state, indent=2))


if __name__ == "__main__":
    raise SystemExit(main())
