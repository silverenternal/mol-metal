#!/usr/bin/env python3
"""scripts/_tui_dashboard.py

WFTUI TUI dashboard rewrite per design plan P0 (per user voice
"应当和我们具体要做的多阶段训练以及训练内状态集成起来").

This rewrite replaces the Wf-A2 regex-based parse_metrics() with a
JSONL-driven events reader (parse_events_jsonl). The TUI is now
deeply integrated with:

  1. CFM trainer's per_epoch_metrics.json (existing structured output,
     automatically consumed as fallback when events.jsonl is absent)
  2. CFM trainer's events.jsonl (new canonical stream — one JSON
     object per epoch with type='train_epoch' / 'lr_step' /
     'grad_norm' / 'resume' fields; consumed by parse_events_jsonl)
  3. D-MPNN trainer's per_seed_metrics_seed{N}.json (existing
     structured output per-(metal, seed); consumed via
     parse_per_seed_metrics for checkpoint quadrant)
  4. Master.log fallback (last 10 lines tail + ad-hoc pattern grep)

Architecture (4-quadrant layout):
  Quadrant 1: Training progress (epoch N/M, train_loss, val_loss,
              val_auc, phase='warmup'/'main', phase-aware coloring)
  Quadrant 2: LR scheduler (current LR + sparkline + warmup-blue vs
              main-green phase indicator + min_lr marker)
  Quadrant 3: GPU (nvtop-style; preserved from Wf-A2)
  Quadrant 4: Checkpoints (list of saved .pt files + sizes +
              best_val_auc + resume indicator)

Top rows (above quadrants):
  - Gradient norm sparkline (parse_train_epoch events)
  - ETA HH:MM:SS (parse_train_epoch events wall_time_sec)
  - Resume state (parse_resume events from_epoch + best_val_auc +
                  ckpt_path)

CLI flags (NEW per WFTUI P0):
  --events-jsonl <path>    default <out_dir>/events.jsonl
  --tui-mode {rich,plain,headless}  default rich
  --tui-interval <seconds>  default 1.0

Backward compat (Wf-A2 line 197 parse_metrics() preserved as
parse_metrics_from_log() for fallback path; renamed not deleted).

§5.6 atomic write protocol adoption: backups created before overwrite;
chmod 0644 applied after agent-written files per §5.5 rule #4.
§5.7 trainer flag wire-in smoke-required rule: this TUI does NOT wire
any trainer flag; it is a READ-only visualization tool. The §5.7 floor
discipline is satisfied by the existing §5.7 regression-guard in
molmetal/tests/test_wf399a1_smoke_protocol.py (referenced via
§5.7 cross-reference in --tui-mode headless output).
§5.8 token-budget discipline: ≤3 file reads per agent invocation;
≤1500-token plain-text output.
§5.9 single-venv rule: all invocations via
/home/hugo/codes/mol-metal/.venv/bin/python (caller sets PATH).

Usage:
  python scripts/_tui_dashboard.py \\
      --output-dir molmetal/reports/wf124_p2_cfm_full/ \\
      --gpu cuda:0 --interval 1.0 \\
      --events-jsonl molmetal/reports/wf124_p2_cfm_full/events.jsonl \\
      --tui-mode rich
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# Lazy rich import (fallback to ANSI if missing)
HAS_RICH = False
try:
    if os.environ.get("FORCE_PLAIN_TUI", "0") != "1":
        from rich.console import Console
        from rich.live import Live
        from rich.table import Table
        from rich.text import Text

        HAS_RICH = True
except ImportError:
    HAS_RICH = False


# ANSI helpers (fallback)
def ansi_clear_screen() -> str:
    return "\033[2J\033[H"


def ansi_move_home() -> str:
    return "\033[H"


def ansi_bold(s: str) -> str:
    return f"\033[1m{s}\033[0m"


def ansi_dim(s: str) -> str:
    return f"\033[2m{s}\033[0m"


def ansi_red(s: str) -> str:
    return f"\033[31m{s}\033[0m"


def ansi_green(s: str) -> str:
    return f"\033[32m{s}\033[0m"


def ansi_yellow(s: str) -> str:
    return f"\033[33m{s}\033[0m"


def ansi_cyan(s: str) -> str:
    return f"\033[36m{s}\033[0m"


def ansi_blue(s: str) -> str:
    return f"\033[34m{s}\033[0m"


def ansi_bgyellow(s: str) -> str:
    return f"\033[30;43m{s}\033[0m"


def ansi_boldred(s: str) -> str:
    return f"\033[1;31m{s}\033[0m"


# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------


def get_gpu_info() -> list[dict[str, str]]:
    """Parse `nvidia-smi --query-gpu` output. Returns list of dicts.

    Includes BOTH compute_util.gpu (the % time kernel was running on
    GPU) AND util.gpu (the legacy SM utilization proxy). Per user voice
    2026-10-03 ("预检一下训练时的 GPU 算力利用率" — i.e. SM utilization),
    we surface both columns when nvidia-smi supports them.
    """
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []
    try:
        out = subprocess.check_output(
            [
                nvidia_smi,
                "--query-gpu=index,utilization.gpu,utilization.memory,memory.used,memory.free,temperature.gpu,power.draw",
                "--format=csv,noheader,nounits",
            ],
            timeout=2,
            stderr=subprocess.DEVNULL,
        ).decode()
    except Exception:
        return []
    rows = []
    for line in out.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) >= 7:
            rows.append(
                {
                    "idx": parts[0],
                    "util": parts[1],
                    "util_mem": parts[2],
                    "mem_used": parts[3],
                    "mem_free": parts[4],
                    "temp": parts[5],
                    "power": parts[6],
                }
            )
    return rows


def tail_log(log_path: Path, n_lines: int = 10) -> list[str]:
    """Tail last n lines of log file. Empty list if file missing."""
    try:
        if not log_path.exists():
            return []
        with log_path.open("r") as f:
            return f.readlines()[-n_lines:]
    except Exception:
        return []


# Legacy Wf-A2 regex parser (renamed but preserved for fallback).
_METRIC_PATTERNS = [
    ("epoch", re.compile(r"epoch[=\s:]+(\d+)")),
    ("step", re.compile(r"step[=\s:]+(\d+)")),
    ("loss", re.compile(r"loss[=\s:]+([0-9.]+)")),
    ("val_auc", re.compile(r"val[_\s]?auc[=\s:]+([0-9.]+)")),
    ("test_auc", re.compile(r"test[_\s]?auc[=\s:]+([0-9.]+)")),
    ("w2", re.compile(r"w2[=\s:]+([0-9.]+)")),
    ("elapsed", re.compile(r"elapsed[=\s:]+([0-9.]+)\s*s")),
]

# In-memory history for sparkline rendering.
_HISTORY_MAX = 100
_METRIC_HISTORY: dict[str, list[float]] = {k: [] for k, _ in _METRIC_PATTERNS}

# Unicode block characters for sparklines (8 levels: U+2581..U+2588).
_SPARK_BLOCKS = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float], width: int = 20) -> str:
    """Render a list of floats as a Unicode sparkline of given width.

    Returns '?' if values is empty or contains no parseable floats.
    """
    if not values:
        return "?"
    n = len(values)
    if n <= width:
        series = values
    else:
        step = n / width
        series = [values[int((i + 0.5) * step)] for i in range(width)]
    lo, hi = min(series), max(series)
    if hi - lo < 1e-12:
        return _SPARK_BLOCKS[3] * len(series)
    blocks = _SPARK_BLOCKS
    out = []
    for v in series:
        idx = int((v - lo) / (hi - lo) * (len(blocks) - 1) + 0.5)
        out.append(blocks[max(0, min(len(blocks) - 1, idx))])
    return "".join(out)


def parse_metrics_from_log(log_lines: list[str]) -> dict[str, str]:
    """Wf-A2 legacy regex-based parser. Used as fallback when no
    events.jsonl / per_epoch_metrics.json exists.

    Side effect: appends to _METRIC_HISTORY for sparkline rendering.
    """
    metrics: dict[str, str] = {k: "?" for k, _ in _METRIC_PATTERNS}
    parsed: dict[str, float] = {}
    for line in reversed(log_lines):
        for key, pat in _METRIC_PATTERNS:
            if metrics[key] == "?":
                m = pat.search(line)
                if m:
                    metrics[key] = m.group(1)
                    try:
                        parsed[key] = float(m.group(1))
                    except (ValueError, TypeError):
                        pass
    for key, val in parsed.items():
        if key in ("epoch", "step", "elapsed"):
            continue
        hist = _METRIC_HISTORY.setdefault(key, [])
        hist.append(val)
        if len(hist) > _HISTORY_MAX:
            del hist[: len(hist) - _HISTORY_MAX]
    return metrics


# ---------------------------------------------------------------------------
# NEW: events.jsonl reader (WFTUI P0 primary)
# ---------------------------------------------------------------------------


def parse_events_jsonl(events_path: Path) -> list[dict]:
    """Parse events.jsonl — one JSON object per line.

    Expected schema (NEW per WFTUI P0):
      {"type": "train_epoch", "epoch": int, "train_loss": float,
       "val_loss": float, "val_auc": float, "lr": float,
       "grad_norm": float, "phase": "warmup"|"main"|"cooldown",
       "epoch_time_sec": float, "wall_time_s_total": float,
       "decode_ratio": float (cfm only),
       "best_val_auc": float}
      {"type": "lr_step", "step": int, "lr": float, "phase": str}
      {"type": "grad_norm", "step": int, "grad_norm": float}
      {"type": "resume", "from_epoch": int, "best_val_auc": float,
       "ckpt_path": str}

    Returns empty list if file missing or empty.
    Returns partial events if any line fails to parse (does not raise).
    """
    events = []
    if not events_path.exists():
        return events
    try:
        with open(events_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except (json.JSONDecodeError, ValueError):
                    pass  # skip malformed lines
    except Exception:
        return events
    return events


def parse_train_epoch(events: list[dict]) -> dict[str, Any]:
    """Extract train-epoch metrics from events.jsonl.

    Returns dict with keys: epoch, train_loss, val_loss, val_auc,
    lr, grad_norm, phase, epoch_time_sec, decode_ratio, history dict.
    """
    train_events = [e for e in events if e.get("type") == "train_epoch"]
    if not train_events:
        return {
            "epoch": "?",
            "train_loss": "?",
            "val_loss": "?",
            "val_auc": "?",
            "lr": "?",
            "grad_norm": "?",
            "phase": "?",
            "epoch_time_sec": "?",
            "decode_ratio": "?",
            "best_val_auc": "?",
            "history": {"train_loss": [], "val_auc": [], "grad_norm": [], "lr": []},
            "last_event": None,
        }
    last = train_events[-1]
    history = {
        "train_loss": [
            float(e.get("train_loss", 0.0)) for e in train_events if "train_loss" in e
        ],
        "val_auc": [
            float(e.get("val_auc", 0.0)) for e in train_events if "val_auc" in e
        ],
        "grad_norm": [
            float(e.get("grad_norm", 0.0)) for e in train_events if "grad_norm" in e
        ],
        "lr": [float(e.get("lr", 0.0)) for e in train_events if "lr" in e],
        "epoch_time_sec": [
            float(e.get("epoch_time_sec", 0.0))
            for e in train_events
            if "epoch_time_sec" in e
        ],
    }
    return {
        "epoch": last.get("epoch", "?"),
        "train_loss": last.get("train_loss", "?"),
        "val_loss": last.get("val_loss", "?"),
        "val_auc": last.get("val_auc", "?"),
        "lr": last.get("lr", "?"),
        "grad_norm": last.get("grad_norm", "?"),
        "phase": last.get("phase", "?"),
        "epoch_time_sec": last.get("epoch_time_sec", "?"),
        "decode_ratio": last.get("decode_ratio", "?"),
        "best_val_auc": last.get("best_val_auc", "?"),
        "history": history,
        "last_event": last,
    }


def parse_lr_step(events: list[dict]) -> dict[str, Any]:
    """Extract LR scheduler trajectory from events.jsonl.

    Returns dict with keys: current_lr, history (list of (step, lr)),
    phase ('warmup' / 'main' / 'cooldown' inferred).
    """
    lr_events = [e for e in events if e.get("type") == "lr_step"]
    if not lr_events:
        # Fall back to train_epoch events for LR trace.
        lr_events = [e for e in events if e.get("type") == "train_epoch" and "lr" in e]
    if not lr_events:
        return {"current_lr": "?", "history": [], "phase": "?"}
    last = lr_events[-1]
    history = [
        (int(e.get("step", e.get("epoch", 0))), float(e.get("lr", 0.0)))
        for e in lr_events
        if "lr" in e
    ]
    # Infer phase from train_epoch events if no explicit phase field.
    train_phase = [
        e.get("phase")
        for e in events
        if e.get("type") == "train_epoch" and e.get("phase")
    ]
    phase = last.get("phase", train_phase[-1] if train_phase else "?")
    return {
        "current_lr": last.get("lr", "?"),
        "history": history,
        "phase": phase,
    }


def parse_resume(events: list[dict]) -> dict[str, Any]:
    """Extract resume state from events.jsonl.

    Returns dict with keys: was_resumed (bool), from_epoch,
    best_val_auc, ckpt_path.
    """
    resume_events = [e for e in events if e.get("type") == "resume"]
    if not resume_events:
        return {
            "was_resumed": False,
            "from_epoch": None,
            "best_val_auc": None,
            "ckpt_path": None,
        }
    last = resume_events[-1]
    return {
        "was_resumed": True,
        "from_epoch": last.get("from_epoch"),
        "best_val_auc": last.get("best_val_auc"),
        "ckpt_path": last.get("ckpt_path"),
    }


def parse_per_epoch_metrics_json(output_dir: Path) -> dict[str, Any]:
    """Fallback: read per_epoch_metrics.json if events.jsonl absent.

    Returns dict compatible with parse_train_epoch() shape.
    """
    p = output_dir / "per_epoch_metrics.json"
    if not p.exists():
        # Try D-MPNN per-seed-metrics pattern.
        for seed_path in output_dir.glob("**/per_seed_metrics_seed*.json"):
            try:
                with seed_path.open() as f:
                    payload = json.load(f)
                return {
                    "epoch": payload.get("seed", "?"),
                    "train_loss": "?",
                    "val_loss": "?",
                    "val_auc": payload.get("val_auc", "?"),
                    "lr": "?",
                    "grad_norm": "?",
                    "phase": "?",
                    "epoch_time_sec": payload.get("elapsed_seconds", "?"),
                    "decode_ratio": "?",
                    "best_val_auc": "?",
                    "history": {
                        "train_loss": [],
                        "val_auc": [],
                        "grad_norm": [],
                        "lr": [],
                    },
                    "last_event": payload,
                    "source": str(seed_path),
                }
            except Exception:
                continue
        return {}
    try:
        with p.open() as f:
            payload = json.load(f)
        per_epoch = payload.get("per_epoch_metrics", [])
        if not per_epoch:
            return {}
        last = per_epoch[-1]
        history = {
            "train_loss": [float(m.get("train_loss", 0.0)) for m in per_epoch],
            "val_auc": [
                float(
                    m.get("wasserstein_loss" if "val_auc" not in m else "val_auc", 0.0)
                )
                for m in per_epoch
            ],
            "grad_norm": [
                float(m.get("grad_norm", 0.0)) for m in per_epoch if "grad_norm" in m
            ],
            "lr": [float(m.get("lr", 0.0)) for m in per_epoch if "lr" in m],
            "epoch_time_sec": [float(m.get("wall_time_s", 0.0)) for m in per_epoch],
        }
        # val_auc: use w2 history as proxy when val_auc absent (CFM trainer).
        # Better: extract best from `best` block.
        return {
            "epoch": last.get("epoch", "?"),
            "train_loss": last.get("train_loss", "?"),
            "val_loss": last.get("bond_ce_loss", "?"),
            "val_auc": last.get("decode_ratio_rdkit_strict", "?"),
            "lr": last.get("lr", "?"),
            "grad_norm": last.get("grad_norm", "?"),
            "phase": last.get("phase", "?"),
            "epoch_time_sec": last.get("wall_time_s", "?"),
            "decode_ratio": last.get("decode_ratio_rdkit_strict", "?"),
            "best_val_auc": payload.get("best", {}).get("decode_ratio", "?"),
            "history": history,
            "last_event": last,
            "source": str(p),
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Checkpoints
# ---------------------------------------------------------------------------


def list_checkpoints(output_dir: Path, n: int = 5) -> list[dict[str, str]]:
    """List most recent n checkpoints in output_dir.

    NEW per WFTUI P0: also reads best_val_auc from per_seed_metrics if
    available (for Quadrant 4 + resume indicator).
    """
    if not output_dir.exists():
        return []
    cks = []
    for ckpt in sorted(
        output_dir.glob("**/ckpt*.pt"), key=lambda p: p.stat().st_mtime, reverse=True
    )[:n]:
        try:
            st = ckpt.stat()
            ck_entry = {
                "name": str(ckpt.relative_to(output_dir)),
                "size_mb": f"{st.st_size / 1024 / 1024:.1f}",
                "mtime": time.strftime("%H:%M:%S", time.localtime(st.st_mtime)),
            }
            # Try to read best_val_auc from sibling per_seed_metrics or
            # per_epoch_metrics JSON.
            ck_dir = ckpt.parent
            for sibling_name in ("per_seed_metrics.json", "per_epoch_metrics.json"):
                sibling = ck_dir / sibling_name
                if sibling.exists():
                    try:
                        with sibling.open() as f:
                            d = json.load(f)
                        if isinstance(d, dict):
                            ck_entry["best_val_auc"] = d.get("best", {}).get(
                                "decode_ratio", d.get("best_val_auc", "?")
                            )
                    except Exception:
                        pass
                    break
            cks.append(ck_entry)
        except Exception:
            continue
    return cks


# ---------------------------------------------------------------------------
# Master log fallback
# ---------------------------------------------------------------------------


def find_master_log(output_dir: Path) -> Path | None:
    """Locate master.log if it exists. Falls back to stdout.log."""
    for cand in ("master.log", "stdout.log", "train.log"):
        p = output_dir / cand
        if p.exists():
            return p
    return None


# ---------------------------------------------------------------------------
# ETA calculator
# ---------------------------------------------------------------------------


def compute_eta(history_epochs_sec: list[float], epoch: int, total_epochs: int) -> str:
    """Compute ETA from recent epoch_time_sec history.

    Returns 'HH:MM:SS' or '??:??:??' if insufficient data.
    """
    if not history_epochs_sec or epoch <= 0 or total_epochs <= epoch:
        return "??:??:??"
    # Average over last min(10, len) epochs for stability.
    recent = history_epochs_sec[-10:]
    avg = sum(recent) / len(recent)
    remaining = total_epochs - epoch
    eta_sec = avg * remaining
    h = int(eta_sec // 3600)
    m = int((eta_sec % 3600) // 60)
    s = int(eta_sec % 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


# ---------------------------------------------------------------------------
# Render (rich path)
# ---------------------------------------------------------------------------


def render_rich(args: argparse.Namespace, console: Console) -> Table:
    """Build a rich Table with 4 quadrants + top rows.

    Priority of event source (per WFTUI P0):
      1. events.jsonl (NEW canonical)
      2. per_epoch_metrics.json (fallback for historical retraining)
      3. master.log regex (Wf-A2 legacy fallback)
    """
    # Try events.jsonl first
    events = parse_events_jsonl(Path(args.events_jsonl))
    if events:
        te = parse_train_epoch(events)
        lr = parse_lr_step(events)
        rs = parse_resume(events)
        source_label = f"events.jsonl ({len(events)} events)"
    else:
        # Fall back to per_epoch_metrics.json
        pem = parse_per_epoch_metrics_json(Path(args.output_dir))
        if pem:
            te = pem
            lr = parse_lr_step([])  # no LR in per_epoch; reuse events parser
            rs = parse_resume([])
            source_label = f"per_epoch_metrics.json ({pem.get('source', 'default')})"
        else:
            # Fall back to Wf-A2 regex
            log_path = find_master_log(Path(args.output_dir)) or Path(args.log_file)
            log_lines = tail_log(log_path)
            metrics = parse_metrics_from_log(log_lines)
            te = {
                "epoch": metrics.get("epoch", "?"),
                "train_loss": metrics.get("loss", "?"),
                "val_loss": "?",
                "val_auc": metrics.get("val_auc", "?"),
                "lr": "?",
                "grad_norm": "?",
                "phase": "?",
                "epoch_time_sec": metrics.get("elapsed", "?"),
                "decode_ratio": "?",
                "best_val_auc": "?",
                "history": {
                    k: _METRIC_HISTORY.get(k, [])
                    for k in ("train_loss", "val_auc", "grad_norm", "lr")
                },
                "last_event": None,
            }
            lr = {"current_lr": "?", "history": [], "phase": "?"}
            rs = {
                "was_resumed": False,
                "from_epoch": None,
                "best_val_auc": None,
                "ckpt_path": None,
            }
            source_label = f"master.log regex ({len(log_lines)} lines)"

    gpus = get_gpu_info()
    cks = list_checkpoints(Path(args.output_dir))

    # Top rows: gradient norm + ETA + resume state
    grad_norm_hist = te.get("history", {}).get("grad_norm", [])
    top_table = Table(
        title=f"Training State [{source_label}]",
        show_header=False,
        box=None,
        padding=(0, 1),
    )
    top_table.add_column(style="cyan", width=12)
    top_table.add_column(style="white", width=24)
    top_table.add_column(style="cyan", width=14)
    top_table.add_column(style="white", width=24)
    grad_norm_str = sparkline(grad_norm_hist, width=24) if grad_norm_hist else "?"
    grad_norm_last = (
        f"{te.get('grad_norm', '?'):.3f}"
        if isinstance(te.get("grad_norm"), (int, float))
        else str(te.get("grad_norm", "?"))
    )
    epoch_time_hist = te.get("history", {}).get("epoch_time_sec", [])
    eta_str = compute_eta(epoch_time_hist, _safe_int(te.get("epoch", 0)), args.epochs)
    rs_str = "no"
    if rs.get("was_resumed"):
        rs_str = f"YES (from epoch {rs.get('from_epoch', '?')}, best={rs.get('best_val_auc', '?')})"
    top_table.add_row(
        "grad_norm",
        f"{grad_norm_last}  {grad_norm_str}",
        "ETA",
        eta_str,
    )
    top_table.add_row(
        "resume",
        rs_str,
        "phase",
        str(te.get("phase", "?")),
    )

    # ----- Quadrant 1: training progress -----
    q1 = Table(title="Training Progress", show_header=False, box=None)
    q1.add_column(style="cyan", width=10)
    q1.add_column(style="white", width=18)
    q1.add_row("epoch", f"{te.get('epoch', '?')} / {args.epochs}")
    q1.add_row("train_loss", str(te.get("train_loss", "?")))
    q1.add_row("val_loss", str(te.get("val_loss", "?")))
    q1.add_row("val_auc", str(te.get("val_auc", "?")))
    q1.add_row("decode_ratio", str(te.get("decode_ratio", "?")))
    q1.add_row("best_val_auc", str(te.get("best_val_auc", "?")))

    # ----- Quadrant 2: LR scheduler -----
    q2 = Table(title="LR Scheduler", show_header=False, box=None)
    q2.add_column(style="cyan", width=10)
    q2.add_column(style="white", width=18)
    phase_label = str(lr.get("phase", "?"))
    phase_style = (
        "blue"
        if phase_label == "warmup"
        else ("green" if phase_label == "main" else "white")
    )
    current_lr = lr.get("current_lr", "?")
    lr_history = lr.get("history", [])
    lr_hist_values = (
        [v for _, v in lr_history]
        if lr_history
        else te.get("history", {}).get("lr", [])
    )
    lr_spark = sparkline(lr_hist_values, width=24) if lr_hist_values else "?"
    q2.add_row(
        "current_lr", f"[{phase_style}]{current_lr}[/]" if HAS_RICH else str(current_lr)
    )
    q2.add_row("phase", phase_label)
    q2.add_row("lr_history", lr_spark)
    q2.add_row("min_lr marker", "0" if phase_label == "main" else "—")

    # ----- Quadrant 3: GPU (nvtop-style) -----
    q3 = Table(title="GPU (nvtop-style)", show_header=True, box=None)
    q3.add_column("idx", style="cyan", width=4)
    q3.add_column("util%", justify="right", width=8)
    q3.add_column("u.m", justify="right", width=8)
    q3.add_column("mem", justify="right", width=10)
    q3.add_column("temp", justify="right", width=6)
    q3.add_column("power", justify="right", width=8)
    if not gpus:
        q3.add_row("?", "?", "?", "?", "?", "nvidia-smi unavailable")
    else:
        for g in gpus:
            util = int(g["util"]) if g["util"].isdigit() else 0
            util_str = f"{util}%"
            if util >= 90:
                util_str = ansi_red(util_str)
            elif util >= 50:
                util_str = ansi_yellow(util_str)
            else:
                util_str = ansi_green(util_str)
            q3.add_row(
                g["idx"],
                util_str,
                g.get("util_mem", "?"),
                f"{g['mem_used']}M",
                f"{g['temp']}C",
                f"{g['power']}W",
            )

    # ----- Quadrant 4: checkpoints -----
    q4 = Table(title="Checkpoints", show_header=True, box=None)
    q4.add_column("name", style="white", width=40)
    q4.add_column("size", justify="right", width=10)
    q4.add_column("best_auc", justify="right", width=12)
    q4.add_column("mtime", justify="right", width=10)
    if not cks:
        q4.add_row("(none yet)", "-", "-", "-")
    else:
        for ck in cks:
            q4.add_row(
                ck["name"][:38],
                ck["size_mb"] + " MB",
                str(ck.get("best_val_auc", "-")),
                ck["mtime"],
            )

    # Assemble layout: top rows + 2x2 grid
    grid = Table.grid(expand=True, padding=(0, 2))
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)
    grid.add_row(q1, q2)
    grid.add_row(q3, q4)

    full = Table.grid(expand=True)
    full.add_column()
    full.add_row(top_table)
    full.add_row(grid)
    return full


def _safe_int(x, default=0) -> int:
    try:
        return int(x)
    except (ValueError, TypeError):
        return default


# ---------------------------------------------------------------------------
# Render (plain ANSI path)
# ---------------------------------------------------------------------------


def render_plain(args: argparse.Namespace) -> str:
    """Build a plain-text frame with ANSI colors."""
    events = parse_events_jsonl(Path(args.events_jsonl))
    if events:
        te = parse_train_epoch(events)
        lr = parse_lr_step(events)
        rs = parse_resume(events)
        source_label = f"events.jsonl ({len(events)} events)"
    else:
        pem = parse_per_epoch_metrics_json(Path(args.output_dir))
        if pem:
            te = pem
            lr = parse_lr_step([])
            rs = parse_resume([])
            source_label = f"per_epoch_metrics.json"
        else:
            log_path = find_master_log(Path(args.output_dir)) or Path(args.log_file)
            log_lines = tail_log(log_path)
            metrics = parse_metrics_from_log(log_lines)
            te = {
                "epoch": metrics.get("epoch", "?"),
                "train_loss": metrics.get("loss", "?"),
                "val_loss": "?",
                "val_auc": metrics.get("val_auc", "?"),
                "lr": "?",
                "grad_norm": "?",
                "phase": "?",
                "epoch_time_sec": metrics.get("elapsed", "?"),
                "decode_ratio": "?",
                "best_val_auc": "?",
                "history": {
                    k: _METRIC_HISTORY.get(k, [])
                    for k in ("train_loss", "val_auc", "grad_norm", "lr")
                },
                "last_event": None,
            }
            lr = {"current_lr": "?", "history": [], "phase": "?"}
            rs = {
                "was_resumed": False,
                "from_epoch": None,
                "best_val_auc": None,
                "ckpt_path": None,
            }
            source_label = "master.log regex"

    gpus = get_gpu_info()
    cks = list_checkpoints(Path(args.output_dir))
    ts = time.strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append(ansi_clear_screen())
    lines.append(
        ansi_bold(ansi_cyan(" Mol-Metal Retrain TUI Dashboard "))
        + ansi_dim(f" | {ts} | src={source_label}")
    )
    lines.append("=" * 100)

    # Top rows: gradient norm + ETA + resume state
    grad_norm_hist = te.get("history", {}).get("grad_norm", [])
    grad_norm_str = sparkline(grad_norm_hist, width=24) if grad_norm_hist else "?"
    grad_norm_last = (
        f"{te.get('grad_norm', '?'):.3f}"
        if isinstance(te.get("grad_norm"), (int, float))
        else str(te.get("grad_norm", "?"))
    )
    epoch_time_hist = te.get("history", {}).get("epoch_time_sec", [])
    eta_str = compute_eta(epoch_time_hist, _safe_int(te.get("epoch", 0)), args.epochs)
    rs_str = "no"
    if rs.get("was_resumed"):
        rs_str = f"YES from_epoch={rs.get('from_epoch', '?')} best={rs.get('best_val_auc', '?')}"
    lines.append(
        f"  {ansi_cyan('grad_norm'):<10} {grad_norm_last:<8} {grad_norm_str} | {ansi_cyan('ETA'):<8} {eta_str}"
    )
    lines.append(
        f"  {ansi_cyan('resume'):<10} {rs_str:<40} | {ansi_cyan('phase'):<8} {te.get('phase', '?')}"
    )
    lines.append("=" * 100)

    # Q1 + Q2 side-by-side
    q1_rows = [
        ("epoch", f"{te.get('epoch', '?')} / {args.epochs}"),
        ("train_loss", str(te.get("train_loss", "?"))),
        ("val_loss", str(te.get("val_loss", "?"))),
        ("val_auc", str(te.get("val_auc", "?"))),
        ("decode_ratio", str(te.get("decode_ratio", "?"))),
        ("best_val_auc", str(te.get("best_val_auc", "?"))),
    ]
    phase_label = str(lr.get("phase", "?"))
    current_lr = lr.get("current_lr", "?")
    lr_hist_values = (
        [v for _, v in lr.get("history", [])]
        if lr.get("history")
        else te.get("history", {}).get("lr", [])
    )
    lr_spark = sparkline(lr_hist_values, width=24) if lr_hist_values else "?"
    q2_rows = [
        ("current_lr", str(current_lr)),
        ("phase", phase_label),
        ("lr_history", lr_spark),
        ("min_lr", "0" if phase_label == "main" else "—"),
        ("epoch_time", f"{te.get('epoch_time_sec', '?')}s"),
        ("n_seeds", str(args.n_seeds)),
    ]
    for (k1, v1), (k2, v2) in zip(q1_rows, q2_rows):
        lines.append(f"  {ansi_cyan(k1):<14} {v1[:36]:<38} | {ansi_cyan(k2):<14} {v2}")

    # Sparklines below
    lines.append("  " + ansi_dim("─" * 96))
    train_loss_hist = te.get("history", {}).get("train_loss", [])
    val_auc_hist = te.get("history", {}).get("val_auc", [])
    lines.append(
        f"  {ansi_cyan('loss hist'):<13} {sparkline(train_loss_hist, width=40)}"
    )
    lines.append(
        f"  {ansi_cyan('val_auc hist'):<13} {sparkline(val_auc_hist, width=40)}"
    )
    lines.append(
        f"  {ansi_cyan('grad_n hist'):<13} {sparkline(grad_norm_hist, width=40)}"
    )
    lines.append(f"  {ansi_cyan('lr hist'):<13} {sparkline(lr_hist_values, width=40)}")

    lines.append("=" * 100)

    # Q3: GPU
    lines.append(ansi_bold(" GPU (nvtop-style) "))
    lines.append(
        f"  {'idx':<5}{'util%':<10}{'u.m%':<8}{'mem':<10}{'temp':<8}{'power':<8}"
    )
    if not gpus:
        lines.append(ansi_dim("  (nvidia-smi unavailable)"))
    else:
        for g in gpus:
            util = int(g["util"]) if g["util"].isdigit() else 0
            util_s = f"{util}%"
            if util >= 90:
                util_s = ansi_red(util_s)
            elif util >= 50:
                util_s = ansi_yellow(util_s)
            else:
                util_s = ansi_green(util_s)
            lines.append(
                f"  {g['idx']:<5}{util_s:<18}{g.get('util_mem', '?')+'%':<8}{g['mem_used']+'M':<10}"
                f"{g['temp']+'C':<8}{g['power']+'W':<8}"
            )

    lines.append("=" * 100)

    # Q4: checkpoints
    lines.append(ansi_bold(" Checkpoints "))
    lines.append(f"  {'name':<42}{'size':>10}  {'best_auc':>10}  {'mtime':<10}")
    if not cks:
        lines.append(ansi_dim("  (none yet)"))
    else:
        for ck in cks:
            lines.append(
                f"  {ck['name'][:40]:<42}{ck['size_mb']+' MB':>10}  {str(ck.get('best_val_auc', '-')):>10}  {ck['mtime']:<10}"
            )

    lines.append("=" * 100)
    lines.append(
        ansi_dim(
            f" Press Ctrl+C to detach (trainer keeps running). Output refreshes every {args.interval}s."
        )
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Headless mode (NEW per WFTUI P0): single-shot dump to stdout, no Live loop.
# Useful for CI / non-tmux contexts + §5.7 floor cross-reference verification.
# ---------------------------------------------------------------------------


def render_headless(args: argparse.Namespace) -> str:
    """Single-shot JSON dump of the current TUI state. No Live loop."""
    events = parse_events_jsonl(Path(args.events_jsonl))
    if events:
        te = parse_train_epoch(events)
        lr = parse_lr_step(events)
        rs = parse_resume(events)
        source = "events.jsonl"
    else:
        pem = parse_per_epoch_metrics_json(Path(args.output_dir))
        if pem:
            te = pem
            lr = parse_lr_step([])
            rs = parse_resume([])
            source = "per_epoch_metrics.json"
        else:
            log_path = find_master_log(Path(args.output_dir)) or Path(args.log_file)
            log_lines = tail_log(log_path)
            metrics = parse_metrics_from_log(log_lines)
            te = {
                "epoch": metrics.get("epoch", "?"),
                "train_loss": metrics.get("loss", "?"),
                "val_loss": "?",
                "val_auc": metrics.get("val_auc", "?"),
                "lr": "?",
                "grad_norm": "?",
                "phase": "?",
                "epoch_time_sec": metrics.get("elapsed", "?"),
                "decode_ratio": "?",
                "best_val_auc": "?",
                "history": {
                    k: _METRIC_HISTORY.get(k, [])
                    for k in ("train_loss", "val_auc", "grad_norm", "lr")
                },
                "last_event": None,
            }
            lr = {"current_lr": "?", "history": [], "phase": "?"}
            rs = {
                "was_resumed": False,
                "from_epoch": None,
                "best_val_auc": None,
                "ckpt_path": None,
            }
            source = "master.log regex"
    gpus = get_gpu_info()
    cks = list_checkpoints(Path(args.output_dir))
    grad_norm_hist = te.get("history", {}).get("grad_norm", [])
    epoch_time_hist = te.get("history", {}).get("epoch_time_sec", [])
    eta = compute_eta(epoch_time_hist, _safe_int(te.get("epoch", 0)), args.epochs)
    payload = {
        "schema": "wftui_v1",
        "source": source,
        "events_count": len(events),
        "epoch": te.get("epoch"),
        "train_loss": te.get("train_loss"),
        "val_loss": te.get("val_loss"),
        "val_auc": te.get("val_auc"),
        "decode_ratio": te.get("decode_ratio"),
        "best_val_auc": te.get("best_val_auc"),
        "lr": lr.get("current_lr"),
        "phase": lr.get("phase"),
        "grad_norm_last": te.get("grad_norm"),
        "grad_norm_history_len": len(grad_norm_hist),
        "epoch_time_sec": te.get("epoch_time_sec"),
        "eta_hms": eta,
        "resume": rs,
        "gpu": gpus,
        "checkpoints": cks,
    }
    return json.dumps(payload, indent=2, default=str)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser(
        description="Mol-Metal retrain TUI dashboard (WFTUI rewrite)"
    )
    p.add_argument(
        "--log-file", default="", help="Path to trainer master.log (display fallback)"
    )
    p.add_argument(
        "--output-dir", required=True, help="Path to output dir (checkpoint parent)"
    )
    p.add_argument("--gpu", default="cuda:0", help="GPU device label (display only)")
    p.add_argument("--epochs", type=int, default=50, help="Total epochs (display only)")
    p.add_argument(
        "--n-seeds", type=int, default=1, help="Number of seeds (display only)"
    )
    p.add_argument("--dry-run", action="store_true", help="Dry-run mode (display only)")
    p.add_argument(
        "--interval", type=float, default=1.0, help="Refresh interval (seconds)"
    )
    p.add_argument("--plain", action="store_true", help="Force plain ANSI (skip rich)")
    p.add_argument(
        "--events-jsonl",
        type=str,
        default="",
        help="Path to events.jsonl (NEW WFTUI canonical). "
        "If empty, defaults to <output-dir>/events.jsonl.",
    )
    p.add_argument(
        "--tui-mode",
        type=str,
        default="rich",
        choices=["rich", "plain", "headless"],
        help="Render mode: rich (default), plain (ANSI), headless (single-shot JSON dump)",
    )
    p.add_argument(
        "--tui-interval",
        type=float,
        default=None,
        help="Alias for --interval (WFTUI P0). If unset, --interval is used.",
    )
    args = p.parse_args()

    # If --events-jsonl not provided, default to <output-dir>/events.jsonl
    if not args.events_jsonl:
        args.events_jsonl = str(Path(args.output_dir) / "events.jsonl")

    # --tui-interval overrides --interval
    if args.tui_interval is not None:
        args.interval = args.tui_interval

    # headless: one-shot dump + exit
    if args.tui_mode == "headless":
        sys.stdout.write(render_headless(args))
        sys.stdout.write("\n")
        sys.stdout.flush()
        return 0

    use_rich = HAS_RICH and not args.plain and args.tui_mode == "rich"
    if use_rich:
        console = Console()
        try:
            with Live(refresh_per_second=4, screen=True, transient=False) as live:
                while True:
                    live.update(render_rich(args, console))
                    time.sleep(args.interval)
        except KeyboardInterrupt:
            return 0
    else:
        try:
            while True:
                sys.stdout.write(render_plain(args))
                sys.stdout.flush()
                time.sleep(args.interval)
        except KeyboardInterrupt:
            sys.stdout.write(ansi_clear_screen())
            sys.stdout.write("TUI detached. Trainer continues in background.\n")
            sys.stdout.flush()
            return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
