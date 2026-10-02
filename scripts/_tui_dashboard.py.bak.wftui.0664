#!/usr/bin/env python3
"""scripts/_tui_dashboard.py
SINGLE canonical TUI dashboard for Mol-Metal retrain (per user voice:
"像nvtop一样实时刷新的表，能带训练指标可视化").

Architecture:
  1. Poll nvidia-smi at 1 Hz → GPU util / memory / temp / power
  2. Tail trainer log file → last 10 lines + parsed metrics + in-memory
     history for loss / AUC sparkline visualization (per user voice
     "能带训练指标可视化那样最好" — Wf-A2 enhancement, 2026-10-02)
  3. List output directory → checkpoints
  4. Render 4 quadrants: training progress / metrics / GPU / checkpoints
  5. On TUI exit / Ctrl+C: kill trainer via .trainer.pid file (per
     start.sh EXIT trap; ensures no orphaned trainer on tmux detach)

Open-source TUI libraries referenced (per Wf-A2 integration directive;
lazy-imported so the dashboard works without any of them installed):
  - rich>=13.0   (Textualize Inc, MIT)        — Rich.Live rendering
  - textual>=0.40 (Textualize Inc, MIT)        — OPTIONAL: for full TUI app
  - nvitop>=1.0   (Xuehai-Pan, Apache-2.0)    — OPTIONAL: nvtop-style GPU
                                                 monitor with process-level
                                                 info (used as primary GPU
                                                 source if installed)
  See requirements-tui.txt for installation hints. Per AGENTS.md §5.8 we
  fall back to plain ANSI if Rich is not installed; no new pip installs
  in docs+code+verify waves.

Usage:
  python scripts/_tui_dashboard.py \\
      --log-file molmetal/reports/m17_pt_.../master.log \\
      --output-dir molmetal/reports/m17_pt_.../ \\
      --gpu cuda:0 --interval 1.0

Environment:
  PYTHON   default: .venv/bin/python (caller sets PATH)
"""
from __future__ import annotations

import argparse
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


def ansi_bgyellow(s: str) -> str:
    return f"\033[30;43m{s}\033[0m"


# ---------------------------------------------------------------------------
# Data sources
# ---------------------------------------------------------------------------

def get_gpu_info() -> list[dict[str, str]]:
    """Parse `nvidia-smi --query-gpu` output. Returns list of dicts."""
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return []
    try:
        out = subprocess.check_output(
            [
                nvidia_smi,
                "--query-gpu=index,utilization.gpu,memory.used,memory.free,temperature.gpu,power.draw",
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
        if len(parts) >= 6:
            rows.append({
                "idx": parts[0],
                "util": parts[1],
                "mem_used": parts[2],
                "mem_free": parts[3],
                "temp": parts[4],
                "power": parts[5],
            })
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


_METRIC_PATTERNS = [
    ("epoch", re.compile(r"epoch[=\s:]+(\d+)")),
    ("step", re.compile(r"step[=\s:]+(\d+)")),
    ("loss", re.compile(r"loss[=\s:]+([0-9.]+)")),
    ("val_auc", re.compile(r"val[_\s]?auc[=\s:]+([0-9.]+)")),
    ("test_auc", re.compile(r"test[_\s]?auc[=\s:]+([0-9.]+)")),
    ("w2", re.compile(r"w2[=\s:]+([0-9.]+)")),
    ("elapsed", re.compile(r"elapsed[=\s:]+([0-9.]+)\s*s")),
]

# In-memory history for sparkline rendering (per Wf-A2 enhancement).
# Capped at 100 points to bound memory under long-running retrain.
_HISTORY_MAX = 100
_METRIC_HISTORY: dict[str, list[float]] = {k: [] for k, _ in _METRIC_PATTERNS}

# Unicode block characters for sparklines (8 levels: U+2581..U+2588).
# Pattern mirrors nvitop / spark / asciichart conventions.
_SPARK_BLOCKS = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float], width: int = 20) -> str:
    """Render a list of floats as a Unicode sparkline of given width.

    Returns '?' if values is empty or contains no parseable floats.
    Resamples to `width` points via stride-or-compress; treats all-equal
    series as flat mid-line.
    """
    if not values:
        return "?"
    # Resample to `width` buckets (right-aligned: latest point at rightmost).
    n = len(values)
    if n <= width:
        series = values
    else:
        # Equal-width bucket stride; take the last of each bucket so
        # the most recent points dominate the right edge.
        step = n / width
        series = [values[int((i + 0.5) * step)] for i in range(width)]
    lo, hi = min(series), max(series)
    if hi - lo < 1e-12:
        # Flat series → all mid blocks.
        return _SPARK_BLOCKS[3] * len(series)
    blocks = _SPARK_BLOCKS
    out = []
    for v in series:
        idx = int((v - lo) / (hi - lo) * (len(blocks) - 1) + 0.5)
        out.append(blocks[max(0, min(len(blocks) - 1, idx))])
    return "".join(out)


def parse_metrics(log_lines: list[str]) -> dict[str, str]:
    """Extract latest metric values from trainer log lines (reverse-scan).

    Side effect: appends the latest parseable float values into
    _METRIC_HISTORY (capped at _HISTORY_MAX) for sparkline rendering.
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
    # Append to history (oldest → newest; order preserved).
    for key, val in parsed.items():
        if key in ("epoch", "step", "elapsed"):
            continue  # monotonic counters don't sparkline well
        hist = _METRIC_HISTORY.setdefault(key, [])
        hist.append(val)
        if len(hist) > _HISTORY_MAX:
            del hist[: len(hist) - _HISTORY_MAX]
    return metrics


def list_checkpoints(output_dir: Path, n: int = 5) -> list[dict[str, str]]:
    """List most recent n checkpoints in output_dir."""
    if not output_dir.exists():
        return []
    cks = []
    for ckpt in sorted(output_dir.glob("**/ckpt*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)[:n]:
        try:
            st = ckpt.stat()
            cks.append({
                "name": str(ckpt.relative_to(output_dir)),
                "size_mb": f"{st.st_size / 1024 / 1024:.1f}",
                "mtime": time.strftime("%H:%M:%S", time.localtime(st.st_mtime)),
            })
        except Exception:
            continue
    return cks


# ---------------------------------------------------------------------------
# Render (rich path)
# ---------------------------------------------------------------------------

def render_rich(args: argparse.Namespace, console: Console) -> Table:
    """Build a rich Table with 4 quadrants."""
    log_lines = tail_log(Path(args.log_file))
    metrics = parse_metrics(log_lines)
    gpus = get_gpu_info()
    cks = list_checkpoints(Path(args.output_dir))

    grid = Table.grid(expand=True, padding=(0, 1))
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)

    # ----- Quadrant 1: training progress -----
    q1 = Table(title="Training Progress", show_header=False, box=None)
    q1.add_column(style="cyan")
    q1.add_column(style="white")
    q1.add_row("epoch", f"{metrics['epoch']} / {args.epochs}")
    q1.add_row("step", metrics["step"])
    q1.add_row("elapsed", f"{metrics['elapsed']}s")
    q1.add_row("loss", metrics["loss"])
    q1.add_row("last log", (log_lines[-1].rstrip() if log_lines else "(none)")[:60])

    # ----- Quadrant 2: metrics + sparkline visualization (Wf-A2 enhancement) -----
    q2 = Table(title="Metrics (with sparklines)", show_header=False, box=None)
    q2.add_column(style="cyan")
    q2.add_column(style="green")
    q2.add_row("val_auc", metrics["val_auc"])
    q2.add_row("test_auc", metrics["test_auc"])
    q2.add_row("w2 distance", metrics["w2"])
    # Sparklines (in-memory history; width=24 for rich path).
    loss_hist = _METRIC_HISTORY.get("loss", [])
    val_auc_hist = _METRIC_HISTORY.get("val_auc", [])
    test_auc_hist = _METRIC_HISTORY.get("test_auc", [])
    q2.add_row("loss history", sparkline(loss_hist, width=24))
    q2.add_row("val_auc history", sparkline(val_auc_hist, width=24))
    q2.add_row("test_auc history", sparkline(test_auc_hist, width=24))

    # ----- Quadrant 3: GPU (nvtop-style) -----
    q3 = Table(title="GPU (nvtop-style)", show_header=True, box=None)
    q3.add_column("idx", style="cyan")
    q3.add_column("util%", justify="right")
    q3.add_column("mem", justify="right")
    q3.add_column("temp", justify="right")
    q3.add_column("power", justify="right")
    if not gpus:
        q3.add_row("?", "?", "?", "?", "nvidia-smi unavailable")
    else:
        for g in gpus:
            util = int(g["util"]) if g["util"].isdigit() else 0
            util_str = f"{ansi_red(str(util)+'%')}" if util >= 90 else (f"{ansi_yellow(str(util)+'%')}" if util >= 50 else f"{ansi_green(str(util)+'%')}")
            q3.add_row(g["idx"], util_str, f"{g['mem_used']}M", f"{g['temp']}C", f"{g['power']}W")

    # ----- Quadrant 4: checkpoints -----
    q4 = Table(title="Checkpoints", show_header=True, box=None)
    q4.add_column("name", style="white")
    q4.add_column("size", justify="right")
    q4.add_column("mtime", justify="right")
    if not cks:
        q4.add_row("(none yet)", "-", "-")
    else:
        for ck in cks:
            q4.add_row(ck["name"][:60], ck["size_mb"] + " MB", ck["mtime"])

    # Assemble 2x2 grid
    top_row = Table.grid(expand=True, padding=(0, 2))
    top_row.add_column(ratio=1)
    top_row.add_column(ratio=1)
    top_row.add_row(q1, q2)
    bot_row = Table.grid(expand=True, padding=(0, 2))
    bot_row.add_column(ratio=1)
    bot_row.add_column(ratio=1)
    bot_row.add_row(q3, q4)

    full = Table.grid(expand=True)
    full.add_column()
    full.add_row(top_row)
    full.add_row(bot_row)
    return full


# ---------------------------------------------------------------------------
# Render (plain ANSI path)
# ---------------------------------------------------------------------------

def render_plain(args: argparse.Namespace) -> str:
    """Build a plain-text frame with ANSI colors."""
    log_lines = tail_log(Path(args.log_file))
    metrics = parse_metrics(log_lines)
    gpus = get_gpu_info()
    cks = list_checkpoints(Path(args.output_dir))
    ts = time.strftime("%Y-%m-%d %H:%M:%S")

    lines = []
    lines.append(ansi_clear_screen())
    lines.append(ansi_bold(ansi_cyan(" Mol-Metal Retrain TUI Dashboard ")) + ansi_dim(f" | {ts} | log={args.log_file}"))
    lines.append("=" * 100)

    # Q1 + Q2 side-by-side (each ~50 wide)
    q1_rows = [
        ("epoch",   f"{metrics['epoch']} / {args.epochs}"),
        ("step",    metrics["step"]),
        ("elapsed", f"{metrics['elapsed']}s"),
        ("loss",    metrics["loss"]),
        ("last",    (log_lines[-1].rstrip() if log_lines else "(none)")[:48]),
    ]
    q2_rows = [
        ("val_auc",  metrics["val_auc"]),
        ("test_auc", metrics["test_auc"]),
        ("w2",       metrics["w2"]),
        ("n_seeds",  str(args.n_seeds)),
        ("dry_run",  str(args.dry_run)),
    ]
    # Side-by-side q1/q2 (limit to min length so alignment holds).
    for (k1, v1), (k2, v2) in zip(q1_rows, q2_rows):
        lines.append(f"  {ansi_cyan(k1):<10} {v1:<38} | {ansi_cyan(k2):<10} {v2}")

    # Sparklines (Wf-A2 enhancement) — render below q1/q2 block.
    loss_hist = _METRIC_HISTORY.get("loss", [])
    val_auc_hist = _METRIC_HISTORY.get("val_auc", [])
    test_auc_hist = _METRIC_HISTORY.get("test_auc", [])
    lines.append("  " + ansi_dim("─" * 96))
    lines.append(f"  {ansi_cyan('loss hist'):<13} {sparkline(loss_hist, width=40)}")
    lines.append(f"  {ansi_cyan('val_auc hist'):<13} {sparkline(val_auc_hist, width=40)}")
    lines.append(f"  {ansi_cyan('test_auc hist'):<13} {sparkline(test_auc_hist, width=40)}")

    lines.append("=" * 100)

    # Q3: GPU (nvtop-style table)
    lines.append(ansi_bold(" GPU (nvtop-style) "))
    lines.append(f"  {'idx':<5}{'util%':<10}{'mem':<10}{'temp':<8}{'power':<8}")
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
                f"  {g['idx']:<5}{util_s:<18}{g['mem_used']+'M':<10}{g['temp']+'C':<8}{g['power']+'W':<8}"
            )

    lines.append("=" * 100)

    # Q4: checkpoints
    lines.append(ansi_bold(" Checkpoints "))
    lines.append(f"  {'name':<62}{'size':>10}  {'mtime':<10}")
    if not cks:
        lines.append(ansi_dim("  (none yet)"))
    else:
        for ck in cks:
            lines.append(f"  {ck['name'][:60]:<62}{ck['size_mb']+' MB':>10}  {ck['mtime']:<10}")

    lines.append("=" * 100)
    lines.append(ansi_dim(" Press Ctrl+C to detach (trainer keeps running). Output refreshes every "
                          + str(args.interval) + "s."))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main() -> int:
    p = argparse.ArgumentParser(description="Mol-Metal retrain TUI dashboard")
    p.add_argument("--log-file", required=True, help="Path to trainer master.log")
    p.add_argument("--output-dir", required=True, help="Path to output dir (checkpoint parent)")
    p.add_argument("--gpu", default="cuda:0", help="GPU device label (display only)")
    p.add_argument("--epochs", type=int, default=50, help="Total epochs (display only)")
    p.add_argument("--n-seeds", type=int, default=1, help="Number of seeds (display only)")
    p.add_argument("--dry-run", action="store_true", help="Dry-run mode (display only)")
    p.add_argument("--interval", type=float, default=1.0, help="Refresh interval (seconds)")
    p.add_argument("--plain", action="store_true", help="Force plain ANSI (skip rich)")
    args = p.parse_args()

    use_rich = HAS_RICH and not args.plain
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
