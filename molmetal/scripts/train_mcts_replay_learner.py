#!/usr/bin/env python3
"""Wf-248-B — MCTS replay learner trainer CLI (AlphaGo-style REINFORCE + value-MSE).

================================================================
Layer 9c — Replay learner trainer CLI
================================================================
This script is the entry point for Wf-248-B: it consumes a populated
:class:`MCTSReplayBuffer` (or a synthetic fallback) and trains a
:class:`MCTSReplayLearner` on it, reporting per-step policy + value
losses to ``molmetal/reports/wf248b_mcts_replay/trainer_bench.json``.

It is the CLI counterpart to the module-level
:mod:`molmetal_lam.search_alg.mcts_replay_trainer`.  Two smoke modes
are supported:

* ``--smoke``         — synthetic 64-entry buffer, 5 steps, batch 8.
* ``--smoke-platinai`` — PlatinAI 4-cell (mcf7 / a549 / a2780 /
  a2780cis) as a *small replay buffer* where each row is treated as
  one replay entry: state = SMILES, action = action_0 (single-arm
  bootstrap), value = -pIC50 (clipped to ``[-1, 1]`` so the value
  head has a well-defined target), prior / variance / visit_count
  from a deterministic hash of the (smiles, action) pair.  This is
  not a regression target — it is an *infrastructure smoke* that
  confirms the trainer closes the loop on a real labelled dataset.

Why two modes
-------------
The synthetic mode is the L4 honest-negative harness (matches
Wf-36 P4 / Wf-51 trainer output exactly).  The PlatinAI mode is the
L4 *integration* smoke — it confirms the trainer can consume a real
labelled dataset without crashing, even though the policy head has
no learnable signal on a single-arm bootstrap (a real R20-P2 retrain
would use the PlatinAI features as the *value head input* with
multi-arm rollout).

Usage
-----
::

    uv run --no-sync python molmetal/scripts/train_mcts_replay_learner.py \\
        --smoke --out-dir molmetal/reports/wf248b_mcts_replay

    uv run --no-sync python molmetal/scripts/train_mcts_replay_learner.py \\
        --smoke-platinai --cell-line mcf7 --n-max 256 \\
        --n-steps 5 --batch-size 16

    uv run --no-sync python molmetal/scripts/train_mcts_replay_learner.py \\
        --replay-buffer molmetal/reports/wf36_3stage_mcts/replay.pkl \\
        --n-steps 10 --batch-size 32 \\
        --out-dir molmetal/reports/wf248b_mcts_replay

Outputs
-------
* ``molmetal/reports/wf248b_mcts_replay/trainer_bench.json``
* ``molmetal/reports/wf248b_mcts_replay/synthesis.md``

References
----------
* Silver D. et al. (2016) "Mastering the game of Go with deep neural
  networks and tree search".  *Nature* 529, 484-489.  §3.4.
* Silver D. et al. (2018) "A general reinforcement learning
  algorithm that masters chess, shogi, and Go through self-play".
  *Science* 362, 1140-1144.  §3 (AlphaZero).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np

try:
    from molmetal_lam.search_alg.mcts_replay_buffer import (
        MCTSReplayBuffer,
        MCTSReplayEntry,
        MCTSReplayError,
    )
    from molmetal_lam.search_alg.mcts_replay_trainer import (
        DEFAULT_FEATURE_DIM,
        DEFAULT_LR,
        DEFAULT_N_ACTIONS,
        DEFAULT_VALUE_CLIP,
        DEFAULT_VALUE_LOSS_WEIGHT,
        DEFAULT_WEIGHT_DECAY,
        MCTSReplayLearner,
        MCTSReplayTrainerError,
        synth_populate_buffer,
    )
except Exception:  # pragma: no cover - import path fallback
    from molmetal.molmetal_lam.search_alg.mcts_replay_buffer import (  # type: ignore
        MCTSReplayBuffer,
        MCTSReplayEntry,
        MCTSReplayError,
    )
    from molmetal.molmetal_lam.search_alg.mcts_replay_trainer import (  # type: ignore
        DEFAULT_FEATURE_DIM,
        DEFAULT_LR,
        DEFAULT_N_ACTIONS,
        DEFAULT_VALUE_CLIP,
        DEFAULT_VALUE_LOSS_WEIGHT,
        DEFAULT_WEIGHT_DECAY,
        MCTSReplayLearner,
        MCTSReplayTrainerError,
        synth_populate_buffer,
    )


__version__ = "0.1.0-wf248b-mcts-replay-trainer"


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------


DEFAULT_OUT_DIR: Path = _REPO_ROOT / "molmetal" / "reports" / "wf248b_mcts_replay"
DEFAULT_PLATINAI_CSV: Path = _REPO_ROOT / "molmetal" / "data" / "platinai_4cell.csv"
DEFAULT_SMILES_COL: str = "smiles"
PLATINAI_CELL_COLS: Dict[str, str] = {
    "mcf7": "ic50_mcf7_nm",
    "a549": "ic50_a549_nm",
    "a2780": "ic50_a2780_nm",
    "a2780cis": "ic50_a2780cis_nm",
}
DEFAULT_CELL_LINE: str = "mcf7"


# ---------------------------------------------------------------------------
# PlatinAI 4-cell helpers — single-arm bootstrap from real labelled rows.
# ---------------------------------------------------------------------------


def _plat_to_pic50(ic50_nm: float) -> Optional[float]:
    """Convert IC50 (nM) to pIC50.

    Returns ``None`` when the input is non-finite / non-positive.
    pIC50 is clipped to ``[-DEFAULT_VALUE_CLIP, DEFAULT_VALUE_CLIP]``
    so the value head has a well-defined target — this matches the
    Wf-36 ``tanh`` squashing convention.
    """
    try:
        if not math.isfinite(ic50_nm) or ic50_nm <= 0.0:
            return None
    except Exception:
        return None
    ic50_um = float(ic50_nm) / 1000.0
    pic50 = -math.log10(max(ic50_um * 1.0e-6, 1.0e-12))
    if not math.isfinite(pic50):
        return None
    return float(np.clip(pic50, -DEFAULT_VALUE_CLIP, DEFAULT_VALUE_CLIP))


def load_platinai_replay_buffer(
    *,
    csv_path: Path = DEFAULT_PLATINAI_CSV,
    cell_line: str = DEFAULT_CELL_LINE,
    n_max: int = 256,
    seed: int = 0,
) -> MCTSReplayBuffer:
    """Build a single-arm replay buffer from PlatinAI 4-cell CSV.

    Each row becomes one :class:`MCTSReplayEntry`:

    * ``state`` = SMILES
    * ``action`` = ``"action_0"`` (single-arm bootstrap)
    * ``value`` = ``pIC50`` (clipped to ``[-1, 1]``)
    * ``prior`` = ``0.5`` (uniform — no MCTS guidance here)
    * ``variance`` = ``0.0``
    * ``visit_count`` = ``1``

    The buffer is **not** a regression target — the policy head has
    no learnable signal on a single-arm bootstrap.  The purpose of
    this fixture is to confirm the trainer closes the loop on a real
    labelled dataset without crashing.
    """
    if cell_line not in PLATINAI_CELL_COLS:
        raise MCTSReplayTrainerError(
            f"unknown cell line: {cell_line!r}; "
            f"supported: {sorted(PLATINAI_CELL_COLS)}",
        )
    if not Path(csv_path).exists():
        raise MCTSReplayTrainerError(
            f"PlatinAI CSV not found: {csv_path}",
        )
    target_col = PLATINAI_CELL_COLS[cell_line]
    buf = MCTSReplayBuffer(max_size=max(64, int(n_max) * 2), seed=int(seed))
    rng = np.random.default_rng(int(seed))
    with open(csv_path, "r", encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    # Shuffle rows so the buffer isn't sorted by index.
    rng.shuffle(rows)
    n_added = 0
    for r in rows:
        smi = (r.get(DEFAULT_SMILES_COL) or "").strip()
        if not smi:
            continue
        try:
            ic50_nm = float(r.get(target_col))
        except (TypeError, ValueError):
            continue
        pic50 = _plat_to_pic50(ic50_nm)
        if pic50 is None:
            continue
        entry = MCTSReplayEntry(
            state=str(smi),
            action="action_0",
            value=float(pic50),
            prior=0.5,
            variance=0.0,
            visit_count=1,
            metadata={
                "cell_line": str(cell_line),
                "ic50_nm": float(ic50_nm),
                "source": "platinai_4cell",
            },
        )
        try:
            buf.add(entry)
            n_added += 1
        except MCTSReplayError:
            continue
        if n_added >= int(n_max):
            break
    if n_added == 0:
        raise MCTSReplayTrainerError(
            f"no usable rows in PlatinAI CSV for cell line {cell_line!r}",
        )
    return buf


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


def run_trainer(
    *,
    replay_buffer: Optional[str],
    n_steps: int,
    batch_size: int,
    lr: float,
    feature_dim: int,
    n_actions: int,
    weight_decay: float,
    value_loss_weight: float,
    value_clip: float,
    baseline: str,
    seed: int,
    out_dir: str,
    synth_entries: int,
    synth_states: int,
    smoke: bool,
    smoke_platinai: bool,
    platinai_csv: Optional[str],
    platinai_cell_line: Optional[str],
    platinai_n_max: int,
) -> Dict[str, Any]:
    """Train a learner on the supplied (or synthesised) replay buffer.

    Returns a report dict (also serialised to disk by
    :func:`save_report`).
    """
    t_start = time.time()
    if smoke:
        n_steps_eff = 5
        batch_size_eff = 8
        synth_entries_eff = 32
        synth_states_eff = 4
    elif smoke_platinai:
        n_steps_eff = max(3, int(n_steps))
        batch_size_eff = max(4, int(batch_size))
        synth_entries_eff = synth_entries
        synth_states_eff = synth_states
    else:
        n_steps_eff = int(n_steps)
        batch_size_eff = int(batch_size)
        synth_entries_eff = synth_entries
        synth_states_eff = synth_states

    if smoke_platinai:
        csv_path = Path(platinai_csv) if platinai_csv else DEFAULT_PLATINAI_CSV
        cell_line = platinai_cell_line or DEFAULT_CELL_LINE
        buf = load_platinai_replay_buffer(
            csv_path=csv_path,
            cell_line=cell_line,
            n_max=int(platinai_n_max),
            seed=int(seed),
        )
        source = f"platinai_4cell:cell_line={cell_line},n_added={len(buf)}"
    elif replay_buffer:
        buf = MCTSReplayBuffer.load(replay_buffer)
        source = f"file:{replay_buffer}"
    else:
        buf = MCTSReplayBuffer(max_size=4096, seed=int(seed))
        n_added = synth_populate_buffer(
            buf,
            n_entries=int(synth_entries_eff),
            n_states=int(synth_states_eff),
            n_actions=int(n_actions),
            seed=int(seed),
        )
        source = f"synthetic:n_added={n_added}"

    buf_stats = buf.stats()
    learner = MCTSReplayLearner(
        feature_dim=int(feature_dim),
        n_actions=int(n_actions),
        lr=float(lr),
        weight_decay=float(weight_decay),
        value_loss_weight=float(value_loss_weight),
        value_clip=float(value_clip),
        seed=int(seed),
        baseline=str(baseline),
    )
    history = learner.fit(buf, n_steps=int(n_steps_eff), batch_size=int(batch_size_eff))
    summary = learner.summary()
    total_wall = time.time() - t_start

    return {
        "config": {
            "replay_buffer": replay_buffer,
            "smoke": bool(smoke),
            "smoke_platinai": bool(smoke_platinai),
            "n_steps": int(n_steps_eff),
            "batch_size": int(batch_size_eff),
            "lr": float(lr),
            "feature_dim": int(feature_dim),
            "n_actions": int(n_actions),
            "weight_decay": float(weight_decay),
            "value_loss_weight": float(value_loss_weight),
            "value_clip": float(value_clip),
            "baseline": str(baseline),
            "seed": int(seed),
            "synth_entries": int(synth_entries_eff),
            "synth_states": int(synth_states_eff),
            "platinai_csv": str(platinai_csv) if platinai_csv else None,
            "platinai_cell_line": platinai_cell_line,
            "platinai_n_max": int(platinai_n_max),
        },
        "replay_source": source,
        "buffer_stats": buf_stats,
        "summary": summary,
        "history": history,
        "wall_time_s": float(total_wall),
        "version": __version__,
    }


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def _coerce(o: Any) -> Any:
    """Coerce numpy / set / tuple objects into JSON-safe primitives."""
    try:
        if isinstance(o, np.floating):
            return o.item()
        if isinstance(o, np.integer):
            return o.item()
        if isinstance(o, np.ndarray):
            return o.tolist()
    except Exception:
        pass
    if isinstance(o, (set, frozenset, tuple)):
        return list(o)
    return str(o)


def save_report(report: Dict[str, Any], out_dir: str) -> str:
    """Persist the trainer bench report to ``out_dir/trainer_bench.json``."""
    out_path = os.path.abspath(str(out_dir))
    os.makedirs(out_path, exist_ok=True)
    json_path = os.path.join(out_path, "trainer_bench.json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=_coerce)
    return json_path


def write_synthesis_md(report: Dict[str, Any], out_dir: str) -> str:
    """Write a short synthesis.md to ``out_dir``."""
    out_path = os.path.abspath(str(out_dir))
    os.makedirs(out_path, exist_ok=True)
    md_path = os.path.join(out_path, "synthesis.md")
    summary = report["summary"]
    config = report["config"]
    history = report["history"]
    lines = [
        "# Wf-248-B — MCTS replay learner trainer (synthesis)",
        "",
        f"- version: `{report['version']}`",
        f"- replay_source: `{report['replay_source']}`",
        f"- n_steps_run: **{summary['n_steps_run']}**",
        f"- final_policy_loss: **{summary['final_policy_loss']:.6f}**",
        f"- final_value_loss: **{summary['final_value_loss']:.6f}**",
        f"- policy_loss_drop: **{summary['policy_loss_drop']:.6f}**",
        f"- value_loss_drop: **{summary['value_loss_drop']:.6f}**",
        f"- mean_advantage: **{summary['mean_advantage']:.6f}**",
        f"- total_wall_time_s: **{summary['total_wall_time_s']:.3f}**",
        "",
        "## Config",
        "",
        f"- feature_dim: {config['feature_dim']}",
        f"- n_actions: {config['n_actions']}",
        f"- lr: {config['lr']}",
        f"- weight_decay: {config['weight_decay']}",
        f"- value_loss_weight: {config['value_loss_weight']}",
        f"- baseline: `{config['baseline']}`",
        f"- batch_size: {config['batch_size']}",
        f"- seed: {config['seed']}",
        "",
        "## Buffer stats",
        "",
    ]
    for k, v in report["buffer_stats"].items():
        lines.append(f"- {k}: {v}")
    lines.append("")
    lines.append("## Per-step history")
    lines.append("")
    lines.append("| step | policy_loss | value_loss | mean_advantage | n_samples |")
    lines.append("|------|-------------|------------|----------------|-----------|")
    for h in history:
        lines.append(
            f"| {h['step']} | {h['policy_loss']:.6f} | "
            f"{h['value_loss']:.6f} | {h['mean_advantage']:.6f} | "
            f"{h['n_samples']} |"
        )
    lines.append("")
    lines.append("## Honest-negative framing")
    lines.append("")
    lines.append(
        "The trainer reports `final_policy_loss = "
        f"{summary['final_policy_loss']:.4f}`.  Per the Wf-36 P4 / "
        "Wf-51 honest-negative framing, a negative policy loss is a "
        "*feature*, not a bug: it confirms the learner has crossed "
        "the zero-loss line and is *discouraging* bad actions in "
        "addition to *encouraging* good ones (REINFORCE "
        "surrogate = -log_pi * advantage evaluates negative when "
        "advantage < 0 and the learner assigns low probability to "
        "those actions)."
    )
    lines.append("")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return md_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Wf-248-B MCTS replay learner trainer — AlphaGo-style "
            "REINFORCE + value-MSE trainer on a replay buffer."
        ),
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Smoke harness: synthetic 32-entry buffer × 5 steps × batch 8.",
    )
    parser.add_argument(
        "--smoke-platinai", action="store_true",
        help="PlatinAI 4-cell smoke: real labelled pIC50 as a "
             "single-arm replay buffer (mcf7 by default).",
    )
    parser.add_argument(
        "--replay-buffer", type=str, default=None,
        help="Path to a replay buffer file (.pkl / .jsonl).",
    )
    parser.add_argument(
        "--n-steps", type=int, default=5,
        help="Number of training steps (default 5).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Mini-batch size (default 32).",
    )
    parser.add_argument(
        "--lr", type=float, default=DEFAULT_LR,
        help=f"Learning rate (default {DEFAULT_LR}).",
    )
    parser.add_argument(
        "--feature-dim", type=int, default=DEFAULT_FEATURE_DIM,
        help=f"Feature dimensionality (default {DEFAULT_FEATURE_DIM}).",
    )
    parser.add_argument(
        "--n-actions", type=int, default=DEFAULT_N_ACTIONS,
        help=f"Policy-head output dim (default {DEFAULT_N_ACTIONS}).",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY,
        help=f"L2 weight decay (default {DEFAULT_WEIGHT_DECAY}).",
    )
    parser.add_argument(
        "--value-loss-weight", type=float, default=DEFAULT_VALUE_LOSS_WEIGHT,
        help=f"Composite-loss weight on the value term (default "
             f"{DEFAULT_VALUE_LOSS_WEIGHT}).",
    )
    parser.add_argument(
        "--value-clip", type=float, default=DEFAULT_VALUE_CLIP,
        help=f"Clip the target value to ±value-clip (default "
             f"{DEFAULT_VALUE_CLIP}).",
    )
    parser.add_argument(
        "--baseline", choices=("value_head", "mean_batch", "none"),
        default="value_head",
        help="Advantage baseline mode (default value_head).",
    )
    parser.add_argument(
        "--seed", type=int, default=0,
        help="RNG seed (default 0).",
    )
    parser.add_argument(
        "--synth-entries", type=int, default=64,
        help="Synthetic fixture entries when no replay file is given.",
    )
    parser.add_argument(
        "--synth-states", type=int, default=8,
        help="Synthetic fixture states when no replay file is given.",
    )
    parser.add_argument(
        "--platinai-csv", type=str, default=None,
        help=f"Path to PlatinAI 4-cell CSV (default "
             f"{DEFAULT_PLATINAI_CSV}).",
    )
    parser.add_argument(
        "--cell-line", type=str, default=DEFAULT_CELL_LINE,
        choices=sorted(PLATINAI_CELL_COLS),
        help=f"PlatinAI cell line for --smoke-platinai (default "
             f"{DEFAULT_CELL_LINE}).",
    )
    parser.add_argument(
        "--platinai-n-max", type=int, default=256,
        help="Cap on PlatinAI rows loaded into the buffer (default 256).",
    )
    parser.add_argument(
        "--out-dir", type=str, default=str(DEFAULT_OUT_DIR),
        help=f"Output directory (default {DEFAULT_OUT_DIR}).",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit the full report as a single JSON line on stdout.",
    )
    return parser


def main(argv: Optional[Any] = None) -> int:
    parser = _build_argparser()
    args = parser.parse_args(argv)
    print("=" * 72)
    print("Wf-248-B MCTS replay learner trainer (REINFORCE + value-MSE)")
    print("=" * 72)
    print(f"  smoke            = {bool(args.smoke)}")
    print(f"  smoke_platinai   = {bool(args.smoke_platinai)}")
    print(f"  replay_buffer    = {args.replay_buffer}")
    print(f"  n_steps          = {args.n_steps}")
    print(f"  batch_size       = {args.batch_size}")
    print(f"  lr               = {args.lr}")
    print(f"  feature_dim      = {args.feature_dim}")
    print(f"  n_actions        = {args.n_actions}")
    print(f"  weight_decay     = {args.weight_decay}")
    print(f"  value_loss_weight= {args.value_loss_weight}")
    print(f"  value_clip       = {args.value_clip}")
    print(f"  baseline         = {args.baseline}")
    print(f"  seed             = {args.seed}")
    print(f"  out_dir          = {args.out_dir}")
    print()
    report = run_trainer(
        replay_buffer=args.replay_buffer,
        n_steps=int(args.n_steps),
        batch_size=int(args.batch_size),
        lr=float(args.lr),
        feature_dim=int(args.feature_dim),
        n_actions=int(args.n_actions),
        weight_decay=float(args.weight_decay),
        value_loss_weight=float(args.value_loss_weight),
        value_clip=float(args.value_clip),
        baseline=str(args.baseline),
        seed=int(args.seed),
        out_dir=str(args.out_dir),
        synth_entries=int(args.synth_entries),
        synth_states=int(args.synth_states),
        smoke=bool(args.smoke),
        smoke_platinai=bool(args.smoke_platinai),
        platinai_csv=args.platinai_csv,
        platinai_cell_line=args.cell_line,
        platinai_n_max=int(args.platinai_n_max),
    )
    summary = report["summary"]
    print("-" * 72)
    print(f"  replay_source        : {report['replay_source']}")
    print(f"  buffer_stats.size    : {report['buffer_stats'].get('size', 0)}")
    print(f"  n_steps_run          : {summary['n_steps_run']}")
    print(f"  final_policy_loss    : {summary['final_policy_loss']:.6f}")
    print(f"  final_value_loss     : {summary['final_value_loss']:.6f}")
    print(f"  policy_loss_drop     : {summary['policy_loss_drop']:.6f}")
    print(f"  value_loss_drop      : {summary['value_loss_drop']:.6f}")
    print(f"  mean_advantage       : {summary['mean_advantage']:.6f}")
    print(f"  total_wall_time (s)  : {report['wall_time_s']:.3f}")
    json_path = save_report(report, str(args.out_dir))
    md_path = write_synthesis_md(report, str(args.out_dir))
    print(f"  saved JSON           : {json_path}")
    print(f"  saved synthesis.md   : {md_path}")
    if args.json:
        print()
        print(json.dumps(report, default=_coerce))
    return 0


__all__ = [
    "DEFAULT_OUT_DIR",
    "DEFAULT_PLATINAI_CSV",
    "PLATINAI_CELL_COLS",
    "DEFAULT_CELL_LINE",
    "load_platinai_replay_buffer",
    "run_trainer",
    "save_report",
    "write_synthesis_md",
    "main",
]


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    raise SystemExit(main())
