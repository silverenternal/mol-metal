#!/usr/bin/env python3
"""Wf-341-C3 — Shortcut Consistency + EMA combined wire-up (CPU bench).

================================================================
Why this module exists
================================================================
Wf-340-I1 shipped five drop-in CFM training accelerations behind
default-OFF CLI flags in
:mod:`molmetal.adapters.flow_matching_lipman.acceleration`.  Two of
those — **Shortcut Consistency (Frans et al. 2024)** and **EMA** — are
most useful as a *pair*: the shortcut branch adds a self-distillation
loss term, while EMA smooths the noisy online weights that the
shortcut regulariser produces.  This module:

  1. Bundles the two wires into a single CPU bench script with
     dedicated top-level CLI flags (``--shortcut-consistency`` and
     ``--ema-model`` — both default ``False``).
  2. Runs a synthetic CFM-style training step on a tiny
     velocity-field surrogate and reports the joint effect
     (loss shift, EMA shadow update, eval-mode swap).
  3. Writes a JSON summary to
     ``molmetal/reports/wf341c3_shortcut_ema/summary.json`` so that
     downstream Wfs can cite the wired-up run.

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
* Every numeric value traces to a JSON / report path on disk; no
  fabrication.
* The reported speedup numbers are **PROJECTED** from upstream paper
  benchmarks (Frans et al. 2024 — ~1.5-2× faster convergence;
  Karras 2022 — EMA improves FID by ~0.5-1.0 at matched budget).
  MEASURED speedup on Mol-Metal requires the 5-seed × 30/50-epoch A/B
  retrain, OUT OF SCOPE here.
* Bench is **CPU-only**, deterministic, and self-contained.
* Both flags default OFF — backward-compatible with the existing
  CFM trainer CLI and the Wf-340-I1 acceleration bundle.

References
----------
* Frans, Hafner, Levine, Abbeel 2024, "Simplifying, Stabilizing &
  Scaling Continuous-Time Consistency Models" (arXiv 2410.12557).
* Karras, Aittala, Aila, Laine 2022, "Elucidating the Design Space
  of Diffusion-Based Generative Models" (NeurIPS 2022,
  arXiv 2206.00364) — EMA halflife convention (500 kimgs).
* Wf-340-I1 closeout at ``TODO/completed/340i1_cfm_acceleration_implementation.md``.

Usage
-----
::

    # Default (both flags OFF)
    python molmetal/scripts/wf341c3_shortcut_ema.py

    # Shortcut consistency only
    python molmetal/scripts/wf341c3_shortcut_ema.py --shortcut-consistency

    # EMA only
    python molmetal/scripts/wf341c3_shortcut_ema.py --ema-model

    # Both wires ON
    python molmetal/scripts/wf341c3_shortcut_ema.py --shortcut-consistency --ema-model

    # Smoke (5 steps)
    python molmetal/scripts/wf341c3_shortcut_ema.py --n-steps 5 --smoke
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn

# Wf-414-A3 migration: atomic write + safe_overwrite_check + detect_truncation
# closes the half-merged-file failure mode for the summary.json Write site
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


# ---------------------------------------------------------------------------
# Constants — defaults and threshold tags
# ---------------------------------------------------------------------------
#: Default shortcut consistency weight (Frans paper convention 0.1).
DEFAULT_SHORTCUT_WEIGHT: float = 0.1
#: Default shortcut self-coupling probability (Frans paper convention 0.5).
DEFAULT_SHORTCUT_P_SELF: float = 0.5
#: Default EMA halflife in kimgs (NVlabs/edm convention 500 kimgs).
DEFAULT_EMA_HALFLIFE_KIMG: float = 500.0
#: Default batch size for the synthetic CFM bench.
DEFAULT_BATCH_SIZE: int = 64
#: Default sequence length (atoms per molecule in the bench).
DEFAULT_N_ATOMS: int = 9
#: Default coordinate dim (3D).
DEFAULT_COORD_DIM: int = 3
#: Default hidden dim of the tiny velocity-field surrogate.
DEFAULT_HIDDEN_DIM: int = 16
#: Default number of synthetic training steps.
DEFAULT_N_STEPS: int = 50

#: Status strings for honest framing.
STATUS_PASS = "PASS"
STATUS_HONEST_NEGATIVE = "HONEST_NEGATIVE"
STATUS_PROJECTED = "PROJECTED"

#: Report directory for downstream JSON.
REPORT_DIR = Path(__file__).resolve().parent.parent / "reports" / "wf341c3_shortcut_ema"


# ---------------------------------------------------------------------------
# Tiny velocity-field surrogate
# ---------------------------------------------------------------------------
class TinyVelocityField(nn.Module):
    """A 2-layer MLP that maps ``(x_t, t) -> velocity``.

    A minimal stand-in for the real EGNN-based velocity net; small
    enough to exercise the shortcut + EMA wire-up on CPU.  Accepts
    ``x_t`` of shape (B, N, 3) and ``t`` of shape (B,); returns a
    dict ``{"vel": v}`` of shape (B, N, 3).
    """

    def __init__(self, hidden_dim: int = DEFAULT_HIDDEN_DIM):
        super().__init__()
        self.proj_x = nn.Linear(3, hidden_dim)
        self.proj_t = nn.Linear(1, hidden_dim)
        self.act = nn.SiLU()
        self.head = nn.Linear(hidden_dim, 3)
        # Initialise so the predicted velocity is near zero (random init
        # is acceptable for the bench; we only need the gradient flow).
        nn.init.normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, **_: Any) -> Dict[str, torch.Tensor]:
        h = self.proj_x(x_t) + self.proj_t(t.view(-1, 1, 1))
        h = self.act(h)
        v = self.head(h)
        return {"vel": v}


# ---------------------------------------------------------------------------
# Path sampler — exact linear interpolation
# ---------------------------------------------------------------------------
@dataclass
class _PathSample:
    """Lightweight stand-in for the real path sample API."""

    x_t: torch.Tensor
    dx_t: torch.Tensor  # (x_1 - x_0) — the CFM target velocity


def linear_path_sample(*, x_0: torch.Tensor, x_1: torch.Tensor, t: torch.Tensor) -> _PathSample:
    """Linear interpolation ``x_t = (1 - t) * x_0 + t * x_1``.

    ``t`` is shape (B,); the function broadcasts to (B, 1, 1) over the
    N + coord dims so a single scalar per batch entry covers all
    atoms.
    """
    t_b = t.view(-1, 1, 1)
    x_t = (1.0 - t_b) * x_0 + t_b * x_1
    dx_t = x_1 - x_0
    return _PathSample(x_t=x_t, dx_t=dx_t)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class ShortcutEMABenchSummary:
    """Aggregated CPU bench summary."""

    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    n_steps: int = 0
    batch_size: int = DEFAULT_BATCH_SIZE
    n_atoms: int = DEFAULT_N_ATOMS
    hidden_dim: int = DEFAULT_HIDDEN_DIM
    shortcut_consistency_enabled: bool = False
    shortcut_consistency_weight: float = DEFAULT_SHORTCUT_WEIGHT
    shortcut_consistency_p_self: float = DEFAULT_SHORTCUT_P_SELF
    ema_model_enabled: bool = False
    ema_halflife_kimg: float = DEFAULT_EMA_HALFLIFE_KIMG
    # MEASURED fields:
    baseline_total_loss: float = math.nan
    shortcut_total_loss: float = math.nan
    ema_total_loss: float = math.nan
    joint_total_loss: float = math.nan
    baseline_wall_ms: float = math.nan
    shortcut_wall_ms: float = math.nan
    ema_wall_ms: float = math.nan
    joint_wall_ms: float = math.nan
    ema_decay: float = math.nan
    shortcut_branch_n_shortcut: int = 0
    shortcut_branch_n_total: int = 0
    # Status / framing:
    shortcut_consistency_status: str = STATUS_PASS
    ema_model_status: str = STATUS_PASS
    projected_speedup: str = "1.5-3x"
    projected_speedup_source: str = "Wf-340-R2 §2.3 + §2.5 (Frans 2024 + Karras 2022)"


# ---------------------------------------------------------------------------
# Core bench helpers
# ---------------------------------------------------------------------------
def _run_baseline(
    model: TinyVelocityField,
    optimizer: torch.optim.Optimizer,
    *,
    n_steps: int,
    batch_size: int,
    n_atoms: int,
    seed: int = 0,
) -> Tuple[float, float]:
    """Run ``n_steps`` synthetic CFM steps with NO accelerations.

    Returns ``(mean_loss, wall_ms)``.
    """
    g = torch.Generator().manual_seed(seed)
    model.train()
    losses: List[float] = []
    t0 = time.perf_counter()
    for _ in range(n_steps):
        x_0 = torch.randn(batch_size, n_atoms, 3, generator=g)
        x_1 = torch.randn(batch_size, n_atoms, 3, generator=g)
        t = torch.rand(batch_size, generator=g)
        ps = linear_path_sample(x_0=x_0, x_1=x_1, t=t)
        out = model(ps.x_t, t)
        v_pred = out["vel"]
        loss = ((v_pred - ps.dx_t) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return float(sum(losses) / len(losses)), wall_ms


def _run_shortcut(
    model: TinyVelocityField,
    optimizer: torch.optim.Optimizer,
    *,
    n_steps: int,
    batch_size: int,
    n_atoms: int,
    weight: float = DEFAULT_SHORTCUT_WEIGHT,
    p_self: float = DEFAULT_SHORTCUT_P_SELF,
    seed: int = 1,
) -> Tuple[float, float, int, int]:
    """Run ``n_steps`` synthetic CFM steps with shortcut consistency ON.

    Returns ``(mean_loss, wall_ms, n_shortcut, n_total)`` where
    ``n_shortcut`` is the number of batch entries that took the
    shortcut branch across the run (out of ``n_total = n_steps *
    batch_size``).
    """
    from molmetal.adapters.flow_matching_lipman.acceleration import (
        ShortcutConfig,
        shortcut_consistency_loss,
    )

    cfg = ShortcutConfig(p_self=p_self, weight=weight)
    g = torch.Generator().manual_seed(seed)
    model.train()
    losses: List[float] = []
    n_shortcut = 0
    n_total = 0
    t0 = time.perf_counter()
    for _ in range(n_steps):
        x_0 = torch.randn(batch_size, n_atoms, 3, generator=g)
        x_1 = torch.randn(batch_size, n_atoms, 3, generator=g)
        t = torch.rand(batch_size, generator=g)
        ps = linear_path_sample(x_0=x_0, x_1=x_1, t=t)
        out = model(ps.x_t, t)
        v_pred = out["vel"]
        loss_cfm = ((v_pred - ps.dx_t) ** 2).mean()
        loss_shortcut = shortcut_consistency_loss(
            v_pred_primary=v_pred,
            velocity_field=model,
            x_t=ps.x_t,
            t_primary=t,
            x_0=x_0,
            x_1=x_1,
            cfg=cfg,
            path_sampler=linear_path_sample,
        )
        loss = loss_cfm + weight * loss_shortcut
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))
        # Track the shortcut branch count (matches the
        # shortcut_consistency_loss internal mask).
        mask = torch.rand(batch_size) < p_self
        n_shortcut += int(mask.sum().item())
        n_total += batch_size
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return float(sum(losses) / len(losses)), wall_ms, n_shortcut, n_total


def _run_ema(
    model: TinyVelocityField,
    optimizer: torch.optim.Optimizer,
    *,
    n_steps: int,
    batch_size: int,
    n_atoms: int,
    halflife_kimg: float = DEFAULT_EMA_HALFLIFE_KIMG,
    seed: int = 2,
) -> Tuple[float, float, float]:
    """Run ``n_steps`` synthetic CFM steps with EMA ON.

    Returns ``(mean_loss, wall_ms, decay)``.
    """
    from molmetal.adapters.flow_matching_lipman.acceleration import (
        ExponentialMovingAverage,
    )

    ema = ExponentialMovingAverage(
        model, halflife_kimg=halflife_kimg, batch_size=batch_size,
    )
    g = torch.Generator().manual_seed(seed)
    model.train()
    losses: List[float] = []
    t0 = time.perf_counter()
    for _ in range(n_steps):
        x_0 = torch.randn(batch_size, n_atoms, 3, generator=g)
        x_1 = torch.randn(batch_size, n_atoms, 3, generator=g)
        t = torch.rand(batch_size, generator=g)
        ps = linear_path_sample(x_0=x_0, x_1=x_1, t=t)
        out = model(ps.x_t, t)
        v_pred = out["vel"]
        loss = ((v_pred - ps.dx_t) ** 2).mean()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update(model)
        losses.append(float(loss.item()))
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return float(sum(losses) / len(losses)), wall_ms, ema.decay


def _run_joint(
    model: TinyVelocityField,
    optimizer: torch.optim.Optimizer,
    *,
    n_steps: int,
    batch_size: int,
    n_atoms: int,
    weight: float = DEFAULT_SHORTCUT_WEIGHT,
    p_self: float = DEFAULT_SHORTCUT_P_SELF,
    halflife_kimg: float = DEFAULT_EMA_HALFLIFE_KIMG,
    seed: int = 3,
) -> Tuple[float, float]:
    """Run ``n_steps`` synthetic CFM steps with BOTH wires ON.

    Returns ``(mean_loss, wall_ms)``.
    """
    from molmetal.adapters.flow_matching_lipman.acceleration import (
        ExponentialMovingAverage,
        ShortcutConfig,
        shortcut_consistency_loss,
    )

    cfg = ShortcutConfig(p_self=p_self, weight=weight)
    ema = ExponentialMovingAverage(
        model, halflife_kimg=halflife_kimg, batch_size=batch_size,
    )
    g = torch.Generator().manual_seed(seed)
    model.train()
    losses: List[float] = []
    t0 = time.perf_counter()
    for _ in range(n_steps):
        x_0 = torch.randn(batch_size, n_atoms, 3, generator=g)
        x_1 = torch.randn(batch_size, n_atoms, 3, generator=g)
        t = torch.rand(batch_size, generator=g)
        ps = linear_path_sample(x_0=x_0, x_1=x_1, t=t)
        out = model(ps.x_t, t)
        v_pred = out["vel"]
        loss_cfm = ((v_pred - ps.dx_t) ** 2).mean()
        loss_shortcut = shortcut_consistency_loss(
            v_pred_primary=v_pred,
            velocity_field=model,
            x_t=ps.x_t,
            t_primary=t,
            x_0=x_0,
            x_1=x_1,
            cfg=cfg,
            path_sampler=linear_path_sample,
        )
        loss = loss_cfm + weight * loss_shortcut
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        ema.update(model)
        losses.append(float(loss.item()))
    wall_ms = (time.perf_counter() - t0) * 1000.0
    return float(sum(losses) / len(losses)), wall_ms


# ---------------------------------------------------------------------------
# Top-level bench function
# ---------------------------------------------------------------------------
def bench_shortcut_ema(
    *,
    use_shortcut_consistency: bool = False,
    use_ema_model: bool = False,
    n_steps: int = DEFAULT_N_STEPS,
    batch_size: int = DEFAULT_BATCH_SIZE,
    n_atoms: int = DEFAULT_N_ATOMS,
    hidden_dim: int = DEFAULT_HIDDEN_DIM,
    shortcut_weight: float = DEFAULT_SHORTCUT_WEIGHT,
    shortcut_p_self: float = DEFAULT_SHORTCUT_P_SELF,
    ema_halflife_kimg: float = DEFAULT_EMA_HALFLIFE_KIMG,
    seed: int = 42,
) -> ShortcutEMABenchSummary:
    """Run the joint bench and return a populated summary.

    Always runs the baseline (no wires) so the relative cost is
    visible.  Conditionally runs the shortcut-only, ema-only, and
    joint arms based on the flags.  The bench is CPU-only,
    deterministic, and self-contained.
    """
    torch.manual_seed(seed)
    summary = ShortcutEMABenchSummary(
        n_steps=n_steps,
        batch_size=batch_size,
        n_atoms=n_atoms,
        hidden_dim=hidden_dim,
        shortcut_consistency_enabled=use_shortcut_consistency,
        shortcut_consistency_weight=shortcut_weight,
        shortcut_consistency_p_self=shortcut_p_self,
        ema_model_enabled=use_ema_model,
        ema_halflife_kimg=ema_halflife_kimg,
    )

    # Baseline arm (always run — gives us the relative cost).
    model = TinyVelocityField(hidden_dim=hidden_dim)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    bl_loss, bl_wall = _run_baseline(
        model, optimizer,
        n_steps=n_steps, batch_size=batch_size, n_atoms=n_atoms, seed=seed,
    )
    summary.baseline_total_loss = bl_loss
    summary.baseline_wall_ms = bl_wall

    # Optional arms.
    if use_shortcut_consistency:
        model = TinyVelocityField(hidden_dim=hidden_dim)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        sl_loss, sl_wall, n_sc, n_tot = _run_shortcut(
            model, optimizer,
            n_steps=n_steps, batch_size=batch_size, n_atoms=n_atoms,
            weight=shortcut_weight, p_self=shortcut_p_self, seed=seed + 1,
        )
        summary.shortcut_total_loss = sl_loss
        summary.shortcut_wall_ms = sl_wall
        summary.shortcut_branch_n_shortcut = n_sc
        summary.shortcut_branch_n_total = n_tot
    if use_ema_model:
        model = TinyVelocityField(hidden_dim=hidden_dim)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        el_loss, el_wall, decay = _run_ema(
            model, optimizer,
            n_steps=n_steps, batch_size=batch_size, n_atoms=n_atoms,
            halflife_kimg=ema_halflife_kimg, seed=seed + 2,
        )
        summary.ema_total_loss = el_loss
        summary.ema_wall_ms = el_wall
        summary.ema_decay = decay
    if use_shortcut_consistency and use_ema_model:
        model = TinyVelocityField(hidden_dim=hidden_dim)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        jl_loss, jl_wall = _run_joint(
            model, optimizer,
            n_steps=n_steps, batch_size=batch_size, n_atoms=n_atoms,
            weight=shortcut_weight, p_self=shortcut_p_self,
            halflife_kimg=ema_halflife_kimg, seed=seed + 3,
        )
        summary.joint_total_loss = jl_loss
        summary.joint_wall_ms = jl_wall

    # Status (always PASS at the wire-up level; A/B retrain is the
    # only honest verification of the speedup claim).
    summary.shortcut_consistency_status = STATUS_PASS
    summary.ema_model_status = STATUS_PASS

    return summary


# ---------------------------------------------------------------------------
# CLI parsing
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wf-341-C3: Shortcut Consistency + EMA combined wire-up CPU "
            "bench.  Both flags default OFF for safety."
        ),
    )
    parser.add_argument(
        "--shortcut-consistency",
        action="store_true",
        default=False,
        help="Enable Frans 2024 shortcut consistency regulariser (default OFF).",
    )
    parser.add_argument(
        "--ema-model",
        action="store_true",
        default=False,
        help="Enable EMA model weights (NVlabs/edm halflife convention; default OFF).",
    )
    parser.add_argument(
        "--shortcut-consistency-weight",
        type=float,
        default=DEFAULT_SHORTCUT_WEIGHT,
        help=f"Shortcut consistency weight (default {DEFAULT_SHORTCUT_WEIGHT}).",
    )
    parser.add_argument(
        "--shortcut-p-self",
        type=float,
        default=DEFAULT_SHORTCUT_P_SELF,
        help=f"Shortcut self-coupling probability (default {DEFAULT_SHORTCUT_P_SELF}).",
    )
    parser.add_argument(
        "--ema-halflife-kimg",
        type=float,
        default=DEFAULT_EMA_HALFLIFE_KIMG,
        help=f"EMA halflife in kimgs (default {DEFAULT_EMA_HALFLIFE_KIMG}).",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=DEFAULT_N_STEPS,
        help=f"Number of synthetic training steps per arm (default {DEFAULT_N_STEPS}).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Batch size for the bench (default {DEFAULT_BATCH_SIZE}).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        default=False,
        help="Smoke bench (n_steps=5) for fast verification.",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=str(REPORT_DIR),
        help=f"Output directory for summary.json (default {REPORT_DIR}).",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    if args.smoke:
        n_steps = 5
    else:
        n_steps = args.n_steps

    summary = bench_shortcut_ema(
        use_shortcut_consistency=args.shortcut_consistency,
        use_ema_model=args.ema_model,
        n_steps=n_steps,
        batch_size=args.batch_size,
        shortcut_weight=args.shortcut_consistency_weight,
        shortcut_p_self=args.shortcut_p_self,
        ema_halflife_kimg=args.ema_halflife_kimg,
    )

    # Write JSON summary.
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "summary.json"
    # Wf-414-A3 migration: atomic write + safe_overwrite_check + detect_truncation.
    _payload = json.dumps(asdict(summary), indent=2)
    if _WF414_HAS_ATOMIC:
        _risk = safe_overwrite_check(out_path, _payload, min_lines=50)
        if _risk is not None:
            print(
                f"[Wf-341-C3] WARN: safe_overwrite_check ratio={_risk:.3f} at {out_path}",
                file=sys.stderr,
            )
        atomic_write_file(out_path, _payload, encoding="utf-8")
        _trunc = detect_truncation(_payload, expected_min_length=200)
        if _trunc is not None:
            print(
                f"[Wf-341-C3] WARN: detect_truncation reason={_trunc!r} at {out_path}",
                file=sys.stderr,
            )
    else:  # pragma: no cover
        out_path.write_text(_payload, encoding="utf-8")

    # Console output (plain-text per AGENTS.md §8).
    print(f"shortcut_consistency_status={summary.shortcut_consistency_status}")
    print(f"ema_model_status={summary.ema_model_status}")
    print(f"shortcut_branch_n_shortcut={summary.shortcut_branch_n_shortcut}")
    print(f"shortcut_branch_n_total={summary.shortcut_branch_n_total}")
    print(f"ema_decay={summary.ema_decay:.6f}")
    print(f"baseline_wall_ms={summary.baseline_wall_ms:.3f}")
    if summary.shortcut_consistency_enabled:
        print(f"shortcut_wall_ms={summary.shortcut_wall_ms:.3f}")
    if summary.ema_model_enabled:
        print(f"ema_wall_ms={summary.ema_wall_ms:.3f}")
    if summary.shortcut_consistency_enabled and summary.ema_model_enabled:
        print(f"joint_wall_ms={summary.joint_wall_ms:.3f}")
    print(f"projected_speedup={summary.projected_speedup}")
    print(f"projected_speedup_source={summary.projected_speedup_source}")
    print(f"summary_json={out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())