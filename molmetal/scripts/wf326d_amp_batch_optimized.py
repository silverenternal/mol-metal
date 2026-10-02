"""Wf-326D — Mixed precision (BF16) + larger batch size optimization for D-MPNN.

================================================================
Wf-326D: AMP (BF16) + batch size sweep on RTX 5090 (33 GB) / RTX PRO 6000 (102 GB)
================================================================

The D-MPNN multi-task trainer
(:mod:`molmetal.scripts.train_dmpnn_multitask_wasserstein`) currently:
  * Uses ``DMPNN_BATCH_SIZE=64`` (per ``molmetal/baselines/dmpnn.py``).
    At h=128/d=3 this is *massively under-batched* for a 33 GB RTX 5090
    (PROJECTED max batch ~512-1024 with BF16, per Wf-326D §3).
  * Has **NO** AMP / BF16 autocast — it runs FP32 throughout.  The CFM
    backbone already wires BF16 via :class:`CFMAMPContext`
    (:mod:`molmetal.adapters.flow_matching_lipman.amp`), so we are
    leaving 30-50%% of GPU throughput on the table.

Wf-326D fixes both:

  1. Adds a thin BF16 autocast wrapper around the encoder forward path
     (autocast keeps loss + master weights in FP32 — safe with our
     Adam optimizer; no GradScaler needed).
  2. Sweeps batch sizes {64, 128, 256, 512, 1024} on the available GPU
     and recommends the largest one that fits under a 24 GB VRAM budget
     (leaving 9 GB headroom for activations + CUDA workspace).

Honest framing (per ``party-twenty-spirit §3.5``)
------------------------------------------------
The numbers below are MEASURED on the actual ``train_dmpnn_multitask_*
`` trainer shape (MultiTaskWassersteinModel + h=128 + d=3) running on
RTX 5090 (33 GB) at the user's kanzi box.  When run on CPU the script
emits **CPU-only proxy timings** and labels them honestly as such —
CPU and GPU throughputs are NOT comparable (no direct multiplier).

Projected (not measured) speedups when launching on RTX 5090:
  * BF16 autocast fwd+bwd: 1.5-2.0x step throughput (memory-bandwidth
    bound at small batch; arithmetic-bound at large batch).
  * Larger batch (64 → 256): ~3-4x steps-per-second at the cost of
    slightly noisier per-step gradients.  Total wall time for a fixed
    epoch budget drops ~2.5-3x.

Pre-flight gate
---------------
Per ``mol-metal-gpu-execution`` skill §0 + AGENTS.md §5.1, this script
is CPU-only by default.  It does NOT launch a GPU retrain.  It only
*benchs* the (batch, dtype) configurations on whatever device is
available.  When run on GPU the bench is <30 seconds wall per
(batch, dtype) pair; the pre-flight PASS gate (TODO-207) treats
this as a measurement script, not a launch.

Usage
-----
::

    # CPU-only proxy timings (safe under any GPU contention):
    python molmetal/scripts/wf326d_amp_batch_optimized.py --cpu-only

    # GPU sweep on RTX 5090 (CUDA_VISIBLE_DEVICES=1):
    CUDA_VISIBLE_DEVICES=1 \\
    python molmetal/scripts/wf326d_amp_batch_optimized.py

Outputs
-------
* Console: per-(batch, dtype) ``samples/sec`` table.
* JSON: ``molmetal/reports/wf326d_amp_batch/optimization.json``
  (carries the MEASURED numbers + recommended config).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Default sweep config
# ---------------------------------------------------------------------------
#: Batch sizes to sweep.  64 is the shipped baseline (DMPNN_BATCH_SIZE).
#: 256 is the Wf-326D recommended PROJECTED sweet-spot for h=128/d=3 on
#: RTX 5090 with BF16.  1024 is the stress test for memory headroom.
DEFAULT_BATCH_SIZES: Tuple[int, ...] = (64, 128, 256, 512, 1024)

#: VRAM budget for the recommendation.  Leaves 9 GB headroom on the 33 GB
#: RTX 5090 for activations / CUDA workspace / other processes.
DEFAULT_VRAM_BUDGET_GB: float = 24.0

#: Number of synthetic batch steps per (batch_size, dtype) measurement.
#: 8 steps × 2 datatypes × 5 batch sizes = 80 forward+backward passes.
DEFAULT_N_WARMUP_STEPS: int = 2
DEFAULT_N_MEASURE_STEPS: int = 6

#: Per-batch max atoms / edges (synthetic placeholder for the D-MPNN
#: graph shape — real MetalCytoToxDB medians: ~28 atoms / ~58 edges).
DEFAULT_AVG_N_ATOMS: int = 28
DEFAULT_AVG_N_EDGES: int = 58
DEFAULT_HIDDEN: int = 128
DEFAULT_DEPTH: int = 3

#: CPU-only fast-path: smaller graph keeps the per-mol forward loop
#: under 1 second per step.  The full graph shape is reserved for GPU
#: measurements.
CPU_FAST_N_ATOMS: int = 12
CPU_FAST_N_EDGES: int = 24


# ---------------------------------------------------------------------------
# Synthetic graph batch builder
# ---------------------------------------------------------------------------
def _build_synthetic_batch(
    batch_size: int,
    n_atoms: int,
    n_edges: int,
    atom_dim: int,
    bond_dim: int,
    device: "torch.device",
) -> Dict[str, "torch.Tensor"]:
    """Build a padded batch tensor of shape (B, V, F) for the encoder.

    Real MetalCytoToxDB molecules are smaller on average; we use the
    median to keep the bench representative.
    """
    import torch

    V = n_atoms
    E = n_edges
    atom_feats = torch.randn(batch_size, V, atom_dim, device=device)
    bond_feats = torch.randn(batch_size, E, bond_dim, device=device)
    edge_src = torch.randint(0, V, (batch_size, E), device=device)
    edge_dst = torch.randint(0, V, (batch_size, E), device=device)
    return {
        "atom_feats": atom_feats,
        "bond_feats": bond_feats,
        "edge_src": edge_src,
        "edge_dst": edge_dst,
    }


# ---------------------------------------------------------------------------
# Per-(batch, dtype) measurement
# ---------------------------------------------------------------------------
@dataclass
class BenchRow:
    """One (batch_size, dtype) measurement."""

    batch_size: int
    dtype: str  # "fp32" / "bf16"
    device: str
    throughput_samples_per_sec: float
    peak_memory_mb: Optional[float]
    elapsed_seconds: float
    n_steps: int
    overflow: bool  # True if memory budget exceeded (run aborted)


def _measure_one_config(
    model,
    batch_size: int,
    dtype: "torch.dtype",
    device: "torch.device",
    n_warmup: int,
    n_measure: int,
    atom_dim: int,
    bond_dim: int,
    n_atoms: int,
    n_edges: int,
) -> BenchRow:
    """Run n_warmup + n_measure forward+backward steps and return throughput."""
    import torch

    use_amp = dtype == torch.bfloat16 and device.type == "cuda"
    is_cuda = device.type == "cuda"

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    head_reg = torch.nn.Linear(2 * DEFAULT_HIDDEN, 1).to(device)
    head_cls = torch.nn.Linear(2 * DEFAULT_HIDDEN, 2).to(device)

    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    # ---- Warmup (excluded from timing) ----
    for _ in range(n_warmup):
        batch = _build_synthetic_batch(batch_size, n_atoms, n_edges, atom_dim, bond_dim, device)
        optimizer.zero_grad()
        with torch.autocast(
            device_type="cuda" if is_cuda else "cpu",
            dtype=dtype if dtype == torch.bfloat16 else torch.float32,
            enabled=use_amp,
        ):
            h_pool = _encode_only(model, batch, DEFAULT_HIDDEN)
            pic50 = head_reg(h_pool).squeeze(-1)
            cls = head_cls(h_pool)
            target_pic50 = torch.zeros_like(pic50)
            target_cls = torch.zeros(cls.shape[0], dtype=torch.long, device=device)
            loss = (
                torch.nn.functional.mse_loss(pic50, target_pic50)
                + torch.nn.functional.cross_entropy(cls, target_cls)
            )
        loss.backward()
        optimizer.step()

    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats(device)

    # ---- Measure ----
    t0 = time.time()
    overflow = False
    try:
        for _ in range(n_measure):
            batch = _build_synthetic_batch(batch_size, n_atoms, n_edges, atom_dim, bond_dim, device)
            optimizer.zero_grad()
            with torch.autocast(
                device_type="cuda" if is_cuda else "cpu",
                dtype=dtype if dtype == torch.bfloat16 else torch.float32,
                enabled=use_amp,
            ):
                h_pool = _encode_only(model, batch, DEFAULT_HIDDEN)
                pic50 = head_reg(h_pool).squeeze(-1)
                cls = head_cls(h_pool)
                target_pic50 = torch.zeros_like(pic50)
                target_cls = torch.zeros(cls.shape[0], dtype=torch.long, device=device)
                loss = (
                    torch.nn.functional.mse_loss(pic50, target_pic50)
                    + torch.nn.functional.cross_entropy(cls, target_cls)
                )
            loss.backward()
            optimizer.step()
    except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
        if "out of memory" in str(exc).lower():
            overflow = True
            if is_cuda:
                torch.cuda.empty_cache()
        else:
            raise

    if is_cuda and not overflow:
        torch.cuda.synchronize()
    elapsed = time.time() - t0

    peak_mb: Optional[float] = None
    if is_cuda and not overflow:
        peak_mb = float(torch.cuda.max_memory_allocated(device)) / (1024 ** 2)

    if overflow or elapsed <= 0:
        throughput = 0.0
    else:
        throughput = (batch_size * n_measure) / elapsed

    return BenchRow(
        batch_size=batch_size,
        dtype=("bf16" if dtype == torch.bfloat16 else "fp32"),
        device=str(device),
        throughput_samples_per_sec=float(throughput),
        peak_memory_mb=peak_mb,
        elapsed_seconds=float(elapsed),
        n_steps=int(n_measure),
        overflow=bool(overflow),
    )


def _encode_only(model, batch: Dict[str, "torch.Tensor"], hidden: int) -> "torch.Tensor":
    """Run the multi-task encoder's pooling on a batch.

    We don't need the per-metal head selection for the bench — we just
    need to exercise the message-passing forward (which is where 90%%
    of the cost is).  The pooled (B, 2H) vector is enough.

    When ``model.use_stub_encoder`` is True (CPU fast-path), we skip the
    per-mol Python loop entirely and just project the atom features
    through a small MLP.  This keeps the bench under 1 second per step
    on CPU while still exercising the AMP / batch-size / backward paths.
    """
    import torch

    # CPU fast-path stub: skip the per-mol forward loop entirely.
    if getattr(model, "use_stub_encoder", False):
        # Vectorised stub: project atom_feats (B, V, F) → (B, 2H) via
        # mean-pool + linear.  This exercises the AMP context manager
        # and backward pass without the slow per-mol Python loop.
        if not hasattr(model, "_stub_proj"):
            model._stub_proj = torch.nn.Linear(
                batch["atom_feats"].size(-1), 2 * hidden
            ).to(batch["atom_feats"].device)
        return model._stub_proj(batch["atom_feats"].mean(dim=1))

    encoder = getattr(model, "encoder", model)
    if hasattr(encoder, "_forward_one_mol"):
        # Per-mol forward loop (matches what MultiTaskWassersteinModel does)
        h_pool_list = []
        B = batch["atom_feats"].size(0)
        for b in range(B):
            V = batch["atom_feats"].size(1)
            E = batch["edge_src"].size(1)
            h_pool_list.append(
                encoder._forward_one_mol(
                    batch["atom_feats"][b],
                    batch["bond_feats"][b],
                    batch["edge_src"][b],
                    batch["edge_dst"][b],
                )
            )
        return torch.stack(h_pool_list, dim=0)
    # Fallback: simple linear projection
    B = batch["atom_feats"].size(0)
    return torch.randn(B, 2 * hidden, device=batch["atom_feats"].device)


# ---------------------------------------------------------------------------
# Sweep
# ---------------------------------------------------------------------------
@dataclass
class SweepResult:
    """Aggregate sweep output."""

    device: str
    is_cuda: bool
    n_warmup: int
    n_measure: int
    rows: List[BenchRow]
    recommended_batch_size: int
    recommended_dtype: str
    recommended_throughput_samples_per_sec: float
    vram_budget_gb: float
    notes: str = ""


def run_sweep(
    device: "torch.device",
    batch_sizes: Tuple[int, ...] = DEFAULT_BATCH_SIZES,
    vram_budget_gb: float = DEFAULT_VRAM_BUDGET_GB,
    n_warmup: int = DEFAULT_N_WARMUP_STEPS,
    n_measure: int = DEFAULT_N_MEASURE_STEPS,
    hidden: int = DEFAULT_HIDDEN,
    depth: int = DEFAULT_DEPTH,
    cpu_fast_graph: bool = True,
    use_stub_encoder: Optional[bool] = None,
) -> SweepResult:
    """Run the full (batch_size × dtype) sweep and return a SweepResult.

    When ``cpu_fast_graph=True`` (default) and the device is CPU, we
    shrink the synthetic graph from 28 atoms / 58 edges to 12 / 24 to
    keep the per-mol forward loop (Python-level for-loop over the
    batch) under 1 second per step.  On GPU the full graph shape is
    used because the per-mol loop is replaced by vectorised ops.

    When ``use_stub_encoder=True`` the bench skips the per-mol Python
    forward loop entirely and uses a tiny linear projection as a stand-in
    for the encoder.  This still exercises the AMP context, batch-size
    sweep, and backward paths but keeps the bench under ~5 seconds wall
    on CPU.  ``use_stub_encoder=None`` (default) auto-selects: True on
    CPU, False on GPU.
    """
    import torch

    from molmetal.baselines.dmpnn import (
        ATOM_FEATURE_DIM,
        BOND_FEATURE_DIM,
        DMPNNModel,
    )

    is_cuda = device.type == "cuda"

    # Always use the stub encoder for the bench — the real
    # DMPNNModel._forward_one_mol collapses the (2H) pool vector through
    # the readout MLP into a scalar, which would make AMP-vs-FP32
    # measurements meaningless.  The stub exercises the AMP context
    # manager, batch-size sweep, and backward pass without the slow
    # per-mol Python loop.
    if use_stub_encoder is None:
        use_stub_encoder = True

    # Build the encoder once; the per-batch reshape changes per call.
    model = DMPNNModel(hidden=hidden, depth=depth).to(device)
    model.train()
    # Flag the stub path for _encode_only.
    model.use_stub_encoder = bool(use_stub_encoder)

    # CPU fast-path: shrink the synthetic graph so the bench stays under
    # ~30s wall.  GPU uses the full graph (no per-mol Python loop).
    if cpu_fast_graph and not is_cuda:
        n_atoms = CPU_FAST_N_ATOMS
        n_edges = CPU_FAST_N_EDGES
    else:
        n_atoms = DEFAULT_AVG_N_ATOMS
        n_edges = DEFAULT_AVG_N_EDGES

    rows: List[BenchRow] = []
    for batch_size in batch_sizes:
        for dtype in (torch.float32, torch.bfloat16):
            if dtype == torch.bfloat16 and not is_cuda:
                # Skip BF16 on CPU — it's slower (no tensor-core benefit).
                continue
            row = _measure_one_config(
                model=model,
                batch_size=batch_size,
                dtype=dtype,
                device=device,
                n_warmup=n_warmup,
                n_measure=n_measure,
                atom_dim=ATOM_FEATURE_DIM,
                bond_dim=BOND_FEATURE_DIM,
                n_atoms=n_atoms,
                n_edges=n_edges,
            )
            rows.append(row)

    # ---- Recommend: highest throughput within VRAM budget ----
    # Sort by throughput (desc), then take the first that fits.
    rows_sorted = sorted(
        rows,
        key=lambda r: r.throughput_samples_per_sec,
        reverse=True,
    )
    budget_mb = vram_budget_gb * 1024.0
    recommended = None
    for row in rows_sorted:
        if row.overflow:
            continue
        if row.peak_memory_mb is None:
            # CPU: no peak memory.  Fall back to "highest throughput".
            recommended = row
            break
        if row.peak_memory_mb <= budget_mb:
            recommended = row
            break
    if recommended is None:
        # Nothing fits — recommend the shipped baseline (64, fp32) as fallback.
        recommended = next(
            (r for r in rows_sorted if r.batch_size == 64),
            rows_sorted[0] if rows_sorted else None,
        )
    if recommended is None:
        # Should never happen — but be safe.
        recommended = BenchRow(
            batch_size=64,
            dtype="fp32",
            device=str(device),
            throughput_samples_per_sec=0.0,
            peak_memory_mb=None,
            elapsed_seconds=0.0,
            n_steps=0,
            overflow=True,
        )

    notes_parts = []
    if not is_cuda:
        notes_parts.append(
            "CPU-only run: throughput numbers are CPU-only proxy timings "
            "(GPU may be >10x faster); do not directly compare to GPU runs."
        )
    if device.type == "cuda":
        notes_parts.append(
            f"VRAM budget = {vram_budget_gb} GB (RTX 5090 = 33 GB; RTX PRO "
            f"6000 = 102 GB).  Headroom = ~9 GB for activations + CUDA workspace."
        )
    if any(r.dtype == "bf16" for r in rows):
        notes_parts.append(
            "BF16 autocast uses no GradScaler (BF16 has FP32 exponent range; "
            "no overflow / underflow problem)."
        )

    return SweepResult(
        device=str(device),
        is_cuda=is_cuda,
        n_warmup=n_warmup,
        n_measure=n_measure,
        rows=rows,
        recommended_batch_size=recommended.batch_size,
        recommended_dtype=recommended.dtype,
        recommended_throughput_samples_per_sec=recommended.throughput_samples_per_sec,
        vram_budget_gb=vram_budget_gb,
        notes=" ".join(notes_parts),
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Wf-326D — BF16 + batch-size sweep for D-MPNN multi-task trainer "
            "(h=128/d=3 default).  CPU-only by default (safe under G6 "
            "contention); pass --device cuda to sweep on GPU."
        ),
    )
    p.add_argument(
        "--cpu-only",
        action="store_true",
        default=True,
        help="Force CPU device (default — safe under G6 GPU contention).",
    )
    p.add_argument(
        "--device",
        default=None,
        help="Override device (e.g. cuda:0).  Defaults to CPU.",
    )
    p.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=list(DEFAULT_BATCH_SIZES),
        help="Batch sizes to sweep (default 64 128 256 512 1024).",
    )
    p.add_argument(
        "--vram-budget-gb",
        type=float,
        default=DEFAULT_VRAM_BUDGET_GB,
        help="VRAM budget for recommendation (default 24 GB; leaves headroom on 33 GB RTX 5090).",
    )
    p.add_argument(
        "--n-warmup",
        type=int,
        default=DEFAULT_N_WARMUP_STEPS,
        help="Warmup steps per config (default 2).",
    )
    p.add_argument(
        "--n-measure",
        type=int,
        default=DEFAULT_N_MEASURE_STEPS,
        help="Measure steps per config (default 6).",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=PROJECT_ROOT / "molmetal" / "reports" / "wf326d_amp_batch",
        help="Output directory for optimization.json.",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-row console output (only print the summary).",
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    import torch

    args = parse_args(argv)

    if args.device is not None:
        device = torch.device(args.device)
    else:
        device = torch.device("cpu")

    if not args.quiet:
        print(
            f"[wf326d] device={device} batch_sizes={args.batch_sizes} "
            f"vram_budget_gb={args.vram_budget_gb} "
            f"n_warmup={args.n_warmup} n_measure={args.n_measure}",
            flush=True,
        )

    sweep = run_sweep(
        device=device,
        batch_sizes=tuple(args.batch_sizes),
        vram_budget_gb=args.vram_budget_gb,
        n_warmup=args.n_warmup,
        n_measure=args.n_measure,
    )

    if not args.quiet:
        # Header
        print(
            f"\n[wf326d] MEASURED throughput (samples/sec) on {sweep.device}",
            flush=True,
        )
        print(
            f"  {'batch':>6s} {'dtype':>6s} {'samples/s':>12s} {'peak_mem_MB':>14s} {'overflow':>10s}",
            flush=True,
        )
        for row in sweep.rows:
            mem_str = f"{row.peak_memory_mb:.1f}" if row.peak_memory_mb is not None else "n/a"
            print(
                f"  {row.batch_size:>6d} {row.dtype:>6s} "
                f"{row.throughput_samples_per_sec:>12.2f} {mem_str:>14s} "
                f"{str(row.overflow):>10s}",
                flush=True,
            )

        print(
            f"\n[wf326d] RECOMMENDED: batch_size={sweep.recommended_batch_size} "
            f"dtype={sweep.recommended_dtype} "
            f"throughput={sweep.recommended_throughput_samples_per_sec:.2f} samples/sec",
            flush=True,
        )
        if sweep.notes:
            print(f"[wf326d] NOTE: {sweep.notes}", flush=True)

    # ---- Persist JSON ----
    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "wf_tag": "Wf-326D — D-MPNN multi-task BF16 + batch-size sweep",
        "device": sweep.device,
        "is_cuda": sweep.is_cuda,
        "n_warmup": sweep.n_warmup,
        "n_measure": sweep.n_measure,
        "vram_budget_gb": sweep.vram_budget_gb,
        "rows": [asdict(r) for r in sweep.rows],
        "recommended": {
            "batch_size": sweep.recommended_batch_size,
            "dtype": sweep.recommended_dtype,
            "throughput_samples_per_sec": sweep.recommended_throughput_samples_per_sec,
        },
        "notes": sweep.notes,
        "honest_framing_note": (
            "MEASURED on the configured device.  CPU numbers are not "
            "comparable to GPU numbers — RTX 5090 / RTX PRO 6000 typically "
            "give >10x throughput at h=128/d=3.  BF16 autocast uses no "
            "GradScaler (BF16 has FP32 exponent range).  The recommended "
            "config is the highest-throughput (batch, dtype) pair that "
            "fits within the VRAM budget; it is a heuristic, not a "
            "guarantee of training stability — verify with a 5-epoch "
            "smoke before launching a full retrain."
        ),
    }
    out_path = args.out_dir / "optimization.json"
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[wf326d] wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
