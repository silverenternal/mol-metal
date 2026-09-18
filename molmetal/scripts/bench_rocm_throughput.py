"""P5 ROCm throughput benchmark on AMD RX 7800 XT.

Goal: validate the predictions in TODO/05_compute/7800xt_budget.md against
the ACTUAL measured throughput of the kernels / models that drive the
Mol-Metal training pipeline.

Sections
--------
(a) FP32 matmul 1024^3                 — sanity roofline (fp32 peak ~52 TF/s)
(b) BF16 matmul 2048^3                 — peak with mixed-precision (bf16 ~104 TF/s)
(c) FP32 matmul 4096^3                 — memory-bound fallback (slow!)
(d) D-MPNN forward on 100 mols × 30 atoms, batch=10
(e) EGNN layer forward + backward on 50 graphs × 20 nodes
(f) LipmanFlowMatchingAdapter.train_step on 16 mols × 20 atoms, batch=4
(g) Triton scatter_sum (from models._scatter) on (N=8192, R=2048)

Each section runs 3 warmup iterations + 20 timed iterations using
``torch.cuda.Event`` (works for both AMD ROCm and NVIDIA CUDA back-ends).
Reports mean / std / throughput:

* matmul → GFLOP/s   (2 × M × N × K, no FMA factor)
* nn      → samples/sec
* scatter → edges/sec

Usage
-----

::

    source .venv/bin/activate && cd /home/hugo/codes/try_triton_on_rocm
    python -m molmetal.scripts.bench_rocm_throughput 2>&1 | tee /tmp/bench.log

Exit code 0 regardless of measured throughput — this script is a probe,
not a test.  See molmetal/tests/test_rocm_throughput.py for the
hard-failing sanity assertions.
"""

from __future__ import annotations

import argparse
import os
import platform
import subprocess
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch

# Make sure the project root is on sys.path so `models._scatter` and
# `models.velocity_net` resolve without an installed `molmetal` package.
_PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from molmetal.utils.device import (
    DEFAULT_DEVICE,
    ROCM_AVAILABLE,
    get_device,
    verify_rocm_active,
)

# Try to import the velocity_net — used by (e) and (f).  We import
# lazily so this script can still print GPU info even if the optional
# model code is broken on the current machine.
try:
    from models.velocity_net import EGNNLayer  # type: ignore

    _HAS_VELOCITY_NET = True
except Exception as exc:  # pragma: no cover - depends on env
    EGNNLayer = None  # type: ignore
    _HAS_VELOCITY_NET = False
    _VELOCITY_NET_IMPORT_ERROR: Optional[Exception] = exc

# Try to import the Lipman FM adapter for (f).  Lazy: the adapter pulls
# the cloned flow_matching repo, which may not be on disk in CI.
try:
    from molmetal.adapters.flow_matching_lipman import (
        LipmanFlowMatchingAdapter,  # type: ignore
    )

    _HAS_LIPMAN = True
except Exception as exc:  # pragma: no cover - depends on env
    LipmanFlowMatchingAdapter = None  # type: ignore
    _HAS_LIPMAN = False
    _LIPMAN_IMPORT_ERROR: Optional[Exception] = exc

# Try to import the Triton scatter kernel for (g).
try:
    from models._scatter import (  # type: ignore
        scatter_sum,
        scatter_backend,
        set_scatter_backend,
    )

    _HAS_SCATTER = True
except Exception as exc:  # pragma: no cover - depends on env
    scatter_sum = None  # type: ignore
    scatter_backend = "unavailable"  # type: ignore
    set_scatter_backend = None  # type: ignore
    _HAS_SCATTER = False
    _SCATTER_IMPORT_ERROR: Optional[Exception] = exc

# Try to import D-MPNN for (d).  The molmetal copy lives at
# molmetal.models.dmpnn.DirectedMPNN — uses a manual _scatter_sum.
try:
    from molmetal.models.dmpnn import DirectedMPNN  # type: ignore

    _HAS_DMPNN = True
except Exception as exc:  # pragma: no cover - depends on env
    DirectedMPNN = None  # type: ignore
    _HAS_DMPNN = False
    _DMPNN_IMPORT_ERROR: Optional[Exception] = exc


# ---------------------------------------------------------------------------
# Benchmark helpers
# ---------------------------------------------------------------------------
@dataclass
class BenchResult:
    """One timed block — mean / std / throughput."""

    name: str
    n_iters: int
    mean_s: float
    std_s: float
    min_s: float
    max_s: float
    throughput: Optional[float] = None
    throughput_unit: str = ""
    extras: dict = field(default_factory=dict)

    def __str__(self) -> str:
        msg = (
            f"  {self.name:<60s}  "
            f"mean={self.mean_s * 1000:8.3f} ms  "
            f"std={self.std_s * 1000:6.3f} ms  "
            f"min={self.min_s * 1000:6.3f}  "
            f"max={self.max_s * 1000:6.3f}"
        )
        if self.throughput is not None:
            msg += f"   | {self.throughput:>10.2f} {self.throughput_unit}"
        return msg


def _matmul_flops(M: int, N: int, K: int) -> int:
    """2 × M × N × K FLOPs (one multiply + one add per output element)."""
    return 2 * M * N * K


def _time_block(
    fn: Callable[[], None],
    n_warmup: int = 3,
    n_iters: int = 20,
) -> List[float]:
    """Time ``fn()`` with ``torch.cuda.Event`` if available, else wall-clock.

    Returns the per-iteration latency in seconds (length ``n_iters``).
    Includes a ``torch.cuda.synchronize()`` before & after so the timings
    are well-defined on ROCm.
    """
    use_cuda_events = torch.cuda.is_available()
    if use_cuda_events:
        # Warmup
        for _ in range(n_warmup):
            fn()
        torch.cuda.synchronize()
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(n_iters)]
        for i in range(n_iters):
            starts[i].record()
            fn()
            ends[i].record()
        torch.cuda.synchronize()
        return [s.elapsed_time(e) / 1000.0 for s, e in zip(starts, ends)]
    # CPU fallback — wall clock
    for _ in range(n_warmup):
        fn()
    latencies: List[float] = []
    for _ in range(n_iters):
        t0 = time.perf_counter()
        fn()
        latencies.append(time.perf_counter() - t0)
    return latencies


def _summarise(
    name: str,
    latencies: List[float],
    throughput: Optional[float] = None,
    unit: str = "",
    extras: Optional[dict] = None,
) -> BenchResult:
    import math
    mean = sum(latencies) / len(latencies)
    var = sum((t - mean) ** 2 for t in latencies) / max(1, len(latencies) - 1)
    std = math.sqrt(var)
    return BenchResult(
        name=name,
        n_iters=len(latencies),
        mean_s=mean,
        std_s=std,
        min_s=min(latencies),
        max_s=max(latencies),
        throughput=throughput,
        throughput_unit=unit,
        extras=extras or {},
    )


def _sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


# ---------------------------------------------------------------------------
# GPU info
# ---------------------------------------------------------------------------
def _gpu_info_block() -> dict:
    """Print and return basic GPU + ROCm info."""
    info: dict = {}
    info["torch_version"] = torch.__version__
    info["cuda_available"] = torch.cuda.is_available()
    info["device_count"] = torch.cuda.device_count()
    if torch.cuda.is_available():
        info["device_0_name"] = torch.cuda.get_device_name(0)
        info["device_0_capability"] = torch.cuda.get_device_capability(0)
        info["hip_version"] = getattr(torch.version, "hip", None)
        info["cuda_version"] = getattr(torch.version, "cuda", None)
        # Free / total VRAM (MiB)
        free, total = torch.cuda.mem_get_info(0)
        info["vram_free_mib"] = int(free // (1024 * 1024))
        info["vram_total_mib"] = int(total // (1024 * 1024))
    info["python_version"] = platform.python_version()
    info["platform"] = platform.platform()
    info["default_device"] = DEFAULT_DEVICE
    info["rocm_available"] = ROCM_AVAILABLE

    # Try to surface rocm-smi / rocminfo
    try:
        rocminfo = subprocess.check_output(["rocminfo"], stderr=subprocess.DEVNULL, timeout=2).decode(errors="replace")
        for line in rocminfo.splitlines():
            if "Marketing Name" in line or "Marketing Name:" in line:
                info.setdefault("rocminfo_marketing", line.strip())
            if "Driver Version" in line or "ROCk Module Version" in line:
                info.setdefault("rocminfo", line.strip())
    except (FileNotFoundError, subprocess.SubprocessError, subprocess.TimeoutExpired):
        pass
    try:
        rocm_smi = subprocess.check_output(["rocm-smi", "--showdriverversion"], stderr=subprocess.DEVNULL, timeout=2).decode(errors="replace")
        info["rocm_smi_driver"] = rocm_smi.strip()
    except (FileNotFoundError, subprocess.SubprocessError, subprocess.TimeoutExpired):
        pass
    return info


# ---------------------------------------------------------------------------
# Benchmark sections
# ---------------------------------------------------------------------------
def bench_matmul(M: int, N: int, K: int, dtype: torch.dtype = torch.float32) -> BenchResult:
    device = get_device()
    a = torch.randn(M, K, device=device, dtype=dtype)
    b = torch.randn(K, N, device=device, dtype=dtype)
    # Warmup op so cuBLAS / hipBLASLt picks the best kernel for this size
    for _ in range(3):
        c = torch.matmul(a, b)
    _sync()
    latencies = _time_block(lambda: torch.matmul(a, b))
    flops = _matmul_flops(M, N, K)
    mean = sum(latencies) / len(latencies)
    tflops = flops / mean / 1e12
    return _summarise(
        f"matmul {M}×{K} @ {K}×{N}  dtype={str(dtype).replace('torch.', '')}",
        latencies,
        throughput=tflops,
        unit="TFLOP/s",
        extras={"M": M, "N": N, "K": K, "dtype": str(dtype), "gflops": flops / 1e9},
    )


def bench_dmpnn(n_mols: int = 100, n_atoms: int = 30, batch_size: int = 10) -> BenchResult:
    if not _HAS_DMPNN:
        return _summarise(
            f"D-MPNN forward {n_mols} mols × {n_atoms} atoms (batch={batch_size}) — SKIPPED (DirectedMPNN unavailable)",
            [0.0],
            throughput=0.0,
            unit="samples/s",
            extras={"reason": str(_DMPNN_IMPORT_ERROR)},
        )
    device = get_device()
    model = DirectedMPNN().to(device).eval()
    # Fully-connected directed edges (no self-loops): E = N*(N-1) per mol
    src = torch.arange(n_atoms, device=device).view(1, -1, 1).expand(batch_size, n_atoms, n_atoms)
    dst = torch.arange(n_atoms, device=device).view(1, 1, -1).expand(batch_size, n_atoms, n_atoms)
    mask = src != dst
    src_flat = src[mask].view(batch_size, -1)
    dst_flat = dst[mask].view(batch_size, -1)
    edge_index = torch.stack([src_flat, dst_flat], dim=1)  # (B, 2, E)
    h_atom = torch.randn(batch_size, n_atoms, 39, device=device)
    edge_attr = torch.randn(batch_size, edge_index.shape[-1], 6, device=device)
    batch_idx = torch.arange(batch_size, device=device).view(-1, 1).expand(-1, n_atoms).contiguous()

    n_batches = n_mols // batch_size

    def step() -> None:
        with torch.no_grad():
            for _ in range(n_batches):
                _ = model(h_atom, edge_index, edge_attr, batch_idx)

    _sync()
    latencies = _time_block(step)
    mean = sum(latencies) / len(latencies)
    samples_per_sec = n_mols / mean
    return _summarise(
        f"D-MPNN forward {n_mols} mols × {n_atoms} atoms, batch={batch_size}",
        latencies,
        throughput=samples_per_sec,
        unit="samples/s",
        extras={"n_mols": n_mols, "n_atoms": n_atoms, "batch_size": batch_size},
    )


def bench_egnn(n_graphs: int = 50, n_nodes: int = 20) -> BenchResult:
    """One round of EGNNLayer fwd+bwd on a fully-connected batch of graphs.

    EGNNLayer.forward signature is ``forward(h_node, positions, edge_index,
    cond_per_node=None, edge_mask=None)`` with batched shapes
    ``(B, N, H)`` / ``(B, N, 3)`` / ``(B, 2, E)``.  We treat every graph
    as a fully-connected directed graph (no self-loops) ⇒ E = N*(N-1).
    """
    if not _HAS_VELOCITY_NET:
        return _summarise(
            f"EGNN forward+backward {n_graphs} graphs × {n_nodes} nodes — SKIPPED (EGNNLayer unavailable)",
            [0.0],
            throughput=0.0,
            unit="graphs/s",
            extras={"reason": str(_VELOCITY_NET_IMPORT_ERROR)},
        )
    device = get_device()
    layer = EGNNLayer(hidden_dim=128).to(device)
    n_edges = n_nodes * (n_nodes - 1)
    # Build per-graph edges as (B, 2, E): src/dst live in [0, N).
    idx = torch.arange(n_nodes, device=device)
    src = idx.view(1, -1, 1).expand(n_graphs, n_nodes, n_nodes)
    dst = idx.view(1, 1, -1).expand(n_graphs, n_nodes, n_nodes)
    mask = src != dst
    edge_index = torch.stack(
        [src[mask].view(n_graphs, -1), dst[mask].view(n_graphs, -1)], dim=1
    )  # (B, 2, E)

    h = torch.randn(n_graphs, n_nodes, 128, device=device, requires_grad=True)
    x = torch.randn(n_graphs, n_nodes, 3, device=device, requires_grad=True)
    # The EGNNLayer's update_mlp is sized 3H+1 → it requires cond_per_node
    # matching h_node (B, N, H) when no condition is supplied we still
    # need to pass *something* of shape (B, N, H).  We use a zero cond
    # here so the layer's output equals the unconditional update.
    cond = torch.zeros(n_graphs, n_nodes, 128, device=device)
    target_v = torch.randn(n_graphs, n_nodes, 3, device=device)
    target_h = torch.randn(n_graphs, n_nodes, 128, device=device)

    def step() -> None:
        if h.grad is not None:
            h.grad = None
        if x.grad is not None:
            x.grad = None
        h_out, v_agg = layer(h, x, edge_index, cond_per_node=cond)
        loss = ((v_agg - target_v) ** 2).mean() + ((h_out - target_h) ** 2).mean()
        loss.backward()

    _sync()
    latencies = _time_block(step)
    mean = sum(latencies) / len(latencies)
    return _summarise(
        f"EGNNLayer fwd+bwd {n_graphs} graphs × {n_nodes} nodes",
        latencies,
        throughput=n_graphs / mean,
        unit="graphs/s",
        extras={"n_graphs": n_graphs, "n_nodes": n_nodes},
    )


def bench_lipman_train_step(n_mols: int = 16, n_atoms: int = 20, batch_size: int = 4) -> BenchResult:
    if not _HAS_LIPMAN:
        return _summarise(
            f"LipmanFlowMatchingAdapter.train_step {n_mols} mols × {n_atoms} atoms (batch={batch_size}) — SKIPPED (adapter unavailable)",
            [0.0],
            throughput=0.0,
            unit="samples/s",
            extras={"reason": str(_LIPMAN_IMPORT_ERROR)},
        )
    device = get_device()
    # The FM adapter ships with hidden_dim=128, n_layers=3 — that is the
    # production size from the budget doc, so we keep it.
    adapter = LipmanFlowMatchingAdapter(hidden_dim=64, n_layers=2, lr=1e-3)
    try:
        adapter.setup(device=str(device))
    except Exception as exc:
        return _summarise(
            f"LipmanFlowMatchingAdapter.train_step — SKIPPED (setup failed: {exc})",
            [0.0],
            throughput=0.0,
            unit="samples/s",
            extras={"reason": str(exc)},
        )

    from molmetal.domain import Molecule
    torch.manual_seed(0)
    mols: list = []
    for i in range(batch_size):
        mols.append(Molecule(
            coords=torch.randn(n_atoms, 3, dtype=torch.float32),
            atom_types=torch.randint(1, 10, (n_atoms,), dtype=torch.long),
            bonds=torch.zeros(2, 0, dtype=torch.long),
            bond_types=torch.zeros(0, dtype=torch.long),
            formal_charges=torch.zeros(n_atoms, dtype=torch.long),
        ))

    n_steps = n_mols // batch_size

    def step() -> None:
        for _ in range(n_steps):
            adapter.train_step(pocket=None, mols=mols)

    _sync()
    latencies = _time_block(step)
    mean = sum(latencies) / len(latencies)
    return _summarise(
        f"LipmanFlowMatchingAdapter.train_step {n_mols} mols × {n_atoms} atoms (batch={batch_size})",
        latencies,
        throughput=n_mols / mean,
        unit="samples/s",
        extras={"n_mols": n_mols, "n_atoms": n_atoms, "batch_size": batch_size},
    )


def bench_scatter_sum(n_atoms: int = 8192, rbf_dim: int = 32, repeat: int = 2048) -> BenchResult:
    if not _HAS_SCATTER:
        return _summarise(
            f"models._scatter.scatter_sum N={n_atoms}, R={repeat} — SKIPPED (kernel unavailable)",
            [0.0],
            throughput=0.0,
            unit="edges/s",
            extras={"reason": str(_SCATTER_IMPORT_ERROR)},
        )
    device = get_device()
    # N atoms, each with R neighbours on average ⇒ N*R edges.
    # Build a random (E, F) feature tensor + (E,) src/dst indices.
    n_edges = n_atoms * repeat
    src = torch.randint(0, n_atoms, (n_edges,), device=device)
    dst = torch.randint(0, n_atoms, (n_edges,), device=device)
    features = torch.randn(n_edges, rbf_dim, device=device, requires_grad=True)

    def step() -> None:
        if features.grad is not None:
            features.grad = None
        out = scatter_sum(src, dst, features, n_atoms=n_atoms)
        # A no-op loss so autograd kicks in.
        out.sum().backward()

    _sync()
    latencies = _time_block(step)
    mean = sum(latencies) / len(latencies)
    return _summarise(
        f"models._scatter.scatter_sum N={n_atoms}, R={repeat} (E={n_edges}, F={rbf_dim}) backend={scatter_backend}",
        latencies,
        throughput=n_edges / mean,
        unit="edges/s",
        extras={
            "n_atoms": n_atoms,
            "repeat": repeat,
            "n_edges": n_edges,
            "rbf_dim": rbf_dim,
            "backend": scatter_backend,
        },
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="P5 ROCm throughput benchmark on RX 7800 XT.",
    )
    p.add_argument(
        "--scatter-backend",
        choices=("auto", "torch", "triton"),
        default="auto",
        help=(
            "Force the scatter_sum backend.  'auto' picks the "
            "architecture-aware default (torch on RDNA3 unless shape "
            "exceeds threshold, triton on CDNA unless shape is tiny)."
        ),
    )
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    if _HAS_SCATTER and set_scatter_backend is not None:
        if args.scatter_backend == "auto":
            set_scatter_backend(None)
        else:
            set_scatter_backend(args.scatter_backend)
    print("=" * 90)
    print("P5 ROCm Throughput Benchmark — Mol-Metal on RX 7800 XT")
    print("=" * 90)
    info = _gpu_info_block()
    for k, v in info.items():
        print(f"  {k:<24s}: {v}")
    # Reuse the canonical probe too
    probe = verify_rocm_active()
    print(f"  {'verify_rocm_active':<24s}: roc_active={probe.get('roc_active')}, "
          f"device_count={probe.get('device_count')}, "
          f"hip_version={probe.get('hip_version')}")
    print(f"  {'scatter-backend':<24s}: {args.scatter_backend} "
          f"(resolved={scatter_backend})")

    # Sectioned benchmarks
    results: List[BenchResult] = []

    print()
    print("-" * 90)
    print("(a) FP32 matmul 1024^3")
    print("-" * 90)
    results.append(bench_matmul(1024, 1024, 1024, dtype=torch.float32))
    print(results[-1])

    print()
    print("-" * 90)
    print("(b) BF16 matmul 2048^3")
    print("-" * 90)
    results.append(bench_matmul(2048, 2048, 2048, dtype=torch.bfloat16))
    print(results[-1])

    print()
    print("-" * 90)
    print("(c) FP32 matmul 4096^3")
    print("-" * 90)
    results.append(bench_matmul(4096, 4096, 4096, dtype=torch.float32))
    print(results[-1])

    print()
    print("-" * 90)
    print("(d) D-MPNN forward on 100 mols × 30 atoms, batch=10")
    print("-" * 90)
    results.append(bench_dmpnn(n_mols=100, n_atoms=30, batch_size=10))
    print(results[-1])

    print()
    print("-" * 90)
    print("(e) EGNNLayer forward + backward on 50 graphs × 20 nodes")
    print("-" * 90)
    results.append(bench_egnn(n_graphs=50, n_nodes=20))
    print(results[-1])

    print()
    print("-" * 90)
    print("(f) LipmanFlowMatchingAdapter.train_step on 16 mols × 20 atoms, batch=4")
    print("-" * 90)
    results.append(bench_lipman_train_step(n_mols=16, n_atoms=20, batch_size=4))
    print(results[-1])

    print()
    print("-" * 90)
    print("(g) Triton scatter_sum (from models._scatter) on (N=8192, R=2048)")
    print("-" * 90)
    results.append(bench_scatter_sum(n_atoms=8192, rbf_dim=32, repeat=2048))
    print(results[-1])

    print()
    print("=" * 90)
    print("Summary (machine-readable)")
    print("=" * 90)
    import json
    payload = {
        "gpu_info": info,
        "results": [
            {
                "name": r.name,
                "n_iters": r.n_iters,
                "mean_s": r.mean_s,
                "std_s": r.std_s,
                "min_s": r.min_s,
                "max_s": r.max_s,
                "throughput": r.throughput,
                "throughput_unit": r.throughput_unit,
                "extras": r.extras,
            }
            for r in results
        ],
    }
    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
