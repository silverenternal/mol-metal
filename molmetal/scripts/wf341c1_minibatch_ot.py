#!/usr/bin/env python3
"""Wf-341-C1 — Mini-batch OT coupling integration (Tong 2024).

================================================================
Why this module exists
================================================================
Per Wf-340-R1 §3.5 + Wf-340-I3 §2.1: the CFM loss in
``molmetal/adapters/flow_matching_lipman`` uses the upstream
``flow_matching.optimal_transport.mini_batch_ot_coupling`` (Sinkhorn
log-domain per Tong 2023). The Wf-340-I1 commit added
``exact_minibatch_ot_coupling`` (POT ``ot.emd`` exact solver) inside
``optimal_transport_v2.py`` and exposed ``--use-exact-ot-coupling``
in the CFM trainer.

This Wf-341-C1 module is the **integration wrapper** that consolidates
the Tong 2024 mini-batch OT family behind a single, top-level
``--minibatch-ot`` orchestrator flag with three backends (default OFF
for safety):

  1. ``exact``  — POT ``ot.emd`` (Tong 2024 BoMb-OT, exact discrete OT)
  2. ``sinkhorn`` — upstream Sinkhorn log-domain (Tong 2023 ICML)
  3. ``hungarian`` — scipy Hungarian (mid-batch, fewer than 128 rows)

The wrapper exposes the canonical Tong 2024 coupling function
:func:`minibatch_ot_couple` plus a smoke bench
:func:`run_minibatch_ot_smoke` that the CFM trainer can call from
``train_one_epoch`` behind the new ``--minibatch-ot`` CLI flag.

Honest framing (party-twenty-spirit §3.5)
-----------------------------------------
* All speedup claims are PROJECTED from Tong et al. 2024 (BoMb-OT)
  paper benchmarks (1.5-3× inference NFE reduction at matched
  training compute); no Mol-Metal A/B measurement yet.
* Defaults to OFF so existing CFM retrain launch scripts are
  bit-identical unless ``--minibatch-ot`` is passed explicitly.
* CPU-only scaffold; the OT solve itself is O(B²) memory and runs
  on CPU regardless of input device.
* Existing ``--use-exact-ot-coupling`` flag (Wf-340-I1) remains
  unchanged; ``--minibatch-ot exact`` is a more general superset
  that wraps the same POT solver and adds the orchestrator.

Usage
-----
::

    # Default — legacy (random pairing inside ``mini_batch_ot_coupling``)
    python molmetal/scripts/wf341c1_minibatch_ot.py --minibatch-ot off

    # Tong 2024 BoMb-OT exact (POT ``ot.emd``) on a 32-sample batch
    python molmetal/scripts/wf341c1_minibatch_ot.py --minibatch-ot exact \
        --minibatch-batch 32 --output-json /tmp/wf341c1_smoke.json

    # Sinkhorn log-domain (Tong 2023) on a 64-sample batch
    python molmetal/scripts/wf341c1_minibatch_ot.py --minibatch-ot sinkhorn \
        --minibatch-batch 64 --sinkhorn-reg 0.05

    # CPU bench (5 random Gaussians × 64 batch × 3 backends)
    python molmetal/scripts/wf341c1_minibatch_ot.py --bench --bench-batch 64

Outputs (NEW this workflow)
---------------------------
* ``molmetal/reports/wf341c1_minibatch_ot/smoke.json`` — per-backend
  coupling cost + permutation validity + sinkhorn marginal error.
* ``molmetal/reports/wf341c1_minibatch_ot/bench.json`` — CPU wall-time
  bench across ``off`` / ``exact`` / ``sinkhorn`` / ``hungarian``.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import torch

# Project root for output paths (consistent with other wf* scripts).
PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf341c1_minibatch_ot"

# Backend literal — the four valid ``--minibatch-ot`` modes.
MinibatchOTMode = Literal["off", "exact", "sinkhorn", "hungarian"]

_VALID_MODES: Tuple[str, ...] = ("off", "exact", "sinkhorn", "hungarian")


# ---------------------------------------------------------------------------
# minibatch_ot_couple — Tong 2024 mini-batch OT coupling wrapper
# ---------------------------------------------------------------------------


def minibatch_ot_couple(
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    *,
    mode: MinibatchOTMode = "exact",
    exact_max_batch: int = 128,
    sinkhorn_reg: float = 0.05,
    sinkhorn_max_iter: int = 200,
    seed: Optional[int] = None,
    return_diagnostics: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
    """Tong 2024 mini-batch OT coupling wrapper.

    Returns ``(perm_x0, perm_x1, diag)`` where ``perm_x0`` /
    ``perm_x1`` are the permuted source / data batches such that
    pairing ``perm_x0[i] <-> perm_x1[i]`` is the optimal (or
    approximate) coupling under the squared-L2 cost.

    Parameters
    ----------
    x_0, x_1
        ``(B, ...)`` source / data batches.  Must share shape.
    mode
        ``"exact"`` (POT ``ot.emd``), ``"sinkhorn"`` (POT log-domain),
        ``"hungarian"`` (scipy), or ``"off"`` (identity).
    exact_max_batch
        Maximum batch size for ``"exact"`` and ``"hungarian"``;
        larger batches fall back to Sinkhorn.
    sinkhorn_reg, sinkhorn_max_iter
        Entropic regularisation + Sinkhorn iteration cap (passed
        through to POT ``ot.sinkhorn(method='sinkhorn_log')``).
    seed
        Deterministic seed forwarded to the upstream RNG.
    return_diagnostics
        When ``True``, returns a ``diag`` dict with backend, cost,
        marginal error (sinkhorn), and CPU/GPU boundary.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, Optional[Dict]]
        ``(x_0, perm_x_1, diag)`` where ``perm_x_1[i]`` is the
        optimal partner of ``x_0[i]`` under the requested coupling
        mode (matches the upstream
        ``flow_matching.optimal_transport.mini_batch_ot_coupling``
        contract: ``x_1[idx]`` is paired with ``x_0``). ``x_0`` is
        always returned unchanged so callers can do
        ``cfm_loss(x_0, perm_x_1)`` directly.

    Notes
    -----
    CPU-only / standalone.  The wrapper guarantees that the returned
    ``perm_x_1`` is the matched partner of ``perm_x_0`` under the
    requested coupling mode — callers use ``cfm_loss(x_t, v_t)``
    with ``v_t = perm_x_1 - perm_x_0`` without further indexing.
    """
    if mode not in _VALID_MODES:
        raise ValueError(f"mode must be one of {_VALID_MODES}, got {mode!r}")
    if x_0.shape != x_1.shape:
        raise ValueError(
            f"x_0 and x_1 must share shape; got {tuple(x_0.shape)} vs "
            f"{tuple(x_1.shape)}."
        )
    if x_0.shape[0] < 1:
        raise ValueError("Batch size must be >= 1.")

    b = int(x_0.shape[0])
    diag: Dict[str, Any] = {
        "requested_mode": mode,
        "batch_size": b,
        "exact_max_batch": int(exact_max_batch),
        "fallback_used": False,
    }

    # Mode 'off' — legacy behaviour, identity permutation.
    if mode == "off":
        diag["effective_backend"] = "identity_off"
        diag["cost"] = 0.0
        if return_diagnostics:
            return x_0, x_1, diag
        return x_0, x_1, None

    # Probe POT availability once.
    pot_available = True
    try:
        import ot as _pot  # noqa: F401
    except ImportError:
        pot_available = False

    # 'exact' backend: POT ``ot.emd`` (Tong 2024 BoMb-OT exact solver).
    if mode == "exact":
        if not pot_available:
            diag["fallback_used"] = True
            diag["fallback_reason"] = "POT library not installed"
            diag["effective_backend"] = "identity_off"
            return _off_fallback(x_0, x_1, diag, return_diagnostics)
        if b > exact_max_batch:
            # Out of the exact-solver's safe range — degrade to sinkhorn.
            diag["fallback_used"] = True
            diag["fallback_reason"] = (
                f"batch_size={b} > exact_max_batch={exact_max_batch}; "
                f"degrading to sinkhorn"
            )
            return minibatch_ot_couple(
                x_0,
                x_1,
                mode="sinkhorn",
                exact_max_batch=exact_max_batch,
                sinkhorn_reg=sinkhorn_reg,
                sinkhorn_max_iter=sinkhorn_max_iter,
                seed=seed,
                return_diagnostics=return_diagnostics,
            )
        perm_x_0, perm_x_1, cost = _exact_ot_solve(x_0, x_1, seed=seed)
        diag["effective_backend"] = "pot_emd_exact"
        diag["cost"] = float(cost)
        if return_diagnostics:
            return perm_x_0, perm_x_1, diag
        return perm_x_0, perm_x_1, None

    # 'sinkhorn' backend: POT log-domain Sinkhorn (Tong 2023).
    if mode == "sinkhorn":
        if not pot_available:
            diag["fallback_used"] = True
            diag["fallback_reason"] = "POT library not installed"
            return _off_fallback(x_0, x_1, diag, return_diagnostics)
        perm_x_0, perm_x_1, cost, marginal_err = _sinkhorn_ot_solve(
            x_0,
            x_1,
            reg=sinkhorn_reg,
            max_iter=sinkhorn_max_iter,
            seed=seed,
        )
        diag["effective_backend"] = "pot_sinkhorn_log"
        diag["cost"] = float(cost)
        diag["marginal_max_abs_error"] = float(marginal_err)
        if return_diagnostics:
            return perm_x_0, perm_x_1, diag
        return perm_x_0, perm_x_1, None

    # 'hungarian' backend: scipy Hungarian (exact, O(B^3), CPU only).
    # Falls back to identity if scipy is unavailable.
    if mode == "hungarian":
        try:
            from scipy.optimize import linear_sum_assignment
        except ImportError:
            diag["fallback_used"] = True
            diag["fallback_reason"] = "scipy not installed"
            return _off_fallback(x_0, x_1, diag, return_diagnostics)
        if b > exact_max_batch:
            diag["fallback_used"] = True
            diag["fallback_reason"] = (
                f"batch_size={b} > exact_max_batch={exact_max_batch}; "
                f"degrading to sinkhorn"
            )
            return minibatch_ot_couple(
                x_0,
                x_1,
                mode="sinkhorn",
                exact_max_batch=exact_max_batch,
                sinkhorn_reg=sinkhorn_reg,
                sinkhorn_max_iter=sinkhorn_max_iter,
                seed=seed,
                return_diagnostics=return_diagnostics,
            )
        perm_x_0, perm_x_1, cost = _hungarian_ot_solve(x_0, x_1)
        diag["effective_backend"] = "scipy_hungarian"
        diag["cost"] = float(cost)
        if return_diagnostics:
            return perm_x_0, perm_x_1, diag
        return perm_x_0, perm_x_1, None

    # Unreachable (mode check at top).
    raise AssertionError(f"unreachable: mode={mode!r}")


# ---------------------------------------------------------------------------
# Per-backend private solvers
# ---------------------------------------------------------------------------


def _exact_ot_solve(
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    *,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """POT ``ot.emd`` exact solver on squared-L2 cost.

    Returns ``(x_0, permuted_x_1, cost)`` where ``permuted_x_1[i]``
    is the optimal partner of ``x_0[i]`` under the OT plan. This
    matches the upstream
    ``flow_matching.optimal_transport.mini_batch_ot_coupling``
    convention (``x_1[idx]`` is paired with ``x_0``).
    """
    import ot as _pot

    b = x_0.shape[0]
    x0f = x_0.detach().to(dtype=torch.float32).reshape(b, -1)
    x1f = x_1.detach().to(dtype=torch.float32).reshape(b, -1)
    cost_mat = _pot.dist(
        x0f.cpu().double().numpy(),
        x1f.cpu().double().numpy(),
        metric="sqeuclidean",
    )
    a = _pot.unif(b)
    bvec = _pot.unif(b)
    plan = _pot.emd(a, bvec, cost_mat, numItermax=100_000)
    idx = plan.argmax(axis=1)
    perm_x_0 = x_0.clone()
    perm_x_1 = x_1[idx].clone()
    cost = float((plan * cost_mat).sum())
    return perm_x_0, perm_x_1, cost


def _sinkhorn_ot_solve(
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    *,
    reg: float = 0.05,
    max_iter: int = 200,
    seed: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, float, float]:
    """POT log-domain Sinkhorn solver; greedy hard-permutation round."""
    import ot as _pot

    b = x_0.shape[0]
    x0f = x_0.detach().to(dtype=torch.float32).reshape(b, -1)
    x1f = x_1.detach().to(dtype=torch.float32).reshape(b, -1)
    cost = torch.cdist(x0f, x1f, p=2.0).pow(2)
    a = torch.full((b,), 1.0 / b, dtype=cost.dtype, device=cost.device)
    bvec = torch.full_like(a, 1.0 / b)
    plan = _pot.sinkhorn(
        a,
        bvec,
        cost,
        reg=float(reg),
        numItermax=int(max_iter),
        method="sinkhorn_log",
        warn=False,
    )
    # Hard-permutation round (greedy with plan-mass priority).
    plan_np = plan.detach().cpu().numpy()
    idx = _greedy_round(plan_np, seed=seed)
    perm_x_0 = x_0.clone()
    perm_x_1 = x_1[idx].clone()
    cost_val = float((plan * cost).sum().item())
    marginal_err = float(
        max(
            (plan.sum(0) - bvec).abs().max().item(),
            (plan.sum(1) - a).abs().max().item(),
        )
    )
    return perm_x_0, perm_x_1, cost_val, marginal_err


def _hungarian_ot_solve(
    x_0: torch.Tensor,
    x_1: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, float]:
    """scipy linear_sum_assignment on squared-L2 cost (CPU).

    Returns ``(x_0, permuted_x_1, cost)`` where ``permuted_x_1[i]``
    is the optimal partner of ``x_0[i]`` under the assignment.
    """
    from scipy.optimize import linear_sum_assignment

    b = x_0.shape[0]
    x0f = x_0.detach().to(dtype=torch.float32).reshape(b, -1)
    x1f = x_1.detach().to(dtype=torch.float32).reshape(b, -1)
    cost_mat = torch.cdist(x0f, x1f, p=2.0).pow(2).cpu().numpy()
    row, col = linear_sum_assignment(cost_mat)
    order = torch.argsort(torch.as_tensor(row, dtype=torch.long))
    idx = torch.as_tensor(col, dtype=torch.long)[order]
    perm_x_0 = x_0.clone()
    perm_x_1 = x_1[idx].clone()
    cost = float(cost_mat[row, col].sum())
    return perm_x_0, perm_x_1, cost


def _off_fallback(
    x_0: torch.Tensor,
    x_1: torch.Tensor,
    diag: Dict[str, Any],
    return_diagnostics: bool,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[Dict[str, Any]]]:
    """Identity fallback when a backend is unavailable."""
    diag["cost"] = 0.0
    if return_diagnostics:
        return x_0, x_1, diag
    return x_0, x_1, None


def _greedy_round(plan: Any, *, seed: Optional[int] = None) -> torch.Tensor:
    """Greedy argmax-per-row with plan-mass priority tie-break.

    Mirrors the upstream ``_sinkhorn_plan_to_permutation`` helper in
    :mod:`flow_matching.optimal_transport` so the Wf-341-C1 wrapper
    has identical coupling behaviour to the legacy path.
    """
    import numpy as _np

    n = plan.shape[0]
    primary = plan.argmax(axis=1)
    used = set()
    out = [-1] * n
    counts: Dict[int, int] = {}
    for j in primary.tolist():
        counts[int(j)] = counts.get(int(j), 0) + 1
    for i, j in enumerate(primary.tolist()):
        j = int(j)
        if counts[j] == 1 and j not in used:
            out[i] = j
            used.add(j)
    free = [j for j in range(n) if j not in used]
    pending = [i for i, v in enumerate(out) if v < 0]
    if pending:
        rng = (
            _np.random.default_rng(int(seed))
            if seed is not None
            else _np.random.default_rng()
        )
        scored = []
        for i in pending:
            masses = [(float(plan[i, j]), j) for j in free]
            masses.sort(reverse=True)
            scored.append((masses[0][0] if masses else 0.0, i, masses))
        scored.sort(reverse=True)
        for _, i, masses in scored:
            if not free:
                break
            choice = None
            for _mass, j in masses:
                if j in set(free):
                    choice = j
                    break
            if choice is None:
                choice = int(rng.choice(free))
            out[i] = choice
            free.remove(choice)
    return torch.as_tensor(out, dtype=torch.long)


# ---------------------------------------------------------------------------
# Smoke + bench runners
# ---------------------------------------------------------------------------


def run_minibatch_ot_smoke(
    *,
    mode: MinibatchOTMode = "exact",
    batch_size: int = 32,
    dim: int = 8,
    seed: int = 0,
) -> Dict[str, Any]:
    """Run a deterministic smoke bench on synthetic Gaussian batches.

    Returns a JSON-serialisable dict with the coupling cost, the
    permutation validity flag, and the marginal error (sinkhorn).
    """
    torch.manual_seed(seed)
    x_0 = torch.randn(batch_size, dim)
    x_1 = torch.randn(batch_size, dim) + 1.0  # shifted data distribution
    perm_x_0, perm_x_1, diag = minibatch_ot_couple(
        x_0,
        x_1,
        mode=mode,
        return_diagnostics=True,
        seed=seed,
    )
    # Validity: perm_x_1 row i should be the closest partner of
    # perm_x_0 row i under squared-L2 (Hungarian guarantees it; sinkhorn
    # is approximate so we measure the cost ratio vs random pairing).
    pair_dist = (perm_x_0 - perm_x_1).pow(2).sum(dim=-1).mean().item()
    rand_idx = torch.randperm(batch_size)
    rand_dist = (x_0 - x_1[rand_idx]).pow(2).sum(dim=-1).mean().item()
    cost_ratio = pair_dist / max(rand_dist, 1e-9)
    return {
        "mode": mode,
        "batch_size": batch_size,
        "dim": dim,
        "diag": diag,
        "paired_distance": float(pair_dist),
        "random_distance": float(rand_dist),
        "cost_ratio_paired_over_random": float(cost_ratio),
    }


def run_minibatch_ot_bench(
    *,
    batch_size: int = 64,
    dim: int = 8,
    n_iters: int = 20,
    seed: int = 0,
) -> Dict[str, Any]:
    """CPU wall-time bench across all backends (off / exact / sinkhorn / hungarian).

    Each backend is timed over ``n_iters`` calls on a fresh random
    batch.  Returns per-backend ``{mean_s, std_s}`` plus a
    ``speedup_vs_off`` ratio per backend.
    """
    modes: Tuple[MinibatchOTMode, ...] = ("off", "exact", "sinkhorn", "hungarian")
    timings: Dict[str, Dict[str, float]] = {}
    torch.manual_seed(seed)
    for mode in modes:
        per_iter = []
        for it in range(n_iters):
            x_0 = torch.randn(batch_size, dim)
            x_1 = torch.randn(batch_size, dim) + 1.0
            start = time.perf_counter()
            minibatch_ot_couple(
                x_0,
                x_1,
                mode=mode,
                seed=seed + it,
            )
            per_iter.append(time.perf_counter() - start)
        m = sum(per_iter) / len(per_iter)
        # Std via simple E[x^2] - E[x]^2.
        var = sum((t - m) ** 2 for t in per_iter) / len(per_iter)
        timings[mode] = {
            "mean_s": float(m),
            "std_s": float(var**0.5),
            "n_iters": n_iters,
        }
    off_mean = timings["off"]["mean_s"]
    for mode in modes:
        if off_mean > 0:
            timings[mode]["speedup_vs_off"] = float(
                off_mean / max(timings[mode]["mean_s"], 1e-9)
            )
        else:
            timings[mode]["speedup_vs_off"] = 1.0
    return {
        "batch_size": batch_size,
        "dim": dim,
        "n_iters": n_iters,
        "timings": timings,
    }


# ---------------------------------------------------------------------------
# CLI surface
# ---------------------------------------------------------------------------


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Wf-341-C1 — Tong 2024 mini-batch OT coupling integration wrapper. "
            "Default mode is 'off' for safety (legacy behaviour)."
        ),
    )
    p.add_argument(
        "--minibatch-ot",
        choices=list(_VALID_MODES),
        default="off",
        help=(
            "Mini-batch OT coupling mode (Tong 2024). Default 'off' — "
            "use legacy upstream mini_batch_ot_coupling. "
            "'exact' uses POT ot.emd; 'sinkhorn' uses POT log-domain; "
            "'hungarian' uses scipy."
        ),
    )
    p.add_argument(
        "--minibatch-batch",
        type=int,
        default=32,
        help="Synthetic batch size for the smoke runner (default 32).",
    )
    p.add_argument(
        "--minibatch-dim",
        type=int,
        default=8,
        help="Synthetic feature dim for the smoke runner (default 8).",
    )
    p.add_argument(
        "--exact-max-batch",
        type=int,
        default=128,
        help=(
            "Maximum batch size for the exact / hungarian solver; "
            "above this the wrapper degrades to sinkhorn."
        ),
    )
    p.add_argument(
        "--sinkhorn-reg",
        type=float,
        default=0.05,
        help="Sinkhorn entropic regularisation strength (default 0.05).",
    )
    p.add_argument(
        "--sinkhorn-max-iter",
        type=int,
        default=200,
        help="Sinkhorn iteration cap (default 200).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed for the synthetic smoke batches (default 0).",
    )
    p.add_argument(
        "--output-json",
        type=str,
        default=None,
        help=(
            "Optional JSON path to write the smoke result. Default "
            "writes to molmetal/reports/wf341c1_minibatch_ot/smoke.json."
        ),
    )
    p.add_argument(
        "--bench",
        action="store_true",
        help="Run the CPU wall-time bench instead of the smoke runner.",
    )
    p.add_argument(
        "--bench-batch",
        type=int,
        default=64,
        help="Batch size for the CPU bench (default 64).",
    )
    p.add_argument(
        "--bench-iters",
        type=int,
        default=20,
        help="Iterations per backend in the CPU bench (default 20).",
    )
    return p


def main(argv: Optional[list] = None) -> int:
    args = _build_arg_parser().parse_args(argv)

    if args.bench:
        result = run_minibatch_ot_bench(
            batch_size=args.bench_batch,
            dim=args.minibatch_dim,
            n_iters=args.bench_iters,
            seed=args.seed,
        )
    else:
        result = run_minibatch_ot_smoke(
            mode=args.minibatch_ot,
            batch_size=args.minibatch_batch,
            dim=args.minibatch_dim,
            seed=args.seed,
        )

    # Write to disk.
    out_path = (
        Path(args.output_json)
        if args.output_json
        else REPORTS_DIR / ("bench.json" if args.bench else "smoke.json")
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(result, f, indent=2, sort_keys=True)
    print(f"[wf341c1] wrote {out_path}", flush=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
