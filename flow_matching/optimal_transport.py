"""Optimal-transport plan utilities for conditional flow matching.

In conditional flow matching the choice of pairing between the source
samples ``x0`` and the data samples ``x1`` matters a great deal: an
independent random pairing produces paths that cross each other and
the velocity field has to work harder to disentangle them; an
optimal-transport pairing (e.g. computed by solving a discrete OT
problem in a cheap cost metric) reduces the variance of the training
gradient substantially (see Tong et al., 2024, "Improving and
Generalizing Flow-Based Generative Models with Minibatch Optimal
Transport").

This module deliberately keeps the implementation minimal:

* :func:`compute_ot_plan` returns, given a batch of source points
  ``x0`` and data points ``x1``, the permutation that reorders
  ``x1`` so that each ``x0_i`` is paired with an ``x1_pi[i]``.
* The default mode is ``"random"`` - a fresh random permutation per
  call - which is a useful baseline for ablations and is what most
  non-OT flow-matching papers use.
* A ``"hungarian"`` mode is provided that solves the assignment
  problem exactly using ``scipy.optimize.linear_sum_assignment`` on
  a cheap pairwise cost (``||x0_i - x1_j||^2``).  ``scipy`` is
  already a transitive dependency of the scientific Python stack
  used elsewhere in the project, so this does not add a new
  requirement.

The function returns indices only (no copies of the tensors) so
callers can do ``x1_permuted = x1[idx]`` themselves.
"""

from __future__ import annotations

from typing import Literal
import warnings

import torch


PlanMode = Literal["random", "hungarian"]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_ot_plan(
    x0: torch.Tensor,
    x1: torch.Tensor,
    mode: PlanMode = "random",
    *,
    seed: int | None = None,
) -> torch.Tensor:
    """Compute a pairing between a source batch ``x0`` and a data batch ``x1``.

    Parameters
    ----------
    x0 : ``(B, ...)`` tensor
        Source batch (typically noise).
    x1 : ``(B, ...)`` tensor
        Target batch (data).  Must have the same leading batch size
        and the same trailing shape as ``x0``.
    mode : ``"random"`` | ``"hungarian"``
        Pairing strategy.
        * ``"random"`` returns a uniformly random permutation of the
          batch indices.  This is the no-OT baseline.
        * ``"hungarian"`` solves the exact min-cost assignment with
          cost ``||x0_i - x1_j||^2`` using
          :func:`scipy.optimize.linear_sum_assignment`.
    seed : int, optional
        Seed for the random permutation (only used in ``"random"``
        mode).  When ``None`` the current global RNG is used.

    Returns
    -------
    idx : ``(B,)`` long tensor
        Permutation such that ``x1[idx]`` is paired with ``x0``.  On
        the same device as the inputs.
    """
    if x0.shape != x1.shape:
        raise ValueError(
            f"`x0` and `x1` must share shape; got {tuple(x0.shape)} vs "
            f"{tuple(x1.shape)}."
        )
    if x0.shape[0] < 1:
        raise ValueError("Batch size must be >= 1.")
    if mode not in ("random", "hungarian"):
        raise ValueError(
            f"`mode` must be 'random' or 'hungarian'; got {mode!r}."
        )

    b = x0.shape[0]

    if mode == "random":
        # Build a generator so we don't touch the global RNG unless the
        # caller has not asked for a fixed seed.
        if seed is not None:
            gen = torch.Generator(device=x0.device)
            gen.manual_seed(int(seed))
            idx = torch.randperm(b, generator=gen, device=x0.device)
        else:
            idx = torch.randperm(b, device=x0.device)
        return idx.to(dtype=torch.long)

    # Hungarian: exact min-cost assignment on a flat squared-distance
    # cost matrix.  We flatten trailing dims so 3-D positions, latent
    # vectors and graph-level features all work the same way.
    x0_flat = x0.detach().to(dtype=torch.float32).reshape(b, -1)
    x1_flat = x1.detach().to(dtype=torch.float32).reshape(b, -1)

    # Cost matrix is small (B x B) and lives on CPU; solving it on the
    # GPU is not worth the latency for typical batch sizes (<=1024).
    cost = torch.cdist(x0_flat, x1_flat, p=2.0).pow(2).cpu().numpy()

    # Lazy import: scipy is only needed for the optimal branch and is
    # not a hard dependency of the package.
    from scipy.optimize import linear_sum_assignment

    row, col = linear_sum_assignment(cost)

    # `linear_sum_assignment` returns row/col arrays sorted by `row`;
    # we re-order so the output index lines up element-wise with x0.
    order = torch.argsort(torch.as_tensor(row, dtype=torch.long))
    idx = torch.as_tensor(col, dtype=torch.long)[order]
    return idx.to(device=x0.device)


# ---------------------------------------------------------------------------
# Mini-batch OT coupling
# ---------------------------------------------------------------------------
def mini_batch_ot_coupling(
    x1: torch.Tensor,
    x0: torch.Tensor,
    batch_idx: torch.Tensor,
    *,
    method: Literal["sinkhorn", "hungarian"] = "sinkhorn",
    reg: float = 0.05,
    max_iter: int = 200,
    seed: int | None = None,
    diagnostics: list[dict] | None = None,
) -> torch.Tensor:
    """Run OT coupling independently within each group defined by ``batch_idx``.

    This is the per-pseudo-batch variant of the mini-batch OT plan of
    Tong et al., 2024 ("Improving and Generalizing Flow-Based
    Generative Models with Minibatch Optimal Transport").  Instead of
    solving a single ``B x B`` assignment over the whole batch, we
    group rows by ``batch_idx`` and solve a separate assignment
    problem inside each group.  This is the right knob when each
    "sample" in the batch already belongs to a distinct molecule /
    pocket / system (e.g. SE(3) flow matching on pockets) and OT
    across heterogeneous systems is meaningless.

    Parameters
    ----------
    x1 : ``(B, ...)`` tensor
        Target / data batch.
    x0 : ``(B, ...)`` tensor
        Source / noise batch.  Must share shape with ``x1``.
    batch_idx : ``(B,)`` long tensor
        Group identifier for each row.  Rows with the same value are
        coupled together; rows in a singleton group are returned
        unchanged (identity coupling for that row).
    method : ``"sinkhorn"`` | ``"hungarian"``
        Solver used inside each mini-batch:

        * ``"sinkhorn"`` uses POT's public ``ot.sinkhorn`` with the
          log-domain implementation of entropic OT. Cost and soft plan
          stay on the input device. Missing/failing POT emits a warning
          before falling back to Hungarian (or random pairing without scipy).
        * ``"hungarian"`` uses
          :func:`scipy.optimize.linear_sum_assignment` for an exact
          min-cost assignment on the squared-distance cost matrix.
    reg : float, default ``0.05``
        Entropic regularisation strength passed to POT's Sinkhorn
        solver.  Ignored when ``method="hungarian"``.
    max_iter : int, default ``200``
        Maximum Sinkhorn iterations.  Ignored when
        ``method="hungarian"``.
    seed : int, optional
        Seed forwarded to :func:`compute_ot_plan` only as a
        *tie-breaker* in the random fallback branch.
    diagnostics : list[dict], optional
        Append one record per group describing the effective solver,
        computation device, CPU boundary, and any fallback reason.

    Returns
    -------
    idx : ``(B,)`` long tensor
        Permutation of the data batch such that ``x1[idx]`` is the
        selected partner of ``x0`` within each ``batch_idx``
        group.  On the same device as the inputs.

    Notes
    -----
    * Sinkhorn returns a soft transport plan; we recover a hard
      permutation by taking ``argmax_j plan[i, j]`` with a random
      tie-break so that ``idx`` stays a valid permutation (no two
      rows receive the same data point). The existing greedy rounding
      runs on CPU after copying the soft plan; it is not a Hungarian
      assignment. Its hard permutation is not guaranteed to be the exact
      minimum-cost assignment.
    * POT is treated as an optional dependency: when missing we fall
      back to Hungarian on the same cost matrix, and when both POT
      and scipy are missing we fall back to random coupling within each
      group. Both fallbacks are reported and return a valid permutation.
    * Group order in the returned ``idx`` matches the input row order
      so callers can do ``x1_permuted = x1[idx]`` exactly as with
      :func:`compute_ot_plan`.
    """
    if x0.shape != x1.shape:
        raise ValueError(
            f"`x0` and `x1` must share shape; got {tuple(x0.shape)} vs "
            f"{tuple(x1.shape)}."
        )
    if batch_idx.shape[0] != x0.shape[0]:
        raise ValueError(
            f"`batch_idx` must have the same leading dim as `x0`; got "
            f"{tuple(batch_idx.shape)} vs {tuple(x0.shape)}."
        )
    if method not in ("sinkhorn", "hungarian"):
        raise ValueError(
            f"`method` must be 'sinkhorn' or 'hungarian'; got {method!r}."
        )
    if method == "sinkhorn" and (reg <= 0 or max_iter < 1):
        raise ValueError("Sinkhorn requires reg > 0 and max_iter >= 1")

    b = x0.shape[0]
    device = x0.device
    out = torch.arange(b, dtype=torch.long, device=device)

    # Group indices by batch_idx value.  We use Python-level bucketing
    # because group counts are tiny (typically <64 molecules per batch)
    # and we need a contiguous sub-index per group for Hungarian.
    batch_idx_long = batch_idx.detach().to(device="cpu", dtype=torch.long)
    groups: dict[int, list[int]] = {}
    for row, g in enumerate(batch_idx_long.tolist()):
        groups.setdefault(int(g), []).append(row)

    # Probe POT / scipy availability once.  POT is preferred for the
    # Sinkhorn method (entropic OT, Tong et al. 2023); scipy is the
    # fallback for either method when POT is missing.
    try:
        import ot as _pot  # noqa: F401
        has_pot = True
    except Exception:  # pragma: no cover - POT absent
        has_pot = False
    try:
        from scipy.optimize import linear_sum_assignment  # noqa: F401
        has_scipy = True
    except Exception:  # pragma: no cover - scipy absent
        has_scipy = False

    for _g, rows in groups.items():
        if len(rows) <= 1:
            # Singleton group: identity coupling; nothing to do.
            if diagnostics is not None:
                diagnostics.append({"effective_backend": "singleton_identity",
                                    "group_size": len(rows), "device": str(device),
                                    "cpu_boundary": "group bookkeeping", "fallback_reason": None})
            continue
        sub_idx = torch.as_tensor(rows, dtype=torch.long, device=device)
        x0_sub = x0.index_select(0, sub_idx).detach()
        x1_sub = x1.index_select(0, sub_idx).detach()

        local = _solve_minibatch_ot(
            x0_sub,
            x1_sub,
            method=method,
            reg=reg,
            max_iter=max_iter,
            has_pot=has_pot,
            has_scipy=has_scipy,
            seed=seed,
            device=device,
            diagnostics=diagnostics,
        )

        # Map the local permutation back to absolute row indices.
        out[sub_idx] = sub_idx[local.to(device=device)]

    return out


def _solve_minibatch_ot(
    x0_sub: torch.Tensor,
    x1_sub: torch.Tensor,
    *,
    method: Literal["sinkhorn", "hungarian"],
    reg: float,
    max_iter: int,
    has_pot: bool,
    has_scipy: bool,
    seed: int | None,
    device: torch.device,
    diagnostics: list[dict] | None = None,
) -> torch.Tensor:
    """Solve OT coupling for a single mini-batch group.

    Returns a ``(n,)`` long permutation on ``device``.  Encapsulates
    the POT Sinkhorn → scipy Hungarian → random fallback ladder so
    :func:`mini_batch_ot_coupling` stays readable.
    """
    n = x0_sub.shape[0]
    x0_flat = x0_sub.to(dtype=torch.float32).reshape(n, -1)
    x1_flat = x1_sub.to(dtype=torch.float32).reshape(n, -1)
    cost = torch.cdist(x0_flat, x1_flat, p=2.0).pow(2)
    record = {
        "requested_method": method, "group_size": n,
        "device": str(device), "cost_device": str(cost.device),
        "fallback_reason": None,
    }

    def report(**fields):
        if diagnostics is not None:
            diagnostics.append({**record, **fields})

    if method == "sinkhorn" and has_pot:
        # Sinkhorn path (Tong et al. 2023): entropic OT in the squared
        # Euclidean ground metric.  POT expects (a, b, M) marginals; we
        # use the uniform marginals that the paper recommends.
        try:
            import ot as _pot
            a = torch.full((n,), 1.0 / n, dtype=cost.dtype, device=cost.device)
            b = torch.full_like(a, 1.0 / n)
            # At molecular coordinate scale, exp(-cost / 0.05) underflows
            # float32. Log-domain Sinkhorn solves the same entropic problem
            # without changing the cost or regularization strength.
            plan = _pot.sinkhorn(
                a, b, cost, reg=float(reg), numItermax=int(max_iter),
                method="sinkhorn_log", warn=False,
            )
            if not isinstance(plan, torch.Tensor) or plan.device != cost.device:
                raise RuntimeError("POT did not return a tensor on the input device")
            if not torch.isfinite(plan).all():
                raise FloatingPointError("POT returned a non-finite transport plan")
            # Preserve the established greedy hard-permutation projection.
            # Only this small plan crosses to CPU, not the Sinkhorn solve.
            local = _sinkhorn_plan_to_permutation(plan.detach().cpu().numpy(), seed=seed)
            if diagnostics is not None:
                error = torch.maximum((plan.sum(0) - b).abs().max(),
                                      (plan.sum(1) - a).abs().max()).item()
                report(effective_backend="pot_sinkhorn_log", plan_device=str(plan.device),
                       cpu_boundary="group bookkeeping and greedy hard-permutation rounding",
                       rounding="greedy_plan_permutation", marginal_max_abs_error=error)
            return local.to(device=device)
        except Exception as exc:
            # POT raised (e.g. unstable Sinkhorn on degenerate cost);
            # drop through to Hungarian if available.
            record["fallback_reason"] = f"{type(exc).__name__}: {exc}"
    elif method == "sinkhorn":
        record["fallback_reason"] = "POT unavailable"

    if record["fallback_reason"] is not None:
        warnings.warn(
            f"Sinkhorn unavailable ({record['fallback_reason']}); using "
            f"{'SciPy Hungarian on CPU' if has_scipy else 'random pairing'}.",
            RuntimeWarning, stacklevel=2,
        )

    # Hungarian fallback (exact discrete OT on the same cost matrix).
    if has_scipy:
        from scipy.optimize import linear_sum_assignment
        row, col = linear_sum_assignment(cost.detach().cpu().numpy())
        order = torch.argsort(torch.as_tensor(row, dtype=torch.long))
        local = torch.as_tensor(col, dtype=torch.long)[order]
        report(effective_backend="scipy_hungarian", plan_device="cpu",
               cpu_boundary="group bookkeeping and SciPy hard assignment", rounding=None)
        return local.to(device=device)

    # Final fallback: random permutation (seeded when requested).
    # Keeps the API contract alive in
    # environments without scipy/POT.
    if seed is not None:
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        perm = torch.randperm(n, generator=gen)
    else:
        perm = torch.randperm(n)
    if record["fallback_reason"] is None:
        record["fallback_reason"] = "SciPy unavailable"
        warnings.warn("SciPy unavailable; using random OT pairing.", RuntimeWarning, stacklevel=2)
    report(effective_backend="random_pairing", plan_device="cpu",
           cpu_boundary="group bookkeeping and random permutation", rounding=None)
    return perm.to(device=device, dtype=torch.long)


def _sinkhorn_plan_to_permutation(
    plan: "numpy.ndarray",
    *,
    seed: int | None,
) -> torch.Tensor:
    """Convert a soft Sinkhorn transport plan into a hard permutation.

    For each row ``i`` we pick ``argmax_j plan[i, j]``.  When two
    rows share the same argmax (degenerate plan), we keep the first
    occurrence and assign the others to whichever columns are still
    free, breaking ties with a deterministic RNG when ``seed`` is
    given.  This produces a valid permutation even when Sinkhorn has
    not fully converged.
    """
    import numpy as _np
    n = plan.shape[0]
    # Primary assignment: argmax per row.
    primary = plan.argmax(axis=1)
    used = set()
    out = [-1] * n
    # Pass 1: place rows whose argmax is unique.
    counts: dict[int, int] = {}
    for j in primary.tolist():
        counts[int(j)] = counts.get(int(j), 0) + 1
    for i, j in enumerate(primary.tolist()):
        j = int(j)
        if counts[j] == 1 and j not in used:
            out[i] = j
            used.add(j)
    # Pass 2: any row still unassigned gets a free column (tie-break
    # by descending plan mass, then RNG if ``seed`` is provided).
    free = [j for j in range(n) if j not in used]
    pending = [i for i, v in enumerate(out) if v < 0]
    if pending:
        if seed is not None:
            rng = _np.random.default_rng(int(seed))
        else:
            rng = _np.random.default_rng()
        # Sort pending rows by their best free-column plan mass so the
        # most-confident rows get first pick.
        scored = []
        for i in pending:
            masses = [(float(plan[i, j]), j) for j in free]
            masses.sort(reverse=True)
            scored.append((masses[0][0] if masses else 0.0, i, masses))
        scored.sort(reverse=True)
        for _, i, masses in scored:
            if not free:
                break
            # Pick the highest-mass free column for this row.
            choice = None
            for _mass, j in masses:
                if j in set(free):
                    choice = j
                    break
            if choice is None:
                # Plan mass all on used columns (pathological); take
                # a random free column.
                choice = int(rng.choice(free))
            out[i] = choice
            free.remove(choice)
    return torch.as_tensor(out, dtype=torch.long)


__all__ = ["compute_ot_plan", "mini_batch_ot_coupling", "PlanMode"]
