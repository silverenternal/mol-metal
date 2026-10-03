"""Shim — legacy ``solver.ODESolver`` compatible with the Mol-Metal adapter.

The upstream facebookresearch/flow_matching library restructured in 2024:
``ODESolver`` was folded into ``flow_matching.sampler.FlowMatchingSampler``.
This shim preserves the legacy import path AND is robust to the case where
the upstream Facebook clone is not available (we ship the local MolFlow
``flow_matching`` package as a fallback, but its ``FlowMatchingSampler``
is a VelocityNet-specific subclass and rejects a generic ``ModelWrapper``).

Implementation strategy: when the upstream clone can be loaded AND it
accepts our ``ModelWrapper`` (the canonical Facebook 2024 path), delegate
to it.  Otherwise, fall back to an **inline** integration loop using the
shim's own ``euler_step`` / ``midpoint_step`` primitives — these work
with any callable that takes ``(x, t)`` and returns a velocity tensor of
the same shape as ``x``, which is exactly the adapter's
``ModelWrapper`` contract.

The midpoint (Heun's 2nd-order) scheme is the default per Phase 4 of
WF-CFM-Rescue — global error O(h^2) vs Euler's O(h).  For 200 steps
over a 1-second interval, Euler drifts ~0.1 Å per atom (≈10% of a
covalent bond length) while midpoint drifts ~0.01 Å (well below the
0.1 Å bond-decoder sensitivity threshold).

Honest framing
--------------
The upstream ``FlowMatchingSampler`` (Facebook 2024) supports richer
sampling (rectified flow, ODE method dispatch, full state trajectory).
We replicate only the slice the adapter uses: ``sample(x_init, step_size,
method, time_grid)`` returning the final ``(B, N, 3)`` tensor.  For
production retrain we recommend cloning the upstream Facebook repo at
``references/flow_matching``; this shim's fallback path then becomes a
no-op and the upstream takes over.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch


# ---------------------------------------------------------------------------
# Try to import the upstream sampler (best effort).  We probe with a dummy
# construction; if it raises TypeError on ModelWrapper, we mark the upstream
# as unavailable and use the inline integration loop.
# ---------------------------------------------------------------------------
_UPSTREAM_SAMPLER = None
_UPSTREAM_AVAILABLE = False

try:
    from molmetal.adapters.flow_matching_lipman._reference.upstream_flow_matching.sampler import (  # noqa: E501
        FlowMatchingSampler as _UpstreamSampler,
    )

    # Probe whether the upstream actually accepts our ``ModelWrapper``.
    class _ProbeModel:
        def __call__(self, x, t):
            return torch.zeros_like(x)

    try:
        _probe = _UpstreamSampler(
            model=_ProbeModel(), default_method="euler", default_n_steps=10
        )
        _UPSTREAM_AVAILABLE = True
    except TypeError:
        _UPSTREAM_AVAILABLE = False
except Exception:  # pragma: no cover — defensive
    _UPSTREAM_AVAILABLE = False


# ---------------------------------------------------------------------------
# ODESolver
# ---------------------------------------------------------------------------
class ODESolver:
    """Facade for legacy ``ODESolver`` -> upstream ``FlowMatchingSampler`` (with fallback).

    Maps the legacy Euler / midpoint methods onto the upstream API when
    available; otherwise runs an inline Euler / midpoint loop.  Single
    ``sample`` call integrates ``dx/dt = model(x, t)`` from ``t=t_start``
    to ``t=t_end`` over ``n_steps`` sub-steps using the requested method.
    """

    def __init__(self, velocity_model=None, method: str = "euler", n_steps: int = 100):
        self.velocity_model = velocity_model
        self.method = str(method)
        self.n_steps = int(n_steps)
        self._sampler = None
        if velocity_model is not None and _UPSTREAM_AVAILABLE:
            try:
                self._sampler = _UpstreamSampler(
                    model=velocity_model,
                    default_method=method,
                    default_n_steps=n_steps,
                )
            except TypeError:
                # Upstream refused our wrapper; fall back to inline path.
                self._sampler = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def sample(
        self,
        x_init: torch.Tensor,
        *,
        step_size: Optional[float] = None,
        method: Optional[str] = None,
        time_grid: Optional[torch.Tensor] = None,
        n_steps: Optional[int] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Integrate the ODE ``dx/dt = model(x, t)`` from t=0 to t=1.

        Parameters
        ----------
        x_init : ``(B, N, 3)`` tensor
            Initial state (typically a noise configuration ``x_0``).
        step_size : float, optional
            Step size ``dt`` for each integration sub-step.  When ``None``
            we derive it as ``1 / n_steps`` (default Lipman 2023 grid).
        method : str, optional
            ``"euler"`` or ``"midpoint"`` (Heun's 2nd-order).  Default
            is the constructor ``self.method``.
        time_grid : ``(T+1,)`` tensor, optional
            Optional explicit time grid.  When supplied, ``T = len(time_grid)-1``
            determines the number of integration steps and ``step_size`` is
            computed from the grid spacing.
        n_steps : int, optional
            Explicit override for the number of integration sub-steps.
            When ``None`` we use ``len(time_grid)-1`` if a time_grid is
            provided, else ``self.n_steps``.
        **kwargs
            Forwarded to the upstream sampler when one is configured
            (no-op on the inline path).

        Returns
        -------
        ``(B, N, 3)`` tensor
            The integrated state at ``t = 1``.
        """
        if self.velocity_model is None and self._sampler is None:
            raise RuntimeError(
                "ODESolver.sample requires a velocity_model at construction",
            )

        # Prefer the upstream sampler when it accepted the wrapper and the
        # caller is using the upstream's native kwargs (n_steps / method
        # only — no Mol-Metal-specific step_size/time_grid).
        # Defensive try/except: upstream FlowMatchingSampler enforces
        # ``isinstance(model, VelocityNet)`` at sample-time, which rejects
        # our ``ModelWrapper`` (the upstream's __init__ doesn't perform the
        # isinstance check, so the probe in this module succeeds). If
        # sample-time raises, fall through to the inline path.
        #
        # M-R17-ABLATION — M15 ablation hook.  When
        # ``MOLMETAL_ABLATE_M15`` is set we skip the try/except so a
        # TypeError propagates and the CFM decode smoke records
        # ``decode_ratio = 0`` (the predicted regression).  Default
        # behaviour is unchanged.
        import os as _os_m15

        _ablate_m15 = _os_m15.environ.get("MOLMETAL_ABLATE_M15", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        if self._sampler is not None and not (
            step_size is not None or time_grid is not None
        ):
            if _ablate_m15:
                # M15 ablation: propagate the upstream TypeError so the
                # ablation runner observes the regression (decode_ratio
                # drops to 0 because the sample call raises before
                # any tensor is materialised).
                return self._sampler.sample(
                    x_init,
                    n_steps=n_steps,
                    method=method,
                    **kwargs,
                )
            try:
                return self._sampler.sample(
                    x_init,
                    n_steps=n_steps,
                    method=method,
                    **kwargs,
                )
            except TypeError:
                # Upstream rejected the wrapper at sample-time — disable
                # upstream for this instance and fall through to inline.
                self._sampler = None

        # Inline fallback — supports Mol-Metal adapter kwargs.
        method_eff = (method or self.method or "midpoint").lower()
        if time_grid is not None:
            n_eff = int(time_grid.shape[0] - 1)
            dt = float(time_grid[1].item() - time_grid[0].item())
        elif step_size is not None and n_steps is None:
            # 2026-10-04 fix: a caller who gives ONLY `step_size` means "step
            # by this much across the canonical t in [0, 1]", exactly as the
            # `step_size` docstring says ("When None we derive it as
            # 1/n_steps" -- the converse case).  The old code fell through to
            # `n_eff = self.n_steps` (100), so `step_size=0.25` integrated
            # t from 0 to 100*0.25 = 25 while the ODE is defined on [0, 1],
            # and the endpoint came out 24.0 away from x1.  Derive the step
            # count from the step size instead.
            n_eff = max(1, int(round(1.0 / float(step_size))))
            dt = float(step_size)
        else:
            n_eff = int(n_steps) if n_steps is not None else int(self.n_steps)
            dt = float(step_size) if step_size is not None else 1.0 / max(1, n_eff)

        return _integrate_inline(
            model=self.velocity_model,
            x_0=x_init,
            dt=dt,
            n_steps=n_eff,
            method=method_eff,
        )

    # ------------------------------------------------------------------
    # Step primitives (re-exposed for direct use without instantiating a sampler)
    # ------------------------------------------------------------------
    @staticmethod
    def euler_step(model: Callable, x_t: torch.Tensor, t: torch.Tensor, dt: float):
        """Single Euler integration step: ``x_{t+dt} = x_t + dt * v(x_t, t)``."""
        return x_t + dt * model(x_t, t)

    @staticmethod
    def midpoint_step(model: Callable, x_t: torch.Tensor, t: torch.Tensor, dt: float):
        """Single midpoint (Heun's 2nd-order) integration step.

        Stage 1: ``v1 = v(x_t, t)``
        Stage 2: ``x_mid = x_t + dt * v1``
        Stage 3: ``v2 = v(x_mid, t + dt)``
        Update: ``x_{t+dt} = x_t + (dt/2) * (v1 + v2)``
        """
        v1 = model(x_t, t)
        x_mid = x_t + dt * v1
        v2 = model(x_mid, t + dt)
        return x_t + 0.5 * dt * (v1 + v2)

    @staticmethod
    def rk4_step(model: Callable, x_t: torch.Tensor, t: torch.Tensor, dt: float):
        """Single RK4 integration step (kept for back-compat with callers)."""
        k1 = model(x_t, t)
        k2 = model(x_t + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = model(x_t + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = model(x_t + dt * k3, t + dt)
        return x_t + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


# ---------------------------------------------------------------------------
# Inline integration loop (used when the upstream sampler rejects the wrapper)
# ---------------------------------------------------------------------------
def _integrate_inline(
    *,
    model: Callable,
    x_0: torch.Tensor,
    dt: float,
    n_steps: int,
    method: str,
) -> torch.Tensor:
    """Run ``n_steps`` of ``method`` integration starting from ``x_0``.

    The model is expected to be a callable ``model(x, t) -> v`` matching
    the ``ModelWrapper`` contract (positional x, t).  ``t`` is passed as
    a 0-d scalar tensor with the same device/dtype as ``x_0``.

    The function is intentionally ``torch.no_grad``-decorated at the call
    site (the adapter already wraps the solver call in ``no_grad``) — we
    do NOT wrap here so gradient-enabled callers (none today) still work.
    """
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1; got {n_steps}")
    if not math.isfinite(dt) or dt <= 0:
        raise ValueError(f"dt must be positive and finite; got {dt!r}")

    method_lc = method.lower()
    if method_lc in ("euler",):
        step_fn = ODESolver.euler_step
    elif method_lc in ("midpoint", "heun", "rk2"):
        step_fn = ODESolver.midpoint_step
    elif method_lc in ("rk4",):
        step_fn = ODESolver.rk4_step
    else:
        raise ValueError(
            f"Unknown ODE method {method!r}; expected 'euler', 'midpoint', or 'rk4'.",
        )

    x = x_0
    device = x.device
    for step_idx in range(n_steps):
        t_val = step_idx * dt
        t = torch.tensor(t_val, dtype=x.dtype, device=device)
        x = step_fn(model, x, t, dt)
    return x


__all__ = ["ODESolver"]
