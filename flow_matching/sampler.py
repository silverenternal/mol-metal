"""Flow-matching sampler for MolFlow-Triton.

This module wraps the custom ODE integrator kernels in
:mod:`triton_kernels.ode_solver` and drives a
:class:`models.velocity_net.VelocityNet` to produce samples from the
learned vector field.

The sampler exposes a single :class:`FlowMatchingSampler` object with
a :meth:`sample` method that integrates the ODE

.. math::

    \\frac{dx}{dt} = v_\\theta(x, t, \\text{cond})

from ``t = 0`` (a noise configuration ``x0``) to ``t = 1`` (a data
configuration ``x1``) using either an explicit Euler or a classical
RK4 step.  The Euler path calls :func:`triton_kernels.euler_step`
once per sub-step; the RK4 path evaluates the velocity field four
times per sub-step and combines the four stages with
:func:`triton_kernels.rk4_step`.

The sampler itself is a pure inference helper: it does not own an
optimizer and never touches the model's parameters.  It is intended
to be used in evaluation / generation scripts, not inside the
training loop.

The integration hyperparameters live in
:class:`flow_matching.sampler_config.SamplerConfig`, which is
re-exported from this module for backwards compatibility.
"""

from __future__ import annotations

import torch
from torch import nn

from flow_matching.sampler_config import SamplerConfig
from models.velocity_net import VelocityNet
from triton_kernels import euler_step, rk4_step


# ---------------------------------------------------------------------------
# Sampler
# ---------------------------------------------------------------------------
class FlowMatchingSampler(nn.Module):
    """Integrate the learned flow-matching ODE using Triton step kernels.

    Parameters
    ----------
    model : :class:`VelocityNet`
        The trained velocity network.  Held by reference; not copied.
    encoder : optional callable
        ``encoder(atomic_numbers, positions, edge_index)`` -> ``h_node``.
        When ``None`` the caller must pass ``h_node`` to :meth:`sample`
        directly.
    default_method : ``"euler"`` | ``"rk4"``, default ``"euler"``
        Default integration scheme used when ``method`` is not provided
        to :meth:`sample`.
    default_n_steps : int, default 100
        Default number of integration steps used when ``n_steps`` is
        not provided to :meth:`sample`.

    Notes
    -----
    The sampler is implemented as an ``nn.Module`` so that PyTorch's
    ``.to(device)`` / ``.eval()`` / ``state_dict`` machinery works as
    expected, but it does not introduce any learnable parameters of
    its own.
    """

    def __init__(
        self,
        model: VelocityNet,
        encoder: nn.Module | None = None,
        default_method: str = "euler",
        default_n_steps: int = 100,
    ) -> None:
        super().__init__()
        if not isinstance(model, VelocityNet):
            raise TypeError(
                f"`model` must be a VelocityNet; got {type(model).__name__}."
            )
        if default_method not in ("euler", "rk4"):
            raise ValueError(
                f"`default_method` must be 'euler' or 'rk4'; got "
                f"{default_method!r}."
            )
        if default_n_steps < 1:
            raise ValueError("`default_n_steps` must be >= 1.")

        self.model = model
        self.encoder = encoder
        self.default_method = default_method
        self.default_n_steps = default_n_steps

    # ------------------------------------------------------------------
    # Helper: predict velocity at a given (x, t).
    # ------------------------------------------------------------------
    def _velocity(
        self,
        x: torch.Tensor,
        t_scalar: torch.Tensor,
        cond: torch.Tensor | None,
        h_node: torch.Tensor | None,
        atomic_numbers: torch.Tensor | None,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Evaluate ``v_theta(x, t, cond)``."""
        if h_node is None:
            if self.encoder is None or atomic_numbers is None:
                raise ValueError(
                    "Either `h_node` or `(encoder, atomic_numbers)` must be "
                    "supplied so that the velocity net can be evaluated."
                )
            h_node_eff, _ = self.encoder(
                atomic_numbers, x, edge_index,
                edge_mask=edge_mask, node_mask=node_mask,
            )
        else:
            h_node_eff = h_node

        b = x.shape[0]
        t_batch = t_scalar.expand(b).to(dtype=x.dtype if x.is_floating_point() else torch.float32)
        v = self.model(
            h_node_eff, x, t_batch, edge_index, cond=cond,
            edge_mask=edge_mask, node_mask=node_mask,
        )
        if v.shape != x.shape:
            raise ValueError(
                "VelocityNet output shape "
                f"{tuple(v.shape)} does not match state shape {tuple(x.shape)}."
            )
        return v

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample(
        self,
        x0: torch.Tensor,
        cond: torch.Tensor | None = None,
        n_steps: int | None = None,
        method: str | None = None,
        *,
        h_node: torch.Tensor | None = None,
        atomic_numbers: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_mask: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
        t_start: float | None = None,
        t_end: float | None = None,
    ) -> torch.Tensor:
        """Integrate the learned ODE from ``t_start`` to ``t_end``.

        Parameters
        ----------
        x0 : ``(B, N, 3)`` tensor
            Initial noise configuration.
        cond : optional ``(B, cond_dim)`` tensor
            Per-sample condition.  May be ``None`` for an unconditional
            velocity net.
        n_steps : int, optional
            Number of integration steps.  Defaults to
            ``self.default_n_steps``.
        method : ``"euler"`` | ``"rk4"``, optional
            Integration scheme.  Defaults to ``self.default_method``.
        h_node : optional ``(B, N, hidden_dim)`` tensor
            Pre-computed hidden states.  When supplied the encoder is
            skipped (useful when the same encoder is being used to
            share features across many calls).
        atomic_numbers : optional ``(B, N)`` int64 tensor
            Required together with ``encoder`` when ``h_node`` is not
            supplied.
        edge_index : ``(B, 2, E)`` int64 tensor
            Required graph topology.
        edge_mask : optional ``(B, E)`` bool tensor
            Real-edge mask forwarded to the encoder / velocity net.
        node_mask : optional ``(B, N)`` bool tensor
            Real-atom mask forwarded to the encoder / velocity net
            and applied to the integrated state so padding positions
            stay at zero.
        t_start, t_end : float, optional
            Override the integration interval.  Defaults to ``0`` and
            ``1`` respectively.

        Returns
        -------
        ``(B, N, 3)`` tensor
            The integrated state at ``t_end``.
        """
        if x0.dim() != 3 or x0.shape[-1] != 3:
            raise ValueError(
                f"`x0` must be (B, N, 3); got {tuple(x0.shape)}."
            )
        if edge_index is None:
            raise ValueError("`edge_index` is required for the velocity net.")

        cfg = SamplerConfig(
            n_steps=self.default_n_steps if n_steps is None else int(n_steps),
            method=self.default_method if method is None else method,
            t_start=0.0 if t_start is None else float(t_start),
            t_end=1.0 if t_end is None else float(t_end),
        )

        device = x0.device
        b = x0.shape[0]
        dt = (cfg.t_end - cfg.t_start) / cfg.n_steps

        # Integrate on the same device as the input.  fp32 keeps the
        # numerical behaviour identical across dtypes; the velocity
        # net's first layer will cast it as needed.
        x = x0.to(dtype=torch.float32, device=device)

        # Cache the encoder output once if we have an encoder: the
        # hidden state depends on positions, so we re-run it every
        # step rather than caching a stale tensor.
        for step in range(cfg.n_steps):
            # Time of the current sub-step in [t_start, t_end].
            t_val = cfg.t_start + step * dt
            t_tensor = torch.tensor(t_val, dtype=torch.float32, device=device)

            if cfg.method == "euler":
                v = self._velocity(
                    x, t_tensor, cond, h_node, atomic_numbers, edge_index,
                    edge_mask=edge_mask, node_mask=node_mask,
                )
                # Euler kernel expects `velocity` already scaled by dt.
                x = euler_step(x, v * dt, dt, out=None)

            else:  # rk4
                # RK4 kernel takes the four stage *velocities* (not
                # pre-multiplied by dt) and folds dt/6 * (k1+2 k2+2 k3+k4)
                # into the update itself.  Multiplying by dt here as well
                # would yield dt^2 * v, which is wrong.
                v1 = self._velocity(
                    x, t_tensor, cond, h_node, atomic_numbers, edge_index,
                    edge_mask=edge_mask, node_mask=node_mask,
                )
                x2 = x + 0.5 * dt * v1

                t_mid = t_tensor + 0.5 * dt
                v2 = self._velocity(
                    x2, t_mid, cond, h_node, atomic_numbers, edge_index,
                    edge_mask=edge_mask, node_mask=node_mask,
                )
                x3 = x + 0.5 * dt * v2

                v3 = self._velocity(
                    x3, t_mid, cond, h_node, atomic_numbers, edge_index,
                    edge_mask=edge_mask, node_mask=node_mask,
                )
                x4 = x + dt * v3

                t_end = t_tensor + dt
                v4 = self._velocity(
                    x4, t_end, cond, h_node, atomic_numbers, edge_index,
                    edge_mask=edge_mask, node_mask=node_mask,
                )
                x = rk4_step(x, v1, v2, v3, v4, dt, out=None)

        # Keep padding positions pinned at zero (the velocity net's
        # node_mask zeroes the velocity there, but Euler/RK4 still
        # propagate them through the update; explicit masking is
        # cheap and avoids drift).
        if node_mask is not None:
            x = x * node_mask.unsqueeze(-1).float()

        # Cast back to the input dtype so the caller doesn't see a
        # silent up-cast.
        return x.to(dtype=x0.dtype)


__all__ = ["FlowMatchingSampler", "SamplerConfig"]