"""Conditional flow-matching loss for MolFlow-Triton.

This module hosts the :class:`ConditionalFlowMatchingLoss` training
objective of :mod:`MolFlow-Triton`.  Given a
:class:`models.velocity_net.VelocityNet`, the loss draws random ``t``,
constructs ``(x_t, t, x1 - x0)`` using the helpers in
:mod:`flow_matching.interpolation` and returns the MSE between the
predicted and the target velocities.

The loss is computed in ``float32`` regardless of the input precision
for numerical stability; the predicted velocity is cast back to the
input dtype before being returned so callers can chain it with the
rest of their pipeline without surprises.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from flow_matching.interpolation import interpolate, target_velocity
from flow_matching.optimal_transport import mini_batch_ot_coupling
from models.velocity_net import VelocityNet


# ---------------------------------------------------------------------------
# Loss container
# ---------------------------------------------------------------------------
@dataclass
class LossOutput:
    """Bundle returned by :meth:`ConditionalFlowMatchingLoss.forward`.

    Attributes
    ----------
    loss : scalar tensor
        The MSE between predicted and target velocity, in ``float32``.
    pred_velocity : ``(B, ...)`` tensor
        Predicted velocity, in the same dtype as the model output
        (typically the dtype of ``x1`` / ``cond``).
    target_velocity_ : ``(B, ...)`` tensor
        Ground-truth conditional velocity ``x1 - x0``, in ``float32``
        for numerical stability.
    x_t : ``(B, ...)`` tensor
        Interpolated state at the sampled ``t``.
    t : ``(B,)`` tensor
        Sampled times.
    """

    loss: torch.Tensor
    pred_velocity: torch.Tensor
    target_velocity_: torch.Tensor
    x_t: torch.Tensor
    t: torch.Tensor


# ---------------------------------------------------------------------------
# Conditional flow matching loss
# ---------------------------------------------------------------------------
class ConditionalFlowMatchingLoss(nn.Module):
    """Conditional flow-matching loss using a :class:`VelocityNet`.

    Given a batch of ``(x0, x1, cond)`` triples, this module:

    1. Samples ``t ~ U(0, 1)`` per element of the batch (broadcast as a
       scalar per atom).
    2. Computes ``x_t = (1 - t) * x0 + t * x1``.
    3. Asks the velocity network to predict ``v_theta(x_t, t, cond)``.
    4. Returns ``MSE(v_theta, x1 - x0)`` computed in ``float32``.

    Parameters
    ----------
    model : :class:`VelocityNet`
        The SE(3)-equivariant velocity network.  Its signature must
        match ``model(h_node, positions, t, edge_index, cond)``.
    encoder : callable, optional
        Callable ``encoder(atomic_numbers, positions, edge_index)``
        returning the ``h_node`` tensor consumed by the velocity net.
        When ``None`` the caller must supply ``h_node`` directly to
        :meth:`forward`.  Splitting encoder and velocity net keeps the
        training loop free of module-graph surgery when the encoder is
        frozen or shared.
    conditioner : :class:`models.conditioner.Conditioner`, optional
        When provided, the raw ``(B, n_scalar_channels)`` ``cond`` is
        first encoded through the conditioner to produce the
        ``(B, hidden_dim)`` embedding the velocity net expects.  When
        ``None`` the caller is expected to pass an already-encoded
        condition tensor (and the velocity net must be constructed with
        a matching ``cond_dim``).
    loss_reduction : str, default ``"mean"``
        Reduction applied to the per-element squared errors; one of
        ``"mean"``, ``"sum"`` or ``"none"``.
    use_minibatch_ot : bool, default ``False``
        When ``True`` the forward pass first permutes ``x1`` using
        :func:`flow_matching.optimal_transport.mini_batch_ot_coupling`
        before interpolating.  OT is solved independently within each
        group defined by the ``batch_idx`` keyword argument.  When
        ``False`` the input order is preserved (the historical
        behaviour).  Off by default for backwards compatibility.

    Notes
    -----
    * The loss is always computed in ``float32``; predictions and
      targets are up-cast before subtraction.
    * No device string is hard-coded; every tensor inherits the device
      of its source.
    """

    def __init__(
        self,
        model: VelocityNet,
        encoder: nn.Module | None = None,
        conditioner: nn.Module | None = None,
        loss_reduction: str = "mean",
        use_minibatch_ot: bool = False,
    ) -> None:
        super().__init__()
        if not isinstance(model, VelocityNet):
            raise TypeError(
                f"`model` must be a VelocityNet; got {type(model).__name__}."
            )
        if loss_reduction not in {"mean", "sum", "none"}:
            raise ValueError(
                f"`loss_reduction` must be one of 'mean', 'sum', 'none'; "
                f"got {loss_reduction!r}."
            )
        if not isinstance(use_minibatch_ot, bool):
            raise TypeError(
                f"`use_minibatch_ot` must be a bool; got "
                f"{type(use_minibatch_ot).__name__}."
            )

        self.model = model
        self.encoder = encoder
        self.conditioner = conditioner
        self.loss_reduction = loss_reduction
        self.use_minibatch_ot = use_minibatch_ot

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _sample_t(batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample ``t`` per element.  Shape ``(B,)`` in ``[0, 1]``."""
        # Use float32 sampling then broadcast as needed downstream.
        return torch.rand(batch_size, dtype=torch.float32, device=device)

    def _get_h_node(
        self,
        h_node: torch.Tensor | None,
        atomic_numbers: torch.Tensor | None,
        positions: torch.Tensor,
        edge_index: torch.Tensor,
        edge_mask: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return ``h_node``, either supplied directly or produced by the encoder."""
        if h_node is not None:
            return h_node
        if self.encoder is None or atomic_numbers is None:
            raise ValueError(
                "Either `h_node` must be provided, or an `encoder` together "
                "with `atomic_numbers` must be supplied to "
                "`ConditionalFlowMatchingLoss.forward`."
            )
        h_node, _ = self.encoder(
            atomic_numbers, positions, edge_index,
            edge_mask=edge_mask, node_mask=node_mask,
        )
        return h_node

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        cond: torch.Tensor | None = None,
        *,
        h_node: torch.Tensor | None = None,
        atomic_numbers: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_mask: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
        batch_idx: torch.Tensor | None = None,
        ot_diagnostics: list[dict] | None = None,
    ) -> LossOutput:
        """Compute the conditional flow-matching loss.

        Parameters
        ----------
        x0 : ``(B, N, 3)`` tensor
            Noise / source positions.
        x1 : ``(B, N, 3)`` tensor
            Data / target positions.  Must share shape with ``x0``.
        cond : optional ``(B, n_scalar_channels)`` or ``(B, cond_dim)`` tensor
            Per-sample condition.  When a :class:`Conditioner` was
            registered at construction time, ``cond`` is the raw scalar
            input ``(B, n_scalar_channels)`` (e.g. ``(B, 1)`` for a
            single QM9 property); the conditioner maps it to the
            ``(B, hidden_dim)`` embedding the velocity net expects.
            When no conditioner is configured, ``cond`` is taken as the
            already-encoded embedding and forwarded directly.  May be
            ``None`` when the velocity net was constructed with
            ``cond_dim=0``.
        h_node : optional ``(B, N, hidden_dim)`` tensor
            Pre-computed per-atom hidden states.  When supplied the
            encoder is skipped.
        atomic_numbers : optional ``(B, N)`` int64 tensor
            Required when an encoder is configured and ``h_node`` is
            not provided.
        edge_index : ``(B, 2, E)`` int64 tensor
            Required; the velocity net needs the graph topology.
        edge_mask : optional ``(B, E)`` bool tensor
            Real-edge mask; forwarded to the encoder and the
            velocity net.
        node_mask : optional ``(B, N)`` bool tensor
            Real-atom mask; the loss is averaged only over positions
            where this is ``True`` (padding positions contribute
            zero and don't pollute the gradient).
        batch_idx : optional ``(B,)`` long tensor
            Group identifier per row.  Only used when the loss was
            constructed with ``use_minibatch_ot=True``: rows sharing
            the same ``batch_idx`` value are coupled by
            :func:`mini_batch_ot_coupling` before interpolation.
            Ignored when ``use_minibatch_ot=False``.
        ot_diagnostics : optional list of dicts
            When supplied, append effective OT backend/device and fallback
            records for each coupled group. Does not alter the loss or
            pairing and is unused when mini-batch OT is disabled.

        Returns
        -------
        :class:`LossOutput`
        """
        if x0.shape != x1.shape:
            raise ValueError(
                f"`x0` and `x1` must share shape; got {tuple(x0.shape)} vs "
                f"{tuple(x1.shape)}."
            )
        if x0.dim() != 3 or x0.shape[-1] != 3:
            raise ValueError(
                f"`x0` and `x1` must be (B, N, 3); got {tuple(x0.shape)}."
            )
        if edge_index is None:
            raise ValueError("`edge_index` is required for the velocity net.")

        b = x0.shape[0]

        # 0. Optional mini-batch OT coupling: permute x1 within each
        # batch_idx group using the configured OT plan and rounding. Skipped
        # entirely when the flag is off (default) so existing call
        # sites keep their input ordering.
        if self.use_minibatch_ot:
            if batch_idx is None:
                raise ValueError(
                    "`batch_idx` is required when `use_minibatch_ot=True`."
                )
            ot_idx = mini_batch_ot_coupling(x1, x0, batch_idx, diagnostics=ot_diagnostics)
            x1 = x1.index_select(0, ot_idx.to(device=x1.device))

        # 1. Sample t, build interpolated state and target velocity.
        t = self._sample_t(b, x0.device)
        x_t = interpolate(x0, x1, t)
        with torch.no_grad():
            target_v = target_velocity(x0, x1).detach()

        # 2. Encode (if needed) and predict velocity.
        h_node_eff = self._get_h_node(
            h_node, atomic_numbers, x_t, edge_index,
            edge_mask=edge_mask, node_mask=node_mask,
        )
        # If a conditioner was supplied, treat ``cond`` as raw scalar
        # features (e.g. a single QM9 property) and lift them through the
        # conditioner into the velocity net's expected embedding.
        cond_eff: torch.Tensor | None = cond
        if cond_eff is not None and self.conditioner is not None:
            cond_eff = self.conditioner(cond_eff)
        pred_v = self.model(
            h_node_eff, x_t, t, edge_index, cond=cond_eff,
            edge_mask=edge_mask, node_mask=node_mask,
        )

        if pred_v.shape != x_t.shape:
            raise ValueError(
                "VelocityNet returned a tensor of shape "
                f"{tuple(pred_v.shape)}, expected {tuple(x_t.shape)}."
            )

        # 3. Loss in fp32 for numerical stability.  When a node_mask is
        # supplied, average only over real atoms (padding positions
        # contribute zero and are excluded from the denominator).
        pred_v_f32 = pred_v.to(dtype=torch.float32)
        target_v_f32 = target_v.to(dtype=torch.float32)
        per_element = (pred_v_f32 - target_v_f32).pow(2)
        if node_mask is not None:
            mask_3d = node_mask.unsqueeze(-1).to(dtype=per_element.dtype)
            per_element = per_element * mask_3d
            denom = mask_3d.sum().clamp(min=1.0)
        else:
            denom = None
        if self.loss_reduction == "mean":
            if denom is not None:
                loss = per_element.sum() / denom
            else:
                loss = per_element.mean()
        elif self.loss_reduction == "sum":
            loss = per_element.sum()
        else:  # "none"
            loss = per_element

        return LossOutput(
            loss=loss,
            pred_velocity=pred_v,
            target_velocity_=target_v_f32,
            x_t=x_t,
            t=t,
        )


__all__ = ["LossOutput", "ConditionalFlowMatchingLoss"]
