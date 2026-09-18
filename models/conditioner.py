"""Property / condition embedding for MolFlow-Triton.

The flow-matching velocity model is conditioned on scalar physical /
molecular properties such as temperature, pressure, target pIC50,
etc.  This module provides:

- :class:`GaussianSmearing` - radial-basis (RBF) expansion of a scalar
  onto a fixed Gaussian grid; this is the same trick used in SchNet /
  EGNN / SphereNet to give the network a smoothly varying basis.
- :class:`Conditioner` - small MLP that maps the flattened RBF (and
  optional raw) features into a single condition vector, ready to be
  broadcast-added to per-atom hidden states.
"""

from __future__ import annotations

import torch
from torch import nn


class GaussianSmearing(nn.Module):
    """Expand scalar inputs onto a fixed Gaussian basis.

    Implements the standard RBF expansion:

    .. math::

        e_k(x) = \\exp(-\\gamma \\,(x - \\mu_k)^2)

    where the centres ``\\mu_k`` are equally spaced in
    ``[start, stop]`` and ``\\gamma = 1 / (2 \\sigma^2)`` is derived
    from the spacing.

    Parameters
    ----------
    start : float
        Lower edge of the RBF domain.
    stop : float
        Upper edge of the RBF domain.
    num_gaussians : int
        Number of Gaussian basis functions (output width).
    trainable : bool, default ``False``
        If ``True``, the widths and centres become learnable
        parameters; otherwise they are fixed buffers.

    Notes
    -----
    Inputs outside ``[start, stop]`` are still expanded - the Gaussian
    simply decays, no clamping is applied.  This matches typical SchNet
    behaviour and avoids creating hard decision boundaries.
    """

    def __init__(
        self,
        start: float = 0.0,
        stop: float = 10.0,
        num_gaussians: int = 64,
        trainable: bool = False,
    ) -> None:
        super().__init__()
        if num_gaussians < 1:
            raise ValueError("`num_gaussians` must be >= 1")
        if stop <= start:
            raise ValueError(
                f"`stop` ({stop}) must be strictly greater than `start` ({start})."
            )

        self.num_gaussians = num_gaussians
        self.trainable = trainable

        offset = (stop - start) / (num_gaussians - 1) if num_gaussians > 1 else 0.0
        centres = torch.linspace(start, stop, num_gaussians)
        # Width: 1 / (2 * sigma^2); choose sigma ~= offset so neighbouring
        # Gaussians overlap meaningfully.
        sigma = offset if offset > 0 else (stop - start)
        widths = torch.full((num_gaussians,), 1.0 / (2.0 * sigma * sigma))

        if trainable:
            self.centres = nn.Parameter(centres)
            self.widths = nn.Parameter(widths)
        else:
            self.register_buffer("centres", centres)
            self.register_buffer("widths", widths)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Expand ``x`` onto the Gaussian basis.

        Parameters
        ----------
        x : ``(...)`` tensor
            Scalar inputs of any shape.

        Returns
        -------
        ``(..., num_gaussians)`` tensor
            RBF expansion of ``x``.  All leading dimensions of ``x``
            are preserved.
        """
        # (..., 1) - (G,) -> (..., G)
        diff = x.unsqueeze(-1) - self.centres
        return torch.exp(-self.widths * diff * diff)


class Conditioner(nn.Module):
    """Encode scalar conditions into a dense vector.

    The conditioner first expands each scalar input with its own
    :class:`GaussianSmearing` (one smearing per channel), concatenates
    the resulting RBF vectors and (optionally) the raw inputs, and
    passes them through an MLP.  The output is a single embedding
    suitable for broadcast-addition into a per-atom hidden state.

    Parameters
    ----------
    n_scalar_channels : int
        Number of scalar inputs expected per sample.  E.g. 2 for
        ``(temperature, pressure)``.
    smear_bonds : list[tuple[float, float, int]], optional
        Per-channel RBF configuration, given as
        ``(start, stop, num_gaussians)``.  When ``None`` a sensible
        default of ``(channel_min - 3, channel_max + 3, 32)`` is used
        once ``fit_defaults`` is called.
    hidden_dim : int, default 128
        Width of the conditioner MLP and the output embedding.
    n_layers : int, default 2
        Number of linear layers in the conditioner MLP (excluding the
        final projection).
    include_raw : bool, default ``True``
        If ``True``, the raw scalar values are concatenated to the
        RBF expansion before the MLP.  Useful for keeping exact
        magnitudes available.
    """

    def __init__(
        self,
        n_scalar_channels: int,
        smear_bonds: list[tuple[float, float, int]] | None = None,
        hidden_dim: int = 128,
        n_layers: int = 2,
        include_raw: bool = True,
    ) -> None:
        super().__init__()
        if n_scalar_channels < 1:
            raise ValueError("`n_scalar_channels` must be >= 1")
        if hidden_dim < 1:
            raise ValueError("`hidden_dim` must be >= 1")
        if n_layers < 1:
            raise ValueError("`n_layers` must be >= 1")

        self.n_scalar_channels = n_scalar_channels
        self.hidden_dim = hidden_dim
        self.include_raw = include_raw

        if smear_bonds is not None:
            if len(smear_bonds) != n_scalar_channels:
                raise ValueError(
                    "`smear_bonds` length must match `n_scalar_channels`; "
                    f"got {len(smear_bonds)} vs {n_scalar_channels}."
                )
            smear_modules = [
                GaussianSmearing(start=s, stop=e, num_gaussians=g)
                for s, e, g in smear_bonds
            ]
        else:
            # Generic default covering temperature, pressure, log-scale
            # toxicity values, etc.  Individual channels can override
            # via `smear_bonds`.
            smear_modules = [
                GaussianSmearing(start=-5.0, stop=5.0, num_gaussians=32)
                for _ in range(n_scalar_channels)
            ]
        self.smearers = nn.ModuleList(smear_modules)

        rbf_width = sum(m.num_gaussians for m in smear_modules)
        in_dim = rbf_width + (n_scalar_channels if include_raw else 0)

        layers: list[nn.Module] = []
        d = in_dim
        for _ in range(n_layers - 1):
            layers += [nn.Linear(d, hidden_dim), nn.SiLU()]
            d = hidden_dim
        layers += [nn.Linear(d, hidden_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, cond: torch.Tensor) -> torch.Tensor:
        """Encode scalar conditions.

        Parameters
        ----------
        cond : ``(B, n_scalar_channels)`` tensor
            Scalar physical / molecular properties per sample.

        Returns
        -------
        ``(B, hidden_dim)`` tensor
            Per-sample condition embedding, ready to be added to a
            ``(B, N, hidden_dim)`` hidden state.
        """
        if cond.dim() != 2 or cond.shape[-1] != self.n_scalar_channels:
            raise ValueError(
                f"`cond` must be (B, {self.n_scalar_channels}); "
                f"got {tuple(cond.shape)}."
            )

        rbf_parts = [m(cond[:, i]) for i, m in enumerate(self.smearers)]
        rbf = torch.cat(rbf_parts, dim=-1)

        if self.include_raw:
            rbf = torch.cat([rbf, cond], dim=-1)

        return self.mlp(rbf)