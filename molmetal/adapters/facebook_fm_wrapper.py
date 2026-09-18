"""FacebookFMWrapper — thin LipmanFlowMatchingAdapter-compatible shim.

Wraps facebookresearch/flow_matching (cloned at
``molmetal/references/flow_matching``) into the same ``MoleculeGenerator``
contract used by :mod:`molmetal.adapters.flow_matching_lipman`:

  - ``name``                       -> "FacebookFM_v1"
  - ``setup(device=None)``         -> construct path, scheduler, model
  - ``train_step(mols, ...)``      -> 1 CFM step with upstream ``AffineProbPath``
  - ``generate(pocket, config)``   -> ``ODESolver.sample`` integration
  - ``last_losses`` dict           -> ``{"cfm": ..., "atom": ..., "total": ...}``

Upstream API used (verified 2026-09-16):

    from flow_matching.path import AffineProbPath        # x_t = α_t x_1 + σ_t x_0
    from flow_matching.path.scheduler import CondOTScheduler  # α_t = t, σ_t = 1 - t
    from flow_matching.solver import ODESolver            # .sample(x_init, step_size, ...)
    from flow_matching.utils import ModelWrapper          # forward(x, t, **extras)
    from flow_matching.loss import MixturePathGeneralizedKL  # (unused here)

References:
  Lipman, Y., Chen, R. T. Q., Ben-Hamu, H., Nickel, M., Le, M. (2023).
    "Flow Matching for Generative Modeling." ICLR 2023.  arXiv:2210.02747.

Constraint: ``molmetal/adapters/flow_matching_lipman/__init__.py`` is NOT
modified — this wrapper is additive.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Upstream import — locate the cloned repo at molmetal/references/flow_matching
# ---------------------------------------------------------------------------
_HERE = Path(__file__).resolve()
_UPSTREAM = _HERE.parents[1] / "references" / "flow_matching"
if _UPSTREAM.is_dir() and str(_UPSTREAM) not in sys.path:
    sys.path.insert(0, str(_UPSTREAM))

from flow_matching.path import AffineProbPath           # noqa: E402
from flow_matching.path.scheduler import CondOTScheduler  # noqa: E402
from flow_matching.solver import ODESolver              # noqa: E402
from flow_matching.utils import ModelWrapper            # noqa: E402


# ---------------------------------------------------------------------------
# Lightweight velocity field — small MLP that mirrors the (B, N, 3) shape
# contract of the upstream AffineProbPath.  Real production code can swap in
# our EGNN by setting ``velocity_field=...`` after construction.
# ---------------------------------------------------------------------------
class _SimpleVelocityField(nn.Module):
    """MLP velocity field: (B, N, 3) + t  -> (B, N, 3).

    Parameters are tiny (~200) so the smoke train below completes in
    seconds on CPU.  Production callers should inject a real EGNN.
    """

    def __init__(self, hidden_dim: int = 32):
        super().__init__()
        self.proj_in = nn.Linear(3 + 1, hidden_dim)
        self.proj_out = nn.Linear(hidden_dim, 3)

    def forward(self, x: torch.Tensor = None, x_t: torch.Tensor = None, t: torch.Tensor = None, **extras) -> torch.Tensor:
        # Accept either ``x`` (upstream ModelWrapper convention) or
        # ``x_t`` (intuitive name).  ``t`` may be (B,) or scalar (0-D);
        # torchdiffeq passes a 0-D tensor during ODE integration.
        if x is None:
            x = x_t
        B, N, _ = x.shape
        if t.ndim == 0:
            t_scalar = t.view(1).expand(B)  # broadcast to (B,)
        elif t.shape[0] == 1:
            t_scalar = t.expand(B)
        else:
            t_scalar = t
        t_b = t_scalar[:, None, None].expand(B, N, 1)
        h = F.silu(self.proj_in(torch.cat([x, t_b], dim=-1)))
        return self.proj_out(h)

    def call_direct(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Convenience: train_step calls this directly to skip the kwargs layer."""
        return self.forward(x=x_t, t=t)


# ---------------------------------------------------------------------------
# Wrapper that exposes the upstream API behind our adapter interface
# ---------------------------------------------------------------------------
class FacebookFMWrapper:
    """LipmanFlowMatchingAdapter-compatible wrapper around facebookresearch/flow_matching.

    The public surface intentionally matches the existing handwritten
    adapter so that future swappable wiring is a one-line import change:

        # before
        from molmetal.adapters.flow_matching_lipman import LipmanFlowMatchingAdapter
        # after
        from molmetal.adapters.facebook_fm_wrapper import FacebookFMWrapper as Adapter
    """

    name = "FacebookFM_v1"

    def __init__(
        self,
        hidden_dim: int = 32,
        lr: float = 1e-3,
        scheduler_kind: str = "condot",
        n_atoms: int = 8,
        device: Optional[str] = None,
    ):
        self._hidden_dim = hidden_dim
        self._lr = lr
        self._scheduler_kind = scheduler_kind
        self._n_atoms = n_atoms
        self._explicit_device = device
        # Filled by setup()
        self.velocity_field: Optional[nn.Module] = None
        self.optimizer: Optional[torch.optim.Optimizer] = None
        self.path: Optional[AffineProbPath] = None
        self.scheduler = None
        self._odesolver: Optional[ODESolver] = None
        self.device: torch.device = torch.device(
            device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.last_losses: dict = {}

    # ------------------------------------------------------------------
    def setup(self, device: Optional[str] = None) -> None:
        """Construct path + scheduler + velocity field + optimizer."""
        if device is not None:
            self.device = torch.device(device)

        # 1) Scheduler (Lipman 2023 §4.7) — CondOT gives α_t = t, σ_t = 1 - t.
        if self._scheduler_kind == "condot":
            self.scheduler = CondOTScheduler()
        else:
            raise ValueError(f"Unknown scheduler_kind={self._scheduler_kind!r}")

        # 2) Probability path (Lipman 2023 §4.8) — AffineProbPath wraps scheduler.
        self.path = AffineProbPath(scheduler=self.scheduler)

        # 3) Velocity field — simple MLP by default; swappable later.
        self.velocity_field = _SimpleVelocityField(self._hidden_dim).to(self.device)

        # 4) ODESolver — the upstream solver expects a ModelWrapper (or callable)
        # that takes ``(x, t, **extras)``.  ModelWrapper stores a model and
        # forwards extras; our SimpleVelocityField already accepts (x, t).
        self._odesolver = ODESolver(
            velocity_model=ModelWrapper(self.velocity_field)
        )

        # 5) Atom-type head + joint optimizer (created lazily in train_step
        # too; we seed it here so the optim knows the params before step 0).
        self._atom_head = nn.Linear(3, 11).to(self.device)
        self.optimizer = torch.optim.Adam(
            list(self.velocity_field.parameters()) + list(self._atom_head.parameters()),
            lr=self._lr,
        )

    # ------------------------------------------------------------------
    def train_step(
        self,
        mols,                         # List[dict-like] with .coords .atom_types
        atom_loss_weight: float = 0.1,
    ) -> float:
        """One CFM step using the upstream AffineProbPath.sample(...).

        Returns the TOTAL loss (CFM + α · atom-CE) — same contract as
        LipmanFlowMatchingAdapter.train_step.
        """
        assert self.path is not None, "call setup() first"

        # ---- 1) Build batched tensors (B, N, 3) -----------------------
        b = len(mols)
        max_n = max(int(getattr(m, "coords").shape[0]) for m in mols)
        x_1 = torch.zeros(b, max_n, 3, device=self.device)
        atom_types = torch.zeros(b, max_n, dtype=torch.long, device=self.device)
        mask = torch.zeros(b, max_n, dtype=torch.bool, device=self.device)
        for i, m in enumerate(mols):
            n = int(m.coords.shape[0])
            x_1[i, :n] = m.coords.to(self.device)
            atom_types[i, :n] = m.atom_types.to(self.device)
            mask[i, :n] = True

        # ---- 2) Sample noise + random t ---------------------------
        x_0 = torch.randn_like(x_1)
        t = torch.rand(b, device=self.device).clamp(min=1e-3, max=1.0 - 1e-3)

        # ---- 3) Upstream conditional path sample ----------------
        path_sample = self.path.sample(x_0=x_0, x_1=x_1, t=t)
        x_t = path_sample.x_t
        dx_t_target = path_sample.dx_t

        # ---- 4) Predict velocity and compute CFM MSE -------------
        v_pred = self.velocity_field.call_direct(x_t, t)
        cfm_loss = F.mse_loss(v_pred[mask], dx_t_target[mask])

        # ---- 5) Atom-type CE loss (same as handwritten adapter) -
        # Project velocity vectors to (B, N, 11) logits via the tiny
        # _atom_head (created in setup()), then CE on the masked subset.
        # We use 11 atomic-number classes (1, 6, 7, 8, 9, 15, 16, 17,
        # 34, 35, 53) — same set as the handwritten adapter's vocab_mask.
        atom_logits = self._atom_head(v_pred)  # (B, N, 11)
        flat_logits = atom_logits[mask]        # (M, 11)
        flat_targets = atom_types[mask].clamp(min=1, max=11) - 1  # to 0..10
        atom_loss = F.cross_entropy(flat_logits, flat_targets)

        loss = cfm_loss + atom_loss_weight * atom_loss

        # ---- 6) Optimise -----------------------------------------
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.velocity_field.parameters(), 1.0)
        self.optimizer.step()

        self.last_losses = {
            "cfm": float(cfm_loss.item()),
            "atom": float(atom_loss.item()),
            "total": float(loss.item()),
        }
        return float(loss.item())

    # ------------------------------------------------------------------
    def generate(self, pocket=None, config=None):
        """Sample via the upstream ODESolver.sample.

        ``config`` is duck-typed: needs ``n_samples``, ``n_steps``, ``seed``,
        ``n_atoms`` (default 8).  Returns a list of (coords, atom_types)
        pairs to mimic the existing adapter's ``List[Molecule]``.
        """
        assert self._odesolver is not None, "call setup() first"
        n_samples = int(getattr(config, "n_samples", 8))
        n_steps = int(getattr(config, "n_steps", 50))
        seed = int(getattr(config, "seed", 0))
        n_atoms = int(getattr(config, "n_atoms", self._n_atoms))

        gen = torch.Generator(device=self.device).manual_seed(seed)
        x_init = torch.randn(n_samples, n_atoms, 3, device=self.device, generator=gen)
        time_grid = torch.linspace(0.0, 1.0, n_steps + 1, device=self.device)

        x_1 = self._odesolver.sample(
            x_init=x_init,
            step_size=None,           # let the time_grid drive stepping
            method="euler",
            time_grid=time_grid,
            return_intermediates=False,
            enable_grad=False,
        )

        # Dummy atom-types for now (round-trip with upstream, no
        # chemistry decoder — that lives in the real EGNNVelocityField).
        atom_types = torch.zeros(n_samples, n_atoms, dtype=torch.long, device=self.device)

        class _M:
            __slots__ = ("coords", "atom_types")
            def __init__(self, coords, atom_types):
                self.coords = coords
                self.atom_types = atom_types
        return [_M(x_1[i].detach().cpu(), atom_types[i].detach().cpu()) for i in range(n_samples)]


__all__ = ["FacebookFMWrapper"]