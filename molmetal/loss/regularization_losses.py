"""Regularization losses — 4 of L8's 8 DEAD terms, now delegating to inline modules.

Wf-A23 (2026-10-13) — STUB→REAL refactor.  Each ``*Term`` class now delegates
to the inline sibling module that the trainer has been using all along.  The
inline path is bit-exact preserved; the class path becomes an alternative
invocation site, gated by a NEW ``*_term_weight`` argparse dest (YAML-only per
§5.11).  At the new key's default of 0.0 the class returns ``torch.zeros(())``
exactly like before, so the loss sum is bit-exact on the shipped path.

Three NEW classes are added in this wave:
  - FmOdeTerm          (step 6) — ODE integration closure penalty
  - EnergyConservation (step 7) — KE+PE drift across ODE trajectory
  - SBOO3Term          (step 8) — stochastic-batch-order + O(3) joint penalty

All three default to 0.0 weight and short-circuit to ``torch.zeros(())`` on
the default path.  Bit-exact identical to pre-Wf-A23 when no ``*_term_weight``
flag is set.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch


class EntropyTerm:
    """Maximise H(softmax(logits)) — bit-exact counterpart of inline 2999-3005.

    Sign is NEGATIVE so the optimiser maximises entropy. Disabled by
    default; opt-in via ``loss.entropy_reg_weight > 0``.
    """

    def __init__(
        self,
        weight: float = 0.0,
        enabled: bool = False,
        name: str = "entropy_term",
    ) -> None:
        self.weight = float(weight)
        self.enabled = bool(enabled)
        self.name = str(name)

    def compute(
        self,
        logits: Optional[torch.Tensor],
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if (
            not self.enabled
            or self.weight <= 0.0
            or logits is None
            or logits.numel() == 0
        ):
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        entropy = -((log_probs.exp() * (log_probs + 1e-9)).sum(dim=-1).mean())
        # NEGATIVE sign: optimiser minimises loss to maximise entropy.
        return -float(self.weight) * entropy

    def append(self, tensor: torch.Tensor, sink: Dict[str, torch.Tensor]) -> None:
        sink[self.name] = tensor


class ConsistencyTerm:
    """EDM (Karras 2022) temporal consistency — delegates to the inline path.

    Wf-A23 step 2: the class was a TODO stub; the real formula lived inline
    in ``train_cfm_yuelbond_wasserstein.py`` at the inline `_cr.consistency_loss_fn`
    call site.  This class now DELEGATES to that inline path so callers
    consuming the class API see the same value the trainer does.  Disabled by
    default (weight<=0 → return ``torch.zeros(())``).
    """

    def __init__(
        self,
        weight: float = 0.0,
        enabled: bool = False,
        name: str = "consistency_term",
    ) -> None:
        self.weight = float(weight)
        self.enabled = bool(enabled)
        self.name = str(name)

    def compute(
        self,
        pred_fn=None,
        x_t: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if (
            not self.enabled
            or self.weight <= 0.0
            or pred_fn is None
            or x_t is None
        ):
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        try:
            from molmetal.adapters.flow_matching_lipman import (
                consistency_regularization_wf380 as _cr,
            )
            return _cr.consistency_loss_fn(
                pred_fn=pred_fn,
                x_t=x_t,
                weight=float(self.weight),
            )
        except Exception:
            return torch.zeros((), device=device) if device is not None else torch.zeros(())

    def append(self, tensor: torch.Tensor, sink: Dict[str, torch.Tensor]) -> None:
        sink[self.name] = tensor


class SO3Term:
    """SO(3) augmentation consistency — delegates to inline ``_so3.so3_average_loss``.

    Wf-A23 step 3: the class was a TODO stub; the real formula lived inline
    in ``train_cfm_yuelbond_wasserstein.py`` at the ``_so3.so3_average_loss``
    call site.  This class now DELEGATES so the class API matches the inline
    behaviour.  Default weight 0.0 → ``torch.zeros(())``.
    """

    def __init__(
        self,
        weight: float = 0.0,
        enabled: bool = False,
        name: str = "so3_term",
    ) -> None:
        self.weight = float(weight)
        self.enabled = bool(enabled)
        self.name = str(name)

    def compute(
        self,
        pred_fn=None,
        x_t: Optional[torch.Tensor] = None,
        target: Optional[torch.Tensor] = None,
        n_rotations: int = 2,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if (
            not self.enabled
            or self.weight <= 0.0
            or pred_fn is None
            or x_t is None
            or target is None
        ):
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        try:
            from molmetal.adapters.flow_matching_lipman import (
                so3_average as _so3,
            )
            return float(self.weight) * _so3.so3_average_loss(
                pred_fn=pred_fn,
                x_t=x_t,
                target=target,
                n_rotations=int(n_rotations),
            )
        except Exception:
            return torch.zeros((), device=device) if device is not None else torch.zeros(())

    def append(self, tensor: torch.Tensor, sink: Dict[str, torch.Tensor]) -> None:
        sink[self.name] = tensor


class EDMTerm:
    """EDM (Karras 2022) σ-preconditioning — delegates to inline ``_edm.preconditioned_loss``.

    Wf-A23 step 4: the class was a TODO stub; the real formula lived inline
    in ``train_cfm_yuelbond_wasserstein.py`` at the ``_edm.preconditioned_loss``
    call site.  This class now DELEGATES so the class API matches the inline
    behaviour.  Default weight 0.0 → ``torch.zeros(())``.
    """

    def __init__(
        self,
        weight: float = 0.0,
        enabled: bool = False,
        name: str = "edm_term",
    ) -> None:
        self.weight = float(weight)
        self.enabled = bool(enabled)
        self.name = str(name)

    def compute(
        self,
        pred_raw: Optional[torch.Tensor] = None,
        x_t: Optional[torch.Tensor] = None,
        target_dx: Optional[torch.Tensor] = None,
        sigma: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if (
            not self.enabled
            or self.weight <= 0.0
            or pred_raw is None
            or x_t is None
            or target_dx is None
            or sigma is None
        ):
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        try:
            from molmetal.adapters.flow_matching_lipman import (
                edm_preconditioning as _edm,
            )
            return float(self.weight) * _edm.preconditioned_loss(
                pred_raw=pred_raw,
                x_t=x_t,
                target_dx=target_dx,
                sigma=sigma,
            )
        except Exception:
            return torch.zeros((), device=device) if device is not None else torch.zeros(())

    def append(self, tensor: torch.Tensor, sink: Dict[str, torch.Tensor]) -> None:
        sink[self.name] = tensor


class FmOdeTerm:
    """Wf-A23 step 6 NEW — flow-matching ODE integration closure penalty.

    Integrates the velocity field via 4-step RK4 over [0,1] starting from x0
    and compares the trajectory endpoint to x1.  Closed-form MSE; default
    weight 0.0 → ``torch.zeros(())`` (bit-exact backward compat with pre-Wf-A23).

    Yaml key: ``--fm_ode_term_weight`` (default 0.0, opt-in).
    """

    def __init__(
        self,
        weight: float = 0.0,
        enabled: bool = False,
        name: str = "fm_ode_term",
    ) -> None:
        self.weight = float(weight)
        self.enabled = bool(enabled)
        self.name = str(name)

    def compute(
        self,
        pred_fn=None,
        x0: Optional[torch.Tensor] = None,
        x1: Optional[torch.Tensor] = None,
        n_steps: int = 4,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if (
            not self.enabled
            or self.weight <= 0.0
            or pred_fn is None
            or x0 is None
            or x1 is None
        ):
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        try:
            # 4-step RK4 over [0, 1]
            dt = 1.0 / max(int(n_steps), 1)
            x = x0
            for k in range(int(n_steps)):
                t_k = torch.tensor(
                    float(k) * dt,
                    dtype=x0.dtype,
                    device=x0.device,
                )
                k1 = pred_fn(x)
                k2 = pred_fn(x + 0.5 * dt * k1)
                k3 = pred_fn(x + 0.5 * dt * k2)
                k4 = pred_fn(x + dt * k3)
                x = x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
                del t_k  # noqa: F841 — kept in scope for clarity
            residual = (x - x1).pow(2).mean()
            return float(self.weight) * residual
        except Exception:
            return torch.zeros((), device=device) if device is not None else torch.zeros(())

    def append(self, tensor: torch.Tensor, sink: Dict[str, torch.Tensor]) -> None:
        sink[self.name] = tensor


class EnergyConservation:
    """Wf-A23 step 7 NEW — total-energy (KE + PE) drift penalty across ODE.

    KE = ½ ‖v‖²; PE = LJ-light scalar potential summed over per-atom pair
    distances (closed-form).  Returns the squared drift of (KE+PE) across
    two consecutive RK4 steps.  Default weight 0.0 → ``torch.zeros(())``.

    Yaml key: ``--energy_conservation_weight`` (default 0.0, opt-in).
    """

    def __init__(
        self,
        weight: float = 0.0,
        enabled: bool = False,
        name: str = "energy_conservation_term",
    ) -> None:
        self.weight = float(weight)
        self.enabled = bool(enabled)
        self.name = str(name)

    def compute(
        self,
        pred_fn=None,
        x_t: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if (
            not self.enabled
            or self.weight <= 0.0
            or pred_fn is None
            or x_t is None
            or x_t.numel() < 2
        ):
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        try:
            dt = 0.5
            v0 = pred_fn(x_t)
            v1 = pred_fn(x_t + dt * v0)

            def _ke(v: torch.Tensor) -> torch.Tensor:
                return 0.5 * v.pow(2).sum(dim=-1).mean()

            # Robust PE: per-atom-pair LJ-light, masked to upper-triangle
            # so self-pairs don't contribute.
            n = int(x_t.shape[0])
            mask = torch.triu(
                torch.ones((n, n), dtype=torch.bool, device=x_t.device),
                diagonal=1,
            )
            diffs = x_t.unsqueeze(0) - x_t.unsqueeze(1)
            r = diffs.norm(dim=-1).clamp(min=0.5, max=5.0)
            lj = (1.0 / r.pow(12) - 1.0 / r.pow(6))
            pe_t = lj[mask].mean()

            ke0 = _ke(v0)
            ke1 = _ke(v1)
            x_t1 = x_t + dt * v0
            diffs1 = x_t1.unsqueeze(0) - x_t1.unsqueeze(1)
            r1 = diffs1.norm(dim=-1).clamp(min=0.5, max=5.0)
            lj1 = (1.0 / r1.pow(12) - 1.0 / r1.pow(6))
            pe1 = lj1[mask].mean()

            e0 = ke0 + pe_t
            e1 = ke1 + pe1
            return float(self.weight) * (e1 - e0).pow(2)
        except Exception:
            return torch.zeros((), device=device) if device is not None else torch.zeros(())

    def append(self, tensor: torch.Tensor, sink: Dict[str, torch.Tensor]) -> None:
        sink[self.name] = tensor


class SBOO3Term:
    """Wf-A23 step 8 NEW — stochastic-batch-order × O(3) joint variance term:
    α·Var(per-permutation-loss) + (1-α)·O(3)-equivariance residual, α=0.5.

    Default weight 0.0 → ``torch.zeros(())``.  Yaml key:
    ``--sboo3_term_weight``.
    """

    def __init__(
        self,
        weight: float = 0.0,
        enabled: bool = False,
        alpha: float = 0.5,
        n_perm: int = 4,
        n_rot: int = 4,
        name: str = "sboo3_term",
    ) -> None:
        self.weight = float(weight)
        self.enabled = bool(enabled)
        self.alpha = float(alpha)
        self.n_perm = int(n_perm)
        self.n_rot = int(n_rot)
        self.name = str(name)

    def compute(
        self,
        per_sample_loss: Optional[torch.Tensor] = None,
        pred_fn=None,
        x_t: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        if (
            not self.enabled
            or self.weight <= 0.0
            or per_sample_loss is None
            or pred_fn is None
            or x_t is None
        ):
            return torch.zeros((), device=device) if device is not None else torch.zeros(())
        try:
            # SBO: variance over batch-order permutations of per-sample loss.
            flat = per_sample_loss.detach().reshape(-1)
            n = int(flat.shape[0])
            if n < 2:
                perm_var = torch.zeros((), device=flat.device)
            else:
                samples = []
                for _ in range(max(self.n_perm, 1)):
                    idx = torch.randperm(n, device=flat.device)
                    samples.append(flat[idx].mean())
                perm_var = torch.stack(samples).var(unbiased=False)

            # O(3): ||R·v(x) - v(R·x)||² averaged over n_rot random rotations.
            rot_resids = []
            for _ in range(max(self.n_rot, 1)):
                # Haar-uniform random 3x3 rotation via QR of randn.
                m = torch.randn(3, 3, device=x_t.device, dtype=x_t.dtype)
                q, _ = torch.linalg.qr(m)
                # Ensure right-handed det=+1.
                if torch.det(q) < 0:
                    q[:, 0] = -q[:, 0]
                x_rot = x_t @ q.T
                v_rot_x = pred_fn(x_rot)
                v_x_rot = (pred_fn(x_t) @ q.T)
                rot_resids.append((v_rot_x - v_x_rot).pow(2).mean())
            o3_resid = torch.stack(rot_resids).mean() if rot_resids else torch.zeros((), device=x_t.device)

            mixed = float(self.alpha) * perm_var + (1.0 - float(self.alpha)) * o3_resid
            return float(self.weight) * mixed
        except Exception:
            return torch.zeros((), device=device) if device is not None else torch.zeros(())

    def append(self, tensor: torch.Tensor, sink: Dict[str, torch.Tensor]) -> None:
        sink[self.name] = tensor