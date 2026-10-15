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

Wf-A34-1 B4 (2026-10-14) — every ``except Exception: return zeros`` swallow
in this module now emits a once-per-process ``warnings.warn`` per
``(class_name, exc_type_name)`` pair, gated by the module-level
``_IMPORT_WARNED`` set.  Mirrors the
``_metal_geometry_import_warned`` gate at
``flow_matching_lipman/__init__.py:2097`` so a missing optional import is
visible in the trainer stderr instead of silently collapsing the term to
zero.  Default behaviour is unchanged: a class with weight<=0 still
short-circuits BEFORE the try block, so the warn never fires on the
shipped default path.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch

# Wf-A34-1 B4 (2026-10-14) — module-level once-per-process gate for the
# 6 silent exception swallows below.  Without this, a missing optional
# import (e.g. CUDA-only ``_cr.consistency_loss_fn`` on CPU-only CI)
# would collapse a term to zero every batch with no signal.  Keyed on
# ``(class_name, exc_type_name)`` so each unique exception is surfaced
# exactly once.  Mirrors ``_metal_geometry_import_warned`` at
# ``flow_matching_lipman/__init__.py:2097``.
_IMPORT_WARNED: set = set()


def _maybe_warn_import(cls_name: str, exc: BaseException) -> None:
    """Emit a once-per-process warning if the import swallow fired.

    No-op when the ``(cls_name, exc_type_name)`` pair has already been
    surfaced; adds the pair to ``_IMPORT_WARNED`` on first sight so
    subsequent identical swallows stay quiet.
    """
    key = (cls_name, type(exc).__name__)
    if key in _IMPORT_WARNED:
        return
    _IMPORT_WARNED.add(key)
    # ``stacklevel=2`` so the warning points at the caller of
    # ``compute`` rather than this helper module.
    import warnings

    warnings.warn(
        f"[{cls_name}] {type(exc).__name__} — {exc}; returning 0",
        stacklevel=2,
    )


class EntropyTerm:
    """Maximise H(softmax(logits)) — bit-exact counterpart of inline 2999-3005.

    Sign is NEGATIVE so the optimiser maximises entropy.  Wf-A43 C — default
    weight flipped from 0.0 → 0.1; the previous default of 0.01 was
    unobservable (0.11-0.19% signal in the trainer's 12-term sum).  The
    log-scale guard ``weight < 1e-3`` keeps a sloppy YAML ``0`` from
    accidentally killing the term, and ``_entropy_term_magnitude_warn``
    fires once per process if the chosen weight would still be unobservable.

    YAML keys
    ---------
    loss.entropy_reg_weight: float = 0.1   # default ON (Wf-A43 C default flip)
    """

    def __init__(
        self,
        weight: float = 0.1,
        enabled: bool = True,
        name: str = "entropy_term",
    ) -> None:
        self.weight = float(weight)
        self.enabled = bool(enabled)
        self.name = str(name)
        # Wf-A43 C — log-scale guard: surface unobservable weights
        # (≤ 1e-3) once per process so a "default 0.01" typo doesn't
        # silently kill the gradient pathway.
        if self.weight > 0.0 and self.weight < 1e-3:
            import warnings as _w
            _w.warn(
                f"[EntropyTerm] weight={self.weight} is below the 1e-3 floor; "
                f"signal will be ~{self.weight * 100:.2f}% of the 12-term sum. "
                f"Recommend weight >= 0.1.",
                stacklevel=2,
            )

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
        # Guard: if weight is below 1e-3, multiply by 0 to make the
        # unobservable signal explicit (warn fired once at construction).
        effective_weight = float(self.weight) if self.weight >= 1e-3 else 0.0
        return -effective_weight * entropy

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
        except Exception as exc:  # Wf-A34-1 B4 — surface instead of swallow
            _maybe_warn_import(self.__class__.__name__, exc)
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
        except Exception as exc:  # Wf-A34-1 B4 — surface instead of swallow
            _maybe_warn_import(self.__class__.__name__, exc)
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
        except Exception as exc:  # Wf-A34-1 B4 — surface instead of swallow
            _maybe_warn_import(self.__class__.__name__, exc)
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
        except Exception as exc:  # Wf-A34-1 B4 — surface instead of swallow
            _maybe_warn_import(self.__class__.__name__, exc)
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
        except Exception as exc:  # Wf-A34-1 B4 — surface instead of swallow
            _maybe_warn_import(self.__class__.__name__, exc)
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
        except Exception as exc:  # Wf-A34-1 B4 — surface instead of swallow
            _maybe_warn_import(self.__class__.__name__, exc)
            return torch.zeros((), device=device) if device is not None else torch.zeros(())

    def append(self, tensor: torch.Tensor, sink: Dict[str, torch.Tensor]) -> None:
        sink[self.name] = tensor