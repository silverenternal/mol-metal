"""Wf-309B — Frontier Math Integration Wrapper (4 standalone adapters composed).

================================================================
Background — composition layer for the 4 R18 frontier-math adapters
================================================================

Per TODO-234 §4 + Wf-305B honest-negative + Wf-306B verification,
the 4 frontier-math adapters ship as **standalone drop-in modules** at
the top level of :mod:`molmetal.adapters`:

* :mod:`molmetal.adapters.multi_ph_integration` — W2.3 ship
  (multi-parameter persistent homology features + metal-coordination
  classifier head).
* :mod:`molmetal.adapters.se3_lipman_integration` — Integration 1 ship
  (SE(3)-aware ODE solver wrapping the R17 ODESolver shim with the
  Lie-group velocity field from W2.1).
* :mod:`molmetal.adapters.se3_sb_integration` — W2.1 ship
  (SE(3) Schrödinger-Bridge velocity field + decoder head +
  ``integrate_with_cfm`` ODE wrapper).
* :mod:`molmetal.adapters.transport_relaxation_integration` — W2.2
  ship (soft endpoint OT penalty = ``λ · KL(p_T ‖ q_T(ε))`` for
  reward shaping).

R17 invariant (R17) forbids wiring into
:mod:`molmetal.adapters.flow_matching_lipman.__init__` — the R17
close-out is locked.  This module is therefore a **NEW pure
composition layer** that:

1. Imports the 4 standalone adapters.
2. Exposes a single :class:`FrontierMathIntegration` class whose
   factory methods return the canonical public objects from each
   adapter.
3. Stores a shared :class:`config` dict so callers can tune every
   adapter from one place (one ``config["multi_ph"]["n_bins"]`` etc.).

Design constraints (per Wf-309B brief)
--------------------------------------
1. **No modification of ``flow_matching_lipman/``** — this module is
   the *only* new file added under Wf-309B.
2. **CPU-only** — every adapter is CPU-only by design (matches the
   brief's pre-flight gate).
3. **Standalone** — no internal imports from
   :mod:`molmetal.molmetal_lam.*`.  Only :mod:`torch` +
   the 4 standalone adapter modules + :mod:`molmetal.math_skills`.
5. **Composition-only** — this layer does NOT redefine the adapters'
   semantics; it is a thin factory.

Module map
----------
* :class:`FrontierMathIntegration` — main composition class with
  factories for each adapter's public object:
  :meth:`get_multi_ph_features` → :class:`MultiPHPocketFeatures`
  :meth:`get_se3_lipman_solver` → :class:`SE3LipmanSolver`
  :meth:`get_se3_sb_decoder` → :class:`SE3SBDecoder`
  :meth:`get_transport_relaxation` → :class:`RelaxedEndpointTerm`
* :func:`default_config` — return the default config dict with the
  recommended hyper-parameters from each adapter's docstring.

Honest framing
--------------
* This is INTEGRATION wrapper work (NEW file), NOT modification of
  :mod:`flow_matching_lipman.__init__`.
* 4 adapters are STANDALONE; wrappers just compose them.
* 3 research-only modules
  (``bass_existence`` / ``sb_bass_martingale`` /
  ``monge_ampere_regularity``) are honestly research-only by design
  and are NOT wired in here.

References
----------
* TODO-234 §4 — frontier math wire-in plan.
* Wf-305B — honest-negative verification of the 4-stand-alone-adapter
  layout (R17 invariant holds: no ``flow_matching_lipman/`` changes).
* Wf-306B — verification pass on the standalone-adapter public APIs.
* Kim & Jung 2026, *Fourier-Accelerated Computation of Multi-Parameter
  Persistent Homology for Molecular Shape Analysis*, AAM **174**, 103023.
* Mahmood, Halder, Akhtar 2026, *Schrödinger Bridges on Compact
  Connected Lie Groups with Application to Molecular Generation*,
  arXiv:2603.14049.
* Jiang, Xu, Zhang 2026, *Schrödinger Bridge with Transport
  Relaxation*, arXiv:2602.08118.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import torch

# 4 STANDALONE frontier-math adapters.  All four live at the top level
# of :mod:`molmetal.adapters` per Wf-305B honest-negative verification.
from molmetal.adapters.multi_ph_integration import (
    MultiPHPocketFeatures,
    MultiPHMetalCoordHead,
)
from molmetal.adapters.se3_lipman_integration import (
    SE3LipmanSolver,
    SE3CFMComposite,
)
from molmetal.adapters.se3_sb_integration import (
    SE3SBDecoder,
    SE3SBVelocityField,
    integrate_with_cfm,
)
from molmetal.adapters.transport_relaxation_integration import (
    RelaxedEndpointTerm,
    LambdaRelaxSweep,
)

__all__ = [
    "FrontierMathIntegration",
    "default_config",
]


def default_config() -> Dict[str, Dict[str, Any]]:
    """Return the recommended hyper-parameters for every adapter.

    These defaults match the canonical values pinned in each
    adapter's docstring:

    * ``multi_ph`` → W2.3 smoke (max_radius=5.0, max_coord=4.0,
      n_bins=32, n_metals=12).
    * ``se3_sb`` → W2.1 default (eps=0.1, n_iters=50,
      use_sinkhorn=True).
    * ``se3_lipman`` → Integration 1 default (n_steps=50,
      method="midpoint", ortho_tol=1e-3).
    * ``transport_relaxation`` → W2.2 default (lambda_relax=1e-3,
      eps=0.1).

    Returns
    -------
    ``Dict[str, Dict[str, Any]]``
        Nested config dict keyed by adapter name.

    Notes
    -----
    The keys are deliberately flat strings (``"multi_ph"``,
    ``"se3_sb"``, ...) so they survive JSON round-trips (callers can
    load / save the config to disk without losing structure).
    """
    return {
        "multi_ph": {
            "max_radius": 5.0,
            "max_coord": 4.0,
            "n_bins": 32,
            "n_metals": 12,
            "kernel": "sliced_wasserstein",
            "sigma": 1.0,
            "n_projections": 8,
        },
        "se3_sb": {
            "eps": 0.1,
            "n_iters": 50,
            "use_sinkhorn": True,
            "seed": None,
        },
        "se3_lipman": {
            "n_steps": 50,
            "method": "midpoint",
            "ortho_tol": 1e-3,
        },
        "transport_relaxation": {
            "lambda_relax": 1e-3,
            "eps": 0.1,
        },
    }


@dataclass
class FrontierMathIntegration:
    """Composition layer for the 4 frontier-math adapters.

    The class is intentionally a thin factory: it stores a shared
    config dict and exposes one factory method per adapter that
    returns a freshly-constructed public object from the underlying
    standalone module.  No adapter semantics are redefined here.

    Parameters
    ----------
    config : dict, optional
        Nested config dict (same shape as :func:`default_config`).
        When ``None`` we fall back to :func:`default_config`.
        Callers may override individual keys without touching the
        others (the factory methods deep-copy the relevant sub-dict
        so mutations do not leak back into the stored config).

    Notes
    -----
    The dataclass form keeps the wrapper trivially inspectable for
    tests + dashboards (``fmi.config["multi_ph"]["n_bins"]`` etc.).
    We do not freeze the dataclass — the brief explicitly allows
    callers to mutate the config in place to retune the adapters
    between runs.
    """

    config: Dict[str, Dict[str, Any]] = field(default_factory=default_config)

    def __post_init__(self) -> None:
        # Validate the config shape — fail-fast on a missing key so
        # callers get a clear error rather than a ``KeyError`` deep
        # inside an adapter's ``__init__``.
        required_keys = {"multi_ph", "se3_sb", "se3_lipman", "transport_relaxation"}
        missing = required_keys - set(self.config.keys())
        if missing:
            raise ValueError(
                f"config missing required keys: {sorted(missing)}; "
                f"got keys {sorted(self.config.keys())}"
            )

    # ------------------------------------------------------------------
    # Multi-PH pocket features (W2.3)
    # ------------------------------------------------------------------
    def get_multi_ph_features(
        self,
        *,
        override: Optional[Dict[str, Any]] = None,
    ) -> MultiPHPocketFeatures:
        """Return a freshly-constructed :class:`MultiPHPocketFeatures`.

        Parameters
        ----------
        override : dict, optional
            Per-call overrides for the ``multi_ph`` sub-dict.  When
            supplied the keys are merged on top of the stored
            config (the stored config is not modified).
        """
        cfg = dict(self.config["multi_ph"])
        if override:
            cfg.update(override)
        return MultiPHPocketFeatures(
            max_radius=float(cfg["max_radius"]),
            max_coord=float(cfg["max_coord"]),
            n_bins=int(cfg["n_bins"]),
        )

    def get_multi_ph_metal_head(
        self,
        *,
        override: Optional[Dict[str, Any]] = None,
    ) -> MultiPHMetalCoordHead:
        """Return a freshly-constructed :class:`MultiPHMetalCoordHead`.

        Provided as a convenience — the multi-PH metal-coordination
        head is the natural consumer of the pocket features returned
        by :meth:`get_multi_ph_features`.
        """
        cfg = dict(self.config["multi_ph"])
        if override:
            cfg.update(override)
        return MultiPHMetalCoordHead(
            n_metals=int(cfg["n_metals"]),
            kernel=str(cfg["kernel"]),
            sigma=float(cfg["sigma"]),
            n_projections=int(cfg["n_projections"]),
            max_radius=float(cfg["max_radius"]),
            max_coord=float(cfg["max_coord"]),
            n_bins=int(cfg["n_bins"]),
        )

    # ------------------------------------------------------------------
    # SE(3) Schrödinger-Bridge (W2.1)
    # ------------------------------------------------------------------
    def get_se3_sb_decoder(
        self,
        *,
        override: Optional[Dict[str, Any]] = None,
    ) -> SE3SBDecoder:
        """Return a freshly-constructed :class:`SE3SBDecoder`.

        Parameters
        ----------
        override : dict, optional
            Per-call overrides for the ``se3_sb`` sub-dict.
        """
        cfg = dict(self.config["se3_sb"])
        if override:
            cfg.update(override)
        return SE3SBDecoder(
            eps=float(cfg["eps"]),
            n_iters=int(cfg["n_iters"]),
            use_sinkhorn=bool(cfg["use_sinkhorn"]),
            seed=cfg["seed"],
        )

    # ------------------------------------------------------------------
    # SE(3) Lipman solver (Integration 1)
    # ------------------------------------------------------------------
    def get_se3_lipman_solver(
        self,
        *,
        se3_adapter: Optional[SE3SBVelocityField] = None,
        override: Optional[Dict[str, Any]] = None,
    ) -> SE3LipmanSolver:
        """Return a freshly-constructed :class:`SE3LipmanSolver`.

        Parameters
        ----------
        se3_adapter : :class:`SE3SBVelocityField`, optional
            Optional SE(3) velocity field wrapper.  When ``None``
            the solver instantiates its default zero-velocity
            fallback (matches the W2.1 smoke contract).
        override : dict, optional
            Per-call overrides for the ``se3_lipman`` sub-dict.
        """
        cfg = dict(self.config["se3_lipman"])
        if override:
            cfg.update(override)
        return SE3LipmanSolver(
            se3_adapter=se3_adapter,
            n_steps=int(cfg["n_steps"]),
            method=str(cfg["method"]),
            ortho_tol=float(cfg["ortho_tol"]),
        )

    # ------------------------------------------------------------------
    # Transport relaxation (W2.2)
    # ------------------------------------------------------------------
    def get_transport_relaxation(
        self,
        *,
        override: Optional[Dict[str, Any]] = None,
    ) -> RelaxedEndpointTerm:
        """Return a freshly-constructed :class:`RelaxedEndpointTerm`.

        Parameters
        ----------
        override : dict, optional
            Per-call overrides for the ``transport_relaxation``
            sub-dict.
        """
        cfg = dict(self.config["transport_relaxation"])
        if override:
            cfg.update(override)
        return RelaxedEndpointTerm(
            lambda_relax=float(cfg["lambda_relax"]),
            eps=float(cfg["eps"]),
        )

    # ------------------------------------------------------------------
    # Diagnostics — composed adapter count
    # ------------------------------------------------------------------
    @property
    def n_adapters_composed(self) -> int:
        """Return the number of frontier-math adapters composed here.

        Always ``4`` — the canonical Wf-309B count.  Exposed as a
        property so test code + dashboards can audit the wrapper
        without poking at the internal config.
        """
        return 4

    def __repr__(self) -> str:
        return (
            f"FrontierMathIntegration("
            f"n_adapters_composed={self.n_adapters_composed}, "
            f"keys={sorted(self.config.keys())})"
        )


# ---------------------------------------------------------------------------
# Convenience module-level helpers — keep callers terse
# ---------------------------------------------------------------------------
def get_default_integration() -> FrontierMathIntegration:
    """Return a :class:`FrontierMathIntegration` with the default config.

    One-line convenience for callers that want the canonical
    composition without manually building the config dict.
    """
    return FrontierMathIntegration(config=default_config())
