"""Per-pocket Vina configuration adapter (Wf-248-C / TODO-125).

================================================================
What this replaces
================================================================
Previously the project relied on a single global Vina configuration
(``DockingConfig(n_poses=5, exhaustiveness=8)``) across ALL pockets.
TODO-125 §0 documented that per-pocket characteristics (binding-site
volume, hydrophobic depth, ligand chemistry) benefit from per-pocket
DRS (docking-resolution) tuning:

* Small pockets (e.g. 1h36, MMP13 Zn-coordination) need a higher
  ``exhaustiveness`` + more ``n_poses`` because the binding pose
  distribution is narrow and needs finer sampling.
* Large pockets (e.g. 4rlu, GST) tolerate the default values — the
  pose distribution is wider so 8/5 captures the mode.

This module exposes a tiny adapter that maps a ``pocket_id`` (string)
to a :class:`PocketVinaConfig` (exhaustiveness + n_poses + optional
box center + box size) and converts it into the canonical
:class:`DockingConfig` consumed by :class:`VinaDockingAdapter`.

Public API
----------
* :class:`PocketVinaConfig`  -- per-pocket dataclass (exhaustiveness,
  n_poses, optional center/size).
* :class:`PerPocketVinaConfig`  -- the adapter: maps pocket_id to
  PocketVinaConfig + ``to_docking_config(pocket_id, base_config)``.
* :func:`default_per_pocket_config()` -- module-level default.

Honest framing
--------------
Per-pocket values are heuristic — small pockets get more sampling,
large pockets get the baseline. They are NOT calibrated against any
external benchmark. When the user passes an explicit ``overrides``
dict those win; otherwise the per-pocket default map is used; if the
pocket_id is unknown, the global fallback
(``exhaustiveness=8, n_poses=5``) is used.

This module is library-first: it does NOT import the (heavy) Vina
Python binding or meeko at module import — only the project's
lightweight ``molmetal.ports.DockingConfig`` dataclass.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Tuple, Union

from molmetal.ports import DockingConfig


# -----------------------------------------------------------------
# Per-pocket dataclass
# -----------------------------------------------------------------
@dataclass(frozen=True)
class PocketVinaConfig:
    """Per-pocket Vina knobs.

    Attributes
    ----------
    exhaustiveness : int
        AutoDock Vina ``--exhaustiveness``. Higher → more thorough
        pose search at the cost of CPU time. Default map below
        uses 16 for small pockets (1h36) and 8 for large (4rlu).
    n_poses : int
        Number of docked poses to keep. Default map uses 10 for
        small pockets and 5 for large.
    center : Optional[Tuple[float, float, float]]
        Optional explicit box centre (Å). ``None`` means "use the
        pocket's geometric centre from the Pocket dataclass".
    size : Optional[Tuple[float, float, float]]
        Optional explicit box size (Å). ``None`` means "use the
        pocket radius + adapter padding".
    label : str
        Human-readable label (e.g. "small-tight", "large-loose")
        used for telemetry + reporting.
    """

    exhaustiveness: int = 8
    n_poses: int = 5
    center: Optional[Tuple[float, float, float]] = None
    size: Optional[Tuple[float, float, float]] = None
    label: str = "default"

    def __post_init__(self) -> None:
        if int(self.exhaustiveness) < 1:
            raise ValueError(
                f"PocketVinaConfig.exhaustiveness must be >= 1, "
                f"got {self.exhaustiveness!r}"
            )
        if int(self.n_poses) < 1:
            raise ValueError(
                f"PocketVinaConfig.n_poses must be >= 1, "
                f"got {self.n_poses!r}"
            )


# -----------------------------------------------------------------
# Default per-pocket map (TODO-125 + Wf-248-C spec)
# -----------------------------------------------------------------
# Per-pocket defaults. The task description (Wf-248-C) fixes
# 1h36 = small + tight + many poses + high exhaustiveness, and
# 4rlu = large + loose + few poses + low exhaustiveness. The other
# three pockets fall on a smooth interpolation between the two
# extremes based on pocket radius (TODO-125 §0 spread).
_DEFAULT_POCKET_MAP: Dict[str, PocketVinaConfig] = {
    # Small tight pocket — MMP13 Zn-coordination (TODO-125 §0)
    # "well-suited to small organic" but pose distribution is narrow,
    # so we sample harder.
    "1h36": PocketVinaConfig(
        exhaustiveness=16,
        n_poses=10,
        label="small-tight",
    ),
    # Mid-size pocket — DHFR folate site.
    "3eml": PocketVinaConfig(
        exhaustiveness=12,
        n_poses=8,
        label="mid-tight",
    ),
    # Mid-size pocket — β2 adrenergic receptor.
    "3ny8": PocketVinaConfig(
        exhaustiveness=10,
        n_poses=6,
        label="mid-loose",
    ),
    # Mid-size pocket — Carbonic anhydrase II.
    "4aaw": PocketVinaConfig(
        exhaustiveness=10,
        n_poses=6,
        label="mid-loose",
    ),
    # Large loose pocket — GST (TODO-125 §0).
    "4rlu": PocketVinaConfig(
        exhaustiveness=8,
        n_poses=5,
        label="large-loose",
    ),
}


# Global fallback used when pocket_id is unknown.
_GLOBAL_FALLBACK: PocketVinaConfig = PocketVinaConfig(
    exhaustiveness=8,
    n_poses=5,
    label="global-fallback",
)


# -----------------------------------------------------------------
# Adapter class
# -----------------------------------------------------------------
class PerPocketVinaConfig:
    """Map ``pocket_id`` → :class:`PocketVinaConfig`.

    Parameters
    ----------
    overrides : Optional[Mapping[str, PocketVinaConfig]]
        Per-pocket overrides. Keys not present fall back to the
        module-level ``_DEFAULT_POCKET_MAP``; pocket_ids not present
        in either map fall back to the global baseline
        (``exhaustiveness=8, n_poses=5``).
    """

    DEFAULT_POCKET_MAP: Dict[str, PocketVinaConfig] = _DEFAULT_POCKET_MAP
    GLOBAL_FALLBACK: PocketVinaConfig = _GLOBAL_FALLBACK

    def __init__(
        self,
        overrides: Optional[Dict[str, PocketVinaConfig]] = None,
        pocket_map: Optional[Dict[str, PocketVinaConfig]] = None,
    ) -> None:
        # User-supplied overrides win; if both ``overrides`` and
        # ``pocket_map`` are given, ``pocket_map`` replaces the
        # entire default map (used by tests).
        self._pocket_map: Dict[str, PocketVinaConfig] = dict(
            self.DEFAULT_POCKET_MAP
        )
        if pocket_map is not None:
            self._pocket_map = dict(pocket_map)
        if overrides:
            for k, v in overrides.items():
                self._pocket_map[str(k)] = v

    # ---------------------------------------------------------- API
    def get(self, pocket_id: str) -> PocketVinaConfig:
        """Return the :class:`PocketVinaConfig` for ``pocket_id``.

        Lookup order: user overrides → module default map → global
        fallback.
        """
        if pocket_id in self._pocket_map:
            return self._pocket_map[pocket_id]
        return self.GLOBAL_FALLBACK

    def exhaustiveness_for(self, pocket_id: str) -> int:
        return int(self.get(pocket_id).exhaustiveness)

    def n_poses_for(self, pocket_id: str) -> int:
        return int(self.get(pocket_id).n_poses)

    def to_docking_config(
        self,
        pocket_id: str,
        base_config: Optional[DockingConfig] = None,
    ) -> DockingConfig:
        """Return a :class:`DockingConfig` for ``pocket_id``.

        Carries through ``use_confidence`` and ``seed`` from
        ``base_config`` (when supplied) so seed-based reproducibility
        is preserved across the per-pocket switch.
        """
        cfg = self.get(pocket_id)
        if base_config is None:
            return DockingConfig(
                n_poses=int(cfg.n_poses),
                exhaustiveness=int(cfg.exhaustiveness),
            )
        return DockingConfig(
            n_poses=int(cfg.n_poses),
            exhaustiveness=int(cfg.exhaustiveness),
            use_confidence=bool(base_config.use_confidence),
            seed=int(base_config.seed),
        )

    def supported_pockets(self) -> Tuple[str, ...]:
        """Return the pocket_ids covered by the default map."""
        return tuple(sorted(self.DEFAULT_POCKET_MAP.keys()))

    def as_dict(self) -> Dict[str, Dict[str, object]]:
        """Return a JSON-friendly summary (used in reports)."""
        out: Dict[str, Dict[str, object]] = {}
        for pid in sorted(self._pocket_map.keys()):
            c = self._pocket_map[pid]
            out[pid] = {
                "exhaustiveness": int(c.exhaustiveness),
                "n_poses": int(c.n_poses),
                "center": list(c.center) if c.center else None,
                "size": list(c.size) if c.size else None,
                "label": str(c.label),
            }
        return out


# -----------------------------------------------------------------
# Module-level helpers
# -----------------------------------------------------------------
def default_per_pocket_config() -> PerPocketVinaConfig:
    """Return a fresh :class:`PerPocketVinaConfig` with default settings."""
    return PerPocketVinaConfig()


__all__ = [
    "PocketVinaConfig",
    "PerPocketVinaConfig",
    "default_per_pocket_config",
]