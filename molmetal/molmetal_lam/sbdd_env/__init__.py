"""Structure-Based Drug Design (SBDD) environment.

Provides:
- :class:`REINVENT4Scorer`: a lightweight multi-property scorer with weights
  matching REINVENT4 defaults (qed, binding, sas, novelty).
- :func:`pocket_to_spatial_tiles`: voxelize a pocket PDB into a 3D grid of
  :class:`SpatialTile` with element-aware channel flags (hbond_donor,
  hbond_acceptor, hydrophobic, charge).

The scorer is intentionally *fallback* friendly — when REINVENT4's
heavy scorer plugins (QSAR, docking) are unavailable the wrapper degrades
to RDKit-based proxies so that downstream code can still run.
"""

from molmetal_lam.sbdd_env.reinvent_wrapper import (  # noqa: F401
    DEFAULT_SCORER_WEIGHTS,
    REINVENT4Scorer,
    batch_score,
)
from molmetal_lam.sbdd_env.voxelization import (  # noqa: F401
    SpatialTile,
    pocket_to_spatial_tiles,
    voxel_grid_shape,
)

__all__ = [
    "DEFAULT_SCORER_WEIGHTS",
    "REINVENT4Scorer",
    "SpatialTile",
    "batch_score",
    "pocket_to_spatial_tiles",
    "voxel_grid_shape",
]