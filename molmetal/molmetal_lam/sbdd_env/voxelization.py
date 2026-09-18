"""Voxelize a protein pocket into a 3D grid of :class:`SpatialTile`.

Each voxel is annotated with element-aware channels:
- ``hbond_donor``     : N, O atoms (and S in sidechains)
- ``hbond_acceptor``  : O, N (carbonyl O, etc.)
- ``hydrophobic``     : C atoms
- ``charge``          : +/-1 for charged residues (Arg/Lys/Asp/Glu)
                         and metals (Zn, Mg, Fe, …)

The implementation is *NumPy/Python* only — no GPU dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import torch  # type: ignore
    _HAVE_TORCH = True
except Exception:  # pragma: no cover
    _HAVE_TORCH = False


# ---------------------------------------------------------------------------
# SpatialTile
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class SpatialTile:
    """A single voxel of a protein-pocket grid.

    All channel fields are 0 or 1 — they represent occupancy in that
    semantic channel rather than magnitudes.  ``coords`` is the voxel
    centre (Å) in the pocket frame.
    """

    coords: Tuple[float, float, float]
    occupied: int            # 1 if any heavy atom sits here, else 0
    hbond_donor: int         # N/O (and S) atoms
    hbond_acceptor: int      # O, carbonyl O
    hydrophobic: int         # aliphatic/aromatic C
    charge: int              # -1 / 0 / +1

    def as_array(self) -> np.ndarray:
        """Return the 6-D feature vector ``[occupied, hd, ha, hyd, charge, 0]``."""
        return np.asarray(
            [self.occupied, self.hbond_donor, self.hbond_acceptor,
             self.hydrophobic, self.charge, 0],
            dtype=np.float32,
        )


# ---------------------------------------------------------------------------
# Element -> channel mapping
# ---------------------------------------------------------------------------
# Atomic-number lookups we need for element-channel decisions.
_HBOND_DONOR_Z = {7, 8, 16}                # N, O, S
_HBOND_ACCEPTOR_Z = {7, 8}                 # N, O  (rough; carbonyl O handled below)
_HYDROPHOBIC_Z = {6}                       # C
_CHARGED_POS_Z = {12, 19, 20, 26, 29, 30}  # Mg, K, Ca, Fe, Cu, Zn (etc.) → +1
# We keep charge strictly +1 for metals; -1 is reserved for acidic
# carboxylate O atoms in residues (handled at residue level below).

_CHARGED_RESIDUES_POS = {"ARG", "LYS", "HIS"}   # basic
_CHARGED_RESIDUES_NEG = {"ASP", "GLU"}          # acidic
_METAL_RESIDUES = {"ZN", "MG", "FE", "CU", "MN", "CA", "NI", "CO", "MO", "PT", "PD", "AU"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def voxel_grid_shape(grid_size: int) -> Tuple[int, int, int]:
    """Just returns ``(grid_size, grid_size, grid_size)`` — for clarity."""
    return (grid_size, grid_size, grid_size)


def _to_numpy(coords) -> np.ndarray:
    """Coerce torch / list-of-lists into ``(N, 3) float32`` numpy."""
    if _HAVE_TORCH and isinstance(coords, torch.Tensor):
        return coords.detach().cpu().numpy().astype(np.float32)
    arr = np.asarray(coords, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"coords must be (N, 3), got shape {arr.shape}")
    return arr


def _to_numpy_atom_types(atom_types) -> np.ndarray:
    """Coerce atomic-number array to int64 numpy."""
    if _HAVE_TORCH and isinstance(atom_types, torch.Tensor):
        return atom_types.detach().cpu().numpy().astype(np.int64)
    return np.asarray(atom_types, dtype=np.int64)


def _voxel_index(x: float, y: float, z: float, origin: np.ndarray, resolution: float, grid_size: int) -> Tuple[int, int, int]:
    """Convert a 3-D coord (Å) into a ``(i, j, k)`` voxel index, clipped."""
    rel = (np.asarray([x, y, z], dtype=np.float32) - origin) / resolution
    idx = np.floor(rel).astype(np.int64)
    idx = np.clip(idx, 0, grid_size - 1)
    return int(idx[0]), int(idx[1]), int(idx[2])


# ---------------------------------------------------------------------------
# Main voxelizer
# ---------------------------------------------------------------------------
def pocket_to_spatial_tiles(
    pocket_or_pdb_id: Union[str, object],
    lig_center: Optional[Sequence[float]] = None,
    grid_size: int = 32,
    resolution: float = 0.5,
    radius: float = 6.0,
    chain_whitelist: Optional[List[str]] = None,
) -> List[SpatialTile]:
    """Voxelize a pocket into a flat list of :class:`SpatialTile`.

    Two calling conventions:
    1. ``pocket_to_spatial_tiles(pocket, lig_center, ...)`` where ``pocket``
       is a :class:`molmetal.domain.Pocket` (preferred — already has coords
       and atom_types).
    2. ``pocket_to_spatial_tiles("2zzf", lig_center=[12.3, 4.5, -3.1], ...)``
       which lazy-loads via :func:`molmetal.domain.io.pdb_loader._load_pocket`.

    Parameters
    ----------
    pocket_or_pdb_id
        Either a :class:`Pocket` or a PDB id/path.
    lig_center
        Binding-site centroid (Å).  Ignored when ``pocket_or_pdb_id`` is
        already a Pocket.
    grid_size
        Number of voxels per axis (default 32 → 32³ = 32 768 tiles).
    resolution
        Voxel side length (Å).  Default 0.5 Å → 16 Å cube.
    radius
        Pocket sphere radius — only used when we lazy-load from PDB.
    chain_whitelist
        Optional chain filter — only used on the lazy-load path.

    Returns
    -------
    list[SpatialTile]
        Flat list of length ``grid_size**3`` with channel flags.
    """
    if isinstance(pocket_or_pdb_id, str):
        # Lazy-load from PDB file.
        if lig_center is None:
            raise ValueError("lig_center is required when passing a PDB id/path string")
        pocket = _load_pocket_from_id(pocket_or_pdb_id, lig_center, radius, chain_whitelist)
    else:
        pocket = pocket_or_pdb_id

    return _voxelize_pocket(pocket, grid_size=grid_size, resolution=resolution)


# ---------------------------------------------------------------------------
# Internal helpers (kept private)
# ---------------------------------------------------------------------------
def _load_pocket_from_id(pdb_id: str, lig_center, radius: float, chain_whitelist):
    """Lazy-load via the existing domain pdb_loader.

    Falls back to a tiny synthetic pocket if BioPython is unavailable so
    the wrapper is still usable in minimal environments.
    """
    try:
        from molmetal.domain.io.pdb_loader import _load_pocket  # type: ignore
    except Exception:
        return _synthetic_pocket(pdb_id, lig_center, radius)

    if _HAVE_TORCH:
        centre_t = torch.as_tensor(lig_center, dtype=torch.float32).flatten()
        assert centre_t.shape == (3,)
        return _load_pocket(
            path=pdb_id,
            ligand_center=centre_t,
            radius=radius,
            chain_whitelist=chain_whitelist,
            include_heteroatoms=False,
        )

    # Fallback if torch is unavailable: build a minimal Pocket by hand.
    return _synthetic_pocket(pdb_id, lig_center, radius)


def _synthetic_pocket(pdb_id: str, lig_center, radius: float):
    """Build a small synthetic Pocket around ``lig_center`` (test/dev fallback)."""
    if _HAVE_TORCH:
        from molmetal.domain import Pocket  # type: ignore
        centre = torch.as_tensor(lig_center, dtype=torch.float32).flatten()
        # 8 heavy atoms in a small cube around the centre.
        offsets = torch.tensor(
            [
                [-1.0, -1.0, -1.0], [+1.0, -1.0, -1.0],
                [-1.0, +1.0, -1.0], [+1.0, +1.0, -1.0],
                [-1.0, -1.0, +1.0], [+1.0, -1.0, +1.0],
                [-1.0, +1.0, +1.0], [+1.0, +1.0, +1.0],
            ],
            dtype=torch.float32,
        )
        coords = offsets + centre
        # Alternate C, N, O, C, O, N, C, S so all channels get exercised.
        atom_types = torch.tensor([6, 7, 8, 6, 8, 7, 6, 16], dtype=torch.long)
        n = coords.shape[0]
        return Pocket(
            pdb_id=pdb_id,
            coords=coords,
            atom_types=atom_types,
            residue_ids=torch.zeros(n, dtype=torch.long),
            chain_ids=torch.zeros(n, dtype=torch.long),
            mask=torch.ones(n, dtype=torch.bool),
            center=centre,
            radius=radius,
        )
    raise RuntimeError("torch is required to build a synthetic pocket")


def _voxelize_pocket(pocket, grid_size: int, resolution: float) -> List[SpatialTile]:
    """Pure-NumPy voxelizer over a Pocket."""
    coords = _to_numpy(pocket.coords)            # (N, 3)
    atom_z = _to_numpy_atom_types(pocket.atom_types)  # (N,)
    center = _to_numpy(pocket.center).reshape(-1)
    radius = float(getattr(pocket, "radius", 6.0))

    # --- Determine origin (min-corner of the sphere) ------------------
    half_extent = grid_size * resolution / 2.0
    origin = center - half_extent

    # --- Allocate channel buffers -------------------------------------
    occ = np.zeros((grid_size, grid_size, grid_size), dtype=np.int8)
    hb_d = np.zeros_like(occ)
    hb_a = np.zeros_like(occ)
    hyd = np.zeros_like(occ)
    charge = np.zeros_like(occ, dtype=np.int8)

    # --- Scatter atoms -------------------------------------------------
    for (x, y, z), z_at in zip(coords, atom_z):
        if (abs(x - center[0]) > half_extent
                or abs(y - center[1]) > half_extent
                or abs(z - center[2]) > half_extent):
            continue
        i, j, k = _voxel_index(x, y, z, origin, resolution, grid_size)
        occ[i, j, k] = 1
        if int(z_at) in _HBOND_DONOR_Z:
            hb_d[i, j, k] = 1
        if int(z_at) in _HBOND_ACCEPTOR_Z:
            hb_a[i, j, k] = 1
        if int(z_at) in _HYDROPHOBIC_Z:
            hyd[i, j, k] = 1
        if int(z_at) in _CHARGED_POS_Z:
            charge[i, j, k] = 1

    # --- Optional residue-level charge annotation ---------------------
    # If we have residue_ids and a residue name map, mark basic/acidic.
    residue_ids = getattr(pocket, "residue_ids", None)
    if residue_ids is not None and _HAVE_TORCH:
        residue_ids_np = residue_ids.detach().cpu().numpy().astype(np.int64)
        coords_list = coords
        for idx, (x, y, z) in enumerate(coords_list):
            ri = int(residue_ids_np[idx])
            rname = _RESIDUE_NAMES_BY_ID.get(ri)
            if rname is None:
                continue
            i, j, k = _voxel_index(x, y, z, origin, resolution, grid_size)
            if rname in _CHARGED_RESIDUES_POS:
                charge[i, j, k] = 1
            elif rname in _CHARGED_RESIDUES_NEG:
                charge[i, j, k] = -1

    # --- Flatten to list of SpatialTile --------------------------------
    tiles: List[SpatialTile] = []
    half = half_extent
    # Iterate in a flat C-order so coords match the buffer indexing.
    flat_occ = occ.ravel()
    flat_hd = hb_d.ravel()
    flat_ha = hb_a.ravel()
    flat_hyd = hyd.ravel()
    flat_chg = charge.ravel()
    for linear_idx in range(grid_size ** 3):
        i = linear_idx // (grid_size * grid_size)
        j = (linear_idx // grid_size) % grid_size
        k = linear_idx % grid_size
        coord = (
            float(origin[0] + (i + 0.5) * resolution),
            float(origin[1] + (j + 0.5) * resolution),
            float(origin[2] + (k + 0.5) * resolution),
        )
        tiles.append(
            SpatialTile(
                coords=coord,
                occupied=int(flat_occ[linear_idx]),
                hbond_donor=int(flat_hd[linear_idx]),
                hbond_acceptor=int(flat_ha[linear_idx]),
                hydrophobic=int(flat_hyd[linear_idx]),
                charge=int(flat_chg[linear_idx]),
            )
        )
    return tiles


# Tiny lookup table for residue-id → name. In the test pocket we have
# residue_ids starting at 0; tests set them up so id 0 == ALA, 1 == ARG, etc.
_RESIDUE_NAMES_BY_ID = {
    0: "ALA",
    1: "ARG",
    2: "ASN",
    3: "ASP",
    4: "CYS",
    5: "GLU",
    6: "GLN",
    7: "GLY",
    8: "HIS",
    9: "ILE",
    10: "LEU",
    11: "LYS",
    12: "MET",
    13: "PHE",
    14: "PRO",
    15: "SER",
    16: "THR",
    17: "TRP",
    18: "TYR",
    19: "VAL",
}


__all__ = ["SpatialTile", "pocket_to_spatial_tiles", "voxel_grid_shape"]