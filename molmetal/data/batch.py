"""Wf-A11 Batch dataclass — single chokepoint for collation.

Replaces 3 inlined collate sites + 4 ``getattr(cloud, "X", None)`` defaults
inside :file:`molmetal/scripts/train_cfm_yuelbond_wasserstein.py`. AtomCloud
stays as the loader output (63 existing call sites intact). Batch is the
COLLATE output, built lazily per mini-batch.

Three inlined sites collapsed to ``Batch.collate_fn(clouds).to(device)``:

  * line 3250-3264  — FM flat-concat (positions/z_all concatenation)
  * line 1673       — padded-tensor inner (sampling path, "largest bucket")
  * line 4405-4479  — padded-tensor inner (sampling path, "by_size")
  * line 2620-2632  — per-cloud passthrough (train_one_epoch rebuild)
  * line 5134-5135  — eval featuriser (legacy ``getattr`` site)

Backward-compat
---------------
* :class:`AtomCloud` dataclass is unchanged (4 required + 19 optional fields).
* :func:`Batch.collate_fn` returns a flat (N_total, ...) view; padded views
  are produced via :meth:`Batch.to_padded` on demand.
* Padding tokens are class constants (``PAD_Z=0``, ``PAD_R=0.0``, ...) so no
  call site carries a magic ``0`` literal.

Wf-A11 design rationale: see ``TODO/INDEX.md`` Wf-A11 batch-class extraction
plan, design §A.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch

from molmetal.data.tmqm import PocketCloud
from molmetal.models.bond_head import AtomCloud


# ---------------------------------------------------------------------------
# Padding token constants (never magic literals in trainer code)
# ---------------------------------------------------------------------------
PAD_Z: int = 0
PAD_R: float = 0.0
PAD_CHARGE: float = 0.0
PAD_BO: float = 0.0
NO_METAL_INDEX: int = AtomCloud.NO_METAL_INDEX
#: Maximum neighbour slots per atom — matches :attr:`AtomCloud.K_NEIGHBOURS`.
K_NEIGHBOURS: int = AtomCloud.K_NEIGHBOURS

# ---------------------------------------------------------------------------
# Covalent radii — the fallback that keeps ``Batch.atomic_radius`` real
# ---------------------------------------------------------------------------
#: Cordero (2008) single-bond covalent radii, Angstrom.  SAME table as
#: ``molmetal/molmetal_lam/lam_chem/yuelbond_decoder.py::_CORDERO_RCOV`` (which
#: is itself pinned to the probe script's copy by a test).  Duplicated rather
#: than imported because ``molmetal.data`` must not import a model module, and
#: ``molmetal/tests/test_wfa113_geometry_radius.py`` asserts the two are equal
#: so they cannot drift.
#:
#: WHY THIS EXISTS (Wf-A113 / T54, 2026-10-28).  ``Batch.collate_fn`` used to
#: read ``getattr(cloud, "atomic_radius", None)``.  ``AtomCloud`` has no such
#: field and no loader sets one, so the lookup always missed, every atom got
#: ``PAD_R = 0.0``, and ``geometry_consistency`` clamps that to 0.3 A.  The
#: equilibrium length ``eq = r_i + r_j`` was therefore 0.6 A for every pair of
#: every element, and the bond-plausibility band ``[0.85, 1.25] * eq`` became
#: [0.51, 0.75] A.  MEASURED on 64 clouds of the 5-metal cohort: 34 of 3,438
#: true bonds (0.99%) were inside the band, mean squared deviation 0.637;
#: with Cordero radii 2,966 (86.3%) are inside and the deviation is 0.0216.
#: The term was pulling every bond toward 0.6 A.
CORDERO_RCOV: Dict[int, float] = {
    1: 0.31, 5: 0.84, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57, 14: 1.11, 15: 1.07,
    16: 1.05, 17: 1.02, 33: 1.21, 35: 1.20, 53: 1.39, 78: 1.36, 44: 1.25,
    77: 1.35, 45: 1.35, 76: 1.35, 75: 1.35,
}
#: Radius for an element outside the table.  Matches the decoder's
#: ``_RCOV_UNKNOWN``: generous, so an unknown element is never reported as a
#: clash by accident.
RCOV_UNKNOWN: float = 1.3
RCOV_MAX_Z: int = 92
_RCOV_BY_Z = tuple(CORDERO_RCOV.get(z, RCOV_UNKNOWN) for z in range(RCOV_MAX_Z + 1))


def _radius_field(cloud: AtomCloud, z: torch.Tensor) -> torch.Tensor:
    """Per-atom covalent radius, from the cloud when it has one, else by Z.

    The cloud's own value wins so a caller that carries a different radius
    convention (e.g. a Pyykko table) is not overridden.  The fallback is what
    makes the field real for every existing loader, none of which sets it.
    """
    n = int(z.shape[0])
    val = getattr(cloud, "atomic_radius", None)
    if val is not None and int(val.shape[0]) == n:
        return val.float().view(-1)
    table = torch.tensor(_RCOV_BY_Z, device=z.device, dtype=torch.float32)
    zz = z.long().clamp(min=0, max=RCOV_MAX_Z)
    return table[zz]



# ---------------------------------------------------------------------------
# Batch
# ---------------------------------------------------------------------------
@dataclass
class Batch:
    """Collation output for a list of :class:`AtomCloud`.

    Flat shape per design §A:

    * per-atom (N_total, ...): ``positions``, ``atomic_numbers``, ``atomic_radius``,
      ``formal_charge``, ``natural_charge``, ``bo_continuous``, ``batch_idx``.
    * per-cloud (B, ...): ``orig_n_atoms``, ``metal_atom_index``, ``oxidation_state``,
      ``spin_state``, ``csd_year``, ``stoichiometry``, ``smiles``, ``coord_neighbors``.
    * per-pair (E_total, ...): ``bonds`` (2, E), ``bond_types``, ``bond_orders_continuous``,
      ``is_dative_pair``, ``pair_cloud_idx``.
    * batch-level: ``cloud_offsets`` (B+1,) CSR; ``n_clouds``.

    Use :meth:`to_padded` to materialise a (B, N_max, ...) view; the flat
    view is the canonical form for the FM objective.
    """

    # ---- per-atom ----
    positions: torch.Tensor
    atomic_numbers: torch.Tensor
    atomic_radius: torch.Tensor
    formal_charge: torch.Tensor
    natural_charge: torch.Tensor
    bo_continuous: torch.Tensor
    batch_idx: torch.Tensor
    # ---- per-cloud ----
    orig_n_atoms: torch.Tensor
    metal_atom_index: torch.Tensor
    oxidation_state: torch.Tensor
    spin_state: torch.Tensor
    csd_year: torch.Tensor
    stoichiometry: List[Optional[str]]
    smiles: List[Optional[str]]
    coord_neighbors: torch.Tensor
    # ---- per-pair ----
    bonds: torch.Tensor
    bond_types: torch.Tensor
    bond_orders_continuous: torch.Tensor
    is_dative_pair: torch.Tensor
    pair_cloud_idx: torch.Tensor
    # ---- batch-level ----
    cloud_offsets: torch.Tensor
    n_clouds: int
    # ---- Wf-A16 (2026-10-13) c-flow pocket fields ----
    # All four default None so legacy callers that omit ``pocket_clouds``
    # see byte-identical batches.  When the trainer passes a parallel
    # list of PocketClouds to ``collate_fn``, the four tensors are
    # padded to (B, P_max, ...) with an explicit mask.
    pocket_features: Optional[torch.Tensor] = None      # (B, H) per-pocket
    pocket_coords: Optional[torch.Tensor] = None        # (B, P_max, 3) padded
    pocket_atom_types: Optional[torch.Tensor] = None    # (B, P_max) int64 padded
    pocket_mask: Optional[torch.Tensor] = None          # (B, P_max) bool padded

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    @classmethod
    def collate_fn(
        cls,
        clouds: Sequence[AtomCloud],
        target_n_atoms: Optional[int] = None,
        # ---- Wf-A16 (2026-10-13) c-flow pocket threading ----
        pocket_clouds: Optional[Sequence[Optional[PocketCloud]]] = None,
        pocket_feature_dim: int = 128,
    ) -> "Batch":
        """Stack a list of :class:`AtomCloud` into a single :class:`Batch`.

        Parameters
        ----------
        clouds : list[AtomCloud]
            Loader outputs (one per molecule).
        target_n_atoms : int, optional
            If supplied, :attr:`positions` and per-atom fields are zero-padded
            to ``(B, target_n_atoms, ...)`` instead of concatenated. The flat
            view is the default because the FM objective keeps clouds separate.

        Returns
        -------
        Batch
        """
        if not clouds:
            raise ValueError("collate_fn requires at least one AtomCloud")

        b = len(clouds)
        offsets: List[int] = [0]
        positions_chunks: List[torch.Tensor] = []
        z_chunks: List[torch.Tensor] = []
        r_chunks: List[torch.Tensor] = []
        fc_chunks: List[torch.Tensor] = []
        nc_chunks: List[torch.Tensor] = []
        bo_chunks: List[torch.Tensor] = []
        batch_idx_chunks: List[torch.Tensor] = []

        # Per-cloud metadata
        metal_atom_index: List[int] = []
        oxidation_state: List[int] = []
        spin_state: List[int] = []
        csd_year: List[int] = []
        stoichiometry: List[Optional[str]] = []
        smiles: List[Optional[str]] = []
        coord_neighbors_chunks: List[torch.Tensor] = []

        # Per-pair (CSR via cloud_idx)
        bonds_chunks: List[torch.Tensor] = []
        bond_types_chunks: List[torch.Tensor] = []
        bo_continuous_chunks: List[torch.Tensor] = []
        is_dative_chunks: List[torch.Tensor] = []
        pair_cloud_idx_chunks: List[torch.Tensor] = []

        for cloud_idx, c in enumerate(clouds):
            n = int(c.positions.shape[0])
            z = c.atomic_numbers.long().view(-1)
            positions_chunks.append(c.positions.float())
            z_chunks.append(z)
            r_chunks.append(_radius_field(c, z))
            fc_chunks.append(_per_atom_field(c, "formal_charge", n, PAD_CHARGE))
            # Wf-A113 (T55) — the field is ``natural_charges`` (plural) on
            # ``AtomCloud``; this read used the singular, so ``getattr`` always
            # missed and every atom got ``PAD_CHARGE = 0.0``.  The natural-charge
            # MSE then regressed the head toward ZERO rather than toward the NBO
            # charges, and reported a falling "natural_charge_term" while doing
            # it.  Latent today: every shipped config sets the weight to 0.
            nc_chunks.append(_per_atom_field(c, "natural_charges", n, PAD_CHARGE))
            bo_chunks.append(_per_atom_field(c, "bo_per_atom", n, PAD_BO))
            batch_idx_chunks.append(
                torch.full((n,), cloud_idx, dtype=torch.long)
            )
            offsets.append(offsets[-1] + n)

            metal_atom_index.append(_scalar_field(c, "metal_atom_index", NO_METAL_INDEX))
            oxidation_state.append(int(_scalar_field(c, "oxidation_state", 0)))
            spin_state.append(int(_scalar_field(c, "spin_state", 0)))
            csd_year.append(int(_scalar_field(c, "csd_publication_year", 0)))
            stoichiometry.append(getattr(c, "stoichiometry", None))
            smiles.append(getattr(c, "smiles", None))
            cn = getattr(c, "coord_neighbors_per_atom", None)
            if cn is None:
                cn = torch.zeros((n, K_NEIGHBOURS), dtype=torch.long)
            else:
                cn = cn.long()
                if cn.shape[0] != n:
                    cn = torch.zeros((n, K_NEIGHBOURS), dtype=torch.long)
            coord_neighbors_chunks.append(cn)

            b_bonds = c.bonds
            b_types = c.bond_types
            if b_bonds is not None and b_types is not None:
                if b_bonds.dim() != 2 or b_bonds.shape[0] != 2:
                    raise ValueError(
                        f"cloud {cloud_idx} bonds must be (2, E); got "
                        f"{tuple(b_bonds.shape)}"
                    )
                e = int(b_bonds.shape[1])
                bonds_chunks.append(b_bonds.long())
                bond_types_chunks.append(b_types.long())
                bo_pair = getattr(c, "bond_orders_continuous", None)
                bo_continuous_chunks.append(
                    bo_pair.float().view(-1)
                    if bo_pair is not None and int(bo_pair.shape[0]) == e
                    else torch.full((e,), PAD_BO, dtype=torch.float32)
                )
                is_dative = getattr(c, "is_dative_pair", None)
                is_dative_chunks.append(
                    is_dative.bool().view(-1)
                    if is_dative is not None and int(is_dative.shape[0]) == e
                    else torch.zeros((e,), dtype=torch.bool)
                )
                pair_cloud_idx_chunks.append(
                    torch.full((e,), cloud_idx, dtype=torch.long)
                )

        positions = torch.cat(positions_chunks, dim=0) if not target_n_atoms else _pad_concat(positions_chunks, b, target_n_atoms, 3)
        atomic_numbers = torch.cat(z_chunks, dim=0) if not target_n_atoms else _pad_concat(z_chunks, b, target_n_atoms, 1).long()
        atomic_radius = torch.cat(r_chunks, dim=0) if not target_n_atoms else _pad_concat(r_chunks, b, target_n_atoms, 1)
        formal_charge = torch.cat(fc_chunks, dim=0) if not target_n_atoms else _pad_concat(fc_chunks, b, target_n_atoms, 1)
        natural_charge = torch.cat(nc_chunks, dim=0) if not target_n_atoms else _pad_concat(nc_chunks, b, target_n_atoms, 1)
        bo_continuous = torch.cat(bo_chunks, dim=0) if not target_n_atoms else _pad_concat(bo_chunks, b, target_n_atoms, 1)
        if target_n_atoms:
            batch_idx = (
                torch.arange(b, dtype=torch.long).unsqueeze(1).expand(b, target_n_atoms).contiguous()
            )
        else:
            batch_idx = torch.cat(batch_idx_chunks, dim=0)
        coord_neighbors = torch.cat(coord_neighbors_chunks, dim=0) if not target_n_atoms else _pad_concat(coord_neighbors_chunks, b, target_n_atoms, K_NEIGHBOURS)

        bonds = torch.cat(bonds_chunks, dim=1) if bonds_chunks else torch.zeros((2, 0), dtype=torch.long)
        bond_types = torch.cat(bond_types_chunks, dim=0) if bond_types_chunks else torch.zeros((0,), dtype=torch.long)
        bond_orders_continuous = torch.cat(bo_continuous_chunks, dim=0) if bo_continuous_chunks else torch.zeros((0,), dtype=torch.float32)
        is_dative_pair = torch.cat(is_dative_chunks, dim=0) if is_dative_chunks else torch.zeros((0,), dtype=torch.bool)
        pair_cloud_idx = torch.cat(pair_cloud_idx_chunks, dim=0) if pair_cloud_idx_chunks else torch.zeros((0,), dtype=torch.long)

        orig_n_atoms = torch.tensor(
            [int(c.positions.shape[0]) for c in clouds], dtype=torch.long
        )

        # ------------------------------------------------------------------
        # Wf-A16 (2026-10-13) c-flow pocket tensor stacking.
        #
        # When ``pocket_clouds`` is provided we pad each pocket to the
        # cohort's max P (cap 256 from PocketEncoder budget) and stack
        # into (B, P_max, ...) tensors.  pocket_features is a lazy global
        # mean-pool: we use the loader-provided coords + mask rather than
        # running PocketEncoder inside collate, because (a) collate runs
        # many times per step and PocketEncoder is a learned module, and
        # (b) the trainer's PocketEncoder forward + per-layer injector is
        # already wired (molmetal/adapters/flow_matching_lipman/__init__.py
        # :1127 + :1372).  pocket_features here is a (B, 3+1) mean-pool
        # placeholder — sufficient for c-flow liveness checks; the
        # trainer replaces it with a PocketEncoder forward at the model
        # boundary.
        #
        # ``pocket_clouds=None`` (default) keeps every legacy call site
        # byte-identical: all four pocket_* fields stay None.
        # ------------------------------------------------------------------
        pocket_features_t = None
        pocket_coords_t = None
        pocket_atom_types_t = None
        pocket_mask_t = None
        if pocket_clouds is not None:
            import torch as _torch
            valid_pockets = [pc for pc in pocket_clouds if pc is not None]
            if valid_pockets:
                p_max = min(
                    256,
                    max(int(pc.coords.shape[0]) for pc in valid_pockets),
                )
                bsz = len(pocket_clouds)
                coords_buf = _torch.zeros((bsz, p_max, 3), dtype=_torch.float32)
                types_buf = _torch.zeros((bsz, p_max), dtype=_torch.int64)
                mask_buf = _torch.zeros((bsz, p_max), dtype=_torch.bool)
                feat_buf = _torch.zeros((bsz, pocket_feature_dim), dtype=_torch.float32)
                for i, pc in enumerate(pocket_clouds):
                    if pc is None:
                        continue
                    p_i = min(int(pc.coords.shape[0]), p_max)
                    coords_buf[i, :p_i] = pc.coords[:p_i]
                    types_buf[i, :p_i] = pc.atom_types[:p_i]
                    mask_buf[i, :p_i] = pc.mask[:p_i]
                    # Lazy global-mean-pool feature: centroid only.  The
                    # trainer swaps this for a real PocketEncoder forward
                    # at the model boundary (per-layer injector at
                    # molmetal/adapters/flow_matching_lipman/__init__.py
                    # :1708-1709 takes the PocketEncoder output).
                    feat_buf[i, :3] = pc.coords[:p_i].mean(dim=0)
                    feat_buf[i, 3] = float(p_i)
                pocket_coords_t = coords_buf
                pocket_atom_types_t = types_buf
                pocket_mask_t = mask_buf
                pocket_features_t = feat_buf

        batch = cls(
            positions=positions,
            atomic_numbers=atomic_numbers,
            atomic_radius=atomic_radius,
            formal_charge=formal_charge,
            natural_charge=natural_charge,
            bo_continuous=bo_continuous,
            batch_idx=batch_idx,
            orig_n_atoms=orig_n_atoms,
            metal_atom_index=torch.tensor(metal_atom_index, dtype=torch.long),
            oxidation_state=torch.tensor(oxidation_state, dtype=torch.long),
            spin_state=torch.tensor(spin_state, dtype=torch.long),
            csd_year=torch.tensor(csd_year, dtype=torch.long),
            stoichiometry=stoichiometry,
            smiles=smiles,
            coord_neighbors=coord_neighbors,
            bonds=bonds,
            bond_types=bond_types,
            bond_orders_continuous=bond_orders_continuous,
            is_dative_pair=is_dative_pair,
            pair_cloud_idx=pair_cloud_idx,
            cloud_offsets=torch.tensor(offsets, dtype=torch.long),
            n_clouds=b,
            # Wf-A16 pocket fields — default None for legacy callers.
            pocket_features=pocket_features_t,
            pocket_coords=pocket_coords_t,
            pocket_atom_types=pocket_atom_types_t,
            pocket_mask=pocket_mask_t,
        )
        batch._validate()
        return batch

    # ------------------------------------------------------------------
    # Shape-contract validation (E6 design §A)
    # ------------------------------------------------------------------
    def _validate(self) -> None:
        is_padded = self.positions.dim() == 3
        if is_padded:
            b, n_max = int(self.positions.shape[0]), int(self.positions.shape[1])
            n_total = b * n_max
            if int(self.atomic_numbers.shape[0]) != b or int(self.atomic_numbers.shape[1]) != n_max:
                raise ValueError(
                    f"padded atomic_numbers shape {tuple(self.atomic_numbers.shape)} "
                    f"!= positions shape (B={b}, N_max={n_max}, 3)"
                )
            for name, field in [
                ("atomic_radius", self.atomic_radius),
                ("formal_charge", self.formal_charge),
                ("natural_charge", self.natural_charge),
                ("bo_continuous", self.bo_continuous),
                ("batch_idx", self.batch_idx),
                ("coord_neighbors", self.coord_neighbors),
            ]:
                if int(field.shape[0]) != b or (field.dim() > 1 and int(field.shape[1]) not in (n_max, K_NEIGHBOURS)):
                    raise ValueError(
                        f"{name} shape {tuple(field.shape)} disagrees with "
                        f"(B={b}, N_max={n_max}, ...)"
                    )
        else:
            n_total = int(self.positions.shape[0])
            if int(self.atomic_numbers.shape[0]) != n_total:
                raise ValueError(
                    f"positions N={n_total} != atomic_numbers N="
                    f"{int(self.atomic_numbers.shape[0])}"
                )
            for name, field in [
                ("atomic_radius", self.atomic_radius),
                ("formal_charge", self.formal_charge),
                ("natural_charge", self.natural_charge),
                ("bo_continuous", self.bo_continuous),
                ("batch_idx", self.batch_idx),
                ("coord_neighbors", self.coord_neighbors),
            ]:
                if int(field.shape[0]) != n_total:
                    raise ValueError(
                        f"{name} shape {tuple(field.shape)} disagrees with N={n_total}"
                    )
            if int(self.cloud_offsets[-1].item()) != n_total:
                raise ValueError(
                    f"cloud_offsets[-1]={int(self.cloud_offsets[-1].item())} != N={n_total}"
                )
        if int(self.orig_n_atoms.shape[0]) != int(self.n_clouds):
            raise ValueError(
                f"orig_n_atoms B={int(self.orig_n_atoms.shape[0])} != n_clouds={self.n_clouds}"
            )
        if int(self.bonds.shape[1]) != int(self.bond_types.shape[0]):
            raise ValueError(
                f"bonds E={int(self.bonds.shape[1])} != bond_types E="
                f"{int(self.bond_types.shape[0])}"
            )

    # ------------------------------------------------------------------
    # Device moves
    # ------------------------------------------------------------------
    def to(self, device) -> "Batch":
        return Batch(
            positions=self.positions.to(device),
            atomic_numbers=self.atomic_numbers.to(device),
            atomic_radius=self.atomic_radius.to(device),
            formal_charge=self.formal_charge.to(device),
            natural_charge=self.natural_charge.to(device),
            bo_continuous=self.bo_continuous.to(device),
            batch_idx=self.batch_idx.to(device),
            orig_n_atoms=self.orig_n_atoms.to(device),
            metal_atom_index=self.metal_atom_index.to(device),
            oxidation_state=self.oxidation_state.to(device),
            spin_state=self.spin_state.to(device),
            csd_year=self.csd_year.to(device),
            stoichiometry=self.stoichiometry,
            smiles=self.smiles,
            coord_neighbors=self.coord_neighbors.to(device),
            bonds=self.bonds.to(device),
            bond_types=self.bond_types.to(device),
            bond_orders_continuous=self.bond_orders_continuous.to(device),
            is_dative_pair=self.is_dative_pair.to(device),
            pair_cloud_idx=self.pair_cloud_idx.to(device),
            cloud_offsets=self.cloud_offsets.to(device),
            n_clouds=self.n_clouds,
            # Wf-A16 pocket tensors — move when present, leave None
            # otherwise so legacy callers see a byte-identical Batch.
            pocket_features=(
                self.pocket_features.to(device)
                if self.pocket_features is not None
                else None
            ),
            pocket_coords=(
                self.pocket_coords.to(device)
                if self.pocket_coords is not None
                else None
            ),
            pocket_atom_types=(
                self.pocket_atom_types.to(device)
                if self.pocket_atom_types is not None
                else None
            ),
            pocket_mask=(
                self.pocket_mask.to(device)
                if self.pocket_mask is not None
                else None
            ),
        )

    def cpu(self) -> "Batch":
        return self.to(torch.device("cpu"))

    def cuda(self) -> "Batch":
        return self.to(torch.device("cuda"))

    # ------------------------------------------------------------------
    # Padded view (for sampling paths that need (B, N_max, ...) tensors)
    # ------------------------------------------------------------------
    def to_padded(self, target_n_atoms: int) -> "Batch":
        """Return a copy with per-atom fields zero-padded to ``(B, target_n_atoms, ...)``."""
        return Batch.collate_fn(
            _clouds_from_batch(self), target_n_atoms=target_n_atoms
        )

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    @property
    def cloud_sizes(self) -> torch.Tensor:
        """Per-cloud atom counts (post-flat, equals ``orig_n_atoms``)."""
        return self.orig_n_atoms

    def compute_angle_to_metal(self, cloud_idx: int, atom_j: int) -> float:
        """Single source of truth for angle_to_metal consumed by bond_head.

        Replaces the hardcoded 0.0 in
        :meth:`BondAwareDecoder._infer_pair_features` and
        :meth:`ConnectivityAwareDecoder._build_pair_features_from_edges`.
        """
        if cloud_idx < 0 or cloud_idx >= self.n_clouds:
            raise IndexError(
                f"cloud_idx {cloud_idx} out of range [0, {self.n_clouds})"
            )
        metal = int(self.metal_atom_index[cloud_idx].item())
        if metal == NO_METAL_INDEX:
            return 0.0
        start = int(self.cloud_offsets[cloud_idx].item())
        end = int(self.cloud_offsets[cloud_idx + 1].item())
        n_total = end - start
        if atom_j < 0 or atom_j >= n_total:
            raise IndexError(f"atom_j {atom_j} out of range [0, {n_total})")
        p_metal = self.positions[start + metal]
        p_atom = self.positions[start + atom_j]
        diff = p_atom - p_metal
        dist = float(torch.norm(diff).item())
        if dist < 1e-9:
            return 0.0
        # Deviation from radial: angle ~ 0 means the atom lies ON the metal
        # normal direction. For a square-planar metal with 4 equivalent
        # donors, every donor sits at ~0 rad deviation from its own normal
        # vector, which is what ``PairFeature.angle_to_metal`` consumes.
        # We project diff onto a reference vector pointing along +z from
        # the metal. Without an external normal direction, return the
        # angular deviation of the bond vector from the molecular centroid.
        centroid = self.positions[start:end].mean(dim=0)
        normal = centroid - p_metal
        n_norm = float(torch.norm(normal).item())
        if n_norm < 1e-9:
            return 0.0
        cos = float(torch.dot(diff, normal).item()) / (dist * n_norm)
        cos = max(-1.0, min(1.0, cos))
        return float(math.acos(cos))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _per_atom_field(
    cloud: AtomCloud, name: str, n: int, pad_value: float
) -> torch.Tensor:
    """Read a per-atom Optional field from a cloud, returning ``pad_value`` if missing."""
    val = getattr(cloud, name, None)
    if val is None:
        return torch.full((n,), pad_value, dtype=torch.float32)
    if int(val.shape[0]) != n:
        return torch.full((n,), pad_value, dtype=torch.float32)
    return val.float().view(-1)


def _scalar_field(cloud: AtomCloud, name: str, default: int) -> int:
    """Read a per-cloud scalar Optional field, returning ``default`` if missing."""
    val = getattr(cloud, name, None)
    if val is None:
        return int(default)
    try:
        return int(val.item())
    except (AttributeError, RuntimeError):
        return int(default)


def _pad_concat(
    chunks: Sequence[torch.Tensor], b: int, target: int, last_dim: int
) -> torch.Tensor:
    """Zero-pad each ``chunks[i]`` to ``target`` rows then stack to ``(b, target, ...)``."""
    out_shape = (b, target) + tuple(chunks[0].shape[1:])
    out = torch.zeros(out_shape, dtype=chunks[0].dtype)
    for i, t in enumerate(chunks):
        n_i = int(t.shape[0])
        if n_i >= target:
            out[i] = t[:target]
        else:
            out[i, :n_i] = t
    return out


def _clouds_from_batch(batch: "Batch") -> List[AtomCloud]:
    """Inverse of collate_fn — split a flat Batch back into its AtomCloud list.

    Used by :meth:`Batch.to_padded` so the padded view goes through the same
    validation path as a fresh collate.
    """
    clouds: List[AtomCloud] = []
    for i in range(int(batch.n_clouds)):
        start = int(batch.cloud_offsets[i].item())
        end = int(batch.cloud_offsets[i + 1].item())
        mask = batch.batch_idx == i
        n_atoms = end - start  # placeholder, set below
        n_atoms = end - start
        kwargs: Dict[str, object] = {
            "positions": batch.positions[start:end],
            "atomic_numbers": batch.atomic_numbers[start:end],
        }
        if int(batch.bonds.shape[1]) > 0:
            in_pair = batch.pair_cloud_idx == i
            if bool(in_pair.any().item()):
                e = int(in_pair.sum().item())
                kwargs["bonds"] = batch.bonds[:, in_pair]
                kwargs["bond_types"] = batch.bond_types[in_pair]
                kwargs["bond_orders_continuous"] = batch.bond_orders_continuous[in_pair]
                kwargs["is_dative_pair"] = batch.is_dative_pair[in_pair]
        clouds.append(AtomCloud(**kwargs))
    return clouds


__all__ = [
    "Batch",
    "PAD_Z",
    "PAD_R",
    "PAD_CHARGE",
    "PAD_BO",
    "NO_METAL_INDEX",
    "K_NEIGHBOURS",
]
