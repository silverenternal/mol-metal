"""RDKit chemistry helpers for MolFlow-Triton.

Most rdkit calls are wrapped in try/except; missing rdkit yields a clear
RuntimeError instead of an obscure ImportError at call sites.  The
exception is :func:`positions_to_smiles`, which raises ``ImportError``
with the install command since it cannot do anything useful without
rdkit.
"""

from __future__ import annotations

import base64
import hashlib
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

try:  # rdkit is optional
    from rdkit import Chem
    from rdkit.Chem import AllChem

    _HAS_RDKIT = True
except Exception:  # pragma: no cover - rdkit missing
    Chem = None  # type: ignore[assignment]
    AllChem = None  # type: ignore[assignment]
    _HAS_RDKIT = False


_RDKIT_INSTALL_HINT = (
    "rdkit is not installed. Install it with `uv pip install rdkit` "
    "(or `pip install rdkit-pypi`) before using chem_utils."
)


def _require_rdkit():
    if not _HAS_RDKIT:
        raise RuntimeError(_RDKIT_INSTALL_HINT)


def parse_smiles(smiles: str):
    """Parse a SMILES string into a Mol, returning None on failure."""
    if not _HAS_RDKIT:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def canonical_smiles(smiles: str) -> Optional[str]:
    """Return the canonical SMILES of `smiles`, or None on failure."""
    mol = parse_smiles(smiles)
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def check_valence(mol) -> Tuple[bool, str]:
    """Sanity-check that all atoms in `mol` have allowed valences.

    Uses RDKit's periodic table to fetch per-element allowed valences and
    compares them against each atom's total valence. Returns (ok, message).
    """
    _require_rdkit()
    if mol is None:
        return False, "mol is None"
    try:
        pt = Chem.GetPeriodicTable()
        problems: List[str] = []
        for atom in mol.GetAtoms():
            allowed = list(pt.GetValenceList(atom.GetAtomicNum()))
            total = atom.GetTotalValence()
            if total not in allowed:
                problems.append(
                    f"atom {atom.GetIdx()} ({atom.GetSymbol()}) "
                    f"total valence {total} not in allowed {allowed}"
                )
        if problems:
            return False, "; ".join(problems)
        return True, ""
    except Exception as exc:
        return False, f"valence check failed: {exc}"


def check_bonds(mol) -> Tuple[bool, str]:
    """Sanity-check that all bonds in `mol` have a sensible bond type.

    Returns (ok, message).
    """
    _require_rdkit()
    if mol is None:
        return False, "mol is None"
    try:
        problems: List[str] = []
        for bond in mol.GetBonds():
            bt = bond.GetBondType()
            # Anything that isn't one of the standard bond types is flagged.
            if bt not in (
                Chem.BondType.SINGLE,
                Chem.BondType.DOUBLE,
                Chem.BondType.TRIPLE,
                Chem.BondType.AROMATIC,
            ):
                problems.append(
                    f"bond {bond.GetIdx()} {bond.GetBeginAtomIdx()}-"
                    f"{bond.GetEndAtomIdx()} has unusual type {bt}"
                )
        if problems:
            return False, "; ".join(problems)
        return True, ""
    except Exception as exc:
        return False, f"bond check failed: {exc}"


def smiles_to_3d(
    smiles: str,
    *,
    optimize: bool = True,
    random_seed: int = 42,
):
    """Embed a molecule in 3D and optionally MMFF-optimize it.

    Returns the optimized Mol, or None on failure. Adds hydrogens before
    embedding (RDKit best practice for 3D coords).
    """
    _require_rdkit()
    mol = parse_smiles(smiles)
    if mol is None:
        return None
    try:
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = int(random_seed)
        embed_status = AllChem.EmbedMolecule(mol, params)
        if embed_status != 0:
            return None
        if optimize:
            try:
                AllChem.MMFFOptimizeMolecule(mol)
            except Exception:
                # Optimization failure is non-fatal; keep the embedded geometry.
                pass
        return mol
    except Exception:
        return None


def get_3d_coordinates(mol) -> Optional[List[Tuple[float, float, float]]]:
    """Return the 3D coordinates of `mol` as a list of (x, y, z) tuples.

    Returns None if `mol` has no conformer or rdkit is unavailable.
    """
    _require_rdkit()
    if mol is None:
        return None
    try:
        conf = mol.GetConformer()
        coords: List[Tuple[float, float, float]] = []
        for i in range(mol.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            coords.append((float(p.x), float(p.y), float(p.z)))
        return coords
    except Exception:
        return None


# ---------------------------------------------------------------------------
# 3D positions -> SMILES decoder
# ---------------------------------------------------------------------------
# Distance thresholds (Angstrom) for geometric bond perception.  Keys are
# frozensets of atomic numbers so lookup is order-independent.  Each entry
# maps an element pair to an ordered list of ``(max_distance, bond_order)``
# rules, sorted shortest-first: the first rule whose cutoff the measured
# distance falls under wins.  Shortest distance => highest bond order, which
# is the "greedy bond typing" rule.
#
#   C-H  < 1.20   single
#   C=O  < 1.35   double      C-O  < 1.55   single
#   C=C  < 1.45   double      C-C  < 1.70   single
#   C-N  < 1.60   single
#
# Element pairs that do not appear here are never bonded.  This is the exact
# threshold table specified for the decoder, so anything outside it is left
# unbonded rather than guessed -- notably H-H (which is what makes an
# exploded/degenerate geometry fall through to the "[]" branch) but also
# O-H and N-H, so hydroxyl/amine hydrogens come out as disconnected atoms
# (e.g. ethanol decodes to "CCO.[HH]").  Add entries here if those pairs
# should be perceived too.
_BOND_RULES: dict[frozenset, Tuple[Tuple[float, int], ...]] = {
    frozenset({6, 1}): ((1.20, 1),),
    frozenset({6, 7}): ((1.60, 1),),
    frozenset({6, 8}): ((1.35, 2), (1.55, 1)),
    frozenset({6}): ((1.45, 2), (1.70, 1)),
}

# Largest cutoff in the table; used to short-circuit the pair loop.
_MAX_BOND_DISTANCE = max(
    cutoff for rules in _BOND_RULES.values() for cutoff, _ in rules
)


def _bond_order_for(z_a: int, z_b: int, distance: float) -> Optional[int]:
    """Return the greedy bond order for a pair of atoms, or None if unbonded.

    ``distance`` is in Angstrom.  Returns 1 or 2 (the only orders the
    threshold table can produce) when the pair is close enough to be
    considered bonded, otherwise ``None``.
    """
    rules = _BOND_RULES.get(frozenset({int(z_a), int(z_b)}))
    if rules is None:
        return None
    for cutoff, order in rules:
        if distance < cutoff:
            return order
    return None


def infer_bonds(
    coords: Sequence[Sequence[float]],
    atomic_numbers: Sequence[int],
) -> List[Tuple[int, int, int]]:
    """Infer bonds from 3D geometry using the distance thresholds above.

    Parameters
    ----------
    coords:
        ``(N, 3)`` Cartesian coordinates in Angstrom.
    atomic_numbers:
        ``(N,)`` atomic numbers.  Entries ``<= 0`` mark padding atoms and
        are never bonded.

    Returns
    -------
    List of ``(i, j, bond_order)`` triples with ``i < j``.
    """
    bonds: List[Tuple[int, int, int]] = []
    n = len(atomic_numbers)
    max_sq = _MAX_BOND_DISTANCE * _MAX_BOND_DISTANCE
    for i in range(n):
        z_i = int(atomic_numbers[i])
        if z_i <= 0:
            continue
        xi, yi, zi = float(coords[i][0]), float(coords[i][1]), float(coords[i][2])
        for j in range(i + 1, n):
            z_j = int(atomic_numbers[j])
            if z_j <= 0:
                continue
            dx = xi - float(coords[j][0])
            dy = yi - float(coords[j][1])
            dz = zi - float(coords[j][2])
            d_sq = dx * dx + dy * dy + dz * dz
            if d_sq >= max_sq:
                continue
            order = _bond_order_for(z_i, z_j, d_sq ** 0.5)
            if order is not None:
                bonds.append((i, j, order))
    return bonds


def _positions_hash(coords: Sequence[Sequence[float]]) -> str:
    """Stable short hash of a coordinate block, used for INVALID_ tokens."""
    def _q(value: float) -> bytes:
        try:
            scaled = int(round(float(value) * 1000.0))
        except (ValueError, OverflowError):  # nan / inf
            scaled = 0
        # Clamp into int32 so pathological coordinates cannot raise.
        scaled = max(-(2**31), min(2**31 - 1, scaled))
        return scaled.to_bytes(4, "little", signed=True)

    quantized = b"".join(_q(v) for row in coords for v in row[:3])
    digest = hashlib.sha1(quantized).digest()[:9]
    return base64.urlsafe_b64encode(digest).decode("ascii").rstrip("=")


def _mol_from_geometry(
    coords: Sequence[Sequence[float]],
    atomic_numbers: Sequence[int],
    bonds: Sequence[Tuple[int, int, int]],
):
    """Build a sanitized RDKit Mol from atoms + inferred bonds, or None."""
    bond_types = {
        1: Chem.BondType.SINGLE,
        2: Chem.BondType.DOUBLE,
        3: Chem.BondType.TRIPLE,
    }
    mol = Chem.RWMol()
    index_map: dict[int, int] = {}
    for i, z in enumerate(atomic_numbers):
        z = int(z)
        if z <= 0:  # padding atom
            continue
        index_map[i] = mol.AddAtom(Chem.Atom(z))
    for i, j, order in bonds:
        if i not in index_map or j not in index_map:
            continue
        mol.AddBond(index_map[i], index_map[j], bond_types.get(order, Chem.BondType.SINGLE))
    try:
        out = mol.GetMol()
        Chem.SanitizeMol(out)
    except Exception:
        return None
    return out


def positions_to_smiles(
    positions: "torch.Tensor",
    atomic_numbers: "torch.Tensor | None" = None,
) -> List[str]:
    """Decode a batch of 3D conformations into canonical SMILES strings.

    Atom identities come from ``atomic_numbers`` (they are *not* guessed
    from the geometry); bonds are perceived from interatomic distances
    using the thresholds in :data:`_BOND_RULES`.  The resulting
    atoms + bonds graph is handed to RDKit and canonicalised with
    ``Chem.MolToSmiles(mol, canonical=True)``.

    Parameters
    ----------
    positions:
        ``(B, N, 3)`` coordinates in Angstrom.  A bare ``(N, 3)`` tensor is
        also accepted and treated as a batch of one.
    atomic_numbers:
        ``(B, N)`` integer atomic numbers.  Values ``<= 0`` are treated as
        padding and dropped.  When ``None`` every atom defaults to carbon,
        which preserves the old single-argument call signature.

    Returns
    -------
    One string per batch element:

    * a canonical SMILES on success,
    * ``"[]"`` when no bonds could be inferred (isolated / exploded
      geometry),
    * ``"INVALID_<hash>"`` when RDKit refuses to sanitize the perceived
      graph, where ``<hash>`` is a stable digest of the coordinates.

    Raises
    ------
    ImportError
        If rdkit is not importable.
    """
    if not _HAS_RDKIT:
        raise ImportError(
            "positions_to_smiles requires rdkit, which is not installed. "
            "Install it with `uv pip install rdkit` (or `pip install rdkit-pypi`)."
        )

    pos = positions.detach().to("cpu")
    if pos.dim() == 2:  # (N, 3) -> (1, N, 3)
        pos = pos.unsqueeze(0)
    if pos.dim() != 3 or pos.shape[-1] != 3:
        raise ValueError(
            f"positions must have shape (B, N, 3) or (N, 3); got {tuple(pos.shape)}."
        )
    batch, n_atoms, _ = pos.shape

    if atomic_numbers is None:
        numbers = [[6] * n_atoms for _ in range(batch)]
    else:
        z = atomic_numbers.detach().to("cpu")
        if z.dim() == 1:
            z = z.unsqueeze(0)
        if z.shape[0] != batch or z.shape[1] != n_atoms:
            raise ValueError(
                "atomic_numbers must have shape (B, N) matching positions "
                f"{(batch, n_atoms)}; got {tuple(z.shape)}."
            )
        numbers = z.long().tolist()

    coords_batch = pos.float().tolist()

    out: List[str] = []
    for coords, zs in zip(coords_batch, numbers):
        bonds = infer_bonds(coords, zs)
        if not bonds:
            out.append("[]")
            continue
        mol = _mol_from_geometry(coords, zs, bonds)
        if mol is None:
            out.append(f"INVALID_{_positions_hash(coords)}")
            continue
        try:
            # Drop explicit hydrogens so methane canonicalises to "C" rather
            # than "[H]C([H])([H])[H]".  Failure here is non-fatal.
            try:
                mol = Chem.RemoveHs(mol)
            except Exception:
                pass
            out.append(Chem.MolToSmiles(mol, canonical=True))
        except Exception:
            out.append(f"INVALID_{_positions_hash(coords)}")
    return out
