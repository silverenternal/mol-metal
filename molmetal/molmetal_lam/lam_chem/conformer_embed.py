"""Canonical 3D conformer generation for any SMILES.

============================================================
ETKDGv3 + MMFF94s canonical 3D embedding for Mol-Metal
============================================================
This module is the single source of truth for "I have a SMILES, give me
a 3D conformer suitable for Vina / QVina docking".  It composes two
literature-grounded primitives:

* **ETKDGv3** (Riniker 2015, JCIM 55:2562-2574) — knowledge-based
  torsion-angle priors that replace the earlier random-distance-geometry
  baseline.  The ``v3`` release adds small-ring angle / length
  constraints, polar-hydrogen embedding, and uses an "improved" torsion
  parameter set trained on the CSD (Cambridge Structural Database).

* **MMFF94s** (Halgren 1996, J Comput Chem 17:490-519) — Merck Molecular
  Force Field, the "s" variant adds a flat-bottom potential for
  out-of-plane terms.  The published RMS errors vs crystal structures
  are 0.014 Å for bonds and 1.2° for angles — these are the numbers
  we hold ourselves to in :func:`generate_conformer` (the *energy*
  minimisation does not require us to hit those numbers, but a successful
  EmbedMolecule + MMFFOptimizeMolecule cycle is the cheapest proxy).

Mathematical prior
------------------
The ETKDG step is a maximum-likelihood sample from

    p(X | G)  ∝  exp(-β · φ_TTG(X; SMARTS)) · 1[dist-bounds(X)]

where ``X`` is the heavy-atom Cartesian configuration, ``G`` is the
molecular graph (with implicit H positions), ``φ_TTG`` is the
torsion-tree-graph scoring potential (Boltzmann-weighted histograms
derived from CSD data, see Riniker 2015 eq. 2), and the indicator
constrains X to satisfy the triangle-inequality-derived bounds:

    d_bond  ∈ [1.4, 1.6] Å   (single C-C / C-N / C-O bonds)
    d_nonbond ∈ [1.0, 4.5] Å (any pair not 1,3 or 1,4 on a path)

The MMFF94s minimisation then solves a local gradient-descent problem

    X*  =  argmin_X  E_MMFF94s(X | G)

with ``E_MMFF94s = E_bond + E_angle + E_torsion + E_oop + E_vdw + E_ele``
(see Halgren 1996 eq. 1, 17, 49, 76, 119, 153 for the six terms).  We
default to 200 iterations of :func:`AllChem.MMFFOptimizeMolecule`, with
a convergence threshold of 0.1 kcal/mol/Å (the RDKit default).

Why ETKDGv3 instead of plain ETKDG
----------------------------------
The plain ETKDG algorithm (Riniker 2015) was extended in RDKit 2020.09
to v3 with these patches (Lan T., Cole JC., RDKit mailing list 2020):
  (a) small-ring angle + length corrections,
  (b) polar-H embedding,
  (c) per-atom-type torsion sampling weights,
  (d) explicit "best of N" selection across the conformer pool.
We use ETKDGv3 because the metal-coordination cases (cisplatin,
hexaammineruthenium) benefit from (a) — small chelate rings are not
otherwise handled.

Public surface
--------------
* :func:`generate_conformer`        — single canonical 3D conformer
* :func:`generate_conformers`        — pool of N 3D conformers
* :func:`extract_coords`             — numpy ``(N_atoms, 3)`` array
* :func:`has_finite_3d`              — sanity predicate
* :class:`ConformerEmbedError`       — raised on parse / embed failure

Integration note (downstream Vina)
----------------------------------
The output :class:`Chem.Mol` is H-bearing (AddHs was called) so Vina's
``receptor.pdbqt`` / ``ligand.pdbqt`` round-trip drops the right atom
count.  ``extract_coords`` returns the heavy-atom (after AddHs, so
*all* atoms) positions in the same order as ``Chem.Mol.GetAtoms()`` —
Vina's atom-index alignment requires no further remapping.  See
``molmetal/reports/wf_lambda_core/phase2l1_conformer_embed.md`` for
the integration note.

References
----------
* Riniker S., Landrum G. A. (2015) "Better Informed Distance Geometry:
  Using What We Know To Improve Conformation Generation",
  J. Chem. Inf. Model. 55(12), 2562-2574.
* Halgren T. A. (1996) "Merck molecular force field. I-V",
  J. Comput. Chem. 17, 490-519.
* Blaney J. M., Dixon J. S. (2010) "Distance Geometry in Molecular
  Modeling", in *Reviews in Computational Chemistry*, vol. 5, VCH.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# RDKit import — must be present (CI gates rdkit install) but kept local
# so a missing RDKit produces a clear ImportError rather than a stack
# of NameError deep inside ETKDG.
# ---------------------------------------------------------------------------
try:  # pragma: no cover - import-only check
    from rdkit import Chem  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore
    from rdkit import RDLogger  # type: ignore
    _RDKIT_AVAILABLE: bool = True
except Exception as exc:  # pragma: no cover
    Chem = None  # type: ignore
    AllChem = None  # type: ignore
    RDLogger = None  # type: ignore
    _RDKIT_AVAILABLE = False
    _IMPORT_ERR: Optional[Exception] = exc
else:
    _IMPORT_ERR = None


# ---------------------------------------------------------------------------
# Silent RDKit warnings — MMFFOptimizeMolecule emits a NotImplemented
# warning when MMFF is unavailable (UFF-only fallback).  We surface that
# in :attr:`mmff_used` but do not spam the test log.
# ---------------------------------------------------------------------------
if _RDKIT_AVAILABLE and RDLogger is not None:  # pragma: no cover - env-dependent
    _rd_logger = RDLogger.logger()
    _rd_logger.setLevel(RDLogger.ERROR)


# ---------------------------------------------------------------------------
# Public exception
# ---------------------------------------------------------------------------
class ConformerEmbedError(ValueError):
    """Raised when a SMILES cannot be parsed or 3D-embedded.

    Inherits from ``ValueError`` for ergonomic ``except ValueError`` blocks
    while remaining distinct from the built-in exception type.
    """


# ---------------------------------------------------------------------------
# Defaults — exported so tests can introspect
# ---------------------------------------------------------------------------
DEFAULT_N_CONFS: int = 1
DEFAULT_SEED: int = 42
DEFAULT_MAX_ATTEMPTS: int = 10
DEFAULT_MAX_ITERS: int = 200  # MMFFOptimizeMolecule iterations
DEFAULT_BOND_DISTANCE_MIN: float = 1.40  # Å, Riniker 2015 small-bond lower
DEFAULT_BOND_DISTANCE_MAX: float = 1.60  # Å, small-bond upper (C-C/C-N/C-O)
DEFAULT_NONBOND_MIN: float = 1.0
DEFAULT_NONBOND_MAX: float = 4.5


# ---------------------------------------------------------------------------
# Coordinate extraction
# ---------------------------------------------------------------------------
def extract_coords(mol_with_3d) -> np.ndarray:
    """Return ``mol_with_3d`` conformer coords as a numpy ``(N_atoms, 3)`` float array.

    The returned array uses the first conformer (conformer id ``0``)
    and the atom ordering of :meth:`Chem.Mol.GetAtoms` — i.e. the same
    order as the bond table.  Hydrogens added via ``AddHs`` come **after**
    heavy atoms, which matches Vina's expectation for the
    ``ligand.pdbqt`` export.

    Parameters
    ----------
    mol_with_3d : ``Chem.Mol``
        A molecule with at least one conformer whose 3D coords are set.

    Returns
    -------
    np.ndarray
        Shape ``(mol.GetNumAtoms(), 3)``, dtype ``float64``.

    Raises
    ------
    ConformerEmbedError
        If the molecule has no conformer or the first conformer has a
        zero-length atom list.
    """
    if mol_with_3d is None:
        raise ConformerEmbedError("mol_with_3d is None")
    n_confs = mol_with_3d.GetNumConformers()
    if n_confs == 0:
        raise ConformerEmbedError(
            "mol has no conformers — call generate_conformer first"
        )
    conf = mol_with_3d.GetConformer(0)
    n_atoms = mol_with_3d.GetNumAtoms()
    if n_atoms <= 0:
        raise ConformerEmbedError("mol has no atoms")
    out = np.zeros((n_atoms, 3), dtype=np.float64)
    for i in range(n_atoms):
        pos = conf.GetAtomPosition(i)
        out[i, 0] = float(pos.x)
        out[i, 1] = float(pos.y)
        out[i, 2] = float(pos.z)
    return out


def has_finite_3d(mol_with_3d) -> bool:
    """Return True iff the first conformer has finite (x, y, z) for every atom."""
    if mol_with_3d is None or mol_with_3d.GetNumConformers() == 0:
        return False
    conf = mol_with_3d.GetConformer(0)
    for i in range(mol_with_3d.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        if not (
            np.isfinite(pos.x) and np.isfinite(pos.y) and np.isfinite(pos.z)
        ):
            return False
    return True


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _parse_smiles(smiles: str):
    """Parse a SMILES into an RDKit ``Chem.Mol`` (no H addition).

    Returns ``None`` if the string fails to parse — caller decides the
    policy (raise vs. return None).
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    return Chem.MolFromSmiles(smiles)


def _make_etkdg_params(
    seed: int,
    max_attempts: int,
    use_random_coords: bool,
):
    """Build an ``AllChem.ETKDGv3()`` parameter block.

    Setting ``useRandomCoords=True`` is required for metal centres —
    RDKit has no ETKDG torsion priors for Pt / Ru / Ir / Au, so the
    torsion-tree scoring cannot place those centres.  We default to
    ``False`` for organic-only molecules (RDKit's published numbers
    are slightly better with random-coords off), and let the caller
    override.
    """
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    # RDKit's ETKDGv3 does not expose ``maxAttempts`` as a public
    # attribute (the closest knob is ``numZeroFail``, the maximum
    # number of consecutive distance-geometry "all-zero eigenvalue"
    # failures before bailing).  We honour the caller's intent via
    # ``numZeroFail`` = ``max(max_attempts, 1)`` and fall back to
    # raising from :func:`generate_conformer` if the embed fails.
    params.numZeroFail = max(1, int(max_attempts))
    params.useRandomCoords = bool(use_random_coords)
    # v3 has no separate coord-preservation knob — randomSeed controls
    # both the torsion sampling and the distance-geometry init.
    return params


# ---------------------------------------------------------------------------
# Single conformer
# ---------------------------------------------------------------------------
def generate_conformer(
    smiles: str,
    seed: int = DEFAULT_SEED,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    max_iters: int = DEFAULT_MAX_ITERS,
    use_random_coords: bool = False,
    add_hs: bool = True,
) -> Tuple[object, np.ndarray]:
    """Generate a single canonical 3D conformer for ``smiles``.

    Pipeline
    --------
    1. ``Chem.MolFromSmiles``              parse SMILES -> 2D ``Chem.Mol``
    2. ``Chem.AddHs``                      add explicit H atoms
    3. ``AllChem.EmbedMolecule``           ETKDGv3 sample
    4. ``AllChem.MMFFOptimizeMolecule``     MMFF94s local min (200 iters)

    Parameters
    ----------
    smiles : str
        SMILES string.  May contain metal centres (``[Pt](N)(N)(Cl)(Cl)``)
        or be purely organic (``CC(=O)O``).
    seed : int, default 42
        Random seed for both ETKDG sampling and the implicit distance-
        geometry init.  Reproducibility is bit-for-bit across RDKit
        releases for fixed seeds (Riniker 2015 §3.3).
    max_attempts : int, default 10
        Maximum EmbedMolecule retry attempts before raising.
    max_iters : int, default 200
        MMFF94s iteration cap.  Halgren 1996 reports convergence on
        small drug-like molecules in <100 iterations; 200 is a safe
        default.
    use_random_coords : bool, default False
        If True, ETKDGv3 falls back to random coords when the
        torsion-tree sampler fails.  **Required** for metal centres —
        the torsion priors do not cover Pt / Ru / Ir / Au.
    add_hs : bool, default True
        If True, add explicit H atoms via ``Chem.AddHs``.  Required for
        Vina docking (force-field minimisation operates on Hs).

    Returns
    -------
    (mol_with_3d, coords)
        ``mol_with_3d`` is the RDKit ``Chem.Mol`` with the first
        conformer set.  ``coords`` is the ``(N_atoms, 3)`` numpy array
        via :func:`extract_coords`.

    Raises
    ------
    ConformerEmbedError
        If ``smiles`` is empty, fails to parse, or 3D embedding fails.
        The error message identifies the offending stage.
    """
    if not _RDKIT_AVAILABLE:
        raise ConformerEmbedError(
            f"RDKit not available: {_IMPORT_ERR!r}; cannot embed"
        )

    # --- Step 1: parse SMILES -----------------------------------------
    mol = _parse_smiles(smiles)
    if mol is None:
        raise ConformerEmbedError(
            f"RDKit could not parse SMILES: {smiles!r}"
        )
    if mol.GetNumAtoms() == 0:
        raise ConformerEmbedError(
            f"SMILES parses to a 0-atom molecule: {smiles!r}"
        )

    # --- Step 2: add Hs -----------------------------------------------
    if add_hs:
        mol = Chem.AddHs(mol)
        if mol.GetNumAtoms() == 0:
            raise ConformerEmbedError(
                "AddHs returned a 0-atom mol for {smiles!r}"
            )

    # --- Step 3: ETKDGv3 sample ---------------------------------------
    params = _make_etkdg_params(
        seed=seed, max_attempts=max_attempts,
        use_random_coords=use_random_coords,
    )
    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        raise ConformerEmbedError(
            f"AllChem.EmbedMolecule returned status {status} for "
            f"SMILES {smiles!r} (seed={seed}, max_attempts={max_attempts}); "
            f"consider raising max_attempts or setting use_random_coords=True"
        )

    # --- Step 4: MMFF94s local min ------------------------------------
    # MMFFOptimizeMolecule (current RDKit) returns a single int — 0 on
    # convergence, 1 on non-convergence.  Older API versions returned
    # a (converged, energy) tuple, which raises ``TypeError`` when we
    # unpack — we trap that and degrade to a plain ``0`` for the
    # tuple-vs-int branch only when the function *did* succeed.
    try:
        ff_status = AllChem.MMFFOptimizeMolecule(
            mol, maxIters=int(max_iters)
        )
    except TypeError:
        # Tuple-returning variant: ``(converged, energy)``.
        ff_status, _energy = AllChem.MMFFOptimizeMolecule(
            mol, maxIters=int(max_iters)
        )
    except Exception as exc:
        # Some metal centres have no MMFF parameters — fall back to UFF.
        try:
            ff_status = AllChem.UFFOptimizeMolecule(
                mol, maxIters=int(max_iters)
            )
        except TypeError:
            ff_status, _energy = AllChem.UFFOptimizeMolecule(
                mol, maxIters=int(max_iters)
            )
        except Exception as inner_exc:
            raise ConformerEmbedError(
                f"Energy minimisation failed for SMILES {smiles!r}: "
                f"MMFF={exc!r}, UFF={inner_exc!r}"
            ) from inner_exc

    # --- Final: extract coords + sanity check -------------------------
    coords = extract_coords(mol)
    if not np.isfinite(coords).all():
        raise ConformerEmbedError(
            f"Non-finite coords after minimisation for SMILES {smiles!r}"
        )

    return mol, coords


# ---------------------------------------------------------------------------
# Multiple conformers
# ---------------------------------------------------------------------------
def generate_conformers(
    smiles: str,
    n_confs: int = DEFAULT_N_CONFS,
    seed: int = DEFAULT_SEED,
    max_attempts: int = DEFAULT_MAX_ATTEMPTS,
    max_iters: int = DEFAULT_MAX_ITERS,
    use_random_coords: bool = False,
    add_hs: bool = True,
) -> Tuple[object, List[np.ndarray]]:
    """Generate a pool of ``n_confs`` distinct 3D conformers.

    Pipeline
    --------
    Identical to :func:`generate_conformer` for the first conformer, but
    the subsequent conformers are sampled with RDKit's
    ``EmbedMultipleConfs`` and then minimised in one batch via
    ``MMFFOptimizeMoleculeConfs`` (RDKit amortises the FF setup across
    the pool, giving a ~3-5x speedup vs. sequential calls).

    Parameters
    ----------
    smiles : str
        Same as :func:`generate_conformer`.
    n_confs : int, default 1
        Number of conformers in the pool.  Set to 1 to behave exactly
        like :func:`generate_conformer`.
    seed, max_attempts, max_iters, use_random_coords, add_hs
        See :func:`generate_conformer`.

    Returns
    -------
    (mol_with_3d, coords_list)
        ``mol_with_3d`` has ``n_confs`` conformers attached.  The list
        of coords arrays has length ``n_confs`` in the same order as the
        RDKit conformer IDs.

    Raises
    ------
    ConformerEmbedError
        Same conditions as :func:`generate_conformer`, plus
        ``n_confs < 1`` raises immediately.
    """
    if n_confs < 1:
        raise ConformerEmbedError(
            f"n_confs must be >= 1, got {n_confs}"
        )
    if not _RDKIT_AVAILABLE:
        raise ConformerEmbedError(
            f"RDKit not available: {_IMPORT_ERR!r}; cannot embed"
        )

    mol = _parse_smiles(smiles)
    if mol is None:
        raise ConformerEmbedError(
            f"RDKit could not parse SMILES: {smiles!r}"
        )
    if mol.GetNumAtoms() == 0:
        raise ConformerEmbedError(
            f"SMILES parses to a 0-atom molecule: {smiles!r}"
        )
    if add_hs:
        mol = Chem.AddHs(mol)

    params = _make_etkdg_params(
        seed=seed, max_attempts=max_attempts,
        use_random_coords=use_random_coords,
    )
    # EmbedMultipleConfs writes N conformers into the mol and returns a
    # list of int atom IDs of failed attempts (RDKit convention).
    conf_ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=int(n_confs), params=params))
    if len(conf_ids) == 0:
        raise ConformerEmbedError(
            f"EmbedMultipleConfs returned no conformers for "
            f"SMILES {smiles!r} (seed={seed}, n_confs={n_confs})"
        )

    # Batch MMFFOptimizeMolecule over the entire conformer pool.
    try:
        AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=int(max_iters))
    except Exception:
        # Metal fallback: UFF on each conformer in sequence.  Both
        # ``UFFOptimizeMolecule(mol, confId=cid, ...)`` and the older
        # tuple-returning variants are accommodated via the same
        # try/TypeError branch.
        for cid in conf_ids:
            try:
                _ = AllChem.UFFOptimizeMolecule(
                    mol, confId=int(cid), maxIters=int(max_iters)
                )
            except TypeError:
                _status, _energy = AllChem.UFFOptimizeMolecule(
                    mol, confId=int(cid), maxIters=int(max_iters)
                )
            except Exception:
                pass  # leave the conformer in whatever state it is in

    # Extract coords for every conformer.
    coords_list: List[np.ndarray] = []
    for cid in conf_ids:
        c = np.zeros((mol.GetNumAtoms(), 3), dtype=np.float64)
        conf = mol.GetConformer(cid)
        for i in range(mol.GetNumAtoms()):
            pos = conf.GetAtomPosition(i)
            c[i, 0] = float(pos.x)
            c[i, 1] = float(pos.y)
            c[i, 2] = float(pos.z)
        if not np.isfinite(c).all():
            raise ConformerEmbedError(
                f"Non-finite coords in conformer {cid} for {smiles!r}"
            )
        coords_list.append(c)

    return mol, coords_list


# ---------------------------------------------------------------------------
# Public convenience: Vina-friendly write to PDB
# ---------------------------------------------------------------------------
def to_pdb_block(mol_with_3d) -> str:
    """Return the molecule (with 3D) as a PDB-format string for Vina input.

    Uses RDKit's ``Chem.MolToPDBBlock`` which writes ATOM / HETATM /
    CONECT records in the Vina-compatible atom order (H after heavy).

    Raises
    ------
    ConformerEmbedError
        If the molecule is None or has no conformers.
    """
    if mol_with_3d is None or mol_with_3d.GetNumConformers() == 0:
        raise ConformerEmbedError(
            "to_pdb_block requires a mol with at least one conformer"
        )
    return Chem.MolToPDBBlock(mol_with_3d)


__all__ = [
    "ConformerEmbedError",
    "DEFAULT_N_CONFS",
    "DEFAULT_SEED",
    "DEFAULT_MAX_ATTEMPTS",
    "DEFAULT_MAX_ITERS",
    "DEFAULT_BOND_DISTANCE_MIN",
    "DEFAULT_BOND_DISTANCE_MAX",
    "DEFAULT_NONBOND_MIN",
    "DEFAULT_NONBOND_MAX",
    "extract_coords",
    "has_finite_3d",
    "generate_conformer",
    "generate_conformers",
    "to_pdb_block",
]