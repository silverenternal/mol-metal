"""MD relaxation stub — Pitfall Rank 13, P2.5 (TODO-30).

Background
----------
Generated small molecules may carry implausible 3D geometry (compressed
bonds, large non-bonded contacts, distorted angles) because the
construction pipeline (``conformer_embed.py``) ships a single
ETKDGv3 sample and one MMFF/UFF local-min pass.  Real docking / scoring
downstream paths want the structure to sit in a local minimum of a
well-defined force field.

A full molecular-dynamics relaxation is overkill for the stub
pipeline — what we want is:

  * a *simple* force field whose energy and gradient are computed
    directly from coordinates + atom-type labels (no bonds/bond-orders,
    no implicit-H bookkeeping), so the stub is **independent** of the
    R17 ``conformer_embed`` /
    ``batched_rdkit`` modules;
  * a *short* steepest-descent / Metropolis Monte Carlo relaxation that
    lowers the energy of a noisy initial conformer without blowing up;
  * a *geometry validator* that flags implausible bond lengths and
    close non-bonded contacts — so downstream docking can refuse a
    poisoned structure.

Force-field design (CPU stub)
----------------------------
We use a sum of three terms, all of which are deliberately simple and
*do not require RDKit's MMFF/UFF atom-type system*.  This keeps the
stub independent of the ship-shipped conformer_embed code path.

  E_bond(r_ij)   =  sum_{(i,j) bonded}  k_b * (r_ij - r_0)^2

      where the bond list is given implicitly by *covalent radii
      proximity* — pairs ``(i, j)`` with ``r_ij < (cov_rad_i + cov_rad_j) *
      bond_tol`` are flagged as bonded.  Default ``k_b = 300 kcal/mol/A^2``
      and ``r_0 = sum(cov_rad_i + cov_rad_j)`` (the equilibrium bond
      length).

  E_nonbond(r_ij) = sum_{(i,j) non-bonded, i<j}  epsilon *
                    [ (sigma_ij / r_ij)^12 - 2 * (sigma_ij / r_ij)^6 ]

      a truncated Lennard-Jones.  ``sigma_ij = (cov_rad_i + cov_rad_j)``
      and ``epsilon = 0.1 kcal/mol`` (a deliberately soft repulsion so
      the stub does not blow up with NaNs if the input has very close
      non-bonded atoms).

  E_angle(theta_ijk) = sum_{(i, j, k) two bonds sharing j}
                       k_a * (theta_ijk - theta_0)^2

      ``k_a = 100 kcal/mol/rad^2`` and ``theta_0 = pi / 2`` (90° — a
      rough sp2 average).

Gradients are computed **numerically** by central differences with a
step ``h = 1e-3`` A — this is the cheapest correct way for a stub and
matches the spirit of an RDKit UFF/MMFF "stub" (RDKit's real
``MMFFOptimizeMolecule`` uses analytical gradients that we cannot
reproduce without the atom-typing layer).  The numerical gradient has
``O(3N)`` energy calls per call, which is fine for the stub (N is the
size of one ligand, ~10-100 atoms).

Monte Carlo step
----------------
The :class:`MDRelaxation` class exposes a single
:meth:`monte_carlo_step` that proposes a uniform-random perturbation of
all atoms (scale ``step_size`` A) and accepts the proposal with the
Metropolis criterion
``min(1, exp(-(E_new - E_old) / kT))`` so the relaxation can escape
shallow local minima.  ``kT`` is taken at the configured temperature
with ``kB = 0.0019872041 kcal/mol/K``.

Public API
----------
:class:`SimpleForceField`       : CPU force field (UFF-like).
:class:`MDRelaxation`           : MD relaxation loop + MC step.
:class:`GeometryValidator`      : bond-length / close-contact sanity.
:func:`register_md_relaxation`  : module-level registration hook.

Constraints honoured
--------------------
* **No modification** of ``conformer_embed.py`` / ``batched_rdkit.py``.
* **CPU-only** — no torch, no GPU, no learned priors.
* **Pure numpy** — the FF / MC step operate on
  ``np.ndarray`` and a ``List[str]`` of atom-type labels.  RDKit is
  only imported lazily (and is not required at all).

Lit anchors
-----------
* Rappé A. K., Casewit C. J., Colwell K. S., Goddard W. A., Skiff W. M.
  (1992) "UFF, a Full Periodic Table Force Field for Molecular
  Mechanics and Molecular Dynamics Simulations", J. Am. Chem. Soc.
  114, 10024-10035.
* Halgren T. A. (1996) "Merck molecular force field. I-V",
  J. Comput. Chem. 17, 490-519.
* Metropolis N., Rosenbluth A. W., Rosenbluth M. N., Teller A. H.,
  Teller E. (1953) "Equation of State Calculations by Fast Computing
  Machines", J. Chem. Phys. 21, 1087-1092.

Honest framing
--------------
The force field is a *CPU stub* with a tiny parameter set
(default ``epsilon = 0.1 kcal/mol``, ``k_b = 300``, ``k_a = 100``).
It is not a substitute for a real UFF/MMFF parameterisation — the
intent is to (a) lower the energy of a noisy conformer by a few
kcal/mol and (b) flag obviously-broken geometries.  Anything that
needs chemical accuracy (e.g. docking re-scoring) should fall back to
the real ``AllChem.MMFFOptimizeMolecule`` pipeline that ships with R17
``conformer_embed.py``.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Defaults — exported so tests can introspect
# ---------------------------------------------------------------------------
DEFAULT_METHOD: str = "UFF"
DEFAULT_BOND_K: float = 300.0  # kcal/mol/A^2 — bond stretch k
DEFAULT_ANGLE_K: float = 100.0  # kcal/mol/rad^2 — angle bend k
DEFAULT_NB_EPSILON: float = 0.1  # kcal/mol — LJ well depth
DEFAULT_BOND_TOL: float = 1.30  # cov-rad sum multiplier for bonding
DEFAULT_GRADIENT_STEP: float = 1.0e-3  # A — central-difference h
DEFAULT_RELAX_STEPS: int = 100
DEFAULT_RELAX_STEP_SIZE: float = 0.01  # A — steepest-descent step
DEFAULT_MC_STEP_SIZE: float = 0.05  # A — MC perturbation scale
DEFAULT_TEMPERATURE_K: float = 300.0
KBOLTZMANN: float = 0.0019872041  # kcal/mol/K
# Geometry validator defaults
DEFAULT_BOND_THRESHOLD: float = 2.0  # A — max distance for "bonded" pair
DEFAULT_BOND_LOWER: float = 0.70  # bond length ratio lower bound
DEFAULT_BOND_UPPER: float = 1.60  # bond length ratio upper bound
DEFAULT_CLOSE_CONTACT: float = 1.20  # A — non-bonded close-contact distance
DEFAULT_ABSOLUTE_MIN: float = 0.80  # A — any pair below this is suspicious


# Covalent radii (Å) — Cordero et al. 2008 Dalton Trans. values
# (rounded to 0.01 Å).  Used for both the bond-length check in
# :class:`GeometryValidator` and the implicit bond list in
# :class:`SimpleForceField`.
_COVALENT_RADII: Dict[str, float] = {
    "H": 0.31,
    "He": 0.28,
    "Li": 1.28,
    "Be": 0.96,
    "B": 0.84,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
    "Ne": 0.58,
    "Na": 1.66,
    "Mg": 1.41,
    "Al": 1.21,
    "Si": 1.11,
    "P": 1.07,
    "S": 1.05,
    "Cl": 1.02,
    "Ar": 1.06,
    "K": 2.03,
    "Ca": 1.76,
    "Ni": 1.24,
    "Cu": 1.32,
    "Zn": 1.22,
    "Br": 1.20,
    "I": 1.39,
    "Pt": 1.36,
    "Ru": 1.26,
    "Pd": 1.39,
    "Au": 1.36,
    "Ir": 1.26,
    "Os": 1.26,
    "Rh": 1.25,
    "Re": 1.28,
}


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _cov_radius(symbol: str) -> float:
    """Return the covalent radius (Å) for ``symbol``, defaulting to 1.0."""
    return float(_COVALENT_RADII.get(str(symbol).strip().capitalize(), 1.0))


def _pair_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Return a symmetric ``(N, N)`` Euclidean distance matrix (Å)."""
    diffs = coords[:, None, :] - coords[None, :, :]
    return np.sqrt(np.sum(diffs * diffs, axis=-1))


def _bond_pairs(
    coords: np.ndarray,
    atom_types: List[str],
    bond_tol: float = DEFAULT_BOND_TOL,
) -> List[Tuple[int, int]]:
    """Return the implicit bond list as ``(i, j)`` index pairs (i < j).

    A pair is considered bonded iff its distance is less than
    ``bond_tol * (cov_rad_i + cov_rad_j)``.  This is a coarse proxy
    for the bond table — it is *deliberately* generous so that the
    stub can be driven without a parsed bond graph.
    """
    n = coords.shape[0]
    dmat = _pair_distance_matrix(coords)
    radii = np.asarray([_cov_radius(t) for t in atom_types], dtype=np.float64)
    thresh = bond_tol * (radii[:, None] + radii[None, :])
    out: List[Tuple[int, int]] = []
    for i in range(n):
        for j in range(i + 1, n):
            if dmat[i, j] <= thresh[i, j]:
                out.append((i, j))
    return out


def _angle_triplets(
    bond_list: List[Tuple[int, int]],
    n_atoms: int,
) -> List[Tuple[int, int, int]]:
    """Return ``(i, j, k)`` triples where ``(i, j)`` and ``(j, k)`` are bonds.

    ``j`` is the central atom.  Self-angles (``i == k``) are excluded.
    """
    neighbours: Dict[int, List[int]] = {i: [] for i in range(n_atoms)}
    for i, j in bond_list:
        neighbours[i].append(j)
        neighbours[j].append(i)
    out: List[Tuple[int, int, int]] = []
    for j in range(n_atoms):
        nbrs = neighbours[j]
        for ii, i in enumerate(nbrs):
            for k in nbrs[ii + 1 :]:
                out.append((i, j, k))
    return out


# ---------------------------------------------------------------------------
# SimpleForceField
# ---------------------------------------------------------------------------
class SimpleForceField:
    """A simple CPU force field — bond + angle + Lennard-Jones nonbonded.

    The parameter set is deliberately tiny (``epsilon``, ``k_b``,
    ``k_a``) and intentionally **not** the full UFF/MMFF.  It is
    sufficient to (a) lower the energy of a noisy conformer by a few
    kcal/mol and (b) produce a useful gradient for steepest descent.
    """

    def __init__(
        self,
        method: str = DEFAULT_METHOD,
        *,
        bond_k: float = DEFAULT_BOND_K,
        angle_k: float = DEFAULT_ANGLE_K,
        nb_epsilon: float = DEFAULT_NB_EPSILON,
        bond_tol: float = DEFAULT_BOND_TOL,
        gradient_step: float = DEFAULT_GRADIENT_STEP,
    ) -> None:
        """Initialise the stub force field.

        Parameters
        ----------
        method : str, default ``"UFF"``
            Cosmetic label (no atom-typing dispatch).  ``"MMFF"`` is
            accepted but maps to the same numerical-gradient stub.
        bond_k : float, default 300.0
            Bond-stretch force constant (``kcal/mol/A^2``).
        angle_k : float, default 100.0
            Angle-bend force constant (``kcal/mol/rad^2``).
        nb_epsilon : float, default 0.1
            Lennard-Jones well depth (``kcal/mol``).
        bond_tol : float, default 1.30
            Multiplier on the covalent-radius sum used to detect
            bonded pairs.
        gradient_step : float, default 1.0e-3
            Central-difference step (Å) for the numerical gradient.
        """
        method_norm = str(method).strip().upper()
        if method_norm not in {"UFF", "MMFF"}:
            raise ValueError(f"Unsupported method {method!r}; expected 'UFF' or 'MMFF'")
        self.method: str = method_norm
        self.bond_k: float = float(bond_k)
        self.angle_k: float = float(angle_k)
        self.nb_epsilon: float = float(nb_epsilon)
        self.bond_tol: float = float(bond_tol)
        self.gradient_step: float = float(gradient_step)

    # ------------------------------------------------------------------
    # Energy
    # ------------------------------------------------------------------
    def energy(
        self,
        coords: np.ndarray,
        atom_types: List[str],
    ) -> float:
        """Return the total energy (kcal/mol) of the configuration.

        The energy is the sum of three terms:

          E_bond    = sum_{bonded} k_b * (r - r_0)^2
          E_angle   = sum_{ijk}   k_a * (theta - pi/2)^2
          E_nonbond = sum_{i<j, nonbonded}  epsilon * (sigma/r)^12
                      - 2 * (sigma/r)^6

        Parameters
        ----------
        coords : np.ndarray
            Shape ``(N, 3)``.
        atom_types : List[str]
            Atomic symbols (e.g. ``["C", "N", "H", ...]``).

        Returns
        -------
        float
            Total energy in kcal/mol.  Returns ``0.0`` for ``N <= 1``.
        """
        c = np.asarray(coords, dtype=np.float64)
        if c.ndim != 2 or c.shape[1] != 3:
            raise ValueError(f"coords must have shape (N, 3); got {coords.shape!r}")
        n = int(c.shape[0])
        if n <= 1:
            return 0.0
        if n != len(atom_types):
            raise ValueError(
                f"coords has {n} rows but atom_types has {len(atom_types)} entries"
            )

        bonds = _bond_pairs(c, atom_types, bond_tol=self.bond_tol)
        radii = np.asarray([_cov_radius(t) for t in atom_types], dtype=np.float64)

        # --- Bond stretch ---
        e_bond = 0.0
        if bonds:
            for i, j in bonds:
                d = float(np.linalg.norm(c[i] - c[j]))
                r0 = float(radii[i] + radii[j])
                e_bond += self.bond_k * (d - r0) ** 2

        # --- Angle bend ---
        e_angle = 0.0
        if self.angle_k > 0.0 and bonds:
            triplets = _angle_triplets(bonds, n)
            for ii, j, k in triplets:
                v1 = c[ii] - c[j]
                v2 = c[k] - c[j]
                n1 = float(np.linalg.norm(v1))
                n2 = float(np.linalg.norm(v2))
                if n1 < 1.0e-9 or n2 < 1.0e-9:
                    continue
                cos_t = float(np.dot(v1, v2) / (n1 * n2))
                cos_t = max(-1.0, min(1.0, cos_t))
                theta = math.acos(cos_t)
                e_angle += self.angle_k * (theta - math.pi / 2.0) ** 2

        # --- Lennard-Jones nonbonded ---
        e_nb = 0.0
        if self.nb_epsilon > 0.0:
            bonded_set = {tuple(sorted((i, j))) for i, j in bonds}
            for i in range(n):
                for j in range(i + 1, n):
                    if (i, j) in bonded_set:
                        continue
                    d = float(np.linalg.norm(c[i] - c[j]))
                    if d < 0.5:  # avoid division by ~0
                        continue
                    sigma = float(radii[i] + radii[j])
                    sr6 = (sigma / d) ** 6
                    e_nb += self.nb_epsilon * (sr6 * sr6 - 2.0 * sr6)

        return float(e_bond + e_angle + e_nb)

    # ------------------------------------------------------------------
    # Gradient (numerical central differences)
    # ------------------------------------------------------------------
    def gradient(
        self,
        coords: np.ndarray,
        atom_types: List[str],
    ) -> np.ndarray:
        """Return the *negative* gradient of ``energy`` at ``coords``.

        Returns an array of shape ``(N, 3)`` where the entry is
        ``-dE/dx`` (so the caller can use it directly as a steepest-
        descent force).  Computed by central differences with step
        :attr:`gradient_step`.

        Parameters
        ----------
        coords : np.ndarray
            Shape ``(N, 3)``.
        atom_types : List[str]
            Atomic symbols.

        Returns
        -------
        np.ndarray
            Shape ``(N, 3)``, dtype ``float64``.
        """
        c = np.asarray(coords, dtype=np.float64)
        if c.ndim != 2 or c.shape[1] != 3:
            raise ValueError(f"coords must have shape (N, 3); got {coords.shape!r}")
        n = int(c.shape[0])
        if n == 0:
            return np.zeros((0, 3), dtype=np.float64)
        if n != len(atom_types):
            raise ValueError(
                f"coords has {n} rows but atom_types has {len(atom_types)} entries"
            )
        h = float(self.gradient_step)
        grad = np.zeros_like(c)
        e0 = self.energy(c, atom_types)
        for i in range(n):
            for d in range(3):
                cp = c.copy()
                cm = c.copy()
                cp[i, d] += h
                cm[i, d] -= h
                ep = self.energy(cp, atom_types)
                em = self.energy(cm, atom_types)
                grad[i, d] = -(ep - em) / (2.0 * h)
        return grad

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------
    def __repr__(self) -> str:
        return (
            f"SimpleForceField(method={self.method!r}, "
            f"bond_k={self.bond_k}, angle_k={self.angle_k}, "
            f"nb_epsilon={self.nb_epsilon})"
        )


# ---------------------------------------------------------------------------
# MDRelaxation
# ---------------------------------------------------------------------------
class MDRelaxation:
    """Simple CPU MD relaxation — steepest descent + Metropolis MC step.

    The relaxation loop alternates:

    1. a steepest-descent step ``X -= step_size * grad``,
    2. a Metropolis Monte Carlo step that perturbs every atom by a
       uniform random ``[-perturb, +perturb]`` A and accepts the
       proposal with probability
       ``min(1, exp(-(E_new - E_old) / kT))``.

    The steepest-descent step lowers the energy monotonically when the
    gradient is exact; the MC step lets the trajectory escape shallow
    local minima where the gradient vanishes.
    """

    def __init__(
        self,
        force_field: SimpleForceField,
        n_steps: int = DEFAULT_RELAX_STEPS,
        step_size: float = DEFAULT_RELAX_STEP_SIZE,
        temperature: float = DEFAULT_TEMPERATURE_K,
        mc_step_size: float = DEFAULT_MC_STEP_SIZE,
        seed: Optional[int] = None,
    ) -> None:
        """Initialise the relaxation loop.

        Parameters
        ----------
        force_field : SimpleForceField
            The force field used to evaluate energies / gradients.
        n_steps : int, default 100
            Number of (steepest-descent + MC) iterations.
        step_size : float, default 0.01
            Step size (Å) for steepest descent.
        temperature : float, default 300.0
            Temperature (K) used by the Metropolis acceptance rule.
        mc_step_size : float, default 0.05
            Per-atom perturbation scale (Å) for the MC step.
        seed : int, optional
            Seed for the internal RNG.  Defaults to ``None`` (random).
        """
        if not isinstance(force_field, SimpleForceField):
            raise TypeError(
                "force_field must be a SimpleForceField instance; "
                f"got {type(force_field).__name__}"
            )
        if int(n_steps) < 0:
            raise ValueError(f"n_steps must be >= 0, got {n_steps!r}")
        if float(step_size) <= 0.0:
            raise ValueError(f"step_size must be > 0, got {step_size!r}")
        if float(temperature) <= 0.0:
            raise ValueError(f"temperature must be > 0, got {temperature!r}")
        self.force_field: SimpleForceField = force_field
        self.n_steps: int = int(n_steps)
        self.step_size: float = float(step_size)
        self.temperature: float = float(temperature)
        self.mc_step_size: float = float(mc_step_size)
        self._rng: random.Random = random.Random(seed)

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------
    def relax(
        self,
        coords: np.ndarray,
        atom_types: List[str],
    ) -> np.ndarray:
        """Run :attr:`n_steps` of (SD + MC) relaxation and return coords.

        The returned array is a fresh copy; the input is not modified.

        Parameters
        ----------
        coords : np.ndarray
            Shape ``(N, 3)``.
        atom_types : List[str]
            Atomic symbols (length ``N``).

        Returns
        -------
        np.ndarray
            Shape ``(N, 3)``.
        """
        x = np.asarray(coords, dtype=np.float64).copy()
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError(f"coords must have shape (N, 3); got {coords.shape!r}")
        for _ in range(self.n_steps):
            grad = self.force_field.gradient(x, atom_types)
            x = x - self.step_size * grad
            x, _accepted = self.monte_carlo_step(x, atom_types)
        return x

    def monte_carlo_step(
        self,
        coords: np.ndarray,
        atom_types: List[str],
    ) -> Tuple[np.ndarray, bool]:
        """One Metropolis Monte Carlo step.

        Proposes a uniform-random perturbation of every atom
        (``[ -mc_step_size, +mc_step_size ]`` A per axis) and accepts
        with probability ``min(1, exp(-dE / kT))``.

        Parameters
        ----------
        coords : np.ndarray
            Shape ``(N, 3)``.
        atom_types : List[str]
            Atomic symbols (length ``N``).

        Returns
        -------
        (new_coords, accepted) : Tuple[np.ndarray, bool]
            The new coords (a fresh copy) and whether the proposal was
            accepted.  The input ``coords`` is not modified.
        """
        x = np.asarray(coords, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != 3:
            raise ValueError(f"coords must have shape (N, 3); got {coords.shape!r}")
        if x.shape[0] != len(atom_types):
            raise ValueError(
                f"coords has {x.shape[0]} rows but atom_types has "
                f"{len(atom_types)} entries"
            )
        if x.shape[0] == 0:
            return x.copy(), False

        e_old = self.force_field.energy(x, atom_types)
        delta = self.mc_step_size
        rng = self._rng
        perturb = np.asarray(
            [
                [rng.uniform(-delta, delta) for _ in range(3)]
                for _ in range(int(x.shape[0]))
            ],
            dtype=np.float64,
        )
        x_new = x + perturb
        e_new = self.force_field.energy(x_new, atom_types)
        dE = e_new - e_old
        accept: bool
        if dE <= 0.0:
            accept = True
        else:
            beta = KBOLTZMANN * float(self.temperature)
            if beta <= 0.0:
                accept = False
            else:
                prob = math.exp(-dE / beta)
                accept = bool(rng.random() < prob)
        if accept:
            return x_new, True
        return x.copy(), False

    def energy_after_relax(
        self,
        coords: np.ndarray,
        atom_types: List[str],
    ) -> float:
        """Run the relaxation and return the final energy (kcal/mol)."""
        relaxed = self.relax(coords, atom_types)
        return float(self.force_field.energy(relaxed, atom_types))

    def __repr__(self) -> str:
        return (
            f"MDRelaxation(ff={self.force_field.method!r}, "
            f"n_steps={self.n_steps}, step_size={self.step_size}, "
            f"T={self.temperature}K)"
        )


# ---------------------------------------------------------------------------
# GeometryValidator
# ---------------------------------------------------------------------------
class GeometryValidator:
    """Validate 3D geometry — bond-length sanity + close-contact report.

    The validator uses three checks, applied in priority order:

    1. **Absolute minimum** — any pair with distance below
       :attr:`absolute_min` (default 0.80 A) is flagged as a
       ``"close_contact"``.  This catches impossibly-close pairs
       (e.g. two H atoms at 0.5 A, which look "bonded" by covalent
       radius but cannot form a real H-H bond at that distance).
    2. **Bond detection** — pairs with distance at most
       :attr:`bond_threshold` (default 2.0 A) are treated as bonds;
       those with distance outside
       ``[bond_lower * r0, bond_upper * r0]`` are flagged.
    3. **Non-bonded close contact** — pairs with distance at most
       :attr:`close_contact` (default 1.20 A) and not classified as
       bonds are flagged.
    """

    def __init__(
        self,
        bond_threshold: float = DEFAULT_BOND_THRESHOLD,
        bond_lower: float = DEFAULT_BOND_LOWER,
        bond_upper: float = DEFAULT_BOND_UPPER,
        close_contact: float = DEFAULT_CLOSE_CONTACT,
        absolute_min: float = DEFAULT_ABSOLUTE_MIN,
    ) -> None:
        """Initialise the validator.

        Parameters
        ----------
        bond_threshold : float, default 2.0
            Maximum distance (A) for two atoms to be classified as
            bonded.
        bond_lower : float, default 0.70
            Lower bound on the bond-length ratio
            ``d_ij / (cov_rad_i + cov_rad_j)``.
        bond_upper : float, default 1.60
            Upper bound on the bond-length ratio.
        close_contact : float, default 1.20
            Threshold (A) for non-bonded close contacts.
        absolute_min : float, default 0.80
            Absolute minimum distance (A) — any pair below this is
            flagged as ``"close_contact"`` regardless of bonding.
        """
        self.bond_threshold = float(bond_threshold)
        self.bond_lower = float(bond_lower)
        self.bond_upper = float(bond_upper)
        self.close_contact = float(close_contact)
        self.absolute_min = float(absolute_min)

    def is_reasonable_geometry(
        self,
        coords: np.ndarray,
        atom_types: List[str],
        bond_threshold: float = DEFAULT_BOND_THRESHOLD,
    ) -> bool:
        """Return ``True`` iff every pair passes the geometry checks.

        "Reasonable" means: no pair is flagged by
        :meth:`report_bad_contacts`.  The ``bond_threshold`` argument
        overrides :attr:`bond_threshold` for this single call — it is
        exposed as a keyword for backward compatibility with the
        task brief's signature.

        Parameters
        ----------
        coords : np.ndarray
            Shape ``(N, 3)``.
        atom_types : List[str]
            Atomic symbols (length ``N``).
        bond_threshold : float, default 2.0
            Override the bond-detection threshold for this call.

        Returns
        -------
        bool
            ``True`` iff no violations are reported.
        """
        return not self.report_bad_contacts(
            coords, atom_types, bond_threshold=bond_threshold
        )

    def report_bad_contacts(
        self,
        coords: np.ndarray,
        atom_types: List[str],
        bond_threshold: Optional[float] = None,
    ) -> List[Tuple[int, int, float, str]]:
        """Return a list of geometry violations.

        Each entry is ``(i, j, distance_ij, reason)`` where ``reason``
        is one of:

        * ``"bond_too_short"``  — ``d_ij < bond_lower * r0``
        * ``"bond_too_long"``   — ``d_ij > bond_upper * r0``
        * ``"close_contact"``   — ``d_ij < close_contact`` (non-bonded)
                                  or ``d_ij < absolute_min`` (any pair)

        Parameters
        ----------
        coords : np.ndarray
            Shape ``(N, 3)``.
        atom_types : List[str]
            Atomic symbols.
        bond_threshold : float, optional
            Override the bond-detection threshold for this call.

        Returns
        -------
        List[Tuple[int, int, float, str]]
        """
        c = np.asarray(coords, dtype=np.float64)
        if c.ndim != 2 or c.shape[1] != 3:
            raise ValueError(f"coords must have shape (N, 3); got {coords.shape!r}")
        n = int(c.shape[0])
        if n == 0:
            return []
        if n != len(atom_types):
            raise ValueError(
                f"coords has {n} rows but atom_types has {len(atom_types)} entries"
            )

        thresh = float(
            self.bond_threshold if bond_threshold is None else bond_threshold
        )
        radii = np.asarray([_cov_radius(t) for t in atom_types], dtype=np.float64)
        dmat = _pair_distance_matrix(c)

        out: List[Tuple[int, int, float, str]] = []
        for i in range(n):
            for j in range(i + 1, n):
                d = float(dmat[i, j])
                r0 = float(radii[i] + radii[j])

                # Priority 1: absolute minimum — catches impossibly
                # close pairs regardless of bonding status.
                if d < self.absolute_min:
                    out.append((i, j, d, "close_contact"))
                    continue

                # Priority 2: bonded pair
                if d <= thresh:
                    if d < self.bond_lower * r0:
                        out.append((i, j, d, "bond_too_short"))
                    elif d > self.bond_upper * r0:
                        out.append((i, j, d, "bond_too_long"))
                    continue

                # Priority 3: non-bonded close contact
                if d < self.close_contact:
                    out.append((i, j, d, "close_contact"))
        return out

    def bad_contact_count(
        self,
        coords: np.ndarray,
        atom_types: List[str],
    ) -> int:
        """Return the number of bad contacts / bond violations."""
        return len(self.report_bad_contacts(coords, atom_types))

    def __repr__(self) -> str:
        return (
            f"GeometryValidator(bond_threshold={self.bond_threshold}, "
            f"close_contact={self.close_contact}, "
            f"absolute_min={self.absolute_min})"
        )


# ---------------------------------------------------------------------------
# Registration hook
# ---------------------------------------------------------------------------
def register_md_relaxation() -> int:
    """Register the MD relaxation stub and return the registered count.

    The registration is **symbolic**: we instantiate one default
    :class:`MDRelaxation` (backed by the default
    :class:`SimpleForceField`) and stash it on the function as an
    attribute so callers can introspect it via
    ``register_md_relaxation._default``.  The hook always returns
    ``1`` — there is one relaxation rule to register.

    Returns
    -------
    int
        Always ``1``.
    """
    if not hasattr(register_md_relaxation, "_default"):
        ff = SimpleForceField()
        relax = MDRelaxation(ff)
        register_md_relaxation._default = relax  # type: ignore[attr-defined]
    return 1


__all__ = [
    "DEFAULT_METHOD",
    "DEFAULT_BOND_K",
    "DEFAULT_ANGLE_K",
    "DEFAULT_NB_EPSILON",
    "DEFAULT_BOND_TOL",
    "DEFAULT_GRADIENT_STEP",
    "DEFAULT_RELAX_STEPS",
    "DEFAULT_RELAX_STEP_SIZE",
    "DEFAULT_MC_STEP_SIZE",
    "DEFAULT_TEMPERATURE_K",
    "DEFAULT_BOND_THRESHOLD",
    "DEFAULT_BOND_LOWER",
    "DEFAULT_BOND_UPPER",
    "DEFAULT_CLOSE_CONTACT",
    "DEFAULT_ABSOLUTE_MIN",
    "KBOLTZMANN",
    "SimpleForceField",
    "MDRelaxation",
    "GeometryValidator",
    "register_md_relaxation",
]
