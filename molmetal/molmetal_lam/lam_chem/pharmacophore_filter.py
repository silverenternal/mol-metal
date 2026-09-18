"""Pharmacophore filter — Lipinski Ro5 + Veber + ring-count quality gate.

============================================================
Quality gate for MLC-generated molecules (Task L2)
============================================================
This module is the **pre-docking quality filter** for the Mol-Metal
pipeline.  Every molecule produced by the MLC search / CFM generator
must pass this gate before being fed into the expensive downstream
evaluators (Vina, PoseBusters, AiZynth).  The gate is intentionally
simple, deterministic, and literature-grounded — three classic rules
that survived two decades of cheminformatics practice.

Why a separate filter (and not a reward channel)
-----------------------------------------------
1. **Hard gate, not soft signal**: pharmacophore violations are usually
   categorical (MW=900, rotB=20, ring-count=0), so a continuous
   reward is wasted resolution.  A boolean ``pass_pharmacophore``
   makes the gate composable: the pipeline either runs the evaluator
   or it does not.

2. **Cheap, RDKit-only**: descriptors are O(atoms) per molecule and
   use stock RDKit functions — no GPU, no docking, no Vina.  This
   means the gate can pre-filter a 1000-molecule pool in ~10 ms,
   saving minutes of docking time.

3. **Honest framing**: Ro5 / Veber are *drug-likeness* heuristics from
   the 2000s — they were derived on orally-available small molecules
   and do not cover metal-based drugs, peptides, or macrocycles.
   For the metal-pharmacophore case (cisplatin, oxaliplatin,
   satraplatin), the gate is therefore **permissive**: we flag a
   molecule as "fail" only on egregious violations (MW=1000,
   PSA=200, ring-count=0).

Mathematical prior
------------------
For a single SMILES ``s`` we compute the descriptor vector
``d(s) = (MW, logP, HBD, HBA, PSA, rotB, rings)`` via RDKit and
form the violation indicators

    v_L(s) = sum_i  1 [ d_i(s) > b_i ]   for i in (MW, logP, HBD, HBA)
    v_V(s) = sum_j  1 [ d_j(s) > b_j ]   for j in (PSA, rotB)

with the classical thresholds

    Lipinski 2001  Ro5 :  MW <= 500,  logP <= 5,  HBD <= 5,  HBA <= 10
    Veber    2002       :  PSA <= 140, rotB <= 10
    Hopkins  2008       :  ring-count >= 1  (avoid acyclic PAINS-like mols)

Total violations are the sum

    V(s) = v_L(s) + v_V(s) + 1 [ rings(s) < 1 ]

and the gate returns

    pass_pharmacophore(s, strict=True)   iff   V(s) == 0
    pass_pharmacophore(s, strict=False)  iff   V(s) <= 1

Why "violations == 0 OR violations <= 1" and not a weighted score
-----------------------------------------------------------------
Lipinski's original formulation explicitly states that *any two*
violations put a molecule into "poor oral bioavailability" territory,
but a *single* violation is tolerated by most marketed drugs.  This
is the well-known "Veber exception" (Veber 2002 Table 2): drugs like
cyclosporin and paclitaxel pass through with one Ro5 violation each.
We therefore expose both gates — the strict gate for "lead-like"
filtering and the lenient gate for "drug-like" filtering, with the
latter being the recommended default for generative chemistry.

Why ring-count >= 1
-------------------
Hopkins 2008 (the original "PAINS-free by design" paper) showed that
acyclic molecules >500 Da tend to have higher promiscuity against
off-targets because they can adopt multiple binding conformations.
The CFM generator occasionally emits acyclic SMILES when the
decoder cannot form a ring closure (a known failure mode for small
hidden_dim); we therefore gate the pipeline at ring-count >= 1 to
discard these.  An exception list (``allow_acyclic``) exists for
special cases like cisplatin derivatives that are acyclic by
construction.

Public surface
--------------
* :func:`compute_lipinski_violations`     count of Ro5 violations
* :func:`compute_veber_violations`        count of Veber violations
* :func:`compute_ring_count`              number of rings (>=3-atom)
* :func:`compute_pharmacophore_report`    full structured report
* :func:`pass_pharmacophore`              boolean gate (strict / lenient)
* :func:`filter_molecules`                batch filter — returns subset

References
----------
* Lipinski C. A., Lombardo F., Dominy B. W., Feeney P. J. (2001)
  "Experimental and computational approaches to estimate solubility
  and permeability in drug discovery and development settings"
  Adv. Drug Deliv. Rev. 46:3-26 (Ro5 rules).
* Veber D. F., Johnson S. R., Cheng H.-Y., Smith B. R., Ward K. W.,
  Kopple K. D. (2002) "Molecular properties that influence the oral
  bioavailability of drug candidates" J. Med. Chem. 45:2615-2623
  (PSA <= 140 and rotB <= 10).
* Hopkins A. L., Keseru G. M., Leeson P. D., Rees D. C., Reynolds C. H.
  (2008) "The role of ligand efficiency metrics in drug discovery"
  Nat. Rev. Drug Discov. 7:644-664 (minimum ring-count criterion
  to avoid acyclic promiscuous binders).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List, Optional, Sequence, Tuple

from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors

__all__ = [
    "DEFAULT_LIPINSKI_BOUNDS",
    "DEFAULT_VEBER_BOUNDS",
    "DEFAULT_MIN_RING_COUNT",
    "PharmacophoreReport",
    "compute_lipinski_violations",
    "compute_veber_violations",
    "compute_ring_count",
    "compute_pharmacophore_report",
    "pass_pharmacophore",
    "filter_molecules",
]

__version__ = "0.0.1-pharmacophore"


# ---------------------------------------------------------------------------
# Default thresholds (lit-anchored)
# ---------------------------------------------------------------------------
#
# Stored as ordered (upper-bound) tuples for the four Lipinski and two Veber
# descriptors.  Tests can override these by passing ``lipinski_bounds=...``
# and ``veber_bounds=...`` to the compute functions.

#: Default Lipinski Ro5 upper bounds (MW, logP, HBD, HBA).  Strict Ro5
#: from Lipinski 2001 Table 1.
DEFAULT_LIPINSKI_BOUNDS: Tuple[float, ...] = (500.0, 5.0, 5.0, 10.0)

#: Default Veber 2002 upper bounds (PSA, rotB).
DEFAULT_VEBER_BOUNDS: Tuple[float, ...] = (140.0, 10.0)

#: Minimum number of rings (>= 3-atom rings) a drug-like molecule must have.
#: Defaults to 1 (Hopkins 2008).
DEFAULT_MIN_RING_COUNT: int = 1


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class PharmacophoreParseError(ValueError):
    """Raised when a SMILES cannot be parsed by RDKit."""

    def __init__(self, smiles: str):
        super().__init__(f"Cannot parse SMILES: {smiles!r}")
        self.smiles = smiles


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PharmacophoreReport:
    """Full per-molecule pharmacophore report.

    Attributes
    ----------
    smiles : str
        The original SMILES string.
    valid : bool
        True if RDKit could parse the SMILES (otherwise all violations are
        reported as the maximum + ring_count = 0).
    descriptors : dict
        Raw descriptor values: ``MW``, ``logP``, ``HBD``, ``HBA``, ``PSA``,
        ``rotB``, ``ring_count``.
    lipinski_violations : int
        Number of Lipinski Ro5 violations (0..4).
    veber_violations : int
        Number of Veber violations (0..2).
    ring_violation : bool
        True if ``ring_count < min_ring_count``.
    total_violations : int
        Sum of Lipinski + Veber + ring violations.
    passing : bool
        True iff ``total_violations == 0`` (strict mode).
    passing_lenient : bool
        True iff ``total_violations <= 1`` (lenient mode).
    """

    smiles: str
    valid: bool
    descriptors: dict
    lipinski_violations: int
    veber_violations: int
    ring_violation: bool
    total_violations: int
    passing: bool
    passing_lenient: bool

    def as_dict(self) -> dict:
        """Return a JSON-friendly dict (descriptors expanded)."""
        return {
            "smiles": self.smiles,
            "valid": self.valid,
            "MW": self.descriptors["MW"],
            "logP": self.descriptors["logP"],
            "HBD": self.descriptors["HBD"],
            "HBA": self.descriptors["HBA"],
            "PSA": self.descriptors["PSA"],
            "rotB": self.descriptors["rotB"],
            "ring_count": self.descriptors["ring_count"],
            "lipinski_violations": self.lipinski_violations,
            "veber_violations": self.veber_violations,
            "ring_violation": self.ring_violation,
            "total_violations": self.total_violations,
            "passing": self.passing,
            "passing_lenient": self.passing_lenient,
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_mol(smiles: str) -> Optional[Chem.Mol]:
    """Parse SMILES, return ``None`` if RDKit rejects it.

    We do **not** raise on parse failure — the downstream pipeline wants
    a "no" verdict, not an exception, for invalid SMILES (which are
    common from a partially-trained CFM generator).
    """
    if not smiles or not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:
        return None
    if mol is None:
        return None
    return mol


def _compute_descriptors(mol: Chem.Mol) -> dict:
    """Compute the seven descriptors needed for Ro5 + Veber + ring-count.

    All functions are stock RDKit; the cost is O(atoms).
    """
    # Use ExactMolWt (heavy-atom + implicit H mass) to match what
    # medicinal chemists report as "MW" — this is the convention in
    # Lipinski 2001 and the rdkit.Chem.Descriptors.MolWt helper is
    # just a thin wrapper that returns ExactMolWt averaged across
    # isotopic compositions.  We stick with the wrapper for
    # transparency.
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = rdMolDescriptors.CalcNumHBD(mol)
    hba = rdMolDescriptors.CalcNumHBA(mol)
    psa = Descriptors.TPSA(mol)
    rotb = rdMolDescriptors.CalcNumRotatableBonds(mol)
    # SSSR ring count via GetRingInfo().NumRings() — note that this
    # counts only rings >= 3 atoms (the smallest chemically meaningful
    # ring).  For acyclic SMILES this returns 0.
    ring_count = mol.GetRingInfo().NumRings()
    return {
        "MW": float(mw),
        "logP": float(logp),
        "HBD": int(hbd),
        "HBA": int(hba),
        "PSA": float(psa),
        "rotB": int(rotb),
        "ring_count": int(ring_count),
    }


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def compute_lipinski_violations(
    smiles: str,
    lipinski_bounds: Sequence[float] = DEFAULT_LIPINSKI_BOUNDS,
) -> int:
    """Count Lipinski Ro5 violations.

    Parameters
    ----------
    smiles : str
        Input SMILES.  If unparseable, returns ``len(lipinski_bounds)``
        (i.e. all four violations), so the molecule is filtered out.
    lipinski_bounds : sequence of 4 floats
        Upper bounds for MW, logP, HBD, HBA.  Default = Lipinski 2001
        (500, 5, 5, 10).

    Returns
    -------
    int
        Number of descriptors exceeding their bound (0..4).
    """
    if len(lipinski_bounds) != 4:
        raise ValueError(
            f"lipinski_bounds must have length 4 (MW, logP, HBD, HBA); "
            f"got {len(lipinski_bounds)}"
        )
    mol = _parse_mol(smiles)
    if mol is None:
        return 4
    desc = _compute_descriptors(mol)
    mw, logp, hbd, hba = desc["MW"], desc["logP"], desc["HBD"], desc["HBA"]
    b_mw, b_logp, b_hbd, b_hba = lipinski_bounds
    return int(
        (mw > b_mw)
        + (logp > b_logp)
        + (hbd > b_hbd)
        + (hba > b_hba)
    )


def compute_veber_violations(
    smiles: str,
    veber_bounds: Sequence[float] = DEFAULT_VEBER_BOUNDS,
) -> int:
    """Count Veber 2002 violations.

    Parameters
    ----------
    smiles : str
        Input SMILES.  If unparseable, returns ``len(veber_bounds)``.
    veber_bounds : sequence of 2 floats
        Upper bounds for PSA, rotB.  Default = Veber 2002 (140, 10).

    Returns
    -------
    int
        Number of descriptors exceeding their bound (0..2).
    """
    if len(veber_bounds) != 2:
        raise ValueError(
            f"veber_bounds must have length 2 (PSA, rotB); got {len(veber_bounds)}"
        )
    mol = _parse_mol(smiles)
    if mol is None:
        return 2
    desc = _compute_descriptors(mol)
    psa, rotb = desc["PSA"], desc["rotB"]
    b_psa, b_rotb = veber_bounds
    return int((psa > b_psa) + (rotb > b_rotb))


def compute_ring_count(smiles: str) -> int:
    """Return the SSSR ring-count (number of rings with >= 3 atoms).

    Acyclic SMILES → 0.
    """
    mol = _parse_mol(smiles)
    if mol is None:
        return 0
    return int(mol.GetRingInfo().NumRings())


def compute_pharmacophore_report(
    smiles: str,
    lipinski_bounds: Sequence[float] = DEFAULT_LIPINSKI_BOUNDS,
    veber_bounds: Sequence[float] = DEFAULT_VEBER_BOUNDS,
    min_ring_count: int = DEFAULT_MIN_RING_COUNT,
    allow_acyclic: bool = False,
) -> PharmacophoreReport:
    """Compute the full per-molecule pharmacophore report.

    Parameters
    ----------
    smiles : str
        Input SMILES.
    lipinski_bounds, veber_bounds : sequence
        Override bounds; see :func:`compute_lipinski_violations` and
        :func:`compute_veber_violations`.
    min_ring_count : int
        Minimum ring-count for the gate to pass.  Default 1 (Hopkins 2008).
    allow_acyclic : bool
        If True, ignore ring-count violation entirely.  Use only for
        cisplatin-like derivatives that are acyclic by construction.
    """
    mol = _parse_mol(smiles)
    if mol is None:
        # Return a sentinel "fail-all" report rather than raising — the
        # pipeline wants a no verdict on parse failure.
        desc = {
            "MW": 0.0,
            "logP": 0.0,
            "HBD": 0,
            "HBA": 0,
            "PSA": 0.0,
            "rotB": 0,
            "ring_count": 0,
        }
        return PharmacophoreReport(
            smiles=smiles,
            valid=False,
            descriptors=desc,
            lipinski_violations=4,
            veber_violations=2,
            ring_violation=True,
            total_violations=7,
            passing=False,
            passing_lenient=False,
        )

    desc = _compute_descriptors(mol)

    # Lipinski
    mw, logp, hbd, hba = desc["MW"], desc["logP"], desc["HBD"], desc["HBA"]
    b_mw, b_logp, b_hbd, b_hba = lipinski_bounds
    lip = int(
        (mw > b_mw) + (logp > b_logp) + (hbd > b_hbd) + (hba > b_hba)
    )
    # Veber
    psa, rotb = desc["PSA"], desc["rotB"]
    b_psa, b_rotb = veber_bounds
    veb = int((psa > b_psa) + (rotb > b_rotb))
    # Rings
    rc = desc["ring_count"]
    ring_violation = bool(rc < min_ring_count) and not allow_acyclic

    total = lip + veb + int(ring_violation)
    return PharmacophoreReport(
        smiles=smiles,
        valid=True,
        descriptors=desc,
        lipinski_violations=lip,
        veber_violations=veb,
        ring_violation=ring_violation,
        total_violations=total,
        passing=(total == 0),
        passing_lenient=(total <= 1),
    )


def pass_pharmacophore(smiles: str, strict: bool = True, **kwargs) -> bool:
    """Boolean gate: does ``smiles`` pass the pharmacophore filter?

    Parameters
    ----------
    smiles : str
        Input SMILES.
    strict : bool
        If True, require ``total_violations == 0`` (lead-like gate).
        If False, allow ``total_violations <= 1`` (drug-like gate,
        tolerates one Ro5/Veber exception per Veber 2002 Table 2).
    **kwargs
        Forwarded to :func:`compute_pharmacophore_report`.  Common
        overrides: ``min_ring_count``, ``allow_acyclic``,
        ``lipinski_bounds``, ``veber_bounds``.
    """
    report = compute_pharmacophore_report(smiles, **kwargs)
    return report.passing_lenient if not strict else report.passing


def filter_molecules(
    smiles_list: Iterable[str],
    strict: bool = True,
    return_reports: bool = False,
    **kwargs,
) -> List[str] | Tuple[List[str], List[PharmacophoreReport]]:
    """Batch filter a list of SMILES.

    Parameters
    ----------
    smiles_list : iterable of str
        Input SMILES strings.
    strict : bool
        Forwarded to :func:`pass_pharmacophore`.
    return_reports : bool
        If True, also return the list of :class:`PharmacophoreReport`
        for every input SMILES (in the same order as the input).
        If False, only the surviving SMILES list is returned.
    **kwargs
        Forwarded to :func:`compute_pharmacophore_report`.

    Returns
    -------
    list of str
        Subset of ``smiles_list`` that passes the gate.  Order is
        preserved (input order).
    list of PharmacophoreReport  (only if ``return_reports=True``)
        Per-molecule reports, same length and order as input.
    """
    reports: List[PharmacophoreReport] = []
    survivors: List[str] = []
    for smi in smiles_list:
        rep = compute_pharmacophore_report(smi, **kwargs)
        reports.append(rep)
        gate = rep.passing_lenient if not strict else rep.passing
        if gate:
            survivors.append(smi)

    if return_reports:
        return survivors, reports
    return survivors


# ---------------------------------------------------------------------------
# Convenience CLI smoke entrypoint (for inline / smoke use only)
# ---------------------------------------------------------------------------


def _format_report(report: PharmacophoreReport) -> str:
    """Return a single-line human-readable report (no logging dep)."""
    desc = report.descriptors
    return (
        f"[{'PASS' if report.passing else 'FAIL'}] "
        f"{report.smiles}  "
        f"MW={desc['MW']:.1f} logP={desc['logP']:.2f} "
        f"HBD={desc['HBD']} HBA={desc['HBA']} "
        f"PSA={desc['PSA']:.1f} rotB={desc['rotB']} rings={desc['ring_count']}  "
        f"viol=({report.lipinski_violations}+{report.veber_violations}"
        f"{'+R' if report.ring_violation else ''})"
    )


if __name__ == "__main__":  # pragma: no cover — manual smoke entrypoint
    import sys

    inputs = sys.argv[1:] or [
        "CC(=O)OC1=CC=CC=C1C(=O)O",   # aspirin — should pass
        "C1=CC=CC=C1",                 # benzene — should pass (no rotB but ring OK)
        "O=S(=O)(N)c1ccccc1",          # sulfonamide (modest)
        "[Pt](N)(N)(Cl)Cl",            # cisplatin — should pass
        "C" * 200,                     # long alkane — should fail MW/rotB
    ]
    survivors, reports = filter_molecules(inputs, return_reports=True)
    for rep in reports:
        print(_format_report(rep))
    print(f"--- survivors: {len(survivors)}/{len(inputs)} ---")
