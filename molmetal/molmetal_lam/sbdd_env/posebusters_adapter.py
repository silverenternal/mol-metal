"""PoseBusters adapter — physical/chemical validity checks.

================================================================
What this replaces
================================================================
The original ``molmetal/molmetal_lam/scripts/baselines.py`` had no
molecular validity check at all — generated SMILES were assumed
valid, and 3D conformers were assumed sensible. This module wraps
``posebusters`` (Buttenschoen et al., *Chem. Sci.* 2024) to provide
a paper-grade validity score on every generated molecule.

Modes
-----
* ``PoseBusters("mol")``    — chemistry + geometry only (no protein)
* ``PoseBusters("dock")``   — protein-ligand clash checks (needs pdb)
* ``PoseBusters("redock")`` — same as dock + RMSD to reference

For our use case (de novo 3D molecule generation, no reference pose)
the default is ``mol``.

Public API
----------
* :class:`PoseBustersAdapter` — implements the validity port
* :func:`validate_mol(smiles)` — one-shot check returning a dict
* :func:`pass_rate(smiles_list, full_report=False)` — bulk pass-rate

Reference
---------
Buttenschoen, Morris & Deane, *PoseBusters: AI-based docking methods
fail to generate physically valid poses or generalise to novel
sequences*, Chem. Sci. 2024, 15, 3130-3139.
doi:10.1039/D3SC04185A
"""
from __future__ import annotations

import logging
import os
import warnings
from dataclasses import dataclass
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

# Suppress RDKit + pandas warnings that PoseBusters triggers routinely
warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


def _have_posebusters() -> bool:
    try:
        from posebusters import PoseBusters  # noqa: F401
        return True
    except Exception:
        return False


@dataclass(frozen=True)
class ValidityReport:
    """Result of validating a single molecule."""

    smiles: str
    passed: bool                         # True iff ALL checks passed
    n_checks: int                        # total checks run
    n_passed: int                        # # checks that passed
    pass_rate: float                     # n_passed / n_checks
    failed_checks: tuple                 # names of failing checks
    details: dict                        # per-check pass/fail dict

    def to_dict(self) -> dict:
        return {
            "smiles": self.smiles,
            "passed": self.passed,
            "n_checks": self.n_checks,
            "n_passed": self.n_passed,
            "pass_rate": self.pass_rate,
            "failed_checks": list(self.failed_checks),
        }


@dataclass(frozen=True)
class PBResult:
    """Result of a PoseBusters run (mol or dock mode).

    Wraps a :class:`ValidityReport` plus the mode-specific extras:

    * ``mode``            — ``"mol"`` or ``"dock"`` (or ``"redock"``)
    * ``receptor_pdb``    — protein file path for ``dock`` mode; ``None`` for ``mol``
    * ``n_checks``        — chemistry + (in ``dock``) clash check count
    * ``extra_checks``    — names of checks that were not present in
      :class:`ValidityReport`'s default 14 chemistry checks (i.e. protein-aware
      clash / minimum-distance checks for ``dock`` mode)
    """

    report: ValidityReport
    mode: str
    receptor_pdb: Optional[str] = None
    extra_checks: tuple = ()            # clash / distance checks added in dock mode

    @property
    def smiles(self) -> str:
        return self.report.smiles

    @property
    def passed(self) -> bool:
        return self.report.passed

    @property
    def n_checks(self) -> int:
        return self.report.n_checks

    @property
    def n_passed(self) -> int:
        return self.report.n_passed

    @property
    def pass_rate(self) -> float:
        return self.report.pass_rate

    @property
    def failed_checks(self) -> tuple:
        return self.report.failed_checks

    @property
    def details(self) -> dict:
        return self.report.details

    def to_dict(self) -> dict:
        d = self.report.to_dict()
        d.update({
            "mode": self.mode,
            "receptor_pdb": self.receptor_pdb,
            "extra_checks": list(self.extra_checks),
        })
        return d


class PoseBustersAdapter:
    """Wraps ``posebusters`` for in-pipeline molecule validity checks.

    Parameters
    ----------
    mode : str
        PoseBusters configuration mode. Default ``"mol"`` (chemistry +
        geometry, no protein). Other modes: ``"dock"``, ``"redock"``.
    full_report : bool
        If True, ``validate_mol`` returns the full DataFrame; otherwise
        a compact :class:`ValidityReport`.
    """

    @property
    def name(self) -> str:
        return f"PoseBusters_{self._mode}_v1"

    def __init__(self, mode: str = "mol", full_report: bool = False) -> None:
        if not _have_posebusters():
            raise RuntimeError(
                "posebusters not installed. Run: "
                "`source .venv/bin/activate && uv pip install posebusters`"
            )
        from posebusters import PoseBusters  # type: ignore
        self._pb = PoseBusters(config=mode)
        self._mode = str(mode)
        self._full_report = bool(full_report)
        self._version = _pb_version()

    # ---------------------------------------------------------- Public API
    def validate_mol(self, smiles: str) -> ValidityReport:
        """Validate a single SMILES (3D-embedded internally)."""
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore

        if not smiles or not isinstance(smiles, str):
            return ValidityReport(
                smiles=smiles or "", passed=False, n_checks=0, n_passed=0,
                pass_rate=0.0, failed_checks=("empty_smiles",),
                details={"empty_smiles": False},
            )
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            return ValidityReport(
                smiles=smiles, passed=False, n_checks=0, n_passed=0,
                pass_rate=0.0, failed_checks=("smiles_parse",),
                details={"smiles_parse": False},
            )
        mol = Chem.AddHs(mol)
        # 3D embedding: ETKDGv3, then MMFF94OptimizeMolecule so the
        # conformer reaches PoseBusters' MMFF94 reference window.  See
        # reports/lambda_vs_sbdd_paper_numbers.md §3.1 — ETKDGv3+UFF
        # leaves triazoles 1° off the MMFF94 minimum, which trips
        # number_outlier_angles and number_short_outlier_bonds.  If
        # MMFF94 cannot parameterise the molecule (e.g. charged metals
        # / very large systems), fall back to UFF so the geometry checks
        # still run on something.
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        embed_status = AllChem.EmbedMolecule(mol, params)
        if embed_status == -1:
            # Last resort: 2D embed → no 3D validity checks possible.
            Chem.RemoveHs(mol)
            return ValidityReport(
                smiles=smiles, passed=False, n_checks=0, n_passed=0,
                pass_rate=0.0, failed_checks=("embed_3d",),
                details={"embed_3d": False},
            )
        # Prefer MMFF94 over UFF: MMFF has better bonded/non-bonded
        # parameters for drug-like organics (triazoles, aromatics,
        # heterocycles).  MMFF94OptimizeMolecule returns 0 on success
        # and a non-zero code on failure (e.g. unsupported atom types
        # like Cu / charged metals / very large molecules); in that
        # case we fall back to UFF so PoseBusters geometry checks
        # still run on something.
        try:
            mmff_status = AllChem.MMFF94OptimizeMolecule(mol, maxIters=200)
        except Exception:
            mmff_status = -1
        if mmff_status != 0:
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=200)
            except Exception:
                pass
        Chem.RemoveHs(mol)
        df = self._pb.bust(mol, full_report=True)
        return self._df_to_report(smiles, df)

    def validate_docked(
        self,
        smiles: str,
        receptor_pdb: str,
    ) -> PBResult:
        """Validate a docked pose against an explicit receptor PDB.

        Compared with :meth:`validate_mol` (chemistry + geometry only), the
        ``dock`` mode of PoseBusters additionally runs protein-aware checks:

        * ``protein-ligand_maximum_distance``  — the ligand stays close to
          the protein binding site (no detachment)
        * ``minimum_distance_to_protein``      — no severe intermolecular
          clash with protein atoms
        * ``minimum_distance_to_*_cofactors`` — distance to organic /
          inorganic / water cofactors
        * ``volume_overlap_with_protein``      — steric overlap with the
          receptor (a hard clash)
        * ``not_too_far_away_*_cofactors``     — cofactor-distance window

        Parameters
        ----------
        smiles : str
            Ligand SMILES (3D-embedded internally; same path as
            :meth:`validate_mol`).
        receptor_pdb : str
            Absolute path to a receptor PDB file.  Must exist; raises
            :class:`FileNotFoundError` otherwise.

        Returns
        -------
        :class:`PBResult`
            Wraps a :class:`ValidityReport` with mode = ``"dock"``, the
            receptor path, and an ``extra_checks`` tuple of protein-aware
            check names that are NOT present in the ``mol`` mode output.
        """
        from pathlib import Path
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore

        if not receptor_pdb:
            raise ValueError("validate_docked requires receptor_pdb (a path to a receptor PDB)")
        if not Path(receptor_pdb).is_file():
            raise FileNotFoundError(f"Receptor PDB not found: {receptor_pdb}")

        # Reuse the same embed + MMFF94/UFF path as validate_mol so dock
        # and mol runs are comparable on the same 3D conformer.
        if not smiles or not isinstance(smiles, str):
            empty = ValidityReport(
                smiles=smiles or "", passed=False, n_checks=0, n_passed=0,
                pass_rate=0.0, failed_checks=("empty_smiles",),
                details={"empty_smiles": False},
            )
            return PBResult(report=empty, mode=self._mode, receptor_pdb=receptor_pdb,
                            extra_checks=())
        mol = Chem.MolFromSmiles(smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            bad = ValidityReport(
                smiles=smiles, passed=False, n_checks=0, n_passed=0,
                pass_rate=0.0, failed_checks=("smiles_parse",),
                details={"smiles_parse": False},
            )
            return PBResult(report=bad, mode=self._mode, receptor_pdb=receptor_pdb,
                            extra_checks=())
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = 42
        embed_status = AllChem.EmbedMolecule(mol, params)
        if embed_status == -1:
            Chem.RemoveHs(mol)
            bad = ValidityReport(
                smiles=smiles, passed=False, n_checks=0, n_passed=0,
                pass_rate=0.0, failed_checks=("embed_3d",),
                details={"embed_3d": False},
            )
            return PBResult(report=bad, mode=self._mode, receptor_pdb=receptor_pdb,
                            extra_checks=())
        try:
            mmff_status = AllChem.MMFF94OptimizeMolecule(mol, maxIters=200)
        except Exception:
            mmff_status = -1
        if mmff_status != 0:
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=200)
            except Exception:
                pass
        Chem.RemoveHs(mol)
        # PoseBusters.dock requires the conditioning molecule (receptor)
        # to be passed as mol_cond.  We pin full_report=True to surface
        # all 29 bool columns so dock-specific clash checks land in
        # ``details`` for downstream aggregation.
        df = self._pb.bust(mol, mol_cond=str(receptor_pdb), full_report=True)
        report = self._df_to_report(smiles, df)
        # Compute the extra checks: protein-aware names that are NOT in
        # the mol-mode set.  We probe by spinning up a quick mol-mode
        # PB to enumerate its bool columns; if anything goes wrong we
        # fall back to a hard-coded set (stable across posebusters 0.x).
        extras = self._protein_aware_extras(df)
        return PBResult(report=report, mode=self._mode, receptor_pdb=receptor_pdb,
                        extra_checks=extras)

    def validate_list(
        self,
        smiles_list: Iterable[str],
    ) -> List[ValidityReport]:
        """Validate a list of SMILES; returns one report per input."""
        return [self.validate_mol(s) for s in smiles_list]

    def pass_rate(self, smiles_list: Iterable[str]) -> float:
        """Fraction of mols that pass ALL PoseBusters checks."""
        reports = self.validate_list(smiles_list)
        if not reports:
            return 0.0
        return float(sum(r.passed for r in reports)) / len(reports)

    def get_metadata(self) -> dict:
        return {
            "name": self.name,
            "engine": "PoseBusters (Python API)",
            "engine_version": self._version,
            "mode": self._mode,
        }

    # ---------------------------------------------------------- Internals
    def _protein_aware_extras(self, df: pd.DataFrame) -> tuple:
        """Return the protein-aware check names in this DataFrame.

        "Protein-aware" = present in ``dock`` mode but absent in the
        default ``mol`` mode.  We compare against a probe PoseBusters
        instance running in ``mol`` mode so this stays robust against
        upstream column additions.

        Falls back to a hard-coded set when the probe fails (e.g. RDKit
        is unavailable in the worker process).  The hard-coded set is
        what PoseBusters >= 0.4 emits for ``config='dock'`` minus the
        three loaders we already filter.
        """
        fallback = (
            "protein-ligand_maximum_distance",
            "minimum_distance_to_protein",
            "minimum_distance_to_organic_cofactors",
            "minimum_distance_to_inorganic_cofactors",
            "minimum_distance_to_waters",
            "volume_overlap_with_protein",
            "volume_overlap_with_organic_cofactors",
            "volume_overlap_with_inorganic_cofactors",
            "volume_overlap_with_waters",
            "not_too_far_away_organic_cofactors",
            "not_too_far_away_inorganic_cofactors",
            "not_too_far_away_waters",
        )
        try:
            from posebusters import PoseBusters  # type: ignore
            probe = PoseBusters(config="mol")
            from rdkit import Chem  # type: ignore
            from rdkit.Chem import AllChem  # type: ignore
            # Build a probe mol that EMBEDS reliably — bare ``C`` (methane)
            # often fails ETKDGv3 on small systems, so we use ethanol
            # which embeds deterministically and gives a full conformer
            # so PoseBusters returns all chemistry check columns.
            pm = Chem.MolFromSmiles("CCO")
            pm = Chem.AddHs(pm)
            params = AllChem.ETKDGv3()
            params.randomSeed = 42
            if AllChem.EmbedMolecule(pm, params) == 0:
                try:
                    AllChem.MMFF94OptimizeMolecule(pm, maxIters=200)
                except Exception:
                    AllChem.UFFOptimizeMolecule(pm, maxIters=200)
                Chem.RemoveHs(pm)
            probe_df = probe.bust(pm, full_report=True)
            mol_bool_cols = {
                c for c in probe_df.columns
                if probe_df[c].dtype == bool
            }
            dock_bool_cols = {
                c for c in df.columns
                if df[c].dtype == bool
            }
            extras = dock_bool_cols - mol_bool_cols
            # Always exclude loaders.
            extras -= {"mol_pred_loaded", "mol_cond_loaded", "mol_true_loaded"}
            if extras:
                return tuple(sorted(extras))
            return fallback
        except Exception:
            return fallback

    def _df_to_report(self, smiles: str, df: pd.DataFrame) -> ValidityReport:
        # PoseBusters bust() returns a DataFrame with one row per molecule
        # and one boolean column per check. We treat the first row.
        if df is None or df.empty:
            return ValidityReport(
                smiles=smiles, passed=False, n_checks=0, n_passed=0,
                pass_rate=0.0, failed_checks=("empty_report",),
                details={},
            )
        row = df.iloc[0]
        # Drop non-check columns (file paths, identifiers, counts).
        # PoseBusters mixes true booleans (the actual validity checks)
        # with count / metric columns (number_bonds, shortest_bond_relative_length,
        # num_h_added, ...).  We must only treat bool-dtype columns as
        # checks, otherwise int(0) gets misread as "False = failed"
        # (see _df_to_report_old which produced 0% pass-rate spuriously).
        skip = {
            "file", "molecule", "position", "molecule_id", "name",
            # Reference-file loading checks (we have no experimental
            # pose in de novo generation, so these are always False
            # and would otherwise cap pass-rate at ~85%).
            "mol_true_loaded", "mol_cond_loaded",
            # Pred file loading (we always supply one, so this is True;
            # listed for clarity).
            "mol_pred_loaded",
        }
        check_cols = [
            c for c in df.columns
            if c not in skip and df[c].dtype == bool
        ]
        details: dict = {}
        for c in check_cols:
            v = row[c]
            if isinstance(v, (bool, np.bool_)):
                details[c] = bool(v)
            elif isinstance(v, str):
                details[c] = v.lower() in {"true", "pass", "yes"}
            else:
                details[c] = bool(v)
        n_checks = len(details)
        n_passed = sum(1 for v in details.values() if v)
        failed = tuple(c for c, v in details.items() if not v)
        return ValidityReport(
            smiles=smiles,
            passed=(n_checks == n_passed and n_checks > 0),
            n_checks=n_checks,
            n_passed=n_passed,
            pass_rate=(n_passed / n_checks) if n_checks > 0 else 0.0,
            failed_checks=failed,
            details=details,
        )


def _pb_version() -> str:
    try:
        import posebusters  # type: ignore
        return getattr(posebusters, "__version__", "unknown")
    except Exception:
        return "uninstalled"


# ----------------------------------------------------------------
# MMFF94s intra-ligand relaxation — Stage 1 of the
# dock -> MMFF94s relax -> PB check pipeline (WF-PB-MMFF94-Relax).
# ----------------------------------------------------------------
def mmff94s_relax_pose(mol, max_iters: int = 200) -> dict:
    """Run RDKit ``MMFFOptimizeMolecule`` (MMFF94s variant) on a docked pose.

    Background (Halgren 1996, *J. Comput. Chem.* 17, 490-512) —
    MMFF94s adds a "substantially improved" out-of-plane bending term
    over MMFF94 (Halgren 1996 §4) and is the variant RDKit ships as
    ``AllChem.MMFFOptimizeMolecule(mol, mmffVariant="MMFF94s")``
    (Tosco 2014 RDKit MMFF docs).

    The PoseBusters paper (Buttenschoen 2024) uses MMFF94s minimisation
    as the chemistry-reference window: PB validates that bonded
    geometry / angles / torsions are within tight bounds of an MMFF
    minimum, and an un-relaxed Vina pose frequently fails those checks
    (the Vina scoring function does NOT optimise bonded terms).
    Halgren 1996 reports 0.014 Å bond-length and 1.2° angle-angle RMS
    against MP2/6-31G* reference geometries.

    Parameters
    ----------
    mol : rdkit.Chem.Mol
        Single-conformer 3D molecule (the docked pose).  MUST have
        explicit hydrogens (the Vina workflow adds Hs; the pose is
        directly passed here).
    max_iters : int, default 200
        Iteration cap.  Returns 0 on convergence.

    Returns
    -------
    dict
        Status report.  Always returns; never raises:

        * ``{"ok": True, "status": int, "variant": "MMFF94s", "max_iters": int}``
          on success (status==0) or non-fatal partial convergence
          (status!=0).
        * ``{"ok": False, "status": -1, "variant": "MMFF94s",
             "error": "..."}`` if RDKit MMFF94s cannot parameterise
          the molecule (e.g. unsupported atom types like charged
          metals, very large systems).
        * ``{"ok": False, "status": -1, "variant": "MMFF94s",
             "error": "not_a_molecule"}`` on bad input.

    The caller is expected to do ``Chem.Mol(mol)`` to clone before
    relaxation — this helper mutates ``mol`` in place (consistent with
    RDKit's MMFFOptimizeMolecule contract).
    """
    result = {"ok": False, "status": -1, "variant": "MMFF94s",
              "max_iters": int(max_iters), "error": None}
    try:
        from rdkit import Chem  # type: ignore
        from rdkit.Chem import AllChem  # type: ignore
    except Exception as exc:
        result["error"] = f"rdkit_unavailable:{type(exc).__name__}"
        return result
    if mol is None or not isinstance(mol, Chem.Mol) or mol.GetNumAtoms() == 0:
        result["error"] = "not_a_molecule"
        return result
    if mol.GetNumConformers() < 1:
        result["error"] = "no_conformer"
        return result
    try:
        # RDKit accepts mmffVariant="MMFF94s" for the improved variant
        # (Halgren 1996 §4 "MMFF94s").  Tosco 2014 RDKit MMFF docs
        # confirm this is the recommended production variant for
        # drug-like organics.
        status = AllChem.MMFFOptimizeMolecule(
            mol,
            maxIters=int(max_iters),
            mmffVariant="MMFF94s",
        )
        result["ok"] = True
        result["status"] = int(status)
        return result
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}:{exc}"
        return result


# ----------------------------------------------------------------
# Convenience helpers
# ----------------------------------------------------------------
def validate_mol(smiles: str) -> ValidityReport:
    """One-shot validation."""
    return PoseBustersAdapter().validate_mol(smiles)


def pass_rate(smiles_list: List[str]) -> float:
    """Bulk pass-rate over a SMILES list."""
    return PoseBustersAdapter().pass_rate(smiles_list)


__all__ = [
    "PoseBustersAdapter",
    "ValidityReport",
    "PBResult",
    "validate_mol",
    "pass_rate",
    "_have_posebusters",
    "mmff94s_relax_pose",
]
