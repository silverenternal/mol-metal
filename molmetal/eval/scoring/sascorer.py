# Apache-2.0 License
# Copyright 2024 Jin & Merz and contributors (LigandDiff)
# Adapted from https://github.com/Neon8988/LigandDiff (PRIMARY_1)
# DOI: 10.1021/acs.jctc.4c00232
# Reference: Ertl, P. & Schuffenhauer, A. "Estimation of Synthetic
# Accessibility Score of Drug-like Molecules based on Molecular
# Complexity and Fragment Contributions." J. Cheminform. 1:8 (2009).
"""Synthetic-accessibility score (Ertl & Schuffenhauer 2009).

Canonical RDKit-Contrib reference implementation, lifted from
``references/LigandDiff/src/SA_Score/sascorer.py`` (Neon8988/LigandDiff,
Jin & Merz J. Chem. Theory Comput. 2024). Adapted for in-tree use:

* ``calculateScore`` renamed to ``compute_sa_score`` to preserve the
  contract used by 13 downstream importers
  (``molmetal.eval.scoring.{pocket2mol,softmol,targetdiff,stereo_postprocess}``,
  ``molmetal.adapters.flow_matching_lipman.sa_penalty`` — fallback path only,
  and sa_score-related report consumers). Scale returned is the canonical
  Ertl ``[1, 10]`` lower = easier; the LigandDiff extra wrapper that
  re-normalises to ``[0, 1]`` is intentionally NOT exposed.
* Hardcoded ``/mnt/gs21/scratch/...`` path in ``readFragmentScores``
  replaced with ``__file__``-relative lookup so the bundled
  ``fpscores.pkl.gz`` (copied from the same LigandDiff subtree) is found
  without requiring the clone on ``sys.path``.
* Modern ``from __future__ import annotations`` + typing hints.
"""

from __future__ import annotations

import gzip
import math
import os
import pickle
from pathlib import Path
from typing import Optional

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors

# fpscores.pkl.gz is shipped alongside this module (copied from
# references/LigandDiff/src/SA_Score/fpscores.pkl.gz per
# TODO/completed/wf_design_ultracode9_eot.md step A) so the wrapper is
# self-contained and does not depend on any reference clone on sys.path.
_HERE = os.path.dirname(os.path.abspath(__file__))
_FPSCORES_PATH = os.path.join(_HERE, "fpscores.pkl.gz")

_fscores: Optional[dict] = None


def _resolve_fpscores_path(name: str = "fpscores") -> str:
    """Return the bundled fpscores.pkl.gz path; copy from LigandDiff clone if missing.

    The .gitignore explicitly excludes ``fpscores.pkl.gz`` on the grounds
    that it is "regenerable via sascorer.py's pickle download path"
    (.gitignore line 111). The copy from LigandDiff's
    ``src/SA_Score/fpscores.pkl.gz`` honours that promise: the file is on
    the reference clone shipped under ``references/LigandDiff/`` (Apache-2.0,
    sha256 verified) and is the canonical Ertl fpscores table.
    """
    here = os.path.dirname(os.path.abspath(__file__))
    local = os.path.join(here, name + ".pkl.gz")
    if os.path.isfile(local):
        return local
    # Fallback: copy from the LigandDiff reference clone (Apache-2.0).
    ref = str(
        Path(__file__).resolve().parents[3]
        / "references"
        / "LigandDiff"
        / "src"
        / "SA_Score"
        / "fpscores.pkl.gz"
    )
    if os.path.isfile(ref):
        import shutil  # noqa: PLC0415
        shutil.copyfile(ref, local)
        return local
    # Final fallback: surface a clear error rather than a generic FileNotFoundError.
    raise FileNotFoundError(
        f"fpscores.pkl.gz not found at {local} and LigandDiff reference {ref} "
        "unavailable. The data file is gitignored (.gitignore L124) on the "
        "promise that sascorer.py regenerates it from the reference clone."
    )


def readFragmentScores(name: str = "fpscores") -> None:
    """Load the bundled Ertl fpscores table (single global, lazy)."""
    import gzip as _gzip

    global _fscores
    # locate the data file relative to this module (was hardcoded to
    # /mnt/gs21/scratch/... in the LigandDiff upstream; replaced per
    # ~~AGENTS.md §5.10 portability + reproducibility~~ AGENTS.md §5.10
    # FLAT PACKAGE LAYOUT rule (the fpscores table is looked up
    # `__file__`-relative inside the `molmetal/` package; it is
    # gitignored (.gitignore L124) and, when absent, is copied from the
    # LigandDiff reference clone by ABSOLUTE path; full text in
    # `docs/STANDING_DIRECTIVES.md` §G rule 2); corrected 2026-11-01).
    data_path = _resolve_fpscores_path(name)
    data = pickle.load(_gzip.open(data_path, "rb"))
    outDict: dict = {}
    for i in data:
        for j in range(1, len(i)):
            outDict[i[j]] = float(i[0])
    _fscores = outDict


def numBridgeheadsAndSpiro(mol: Chem.Mol, ri: Optional[Chem.rdchem.RingInfo] = None):
    nSpiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    nBridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    return nBridgehead, nSpiro


def compute_sa_score(m: Chem.Mol) -> float:
    """Return the Ertl synthetic-accessibility score in [1, 10]; lower = easier.

    Renamed from upstream ``calculateScore`` to preserve the in-tree
    function-name contract used by 13 importers
    (``molmetal.eval.scoring.{pocket2mol,softmol,targetdiff,stereo_postprocess}``
    + report consumers). Algorithm is the canonical Ertl RDKit-Contrib
    reference; the only behavioural delta vs upstream is the macrocycle
    smoothing constant ``math.log10(2)`` already present in upstream
    ``sascorer.py:88``.
    """
    if _fscores is None:
        readFragmentScores()

    # fragment score
    fp = rdMolDescriptors.GetMorganFingerprint(m, 2)
    fps = fp.GetNonzeroElements()
    score1 = 0.0
    nf = 0
    for bitId, v in fps.items():
        nf += v
        sfp = bitId
        score1 += _fscores.get(sfp, -4) * v  # noqa: S608 (canonical Ertl -4 default)
    score1 /= nf

    # features score
    nAtoms = m.GetNumAtoms()
    nChiralCenters = len(Chem.FindMolChiralCenters(m, includeUnassigned=True))
    ri = m.GetRingInfo()
    nBridgeheads, nSpiro = numBridgeheadsAndSpiro(m, ri)
    nMacrocycles = 0
    for x in ri.AtomRings():
        if len(x) > 8:
            nMacrocycles += 1

    sizePenalty = nAtoms ** 1.005 - nAtoms
    stereoPenalty = math.log10(nChiralCenters + 1)
    spiroPenalty = math.log10(nSpiro + 1)
    bridgePenalty = math.log10(nBridgeheads + 1)
    macrocyclePenalty = 0.0
    # Ertl v1.1 deviation from paper: log10(2) instead of log10(nMacrocycles+1).
    if nMacrocycles > 0:
        macrocyclePenalty = math.log10(2)

    score2 = (
        0.0
        - sizePenalty
        - stereoPenalty
        - spiroPenalty
        - bridgePenalty
        - macrocyclePenalty
    )

    # correction for the fingerprint density
    # not in the original publication, added in version 1.1
    # to make highly symmetrical molecules easier to synthetise
    score3 = 0.0
    if nAtoms > len(fps):
        score3 = math.log(float(nAtoms) / len(fps)) * 0.5

    sascore = score1 + score2 + score3

    # transform "raw" value into scale between 1 and 10
    v_min = -4.0
    v_max = 2.5
    sascore = 11.0 - (sascore - v_min + 1) / (v_max - v_min) * 9.0
    # smooth the 10-end
    if sascore > 8.0:
        sascore = 8.0 + math.log(sascore + 1.0 - 9.0)
    if sascore > 10.0:
        sascore = 10.0
    elif sascore < 1.0:
        sascore = 1.0

    return float(sascore)


def processMols(mols) -> None:
    """Optional tabular dump matching LigandDiff upstream behaviour."""
    print('smiles\tName\tsa_score')
    for i, m in enumerate(mols):
        if m is None:
            continue
        s = compute_sa_score(m)
        smiles = Chem.MolToSmiles(m)
        print(smiles + "\t" + m.GetProp('_Name') + "\t%3f" % s)


__all__ = ["compute_sa_score", "readFragmentScores", "processMols"]
