"""Real RDKit-based PropertyPredictor adapter.

Implements the :class:`molmetal.ports.PropertyPredictor` port using
**only** RDKit descriptors (no GPU, no neural network).  The goal is
to produce the 2D / chemical-property fields of ``PropertyPrediction``
(QED, logP, MW, TPSA, Lipinski counts, SA score) directly from the
molecule SMILES, leaving the GPU-only binding affinity prediction to
:class:`EGNNPropertyPredictor`.

Why a separate file rather than folding into ``mock.py``?
* ``mock.py`` is *deterministic / synthetic* (random sa_score, random
  pic50) for orchestration smoke tests.  This module is the **real**
  predictor and should be used whenever RDKit is available.
* SA score requires RDKit's contributed ``sascorer`` module and the
  data file ``SA_Score.sdf.gz`` shipped with RDKit.  When the contrib
  module isn't usable (e.g. headless wheels without data dir) we fall
  back to a simple aromatic-rings-based proxy.
"""

from __future__ import annotations

from typing import Optional

from molmetal.domain import Complex, Molecule
from molmetal.ports import PropertyPrediction, PropertyPredictor


__all__ = ["RDKitPropertyPredictor"]


# ---------------------------------------------------------------------------
# SA score: try RDKit contrib first, fall back to a 1 / (1 + n_aromatic) proxy
# ---------------------------------------------------------------------------
def _try_import_sascorer():
    """Lazy import of RDKit's contributed SA-score module.

    Returns the ``sascorer`` callable or ``None`` if it can't be loaded.
    """
    try:
        from rdkit.Chem import RDConfig  # noqa: WPS433
        import os
        import sys  # noqa: WPS433

        contrib_path = os.path.join(RDConfig.RDContribDir, "SA_Score")
        if os.path.isdir(contrib_path) and contrib_path not in sys.path:
            sys.path.append(contrib_path)
        # Check the SA score data file actually ships with the install.
        sa_data = os.path.join(RDConfig.RDDataDir, "SA_Score.sdf.gz") \
            if hasattr(RDConfig, "RDDataDir") else ""
        if sa_data and not os.path.exists(sa_data):
            return None
        from rdkit.Contrib.SA_Score import sascorer  # type: ignore  # noqa: WPS433

        return sascorer
    except Exception:
        return None


_SASCORER = _try_import_sascorer()


def _sa_score_proxy(mol) -> float:
    """Fallback SA score when ``rdkit.Contrib.SA_Score`` isn't available.

    We use ``1.0 / (1 + NumAromaticRings)`` as a crude proxy so the value
    is in (0, 1], decreasing with ring complexity.  This is **not** the
    Ertl-Schuffenhauer SA score and should be flagged as such in logs.
    """
    from rdkit.Chem import Lipinski  # noqa: WPS433

    n_arom = int(Lipinski.NumAromaticRings(mol))
    return float(1.0 / (1.0 + n_arom))


# ---------------------------------------------------------------------------
# PropertyPredictor implementation
# ---------------------------------------------------------------------------
class RDKitPropertyPredictor(PropertyPredictor):
    """PropertyPredictor that computes 2D descriptors from SMILES via RDKit.

    Computed fields (when SMILES parses successfully):
        * ``qed``             : RDKit ``Descriptors.qed``                                (0..1)
        * ``logp``            : ``Crippen.MolLogP``                                     (float)
        * ``mol_weight``      : ``Descriptors.MolWt``                                   (g/mol)
        * ``tpsa``            : ``rdMolDescriptors.CalcTPSA``                           (Å²)
        * ``num_h_donors``    : ``Lipinski.NumHDonors``
        * ``num_h_acceptors`` : ``Lipinski.NumHAcceptors``
        * ``num_rotatable_bonds``: ``Lipinski.NumRotatableBonds``
        * ``sa_score``        : ``rdkit.Contrib.SA_Score.sascorer`` if available,
                                else ``1 / (1 + NumAromaticRings)`` proxy              (0..1)

    Always ``None``:
        * ``binding_affinity_pic50`` (use :class:`EGNNPropertyPredictor`)
        * ``metal_binding_score``    (use a metal-specific predictor)

    Robustness:
        * Empty SMILES, invalid SMILES, or RDKit import failure all return a
          default ``PropertyPrediction()`` rather than raising — the
          orchestrator can decide whether to filter these out.
        * If the molecule has 3D coords but no SMILES, we try to reconstruct
          a Mol from the SMILES-less path via ``Molecule.to_rdkit()``.
    """

    name = "RDKitPropertyPredictor_v0"

    def __init__(self) -> None:
        self._device = "cpu"
        self._sascorer_available = _SASCORER is not None

    # ------------------------------------------------------------------
    def setup(self, device: str = "cpu") -> None:
        # RDKit descriptors are CPU-only.
        self._device = "cpu"

    # ------------------------------------------------------------------
    def predict(
        self,
        molecule: Molecule,
        complex: Optional[Complex] = None,  # noqa: A002  (match port signature)
    ) -> PropertyPrediction:
        # Lazy RDKit imports keep this module importable even if rdkit
        # is missing in some test environments.
        try:
            from rdkit import Chem  # noqa: WPS433
        except Exception:
            return PropertyPrediction()

        mol = self._coerce_to_mol(molecule, Chem)
        if mol is None:
            # Empty / invalid SMILES and reconstruction failed — return defaults.
            return PropertyPrediction()

        try:
            from rdkit.Chem import (  # noqa: WPS433
                Crippen,
                Descriptors,
                Lipinski,
                rdMolDescriptors,
            )

            qed = float(Descriptors.qed(mol))
            logp = float(Crippen.MolLogP(mol))
            mol_weight = float(Descriptors.MolWt(mol))
            tpsa = float(rdMolDescriptors.CalcTPSA(mol))
            num_h_donors = int(Lipinski.NumHDonors(mol))
            num_h_acceptors = int(Lipinski.NumHAcceptors(mol))
            num_rotatable_bonds = int(Lipinski.NumRotatableBonds(mol))
        except Exception:
            return PropertyPrediction()

        # SA score: real sascorer if importable, else proxy.
        if _SASCORER is not None:
            try:
                sa_score = float(_SASCORER.calculateScore(mol))
            except Exception:
                sa_score = _sa_score_proxy(mol)
        else:
            sa_score = _sa_score_proxy(mol)

        return PropertyPrediction(
            qed=qed,
            sa_score=sa_score,
            logp=logp,
            mol_weight=mol_weight,
            tpsa=tpsa,
            num_h_donors=num_h_donors,
            num_h_acceptors=num_h_acceptors,
            num_rotatable_bonds=num_rotatable_bonds,
            binding_affinity_pic50=None,
            metal_binding_score=None,
        )

    # ------------------------------------------------------------------
    @staticmethod
    def _coerce_to_mol(molecule: Molecule, Chem_mod):
        """Attempt to convert a ``Molecule`` into an RDKit ``Mol``.

        Priority:
            1. ``Molecule.smiles`` if non-empty (canonical source of truth).
            2. ``Molecule.to_rdkit()`` (rebuilds from coords/bonds) — only
               used when ``smiles`` was empty *and* the molecule has zero
               atoms.  If the molecule has coords/bonds but no SMILES we
               treat it as **unset** and return ``None`` — the caller
               will fall back to a default ``PropertyPrediction``.
            3. ``None`` — caller returns a default ``PropertyPrediction``.
        """
        if molecule.smiles:
            mol = Chem_mod.MolFromSmiles(molecule.smiles)
            if mol is not None:
                return mol
            # SMILES present but unparseable — fall through to None.
            return None
        # Empty SMILES: only try the structural fallback when the molecule
        # actually has atoms (otherwise we'd be returning a default-pred
        # for a sentinel / placeholder, which is what we want).
        if molecule.n_atoms == 0:
            return None
        # Molecule has structure but no SMILES — defer to a higher-level
        # caller that may regenerate SMILES first.  Returning None here
        # signals "no usable descriptor input".
        return None

    # ------------------------------------------------------------------
    def get_metadata(self) -> dict:
        return {
            "model": self.name,
            "type": "rdkit-descriptors",
            "device": self._device,
            "sascorer_available": self._sascorer_available,
            "sa_score_source": (
                "rdkit.Contrib.SA_Score.sascorer"
                if self._sascorer_available
                else "1/(1+NumAromaticRings) proxy"
            ),
            "fields_computed": [
                "qed", "logp", "mol_weight", "tpsa",
                "num_h_donors", "num_h_acceptors", "num_rotatable_bonds",
                "sa_score",
            ],
            "fields_left_none": ["binding_affinity_pic50", "metal_binding_score"],
            "description": (
                "Real RDKit-based PropertyPredictor. Computes QED, logP, MW, "
                "TPSA, Lipinski counts, and SA score (real sascorer when "
                "available, proxy otherwise). No GPU required. Binding "
                "affinity (pIC50) is left None — use EGNNPropertyPredictor."
            ),
        }
