"""ADMET prediction runner with multi-backend fallback.

Layer 8 reward channel used by :class:`RewardAggregator` — exposes a
single public function :func:`predict_admet` that returns an ADMET
property dict for a SMILES string, with a deterministic backend
fallback chain:

1. **admet-ai** — ``admet_ai.ADMETModel.predict(smiles)`` returns a
   104-key dict covering Lipinski, PAINS/BRENK/NIH, AMES, BBB, CYP
   substrate/inhibitor, Caco2, hERG, solubility, Bioavailability_Ma,
   and 13 physchem descriptors (MW, logP, HBA, HBD, TPSA, …).
   This is the canonical path.  The model artefacts live under
   ``admet_ai/resources/models`` and total ~2 GB.
2. **datamol / molfeat** — if admet-ai fails (model load, no GPU, …)
   we fall back to datamol's high-level ``dm.descriptors`` plus a
   molfeat ``RDKitDescriptors`` featurizer to back-fill the physchem
   block.  These are RDKit-derived and stay consistent with the
   canonical path's keys.
3. **RDKit-only** — last resort.  Compute the six Lipinski-relevant
   RDKit descriptors (MW, logP, HBA, HBD, TPSA, rotatable bonds) plus
   QED + SA via RDKit + ``rdkit.Chem.Descriptors``.

All paths return at minimum the six Lipinski-relevant descriptors
(MW, logP, HBD, HBA, TPSA, rotatable_bonds), QED, and SA so the
downstream :func:`admet_desirability` scoring stays well-defined.

The module never raises out of :func:`predict_admet` — invalid SMILES
return ``{}`` (and emit a warning).  Backend detection is memoised at
module level (``_DETECTED_BACKEND``) so the import cost is paid once
per process.

Examples
--------
>>> from molmetal.validation.admet_runner import predict_admet
>>> out = predict_admet("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
>>> round(out["MW"], 2)
180.16
>>> "logP" in out and "Bioavailability_Ma" in out
True
"""
from __future__ import annotations

import logging
import os
import warnings
from functools import lru_cache
from typing import Dict, Optional

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Backend detection (memoised)
# ---------------------------------------------------------------------------

_DETECTED_BACKEND: Optional[str] = None
_ADMET_MODEL = None  # cached admet_ai.ADMETModel instance


def _detect_backend() -> str:
    """Return one of ``"admet-ai"`` / ``"datamol"`` / ``"rdkit"``.

    Probed in priority order on first call.  Subsequent calls return
    the cached value so we don't re-import pytorch-lightning.
    """
    global _DETECTED_BACKEND
    if _DETECTED_BACKEND is not None:
        return _DETECTED_BACKEND

    # 1) admet-ai
    try:
        import admet_ai  # noqa: F401

        _DETECTED_BACKEND = "admet-ai"
        log.info("admet_runner: backend = admet-ai (canonical)")
        return _DETECTED_BACKEND
    except Exception as exc:
        log.info("admet_runner: admet-ai import failed (%s); trying datamol", exc)

    # 2) datamol / molfeat
    try:
        import datamol as dm  # noqa: F401

        try:
            import molfeat  # noqa: F401

            _DETECTED_BACKEND = "datamol+molfeat"
        except Exception:
            _DETECTED_BACKEND = "datamol"
        log.info("admet_runner: backend = %s (fallback 1)", _DETECTED_BACKEND)
        return _DETECTED_BACKEND
    except Exception as exc:
        log.info("admet_runner: datamol import failed (%s); falling back to rdkit", exc)

    # 3) RDKit only
    try:
        from rdkit.Chem import Descriptors  # noqa: F401

        _DETECTED_BACKEND = "rdkit"
        log.info("admet_runner: backend = rdkit-only (last resort)")
        return _DETECTED_BACKEND
    except Exception as exc:
        log.error("admet_runner: RDKit import failed (%s); ADMET unavailable", exc)
        _DETECTED_BACKEND = "none"
        return _DETECTED_BACKEND


def active_backend() -> str:
    """Return the name of the active backend (``"admet-ai"`` /
    ``"datamol"`` / ``"datamol+molfeat"`` / ``"rdkit"`` / ``"none"``).

    Useful for diagnostics / logs.
    """
    return _detect_backend()


# ---------------------------------------------------------------------------
# SMILES normalisation
# ---------------------------------------------------------------------------


def _canonical_smiles(smiles: str) -> Optional[str]:
    """Return RDKit-canonicalised SMILES or ``None`` on parse failure.

    Uses RDKit directly (always available in this stack).  Falls back
    to the input string stripped if RDKit cannot parse.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        from rdkit import Chem
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# admet-ai backend
# ---------------------------------------------------------------------------


def _get_admet_model():
    """Return the cached admet_ai.ADMETModel instance (lazy init).

    On first call we instantiate the model.  This loads the pytorch-
    lightning module + 2 GB of weights — slow on first call, fast on
    subsequent calls (the model object is cached at module level).
    """
    global _ADMET_MODEL
    if _ADMET_MODEL is not None:
        return _ADMET_MODEL
    try:
        # Suppress admet-ai's noisy lightning import-time stdout
        import sys as _sys
        import io as _io

        old_stdout = _sys.stdout
        old_stderr = _sys.stderr
        try:
            _sys.stdout = _io.StringIO()
            _sys.stderr = _io.StringIO()
            from admet_ai import ADMETModel
            _ADMET_MODEL = ADMETModel(num_workers=1)
        finally:
            _sys.stdout = old_stdout
            _sys.stderr = old_stderr
        return _ADMET_MODEL
    except Exception as exc:
        log.warning("admet_runner: failed to instantiate admet_ai.ADMETModel (%s)", exc)
        return None


def _predict_admet_ai(smiles: str) -> Optional[Dict[str, float]]:
    """admet-ai path.  Returns the raw 104-key dict on success, else None."""
    model = _get_admet_model()
    if model is None:
        return None
    try:
        import sys as _sys
        import io as _io

        old_stdout = _sys.stdout
        old_stderr = _sys.stderr
        try:
            _sys.stdout = _io.StringIO()
            _sys.stderr = _io.StringIO()
            out = model.predict(smiles)
        finally:
            _sys.stdout = old_stdout
            _sys.stderr = old_stderr
        if not isinstance(out, dict):
            return None
        # Cast to float (admet-ai returns numpy scalars; json can't
        # serialize those).
        return {str(k): float(v) for k, v in out.items()}
    except Exception as exc:
        log.warning("admet_runner: admet-ai predict failed (%s)", exc)
        return None


# ---------------------------------------------------------------------------
# RDKit fallback (always-on — also used to back-fill missing keys)
# ---------------------------------------------------------------------------


_RDKIT_PHYSCHEM_KEYS = (
    "MW",
    "logP",
    "HBA",
    "HBD",
    "TPSA",
    "RotBonds",
    "QED",
    "SA",
)


def _compute_rdkit_descriptors(smiles: str) -> Dict[str, float]:
    """Return the eight RDKit-derived descriptors for ``smiles``.

    Always returns a dict (possibly empty if RDKit cannot parse the
    SMILES).  Keys are normalised to short Lipinski names (``MW``,
    ``logP``, ``HBA``, ``HBD``, ``TPSA``, ``RotBonds``, ``QED``,
    ``SA``) — ``predict_admet`` re-maps these into the canonical
    admet-ai-style keys so downstream code doesn't branch on backend.
    """
    out: Dict[str, float] = {}
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem, Descriptors, Lipinski, QED
        from rdkit import RDLogger

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return out
        try:
            AllChem.Compute2DCoords(mol)
        except Exception:
            pass  # 2D-only — not strictly needed for these descs
        out["MW"] = float(Descriptors.MolWt(mol))
        out["logP"] = float(Descriptors.MolLogP(mol))
        out["HBA"] = float(Lipinski.NumHAcceptors(mol))
        out["HBD"] = float(Lipinski.NumHDonors(mol))
        out["TPSA"] = float(Descriptors.TPSA(mol))
        out["RotBonds"] = float(Lipinski.NumRotatableBonds(mol))
        out["QED"] = float(QED.qed(mol))
        # Synthetic accessibility — RDKit contrib if present, else NaN
        try:
            from rdkit.Chem import RDConfig  # noqa: F401
            import os as _os
            contrib_path = _os.path.join(
                _os.path.dirname(__file__), "..", "..", ".venv", "lib",
            )
            # fallback: ask rdkit for its contrib path
            from rdkit.Chem import RDConfig as _RC
            sascorer_path = _os.path.join(_RC.RDContribDir, "SA_Score")  # type: ignore[attr-defined]
            import sys as _sys
            if sascorer_path not in _sys.path:
                _sys.path.append(sascorer_path)
            try:
                import sascorer  # type: ignore[import-not-found]
                out["SA"] = float(sascorer.calculateScore(mol))
            except Exception:
                out["SA"] = float("nan")
        except Exception:
            out["SA"] = float("nan")
    except Exception as exc:
        log.warning("admet_runner: RDKit descriptor compute failed (%s)", exc)
    return out


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


# Mapping from admet-ai key → our short key.  The aggregator reads
# ``MW / logP / HBA / HBD / TPSA`` so we normalise both backends onto
# the same short names.
_ADMET_AI_KEY_MAP = {
    "molecular_weight": "MW",
    "logP": "logP",
    "hydrogen_bond_acceptors": "HBA",
    "hydrogen_bond_donors": "HBD",
    "tpsa": "TPSA",
    "rotatable_bonds": "RotBonds",
    "QED": "QED",
}


def predict_admet(smiles: str) -> Dict[str, float]:
    """Return an ADMET property dict for ``smiles``.

    Backend priority: ``admet-ai`` → ``datamol/molfeat`` →
    ``RDKit-only``.  Even when the AI backend is active, RDKit
    descriptors are merged in to back-fill any key the AI path missed
    and to guarantee a consistent schema.

    Returns
    -------
    dict
        Keys always include ``MW, logP, HBD, HBA, TPSA, RotBonds,
        QED`` when the SMILES parses.  Admet-ai additionally supplies
        ``Bioavailability_Ma, Solubility_AqSolDB, AMES, BBB_Martins,
        CYP1A2_Veith, ...`` (~100 endpoints).  Returns ``{}`` and
        emits a warning on parse failure.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        warnings.warn(f"predict_admet: empty/invalid SMILES {smiles!r}; returning {{}}")
        return {}

    can_smi = _canonical_smiles(smiles)
    if can_smi is None:
        warnings.warn(f"predict_admet: cannot parse SMILES {smiles!r}; returning {{}}")
        return {}

    backend = _detect_backend()
    out: Dict[str, float] = {}

    # 1) admet-ai canonical path
    if backend == "admet-ai":
        ai_out = _predict_admet_ai(can_smi)
        if ai_out:
            # Map canonical keys
            for src, dst in _ADMET_AI_KEY_MAP.items():
                if src in ai_out:
                    out[dst] = ai_out[src]
            # Keep the original admet-ai keys too (Bioavailability_Ma,
            # CYP*, AMES, …) — callers that want them just look them up.
            for k, v in ai_out.items():
                out.setdefault(k, v)

    # 2) datamol/molfeat fallback (only used when admet-ai failed)
    elif backend in ("datamol", "datamol+molfeat"):
        try:
            import datamol as dm  # type: ignore[import-not-found]

            mol = dm.to_mol(can_smi)
            if mol is not None:
                # dm.descriptors returns a pandas Series of physchem
                descs = dm.descriptors.compute_many_descriptors(
                    mol,
                    descriptors=["mw", "clogp", "n_lipinski_hbd", "n_lipinski_hba",
                                 "tpsa", "n_rotatable_bonds", "qed"],
                )
                if descs is not None and len(descs) > 0:
                    row = descs.iloc[0] if hasattr(descs, "iloc") else descs
                    mapping = {
                        "mw": "MW",
                        "clogp": "logP",
                        "n_lipinski_hbd": "HBD",
                        "n_lipinski_hba": "HBA",
                        "tpsa": "TPSA",
                        "n_rotatable_bonds": "RotBonds",
                        "qed": "QED",
                    }
                    for src, dst in mapping.items():
                        if src in row:
                            try:
                                out[dst] = float(row[src])
                            except (TypeError, ValueError):
                                pass
        except Exception as exc:
            log.info("admet_runner: datamol fallback failed (%s)", exc)

    # 3) Always back-fill / cross-check with RDKit so the canonical
    #    Lipinski schema is present even if the AI backend was slow or
    #    partial.
    rdkit_out = _compute_rdkit_descriptors(can_smi)
    for k, v in rdkit_out.items():
        # Prefer AI value when present, else RDKit's.
        out.setdefault(k, v)

    if not out:
        warnings.warn(
            f"predict_admet: backend={backend} returned no descriptors for {smiles!r}"
        )
    return out


# ---------------------------------------------------------------------------
# Desirability scoring — used by the RewardAggregator channel
# ---------------------------------------------------------------------------

# Canonical Lipinski-style desirability windows.  Each rule returns
# 0.0 (outside) or its weight (inside).  Total r_admet ∈ [0, 1].
_DESIRABILITY_RULES = (
    ("logP", (1.0, 4.0), 0.30),
    ("MW", (200.0, 500.0), 0.20),
    ("HBD", (0.0, 5.0, "<="), 0.15),  # HBD ≤ 5
    ("HBA", (0.0, 10.0, "<="), 0.15),  # HBA ≤ 10
    ("TPSA", (20.0, 130.0), 0.20),
)


def admet_desirability(admet: Dict[str, float]) -> float:
    """Convert an ADMET dict into a desirability score in [0, 1].

    Uses the five Lipinski-style rules from TODO/13_lambda_clickchem
    Phase-2 / SOTA note:

    * logP ∈ [1, 4]   → +0.30
    * MW   ∈ [200, 500] → +0.20
    * HBD  ≤ 5          → +0.15
    * HBA  ≤ 10         → +0.15
    * TPSA ∈ [20, 130]  → +0.20

    Missing keys contribute 0.0; an empty dict returns 0.0.
    """
    if not admet:
        return 0.0
    score = 0.0
    for key, window, weight in _DESIRABILITY_RULES:
        if key not in admet:
            continue
        try:
            v = float(admet[key])
        except (TypeError, ValueError):
            continue
        if len(window) == 3 and window[2] == "<=":
            lo, hi, _ = window
            if v <= hi:
                score += weight
        else:
            lo, hi = window
            if lo <= v <= hi:
                score += weight
    # Clamp defensively (rounding in admet-ai can occasionally push
    # the sum marginally outside [0, 1]).
    return max(0.0, min(1.0, score))


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO)
    print(f"backend: {active_backend()}")
    for smi in ["CC(=O)Oc1ccccc1C(=O)O", "CN1C=NC2=C1C(=O)N(C(=O)N2C)C", "INVALID"]:
        out = predict_admet(smi)
        d = admet_desirability(out)
        print(f"{smi:32s} → keys={len(out)}  r_admet={d:.3f}  MW={out.get('MW')}")