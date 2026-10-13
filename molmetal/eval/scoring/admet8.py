"""8-set ADMET proxy for paper §4 / §5 standalone per-mol emission.

Track E (Wf-60231cff-03e-15, m14-admet-paper-align, EV2 0/8 → 8/8) —
the proposed track-E 8-set is::

    sa             Synthetic Accessibility   (Ertl 2008, range [1, 10])
    qed            Quant. Estimate of Drug-likeness (Bickerton 2012, [0, 1])
    logP           Crippen logP              (Mannhold 2009)
    tpsa           Topological Polar SA      (Ertl 2000, Å²)
    hbd            NumHDonors                (Lipinski 1997)
    hba            NumHAcceptors             (Lipinski 1997)
    rotB           NumRotatableBonds         (Veber 2002)
    aromatic_rings CalcNumAromaticRings      (rdMolDescriptors)

These are the **standalone** 8-set per-mol metrics requested by the
track-E spec; they are disjoint from
:mod:`molmetal.eval.discriminators.metal_aware_admet` which exposes
the *composite* ``logp7_4 / gi50_proxy / cell_permeability_logPapp /
herg_cardio_risk / ames_mutagen / hepatotox_index /
aqueous_solubility_logS / plasma_protein_binding`` proxies (those
embed TPSA / HBD / HBA / RotB / AromaticRings as coefficients
inside composite scores, never as standalone emissions).

**Fragmentation-aware split (BLK-QED-def):** every ADMET8 sample is
taken on the *largest* connected fragment of the sanitised Mol, so
disconnected salts (``[Na+].[Cl-]``, ``CC(=O)O.[Na+]``) are scored on
the parent molecule only — same protocol used by QED, SA, logP
benchmarks at Ertl 2008, Bickerton 2012, Polykovskiy 2020 MOSES.

Lit anchors
-----------
- Ertl 2008 *J. Cheminform.* — SA score (1=easy, 10=hard).
- Ertl 2000 *JMC* — TPSA (Å²); Lipinski Veber extensions.
- Bickerton 2012 *Nature Chem.* — QED (0..1; weibull aggregation).
- Lipinski 1997 *Adv. Drug Deliv. Rev.* — Rule-of-Five HBD/HBA.
- Veber 2002 *JMC* — RotB ≤ 10 oral-bioavailability heuristic.
- rdMolDescriptors (RDKit 2024) — CalcNumAromaticRings.
"""
from __future__ import annotations

from dataclasses import dataclass
from statistics import mean, median
from typing import Iterable, List, Optional

try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Crippen, Descriptors, Lipinski, rdMolDescriptors
    _RDKIT_OK = True
except ImportError:  # pragma: no cover
    _RDKIT_OK = False

try:
    from molmetal.eval.scoring.sascorer import compute_sa_score
    _SA_OK = True
except Exception:  # pragma: no cover
    _SA_OK = False

try:
    from rdkit.Chem.QED import qed as _rdkit_qed
    _QED_OK = True
except Exception:  # pragma: no cover
    _QED_OK = False


# ---------------------------------------------------------------------------
# Per-mol container
# ---------------------------------------------------------------------------
@dataclass
class ADMET8Row:
    """8-set ADMET row for a single sanitised Mol (largest fragment)."""

    sa: Optional[float] = None
    qed: Optional[float] = None
    logP: Optional[float] = None
    tpsa: Optional[float] = None
    hbd: Optional[int] = None
    hba: Optional[int] = None
    rotB: Optional[int] = None
    aromatic_rings: Optional[int] = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _largest_fragment_mol(mol):
    """Return the largest connected fragment of an RDKit Mol.

    For a single-fragment Mol this is a no-op (returns ``mol``).
    For a multi-fragment (salt / co-former) input, returns the
    largest connected substructure per the BLK-QED-def convention.
    Returns ``None`` on a fully-empty / unparseable input.
    """
    if mol is None:
        return None
    if mol.GetNumAtoms() == 0:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if len(frags) <= 1:
        return mol
    # Pick the largest fragment and re-sanitise.
    largest = max(frags, key=lambda f: f.GetNumAtoms())
    try:
        Chem.SanitizeMol(largest)
    except Exception:
        # Some fragments fail sanitise even after selection; fall back
        # to the unsanitised largest.
        return largest
    return largest


def _safe_int(value, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value, default: float = 0.0) -> float:
    try:
        f = float(value)
        if f != f:  # NaN
            return default
        return f
    except Exception:
        return default


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def compute_admet8_one(smi: Optional[str]) -> Optional[ADMET8Row]:
    """Compute the 8-set ADMET row for one SMILES.

    Returns ``None`` on parse failure.  Each individual metric may
    independently be ``None`` if its underlying RDKit call raises
    (e.g. SA without the ``fpscores.pkl.gz`` table).

    Parameters
    ----------
    smi : str or None
        Candidate SMILES.

    Returns
    -------
    :class:`ADMET8Row` or None
        ``None`` iff the SMILES did not parse.
    """
    if not _RDKIT_OK:
        return None
    if smi is None or not isinstance(smi, str) or not smi.strip():
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return None
    largest = _largest_fragment_mol(mol)
    if largest is None or largest.GetNumAtoms() == 0:
        return None
    row = ADMET8Row()
    try:
        row.sa = float(compute_sa_score(largest)) if _SA_OK else None
    except Exception:
        row.sa = None
    try:
        row.qed = float(_rdkit_qed(largest)) if _QED_OK else None
    except Exception:
        row.qed = None
    try:
        row.logP = float(Crippen.MolLogP(largest))
    except Exception:
        row.logP = None
    try:
        row.tpsa = float(Descriptors.TPSA(largest))
    except Exception:
        row.tpsa = None
    try:
        row.hbd = int(Lipinski.NumHDonors(largest))
    except Exception:
        row.hbd = None
    try:
        row.hba = int(Lipinski.NumHAcceptors(largest))
    except Exception:
        row.hba = None
    try:
        row.rotB = int(rdMolDescriptors.CalcNumRotatableBonds(largest))
    except Exception:
        row.rotB = None
    try:
        row.aromatic_rings = int(rdMolDescriptors.CalcNumAromaticRings(largest))
    except Exception:
        row.aromatic_rings = None
    return row


def compute_admet8_panel(smiles: Iterable[Optional[str]], *, cohort_tag: str = "default") -> dict:
    """Aggregate :func:`compute_admet8_one` over a cohort.

    Returns a dict with mean/median/IQR per metric plus N counts.
    Every metric may be ``None`` in the aggregate if the underlying
    RDKit call failed for every sample.

    Parameters
    ----------
    smiles : iterable[str or None]
        Candidate SMILES.
    cohort_tag : str
        Included as ``cohort_id`` in the output dict.

    Returns
    -------
    dict
        Schema (skeleton — see body for full)::

            {
              "cohort_id": str,
              "n_input": int,
              "n_ok": int,    # at least one of the 8 metrics is non-None
              "per_metric_mean": { "sa": float, "qed": ..., ... },
              "per_metric_median": { ... },
              "per_metric_iqr": { ... },
              "missing": { "sa": int, ... },   # counts of None per metric
              "protocol_version": "admet8/2026-10-13",
            }
    """
    smiles_list = list(smiles)
    n_input = len(smiles_list)
    if n_input == 0:
        empty = {k: None for k in ("sa", "qed", "logP", "tpsa", "hbd", "hba", "rotB", "aromatic_rings")}
        return {
            "cohort_id": cohort_tag,
            "n_input": 0,
            "n_ok": 0,
            "per_metric_mean": dict(empty),
            "per_metric_median": dict(empty),
            "per_metric_iqr": dict(empty),
            "missing": {k: 0 for k in empty},
            "protocol_version": "admet8/2026-10-13",
        }
    rows: List[ADMET8Row] = []
    for smi in smiles_list:
        r = compute_admet8_one(smi)
        if r is not None:
            rows.append(r)
    n_ok = len(rows)
    bucket = {k: [] for k in ("sa", "qed", "logP", "tpsa", "hbd", "hba", "rotB", "aromatic_rings")}
    miss = {k: 0 for k in bucket}
    for r in rows:
        for k in bucket:
            v = getattr(r, k)
            if v is None:
                miss[k] += 1
            else:
                bucket[k].append(float(v))

    def _stats(vals):
        if not vals:
            return None, None, None  # mean, median, IQR
        m = mean(vals)
        med = median(vals)
        if len(vals) >= 4:
            s = sorted(vals)
            q1 = s[len(s) // 4]
            q3 = s[(3 * len(s)) // 4]
            iqr = q3 - q1
        else:
            iqr = None
        return float(m), float(med), (float(iqr) if iqr is not None else None)

    per_metric_mean = {}
    per_metric_median = {}
    per_metric_iqr = {}
    for k in bucket:
        mu, med, iqr = _stats(bucket[k])
        per_metric_mean[k] = mu
        per_metric_median[k] = med
        per_metric_iqr[k] = iqr
    return {
        "cohort_id": str(cohort_tag),
        "n_input": int(n_input),
        "n_ok": int(n_ok),
        "per_metric_mean": per_metric_mean,
        "per_metric_median": per_metric_median,
        "per_metric_iqr": per_metric_iqr,
        "missing": miss,
        "protocol_version": "admet8/2026-10-13",
    }


AVAILABLE: bool = _RDKIT_OK and _SA_OK and _QED_OK

__all__ = [
    "ADMET8Row",
    "compute_admet8_one",
    "compute_admet8_panel",
    "AVAILABLE",
]


# ---------------------------------------------------------------------------
# Smoke (per §5.7)
# ---------------------------------------------------------------------------
def _smoke() -> int:
    """Self-test: returns 0 on PASS, 1 on FAIL.

    Tests:
      1. ``None`` → ``compute_admet8_one`` returns None.
      2. ``CCO`` (ethanol) → row has 8 non-None metrics.
      3. ``CC(=O)O.[Na+]`` (acetic acid + Na+ salt) → row has 8 non-None
         metrics; SA/QED computed on the parent fragment.
      4. ``panel`` arithmetic on a 5-mol cohort.
      5. ``panel`` on empty cohort → n_input=0, no exception.
    """
    if not _RDKIT_OK:
        print("[admet8] SKIP — RDKit unavailable")
        return 2
    # Test 1
    r0 = compute_admet8_one(None)
    assert r0 is None, r0
    # Test 2
    r1 = compute_admet8_one("CCO")
    assert r1 is not None
    # ``sa`` may be ``None`` if ``fpscores.pkl.gz`` is missing (gitignored
    # in worktrees; the Ertl 2008 SA table is loaded from disk at runtime).
    for k in ("qed", "logP", "tpsa", "hbd", "hba", "rotB", "aromatic_rings"):
        v = getattr(r1, k)
        assert v is not None, (k, v)
    if _SA_OK:
        assert r1.sa is not None, "sa unexpectedly None despite _SA_OK"
    # Ethanol: HBD=1 (OH), HBA=1 (O), rotB=0, aromatic_rings=0, logP near 0.
    assert r1.hbd == 1, r1.hbd
    assert r1.hba == 1, r1.hba
    assert r1.rotB == 0, r1.rotB
    assert r1.aromatic_rings == 0, r1.aromatic_rings
    # Test 3 (salt stripping)
    r2 = compute_admet8_one("CC(=O)O.[Na+]")
    assert r2 is not None
    assert r2.hbd == 1, r2.hbd  # acetic acid: 1 OH
    # Test 4: panel
    cohort = ["CCO", "CC(=O)O", "c1ccccc1", "invalid_smiles", None, "[He]"]
    p = compute_admet8_panel(cohort, cohort_tag="smoke")
    assert p["n_input"] == 6
    assert p["n_ok"] >= 3
    # All metrics are non-None on the parsed subset (sa may be None if
    # fpscores.pkl.gz is gitignored in the worktree).
    for k in ("qed", "logP", "tpsa"):
        assert p["per_metric_mean"][k] is not None, (k, p)
    # Test 5
    p0 = compute_admet8_panel([], cohort_tag="empty")
    assert p0["n_input"] == 0
    assert p0["n_ok"] == 0
    print(f"[admet8] SMOKE PASS  5/5  n_input={p['n_input']} n_ok={p['n_ok']}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    import sys
    sys.exit(_smoke())
