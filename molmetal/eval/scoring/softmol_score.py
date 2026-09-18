"""SoftMol scoring wrapper — clone-reuse from SoftMol repo.

Source
------
``molmetal/references/SoftMol/eval_sbdd.py`` defines the canonical
hit-rate pipeline for SBBD:

    deduplicate → (QED > 0.5 & SA < 5.0) → rv > target_threshold → top-k%

with per-target thresholds for ``parp1``, ``fa7``, ``5ht1b``, ``braf``,
``jak2``. The original uses ``tdc.Oracle("qed")`` / ``tdc.Oracle("sa")``
which require the TDC chem library (NOT installed on this machine per
round-7 install report). This wrapper:

- Uses the vendored RDKit + sascorer fallback so it works WITHOUT TDC.
- Optionally calls :mod:`tdc` if the user later installs it; the metric
  keys then come from TDC (which matches the paper's reported numbers).
- Accepts a precomputed ``rv`` series (Vina ``-rv`` column) per SMILES
  since the docking path is NOT lifted (no Vina binary path).

Heavy imports are guarded; missing TDC simply falls back to RDKit.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

try:
    from rdkit import Chem
    from rdkit.Chem.QED import qed as _rdkit_qed
    _RDKIT_OK = True
except ImportError:  # pragma: no cover
    _RDKIT_OK = False

try:
    from molmetal.eval.scoring.sascorer import compute_sa_score
    _SA_OK = True
except Exception:  # pragma: no cover
    _SA_OK = False

try:
    from tdc import Oracle  # type: ignore
    _TDC_OK = True
except Exception:  # pragma: no cover
    _TDC_OK = False


AVAILABLE = _RDKIT_OK and _SA_OK


# Per-target hit thresholds copied verbatim from SoftMol/eval_sbdd.py
HIT_THR_BY_TARGET: Dict[str, float] = {
    "parp1": 10.0,
    "fa7": 8.5,
    "5ht1b": 8.7845,
    "braf": 10.3,
    "jak2": 9.1,
}

QED_THR = 0.5
SA_THR = 5.0
TOP_FRAC = 0.05


def _qed_for(smi: str) -> float:
    if _TDC_OK:
        try:
            return float(Oracle("qed")(smi))
        except Exception:
            pass
    mol = Chem.MolFromSmiles(str(smi)) if smi else None
    return float(_rdkit_qed(mol)) if mol is not None else float("nan")


def _sa_for(smi: str) -> float:
    if _TDC_OK:
        try:
            return float(Oracle("sa")(smi))
        except Exception:
            pass
    mol = Chem.MolFromSmiles(str(smi)) if smi else None
    return float(compute_sa_score(mol)) if mol is not None else float("nan")


# ---------------------------------------------------------------------------
# Public wrapper API
# ---------------------------------------------------------------------------
def score(
    smiles_list: List[str],
    pocket: Optional[Any] = None,
    rv_scores: Optional[Sequence[float]] = None,
    target: str = "parp1",
    top_frac: float = TOP_FRAC,
    **_: Any,
) -> Optional[Dict[str, Any]]:
    """Hit-rate gate + top-k% roll-up à la SoftMol.

    Parameters
    ----------
    smiles_list : list[str]
        SMILES strings. Empty / unparseable rows are skipped.
    pocket : optional, unused.
    rv_scores : optional sequence of float, same length as ``smiles_list``.
        Precomputed Vina ``-rv`` affinity (kcal/mol). Required for the
        hit-rate computation; if missing the wrapper still returns the
        QED/SA filtered counts but with ``n_hits = 0``.
    target : str
        One of ``parp1``, ``fa7``, ``5ht1b``, ``braf``, ``jak2``. The
        threshold comes from SoftMol's HIT_THR_BY_TARGET.
    top_frac : float
        Top-k fraction for the top-5% roll-up. Defaults to ``0.05``.

    Returns
    -------
    dict with keys:
        - ``n_input``, ``n_dedup``, ``n_qs`` (QED/SA-filtered),
          ``n_hits`` (rv > threshold).
        - ``top5pct_rv``, ``mean_rv``, ``top1_rv`` over hits.
        - ``top5pct_qed``, ``top5pct_sa`` over hits.
        - ``target``, ``hit_threshold``, ``qed_threshold``, ``sa_threshold``.
        - ``backend`` — ``"tdc"`` or ``"rdkit"`` to flag which oracle was used.
    """
    if not _RDKIT_OK or not _SA_OK:
        return None
    if target not in HIT_THR_BY_TARGET:
        return None
    hit_thr = HIT_THR_BY_TARGET[target]

    # --- 1. dedup ------------------------------------------------------
    cleaned: List[str] = []
    seen = set()
    for smi in smiles_list:
        if not smi:
            continue
        s = str(smi).strip()
        if not s or s in seen:
            continue
        seen.add(s)
        cleaned.append(s)
    n_dedup = len(cleaned)

    # --- 2. QED + SA ---------------------------------------------------
    qeds = [_qed_for(s) for s in cleaned]
    sas = [_sa_for(s) for s in cleaned]

    # --- 3. QED/SA filter ---------------------------------------------
    qs_mask = [(not math.isnan(q)) and (not math.isnan(s)) and (q > QED_THR) and (s < SA_THR)
               for q, s in zip(qeds, sas)]
    n_qs = sum(qs_mask)

    if n_qs == 0 or rv_scores is None:
        return {
            "n_input": len(smiles_list),
            "n_dedup": n_dedup,
            "n_qs": n_qs,
            "n_hits": 0,
            "top5pct_rv": None,
            "mean_rv": None,
            "top1_rv": None,
            "top5pct_qed": None,
            "top5pct_sa": None,
            "target": target,
            "hit_threshold": hit_thr,
            "qed_threshold": QED_THR,
            "sa_threshold": SA_THR,
            "backend": "tdc" if _TDC_OK else "rdkit",
        }

    # align rv to cleaned list (skip None entries)
    rv_iter = list(rv_scores)
    if len(rv_iter) != n_dedup:
        # caller passed unequal-length arrays — degrade gracefully
        return {
            "n_input": len(smiles_list),
            "n_dedup": n_dedup,
            "n_qs": n_qs,
            "n_hits": 0,
            "top5pct_rv": None,
            "mean_rv": None,
            "top1_rv": None,
            "top5pct_qed": None,
            "top5pct_sa": None,
            "target": target,
            "hit_threshold": hit_thr,
            "qed_threshold": QED_THR,
            "sa_threshold": SA_THR,
            "backend": "tdc" if _TDC_OK else "rdkit",
        }

    rvs = np.asarray([float(v) if v is not None else float("nan") for v in rv_iter])

    # --- 4. hits (rv > threshold) within QS subset --------------------
    hit_mask = [qs and (not math.isnan(rv)) and (rv > hit_thr)
                for qs, rv in zip(qs_mask, rvs)]
    hit_rv = rvs[hit_mask]
    n_hits = int(hit_mask.count(True))
    if n_hits == 0:
        return {
            "n_input": len(smiles_list),
            "n_dedup": n_dedup,
            "n_qs": n_qs,
            "n_hits": 0,
            "top5pct_rv": None,
            "mean_rv": None,
            "top1_rv": None,
            "top5pct_qed": None,
            "top5pct_sa": None,
            "target": target,
            "hit_threshold": hit_thr,
            "qed_threshold": QED_THR,
            "sa_threshold": SA_THR,
            "backend": "tdc" if _TDC_OK else "rdkit",
        }

    sorted_idx = np.argsort(hit_rv)[::-1]
    hit_rv_sorted = hit_rv[sorted_idx]
    k_top5 = max(1, int(math.ceil(n_hits * top_frac)))
    k_top5 = min(k_top5, n_hits)

    cleaned_arr = np.asarray(cleaned)
    hit_smiles_sorted = cleaned_arr[hit_mask][sorted_idx]
    # Re-score QED/SA on the sorted hit subset to get top-k% means. Re-using
    # the original qeds[] array would double-count duplicates across the
    # mask, so it's cleaner to call the oracle on the unique top-k SMILES.
    top5_qed = float(np.mean([_qed_for(s) for s in hit_smiles_sorted[:k_top5]]))
    top5_sa = float(np.mean([_sa_for(s) for s in hit_smiles_sorted[:k_top5]]))

    return {
        "n_input": len(smiles_list),
        "n_dedup": n_dedup,
        "n_qs": n_qs,
        "n_hits": n_hits,
        "top5pct_rv": float(hit_rv_sorted[:k_top5].mean()),
        "mean_rv": float(hit_rv_sorted.mean()),
        "top1_rv": float(hit_rv_sorted[0]),
        "top5pct_qed": top5_qed,
        "top5pct_sa": top5_sa,
        "target": target,
        "hit_threshold": hit_thr,
        "qed_threshold": QED_THR,
        "sa_threshold": SA_THR,
        "backend": "tdc" if _TDC_OK else "rdkit",
    }


__all__ = ["score", "HIT_THR_BY_TARGET", "QED_THR", "SA_THR", "TOP_FRAC"]
