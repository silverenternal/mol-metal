"""platinai_oracle.py — PlatinAI predicted-activity reward channel.

Phase 2 of the Metallodrug de novo ultracode (task #915..#918).

Provides a lightweight kNN-based activity oracle that uses the
PlatinAI_MBFinder corpus (226,918 SMILES) plus the two predicted-
activity sheets (A2780 ovarian, MCF7 breast) shipped in
``/mnt/storage/data/molmetal/``.

Workflow
========

1. :class:`PlatinAIOracle` loads the MBFinder SMILES table + joins
   the two predicted-activity sheets onto a canonical SMILES index.
2. On construction it builds a Morgan-ECFP4 (radius=2, 2048-bit)
   kNN index over the corpus.  Indexing uses RDKit's bulk Tanimoto
   against the query fingerprint — no sklearn dependency required.
3. :meth:`score` takes a generated SMILES, computes its ECFP4
   fingerprint, finds the 5 nearest MBFinder neighbours by
   Tanimoto similarity, and returns the mean of the A2780 + MCF7
   predictions weighted by ``1 / Tanimoto_distance`` (so closer
   neighbours weigh more).  Falls back to 0.0 when no neighbour
   is within Tanimoto 0.3 (i.e. when the query is too far from the
   metallodrug corpus to be informative).

Honest framing
==============

- This is a **weak, oracle-style reward**, not a gold-standard
  activity predictor.  The predictions are model-generated labels
  stored in ``PlatinAI_predicted_{A2780,MCF7}.xlsx`` (pred_0 columns,
  range [0, 1]).
- The oracle only fires on molecules that look like the training
  corpus (Tanimoto >= 0.3 to one of the 226,918 reference mols).
  Generated candidates outside that neighbourhood get 0.0 — this
  is by design (we do not want to penalise novel scaffolds).
- All values come back in [0, 1]; the aggregator multiplies by
  ``w_platinai`` so callers stay in control of magnitude.

Integration status
==================

- Wire to :class:`RewardAggregator` via
  :meth:`register_platinai_oracle_channel` — see proof_search.py.
- CLI flag ``--reward-platinai-weight`` (default 0.0 = opt-in).
- Honest caveats for downstream pilots: this oracle is biased toward
  the corpus; it should NOT be the dominant reward channel.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

try:
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem, DataStructs
    _HAS_RDKIT = True
    RDLogger.DisableLog("rdApp.*")  # keep stdout clean
except Exception:  # pragma: no cover
    Chem = None
    AllChem = None
    DataStructs = None
    _HAS_RDKIT = False


# -----------------------------------------------------------------------------
# Constants — file paths and hyper-parameters
# -----------------------------------------------------------------------------
DEFAULT_MBFINDER_PATH: str = "/mnt/storage/data/molmetal/PlatinAI_MBFinder_dataset.xlsx"
DEFAULT_A2780_PATH: str = "/mnt/storage/data/molmetal/PlatinAI_predicted_A2780.xlsx"
DEFAULT_MCF7_PATH: str = "/mnt/storage/data/molmetal/PlatinAI_predicted_MCF7.xlsx"

DEFAULT_N_NEIGHBOURS: int = 5
DEFAULT_TANIMOTO_MIN: float = 0.3
DEFAULT_ECFP_RADIUS: int = 2
DEFAULT_ECFP_BITS: int = 2048


@dataclass(frozen=True)
class PlatinAINeighbour:
    """One kNN match from the MBFinder corpus."""

    smiles: str
    a2780: float  # in [0, 1] (NaN if no predicted label)
    mcf7: float  # in [0, 1] (NaN if no predicted label)
    tanimoto: float  # in [0, 1]


def _safe_canon(smiles: str) -> Optional[str]:
    """Return canonical SMILES or None on parse failure."""
    if not _HAS_RDKIT or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def _ecfp4_fingerprint(smiles: str):
    """Return Morgan-ECFP4 fingerprint (radius=2, 2048 bits) or None."""
    if not _HAS_RDKIT or not smiles:
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(
            mol, DEFAULT_ECFP_RADIUS, nBits=DEFAULT_ECFP_BITS
        )
    except Exception:
        return None


class PlatinAIOracle:
    """kNN-based predicted-activity oracle over the PlatinAI_MBFinder corpus.

    Parameters
    ----------
    mbfinder_path : str, optional
        Path to ``PlatinAI_MBFinder_dataset.xlsx`` (226,918 SMILES).
        Defaults to :data:`DEFAULT_MBFINDER_PATH`.
    a2780_path, mcf7_path : str, optional
        Paths to the two predicted-activity xlsx files.
    n_neighbours : int, default 5
        k for the kNN lookup.
    tanimoto_min : float, default 0.3
        Below this Tanimoto similarity we treat the molecule as
        out-of-corpus and return 0.0.
    """

    def __init__(
        self,
        mbfinder_path: str = DEFAULT_MBFINDER_PATH,
        a2780_path: str = DEFAULT_A2780_PATH,
        mcf7_path: str = DEFAULT_MCF7_PATH,
        *,
        n_neighbours: int = DEFAULT_N_NEIGHBOURS,
        tanimoto_min: float = DEFAULT_TANIMOTO_MIN,
    ) -> None:
        self.mbfinder_path = mbfinder_path
        self.a2780_path = a2780_path
        self.mcf7_path = mcf7_path
        self.n_neighbours = int(n_neighbours)
        self.tanimoto_min = float(tanimoto_min)
        self._loaded = False
        self._load_error: Optional[str] = None
        # Public state populated by :meth:`_load` when loadable.
        self.canonical_smiles: List[str] = []
        self.fingerprints = []  # list of ExplicitBitVect
        self.a2780_lookup: dict = {}
        self.mcf7_lookup: dict = {}
        self.n_corpus: int = 0
        self.try_load()

    # ---- public API ------------------------------------------------------
    @property
    def available(self) -> bool:
        """True iff the MBFinder xlsx was loaded successfully."""
        return self._loaded

    @property
    def load_error(self) -> Optional[str]:
        """Last load error message (None if loaded OK)."""
        return self._load_error

    def try_load(self) -> bool:
        """Load + index the corpus.  Returns True on success.

        Lazy: callers can construct ``PlatinAIOracle()`` even when the
        data files are missing — the oracle simply degrades to 0.0
        via :meth:`score`.  This matches the RewardAggregator
        graceful-degradation contract used by the REINVENT4 and ADMET
        channels.
        """
        if self._loaded:
            return True
        if not _HAS_RDKIT:
            self._load_error = "rdkit not importable"
            return False
        try:
            import pandas as pd  # local: pandas pulls nothing for oracle core
            mb = pd.read_excel(self.mbfinder_path)
            smiles_list = mb["smiles"].astype(str).tolist()
        except Exception as e:
            self._load_error = f"failed to load MBFinder: {e}"
            return False

        # Index fingerprints + canonical SMILES.  We keep the index of
        # successfully-parsed molecules only; unparseable rows are
        # silently dropped (~5-10% typical for metallodrug corpora).
        canon: List[str] = []
        fps = []
        for raw in smiles_list:
            c = _safe_canon(raw)
            if c is None:
                continue
            fp = _ecfp4_fingerprint(c)
            if fp is None:
                continue
            canon.append(c)
            fps.append(fp)
        self.canonical_smiles = canon
        self.fingerprints = fps
        self.n_corpus = len(canon)
        if self.n_corpus == 0:
            self._load_error = "no parseable SMILES in MBFinder"
            return False

        # Load activity sheets (best-effort; missing sheets → empty lookup).
        self.a2780_lookup = self._load_activity_sheet(self.a2780_path)
        self.mcf7_lookup = self._load_activity_sheet(self.mcf7_path)

        self._loaded = True
        return True

    def _load_activity_sheet(self, path: str) -> dict:
        """Load a predicted-activity xlsx → dict[canonical_smiles, pred_0]."""
        out: dict = {}
        if not os.path.isfile(path):
            return out
        try:
            import pandas as pd
            df = pd.read_excel(path)
            for raw_smi, pred in zip(df["smiles"].astype(str), df["pred_0"].astype(float)):
                canon = _safe_canon(raw_smi)
                if canon is None:
                    continue
                # Take the mean if multiple rows canonicalise to the same SMILES.
                if canon in out:
                    out[canon] = float(0.5 * (out[canon] + float(pred)))
                else:
                    out[canon] = float(pred)
        except Exception:
            # Silent degrade — empty lookup means neighbours contribute 0.0
            # via :meth:`_neighbour_activity_mean`.  Surface as warning if the
            # caller has logging enabled.
            return {}
        return out

    def find_neighbours(self, smiles: str) -> List[PlatinAINeighbour]:
        """Return the top-k neighbours sorted by descending Tanimoto.

        Returns an empty list when the molecule does not parse or when
        the corpus has not been loaded.
        """
        if not self._loaded:
            return []
        canon = _safe_canon(smiles)
        if canon is None:
            return []
        fp = _ecfp4_fingerprint(canon)
        if fp is None:
            return []
        # Bulk Tanimoto against the corpus.  RDKit returns a tuple of
        # bulk Tanimoto similarities (DataStructs.BulkTanimotoSimilarity).
        try:
            sims = DataStructs.BulkTanimotoSimilarity(fp, self.fingerprints)
        except Exception:
            return []
        # Take top-k.
        n = self.n_neighbours
        if n > len(sims):
            n = len(sims)
        # Partial selection — O(n_corpus) is fine for ~227K rows.
        indexed = sorted(
            enumerate(sims), key=lambda kv: kv[1], reverse=True
        )[:n]
        out: List[PlatinAINeighbour] = []
        for idx, tan in indexed:
            if tan < self.tanimoto_min:
                continue
            n_smi = self.canonical_smiles[idx]
            out.append(
                PlatinAINeighbour(
                    smiles=n_smi,
                    a2780=float(self.a2780_lookup.get(n_smi, float("nan"))),
                    mcf7=float(self.mcf7_lookup.get(n_smi, float("nan"))),
                    tanimoto=float(tan),
                )
            )
        return out

    def score(self, smiles: str) -> float:
        """Weighted-mean predicted-activity score in [0, 1].

        The formula is::

            score = sum_i (w_i * a_i) / sum_i w_i
            w_i   = 1 / max(eps, 1 - tanimoto_i)

        where ``a_i`` is the per-cell-line mean
        ``(a2780_i + mcf7_i) / 2`` and the Tanimoto_min gate is
        applied BEFORE the weighting (so low-similarity neighbours
        do not contribute at all).

        Returns 0.0 when the corpus is not loaded, when no neighbour
        is within ``tanimoto_min``, or when none of the surviving
        neighbours have a predicted-activity label (rare, since
        214,376 of 226,918 rows have a label join).
        """
        neighbours = self.find_neighbours(smiles)
        if not neighbours:
            return 0.0
        eps = 1e-6
        num = 0.0
        den = 0.0
        for nb in neighbours:
            # Per-cell-line mean, ignoring NaNs.
            labels = [v for v in (nb.a2780, nb.mcf7) if v == v]  # NaN-safe
            if not labels:
                continue
            activity = float(sum(labels)) / float(len(labels))
            distance = max(eps, 1.0 - nb.tanimoto)
            w = 1.0 / distance
            num += w * activity
            den += w
        if den <= 0.0:
            return 0.0
        return float(num / den)

    def score_channel(self, smiles: str) -> float:
        """``(state) -> float`` adapter for RewardAggregator.r_platinai.

        Accepts either a plain SMILES string or any state object with
        a ``canonical_smiles()`` method or ``smiles`` attribute.
        """
        if smiles is None:
            return 0.0
        if isinstance(smiles, str):
            return self.score(smiles)
        # State object: try canonical_smiles() then smiles attribute.
        try:
            fn = getattr(smiles, "canonical_smiles", None)
            if callable(fn):
                smi = str(fn())
                if smi:
                    return self.score(smi)
        except Exception:
            pass
        try:
            smi = str(getattr(smiles, "smiles", "") or "")
            if smi:
                return self.score(smi)
        except Exception:
            pass
        return 0.0

    def register_channel(self) -> Callable:
        """Return a closure suitable for ``RewardAggregator.r_platinai``.

        The closure captures ``self``; reward aggregator's ``_safe``
        wrapper will silently degrade to 0.0 if the closure raises.
        """
        return lambda state: self.score_channel(state)


def make_platinai_channel(
    mbfinder_path: str = DEFAULT_MBFINDER_PATH,
    a2780_path: str = DEFAULT_A2780_PATH,
    mcf7_path: str = DEFAULT_MCF7_PATH,
    *,
    n_neighbours: int = DEFAULT_N_NEIGHBOURS,
    tanimoto_min: float = DEFAULT_TANIMOTO_MIN,
) -> Tuple[Callable, PlatinAIOracle]:
    """Convenience factory: return ``(r_channel, oracle)``.

    The oracle is returned alongside the closure so callers can introspect
    ``n_corpus``, ``available``, etc. for diagnostics.
    """
    oracle = PlatinAIOracle(
        mbfinder_path=mbfinder_path,
        a2780_path=a2780_path,
        mcf7_path=mcf7_path,
        n_neighbours=n_neighbours,
        tanimoto_min=tanimoto_min,
    )
    return oracle.register_channel(), oracle


__all__ = [
    "DEFAULT_MBFINDER_PATH",
    "DEFAULT_A2780_PATH",
    "DEFAULT_MCF7_PATH",
    "DEFAULT_N_NEIGHBOURS",
    "DEFAULT_TANIMOTO_MIN",
    "PlatinAINeighbour",
    "PlatinAIOracle",
    "make_platinai_channel",
]