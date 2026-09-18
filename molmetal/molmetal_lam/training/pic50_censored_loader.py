"""Censor-aware pIC50 dataset for D-MPNN retraining (WF-Extra-1).

Reads the MetalCytoToxDB export (or its audit-derived conditioned CSV) and
returns one row per formulation with an explicit ``is_censored`` flag plus
the bound value (for right-censored rows such as "IC50 > 10 uM" the bound
is 10.0; for left-censored rows such as "IC50 < 0.1 uM" the bound is 0.1).

Censored rows are NEVER silently re-coded to exact targets. The training
loop is expected to consume the (target, is_censored, bound_value) triple
via a right-censored margin / Huber-on-bound loss. Rows whose SMILES do
not parse under RDKit are dropped with a logged warning rather than
raised — this keeps a single corrupt ligand from killing the sweep.
"""
from __future__ import annotations

import logging
import re
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Column contract (from molmetal/scripts/audit_pic50_assay_data.py)
# ---------------------------------------------------------------------------
SMILES_COL = "SMILES_Ligands"
RAW_IC50_COL = "IC50_Dark(M*10^-6)"  # contains "<", ">", "≤", "≥" for censored
VALUE_IC50_COL = "IC50_Dark_value"   # already numeric (uM)

# Formulation meta columns (preserve verbatim into the dataset row)
FORMULATION_META_COLS: Tuple[str, ...] = (
    "Metal",
    "Cell_line",
    "Time(h)",
    "DOI",
    "Counterion",
    "Oxidation_state",
    "Charge_complex",
)


# ---------------------------------------------------------------------------
# Bound extraction
# ---------------------------------------------------------------------------
_BOUND_RE = re.compile(r"([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")


def _extract_bound(raw_value: Any) -> Optional[float]:
    """Pull the numeric bound out of a raw cell.

    Examples
    --------
    >>> _extract_bound(">10")          # 10.0 (upper bound)
    >>> _extract_bound("> 10.5 uM")    # 10.5
    >>> _extract_bound("<0.1")         # 0.1
    >>> _extract_bound("5.3")          # None (not censored)
    >>> _extract_bound(None)           # None
    """
    if raw_value is None or (isinstance(raw_value, float) and np.isnan(raw_value)):
        return None
    text = str(raw_value)
    if not any(sym in text for sym in ("<", ">", "≤", "≥")):
        return None
    match = _BOUND_RE.search(text)
    if match is None:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def _detect_censor(raw_value: Any) -> bool:
    """Return True iff the raw IC50 cell carries a censor symbol."""
    if raw_value is None or (isinstance(raw_value, float) and np.isnan(raw_value)):
        return False
    return any(sym in str(raw_value) for sym in ("<", ">", "≤", "≥"))


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
@dataclass
class CensoredPIC50Row:
    """One formulation row. Mirrors the loader's __getitem__ contract."""

    smiles: str
    target: float             # numeric IC50 in uM (for uncensored) or the bound (best-effort)
    is_censored: bool         # True iff raw cell carries <, >, ≤, ≥
    bound_value: float        # the numeric bound (upper for ">X", lower for "<X")
    formulation_id: str       # stable hash for traceability
    formulation_meta: dict = field(default_factory=dict)


class CensoredPIC50Dataset(Dataset):
    """Censor-aware pIC50 dataset.

    Parameters
    ----------
    frame
        DataFrame already loaded from the audit export or conditioned CSV.
        Required columns: ``SMILES_Ligands``, ``IC50_Dark(M*10^-6)``,
        ``IC50_Dark_value``. Formulation meta is best-effort; missing
        meta columns are filled with ``None``.
    smiles_col / raw_col / value_col
        Column-name overrides (defaults match the audit script).
    formulation_meta_cols
        Iterable of column names to preserve verbatim into
        ``CensoredPIC50Row.formulation_meta``.
    drop_unparseable
        If True (default), invalid SMILES are dropped with a warning.
        If False, they are kept but ``smiles`` becomes ``""`` — callers
        that need strict behaviour can override.
    """

    def __init__(
        self,
        frame: pd.DataFrame,
        *,
        smiles_col: str = SMILES_COL,
        raw_col: str = RAW_IC50_COL,
        value_col: str = VALUE_IC50_COL,
        formulation_meta_cols: Iterable[str] = FORMULATION_META_COLS,
        drop_unparseable: bool = True,
    ) -> None:
        super().__init__()
        self._smiles_col = smiles_col
        self._raw_col = raw_col
        self._value_col = value_col
        self._meta_cols: Tuple[str, ...] = tuple(formulation_meta_cols)
        self._drop_unparseable = drop_unparseable
        self._rows: List[CensoredPIC50Row] = self._build(frame)
        if self._rows:
            self._n_censored_total = int(sum(r.is_censored for r in self._rows))
        else:
            self._n_censored_total = 0

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build(self, frame: pd.DataFrame) -> List[CensoredPIC50Row]:
        # Lazy import — RDKit is heavy and some unit tests only need the
        # bound-extraction helpers.
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")

        rows: List[CensoredPIC50Row] = []
        n_dropped = 0
        n_invalid_smiles = 0

        for _, src in frame.iterrows():
            raw_smi = src.get(self._smiles_col)
            if raw_smi is None or (isinstance(raw_smi, float) and np.isnan(raw_smi)):
                n_dropped += 1
                continue
            smi_text = str(raw_smi).strip()
            if not smi_text:
                n_dropped += 1
                continue

            try:
                mol = Chem.MolFromSmiles(smi_text)
            except Exception as exc:  # pragma: no cover - defensive
                if self._drop_unparseable:
                    warnings.warn(
                        f"dropping SMILES that crashed RDKit parser: {smi_text!r} ({exc})",
                        stacklevel=2,
                    )
                    n_invalid_smiles += 1
                    continue
                mol = None

            if mol is None or mol.GetNumAtoms() == 0:
                if self._drop_unparseable:
                    warnings.warn(
                        f"dropping unparseable SMILES: {smi_text!r}",
                        stacklevel=2,
                    )
                    n_invalid_smiles += 1
                    continue
                canonical = ""
            else:
                canonical = Chem.MolToSmiles(mol, isomericSmiles=True)

            raw_cell = src.get(self._raw_col)
            value_cell = src.get(self._value_col)
            is_censored = _detect_censor(raw_cell)
            bound = _extract_bound(raw_cell)

            if is_censored and bound is None:
                warnings.warn(
                    f"dropping censored row with no parseable bound: raw={raw_cell!r}",
                    stacklevel=2,
                )
                n_invalid_smiles += 1
                continue

            if not is_censored:
                # Exact value path — value_col is already uM, but guard against NaN.
                if value_cell is None or (
                    isinstance(value_cell, float) and np.isnan(value_cell)
                ):
                    n_dropped += 1
                    continue
                try:
                    target_uM = float(value_cell)
                except (TypeError, ValueError):
                    n_dropped += 1
                    continue
                if target_uM <= 0:
                    # pIC50 undefined for non-positive IC50; drop rather than poison training.
                    n_dropped += 1
                    continue
                bound_value = target_uM  # exact → bound == target
            else:
                # Censored — never re-code to an exact label.
                bound_value = float(bound)  # type: ignore[arg-type]
                # For numerical sanity we still expose *some* target value so that
                # __getitem__ can yield a 4-tuple without NaNs; downstream loss
                # functions must branch on ``is_censored`` and ignore it for fit.
                target_uM = bound_value

            meta = {}
            for col in self._meta_cols:
                if col in src.index:
                    val = src[col]
                    if isinstance(val, float) and np.isnan(val):
                        val = None
                    meta[col] = val
            formulation_id = self._formulation_id(canonical, meta)

            rows.append(
                CensoredPIC50Row(
                    smiles=canonical,
                    target=float(target_uM),
                    is_censored=bool(is_censored),
                    bound_value=float(bound_value),
                    formulation_id=formulation_id,
                    formulation_meta=meta,
                )
            )

        if n_dropped:
            logger.info(
                "CensoredPIC50Dataset: dropped %d rows missing SMILES/IC50", n_dropped
            )
        if n_invalid_smiles:
            logger.info(
                "CensoredPIC50Dataset: dropped %d rows with unparseable SMILES or bound",
                n_invalid_smiles,
            )
        return rows

    @staticmethod
    def _formulation_id(smiles: str, meta: dict) -> str:
        """Stable per-formulation identifier (metal + cell + time + oxidation + charge + ligand)."""
        parts = [
            str(meta.get("Metal", "")),
            str(meta.get("Cell_line", "")),
            str(meta.get("Time(h)", "")),
            str(meta.get("Oxidation_state", "")),
            str(meta.get("Charge_complex", "")),
            str(meta.get("Counterion", "")),
            smiles,
        ]
        return "|".join(parts)

    # ------------------------------------------------------------------
    # Convenience constructors
    # ------------------------------------------------------------------
    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        *,
        smiles_col: str = SMILES_COL,
        raw_col: str = RAW_IC50_COL,
        value_col: str = VALUE_IC50_COL,
        formulation_meta_cols: Iterable[str] = FORMULATION_META_COLS,
        drop_unparseable: bool = True,
        filters: Optional[dict] = None,
    ) -> "CensoredPIC50Dataset":
        frame = pd.read_csv(path, low_memory=False)
        if filters:
            for col, expected in filters.items():
                if col not in frame.columns:
                    raise KeyError(f"filter column missing: {col}")
                frame = frame[frame[col].eq(expected)]
        return cls(
            frame,
            smiles_col=smiles_col,
            raw_col=raw_col,
            value_col=value_col,
            formulation_meta_cols=formulation_meta_cols,
            drop_unparseable=drop_unparseable,
        )

    # ------------------------------------------------------------------
    # Dataset protocol
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> Tuple[str, float, bool, float]:
        row = self._rows[idx]
        return row.smiles, row.target, row.is_censored, row.bound_value

    def get_row(self, idx: int) -> CensoredPIC50Row:
        """Return the full row (including formulation_id + meta), not just the 4-tuple."""
        return self._rows[idx]

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    @property
    def n_censored_total(self) -> int:
        """Total censored rows that survived construction (test/diagnostic use)."""
        return self._n_censored_total

    def formulation_meta(self, idx: int) -> dict:
        return dict(self._rows[idx].formulation_meta)

    def formulation_ids(self) -> List[str]:
        return [r.formulation_id for r in self._rows]

    def to_frame(self) -> pd.DataFrame:
        """Project the dataset back into a flat DataFrame (debug/audit only)."""
        records = []
        for r in self._rows:
            rec = {
                "smiles": r.smiles,
                "target_uM": r.target,
                "is_censored": r.is_censored,
                "bound_value_uM": r.bound_value,
                "formulation_id": r.formulation_id,
            }
            rec.update(r.formulation_meta)
            records.append(rec)
        return pd.DataFrame.from_records(records)


__all__: Sequence[str] = (
    "CensoredPIC50Dataset",
    "CensoredPIC50Row",
    "SMILES_COL",
    "RAW_IC50_COL",
    "VALUE_IC50_COL",
    "FORMULATION_META_COLS",
)
