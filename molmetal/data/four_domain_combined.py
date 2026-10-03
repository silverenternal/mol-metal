"""Four-domain combined dataset for cross-domain Mol-Metal training.

Source mapping (TODO-143 §2.5 Day 4)
------------------------------------
Domain 0 — **MetalCytoToxDB** (5,207 rows)
    :class:`molmetal.data.cytotox.MetalCytotoxDataset.from_csv(...)`
    Ru/Ir/Rh/Os/Re organometallic complexes, t<=24h, IC50_Dark_value present.
    Item yields: (smiles, pIC50, active, metal, cell_line, year, charge, ox_state).

Domain 1 — **DrugOOD native IC50** (1,583,302 rows)
    ``.data/molmetal/drugood/ic50/data/train.csv`` (ChEMBL_29 IC50 split).
    Columns: smiles, y, assay_id, scaffold_split_id, size_split_id.
    Item yields: (smiles, y_binary, assay_id, scaffold_id, size_id).

Domain 2 — **NCI-60 9-cell precious subset** (7,824 rows)
    ``molmetal/data/nci60_precious_9cell.csv`` (pre-filtered to precious-metal
    complexes against 9 representative cancer cell lines).
    Columns: smiles, cell_line, pIC50, active.
    Item yields: (smiles, pIC50, active, cell_line).

Domain 3 — **BindingDB IC50** (~1.07M rows)  --  wired 2026-09-27 (Wf-145).
    ``.data/molmetal/bindingdb_ic50.csv`` (symlink to
    ``/home/hugo/codes/pocket/data/tdc/bindingdb_ic50.csv`` on the host).
    Columns: ID1, X1, ID2, X2, Y where ``X1`` is a drug SMILES, ``X2`` is a
    target amino-acid sequence, ``Y`` is IC50 in nanomolar (nM).
    We convert ``Y`` (nM) → ``pIC50 = 9 - log10(Y)``; rows with non-positive
    ``Y`` or empty SMILES are skipped.  Item yields: (smiles, pIC50, active,
    target_id, protein).
    **License** — BindingDB data is published under
    `Creative Commons Attribution 3.0 <https://creativecommons.org/licenses/by/3.0/>`_
    (CC BY 3.0).  Attribution line required in derivative works:
    "BindingDB data courtesy of BindingDB.org, licensed under CC BY 3.0."
    The Mol-Metal repo **does not redistribute** the BindingDB CSV itself —
    it is referenced via a project-local symlink; users without the symlink
    fall back to :file:`molmetal/data/protein_metal_binding.csv` (5,258 rows).

Combined shape
--------------
* ``__len__`` returns the *exact* sum of the 4 domain lengths (no subsampling).
* ``__getitem__`` returns ``(smiles, y, domain_id, aux)`` where
  ``domain_id ∈ {0, 1, 2, 3}`` and ``aux`` is a per-domain dict (kept tiny —
  metal / cell_line / assay_id / target_id / etc.).

Lazy loading
------------
DrugOOD's 1.58M rows are **NOT** eagerly loaded.  ``__len__`` is computed via
a fast header peek / line-count so construction is O(1).  Per-row reads
are still backed by an in-memory list — DrugOOD's 1.58M CSVs at ~50 chars/row
fit in ~80 MB and are read on first iteration.

BindingDB IC50 (~1.07M rows) is *also* lazy on ``__len__`` (line-count), but
its ``__getitem__`` lazily parses the CSV on first index access; the file is
838 MB on disk so we do **not** eagerly slurp it.  If you want a fully
materialised BindingDB subset, pass ``max_rows_per_domain={DOMAIN_BINDINGDB: N}``.
"""

from __future__ import annotations

import csv
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Domain constants — match TODO-143 §2.5 ordering
# ---------------------------------------------------------------------------
DOMAIN_METALCYTOTOX = 0  # MetalCytoToxDB (Ru/Ir/Rh/Os/Re anticancer)
DOMAIN_DRUGOOD = 1  # DrugOOD native IC50 (ChEMBL_29)
DOMAIN_NCI60 = 2  # NCI-60 9-cell precious subset
DOMAIN_BINDINGDB = 3  # BindingDB IC50 (real ~1.07M, fallback 5,258)

DOMAIN_NAMES = {
    DOMAIN_METALCYTOTOX: "metalcytotox",
    DOMAIN_DRUGOOD: "drugood",
    DOMAIN_NCI60: "nci60_9cell",
    DOMAIN_BINDINGDB: "bindingdb",
}

# Default paths (override via env MOLMETAL_DATA_DIR or constructor kwargs)
DEFAULT_DRUGOOD_TRAIN = (
    PROJECT_ROOT / ".data" / "molmetal" / "drugood" / "ic50" / "data" / "train.csv"
)
DEFAULT_NCI60_CSV = PROJECT_ROOT / "molmetal" / "data" / "nci60_precious_9cell.csv"
# BindingDB IC50 — preferred (real ~1.07M rows, symlink to pocket/data/tdc/)
DEFAULT_BINDINGDB_CSV = PROJECT_ROOT / ".data" / "molmetal" / "bindingdb_ic50.csv"
# Fallback — local precious-metal subset (5,258 rows, no third-party licensing)
DEFAULT_BINDINGDB_FALLBACK_CSV = (
    PROJECT_ROOT / "molmetal" / "data" / "protein_metal_binding.csv"
)


def _resolve_metalcytotox_csv() -> Path:
    """Locate MetalCytoToxDB.csv honouring the ``MOLMETAL_DATA_DIR`` env var.

    The :mod:`molmetal.data.cytotox` loader uses a module-level
    ``DEFAULT_CSV`` constant pointing at ``/mnt/storage/data/molmetal/``,
    which is unavailable on most dev boxes.  We honour the same
    ``MOLMETAL_DATA_DIR`` convention used elsewhere in the project
    (see :file:`molmetal/tests/test_metal_lomo_v2.py`) and fall back to
    the project-local ``.data/molmetal/`` tree.
    """
    env_root = os.environ.get("MOLMETAL_DATA_DIR")
    candidates: List[Path] = []
    if env_root:
        candidates.append(Path(env_root) / "MetalCytoToxDB.csv")
    candidates.extend(
        [
            PROJECT_ROOT / ".data" / "molmetal" / "MetalCytoToxDB.csv",
            Path("/mnt/storage/data/molmetal/MetalCytoToxDB.csv"),
        ]
    )
    for c in candidates:
        if c.exists():
            return c
    # Return the first candidate even if missing — the loader will raise a
    # clearer FileNotFoundError.
    return candidates[0]


# ---------------------------------------------------------------------------
# Per-domain sub-datasets — all expose __len__ + __getitem__ returning a dict
# ---------------------------------------------------------------------------
@dataclass
class _DomainStats:
    """Lightweight stats — exposed by :class:`FourDomainDataset.summary`."""

    domain_id: int
    name: str
    n_rows: int


class _DrugOODRows:
    """DrugOOD-native IC50 sub-dataset (domain 1).

    Returns dict per item with keys: ``smiles``, ``y`` (binary), ``assay_id``,
    ``scaffold_id``, ``size_id``.
    """

    def __init__(self, path: Path = DEFAULT_DRUGOOD_TRAIN):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(
                f"DrugOOD native CSV not found at {self.path}.  "
                "Either run the DrugOOD data-prep step or supply a custom path."
            )
        # Lazy-loaded rows: list[Tuple[smiles, y, assay_id, scaffold_id, size_id]]
        self._rows: Optional[List[Tuple[str, int, int, int, int]]] = None

    def _ensure_loaded(self) -> None:
        if self._rows is not None:
            return
        rows: List[Tuple[str, int, int, int, int]] = []
        with self.path.open("r", encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                smi = (row.get("smiles") or "").strip()
                if not smi:
                    continue
                try:
                    y = int(float(row.get("y", 0) or 0))
                except (ValueError, TypeError):
                    continue
                if y not in (0, 1):
                    continue
                rows.append(
                    (
                        smi,
                        y,
                        int(row.get("assay_id", 0) or 0),
                        int(row.get("scaffold_split_id", 0) or 0),
                        int(row.get("size_split_id", 0) or 0),
                    )
                )
        self._rows = rows

    def __len__(self) -> int:
        # Fast path: count rows without parsing (assumes no embedded newlines in smiles)
        if self._rows is not None:
            return len(self._rows)
        n = -1  # subtract header
        with self.path.open("rb") as fh:
            for _ in fh:
                n += 1
        return max(n, 0)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        self._ensure_loaded()
        assert self._rows is not None
        smi, y, assay, sc, sz = self._rows[int(idx)]
        return {
            "smiles": smi,
            "y": int(y),
            "assay_id": int(assay),
            "scaffold_id": int(sc),
            "size_id": int(sz),
        }


class _NCI60Rows:
    """NCI-60 9-cell precious subset (domain 2).  CSV-backed."""

    def __init__(self, path: Path = DEFAULT_NCI60_CSV):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"NCI-60 CSV not found at {self.path}")
        self._rows: Optional[List[Tuple[str, str, float, bool]]] = None

    def _ensure_loaded(self) -> None:
        if self._rows is not None:
            return
        rows: List[Tuple[str, str, float, bool]] = []
        with self.path.open("r", encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                smi = (row.get("smiles") or "").strip()
                if not smi:
                    continue
                try:
                    pic50 = float(row.get("pIC50", float("nan")) or float("nan"))
                except (ValueError, TypeError):
                    pic50 = float("nan")
                try:
                    active = bool(int(float(row.get("active", 0) or 0)))
                except (ValueError, TypeError):
                    active = False
                rows.append((smi, str(row.get("cell_line", "")), pic50, active))
        self._rows = rows

    def __len__(self) -> int:
        if self._rows is not None:
            return len(self._rows)
        n = -1
        with self.path.open("rb") as fh:
            for _ in fh:
                n += 1
        return max(n, 0)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        self._ensure_loaded()
        assert self._rows is not None
        smi, cl, pic50, act = self._rows[int(idx)]
        return {
            "smiles": smi,
            "pIC50": float(pic50),
            "active": bool(act),
            "cell_line": cl,
        }


class _BindingDBRows:
    """BindingDB IC50 (domain 3).  CSV-backed, lazy-loaded.

    Input format (TDC ``bindingdb_ic50.csv``):
        ``ID1, X1, ID2, X2, Y``
        - ``ID1`` — drug identifier (BindingDB monomer ID, stringified float).
        - ``X1``  — drug SMILES.
        - ``ID2`` — target identifier (typically a UniProt accession).
        - ``X2``  — target amino-acid sequence (may be empty for some rows).
        - ``Y``   — IC50 in **nanomolar** (nM); may be ``<= 0`` for censored rows.

    We convert ``Y`` → ``pIC50 = 9 - log10(Y)`` and skip rows where ``Y <= 0``
    or the SMILES is empty.  Active label = ``pIC50 >= 5`` (i.e. ``Y <= 10 µM``).

    License
    -------
    BindingDB data is published under
    `Creative Commons Attribution 3.0 (CC BY 3.0) <https://creativecommons.org/licenses/by/3.0/>`_.
    Attribution line for derivative works:
        "BindingDB data courtesy of BindingDB.org, licensed under CC BY 3.0."

    Notes
    -----
    The CSV is **NOT** vendored in this repo.  We expect a project-local
    symlink ``.data/molmetal/bindingdb_ic50.csv`` to point at the file on the
    developer's host (here: ``/home/hugo/codes/pocket/data/tdc/bindingdb_ic50.csv``,
    838 MB).  Without that symlink, the loader auto-falls-back to the local
    :file:`molmetal/data/protein_metal_binding.csv` (5,258 rows) so the
    FourDomainDataset still constructs but with a much smaller domain-3.
    """

    def __init__(
        self,
        path: Path = DEFAULT_BINDINGDB_CSV,
        fallback_path: Path = DEFAULT_BINDINGDB_FALLBACK_CSV,
    ):
        self.path = Path(path)
        self.fallback_path = Path(fallback_path)
        if not self.path.exists() and not self.fallback_path.exists():
            raise FileNotFoundError(
                f"BindingDB CSV not found at {self.path} and fallback "
                f"{self.fallback_path} is also missing."
            )
        # Flag: which file are we actually reading?
        self._using_fallback = not self.path.exists()
        # Storage shape: real BindingDB uses (smi, pIC50, target_id, target_seq, active);
        # fallback uses (smi, pIC50, metal, protein, active).
        self._rows: Optional[List[Tuple[object, ...]]] = None

    @property
    def using_fallback(self) -> bool:
        return self._using_fallback

    def _ensure_loaded(self) -> None:
        if self._rows is not None:
            return
        rows: List[Tuple[object, ...]] = []
        if self._using_fallback:
            rows = self._load_fallback()
        else:
            rows = self._load_real_bindingdb()
        self._rows = rows

    def _load_real_bindingdb(self) -> List[Tuple[str, float, str, str, bool]]:
        """Parse ``ID1,X1,ID2,X2,Y`` (TDC bindingdb_ic50.csv format).

        Returns list of ``(smi, pIC50, target_id, target_seq, active)``.
        """
        import math

        rows: List[Tuple[str, float, str, str, bool]] = []
        with self.path.open("r", encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                smi = (row.get("X1") or "").strip()
                if not smi:
                    continue
                try:
                    y_nM = float(row.get("Y", "") or "")
                except (ValueError, TypeError):
                    continue
                # Y is IC50 in nM.  Skip non-positive (censored) and extreme outliers.
                if y_nM <= 0.0:
                    continue
                if y_nM > 1.0e10:  # > 10 mM is not a meaningful IC50
                    continue
                pIC50 = 9.0 - math.log10(y_nM)  # nM → pIC50
                # Clip pIC50 to a sane range to avoid overflow / log-underflow artefacts.
                if not (-2.0 <= pIC50 <= 14.0):
                    continue
                target_id = (row.get("ID2") or "").strip()
                target_seq = (row.get("X2") or "").strip()
                active = pIC50 >= 5.0  # ≤ 10 µM
                rows.append((smi, float(pIC50), target_id, target_seq, bool(active)))
        return rows

    def _load_fallback(self) -> List[Tuple[str, float, str, str, bool]]:
        """Parse the local :file:`protein_metal_binding.csv` fallback.

        Columns: ``smiles, metal, cell_line, protein, pIC50, active``.
        """
        rows: List[Tuple[str, float, str, str, bool]] = []
        with self.fallback_path.open("r", encoding="utf-8", errors="replace") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                smi = (row.get("smiles") or "").strip()
                if not smi:
                    continue
                try:
                    pic50 = float(row.get("pIC50", float("nan")) or float("nan"))
                except (ValueError, TypeError):
                    pic50 = float("nan")
                try:
                    active = bool(int(float(row.get("active", 0) or 0)))
                except (ValueError, TypeError):
                    active = False
                metal = str(row.get("metal", ""))
                protein = str(row.get("protein", ""))
                rows.append((smi, float(pIC50), metal, protein, bool(active)))
        return rows

    def __len__(self) -> int:
        if self._rows is not None:
            return len(self._rows)
        n = -1
        active_path = self.fallback_path if self._using_fallback else self.path
        with active_path.open("rb") as fh:
            for _ in fh:
                n += 1
        return max(n, 0)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        self._ensure_loaded()
        assert self._rows is not None
        rec = self._rows[int(idx)]
        if self._using_fallback:
            smi, pIC50, metal, protein, active = rec
            return {
                "smiles": smi,
                "pIC50": float(pIC50),
                "active": bool(active),
                "metal": metal,
                "protein": protein,
                "target_id": "",
            }
        smi, pIC50, target_id, target_seq, active = rec
        return {
            "smiles": smi,
            "pIC50": float(pIC50),
            "active": bool(active),
            "target_id": target_id,
            "target_seq": target_seq,
        }


def _load_cytotox_df(path: Path, flt) -> "pd.DataFrame":
    """Read + filter MetalCytoToxDB.csv directly (bypass from_csv's hard-coded path)."""
    import pandas as pd

    from molmetal.data.cytotox import _add_active, _add_pic50, _apply_filters

    df = pd.read_csv(path, low_memory=False)
    df = _apply_filters(df, flt)
    if flt.compute_pic50:
        df = _add_pic50(df)
    if flt.compute_active:
        df = _add_active(df)
    return df


# ---------------------------------------------------------------------------
# Combined dataset
# ---------------------------------------------------------------------------
@dataclass
class FourDomainItem:
    """Item yielded by :class:`FourDomainDataset`.

    Per the TODO-143 spec, the item is ``(smiles, y, domain_id)`` plus an
    optional ``aux`` dict carrying domain-specific metadata (metal, cell_line,
    assay_id, etc.).
    """

    smiles: str
    y: object  # float pIC50 or int binary label, depending on domain
    domain_id: int  # 0/1/2/3
    aux: Dict[str, object] = field(default_factory=dict)


class FourDomainDataset:
    """ConcatDataset of the 4 cross-domain sources.

    Parameters
    ----------
    metal_whitelist : list[str] | None
        If given, restrict MetalCytoToxDB to those metals (e.g. ``["Ru"]``).
    drugood_path / nci60_path / bindingdb_path : Path | None
        Override per-domain CSV paths.
    max_rows_per_domain : dict[int, int] | None
        Optional per-domain sub-sample caps (for fast tests / 100k-mode).

    Notes
    -----
    Construction does **not** load DrugOOD's 1.58M rows into memory — it only
    counts them via a header peek (newline count).  Per-row materialization
    happens lazily on first ``__getitem__`` traversal.
    """

    def __init__(
        self,
        metal_whitelist: Optional[List[str]] = None,
        drugood_path: Optional[Path] = None,
        nci60_path: Optional[Path] = None,
        bindingdb_path: Optional[Path] = None,
        max_rows_per_domain: Optional[Dict[int, int]] = None,
        metalcytotox_path: Optional[Path] = None,
    ) -> None:
        # Domain 0 — MetalCytoToxDB via the existing loader.  Pass the
        # resolved path explicitly because :mod:`molmetal.data.cytotox` uses
        # a module-level constant that ignores ``MOLMETAL_DATA_DIR``.
        from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset

        flt = CytotoxFilter(
            time_threshold=24.0,
            ic50_min=0.01,
            compute_pic50=True,
            compute_active=True,
            metal_whitelist=metal_whitelist,
        )
        cytotox_csv = metalcytotox_path or _resolve_metalcytotox_csv()
        # Use a project-local cache dir (cytotox.py's module-level default
        # is /mnt/storage/data/molmetal/3d_cache, which is unwritable on
        # most dev boxes).  We pass it through the from_csv factory by
        # constructing the dataset directly because from_csv() does not
        # expose cache_dir.
        cache_dir = Path(
            os.environ.get("MOLMETAL_3D_CACHE_DIR")
            or str(cytotox_csv.parent / "3d_cache")
        )
        cache_dir.mkdir(parents=True, exist_ok=True)
        try:
            ds0_df = _load_cytotox_df(cytotox_csv, flt)
        except FileNotFoundError:
            # Surface a clear error that names the path tried + the env var
            # hint.  Without this, a developer who has the project-local
            # ``.data/molmetal/`` tree will see the hard-coded
            # ``/mnt/storage`` path and waste time chasing it.
            raise FileNotFoundError(
                f"MetalCytoToxDB.csv not found.  Tried: {cytotox_csv}.  "
                "Set the MOLMETAL_DATA_DIR env var to the directory "
                "containing MetalCytoToxDB.csv, or pass metalcytotox_path=... "
                "to the constructor."
            )
        from molmetal.data.cytotox import MetalCytotoxDataset

        self._domain_0 = MetalCytotoxDataset(df=ds0_df, cache_dir=cache_dir)

        # Domains 1-3 — CSV-backed lazy loaders
        self._domain_1 = _DrugOODRows(drugood_path or DEFAULT_DRUGOOD_TRAIN)
        self._domain_2 = _NCI60Rows(nci60_path or DEFAULT_NCI60_CSV)
        self._domain_3 = _BindingDBRows(bindingdb_path or DEFAULT_BINDINGDB_CSV)

        # Per-domain length cache

        # Per-domain length cache (FULL sizes — caps applied below)
        self._lens: Dict[int, int] = {
            0: len(self._domain_0),
            1: len(self._domain_1),
            2: len(self._domain_2),
            3: len(self._domain_3),
        }
        # Per-domain row caps for fast tests
        self._caps: Dict[int, Optional[int]] = {
            0: None,
            1: None,
            2: None,
            3: None,
        }
        if max_rows_per_domain:
            for d, cap in max_rows_per_domain.items():
                self._caps[int(d)] = min(int(cap), self._lens[int(d)])

        # Per-domain offset into the concat index space (EFFECTIVE / capped)
        self._offsets: Dict[int, int] = {}
        cum = 0
        for d in range(4):
            self._offsets[d] = cum
            cum += self._lens[d] if self._caps[d] is None else self._caps[d]
        self._total_len = cum

    # ------------------------------------------------------------------
    # Length
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        # Sum over actual (possibly capped) per-domain sizes
        total = 0
        for d in range(4):
            cap = self._caps[d]
            total += self._lens[d] if cap is None else cap
        return total

    def domain_lengths(self) -> Dict[int, int]:
        """Return per-domain *effective* lengths (respecting ``max_rows_per_domain``)."""
        return {
            d: self._lens[d] if self._caps[d] is None else self._caps[d]
            for d in range(4)
        }

    def domain_names(self) -> Dict[int, str]:
        return dict(DOMAIN_NAMES)

    # ------------------------------------------------------------------
    # Get item
    # ------------------------------------------------------------------
    def _resolve_global_index(self, idx: int) -> Tuple[int, int]:
        """Map a flat concat index -> (domain_id, local_index).

        Honours per-domain ``max_rows_per_domain`` caps.
        """
        if idx < 0:
            idx += len(self)
        if not 0 <= idx < len(self):
            raise IndexError(
                f"FourDomainDataset index {idx} out of range [0, {len(self)})"
            )
        cum = 0
        for d in range(4):
            cap = self._caps[d]
            d_len = self._lens[d] if cap is None else cap
            if idx < cum + d_len:
                return d, int(idx - cum)
            cum += d_len
        raise IndexError(idx)  # unreachable

    def __getitem__(self, idx: int) -> FourDomainItem:
        d, local = self._resolve_global_index(idx)
        if d == 0:
            row = self._domain_0[int(local)]
            return FourDomainItem(
                smiles=str(row["smiles"]),
                y=float(row["pIC50"]) if "pIC50" in row else float("nan"),
                domain_id=0,
                aux={
                    "metal": str(row.get("metal", "")),
                    "cell_line": str(row.get("cell_line", "")),
                    "active": bool(row.get("active", False)),
                },
            )
        if d == 1:
            row = self._domain_1[int(local)]
            return FourDomainItem(
                smiles=str(row["smiles"]),
                y=int(row["y"]),
                domain_id=1,
                aux={
                    "assay_id": int(row["assay_id"]),
                    "scaffold_id": int(row["scaffold_id"]),
                    "size_id": int(row["size_id"]),
                },
            )
        if d == 2:
            row = self._domain_2[int(local)]
            return FourDomainItem(
                smiles=str(row["smiles"]),
                y=float(row["pIC50"]),
                domain_id=2,
                aux={
                    "cell_line": str(row["cell_line"]),
                    "active": bool(row["active"]),
                },
            )
        if d == 3:
            row = self._domain_3[int(local)]
            aux = {"active": bool(row.get("active", False))}
            # Real BindingDB rows carry target_id + target_seq (UniProt + AA seq).
            # Fallback rows carry metal + protein (legacy protein_metal_binding schema).
            if "target_id" in row:
                aux["target_id"] = str(row.get("target_id", ""))
                aux["target_seq"] = str(row.get("target_seq", ""))
            else:
                aux["metal"] = str(row.get("metal", ""))
                aux["protein"] = str(row.get("protein", ""))
            return FourDomainItem(
                smiles=str(row["smiles"]),
                y=float(row["pIC50"]),
                domain_id=3,
                aux=aux,
            )
        raise IndexError(d)

    # ------------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, object]:
        return {
            "total_rows": len(self),
            "domain_rows_full": dict(self._lens),
            "domain_rows_effective": self.domain_lengths(),
            "domain_names": self.domain_names(),
            "domains": [
                {
                    "domain_id": d,
                    "name": DOMAIN_NAMES[d],
                    "n_rows_full": self._lens[d],
                    "n_rows_effective": (
                        self._lens[d] if self._caps[d] is None else self._caps[d]
                    ),
                    "cap": self._caps[d],
                }
                for d in range(4)
            ],
        }


# ---------------------------------------------------------------------------
# Convenience — iterate as (X, y, domain_id) per TODO-143 §2.5 spec
# ---------------------------------------------------------------------------
def iter_three_tuples(dataset: FourDomainDataset):
    """Yield ``(smiles, y, domain_id)`` per item — convenient for downstream code."""
    for i in range(len(dataset)):
        item = dataset[i]
        yield item.smiles, item.y, item.domain_id


__all__ = [
    "FourDomainDataset",
    "FourDomainItem",
    "DOMAIN_METALCYTOTOX",
    "DOMAIN_DRUGOOD",
    "DOMAIN_NCI60",
    "DOMAIN_BINDINGDB",
    "DOMAIN_NAMES",
    "iter_three_tuples",
]
