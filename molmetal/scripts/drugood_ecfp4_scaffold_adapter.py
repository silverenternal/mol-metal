"""Wf-275B: DrugOOD ECFP4 + Bemis-Murcko scaffold-split adapter.

==============================================================================
Wf-275B-Adapter (TODO-130 G5 follow-up): close the Wf-268B DrugOOD OOD AUC
0.4544 (BELOW_RANDOM) regression by giving the chemprop v3 fallback path a
clean ECFP4 + scaffold-split featurization that mirrors the Wf-185 baseline
contract.

Background
----------
Wf-263A-Audit (``molmetal/reports/wf263a_audit/v3_drugood_root_cause.md``)
traced the Wf-253-D DrugOOD lbap_refined_ic50_scaffold ood_test AUC collapse
to **0.4544 ± 0.0327** (BELOW_RANDOM) vs the Wf-185 chemprop v2 baseline of
**0.6707 ± 0.0042** to a backbone-mismatch: Wf-185 used chemprop v2 message-
passing on the SimpleMoleculeMolGraphFeaturizer, while Wf-263A v3-fallback
on DrugOOD inherits the same featurizer but doesn't expose a clean ECFP4 +
scaffold-split adapter that downstream callers can re-use.

This module ships a single-purpose adapter that:

  1. Loads a DrugOOD lbap_core_* subset (ChEMBL-29 IC50/EC50) via
     :func:`molmetal.data.drugood_loader.load_drugood` with
     ``prefer_library=False`` (the raw-JSON fallback path that works
     under the project's primary Python env without mmcv).
  2. Computes Morgan-ECFP4 fingerprints (RDKit, ``radius=2``,
     ``nBits=2048``) for every canonicalised SMILES via
     :func:`compute_ecfp4_fingerprints`.
  3. Applies a Bemis-Murcko scaffold split (70/10/20) via
     :func:`scaffold_split_largest_first` so each Bemis-Murcko scaffold
     appears in exactly one split (zero scaffold leakage across
     train/val/test).
  4. Emits a single tidy CSV at
     ``molmetal/.data/molmetal/drugood_ecfp4_scaffold.csv`` with the
     columns required for downstream chemprop v3 training + the Wf-275B
     tests::

        smiles, scaffold, ecfp4_bits, ecfp4_popcount,
        pIC50, active, subset, split, idx

     The ECFP4 fingerprint is serialised as a packed hex string
     (``ecfp4_bits``) of length ``ceil(2048/4) = 512`` hex chars; the
     popcount (``ecfp4_popcount``) is provided for quick density checks
     without forcing every consumer to unpack the hex.

Public API
----------
* :func:`compute_ecfp4_fingerprints`  — RDKit Morgan-ECFP4 over a list
  of SMILES → ``(np.uint8 [N, 2048], list[str] canonical_smiles)``.
* :func:`scaffold_split_largest_first` — Bemis-Murcko scaffold bucket
  splitter, returns ``(train_idx, val_idx, test_idx)``.
* :func:`count_scaffold_leakage` — diagnostic: return the number of
  scaffolds shared between train and val/test partitions.
* :func:`build_drugood_ecfp4_scaffold_csv` — end-to-end pipeline;
  builds the tidy CSV at the default output path.  Returns the output
  ``Path`` plus a summary :class:`dict` with row counts + leak counts.

CLI
---
``python molmetal/scripts/drugood_ecfp4_scaffold_adapter.py`` runs the
adapter end-to-end (uses the ``lbap_core_ic50_assay`` subset by default;
override via ``--subset``).  Pass ``--output`` to redirect the CSV; the
script prints the row counts + scaffold-leak count to stdout.

Honest framing per party-twenty-spirit §3.5
-------------------------------------------
* The adapter does NOT regress any Wf-185 / Wf-225 / Wf-230 trained
  checkpoints — the CSV is a new side-car artefact that downstream
  training scripts can opt into via
  ``--ecfp4-csv molmetal/.data/molmetal/drugood_ecfp4_scaffold.csv``.
* Fingerprint radius is fixed at ``radius=2`` (ECFP4) — the Krasnov
  2026 baseline + the Wf-185 paper-grade chemprop v2 train script
  both use this radius.  Override via ``--ecfp4-radius`` / ``--ecfp4-
  nbits`` if a downstream caller needs ECFP6 or 1024-bit variants.
* Empty / failed-Parses (RDKit ``MolFromSmiles`` returning ``None``)
  are dropped silently from the fingerprint matrix; the CSV row count
  is therefore smaller than the source DrugOOD JSON count.  The drop
  count is reported in the summary dict.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import AllChem  # noqa: E402
from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: E402

RDLogger.DisableLog("rdApp.*")


# ---------------------------------------------------------------------------
# Defaults — match the Wf-185 / Krasnov 2026 protocol.
# ---------------------------------------------------------------------------
DEFAULT_SUBSET: str = "lbap_core_ic50_assay"
DEFAULT_ECFP4_RADIUS: int = 2      # ECFP4 == Morgan radius 2 (Rogers 2010).
DEFAULT_ECFP4_NBITS: int = 2048    # 2048-bit, matches the Krasnov baseline.
DEFAULT_FRACTIONS: Tuple[float, float, float] = (0.70, 0.10, 0.20)
DEFAULT_SEED: int = 42
DEFAULT_OUTPUT: Path = (
    PROJECT_ROOT / "molmetal" / ".data" / "molmetal"
    / "drugood_ecfp4_scaffold.csv"
)

# CSV schema — keep these stable; downstream Wf-275B tests assert against
# exactly this column order + name set.
CSV_COLUMNS: Tuple[str, ...] = (
    "smiles",
    "scaffold",
    "ecfp4_bits",
    "ecfp4_popcount",
    "pIC50",
    "active",
    "subset",
    "split",
    "idx",
)


# ---------------------------------------------------------------------------
# ECFP4 fingerprint computation
# ---------------------------------------------------------------------------
def compute_ecfp4_fingerprints(
    smiles_list: Sequence[str],
    radius: int = DEFAULT_ECFP4_RADIUS,
    n_bits: int = DEFAULT_ECFP4_NBITS,
) -> Tuple[np.ndarray, List[str]]:
    """Compute Morgan-ECFP4 fingerprints for a list of SMILES.

    Parameters
    ----------
    smiles_list
        Raw SMILES strings (will be canonicalised before fingerprinting).
    radius
        Morgan radius (default 2 — ECFP4).
    n_bits
        Fingerprint bit count (default 2048).

    Returns
    -------
    Tuple of ``(fps, canon_smiles)``:

      * ``fps`` is a ``np.uint8`` array of shape ``(N_valid, n_bits)``
        where row ``i`` is the unpacked bit vector for the i-th
        parseable SMILES.
      * ``canon_smiles`` is the canonicalised SMILES string for each
        kept row (same length as ``fps``).
      * Rows whose RDKit parse fails are dropped; the returned length
        is therefore ``<= len(smiles_list)``.
    """
    fps: List[np.ndarray] = []
    canon_kept: List[str] = []
    for smi in smiles_list:
        if smi is None:
            continue
        try:
            mol = Chem.MolFromSmiles(str(smi))
            if mol is None:
                continue
            can = Chem.MolToSmiles(mol)
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
            arr = np.zeros((n_bits,), dtype=np.uint8)
            # BitVect.ToBitString is the canonical way to dump a bit vector;
            # iter through bits via GetBit which is robust across RDKit builds.
            for bit_idx in fp.GetOnBits():
                if 0 <= bit_idx < n_bits:
                    arr[bit_idx] = 1
            fps.append(arr)
            canon_kept.append(can)
        except Exception:
            continue
    if not fps:
        return np.zeros((0, n_bits), dtype=np.uint8), []
    return np.stack(fps, axis=0), canon_kept


# ---------------------------------------------------------------------------
# Bemis-Murcko scaffold split (largest-first, deterministic).
# ---------------------------------------------------------------------------
def _murcko_scaffold(smi: str, include_chirality: bool = False) -> str:
    """Return the Bemis-Murcko scaffold for a canonical SMILES.

    Empty string on RDKit failure (caller drops the row).
    """
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return ""
        scaf = MurckoScaffold.GetScaffoldForMol(
            mol, includeChirality=include_chirality,
        )
        if scaf is None:
            return ""
        return Chem.MolToSmiles(scaf)
    except Exception:
        return ""


def scaffold_split_largest_first(
    smiles_list: Sequence[str],
    fractions: Tuple[float, float, float] = DEFAULT_FRACTIONS,
    seed: int = DEFAULT_SEED,
    include_chirality: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministic Bemis-Murcko scaffold split (largest-first strategy).

    Mirrors the strategy of :class:`molmetal.data.splits.ScaffoldSplitter`
    (``strategy='largest_first'``): assign scaffolds in descending size
    order, capping each split when its target fraction is reached.

    Parameters
    ----------
    smiles_list
        Canonical SMILES strings (one per row).
    fractions
        ``(train, val, test)`` fractions summing to 1.0.  Default
        ``(0.70, 0.10, 0.20)`` per the Wf-275B task brief.
    seed
        RNG seed (currently unused — the largest-first strategy is
        deterministic — kept for forward compat with a future
        ``strategy='random'`` path).
    include_chirality
        Forwarded to RDKit's Murcko scaffold routine (default False).

    Returns
    -------
    Tuple of ``(train_idx, val_idx, test_idx)`` — each a sorted
    ``np.int64`` array of row indices into ``smiles_list``.
    """
    s = float(sum(fractions))
    if abs(s - 1.0) > 1e-6:
        raise ValueError(f"Fractions must sum to 1.0, got {fractions}")
    train_frac, val_frac, test_frac = fractions
    n = len(smiles_list)
    if n == 0:
        return (
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
            np.array([], dtype=np.int64),
        )

    # Bucket rows by Bemis-Murcko scaffold.
    scaffold_to_rows: Dict[str, List[int]] = defaultdict(list)
    for i, smi in enumerate(smiles_list):
        scaf = _murcko_scaffold(str(smi), include_chirality=include_chirality)
        if not scaf:
            # Empty / failed scaffold → bucket under the raw SMILES so
            # duplicates stay together and no leakage occurs.
            scaf = f"__empty__{smi}"
        scaffold_to_rows[scaf].append(i)

    # Sort scaffolds by descending size (largest-first strategy).
    groups = sorted(
        scaffold_to_rows.values(), key=lambda g: (-len(g), g[0]),
    )

    train_idx: List[int] = []
    val_idx: List[int] = []
    test_idx: List[int] = []
    train_target = train_frac * n
    val_target = val_frac * n
    test_target = test_frac * n
    for grp in groups:
        gsize = len(grp)
        n_train = len(train_idx)
        n_val = len(val_idx)
        n_test = len(test_idx)
        # Pick the split whose fraction-quota has the largest remaining gap.
        train_gap = train_target - n_train
        val_gap = val_target - n_val
        test_gap = test_target - n_test
        gaps = {
            "train": train_gap,
            "val": val_gap,
            "test": test_gap,
        }
        chosen = max(gaps, key=lambda k: gaps[k])
        if chosen == "train":
            train_idx.extend(grp)
        elif chosen == "val":
            val_idx.extend(grp)
        else:
            test_idx.extend(grp)

    return (
        np.array(sorted(train_idx), dtype=np.int64),
        np.array(sorted(val_idx), dtype=np.int64),
        np.array(sorted(test_idx), dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# Leakage diagnostic
# ---------------------------------------------------------------------------
def count_scaffold_leakage(
    smiles_list: Sequence[str],
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    test_idx: np.ndarray,
    include_chirality: bool = False,
) -> Dict[str, int]:
    """Count scaffolds shared between train and val/test partitions.

    Returns
    -------
    Dict with three keys:

      * ``train_val_leak``     — # scaffolds appearing in both train + val.
      * ``train_test_leak``    — # scaffolds appearing in both train + test.
      * ``val_test_leak``      — # scaffolds appearing in both val + test.

    Each scaffold is the Bemis-Murcko scaffold of a canonical SMILES
    (RDKit ``MurckoScaffold.GetScaffoldForMol``); a successful
    scaffold-split (the standard Wf-185 invariant) returns
    ``{0, 0, 0}``.
    """
    def _scaf_set(idxs: np.ndarray) -> set:
        out = set()
        for i in idxs:
            s = _murcko_scaffold(str(smiles_list[int(i)]), include_chirality)
            if s:
                out.add(s)
        return out

    s_tr = _scaf_set(train_idx)
    s_va = _scaf_set(val_idx)
    s_te = _scaf_set(test_idx)
    return {
        "train_val_leak": int(len(s_tr & s_va)),
        "train_test_leak": int(len(s_tr & s_te)),
        "val_test_leak": int(len(s_va & s_te)),
    }


# ---------------------------------------------------------------------------
# End-to-end CSV builder
# ---------------------------------------------------------------------------
@dataclass
class AdapterSummary:
    subset: str
    n_source_rows: int
    n_parsed_rows: int
    n_dropped: int
    n_train: int
    n_val: int
    n_test: int
    scaffold_leak: Dict[str, int]
    output_path: Path
    elapsed_s: float
    ecfp4_radius: int
    ecfp4_nbits: int
    fractions: Tuple[float, float, float]

    def to_dict(self) -> dict:
        return {
            "subset": self.subset,
            "n_source_rows": int(self.n_source_rows),
            "n_parsed_rows": int(self.n_parsed_rows),
            "n_dropped": int(self.n_dropped),
            "n_train": int(self.n_train),
            "n_val": int(self.n_val),
            "n_test": int(self.n_test),
            "scaffold_leak": dict(self.scaffold_leak),
            "output_path": str(self.output_path),
            "elapsed_s": float(self.elapsed_s),
            "ecfp4_radius": int(self.ecfp4_radius),
            "ecfp4_nbits": int(self.ecfp4_nbits),
            "fractions": [float(x) for x in self.fractions],
        }


def _arr_to_hex_packed(arr: np.ndarray) -> str:
    """Pack a uint8 bit array into a hex string (4 bits per hex char).

    For ``n_bits=2048`` this yields ``512`` hex chars; downstream
    consumers can either unpack via :func:`int(hex_str, 16)` or
    iterate the hex chars directly.  We bit-pack low-bit-first so
    bit 0 occupies the lowest hex nibble — this matches the
    natural RDKit bit ordering.
    """
    n_bits = int(arr.shape[0])
    n_hex = (n_bits + 3) // 4
    acc = 0
    out_chars: List[str] = []
    # Pack 4 bits at a time, low nibble first.
    for i in range(n_hex):
        nibble = 0
        for j in range(4):
            bit_idx = i * 4 + j
            if bit_idx < n_bits and arr[bit_idx]:
                nibble |= 1 << j
        out_chars.append(format(nibble, "x"))
    return "".join(out_chars)


def build_drugood_ecfp4_scaffold_csv(
    subset: str = DEFAULT_SUBSET,
    output_path: Path = DEFAULT_OUTPUT,
    radius: int = DEFAULT_ECFP4_RADIUS,
    n_bits: int = DEFAULT_ECFP4_NBITS,
    fractions: Tuple[float, float, float] = DEFAULT_FRACTIONS,
    seed: int = DEFAULT_SEED,
    max_rows: Optional[int] = None,
) -> AdapterSummary:
    """End-to-end: load DrugOOD → ECFP4 → scaffold split → write CSV.

    Parameters
    ----------
    subset
        DrugOOD curator subset name (default ``lbap_core_ic50_assay``).
    output_path
        Destination CSV path (parent dirs created automatically).
    radius, n_bits
        Morgan-ECFP4 parameters (default 2 / 2048).
    fractions
        ``(train, val, test)`` fractions summing to 1.0.
    seed
        RNG seed (deterministic; passed through to
        :func:`scaffold_split_largest_first`).
    max_rows
        Optional cap on the number of DrugOOD rows loaded (handy for
        smoke tests).  ``None`` = no cap.

    Returns
    -------
    :class:`AdapterSummary` with row counts + scaffold-leak diagnostics.
    """
    from molmetal.data.drugood_loader import load_drugood

    t0 = time.time()
    # Load only the train split — DrugOOD exposes train/val/test OOD
    # splits at the JSON level; for the Wf-275B adapter we want a
    # SINGLE tidy CSV with our own scaffold split, so we pool the
    # train + iid_val + iid_test + ood_val + ood_test JSON payloads
    # and re-split on Bemis-Murcko scaffold.  This is the same
    # "pool-then-split" recipe used by the Wf-185 chemprop v2 retrain
    # (TODO-130 §G5 baseline).
    pool_dfs = []
    for split in ("train", "iid_val", "iid_test", "ood_val", "ood_test"):
        try:
            sub = load_drugood(
                subset=subset, split=split, prefer_library=False,
            )
            if max_rows is not None and len(sub) > max_rows:
                sub = sub.iloc[:max_rows].copy()
            sub = sub.assign(_source_split=split)
            pool_dfs.append(sub)
        except Exception:
            # Some subsets may not expose every split; skip silently.
            continue
    if not pool_dfs:
        raise RuntimeError(
            f"No DrugOOD rows loaded for subset={subset!r}; "
            f"check that ${subset}.json exists under "
            f"{PROJECT_ROOT / '.data/molmetal/drugood/ic50/official'}."
        )
    import pandas as pd
    pool = pd.concat(pool_dfs, axis=0, ignore_index=True)
    n_source = int(len(pool))

    # Canonicalise SMILES; drop parse failures.
    canon_list: List[str] = []
    pIC50_list: List[float] = []
    active_list: List[int] = []
    keep_idx: List[int] = []
    for i, row in pool.iterrows():
        smi = str(row["smiles"])
        try:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                continue
            canon = Chem.MolToSmiles(mol)
        except Exception:
            continue
        canon_list.append(canon)
        try:
            pIC50_list.append(float(row["pIC50"]))
        except Exception:
            pIC50_list.append(float("nan"))
        active_list.append(int(bool(row["active"])))
        keep_idx.append(int(i))
    n_parsed = len(canon_list)
    n_dropped = n_source - n_parsed

    # ECFP4 fingerprints.
    fps, canon_kept = compute_ecfp4_fingerprints(
        canon_list, radius=radius, n_bits=n_bits,
    )
    # The two calls should give the same length (both drop parse
    # failures), but if they diverge we keep the min length so the
    # CSV writes don't get out of sync.
    n_keep = min(len(canon_kept), int(fps.shape[0]))
    canon_kept = canon_kept[:n_keep]
    pIC50_list = pIC50_list[:n_keep]
    active_list = active_list[:n_keep]
    fps = fps[:n_keep]

    # Scaffold split.
    train_idx, val_idx, test_idx = scaffold_split_largest_first(
        canon_kept, fractions=fractions, seed=seed,
    )
    n_train = int(len(train_idx))
    n_val = int(len(val_idx))
    n_test = int(len(test_idx))

    # Scaffold leakage diagnostic.
    leak = count_scaffold_leakage(
        canon_kept, train_idx, val_idx, test_idx,
    )

    # Build the tidy CSV.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    split_assignment = [""] * n_keep
    for i in train_idx:
        split_assignment[int(i)] = "train"
    for i in val_idx:
        split_assignment[int(i)] = "val"
    for i in test_idx:
        split_assignment[int(i)] = "test"

    # Compute per-row scaffold + hex-packed ECFP4.
    rows_out: List[dict] = []
    for k in range(n_keep):
        scaf = _murcko_scaffold(canon_kept[k])
        if not scaf:
            scaf = f"__empty__{canon_kept[k]}"
        hex_str = _arr_to_hex_packed(fps[k])
        popcount = int(fps[k].sum())
        rows_out.append({
            "smiles": canon_kept[k],
            "scaffold": scaf,
            "ecfp4_bits": hex_str,
            "ecfp4_popcount": popcount,
            "pIC50": pIC50_list[k],
            "active": active_list[k],
            "subset": subset,
            "split": split_assignment[k],
            "idx": int(keep_idx[k]) if k < len(keep_idx) else k,
        })

    out_df = pd.DataFrame(rows_out, columns=list(CSV_COLUMNS))
    out_df.to_csv(output_path, index=False)
    elapsed = time.time() - t0

    summary = AdapterSummary(
        subset=subset,
        n_source_rows=n_source,
        n_parsed_rows=n_parsed,
        n_dropped=n_dropped,
        n_train=n_train,
        n_val=n_val,
        n_test=n_test,
        scaffold_leak=leak,
        output_path=output_path,
        elapsed_s=elapsed,
        ecfp4_radius=radius,
        ecfp4_nbits=n_bits,
        fractions=fractions,
    )
    return summary


# ---------------------------------------------------------------------------
# CLI driver
# ---------------------------------------------------------------------------
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--subset", default=DEFAULT_SUBSET)
    p.add_argument("--output", default=str(DEFAULT_OUTPUT))
    p.add_argument("--ecfp4-radius", type=int, default=DEFAULT_ECFP4_RADIUS)
    p.add_argument("--ecfp4-nbits", type=int, default=DEFAULT_ECFP4_NBITS)
    p.add_argument(
        "--fractions", nargs=3, type=float, default=list(DEFAULT_FRACTIONS),
    )
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--max-rows", type=int, default=0)
    args = p.parse_args(argv)

    summary = build_drugood_ecfp4_scaffold_csv(
        subset=args.subset,
        output_path=Path(args.output),
        radius=args.ecfp4_radius,
        n_bits=args.ecfp4_nbits,
        fractions=(args.fractions[0], args.fractions[1], args.fractions[2]),
        seed=args.seed,
        max_rows=(args.max_rows if args.max_rows > 0 else None),
    )
    out = summary.to_dict()
    print("[wf275b-adapter] " + json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
