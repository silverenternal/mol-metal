#!/usr/bin/env python3
"""Wf-44 Phase 3 — Data leakage diagnostic for the Wf-32 affinity trainer.

================================================================
``molmetal/scripts/check_data_leakage.py`` is the Wf-44 Phase 3 diagnostic
that **empirically verifies that the splits used by
``train_affinity_regression.py`` (Wf-32 ship) are strict**: no Bemis-Murcko
scaffold overlap under a *scaffold* split, and no canonical-SMILES overlap
under a *random* split.  This addresses root-cause ``C4`` of the
``Wf-44`` catalog ("Data leakage / split not strict").

Per-dataset checks
------------------
For each input CSV we run two splits — ``random`` (the default that
``train_affinity_regression.py`` currently uses via
``sklearn.model_selection.train_test_split``) and ``scaffold``
(:class:`molmetal.data.splits.ScaffoldSplitter`, Bemis-Murcko grouped) —
and report per-pair overlap on both axes:

* **Random split** — verifies that the *random* split is, at minimum,
  free of canonical-SMILES duplicates between train and test.  We
  *also* report how many *Bemis-Murcko scaffolds* are shared (a stricter
  notion that the random split is *not* designed to honour).
* **Scaffold split** — verifies that the Bemis-Murcko scaffold split is
  strict (zero shared scaffolds between any pair) and reports the
  canonical-SMILES overlap as a downstream sanity check.

For datasets that ship with a pre-computed ``scaffold_id`` and ``split``
column (e.g. ``metalcytotox_5metal_multitask.csv``), we additionally
verify that the **shipped split is strict at the scaffold level**.

Outputs
-------
* ``molmetal/reports/wf44_r2_debug/leakage_check.json`` — a JSON report
  with one entry per dataset + split, listing:

  * ``n_total``            — total rows
  * ``n_unique_smiles``    — unique canonical SMILES
  * ``n_unique_scaffolds`` — unique Bemis-Murcko scaffolds
  * ``splits``             — ``{train, val, test}`` indices + sizes
  * ``overlaps``           — per-pair overlap statistics:
    ``n_intersect_smiles``, ``n_intersect_scaffolds``,
    ``frac_test_smiles_seen_in_train``,
    ``frac_test_scaffolds_seen_in_train``.
  * ``strict_pass``        — ``True`` iff the overlap is zero on the
    axis the split is supposed to honour.
  * ``summary``            — a one-line verdict.

CLI
---
::

    source .venvs/mol-metal-venv/bin/activate
    python molmetal/scripts/check_data_leakage.py
    python molmetal/scripts/check_data_leakage.py \
        --platinai-csv molmetal/data/platinai_5k.csv \
        --metalcytotox-csv .data/molmetal/metalcytotox_5metal_multitask.csv \
        --out-dir molmetal/reports/wf44_r2_debug
    pytest molmetal/tests/test_data_leakage.py -v

Honest framing
--------------
* The **random split** the trainer currently uses is **expected** to
  show non-zero scaffold overlap — that is the entire diagnosis (C4).
  We report both numbers so the diagnostic itself is *transparent*.
* The we **scaffold split** is the **remediation** — it is expected to
  produce zero overlap on the scaffold axis.  If it doesn't, either
  RDKit cannot compute scaffolds on a non-trivial fraction of rows
  (in which case the report flags ``n_parse_failures``) or the
  :class:`ScaffoldSplitter` heuristic is buggy.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, str(_p))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_OUT_DIR: Path = _REPO_ROOT / "molmetal" / "reports" / "wf44_r2_debug"
DEFAULT_OUT_PATH: Path = DEFAULT_OUT_DIR / "leakage_check.json"

DEFAULT_PLATINAI_CSV: Path = _REPO_ROOT / "molmetal" / "data" / "platinai_5k.csv"
DEFAULT_METALCYTOTOX_CSV: Path = (
    _REPO_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask.csv"
)

# Both splits use 80/10/10 — same fractions ``train_affinity_regression.py``
# uses via sklearn ``train_test_split``.
DEFAULT_FRACTIONS: Tuple[float, float, float] = (0.8, 0.1, 0.1)
DEFAULT_SEED: int = 42


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------
@dataclass
class SplitOverlap:
    """Per-pair overlap statistics.

    Each entry reports the number of unique rows / scaffolds / SMILES
    in the *test* split that are *also* present in the *train* split.
    """

    n_intersect_smiles: int = 0
    n_intersect_scaffolds: int = 0
    n_unique_train_smiles: int = 0
    n_unique_test_smiles: int = 0
    n_unique_train_scaffolds: int = 0
    n_unique_test_scaffolds: int = 0
    frac_test_smiles_seen_in_train: float = 0.0
    frac_test_scaffolds_seen_in_train: float = 0.0


@dataclass
class SplitStats:
    """Per-split container — indices + counts."""

    train_idx: List[int] = field(default_factory=list)
    val_idx: List[int] = field(default_factory=list)
    test_idx: List[int] = field(default_factory=list)
    n_train: int = 0
    n_val: int = 0
    n_test: int = 0


@dataclass
class DatasetLeakageReport:
    """Aggregate per-dataset leakage report."""

    dataset_name: str
    csv_path: str
    n_total: int
    n_unique_smiles: int
    n_unique_scaffolds: int
    n_parse_failures: int
    splits: Dict[str, SplitStats] = field(default_factory=dict)
    overlaps: Dict[str, Dict[str, SplitOverlap]] = field(default_factory=dict)
    strict_pass: Dict[str, bool] = field(default_factory=dict)
    summary: Dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# CSV loader
# ---------------------------------------------------------------------------
def load_smiles_csv(path: Path) -> Tuple[List[str], List[Optional[str]]]:
    """Load ``(smiles, scaffold_id)`` from a CSV.

    Accepts both the platinai-style CSV (no ``scaffold_id`` column —
    ``scaffold_id`` is filled with ``None``) and the
    metalcytotox-5metal-style CSV (with a pre-computed ``scaffold_id``
    column).  Robust to BOMs and missing cells.
    """
    smiles: List[str] = []
    scaffolds: List[Optional[str]] = []
    with path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            smi = (row.get("smiles") or "").strip()
            smiles.append(smi)
            scaf = row.get("scaffold_id")
            if scaf is None:
                scaffolds.append(None)
            else:
                scaf = scaf.strip()
                scaffolds.append(scaf if scaf else None)
    return smiles, scaffolds


# ---------------------------------------------------------------------------
# SMILES / scaffold canonicalisation
# ---------------------------------------------------------------------------
def _try_canonicalize(smi: str) -> Optional[str]:
    """Canonical SMILES via RDKit; ``None`` on parse failure."""
    try:
        from rdkit import Chem, RDLogger

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return Chem.MolToSmiles(mol)
    except Exception:
        return None


def _try_murcko_scaffold(canon: str) -> Optional[str]:
    """Bemis-Murcko scaffold SMILES; ``None`` on parse failure."""
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem.Scaffolds import MurckoScaffold

        RDLogger.DisableLog("rdApp.*")
        mol = Chem.MolFromSmiles(canon)
        if mol is None:
            return None
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        if scaffold is None:
            return None
        out = Chem.MolToSmiles(scaffold)
        # An empty scaffold (single-atom / acyclic only) is signalled by
        # an empty string from ``Chem.MolToSmiles``.  Treat it as a parse
        # failure so we don't lump unrelated acyclic molecules into one
        # bucket.
        return out if out else None
    except Exception:
        return None


def canonicalise_smiles(smiles_list: Sequence[str]) -> Tuple[List[str], int]:
    """Canonicalise SMILES list.  Returns ``(canon_list, n_failures)``."""
    canon: List[str] = []
    n_fail = 0
    for s in smiles_list:
        c = _try_canonicalize(s)
        if c is None:
            canon.append("")    # sentinel — never matches a real canon
            n_fail += 1
        else:
            canon.append(c)
    return canon, n_fail


def compute_scaffolds(
    canon_smiles: Sequence[str],
) -> Tuple[List[Optional[str]], int]:
    """Compute Bemis-Murcko scaffolds from canonical SMILES.

    Returns ``(scaffold_list, n_failures)``.  Rows whose SMILES failed
    to parse are also marked as ``None``.
    """
    scaffolds: List[Optional[str]] = []
    n_fail = 0
    for c in canon_smiles:
        if not c:
            scaffolds.append(None)
            n_fail += 1
            continue
        s = _try_murcko_scaffold(c)
        if s is None:
            n_fail += 1
        scaffolds.append(s)
    return scaffolds, n_fail


# ---------------------------------------------------------------------------
# Splitters (random + scaffold)
# ---------------------------------------------------------------------------
def random_split_indices(
    n: int,
    fractions: Tuple[float, float, float] = DEFAULT_FRACTIONS,
    seed: int = DEFAULT_SEED,
) -> SplitStats:
    """Deterministic 80/10/10 random split (mirrors sklearn default)."""
    if n <= 0:
        return SplitStats()
    import numpy as np

    rng = np.random.default_rng(int(seed))
    perm = rng.permutation(n)
    n_train = int(round(fractions[0] * n))
    n_val = int(round(fractions[1] * n))
    n_test = n - n_train - n_val
    train_idx = perm[:n_train].tolist()
    val_idx = perm[n_train : n_train + n_val].tolist()
    test_idx = perm[n_train + n_val :].tolist()
    return SplitStats(
        train_idx=sorted(int(i) for i in train_idx),
        val_idx=sorted(int(i) for i in val_idx),
        test_idx=sorted(int(i) for i in test_idx),
        n_train=len(train_idx),
        n_val=len(val_idx),
        n_test=len(test_idx),
    )


def scaffold_split_indices(
    scaffolds: Sequence[Optional[str]],
    fractions: Tuple[float, float, float] = DEFAULT_FRACTIONS,
    seed: int = DEFAULT_SEED,
) -> SplitStats:
    """Bemis-Murcko scaffold-grouped split (Bemis-Murcko 1996).

    Each unique scaffold is assigned to exactly one of
    ``train / val / test`` using a deterministic largest-first policy
    (mirrors :class:`molmetal.data.splits.ScaffoldSplitter`).  Rows
    whose scaffold is ``None`` are bucketed into a sentinel key so they
    still land in one split together.
    """
    n = len(scaffolds)
    if n == 0:
        return SplitStats()

    scaffold_to_rows: Dict[str, List[int]] = defaultdict(list)
    for i, s in enumerate(scaffolds):
        key = s if s else "__empty__"
        scaffold_to_rows[key].append(int(i))

    groups = list(scaffold_to_rows.values())
    # Sort by descending size, then by group-id (sentinel first) for
    # full determinism.
    groups.sort(key=lambda g: (-len(g), g[0] if g else 0))

    n_groups = len(groups)
    n_train = max(1, int(round(fractions[0] * n_groups)))
    n_val = max(1, int(round(fractions[1] * n_groups)))
    n_test = n_groups - n_train - n_val
    if n_test < 1:
        deficit = 1 - n_test
        n_train = max(1, n_train - deficit)
        n_test = n_groups - n_train - n_val

    capacities = {"train": n_train, "val": n_val, "test": n_test}
    assignment: Dict[int, str] = {}

    def _next_split() -> str:
        return max(
            ("train", "val", "test"),
            key=lambda k: (capacities[k], 0 if k == "train" else 1 if k == "val" else 2),
        )

    for g in groups:
        sp = _next_split()
        capacities[sp] -= 1
        for row in g:
            assignment[row] = sp

    train_idx = sorted(i for i, sp in assignment.items() if sp == "train")
    val_idx = sorted(i for i, sp in assignment.items() if sp == "val")
    test_idx = sorted(i for i, sp in assignment.items() if sp == "test")
    return SplitStats(
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        n_train=len(train_idx),
        n_val=len(val_idx),
        n_test=len(test_idx),
    )


# ---------------------------------------------------------------------------
# Overlap computation
# ---------------------------------------------------------------------------
def _set_lookup(arr: Sequence[Optional[str]], indices: Sequence[int]) -> set:
    """Build a set of non-None values from ``arr[indices]``."""
    out = set()
    for i in indices:
        v = arr[int(i)]
        if v:
            out.add(v)
    return out


def compute_overlap(
    canon_smiles: Sequence[str],
    scaffolds: Sequence[Optional[str]],
    split: SplitStats,
) -> Dict[str, SplitOverlap]:
    """Compute per-pair (train∩val, train∩test, val∩test) overlap.

    Returns a dict keyed by ``"train_vs_val"``, ``"train_vs_test"``,
    ``"val_vs_test"`` — each value is a :class:`SplitOverlap`.
    """
    smiles_set = canon_smiles
    scaf_set = scaffolds

    out: Dict[str, SplitOverlap] = {}
    pairs = (
        ("train_vs_val", split.train_idx, split.val_idx),
        ("train_vs_test", split.train_idx, split.test_idx),
        ("val_vs_test", split.val_idx, split.test_idx),
    )
    for name, idx_a, idx_b in pairs:
        sa = _set_lookup(smiles_set, idx_a)
        sb = _set_lookup(smiles_set, idx_b)
        ta = _set_lookup(scaf_set, idx_a)
        tb = _set_lookup(scaf_set, idx_b)
        n_inter_smiles = len(sa & sb)
        n_inter_scafs = len(ta & tb)
        # Row-level "fraction of b whose canon is in a"
        n_b_rows = len(idx_b)
        b_rows_seen = 0
        for i in idx_b:
            v = smiles_set[int(i)]
            if v and v in sa:
                b_rows_seen += 1
        frac_b_smiles_seen = (b_rows_seen / n_b_rows) if n_b_rows else 0.0

        b_scaf_seen = 0
        n_b_unique_scafs = len(tb)
        for s in tb:
            if s in ta:
                b_scaf_seen += 1
        frac_b_scafs_seen = (b_scaf_seen / n_b_unique_scafs) if n_b_unique_scafs else 0.0

        out[name] = SplitOverlap(
            n_intersect_smiles=n_inter_smiles,
            n_intersect_scaffolds=n_inter_scafs,
            n_unique_train_smiles=len(sa),
            n_unique_test_smiles=len(sb),
            n_unique_train_scaffolds=len(ta),
            n_unique_test_scaffolds=len(tb),
            frac_test_smiles_seen_in_train=frac_b_smiles_seen,
            frac_test_scaffolds_seen_in_train=frac_b_scafs_seen,
        )
    return out


# ---------------------------------------------------------------------------
# Per-dataset driver
# ---------------------------------------------------------------------------
def run_dataset(
    dataset_name: str,
    csv_path: Path,
    seed: int = DEFAULT_SEED,
) -> DatasetLeakageReport:
    """Run the leakage check for one CSV.  See module docstring."""
    raw_smiles, raw_scaffolds = load_smiles_csv(csv_path)
    canon, n_canon_fail = canonicalise_smiles(raw_smiles)
    computed_scaffolds, n_scaf_fail = compute_scaffolds(canon)
    # Prefer the dataset's shipped scaffold_id when available (it is
    # already canonical Bemis-Murcko); fall back to locally computed.
    final_scaffolds: List[Optional[str]] = []
    for shipped, computed in zip(raw_scaffolds, computed_scaffolds):
        if shipped is not None and shipped:
            final_scaffolds.append(shipped)
        else:
            final_scaffolds.append(computed)

    n_total = len(raw_smiles)
    n_unique_smiles = len({s for s in canon if s})
    n_unique_scaffolds = len({s for s in final_scaffolds if s})
    n_parse_failures = max(n_canon_fail, n_scaf_fail)

    rep = DatasetLeakageReport(
        dataset_name=dataset_name,
        csv_path=str(csv_path),
        n_total=n_total,
        n_unique_smiles=n_unique_smiles,
        n_unique_scaffolds=n_unique_scaffolds,
        n_parse_failures=int(n_parse_failures),
    )

    # --- Random split -----------------------------------------------------
    random_split = random_split_indices(n_total, seed=seed)
    rep.splits["random"] = random_split
    rep.overlaps["random"] = compute_overlap(canon, final_scaffolds, random_split)
    # A "strict random split" should still have no canonical-SMILES
    # duplicates between train and test (Wiley 2017 / MoleculeNet
    # baseline).  Scaffold overlap is *expected* for random splits and
    # is the diagnostic signal we report (it is *not* a failure of the
    # random split — it is the motivation for the scaffold split).
    rep.strict_pass["random"] = (
        rep.overlaps["random"]["train_vs_test"].n_intersect_smiles == 0
    )

    # --- Scaffold split ---------------------------------------------------
    scaf_split = scaffold_split_indices(final_scaffolds, seed=seed)
    rep.splits["scaffold"] = scaf_split
    rep.overlaps["scaffold"] = compute_overlap(canon, final_scaffolds, scaf_split)
    # A "strict scaffold split" must have zero scaffold overlap between
    # train and test.
    rep.strict_pass["scaffold"] = (
        rep.overlaps["scaffold"]["train_vs_test"].n_intersect_scaffolds == 0
    )

    # --- Shipped split (only if the CSV has a ``split`` column) -----------
    if (csv_path).exists():
        try:
            with csv_path.open(newline="") as fh:
                reader = csv.DictReader(fh)
                fieldnames = reader.fieldnames or []
        except Exception:
            fieldnames = []
        if "split" in fieldnames:
            shipped_split = SplitStats()
            shipped_assignments: Dict[int, str] = {}
            with csv_path.open(newline="") as fh:
                for i, row in enumerate(csv.DictReader(fh)):
                    sp = (row.get("split") or "").strip()
                    if sp:
                        shipped_assignments[i] = sp
            shipped_split.train_idx = sorted(
                i for i, sp in shipped_assignments.items() if sp == "train"
            )
            shipped_split.val_idx = sorted(
                i for i, sp in shipped_assignments.items() if sp == "val"
            )
            shipped_split.test_idx = sorted(
                i for i, sp in shipped_assignments.items() if sp == "test"
            )
            shipped_split.n_train = len(shipped_split.train_idx)
            shipped_split.n_val = len(shipped_split.val_idx)
            shipped_split.n_test = len(shipped_split.test_idx)
            rep.splits["shipped"] = shipped_split
            rep.overlaps["shipped"] = compute_overlap(
                canon, final_scaffolds, shipped_split,
            )
            rep.strict_pass["shipped"] = (
                rep.overlaps["shipped"]["train_vs_test"].n_intersect_scaffolds == 0
            )

    # --- Summary ----------------------------------------------------------
    rep.summary = {
        "random_split": (
            f"strict={'PASS' if rep.strict_pass['random'] else 'FAIL'} "
            f"({rep.overlaps['random']['train_vs_test'].n_intersect_smiles} "
            f"shared canon SMILES; "
            f"{rep.overlaps['random']['train_vs_test'].n_intersect_scaffolds} "
            f"shared Bemis-Murcko scaffolds)"
        ),
        "scaffold_split": (
            f"strict={'PASS' if rep.strict_pass['scaffold'] else 'FAIL'} "
            f"({rep.overlaps['scaffold']['train_vs_test'].n_intersect_scaffolds} "
            f"shared scaffolds; "
            f"{rep.overlaps['scaffold']['train_vs_test'].frac_test_scaffolds_seen_in_train:.1%} "
            f"of test scaffolds in train)"
        ),
    }
    if "shipped" in rep.strict_pass:
        rep.summary["shipped_split"] = (
            f"strict={'PASS' if rep.strict_pass['shipped'] else 'FAIL'} "
            f"({rep.overlaps['shipped']['train_vs_test'].n_intersect_scaffolds} "
            f"shared scaffolds)"
        )
    return rep


# ---------------------------------------------------------------------------
# Report aggregation
# ---------------------------------------------------------------------------
@dataclass
class LeakageReport:
    """Top-level container for all per-dataset leakage checks."""

    datasets: List[DatasetLeakageReport] = field(default_factory=list)
    n_datasets: int = 0
    workflow: str = "Wf-44 Phase 3"
    title: str = "Data leakage diagnostic — train/test split strictness"
    description: str = (
        "Per-dataset check whether the splits used by "
        "train_affinity_regression.py (Wf-32 ship) are strict on the "
        "canonical-SMILES axis (random split) and the Bemis-Murcko "
        "scaffold axis (scaffold split).  Strict ⇒ zero overlap."
    )

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "workflow": self.workflow,
            "title": self.title,
            "description": self.description,
            "n_datasets": self.n_datasets,
            "datasets": [asdict(d) for d in self.datasets],
        }
        return out


def _report_to_dict(d: DatasetLeakageReport) -> Dict[str, Any]:
    """Convert per-dataset dataclass tree to plain dict (json-safe)."""
    return asdict(d)


def write_report(report: LeakageReport, out_path: Path) -> Path:
    """Serialise the report to JSON."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = report.to_dict()
    payload["generated_by"] = "molmetal/scripts/check_data_leakage.py"
    # Strip raw index lists (huge for 5K-row datasets) — keep only counts.
    for d in payload["datasets"]:
        for split_name, split in d.get("splits", {}).items():
            split["train_idx_size"] = len(split.get("train_idx", []))
            split["val_idx_size"] = len(split.get("val_idx", []))
            split["test_idx_size"] = len(split.get("test_idx", []))
            del split["train_idx"]
            del split["val_idx"]
            del split["test_idx"]
    with out_path.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    return out_path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Wf-44 Phase 3 leakage diagnostic — verify that "
            "train/test splits used by train_affinity_regression.py "
            "are strict on canonical SMILES (random) and Bemis-Murcko "
            "scaffold (scaffold)."
        ),
    )
    parser.add_argument(
        "--platinai-csv",
        type=Path,
        default=DEFAULT_PLATINAI_CSV,
        help=f"PlatinAI CSV (default: {DEFAULT_PLATINAI_CSV})",
    )
    parser.add_argument(
        "--metalcytotox-csv",
        type=Path,
        default=DEFAULT_METALCYTOTOX_CSV,
        help=f"MetalCytoToxDB 5-metal multitask CSV (default: {DEFAULT_METALCYTOTOX_CSV})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUT_DIR})",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        default=None,
        help=(
            "Full output JSON path (default: "
            "<out-dir>/leakage_check.json)"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for the random splitter (default: 42)",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    out_dir: Path = args.out_dir
    out_path: Path = args.out_path or (out_dir / "leakage_check.json")

    datasets: List[Tuple[str, Path]] = []
    if args.platinai_csv.exists():
        datasets.append(("platinai_5k", args.platinai_csv))
    else:
        print(
            f"[check_data_leakage] SKIP platinai CSV missing: "
            f"{args.platinai_csv}"
        )
    if args.metalcytotox_csv.exists():
        datasets.append(("metalcytotox_5metal", args.metalcytotox_csv))
    else:
        print(
            f"[check_data_leakage] SKIP metalcytotox CSV missing: "
            f"{args.metalcytotox_csv}"
        )

    rep = LeakageReport()
    for name, path in datasets:
        print(f"[check_data_leakage] running {name} ({path})")
        d = run_dataset(name, path, seed=int(args.seed))
        rep.datasets.append(d)
        print(
            f"[check_data_leakage]   {name}: n_total={d.n_total} "
            f"unique_smiles={d.n_unique_smiles} "
            f"unique_scaffolds={d.n_unique_scaffolds} "
            f"parse_failures={d.n_parse_failures}"
        )
        for split_name, verdict in d.summary.items():
            print(f"[check_data_leakage]   {split_name}: {verdict}")
    rep.n_datasets = len(rep.datasets)

    written = write_report(rep, out_path)
    print(f"[check_data_leakage] wrote {written}")
    print(
        f"[check_data_leakage] n_datasets={rep.n_datasets} "
        f"strict_random_pass="
        f"{sum(1 for d in rep.datasets if d.strict_pass.get('random', False))} "
        f"strict_scaffold_pass="
        f"{sum(1 for d in rep.datasets if d.strict_pass.get('scaffold', False))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())