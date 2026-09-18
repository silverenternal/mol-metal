"""Bemis-Murcko scaffold split for the metallodrug training pool.

TODO-30 P6.1 — fill the Tier-2 reproducibility gap on the *training-side*
split.  Random splits leak between train and test on chemically near-
identical metallodrug scaffolds; this script enforces a Bemis-Murcko
scaffold split with an optional Tanimoto threshold (default 0.6) so two
SMILES that share a similar Murcko scaffold cannot land in different
splits.

Inputs
------
- molmetal/data/metallo_drugs_500_train.csv (default)
  Columns: smiles, source
  ``source`` may carry a metal token (e.g. ``platinai``, ``tmqm``).
  Rows with the same metal token are summed into the per-metal coverage
  report.

Outputs
-------
- <output-dir>/train.csv
- <output-dir>/val.csv
- <output-dir>/test.csv

Each output row keeps the original ``smiles`` and ``source`` columns,
plus a new ``scaffold_id`` column (canonical Murcko scaffold SMILES).

CLI
---
    uv run python molmetal/scripts/metallo_pool_scaffold_split.py \
        --input  molmetal/data/metallo_drugs_500_train.csv \
        --output-dir molmetal/data/metallo_scaffold_split \
        --fractions 0.7 0.15 0.15 \
        --tanimoto-threshold 0.6

The CLI is intentionally narrow: a single seed (default 42) is used so
re-running reproduces the split bit-for-bit (forward-compat with the
``seeds.json`` hygiene shipped in P6.2).

P6.3 (MMseqs2 + Tanimoto<0.4) is deferred; this script is the
scaffold-only path required by the P6.1 reinforcement item.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import random
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

# Make ``molmetal`` importable when run as ``python molmetal/scripts/...``.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent.parent
for _p in (str(_HERE.parent), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
log = logging.getLogger("metallo_pool_scaffold_split")


# ---------------------------------------------------------------------------
# Metals coverage — the audit asks for Pt/Pd/Au/Ir/Ru coverage.  We use a
# loose metal-token lookup because the metallo_drugs_500_train.csv uses
# vendor prefixes (platinai, tmqm, ...).  The fallback uses an RDKit scan
# over atomic numbers for safety.
# ---------------------------------------------------------------------------
METAL_TOKENS: Dict[str, Tuple[str, ...]] = {
    "Pt": ("Pt", "platin"),
    "Pd": ("Pd", "pallad"),
    "Au": ("Au", "aur", "gold"),
    "Ir": ("Ir", "irid"),
    "Ru": ("Ru", "ruth"),
}


def detect_metal(smiles: str, source: str = "") -> str:
    """Return one of {Pt, Pd, Au, Ir, Ru, OTHER}."""
    s = (smiles or "") + " " + (source or "")
    for metal, tokens in METAL_TOKENS.items():
        for tok in tokens:
            if tok in s:
                return metal
    # Fallback: scan atomic numbers via RDKit.
    try:
        from rdkit import Chem  # type: ignore
        mol = Chem.MolFromSmiles(smiles or "")
        if mol is not None:
            atomic_nums = {a.GetAtomicNum() for a in mol.GetAtoms()}
            for atomic_num, metal in ((78, "Pt"), (46, "Pd"), (79, "Au"),
                                       (77, "Ir"), (44, "Ru")):
                if atomic_num in atomic_nums:
                    return metal
    except Exception:
        pass
    return "OTHER"


# ---------------------------------------------------------------------------
# Scaffold computation — RDKit Bemis-Murcko scaffold canonical SMILES.
# Tanimoto clustering uses Morgan radius-2 fingerprints so two near-
# identical scaffolds collapse into a single split group.
# ---------------------------------------------------------------------------
def _rdkit_imports():
    from rdkit import Chem  # type: ignore
    from rdkit.Chem.Scaffolds import MurckoScaffold  # type: ignore
    from rdkit import DataStructs  # type: ignore
    from rdkit.Chem import AllChem  # type: ignore
    return Chem, MurckoScaffold, DataStructs, AllChem


def _scaffold_smiles(smiles: str) -> Optional[str]:
    try:
        Chem, MurckoScaffold, _, _ = _rdkit_imports()
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None or scaf.GetNumAtoms() == 0:
            return ""
        return Chem.MolToSmiles(scaf, canonical=True)
    except Exception:
        return None


def _tanimoto(fp_a, fp_b) -> float:
    """DataStructs.TanimotoSimilarity; accepts RDKit ExplicitBitVect."""
    from rdkit import DataStructs  # type: ignore
    return float(DataStructs.TanimotoSimilarity(fp_a, fp_b))


def _cluster_scaffolds_by_tanimoto(
    scaffolds: Sequence[str],
    threshold: float,
) -> Dict[str, int]:
    """Assign cluster ids so two scaffolds with Tanimoto >= threshold share an id.

    Returns a mapping scaffold_smiles -> cluster_id (int).  Uses single-
    link clustering: a new scaffold joins the first existing cluster whose
    Tanimoto >= threshold.  Deterministic given input order.
    """
    Chem, _, _, AllChem = _rdkit_imports()
    fps: List = []
    for s in scaffolds:
        if not s:
            fps.append(None)
            continue
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            fps.append(None)
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
    cluster_id: Dict[str, int] = {}
    clusters: List[Dict[str, "object"]] = []  # list of { "rep": fp, "members": set[str] }
    for scaf, fp in zip(scaffolds, fps):
        if scaf == "" or fp is None:
            # Empty / unknown scaffold gets its own bucket to guarantee determinism.
            cluster_id.setdefault(scaf, len(clusters))
            if scaf not in {m for c in clusters for m in c["members"]}:
                clusters.append({"rep": None, "members": {scaf}})
            continue
        joined = False
        for idx, c in enumerate(clusters):
            if c["rep"] is None:
                continue
            if _tanimoto(fp, c["rep"]) >= threshold:
                c["members"].add(scaf)
                cluster_id[scaf] = idx
                joined = True
                break
        if not joined:
            new_id = len(clusters)
            clusters.append({"rep": fp, "members": {scaf}})
            cluster_id[scaf] = new_id
    return cluster_id


# ---------------------------------------------------------------------------
# Deterministic size-balanced assignment across (train, val, test).
# We sort cluster ids by size descending, then assign each to the
# smallest-current bucket.  This keeps the splits close to the requested
# fractions even when cluster sizes are skewed.
# ---------------------------------------------------------------------------
@dataclass
class SplitResult:
    n_clusters: int
    bucket_assignment: Dict[int, str]  # cluster_id -> bucket name
    rows_by_bucket: Dict[str, List[dict]] = field(default_factory=dict)


def _balanced_assign(
    cluster_sizes: Dict[int, int],
    fractions: Tuple[float, float, float],
    bucket_names: Tuple[str, str, str] = ("train", "val", "test"),
    seed: int = 42,
) -> Dict[int, str]:
    """Assign clusters to buckets to keep cumulative size close to fractions."""
    rng = random.Random(seed)
    # Sort by size desc, tie-break by cluster id asc; deterministic with seed.
    ordered = sorted(cluster_sizes.items(), key=lambda kv: (-kv[1], kv[0]))
    target = {name: max(0.0, frac) for name, frac in zip(bucket_names, fractions)}
    total = sum(cluster_sizes.values())
    if total == 0:
        return {cid: bucket_names[0] for cid in cluster_sizes}
    target_count = {name: round(frac * total) for name, frac in target.items()}
    # Greedy fill: assign each cluster to the bucket with the largest deficit.
    assigned: Dict[int, str] = {}
    current = {name: 0 for name in bucket_names}
    for cid, size in ordered:
        # Largest deficit = max(target_count[name] - current[name]); ties broken by order.
        deficits = {name: target_count[name] - current[name] for name in bucket_names}
        best_name = max(bucket_names, key=lambda n: (deficits[n], -ord(n[0])))
        # Add a tiny deterministic tiebreak so e.g. val never wins a 0-0 tie over test
        # via a global Python hash; reuse the seeded rng for reproducibility.
        if all(v == 0 for v in deficits.values()):
            best_name = rng.choice(bucket_names)
        assigned[cid] = best_name
        current[best_name] += size
    return assigned


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def load_input(csv_path: Path) -> List[dict]:
    rows: List[dict] = []
    with csv_path.open(newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            smi = (row.get("smiles") or "").strip()
            if not smi:
                continue
            rows.append({"smiles": smi, "source": (row.get("source") or "").strip()})
    return rows


def compute_scaffolds(rows: Iterable[dict]) -> Tuple[List[str], List[Optional[str]]]:
    """Return (smiles, scaffold_smiles) parallel lists."""
    out: List[str] = []
    out_scaf: List[Optional[str]] = []
    for r in rows:
        out.append(r["smiles"])
        out_scaf.append(_scaffold_smiles(r["smiles"]))
    return out, out_scaf


def split_pool(
    rows: List[dict],
    fractions: Tuple[float, float, float] = (0.7, 0.15, 0.15),
    tanimoto_threshold: float = 0.6,
    seed: int = 42,
) -> SplitResult:
    if abs(sum(fractions) - 1.0) > 1e-3:
        raise ValueError(f"fractions must sum to ~1.0, got {fractions}")
    if len(fractions) != 3:
        raise ValueError("split_pool expects 3 fractions (train, val, test)")
    smiles, scaffolds = compute_scaffolds(rows)
    # Bucket "no scaffold" entries into a single deterministic group.
    cleaned_scaf = [s if s is not None else "__invalid__" for s in scaffolds]
    cluster_id = _cluster_scaffolds_by_tanimoto(cleaned_scaf, tanimoto_threshold)
    # Group rows by cluster.
    cluster_to_rows: Dict[int, List[int]] = defaultdict(list)
    for idx, scaf in enumerate(cleaned_scaf):
        cid = cluster_id[scaf]
        cluster_to_rows[cid].append(idx)
    cluster_sizes = {cid: len(idxs) for cid, idxs in cluster_to_rows.items()}
    assignment = _balanced_assign(cluster_sizes, fractions, seed=seed)
    # Build per-bucket rows; assign scaffold_id (canonical Murcko) per row.
    buckets: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}
    for cid, idxs in cluster_to_rows.items():
        bucket = assignment[cid]
        for idx in idxs:
            row = dict(rows[idx])
            scaf = scaffolds[idx] or ""
            row["scaffold_id"] = scaf
            row["cluster_id"] = cid
            row["metal"] = detect_metal(row["smiles"], row["source"])
            buckets[bucket].append(row)
    return SplitResult(
        n_clusters=len(cluster_to_rows),
        bucket_assignment=assignment,
        rows_by_bucket=buckets,
    )


def coverage_by_metal(rows: List[dict]) -> Dict[str, int]:
    out: Dict[str, int] = defaultdict(int)
    for r in rows:
        out[r["metal"]] += 1
    return dict(sorted(out.items()))


def write_outputs(result: SplitResult, output_dir: Path) -> Tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for name, rows in result.rows_by_bucket.items():
        path = output_dir / f"{name}.csv"
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(
                fh, fieldnames=["smiles", "source", "metal", "scaffold_id", "cluster_id"]
            )
            writer.writeheader()
            for r in rows:
                writer.writerow(r)
        paths.append(path)
        log.info("Wrote %d rows to %s", len(rows), path)
    return tuple(paths)  # type: ignore[return-value]


def write_manifest(
    result: SplitResult,
    output_dir: Path,
    fractions: Tuple[float, float, float],
    tanimoto_threshold: float,
    seed: int,
) -> Path:
    coverage = {name: coverage_by_metal(rows) for name, rows in result.rows_by_bucket.items()}
    summary = {
        "schema_version": 1,
        "tool": "metallo_pool_scaffold_split.py",
        "input": str(output_dir.parent / "metallo_drugs_500_train.csv"),
        "n_clusters": result.n_clusters,
        "fractions": list(fractions),
        "tanimoto_threshold": tanimoto_threshold,
        "seed": seed,
        "bucket_sizes": {name: len(rows) for name, rows in result.rows_by_bucket.items()},
        "bucket_metal_coverage": coverage,
    }
    path = output_dir / "manifest.json"
    path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    log.info("Wrote manifest %s", path)
    return path


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--input",
        default=str(_REPO_ROOT / "molmetal/data/metallo_drugs_500_train.csv"),
        help="CSV with at minimum columns smiles, source",
    )
    p.add_argument(
        "--output-dir",
        default=str(_REPO_ROOT / "molmetal/data/metallo_scaffold_split"),
        help="Where to write train.csv / val.csv / test.csv / manifest.json",
    )
    p.add_argument(
        "--fractions",
        nargs=3,
        type=float,
        default=[0.7, 0.15, 0.15],
        metavar=("TRAIN", "VAL", "TEST"),
        help="Per-bucket fraction (must sum to 1.0)",
    )
    p.add_argument(
        "--tanimoto-threshold",
        type=float,
        default=0.6,
        help="Murcko-scaffold Tanimoto threshold for clustering (default 0.6)",
    )
    p.add_argument("--seed", type=int, default=42, help="Deterministic seed for bucket assignment")
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_argparser().parse_args(argv)
    rows = load_input(Path(args.input))
    log.info("Loaded %d rows from %s", len(rows), args.input)
    result = split_pool(
        rows,
        fractions=tuple(args.fractions),
        tanimoto_threshold=float(args.tanimoto_threshold),
        seed=int(args.seed),
    )
    output_dir = Path(args.output_dir)
    paths = write_outputs(result, output_dir)
    manifest = write_manifest(
        result,
        output_dir,
        fractions=tuple(args.fractions),
        tanimoto_threshold=float(args.tanimoto_threshold),
        seed=int(args.seed),
    )
    coverage = {name: coverage_by_metal(rows) for name, rows in result.rows_by_bucket.items()}
    print(json.dumps({
        "n_rows": len(rows),
        "n_clusters": result.n_clusters,
        "bucket_sizes": {n: len(rs) for n, rs in result.rows_by_bucket.items()},
        "bucket_metal_coverage": coverage,
        "outputs": [str(p) for p in paths],
        "manifest": str(manifest),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
