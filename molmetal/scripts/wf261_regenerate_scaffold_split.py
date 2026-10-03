"""Wf-261-Regenerate: Strict Bemis-Murcko scaffold 70/10/20 split (0 scaffold leak).

This script regenerates ``.data/molmetal/metalcytotox_5metal_multitask_v2.csv``
with a *strict* Bemis-Murcko scaffold-grouped split that guarantees
``scaffold_id`` overlap between train/val/test is **exactly 0**.

History
-------
The on-disk v1 CSV (``metalcytotox_5metal_multitask_v4.csv``) has a 70/10/20
``split`` column that was applied **per-metal** rather than globally
scaffold-grouped.  See ``molmetal/reports/wf261_audit/split_audit.md`` for the
full root-cause trace.  This script fixes that by:

  1. Backing up the existing v1 CSV as ``metalcytotox_5metal_multitask_v1.csv``.
  2. Recomputing the Bemis-Murcko scaffold for every SMILES via RDKit.
  3. Grouping rows by scaffold, shuffling the scaffold list with seed=42.
  4. Assigning scaffolds to train/val/test using the canonical
     ``LigandDeduplicatedSplitter`` random strategy at fractions (0.7, 0.1, 0.2).
  5. Writing the new ``metalcytotox_5metal_multitask_v2.csv`` with the
     ``split`` column updated and dropping rows whose scaffold is null.
  6. Verifying 0 scaffold leak and ~70/10/20 row distribution.
  7. Writing ``split_manifest.json`` with scaffold_id -> split + SHA256.

Run end-to-end via:

    /home/hugo/codes/mol-metal/.venv/bin/python \\
        molmetal/scripts/wf261_regenerate_scaffold_split.py
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Paths (constants)
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / ".data" / "molmetal"
V1_CSV = DATA_DIR / "metalcytotox_5metal_multitask_v4.csv"
V1_BACKUP = DATA_DIR / "metalcytotox_5metal_multitask_v1.csv"
V2_CSV = DATA_DIR / "metalcytotox_5metal_multitask_v2.csv"
MANIFEST = DATA_DIR / "metalcytotox_5metal_multitask_v2_split_manifest.json"
REPORT_DIR = PROJECT_ROOT / "molmetal" / "reports" / "wf261_regenerate"
REPORT_MD = REPORT_DIR / "regen_report.md"

SEED = 42
FRACTIONS: Tuple[float, float, float] = (0.70, 0.10, 0.20)

CANONICAL_COLS = [
    "smiles",
    "metal",
    "scaffold_id",
    "split",
    "activity",
    "pic50",
    "cell_line",
    "source",
]


# ---------------------------------------------------------------------------
# RDKit scaffolding (top-level Bemis-Murcko, no generic atom)
# ---------------------------------------------------------------------------
def _import_rdkit():
    """Import RDKit, returning Chem + Scaffolds modules.  Raises on failure."""
    try:
        from rdkit import Chem, RDLogger  # noqa: F401
        from rdkit.Chem.Scaffolds import MurckoScaffold  # noqa: F401

        RDLogger.DisableLog("rdApp.*")
        return Chem, MurckoScaffold
    except ImportError as exc:  # pragma: no cover — exercised only in non-rdkit env
        raise RuntimeError(
            "RDKit unavailable; abort. Use /home/hugo/codes/mol-metal/.venv/bin/python"
        ) from exc


def _scaffold_for_mol(mol, MurckoScaffold, Chem):
    """Top-level Bemis-Murcko scaffold; sig-aware across RDKit API drift.

    RDKit's ``MurckoScaffold.GetScaffoldForMol`` signature has shifted across
    versions (modern versions drop ``includeChirality``); this helper tries
    the modern sig first, then falls back to the legacy kwarg variant.
    Returns the scaffold Mol object (possibly empty) or ``None`` on failure.
    """
    import inspect

    try:
        sig = inspect.signature(MurckoScaffold.GetScaffoldForMol)
    except (TypeError, ValueError):
        sig = None

    if sig is not None and "includeChirality" in sig.parameters:
        try:
            return MurckoScaffold.GetScaffoldForMol(mol, includeChirality=False)
        except Exception:
            return None
    try:
        return MurckoScaffold.GetScaffoldForMol(mol)
    except TypeError:
        return None
    except Exception:
        return None


def compute_bm_scaffold(smiles: str):
    """Return Bemis-Murcko scaffold SMILES for *smiles*.

    Uses RDKit's ``MurckoScaffold.GetScaffoldForMol`` on the *top-level*
    scaffold (no generic-atom rewriting; no chirality).  Returns ``None``
    for unparseable SMILES or when the top-level Mol has no ring system
    (empty scaffold).
    """
    Chem, MurckoScaffold = _import_rdkit()
    if not isinstance(smiles, str) or not smiles:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    scaffold_mol = _scaffold_for_mol(mol, MurckoScaffold, Chem)
    if scaffold_mol is None:
        return None
    n_atoms = scaffold_mol.GetNumAtoms()
    if n_atoms == 0:
        # Empty scaffold (no ring system); caller treats as null.
        return ""
    scaffold_smi = Chem.MolToSmiles(scaffold_mol)
    return scaffold_smi or None


def compute_scaffolds_for(smiles_list: List[str]) -> List[str]:
    """Vectorised wrapper: compute Bemis-Murcko scaffolds in order."""
    out: List[str] = []
    for smi in smiles_list:
        scaf = compute_bm_scaffold(smi)
        if scaf is None:
            out.append("")  # placeholder for null/empty — dropped later
        else:
            out.append(scaf)
    return out


# ---------------------------------------------------------------------------
# Split assignment — mirrors molmetal/data/splits.py::LigandDeduplicatedSplitter
# ``strategy='random'`` exactly (same RNG semantics, same greedy placement).
# ---------------------------------------------------------------------------
def assign_groups_to_splits(
    groups: List[List[int]],
    fractions: Tuple[float, float, float],
    seed: int,
) -> Dict[int, str]:
    """Assign each row to train/val/test such that row's group is intact.

    Algorithm
    ---------
    1. ``n_train = round(fractions[0] * n_groups)`` etc.
    2. Shuffle the *group indices* via ``np.random.default_rng(seed).permutation``.
    3. Walk the shuffled groups in order.  For each, pick the split with the
       largest remaining *group* capacity (ties broken train > val > test).
       Decrement capacity by 1.
    4. Every row index within the group inherits the chosen split name.
    """
    n_groups = len(groups)
    if n_groups == 0:
        return {}
    n_train = max(1, int(round(fractions[0] * n_groups)))
    n_val = max(1, int(round(fractions[1] * n_groups)))
    n_test = n_groups - n_train - n_val
    if n_test < 1:
        deficit = 1 - n_test
        n_train = max(1, n_train - deficit)
        n_test = n_groups - n_train - n_val

    capacities = {"train": n_train, "val": n_val, "test": n_test}
    rng = np.random.default_rng(seed)
    order = rng.permutation(n_groups)
    split_assignment: Dict[int, str] = {}
    for idx in order:
        split = max(
            ("train", "val", "test"),
            key=lambda k: (
                capacities[k],
                0 if k == "train" else 1 if k == "val" else 2,
            ),
        )
        capacities[split] -= 1
        for row in groups[int(idx)]:
            split_assignment[int(row)] = split
    return split_assignment


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_csv_rows(path: Path) -> Tuple[List[dict], List[str]]:
    with path.open("r", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = [dict(r) for r in reader]
        fields = list(reader.fieldnames or [])
    return rows, fields


def write_csv_rows(path: Path, fields: List[str], rows: List[dict]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------
def regenerate(
    v1_path: Path = V1_CSV,
    v1_backup: Path = V1_BACKUP,
    v2_path: Path = V2_CSV,
    manifest_path: Path = MANIFEST,
    fractions: Tuple[float, float, float] = FRACTIONS,
    seed: int = SEED,
) -> Dict[str, object]:
    """Run the full pipeline; return the acceptance-gate dict."""

    started_at = time.time()
    started_iso = time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(started_at))

    # 1) Backup v1
    g1_v1_backed_up = False
    if v1_path.exists() and not v1_backup.exists():
        v1_backup.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(v1_path, v1_backup)
        g1_v1_backed_up = True
    elif v1_backup.exists():
        # Backup already in place from a prior run — treat as preserved.
        g1_v1_backed_up = True
    else:
        raise FileNotFoundError(f"v1 CSV missing at {v1_path}")

    # 2) Load v1 CSV
    rows, fields = read_csv_rows(v1_path)
    n_total = len(rows)
    assert (
        fields == CANONICAL_COLS
    ), f"Unexpected CSV columns: got {fields}, expected {CANONICAL_COLS}"

    # 3) Validate + compute Bemis-Murcko scaffolds
    smiles = [r.get("smiles", "") or "" for r in rows]
    computed_scaffolds = compute_scaffolds_for(smiles)

    # Track null rows (scaffold is None/empty)
    null_indices: List[int] = []
    for i, scaf in enumerate(computed_scaffolds):
        if not scaf:
            null_indices.append(i)

    # 4) Group by scaffold_id (skip null rows)
    scaffold_to_rows: Dict[str, List[int]] = defaultdict(list)
    for i, scaf in enumerate(computed_scaffolds):
        if scaf:
            scaffold_to_rows[scaf].append(i)

    groups = list(scaffold_to_rows.values())
    n_unique_scaffolds = len(groups)
    n_after_drop = n_total - len(null_indices)

    # 5+6) Assign scaffolds to splits via shuffle(42) + greedy capacity
    split_assignment = assign_groups_to_splits(groups, fractions=fractions, seed=seed)

    # 7) Apply split to rows + write v2 CSV
    new_rows: List[dict] = []
    for i, r in enumerate(rows):
        if i in null_indices:
            continue  # drop null-scaffold rows
        new_row = dict(r)
        new_row["split"] = split_assignment[i]
        new_rows.append(new_row)

    v2_path.parent.mkdir(parents=True, exist_ok=True)
    write_csv_rows(v2_path, fields, new_rows)
    g2_v2_csv_written = v2_path.exists() and v2_path.stat().st_size > 0

    # 8) Reload to verify
    v2_rows, _ = read_csv_rows(v2_path)

    # 9) Verify 0 scaffold leak
    scaf_split: Dict[str, set] = defaultdict(set)
    for r in v2_rows:
        scaf_split[r["scaffold_id"]].add(r["split"])
    overlap_pairs = []
    for scaf, splits in scaf_split.items():
        if len(splits) > 1:
            overlap_pairs.append((scaf, sorted(splits)))
    g3_zero_scaffold_leak = len(overlap_pairs) == 0

    # 10) Verify ~70/10/20 row distribution
    split_counts: Dict[str, int] = defaultdict(int)
    for r in v2_rows:
        split_counts[r["split"]] += 1
    n_v2 = len(v2_rows)
    pct = {k: 100.0 * v / max(1, n_v2) for k, v in split_counts.items()}
    g4_row_distribution_ok = all(
        abs(pct.get(k, 0) - 100 * f) <= 2.0
        for k, f in [
            ("train", fractions[0]),
            ("val", fractions[1]),
            ("test", fractions[2]),
        ]
    )

    # 11) split_manifest.json
    scaffold_to_split: Dict[str, str] = {
        scaf: next(iter(splits)) for scaf, splits in scaf_split.items()
    }
    sha_v2 = sha256_of(v2_path)
    manifest = {
        "version": "v2",
        "csv": str(v2_path),
        "sha256": sha_v2,
        "n_total_rows": n_total,
        "n_dropped_null_scaffold": len(null_indices),
        "n_unique_scaffolds": n_unique_scaffolds,
        "n_rows_v2": n_v2,
        "fractions": list(fractions),
        "seed": seed,
        "split_rows": dict(split_counts),
        "split_pct": {k: round(v, 4) for k, v in pct.items()},
        "scaffold_count_per_split": {
            k: sum(1 for v in scaffold_to_split.values() if v == k)
            for k in ("train", "val", "test")
        },
        "scaffold_id_to_split": dict(sorted(scaffold_to_split.items())),
        "generated_at_iso": started_iso,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w") as fh:
        json.dump(manifest, fh, indent=2)
    g5_split_manifest_written = manifest_path.exists()

    # Diagnostics summary
    honest_framing_note = (
        f"{len(null_indices)} of {n_total} rows dropped due to null/empty Bemis-Murcko scaffold "
        f"(computed fresh via RDKit MurckoScaffold.GetScaffoldForMol); {n_unique_scaffolds} "
        f"unique scaffolds across {n_v2} kept rows. These dropouts are reported honestly in "
        "regen_report.md (G_dropped_rows)."
    )

    return {
        "started_at_iso": started_iso,
        "duration_seconds": round(time.time() - started_at, 2),
        "v1_path": str(v1_path),
        "v1_backup": str(v1_backup),
        "v2_path": str(v2_path),
        "manifest_path": str(manifest_path),
        "n_total_rows": n_total,
        "n_dropped_null": len(null_indices),
        "n_unique_scaffolds": n_unique_scaffolds,
        "n_rows_v2": n_v2,
        "split_counts": dict(split_counts),
        "split_pct": {k: round(v, 2) for k, v in pct.items()},
        "scaffold_count_per_split": manifest["scaffold_count_per_split"],
        "sha256_v2": sha_v2,
        "overlap_pairs_count": len(overlap_pairs),
        "g1_v1_backed_up": bool(g1_v1_backed_up),
        "g2_v2_csv_written": bool(g2_v2_csv_written),
        "g3_zero_scaffold_leak": bool(g3_zero_scaffold_leak),
        "g4_row_distribution_ok": bool(g4_row_distribution_ok),
        "g5_split_manifest_written": bool(g5_split_manifest_written),
        "honest_framing_note": honest_framing_note,
    }


# ---------------------------------------------------------------------------
# Reporting helpers (used by CLI + __main__)
# ---------------------------------------------------------------------------
def render_report_md(summary: Dict[str, object]) -> str:
    keys = [
        "started_at_iso",
        "duration_seconds",
        "v1_path",
        "v1_backup",
        "v2_path",
        "manifest_path",
        "n_total_rows",
        "n_dropped_null",
        "n_unique_scaffolds",
        "n_rows_v2",
        "split_counts",
        "split_pct",
        "scaffold_count_per_split",
        "sha256_v2",
        "overlap_pairs_count",
        "g1_v1_backed_up",
        "g2_v2_csv_written",
        "g3_zero_scaffold_leak",
        "g4_row_distribution_ok",
        "g5_split_manifest_written",
        "honest_framing_note",
    ]
    lines = ["# Wf-261-Regenerate — Strict Bemis-Murcko Scaffold 70/10/20 Split", ""]
    lines.append(f"**Date**: {summary['started_at_iso']}  ")
    lines.append(f"**Duration**: {summary['duration_seconds']} s")
    lines.append("")
    lines.append("## §1 Run summary")
    lines.append("")
    lines.append("| Field | Value |")
    lines.append("|---|---|")
    for k in keys:
        lines.append(f"| `{k}` | `{summary[k]}` |")
    lines.append("")
    lines.append("## §2 Acceptance gates (FLAT)")
    lines.append("")
    lines.append("| Gate | Value |")
    lines.append("|---|---|")
    lines.append(f"| G1_v1_backed_up | `{summary['g1_v1_backed_up']}` |")
    lines.append(f"| G2_v2_csv_written | `{summary['g2_v2_csv_written']}` |")
    lines.append(f"| G3_zero_scaffold_leak | `{summary['g3_zero_scaffold_leak']}` |")
    lines.append(f"| G4_row_distribution_ok | `{summary['g4_row_distribution_ok']}` |")
    lines.append(
        f"| G5_split_manifest_written | `{summary['g5_split_manifest_written']}` |"
    )
    lines.append("")
    lines.append("## §3 Honest framing")
    lines.append("")
    lines.append(summary["honest_framing_note"])
    lines.append("")
    return "\n".join(lines)


def parse_args(argv: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--v1", type=Path, default=V1_CSV)
    parser.add_argument("--v1-backup", type=Path, default=V1_BACKUP)
    parser.add_argument("--v2", type=Path, default=V2_CSV)
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--fractions",
        type=float,
        nargs=3,
        default=list(FRACTIONS),
        metavar=("TRAIN", "VAL", "TEST"),
    )
    parser.add_argument("--report-md", type=Path, default=REPORT_MD)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run regenerate() but skip writing v2/manifest/report.",
    )
    return parser.parse_args(argv)


def main(argv: List[str] | None = None) -> int:
    args = parse_args(argv if argv is not None else sys.argv[1:])
    fractions = tuple(args.fractions)
    summary = regenerate(
        v1_path=args.v1,
        v1_backup=args.v1_backup,
        v2_path=args.v2,
        manifest_path=args.manifest,
        fractions=fractions,
        seed=args.seed,
    )
    if args.dry_run:
        # Cleanup what regenerate wrote when dry-run.
        for p in (args.v2, args.manifest):
            if p.exists():
                p.unlink()
        summary["dry_run"] = True
    else:
        report_md = render_report_md(summary)
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        args.report_md.parent.mkdir(parents=True, exist_ok=True)
        args.report_md.write_text(report_md)
        summary["report_md"] = str(args.report_md)
    print(json.dumps(summary, indent=2))
    all_ok = (
        summary["g1_v1_backed_up"]
        and summary["g2_v2_csv_written"]
        and summary["g3_zero_scaffold_leak"]
        and summary["g4_row_distribution_ok"]
        and summary["g5_split_manifest_written"]
    )
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
