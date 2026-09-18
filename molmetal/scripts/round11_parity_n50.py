"""Round-11 N=50 Vina vs QuickVina2 paired parity experiment.

Picks the first 50 SMILES from the round-11 unique click library
(molmetal/reports/round11_engine_parity/click12_x5rules_unique27.csv),
then docks each one with the same box / exhaustiveness / seed in both
the AutoDock Vina 1.2.7 Python binding and the bundled QVina 2 binary.

Outputs:
    molmetal/reports/round11_engine_parity/parity_raw.csv
    molmetal/reports/round11_engine_parity/parity_metrics.json
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path("/home/hugo/codes/try_triton_on_rocm")
sys.path.insert(0, str(REPO))

import numpy as np
import torch
from rdkit import Chem

from molmetal.domain import Molecule, Pocket
from molmetal.ports import DockingConfig
from molmetal_lam.sbdd_env.vina_adapter import VinaDockingAdapter


MOL_LIST = (
    REPO / "molmetal" / "reports" / "round11_engine_parity"
    / "n50_smiles.csv"
)
EXAMPLE_PDB = (
    REPO / "molmetal" / "references" / "targetdiff" / "examples"
    / "1h36_A_rec_1h36_r88_lig_tt_docked_0_pocket10.pdb"
)
EXAMPLE_LIG = (
    REPO / "molmetal" / "references" / "targetdiff" / "examples"
    / "1h36_A_rec_1h36_r88_lig_tt_docked_0.sdf"
)
REPORT_DIR = REPO / "molmetal" / "reports" / "round11_engine_parity"


@dataclass
class PocketBundle:
    pocket: Pocket
    center: np.ndarray
    box_size: float


def load_pocket() -> PocketBundle:
    ref = next(iter(Chem.SDMolSupplier(str(EXAMPLE_LIG), removeHs=False)))
    heavy = [a.GetIdx() for a in ref.GetAtoms() if a.GetAtomicNum() > 1]
    xyz = ref.GetConformer().GetPositions()[heavy]
    center = xyz.mean(axis=0)
    side = max(12.0, 2.0 * float(np.abs(xyz - center).max()) + 8.0)
    pocket = Pocket.from_pdb_file(
        str(EXAMPLE_PDB),
        torch.tensor(center, dtype=torch.float32),
        radius=side / 2.0,
    )
    # The Vina adapter looks for _pdb_path (private sidecar) on the
    # frozen Pocket dataclass.
    object.__setattr__(pocket, "_pdb_path", str(EXAMPLE_PDB))
    return PocketBundle(pocket=pocket, center=center, box_size=side)


def load_smiles(n: int) -> list[str]:
    rows: list[tuple[int, str]] = []
    with MOL_LIST.open() as fh:
        reader = csv.DictReader(fh)
        for r in reader:
            smi = (r.get("smi") or r.get("canonical_smiles") or "").strip()
            if not smi:
                continue
            rows.append((int(r["idx"]), smi))
    return [s for _, s in rows[:n]]


def dock_one(engine: str, smiles: str, pocket: PocketBundle,
             exhaustiveness: int, n_poses: int, seed: int) -> tuple[float | None, str]:
    adapter = VinaDockingAdapter(
        engine=engine, cpu_count=1, default_box_padding=0,
    )
    pdbqt = adapter._prepare_receptor(pocket.pocket)
    adapter._receptor_pdbqt[pocket.pocket.pdb_id] = pdbqt
    config = DockingConfig(seed=seed, exhaustiveness=exhaustiveness, n_poses=n_poses)
    try:
        # Pre-validate SMILES (the adapter would fail inside dock anyway,
        # but we want a structured failure row).
        if Chem.MolFromSmiles(smiles) is None:
            return None, "rdkit_parse_failed"
        mol_obj = Molecule.from_smiles(smiles)
        complexes = adapter.dock(mol_obj, pocket.pocket, config)
    except Exception as exc:  # noqa: BLE001
        return None, f"{type(exc).__name__}: {exc}".replace("\n", " ")[:200]
    if not complexes:
        return None, "no_poses_returned"
    best = min(c.vina_score for c in complexes)
    return float(best), "ok"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n", type=int, default=50, help="number of molecules (default 50)")
    p.add_argument("--exhaustiveness", type=int, default=8)
    p.add_argument("--n-poses", type=int, default=1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", type=Path, default=REPORT_DIR)
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    pocket = load_pocket()
    smiles_list = load_smiles(args.n)
    if not smiles_list:
        raise SystemExit(f"No SMILES loaded from {MOL_LIST}")

    raw_path = args.out / "parity_raw.csv"
    metrics_path = args.out / "parity_metrics.json"

    rows: list[dict] = []
    t0 = time.time()
    for i, smi in enumerate(smiles_list):
        v_kcal, v_status = dock_one(
            "vina", smi, pocket, args.exhaustiveness, args.n_poses, args.seed,
        )
        q_kcal, q_status = dock_one(
            "quickvina2", smi, pocket, args.exhaustiveness, args.n_poses, args.seed,
        )
        rows.append({
            "idx": i + 1,
            "smi": smi,
            "vina_kcal": "" if v_kcal is None else f"{v_kcal:.4f}",
            "vina_status": v_status,
            "quickvina_kcal": "" if q_kcal is None else f"{q_kcal:.4f}",
            "quickvina_status": q_status,
        })
        print(
            f"[{i+1:02d}/{len(smiles_list)}] V={v_status:>20s}/{v_kcal!s:>8s} "
            f"Q={q_status:>20s}/{q_kcal!s:>8s}  smi={smi[:40]!s}",
            flush=True,
        )

    with raw_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=["idx", "smi", "vina_kcal", "vina_status",
                        "quickvina_kcal", "quickvina_status"],
        )
        writer.writeheader()
        writer.writerows(rows)

    # Pairwise metrics on molecules that succeeded in BOTH engines.
    pairs: list[tuple[float, float]] = []
    failures: list[dict] = []
    for r in rows:
        v_raw, q_raw = r["vina_kcal"], r["quickvina_kcal"]
        if v_raw == "" or q_raw == "":
            failures.append({"idx": r["idx"], "smi": r["smi"],
                             "vina_status": r["vina_status"],
                             "quickvina_status": r["quickvina_status"]})
            continue
        pairs.append((float(v_raw), float(q_raw)))

    n_total = len(rows)
    n_vina_ok = sum(1 for r in rows if r["vina_kcal"] != "")
    n_quickvina_ok = sum(1 for r in rows if r["quickvina_kcal"] != "")
    n_pairs = len(pairs)

    metrics: dict[str, object] = {
        "n_total": n_total,
        "n_vina_ok": n_vina_ok,
        "n_quickvina_ok": n_quickvina_ok,
        "n_pairs": n_pairs,
        "n_failures": len(failures),
        "failures": failures,
        "wall_time_seconds": round(time.time() - t0, 1),
        "exhaustiveness": args.exhaustiveness,
        "n_poses": args.n_poses,
        "seed": args.seed,
        "pocket": str(EXAMPLE_PDB),
        "box_center": pocket.center.tolist(),
        "box_size_A": pocket.box_size,
    }

    if pairs:
        v = np.array([p[0] for p in pairs], dtype=float)
        q = np.array([p[1] for p in pairs], dtype=float)
        diff = v - q  # vina - quickvina
        # Pearson r
        if v.std() > 0 and q.std() > 0:
            pearson_r = float(np.corrcoef(v, q)[0, 1])
        else:
            pearson_r = float("nan")
        # Spearman rho via rank correlation (no scipy dep assumed)
        def _rank(x: np.ndarray) -> np.ndarray:
            order = np.argsort(x, kind="mergesort")
            ranks = np.empty_like(order, dtype=float)
            ranks[order] = np.arange(1, len(x) + 1, dtype=float)
            # average ties
            sorted_x = x[order]
            i = 0
            while i < len(x):
                j = i
                while j + 1 < len(x) and sorted_x[j + 1] == sorted_x[i]:
                    j += 1
                if j > i:
                    avg = (i + j + 2) / 2.0  # mean of (i+1..j+1)
                    ranks[order[i:j + 1]] = avg
                i = j + 1
            return ranks
        rv, rq = _rank(v), _rank(q)
        if rv.std() > 0 and rq.std() > 0:
            spearman_rho = float(np.corrcoef(rv, rq)[0, 1])
        else:
            spearman_rho = float("nan")
        # Within-engine summary stats (only molecules that succeeded in that engine)
        v_all = np.array([float(r["vina_kcal"]) for r in rows if r["vina_kcal"] != ""], dtype=float)
        q_all = np.array([float(r["quickvina_kcal"]) for r in rows if r["quickvina_kcal"] != ""], dtype=float)
        # Paired SE of the mean difference
        if diff.size > 1:
            paired_se = float(diff.std(ddof=1) / math.sqrt(diff.size))
        else:
            paired_se = float("nan")
        metrics.update({
            "pearson_r": pearson_r,
            "spearman_rho": spearman_rho,
            "vina_mean_kcal_mol": float(v_all.mean()),
            "vina_std_kcal_mol": float(v_all.std(ddof=1)) if v_all.size > 1 else float("nan"),
            "quickvina_mean_kcal_mol": float(q_all.mean()),
            "quickvina_std_kcal_mol": float(q_all.std(ddof=1)) if q_all.size > 1 else float("nan"),
            "mean_diff_vina_minus_quickvina": float(diff.mean()),
            "paired_se_diff": paired_se,
            "mean_abs_diff": float(np.mean(np.abs(diff))),
        })

    metrics_path.write_text(json.dumps(metrics, indent=2))
    print(f"\nWrote {raw_path} ({n_total} rows) and {metrics_path}", file=sys.stderr)
    print(json.dumps({k: v for k, v in metrics.items() if k != "failures"}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
