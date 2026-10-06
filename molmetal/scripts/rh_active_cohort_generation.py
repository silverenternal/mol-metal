"""Wf-A3: generate a POSITIVE-labelled synthetic Rh cohort.

Why this exists
---------------
The Wf-125 P1 cohort (``synthetic_rh_cohort.json``) is 500 Catherine-6/6-PASS
Rh compounds that ``train_metal_lomo_wasserstein.py`` appends with
``active=0`` for every row.  Appended to a LOMO fold that holds out Rh, those
500 all-inactive rows land *in the Rh test fold* (see the splitter fix in
:func:`train_metal_lomo_wasserstein.WassersteinLOMOSplitter`), so the model
trains on no synthetic data while the metric is computed on 82 % fabricated
rows.  Even after the routing fix the cohort is still single-class, so the
fold learns a flat "all Rh is inactive" prior for the metal it must score.

This module generates the complementary ACTIVE cohort: 500 additional Rh
compounds, disjoint from the Wf-125 P1 500, labelled ``Active=1``.

Labelling assumption (single inline note, per the A3 brief)
-----------------------------------------------------------
A synthetic compound has no measured IC50, so "active" is a *similarity
transfer*, not a measurement: each compound is ranked by its maximum Morgan
(Tanimoto) similarity to a real, measured active Rh compound, and the
top-N are labelled active.  The real actives are drawn from the 5-metal
multitask CSV and every compound that also appears in the LOMO CSV is
excluded, so the label anchor never touches the LOMO Rh test fold.  Each
emitted record carries its ``max_sim_to_real_active`` so a reader can see
how weak the evidence is (measured values are ~0.1-0.2 -- these are
*weak* labels and are reported as such).

The Wf-125 P1 inactive cohort is NOT modified; it is only read here, to
guarantee the two cohorts are disjoint.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "molmetal") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "molmetal"))

DEFAULT_INACTIVE_JSON = (
    PROJECT_ROOT
    / "molmetal"
    / "reports"
    / "wf125_p1_rh_oversample"
    / "synthetic_rh_cohort.json"
)
DEFAULT_OUT_JSON = (
    PROJECT_ROOT
    / "molmetal"
    / "reports"
    / "wf125_p1_rh_oversample"
    / "synthetic_rh_cohort_active.json"
)
DEFAULT_REAL_CSV = PROJECT_ROOT / ".data" / "molmetal" / "metalcytotox_5metal_multitask_v4.csv"
DEFAULT_LOMO_CSV = PROJECT_ROOT / "molmetal" / "data" / "metal_lomo.csv"


def _smiles_set_from_json(path: Path) -> set:
    """Compound SMILES already present in an existing cohort JSON."""
    with Path(path).open() as fh:
        payload = json.load(fh)
    return {
        str(e.get("smiles", "")).strip()
        for e in payload.get("per_compound", [])
        if str(e.get("smiles", "")).strip()
    }


def _real_active_anchor_smiles(
    real_csv: Path, lomo_csv: Path, metal: str = "Rh"
) -> Tuple[List[str], dict]:
    """Real measured active ``metal`` compounds, minus anything in the LOMO CSV.

    The LOMO exclusion is what keeps this from being a label leak: the LOMO
    Rh rows ARE the Rh fold's test fold, so an anchor drawn from them would
    hand the active cohort its labels from the very compounds it is scored
    against.
    """
    import pandas as pd

    df = pd.read_csv(real_csv, low_memory=False)
    col = "Metal" if "Metal" in df.columns else "metal"
    act = "activity" if "activity" in df.columns else "active"
    smi = "SMILES_Ligands" if "SMILES_Ligands" in df.columns else "smiles"

    sub = df[df[col].astype(str).str.strip() == metal]
    sub = sub[sub[act].fillna(0).astype(float) > 0.5]
    anchors = [s for s in sub[smi].astype(str).str.strip().tolist() if s]

    lomo = pd.read_csv(lomo_csv, low_memory=False)
    lomo_smis = {
        s
        for s in lomo[lomo["Metal"].astype(str).str.strip() == metal][
            "SMILES_Ligands"
        ]
        .astype(str)
        .str.strip()
        .tolist()
        if s
    }

    before = len(set(anchors))
    kept = sorted(set(anchors) - lomo_smis)
    stats = {
        "n_anchor_candidates": before,
        "n_anchor_after_lomo_exclusion": len(kept),
        "n_excluded_as_lomo_test_leak": before - len(kept),
    }
    return kept, stats


def _generate_disjoint_pool(
    exclude: set, needed: int, seed_salt: int = 0
) -> List[Tuple[str, str, int]]:
    """Run the Wf-125 P1 ``ClosureOversampler`` until ``needed`` fresh compounds.

    Returns ``[(smiles, seed_scaffold, depth), ...]`` with no overlap with
    ``exclude``.  The oversampler is deterministic given its inputs, so we
    escalate ``per_scaffold_cap``/``target_cohort_size`` rather than reseed
    to widen the pool.
    """
    from molmetal.molmetal_lam.reward.closure_oversample import ClosureOversampler

    for cap, target in (
        (200, max(1000, needed * 3 + seed_salt)),
        (400, max(2600, needed * 6)),
        (600, max(4000, needed * 9)),
    ):
        ov = ClosureOversampler(
            metal="Rh", max_depth=2, per_scaffold_cap=cap, target_cohort_size=target
        )
        res = ov.run(verbose=False)
        pool = [
            (c.smiles, getattr(c, "seed_scaffold", ""), int(getattr(c, "depth", 0)))
            for c in res.per_compound
            if c.smiles and c.smiles not in exclude
        ]
        # De-duplicate while preserving generator order.
        seen = set()
        uniq = []
        for item in pool:
            if item[0] in seen:
                continue
            seen.add(item[0])
            uniq.append(item)
        if len(uniq) >= needed:
            return uniq
    raise RuntimeError(
        f"ClosureOversampler exhausted at {len(uniq)} fresh compounds, "
        f"needed {needed}."
    )


def _rank_by_active_similarity(
    pool: Sequence[Tuple[str, str, int]], anchor_smiles: Sequence[str]
) -> List[dict]:
    """Rank the pool by max Morgan/Tanimoto similarity to a real active."""
    from rdkit import Chem, DataStructs
    from rdkit.Chem import AllChem

    def fp(s: str):
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)

    anchor_fps = []
    for s in anchor_smiles:
        f = fp(s)
        if f is not None:
            anchor_fps.append((s, f))
    if not anchor_fps:
        raise RuntimeError("no usable real-active anchor fingerprints")
    anchor_vecs = [f for _, f in anchor_fps]
    anchor_smis = [s for s, _ in anchor_fps]

    scored = []
    for smi, scaffold, depth in pool:
        f = fp(smi)
        if f is None:
            continue
        sims = DataStructs.BulkTanimotoSimilarity(f, anchor_vecs)
        j = int(max(range(len(sims)), key=lambda k: sims[k]))
        scored.append(
            {
                "smiles": smi,
                "seed_scaffold": scaffold,
                "depth": depth,
                "max_sim_to_real_active": float(sims[j]),
                "nearest_real_active_smiles": anchor_smis[j],
            }
        )
    scored.sort(key=lambda r: -r["max_sim_to_real_active"])
    return scored


def build_active_cohort(
    n_active: int,
    inactive_json: Path,
    real_csv: Path,
    lomo_csv: Path,
) -> Tuple[List[dict], dict]:
    """Return ``(per_compound, stats)`` for the active Rh cohort."""
    t0 = time.time()
    exclude = _smiles_set_from_json(inactive_json)
    anchors, anchor_stats = _real_active_anchor_smiles(real_csv, lomo_csv)
    print(
        f"[wf-a3] real active anchors: {anchor_stats['n_anchor_after_lomo_exclusion']} "
        f"(excluded {anchor_stats['n_excluded_as_lomo_test_leak']} that appear in the "
        "LOMO Rh test fold)",
        flush=True,
    )
    pool = _generate_disjoint_pool(exclude, n_active)
    print(f"[wf-a3] fresh disjoint pool: {len(pool)}", flush=True)
    ranked = _rank_by_active_similarity(pool, anchors)
    if len(ranked) < n_active:
        raise RuntimeError(f"only {len(ranked)} scored compounds for {n_active} needed")

    chosen = ranked[:n_active]
    sims = [r["max_sim_to_real_active"] for r in chosen]
    per_compound = []
    for i, r in enumerate(chosen):
        per_compound.append(
            {
                "compound_id": f"rh_syn_act_{i + 1:04d}",
                "smiles": r["smiles"],
                "is_active": 1,
                "active": 1,
                "metal": "Rh",
                "catherine_6of6_pass": True,
                "seed_scaffold": r["seed_scaffold"],
                "depth": r["depth"],
                "max_sim_to_real_active": r["max_sim_to_real_active"],
                "nearest_real_active_smiles": r["nearest_real_active_smiles"],
            }
        )
    stats = {
        "n_active": len(per_compound),
        "label_source": "similarity_transfer_to_real_active_rh",
        "label_is_measured": False,
        "label_caveat": (
            "Active=1 is a Tanimoto similarity transfer to a measured real "
            "Rh active, not a measured IC50. Median max-sim is ~0.15, i.e. "
            "weak evidence; treat as a class-balance prior, not ground truth."
        ),
        "anchor_stats": anchor_stats,
        "max_sim_to_real_active": {
            "min": min(sims),
            "median": sorted(sims)[len(sims) // 2],
            "max": max(sims),
            "mean": sum(sims) / len(sims),
        },
        "disjoint_from_inactive_cohort": True,
        "wall_time_s": time.time() - t0,
    }
    return per_compound, stats


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Wf-A3: synthetic Rh ACTIVE cohort")
    p.add_argument("--n-active", type=int, default=500)
    p.add_argument("--inactive-json", default=str(DEFAULT_INACTIVE_JSON))
    p.add_argument("--real-csv", default=str(DEFAULT_REAL_CSV))
    p.add_argument("--lomo-csv", default=str(DEFAULT_LOMO_CSV))
    p.add_argument("--out-json", default=str(DEFAULT_OUT_JSON))
    args = p.parse_args(argv)

    per_compound, stats = build_active_cohort(
        n_active=args.n_active,
        inactive_json=Path(args.inactive_json),
        real_csv=Path(args.real_csv),
        lomo_csv=Path(args.lomo_csv),
    )
    payload = {
        "workflow": "wf_a3",
        "metal": "Rh",
        "parent_inactive_cohort": str(args.inactive_json),
        "n_compounds_kept": len(per_compound),
        "stats": stats,
        "per_compound": per_compound,
    }
    out = Path(args.out_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".tmp")
    with tmp.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    tmp.replace(out)  # atomic (AGENTS.md §5.6)
    print(f"[wf-a3] wrote {out} n={len(per_compound)}", flush=True)
    print(
        f"[wf-a3] max_sim_to_real_active median="
        f"{stats['max_sim_to_real_active']['median']:.4f}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
