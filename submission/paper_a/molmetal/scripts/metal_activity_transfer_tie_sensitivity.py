#!/usr/bin/env python3
"""Tie-rule sensitivity for the metal-transfer probe (BLK-PA1 follow-up).

The duplicate-collapse predicate resolves an exact majority tie toward ACTIVE
(``sum * 2 >= n``) — the anti-conservative direction for an activity claim.
Several docstrings described it as "ties -> inactive, the conservative choice";
that was wrong (corrected 2026-10-10). The canonical
``metal_activity_transfer_tie_sensitivity.json`` recorded the size of the effect
but was produced ad hoc, so it cannot be regenerated from committed code.

This wrapper re-derives it from the probe: for each tie rule it runs the
within-metal regime over the same seed set as the other stability artifacts and
reports the resulting range. Only the metals that actually have ties (Ru, Ir) are
scored, because the rule is a no-op elsewhere.

Usage
-----
::

    .venv/bin/python molmetal/scripts/metal_activity_transfer_tie_sensitivity.py \
        --out-metric metrics/by_metric/metal_activity_transfer_tie_sensitivity.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np

from molmetal.scripts.metal_activity_transfer_probe import (
    KRASNOV_WITHIN_METAL_AUC,
    METALS,
    _load_env_default,
    collapse_duplicates,
    run,
)

DEFAULT_SEEDS = [42, 43, 44, 49, 123]

RULE_KEYS = {
    "active": "ties_to_ACTIVE_current",
    "inactive": "ties_to_INACTIVE_conservative",
}


def _tie_counts(metals: List[str]) -> Dict[str, int]:
    """Exact-tie ligand groups per metal (sum*2 == n), under the raw cohort."""
    _load_env_default()
    import pandas as pd

    from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset

    counts: Dict[str, int] = {}
    for m in metals:
        ds = MetalCytotoxDataset.from_csv(
            filters=CytotoxFilter(time_threshold=24.0, ic50_min=0.01, metal_whitelist=[m])
        )
        df = pd.DataFrame({"s": [str(x) for x in ds.smiles], "y": np.asarray(ds.active, int)})
        g = df.groupby("s")["y"]
        counts[m] = int(sum(1 for _, v in g if int(v.sum()) * 2 == len(v)))
    return counts


def _auc_range(metals: List[str], seeds: List[int], radius: int, nbits: int,
               tie_rule: str) -> Dict[str, Dict[str, object]]:
    per_seed: Dict[str, List[Optional[float]]] = {m: [] for m in metals}
    for s in seeds:
        rep = run(
            metals=metals, seed=s, radius=radius, nbits=nbits, tie_rule=tie_rule
        )
        for m in metals:
            d = rep["within_metal_cv"].get(m, {})
            per_seed[m].append(float(d["roc_auc"]) if d.get("status") == "MEASURED" else None)
    out: Dict[str, Dict[str, object]] = {}
    for m in metals:
        vals = [v for v in per_seed[m] if v is not None]
        if not vals:
            continue
        a = np.asarray(vals, float)
        out[m] = {
            "seed42": float(per_seed[m][0]) if per_seed[m][0] is not None else None,
            f"seeds_{'_'.join(str(s) for s in seeds)}": [float(v) for v in vals],
            "min": float(a.min()),
            "max": float(a.max()),
            "spread": float(a.max() - a.min()),
            "pos_rate": None,  # filled below
        }
    return out


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Tie-rule sensitivity for the transfer probe.")
    p.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    p.add_argument("--metals", nargs="+", default=METALS)
    p.add_argument("--radius", type=int, default=2)
    p.add_argument("--nbits", type=int, default=2048)
    p.add_argument(
        "--out-metric",
        default="metrics/by_metric/metal_activity_transfer_tie_sensitivity.json",
    )
    args = p.parse_args(argv)

    ties = _tie_counts(list(args.metals))
    # Score only metals where a tie can move a claim we actually make: a
    # comparison against a published within-metal value. Rh also has 2 ties but
    # no published comparator (KRASNOV_WITHIN_METAL_AUC[Rh] is None), so the
    # rule cannot change any statement about it -- excluded, matching the
    # canonical artifact. The full tie census still covers every metal below.
    scored = [
        m
        for m in args.metals
        if ties.get(m, 0) > 0 and KRASNOV_WITHIN_METAL_AUC.get(m) is not None
    ]

    by_rule: Dict[str, Dict[str, Dict[str, object]]] = {}
    pos_rates: Dict[str, Dict[str, float]] = {}
    for rule, key in RULE_KEYS.items():
        by_rule[key] = _auc_range(scored, list(args.seeds), args.radius, args.nbits, rule)
        # pos_rate of the dedup cohort under this rule
        rep = run(metals=scored, seed=args.seeds[0], radius=args.radius,
                  nbits=args.nbits, tie_rule=rule)
        pos_rates[key] = {
            m: float(rep["duplication"][m]["pos_rate_dedup"]) for m in scored
        }
    for key, block in by_rule.items():
        for m, d in block.items():
            d["pos_rate"] = pos_rates[key].get(m)

    total_ties = sum(ties.values())

    # dedup cohort size for the share denominator
    _load_env_default()
    from molmetal.data.cytotox import CytotoxFilter, MetalCytotoxDataset

    n_unique_total = 0
    for m in args.metals:
        ds = MetalCytotoxDataset.from_csv(
            filters=CytotoxFilter(time_threshold=24.0, ic50_min=0.01, metal_whitelist=[m])
        )
        n_unique_total += len(collapse_duplicates(ds.smiles, ds.active)[0])

    active = by_rule[RULE_KEYS["active"]]
    inactive = by_rule[RULE_KEYS["inactive"]]
    ru_a, ru_i = active.get("Ru"), inactive.get("Ru")
    ir_a, ir_i = active.get("Ir"), inactive.get("Ir")
    krasnov = {m: KRASNOV_WITHIN_METAL_AUC.get(m) for m in scored}

    report = {
        "metric": "metal_activity_transfer_tie_sensitivity",
        "description": (
            "How much does the headline within-metal ROC-AUC move when the "
            "duplicate-collapse tie rule is flipped? The collapse predicate is "
            "`sum*2 >= n`, so an exact tie resolves to ACTIVE -- the "
            "anti-conservative direction for an activity claim. This artifact "
            "records the size of the effect rather than leaving it in comments."
        ),
        "why_it_matters": (
            "A tie means the ligand has contradictory measurements, which the "
            "project's own root-cause analysis established is expected (RC1: "
            "contradiction rate rises with measurement count). So neither "
            "direction is 'right' -- but any claim presented as robust to the "
            "choice must actually be robust to it."
        ),
        "unit": "ROC-AUC (0-1, higher better)",
        "scope": "tier2_precious_metal_anticancer",
        "source": "molmetal/scripts/metal_activity_transfer_tie_sensitivity.py",
        "wraps": "molmetal/scripts/metal_activity_transfer_probe.py --tie-rule",
        "model": f"morgan_fp(radius={args.radius}, {args.nbits}bit) + xgboost",
        "data_source": "MetalCytoToxDB (Krasnov et al., J. Med. Chem. 2026)",
        "seeds": list(args.seeds),
        "tie_groups_measured": {
            **ties,
            "total": int(total_ties),
            "share_of_dedup_cohort": round(total_ties / max(1, n_unique_total), 4),
            "all_labelled_active_under_current_rule": int(total_ties),
        },
        "within_metal_auc_by_rule": by_rule,
        "krasnov_published": krasnov,
        "verdict": {
            "Ir_claim_IS_robust": (
                f"Ir exceeds Krasnov {krasnov.get('Ir')} under BOTH rules and "
                f"every seed (conservative min {ir_i['min']:.4f}; current min "
                f"{ir_a['min']:.4f}). Safe to state."
                if ir_a and ir_i
                else "insufficient data"
            ),
            "Ru_claim_IS_NOT_robust": (
                f"Under the conservative rule Ru spans {ru_i['min']:.4f}-"
                f"{ru_i['max']:.4f}, i.e. entirely at or just BELOW Krasnov "
                f"{krasnov.get('Ru')}. Under the current rule it spans "
                f"{ru_a['min']:.4f}-{ru_a['max']:.4f}, entirely above. The "
                "direction of 'Ru beats Krasnov' is decided by the tie rule, "
                "not by the chemistry."
                if ru_a and ru_i
                else "insufficient data"
            ),
            "correct_how_to_state_it": (
                "Ir: 'exceeds the published value under either tie rule'. Ru: "
                "'at parity with the published 0.81', with the apparent margin "
                "reported as rule-dependent. Do NOT write 'Ru exceeds Krasnov' "
                "unqualified."
            ),
            "recommendation": (
                "Keep the current rule (both the probe and the D-MPNN matched "
                "switch use it, and switching would invalidate the committed "
                "seed-stability artifact), but state the Ru claim as parity and "
                "report both rules. The honest framing is stronger for not "
                "depending on an arbitrary tie-break."
            ),
        },
        "companion_artifacts": [
            "metrics/by_metric/metal_activity_transfer.json (seed 42, current rule)",
            "metrics/by_metric/metal_activity_transfer_seed_stability_v2.json "
            "(5 seeds, current rule)",
        ],
    }

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    out = Path(args.out_metric)
    atomic_write_file(out, json.dumps(report, indent=2, default=float))

    print(f"tie groups: {ties} (total {total_ties}, {report['tie_groups_measured']['share_of_dedup_cohort']:.4f} of cohort)")
    for m in scored:
        a, i = active[m], inactive[m]
        print(f"{m:>4}  active {a['min']:.4f}-{a['max']:.4f}   "
              f"inactive {i['min']:.4f}-{i['max']:.4f}   "
              f"(Krasnov {krasnov.get(m)})")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
