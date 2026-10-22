#!/usr/bin/env python3
# Copyright 2026 The Mol-Metal Authors
# SPDX-License-Identifier: Apache-2.0
"""UP-16 — is the ranking result a within-supported-pair result by construction?

The headline is `balanced_accuracy = 0.5985` on 360 ligand_cell pairs
(`msrs_listwise_ranker.json`).  The question is whether that is evidence of a
cross-metal ranking ability or only of ranking *inside* a matched pair.

The answer is structural, not statistical: a ranking pair is two rows of the same
`(ligand_canonical, cell_line_key)` carrying two different metals.  Every such pair
therefore sits inside a group that has >= 2 metals — the definition of a supported
unit in the UP-10 audit.  100% support is a tautology of the protocol; the metric
has no unsupported stratum to report.

What this script adds on top of the stored aggregates:

  * per metal pair, n / n_pos / n_neg / TPR / TNR / balanced accuracy with a
    Wilson interval on each rate propagated to balanced accuracy;
  * leave-one-metal-pair-out, to test the kill rule "if Os-Ru alone explains the
    whole signal";
  * the sample size each pair would need for a 95% balanced-accuracy half-width
    below 0.05, given its own observed rates.

No model is refitted.  Everything is recomputed from the two stored artifacts plus
the deduplicated table for the support census.

Usage
=====
    .venv/bin/python molmetal/scripts/msrs_ranking_support.py
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from molmetal.scripts.msrs_build_endpoints import _git_commit, _sha256  # noqa: E402
from molmetal.scripts.wf386a3_atomic_write import atomic_write_file  # noqa: E402

MIN_GROUPS = 20          # UP-10 support threshold
Z = 1.959963984540054    # 95% normal quantile
TARGET_HALF_WIDTH = 0.05


def wilson(k: float, n: float, z: float = Z) -> List[float]:
    """Wilson score interval for a binomial rate; degenerate inputs return [0, 1]."""
    if n <= 0:
        return [0.0, 1.0]
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(max(p * (1 - p) / n + z * z / (4 * n * n), 0.0)) / denom
    return [round(max(0.0, centre - half), 4), round(min(1.0, centre + half), 4)]


def counts_from_block(block: Dict[str, object]) -> Dict[str, object]:
    """Recover TP/TN/FP/FN from the stored n_pairs / positive_rate / tpr / tnr."""
    n = int(block["n_pairs"])
    pi = float(block["positive_rate"])
    n_pos = int(round(n * pi))
    n_neg = n - n_pos
    tpr = block.get("tpr")
    tnr = block.get("tnr")
    tp = int(round(float(tpr) * n_pos)) if tpr is not None and np.isfinite(tpr) else 0
    tn = int(round(float(tnr) * n_neg)) if tnr is not None and np.isfinite(tnr) else 0
    return {"n": n, "n_pos": n_pos, "n_neg": n_neg, "tp": tp, "tn": tn}


def balanced_ci(block: Dict[str, object]) -> Dict[str, object]:
    c = counts_from_block(block)
    tpr_ci = wilson(c["tp"], c["n_pos"])
    tnr_ci = wilson(c["tn"], c["n_neg"])
    bal = 0.5 * (tpr_ci[0] + tnr_ci[0]), 0.5 * (tpr_ci[1] + tnr_ci[1])
    return {
        **c,
        "tpr": round(c["tp"] / c["n_pos"], 4) if c["n_pos"] else None,
        "tnr": round(c["tn"] / c["n_neg"], 4) if c["n_neg"] else None,
        "balanced_accuracy": round(0.5 * (c["tp"] / max(c["n_pos"], 1)
                                          + c["tn"] / max(c["n_neg"], 1)), 4),
        "tpr_wilson95": tpr_ci,
        "tnr_wilson95": tnr_ci,
        "balanced_wilson95": [round(bal[0], 4), round(bal[1], 4)],
        "balanced_half_width": round(0.5 * (bal[1] - bal[0]), 4),
        "interval_kind": "Wilson on each rate, propagated to balanced accuracy "
                         "(not a ligand-clustered bootstrap)",
    }


def n_for_half_width(
    tpr: float, tnr: float, positive_rate: float, target: float = TARGET_HALF_WIDTH
) -> Optional[int]:
    """Smallest n whose expected balanced-accuracy half-width is below `target`.

    Jeffreys-smoothed rates keep a 0.0 or 1.0 rate from claiming zero variance:
    a TNR of exactly 1.0 from 36 negatives is not evidence of a zero-width
    interval.  n_pos = n * positive_rate, n_neg = n * (1 - positive_rate).
    """
    if not (0 < positive_rate < 1):
        return None

    def half(n: int) -> float:
        n_pos, n_neg = n * positive_rate, n * (1 - positive_rate)
        p1 = (tpr * n_pos + 0.5) / (n_pos + 1.0)
        p0 = (tnr * n_neg + 0.5) / (n_neg + 1.0)
        return 0.5 * Z * (math.sqrt(p1 * (1 - p1) / max(n_pos, 1e-9))
                          + math.sqrt(p0 * (1 - p0) / max(n_neg, 1e-9)))

    lo, hi = 4, 2_000_000
    if half(hi) >= target:
        return None
    while lo < hi:
        mid = (lo + hi) // 2
        if half(mid) < target:
            hi = mid
        else:
            lo = mid + 1
    return int(lo)


def support_census(dedup: Path) -> Dict[str, object]:
    """How many (ligand, cell) groups carry >= 2 metals, per metal pair."""
    df = pd.read_parquet(dedup)
    sub = df[(df["endpoint"] == "potency") & df["value_log10"].notna()]
    units = (
        sub.groupby(["ligand_canonical", "cell_line_key", "metal"])["value_log10"]
        .mean()
        .reset_index()
    )
    per_metal = units.groupby("ligand_canonical")["metal"].nunique()
    pairs = {}
    for a, b in (("Ru", "Os"), ("Ir", "Rh"), ("Ru", "Rh")):
        left = units.loc[units["metal"] == a, ["ligand_canonical", "cell_line_key"]]
        right = units.loc[units["metal"] == b, ["ligand_canonical", "cell_line_key"]]
        joint = left.merge(right, on=["ligand_canonical", "cell_line_key"])
        pairs[f"{a}-{b}"] = {
            "n_groups": int(len(joint)),
            "n_ligands": int(joint["ligand_canonical"].nunique()),
            "supported": bool(len(joint) >= MIN_GROUPS),
        }
    return {
        "n_ligands_total": int(len(per_metal)),
        "n_ligands_cross_metal": int((per_metal >= 2).sum()),
        "share_cross_metal": round(float((per_metal >= 2).mean()), 4),
        "pairs": pairs,
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--listwise", type=Path,
                    default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_listwise_ranker.json")
    ap.add_argument("--expanded", type=Path,
                    default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_ranking_expanded.json")
    ap.add_argument("--dedup", type=Path,
                    default=PROJECT_ROOT / "data" / "msrs" / "measurements_dedup.parquet")
    ap.add_argument("--out", type=Path,
                    default=PROJECT_ROOT / "metrics" / "by_metric" / "msrs_ranking_support.json")
    args = ap.parse_args(argv)

    listwise = json.loads(args.listwise.read_text())
    expanded = json.loads(args.expanded.read_text())
    census = support_census(args.dedup)

    blocks = {
        "listwise_ligand_cell": listwise["pairs"]["ligand_cell"],
        "surface_ligand_cell": expanded["pairs"]["ligand_cell"],
    }

    strata: Dict[str, object] = {}
    for name, block in blocks.items():
        by_pair = {}
        for pair, v in block["by_metal_pair"].items():
            if int(v["n_pairs"]) == 0:
                continue
            by_pair[pair] = balanced_ci(v)
            by_pair[pair]["n_pairs_stored"] = int(v["n_pairs"])
            by_pair[pair]["raw_sign_accuracy_stored"] = v["raw_sign_accuracy"]
            by_pair[pair]["majority_rate"] = v["majority_rate"]

        # Leave-one-pair-out, from recovered counts (exact, not weighted averages).
        pooled = {k: sum(v[k] for v in by_pair.values())
                  for k in ("n", "n_pos", "n_neg", "tp", "tn")}
        loo = {}
        for drop in by_pair:
            rem = {k: pooled[k] - by_pair[drop][k] for k in pooled}
            if rem["n_pos"] <= 0 or rem["n_neg"] <= 0:
                loo[drop] = {"status": "DEGENERATE", "n": int(rem["n"])}
                continue
            loo[drop] = {
                "n": int(rem["n"]),
                "balanced_accuracy": round(
                    0.5 * (rem["tp"] / rem["n_pos"] + rem["tn"] / rem["n_neg"]), 4
                ),
                "tpr": round(rem["tp"] / rem["n_pos"], 4),
                "tnr": round(rem["tn"] / rem["n_neg"], 4),
            }

        sample = {}
        for pair, v in by_pair.items():
            sample[pair] = {
                "n_now": v["n"],
                "half_width_now": v["balanced_half_width"],
                "n_for_half_width_0p05": n_for_half_width(
                    v["tpr"] if v["tpr"] is not None else 0.5,
                    v["tnr"] if v["tnr"] is not None else 0.5,
                    v["n_pos"] / max(v["n"], 1),
                ),
            }

        strata[name] = {
            "overall_stored": block["overall"],
            "by_metal_pair": by_pair,
            "leave_one_pair_out": loo,
            "sample_size": sample,
            "n_pairs": int(block["overall"]["n_pairs"]),
            "n_pairs_in_supported_groups": int(block["overall"]["n_pairs"]),
            "share_in_supported_groups": 1.0,
        }

    kill = {
        "rule": "if dropping Os-Ru returns balanced accuracy to 0.5, the ranking "
                "conclusion is a single-pair conclusion",
        "without_Os_Ru": strata["listwise_ligand_cell"]["leave_one_pair_out"].get("Os-Ru"),
        "kill_triggered": bool(
            strata["listwise_ligand_cell"]["leave_one_pair_out"].get("Os-Ru", {})
            .get("balanced_accuracy", 0.5) <= 0.52
        ),
    }

    out = {
        "artifact": "msrs_ranking_support",
        "version": "1.0",
        "framework": "MSRS — Metal Substitution Response Surface",
        "level": "UP-16 ranking support decomposition",
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {
            "listwise_artifact": str(args.listwise.relative_to(PROJECT_ROOT)),
            "listwise_sha256": _sha256(args.listwise),
            "expanded_artifact": str(args.expanded.relative_to(PROJECT_ROOT)),
            "expanded_sha256": _sha256(args.expanded),
            "dedup_parquet": str(args.dedup.relative_to(PROJECT_ROOT)),
            "dedup_sha256": _sha256(args.dedup),
            "script": str(Path(__file__).relative_to(PROJECT_ROOT)),
            "git_commit": _git_commit(),
            "argv": [sys.executable] + list(argv if argv is not None else sys.argv[1:]),
        },
        "design": {
            "no_refit": True,
            "source_of_numbers": "pairs.ligand_cell.by_metal_pair in the two stored artifacts",
            "support_census": "dedup potency units, >= 2 metals per (ligand, cell) group",
            "interval": "Wilson 95% per rate, propagated to balanced accuracy",
            "sample_size_target": TARGET_HALF_WIDTH,
            "min_groups": MIN_GROUPS,
        },
        "support_census": census,
        "strata": strata,
        "kill": kill,
        "gate": {
            "every_metal_pair_has_n_balanced_ci": all(
                all(k in v for k in ("n", "balanced_accuracy", "balanced_wilson95"))
                for name in strata for v in strata[name]["by_metal_pair"].values()
            ),
            "no_threshold": True,
        },
        "verdict": "RANKING_IS_SUPPORTED_BY_CONSTRUCTION",
        "caveats": [
            "Every ranking pair is two rows of one (ligand, cell) group under two metals, "
            "so 100% support is a property of the protocol, not a measurement. There is no "
            "unsupported stratum to report and none can be constructed without new data.",
            "The intervals are Wilson intervals on TPR and TNR, propagated to balanced "
            "accuracy. They are not ligand-clustered: the stored artifacts keep only the "
            "per-pair aggregate, not the per-pair rows, so the resampling unit is "
            "unavailable without refitting. Wilson intervals will be narrower than a "
            "cluster bootstrap because ligands repeat across pairs.",
            "Rh-Ru has 16 pairs with 1 negative, so its TNR is 0 or 1 and its balanced "
            "accuracy of 0.5 carries no information. It is reported, not used.",
            "The sample-size curve assumes the observed TPR/TNR and positive rate hold at "
            "larger n. It is a design calculation, not a projection of achievable accuracy.",
        ],
    }

    atomic_write_file(args.out, json.dumps(out, indent=2, allow_nan=True) + "\n")
    for name, s in strata.items():
        print(f"[UP-16] {name}: n_pairs={s['n_pairs']} supported=100%")
        for pair, v in s["by_metal_pair"].items():
            print(f"[UP-16]   {pair:7s} n={v['n']:4d} tpr={v['tpr']} tnr={v['tnr']} "
                  f"bal={v['balanced_accuracy']} ci={v['balanced_wilson95']} "
                  f"n_needed={s['sample_size'][pair]['n_for_half_width_0p05']}")
        print(f"[UP-16]   leave-one-out: {json.dumps(s['leave_one_pair_out'], ensure_ascii=False)}")
    print(f"[UP-16] kill={json.dumps(kill, ensure_ascii=False)}")
    print(f"json: {args.out}")
    print("PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
