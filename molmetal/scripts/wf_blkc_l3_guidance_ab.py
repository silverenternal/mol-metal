"""Write metrics/by_metric/cfm_l3_guidance_ab.json from the two probe arms.

Arms are produced by molmetal/scripts/wf_blkc_l3_guidance_probe.py with the SAME
checkpoint, SAME seed, SAME clouds, SAME ODE method and step count. The only
difference between arm A (guidance_strength 0.0) and arm B (> 0) is the guidance
term, so any difference in the geometry statistics is attributable to it.

This script only COMPARES two JSON files; it runs no model.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

# Metrics where a value moving TOWARD the real-tmqm number is an improvement.
# Everything here is a "lower is better for generated, real is the target" or
# "higher is better" comparison, spelled out per row rather than assumed.
DIRECTION = {
    "min_pair_distance": "up_toward_real",
    "p01_pair_distance": "up_toward_real",
    "median_pair_distance": "up_toward_real",
    "frac_pairs_hard_clash": "down_toward_real",
    "radius_of_gyration": "toward_real",
    "metal_ligand_min": "up_toward_real",
    "metal_ligand_median": "up_toward_real",
}


def _mean(agg: Dict[str, Any], key: str):
    d = agg.get(key)
    return None if not isinstance(d, dict) else d.get("mean")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-a", required=True, help="unguided JSON (strength 0)")
    ap.add_argument("--arm-b", required=True, help="guided JSON (strength > 0)")
    ap.add_argument("--out", required=True)
    cli = ap.parse_args()

    a = json.loads(Path(cli.arm_a).read_text())
    b = json.loads(Path(cli.arm_b).read_text())

    # The cohort must be identical except for the guidance strength, otherwise
    # the comparison is meaningless. `checkpoint` lives at TOP level, not in
    # `cohort` -- reading it from `cohort` compares None == None and passes
    # vacuously, which would let a cross-checkpoint A/B through unnoticed.
    diffs = []
    for k in ("n_clouds", "n_steps", "seed", "method", "noise_sigma"):
        if k not in a["cohort"] or k not in b["cohort"]:
            raise SystemExit(f"ARM COHORT MISSING KEY {k!r} -- refusing to compare")
        if a["cohort"][k] != b["cohort"][k]:
            diffs.append(f"cohort.{k}: {a['cohort'][k]!r} != {b['cohort'][k]!r}")
    if a.get("checkpoint") != b.get("checkpoint"):
        diffs.append(f"checkpoint: {a.get('checkpoint')!r} != {b.get('checkpoint')!r}")
    if a["cohort"].get("source") != b["cohort"].get("source"):
        diffs.append("cohort.source differs")
    if diffs:
        raise SystemExit("ARMS ARE NOT COMPARABLE: " + "; ".join(diffs))

    # The real-tmqm reference is a function of the cohort alone, so it must be
    # bit-identical across arms. If it is not, the cohort really did differ.
    ra, rb = a.get("real_tmqm_geometry"), b.get("real_tmqm_geometry")
    if ra is not None and rb is not None and ra != rb:
        raise SystemExit("real_tmqm_geometry differs between arms -- cohort mismatch")

    ga, gb = a["generated_geometry"], b["generated_geometry"]
    real = a.get("real_tmqm_geometry") or b.get("real_tmqm_geometry")

    rows = []
    for key, direction in DIRECTION.items():
        va, vb = _mean(ga, key), _mean(gb, key)
        vr = _mean(real, key) if real else None
        row = {
            "metric": key,
            "unguided": va,
            "guided": vb,
            "real_tmqm": vr,
            "direction_wanted": direction,
        }
        if None not in (va, vb):
            row["delta_guided_minus_unguided"] = vb - va
            if vr is not None:
                row["abs_err_unguided"] = abs(va - vr)
                row["abs_err_guided"] = abs(vb - vr)
                row["moved_toward_real"] = abs(vb - vr) < abs(va - vr)
        rows.append(row)

    for key in ("n_clouds_with_any_hard_clash", "total_hard_clash_pairs"):
        row = {
            "metric": key,
            "unguided": ga.get(key),
            "guided": gb.get(key),
            "real_tmqm": (real or {}).get(key),
            "direction_wanted": "down_toward_real",
        }
        if None not in (row["unguided"], row["guided"]) and real:
            row["delta_guided_minus_unguided"] = row["guided"] - row["unguided"]
            row["moved_toward_real"] = row["guided"] < row["unguided"]
        rows.append(row)

    report: Dict[str, Any] = {
        "metric": "cfm_l3_guidance_ab",
        "tag": "MEASURED",
        "date": "2026-10-12",
        "question": (
            "Does injecting a repulsive-core sampling-time guidance force during "
            "ODE integration improve the sampled geometry, WITHOUT retraining?"
        ),
        "design": {
            "controlled_variable": "the guidance term and nothing else",
            "held_fixed": [
                "checkpoint", "template cohort (tmQM val composition only)",
                "initial noise (seed 1234)", "ODE method midpoint", "n_steps 20",
            ],
            "guidance_strength_unguided": a["cohort"].get("guidance_strength"),
            "guidance_strength_guided": b["cohort"].get("guidance_strength"),
            "mode": "eval() + torch.no_grad(), forward ODE integration only -- NO TRAINING",
        },
        "arms_on_disk": {
            "unguided": str(Path(cli.arm_a)),
            "guided": str(Path(cli.arm_b)),
            "script": "molmetal/scripts/wf_blkc_l3_guidance_probe.py",
            "aggregator": "molmetal/scripts/wf_blkc_l3_guidance_ab.py",
        },
        "cohort_caveat": (
            "SMOKE SCALE: 24 clouds, 20 steps, one checkpoint, one seed, one "
            "guidance strength. Directionally informative; not a settled number."
        ),
        "comparison": rows,
    }

    tmp = Path(str(cli.out) + ".tmp")
    tmp.write_text(json.dumps(report, indent=1))
    os.replace(tmp, cli.out)  # atomic, per AGENTS.md 5.6

    for r in rows:
        u, g = r.get("unguided"), r.get("guided")
        if u is None or g is None:
            continue
        flag = ""
        if "moved_toward_real" in r:
            flag = "  toward_real=" + str(r["moved_toward_real"])
        print(f"{r['metric']:>28}: {u:>10.5f} -> {g:>10.5f}{flag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
