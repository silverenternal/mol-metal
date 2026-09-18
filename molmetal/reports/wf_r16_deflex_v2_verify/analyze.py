"""Analyze the 30-cell Deflex v2 verification sweep.

Inputs (after both arms complete):
- molmetal/reports/wf_lambda1_wf_r16_deflex_v2_verify/baseline/report.json
- molmetal/reports/wf_lambda1_wf_r16_deflex_v2_verify/deflex/report.json

Outputs:
- final.json  — per-cell lift + aggregate mean±std + VERDICT
- prints verdict to stdout
"""
import json
import math
import sys
from pathlib import Path

REPO = Path("/home/hugo/codes/try_triton_on_rocm/molmetal")
BASE = REPO / "reports" / "wf_lambda1_wf_r16_deflex_v2_verify" / "baseline" / "report.json"
DEFL = REPO / "reports" / "wf_lambda1_wf_r16_deflex_v2_verify" / "deflex" / "report.json"
OUT = REPO / "reports" / "wf_r16_deflex_v2_verify"


def load(path: Path) -> dict:
    if not path.exists():
        print(f"FATAL: {path} not found", file=sys.stderr)
        sys.exit(1)
    return json.loads(path.read_text())


def per_cell(rows: list, keys=("pocket_id", "seed")) -> dict:
    out = {}
    for r in rows:
        k = (r.get("pocket_id"), r.get("seed"))
        out[k] = r
    return out


METRICS = [
    "diversity_tanimoto",
    "reference_tanimoto",
    "validity",
    "synthesizability_rate",
]


def main() -> int:
    base = load(BASE)
    defl = load(DEFL)
    base_cells = per_cell(base["cells"])
    defl_cells = per_cell(defl["cells"])

    # Per-cell lift (Deflex − Lambda), NaN if both arms missing the cell.
    per_cell_lift = {}
    for k in base_cells:
        if k not in defl_cells:
            continue
        b = base_cells[k]
        d = defl_cells[k]
        cell = {"pocket_id": k[0], "seed": k[1]}
        for m in METRICS:
            bv = b.get(m)
            dv = d.get(m)
            if isinstance(bv, (int, float)) and isinstance(dv, (int, float)):
                cell[f"{m}_baseline"] = float(bv)
                cell[f"{m}_deflex"] = float(dv)
                cell[f"{m}_lift"] = float(dv) - float(bv)
            else:
                cell[f"{m}_baseline"] = bv
                cell[f"{m}_deflex"] = dv
                cell[f"{m}_lift"] = None
        # also n_distinct / n_candidates (collapse indicators)
        for aux in ("n_distinct", "n_candidates", "elapsed_s"):
            cell[f"{aux}_baseline"] = b.get(aux)
            cell[f"{aux}_deflex"] = d.get(aux)
        per_cell_lift[k] = cell

    # Aggregate mean±std per metric
    agg = {}
    n = len(per_cell_lift)
    for m in METRICS:
        lifts = [
            c[f"{m}_lift"] for c in per_cell_lift.values()
            if isinstance(c[f"{m}_lift"], (int, float))
        ]
        if lifts:
            mean = sum(lifts) / len(lifts)
            var = sum((x - mean) ** 2 for x in lifts) / max(1, len(lifts) - 1)
            std = math.sqrt(var) if len(lifts) > 1 else 0.0
        else:
            mean = std = 0.0
            n = 0
        agg[m] = {
            "n": len(lifts),
            "mean_lift": mean,
            "std_lift": std,
            "min_lift": min(lifts) if lifts else None,
            "max_lift": max(lifts) if lifts else None,
        }

    # Verdict on div_tanimoto gate ≥ +0.05
    div_tan_mean = agg["diversity_tanimoto"]["mean_lift"]
    if div_tan_mean >= 0.05:
        verdict = "PASS"
    elif div_tan_mean >= 0.0:
        verdict = "NEUTRAL"
    else:
        verdict = "REGRESSION"

    OUT.mkdir(parents=True, exist_ok=True)
    out_payload = {
        "n_cells_compared": len(per_cell_lift),
        "aggregate": agg,
        "verdict": verdict,
        "verdict_gate": "div_tanimoto mean lift >= +0.05",
        "per_cell_lift": list(per_cell_lift.values()),
    }
    (OUT / "final.json").write_text(json.dumps(out_payload, indent=2))
    print(json.dumps({"verdict": verdict, "n_cells": len(per_cell_lift), "agg": agg}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
