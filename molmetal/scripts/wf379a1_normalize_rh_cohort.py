#!/usr/bin/env python3
"""Wf-379-A1 — normalize Rh cohort JSON for train_metal_lomo_wasserstein.py.

The Wf-139 P7 Rh cohort JSON (synthetic_rh_pos_neg.json) stores entries under
'actives' + 'inactives' arrays, but the trainer expects a unified
'per_compound' array. This script reads the Wf-139 JSON and writes a
side-car JSON with 'per_compound' added, so the trainer can consume it
without source JSON modification.

Reads:
  /home/hugo/codes/mol-metal/molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg.json
Writes:
  /home/hugo/codes/mol-metal/molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg_normalized.json

Usage:
  python3 molmetal/scripts/wf379a1_normalize_rh_cohort.py
"""
import json
import sys
from pathlib import Path

SRC = Path("/home/hugo/codes/mol-metal/molmetal/reports/wf139_p7_rh_active_cohort/synthetic_rh_pos_neg.json")
DST = SRC.parent / "synthetic_rh_pos_neg_normalized.json"


def main() -> int:
    if not SRC.exists():
        print(f"FATAL: source JSON not found at {SRC}", file=sys.stderr)
        return 1
    with SRC.open() as fh:
        d = json.load(fh)
    actives = d.get("actives", [])
    inactives = d.get("inactives", [])
    per_compound = list(actives) + list(inactives)
    out = dict(d)
    out["per_compound"] = per_compound
    out["_wf379a1_normalize_source"] = str(SRC)
    out["_wf379a1_normalize_n_per_compound"] = len(per_compound)
    out["_wf379a1_normalize_n_catherine_pass"] = sum(
        1 for e in per_compound if e.get("catherine_6of6_pass", False)
    )
    DST.write_text(json.dumps(out, indent=2))
    print(f"[OK] wrote {DST}")
    print(f"     per_compound={out['_wf379a1_normalize_n_per_compound']} "
          f"catherine_pass={out['_wf379a1_normalize_n_catherine_pass']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())