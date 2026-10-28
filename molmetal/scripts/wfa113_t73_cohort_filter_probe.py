# SPDX-License-Identifier: Apache-2.0
"""Wf-A113 T73 — what a coordination-number FILTER does to a "five-metal" cohort.

WHY THIS EXISTS
---------------
``--tmqm-coord-number`` defaults to a value that is applied as an EXACT filter
(``tmqm_clouds.py:1034`` compares ``row["coord_number"] == str(value)``), and
until T73 that default was ``4``.  Every one of the 14 ``dataset: tmqm`` configs
on disk overrides it to ``-1``, so no measured number changes -- but a new
config that forgot the key would silently keep only four-coordinate complexes.

This probe states what that cohort looks like, straight from the corpus, so the
footgun is a number rather than an adjective.

Run:
    .venv/bin/python molmetal/scripts/wfa113_t73_cohort_filter_probe.py \
        --out metrics/by_metric/wfa113_t73_cohort_filter.json
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

COHORT = ("Pt", "Ru", "Ir", "Rh", "Os", "Re")
DEFAULT_CSV = ".data/molmetal/tmQM/tmqm_parsed.csv"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    path = Path(args.csv)
    if not path.is_file():
        raise SystemExit(f"corpus not found: {path}")

    total: collections.Counter = collections.Counter()
    by_cn: dict[str, collections.Counter] = {c: collections.Counter() for c in COHORT}
    cn_hist: dict[str, collections.Counter] = {c: collections.Counter() for c in COHORT}
    with path.open(encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            sym = row.get("metal")
            if sym not in COHORT:
                continue
            total[sym] += 1
            try:
                cn = int(float(row["coord_number"]))
            except (KeyError, TypeError, ValueError):
                continue
            cn_hist[sym][cn] += 1
            by_cn[sym][cn] += 1

    def _mix(pred) -> dict:
        keep = {c: sum(v for k, v in by_cn[c].items() if pred(k)) for c in COHORT}
        n = sum(keep.values())
        return {
            "n": n,
            "per_metal": keep,
            "share": {c: round(keep[c] / n, 4) if n else 0.0 for c in COHORT},
        }

    out = {
        "probe": "wfa113_t73_cohort_filter",
        "corpus": str(path),
        "n_rows_cohort": sum(total.values()),
        "per_metal_total": dict(total),
        "cn4_filter": _mix(lambda k: k == 4),
        "no_filter": _mix(lambda k: True),
        "coord_number_hist": {c: dict(sorted(cn_hist[c].items())) for c in COHORT},
        "reading": (
            "A cohort built with the old --tmqm-coord-number=4 default keeps "
            "only four-coordinate complexes. By the corpus that is 81.8% of Pt "
            "against 0.3% of Ru, so a config that calls itself 'five-metal' "
            "would in practice be one metal. The default is now -1 (no filter); "
            "the key remains an exact-equality filter, not a target."
        ),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
        fh.write("\n")
    print(json.dumps({"cn4": out["cn4_filter"]["share"],
                      "none": out["no_filter"]["share"]}, indent=1))
    print("n cn4 =", out["cn4_filter"]["n"], " n all =", out["no_filter"]["n"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
