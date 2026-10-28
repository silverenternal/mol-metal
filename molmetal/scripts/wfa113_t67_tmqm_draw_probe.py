# SPDX-License-Identifier: Apache-2.0
"""Wf-A113 T67 — the tmQM cohort was drawn from a metal-blocked stream.

THE DEFECT
----------
``load_tmqm_clouds`` reservoir-sampled the first ``_CAP = 5 * n_clouds``
candidates in file order.  ``iter_tmqm_blocks`` walks ``tmQM_X1`` then ``X2``
then ``X3``, and those shards are METAL-BLOCKED, so the "unbiased reservoir"
was a reservoir over a single-metal prefix.

``DEFAULT_TRAIN_CLOUDS`` is 256, and 13 of the 14 ``dataset: tmqm`` configs do
not declare ``n_train``.  Every one of them trained on 256 Re complexes.

WHAT THIS MEASURES
------------------
1. The metal composition of the filtered stream at increasing prefixes
   (exact, from the XYZ block headers joined to ``tmqm_parsed.csv``).
2. The old selection's composition, simulated exactly: reservoir of
   ``n_clouds`` over the first ``5 * n_clouds`` candidates with the loader's
   own ``random.Random(seed)``.
3. The new selection's composition, from the shipped ``_stratified_code_draw``.

Run:
    .venv/bin/python molmetal/scripts/wfa113_t67_tmqm_draw_probe.py \
        --out metrics/by_metric/wfa113_t67_tmqm_draw.json
"""

from __future__ import annotations

import argparse
import collections
import csv
import gzip
import json
import random
import re
import sys
from pathlib import Path
from typing import Dict, List

_REPO = Path(__file__).resolve().parents[2]
TMQM = _REPO / ".data" / "molmetal" / "tmQM"
TARGET = ("Pt", "Ru", "Ir", "Rh", "Os", "Re")


def _stream_metals() -> List[str]:
    """Metal symbol of every six-metal complex, in the loader's stream order."""
    meta: Dict[str, str] = {}
    with (TMQM / "tmqm_parsed.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            meta[row["csd_code"]] = row["metal"]
    order: List[str] = []
    for name in ("tmQM_X1.xyz.gz", "tmQM_X2.xyz.gz", "tmQM_X3.xyz.gz"):
        with gzip.open(TMQM / name, "rt", encoding="utf-8") as fh:
            while True:
                line = fh.readline()
                while line and not line.strip():
                    line = fh.readline()
                if not line:
                    break
                n = int(line)
                header = fh.readline()
                m = re.search(r"CSD_code\s*=\s*(\S+)", header)
                code = m.group(1) if m else None
                for _ in range(n):
                    fh.readline()
                metal = meta.get(code)
                if metal in TARGET:
                    order.append(metal)
    return order


def _old_reservoir_mix(order: List[str], n_clouds: int, seed: int) -> Dict[str, int]:
    """Exactly what the pre-T67 loader selected."""
    cap = max(int(n_clouds) * 5, int(n_clouds))
    rng = random.Random(int(seed))
    reservoir: List[str] = []
    n_seen = 0
    for metal in order[:cap]:
        n_seen += 1
        if len(reservoir) < int(n_clouds):
            reservoir.append(metal)
        else:
            j = rng.randrange(n_seen)
            if j < int(n_clouds):
                reservoir[j] = metal
    return dict(collections.Counter(reservoir).most_common())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-clouds", type=int, nargs="+", default=[64, 256, 400, 819, 1024, 2000])
    args = ap.parse_args()

    sys.path.insert(0, str(_REPO))
    from molmetal.data.tmqm_clouds import _stratified_code_draw

    order = _stream_metals()
    pool = collections.Counter(order)
    total = len(order)

    first_pt = next((k for k, m in enumerate(order) if m == "Pt"), -1)
    # Zero Pt for a cohort of n means 5*n <= first_pt (the old window is the
    # first 5*n candidates, 0-based index 5*n-1).
    largest_pt_free_n = (first_pt // 5) if first_pt >= 0 else None
    last_re_only_n = next(
        (k for k, m in enumerate(order) if m != "Re"), len(order)
    ) // 5

    prefix = {}
    cum: collections.Counter = collections.Counter()
    i = 0
    for n in (64, 128, 256, 400, 512, 1024, 2048, 4096, 8192, 16384, total):
        while i < min(n, total):
            cum[order[i]] += 1
            i += 1
        prefix[str(n)] = {
            "n": sum(cum.values()),
            "mix": dict(cum.most_common()),
            "pt_present": cum.get("Pt", 0) > 0,
        }

    rows = []
    for n in args.n_clouds:
        old = _old_reservoir_mix(order, n, seed=0)
        # The shipped draw, applied to the real pool (codes are the index into
        # `order`, which is all the draw needs).
        codes = [str(k) for k in range(total)]
        meta = {str(k): {"metal": order[k]} for k in range(total)}
        drawn, quota, pool_by_metal = _stratified_code_draw(meta, codes, n, 0)
        new = dict(collections.Counter(order[int(c)] for c in drawn).most_common())
        rows.append(
            {
                "n_clouds": n,
                "old_stream_window": max(5 * n, n),
                "old_mix": old,
                "old_distinct_metals": len(old),
                "old_pt": old.get("Pt", 0),
                "new_mix": new,
                "new_quota": quota,
                "new_distinct_metals": len(new),
                "new_pt": new.get("Pt", 0),
            }
        )

    out = {
        "metric": "wfa113_t67_tmqm_draw",
        "tag": "MEASURED",
        "date": "2026-10-28",
        "probe": "molmetal/scripts/wfa113_t67_tmqm_draw_probe.py",
        "corpus": str(TMQM.relative_to(_REPO)),
        "filtered_pool_size": total,
        "filtered_pool_mix": dict(pool.most_common()),
        "first_pt_candidate_index": first_pt,
        "largest_pt_free_n_clouds": largest_pt_free_n,
        "largest_re_only_n_clouds": last_re_only_n,
        "defect": {
            "code": "molmetal/data/tmqm_clouds.py::load_tmqm_clouds (reservoir over a metal-blocked prefix)",
            "mechanism": (
                "iter_tmqm_blocks walks tmQM_X1/X2/X3 in file order, and the "
                "shards are metal-blocked. The reservoir sampled the first "
                "_CAP = 5 * n_clouds candidates of that order, so a small "
                "cohort is a sample of one metal's block, not of the corpus."
            ),
            "fix": (
                "_stratified_code_draw allocates n_clouds across metals by "
                "pool share (largest-remainder rounding, floor 1 per metal, "
                "min_per_metal as a stronger floor) and draws each metal's "
                "subset with random.Random(f'{seed}|{metal}'). The loader "
                "then streams only the drawn codes and reports the metal mix "
                "of the returned cohort on every run."
            ),
        },
        "stream_prefix_composition": prefix,
        "draws": rows,
        "run_log_evidence": [
            "molmetal/reports/wf299_cfm_main/wave_a/train.log: "
            "'256 cloud(s) from 1280 candidate(s)' — DEFAULT_TRAIN_CLOUDS, "
            "window 1280, 100% Re by the prefix table",
            "molmetal/reports/cfm_round3_retrain/d1_smoke/smoke.log: "
            "'400 cloud(s) from 2000 candidate(s)' — Ru/Re/Os, zero Pt",
            "molmetal/reports/cfm_tmqm_all6_retrain/train.log: "
            "'26715 cloud(s) from 26715 candidate(s)' — the only cohort that "
            "was ever representative",
        ],
        "consequences": [
            f"Every tmQM run with n_train <= {largest_pt_free_n} trained on a "
            f"cohort with ZERO Pt (the old window is the first 5*n candidates "
            f"and the first Pt sits at index {first_pt}). n_train <= "
            f"{last_re_only_n} is 100% Re. 13 of the 14 dataset:tmqm configs "
            "inherit DEFAULT_TRAIN_CLOUDS=256, which lands inside both "
            "regions.",
            "Any A/B between a small-n arm and cfm_tmqm_all6 (26715) conflates "
            "cohort size with metal diversity.",
            "The M2 measurement 'swapping the metal's atomic number moved the "
            "velocity by 0.74%' was taken on a checkpoint that may never have "
            "seen more than one metal, so it cannot be read as 'the metal "
            "channel is weak'.",
        ],
        "honest_limits": [
            "The old mix is simulated, not re-loaded: the simulation reproduces "
            "the loader's own rng call sequence (randrange(n_seen) per "
            "candidate) over the filtered stream, and the build step skips 0 "
            "complexes on this corpus (measured: 'skipped 0 + 0'), so the "
            "simulation is exact for the metal counts.",
            "The new draw's mix is the quota applied to the real pool; the "
            "loader can still come up short if a drawn complex fails to build, "
            "and it prints a WARNING naming the metal and the shortfall when "
            "that happens.",
            "Every archived tmQM number was produced on a metal-biased cohort. "
            "None of them is comparable to a run on the fixed draw.",
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=2, ensure_ascii=False)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
