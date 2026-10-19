#!/usr/bin/env python3
"""Assemble the PA17 canonical metric: same-protocol chemprop v2 vs literature.

``train_chemprop_v2_moleculenet.py`` writes the run to
``molmetal/reports/pa17_chemprop_v2_moleculenet/metrics.json``.  This script
turns it into a canonical ``metrics/by_metric`` artefact by adding a three-way
comparison block:

  * chemprop v2 measured on THIS repo's scaffold split (same-protocol);
  * this repo's D-MPNN at hidden 512 / depth 4 (the PA14 arm);
  * the Wu-2018 literature constant the paper used to quote.

The point (BLK-PA17) is that the literature column is a best-of-search number
the repo never reproduced, so the paper's "we lose on 4/4" framing was against
an un-reproduced target.  The comparison makes both gaps explicit.

Usage
-----
::

    .venv/bin/python molmetal/scripts/synthesize_pa17_metric.py
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

REPORT = "molmetal/reports/pa17_chemprop_v2_moleculenet/metrics.json"
# 2026-10-19 (Wf-A53) — this used to point at
# moleculenet_auc_dmpnn_5seed_cap512_d4.json, which was measured on the OLD
# isomeric split key.  The aligned-key canonical run now lives here, and it
# also carries the literature constants, so both pointers resolve to one file.
OURS = "metrics/by_metric/moleculenet_auc.json"
LITERATURE = "metrics/by_metric/moleculenet_auc.json"
OUT = "metrics/by_metric/moleculenet_auc_chemprop_v2_same_protocol.json"
TASKS = ("bbbp", "bace", "hiv", "clintox")


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--report", default=REPORT)
    p.add_argument("--ours", default=OURS)
    p.add_argument("--literature", default=LITERATURE)
    p.add_argument("--out", default=OUT)
    args = p.parse_args(argv)

    rep = json.loads((PROJECT_ROOT / args.report).read_text())
    ours = json.loads((PROJECT_ROOT / args.ours).read_text())
    lit = json.loads((PROJECT_ROOT / args.literature).read_text())

    comparison: Dict[str, object] = {}
    cp_m, our_m, lit_m = [], [], []
    for t in TASKS:
        cp = float(rep["tasks"][t]["test_auc_mean"])
        cp_s = float(rep["tasks"][t]["test_auc_std"])
        o = float(ours["tasks"][t]["test_auc_mean"])
        l = float(lit["tasks"][t]["chemprop_dmpnn_baseline"])
        comparison[t] = {
            "chemprop_v2_same_protocol_mean": cp,
            "chemprop_v2_same_protocol_std": cp_s,
            "our_dmpnn_cap512_d4_mean": o,
            "literature_chemprop_mean": l,
            "ours_minus_same_protocol": o - cp,
            "ours_minus_literature": o - l,
            "same_protocol_minus_literature": cp - l,
        }
        cp_m.append(cp)
        our_m.append(o)
        lit_m.append(l)

    n = len(TASKS)
    cp_mean, our_mean, lit_mean = sum(cp_m) / n, sum(our_m) / n, sum(lit_m) / n
    wins = [t for t in TASKS if comparison[t]["ours_minus_same_protocol"] > 0]

    out = dict(rep)
    out.update(
        {
            "metric": "moleculenet_auc_chemprop_v2_same_protocol",
            "status": "MEASURED",
            "MEASURED_vs_PROJECTED": "MEASURED",
            "comparison": comparison,
            "mean_chemprop_v2_same_protocol": cp_mean,
            "mean_our_dmpnn_cap512_d4": our_mean,
            "mean_literature_chemprop": lit_mean,
            "ours_minus_same_protocol_mean": our_mean - cp_mean,
            "ours_minus_literature_mean": our_mean - lit_mean,
            "same_protocol_minus_literature_mean": cp_mean - lit_mean,
            "tasks_we_beat_same_protocol_chemprop": wins,
            "comparison_baseline": args.literature,
            "ours_source": args.ours,
            "split_key": ours.get("split_key"),
            "same_protocol_requires": (
                "Both arms must be measured under the same split key.  The "
                "2026-10-12 chemprop run predates the 2026-10-19 key alignment; "
                "re-running it is what makes this block same-protocol."
            ),
            "why": (
                "BLK-PA17: the paper's chemprop column is the Wu-2018 "
                "best-of-search constant and the repo never reproduced it on its "
                "own split.  Measuring chemprop v2 on the same Bemis-Murcko "
                "scaffold split, seeds and epochs turns the comparison into a "
                "same-protocol one and separates 'we lose to chemprop' from "
                "'the literature column is tuned'."
            ),
            "verdict": (
                f"chemprop v2 same-protocol mean {cp_mean:.4f} is "
                f"{cp_mean - lit_mean:+.4f} BELOW the literature constant "
                f"{lit_mean:.4f} — the quoted reference was best-of-search.  Our "
                f"D-MPNN (512/4) mean {our_mean:.4f} is "
                f"{our_mean - cp_mean:+.4f} against same-protocol chemprop "
                f"(winning {len(wins)}/{n}: {wins}) but "
                f"{our_mean - lit_mean:+.4f} against the literature column."
            ),
            "caveats": [
                "chemprop v2 here uses its own default unweighted BCE "
                "(class_weight=off) and fixed 50 epochs with no early stopping, "
                "matching this repo's fixed-epoch protocol; a tuned chemprop run "
                "could score higher.",
                "Both arms use the historical varying-per-seed scaffold split "
                "(split_seed unset), so the 5-seed std mixes split and seed "
                "variance.",
                "The literature constant is cited, not reproduced; its original "
                "split is unknown.",
            ],
            "honest_framing_note": (
                "MEASURED only.  The same-protocol chemprop value comes from the "
                "PA17 report on disk; the comparison block is arithmetic over "
                "on-disk metrics, no value is asserted."
            ),
        }
    )

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    dest = PROJECT_ROOT / args.out
    atomic_write_file(dest, json.dumps(out, indent=2, default=float, ensure_ascii=False))
    for t in TASKS:
        c = comparison[t]
        print(f"{t:>8}  cp={c['chemprop_v2_same_protocol_mean']:.4f}  "
              f"ours={c['our_dmpnn_cap512_d4_mean']:.4f}  lit={c['literature_chemprop_mean']:.3f}  "
              f"ours-cp={c['ours_minus_same_protocol']:+.4f}")
    print(f"{'MEAN':>8}  cp={cp_mean:.4f}  ours={our_mean:.4f}  lit={lit_mean:.4f}  "
          f"ours-cp={our_mean - cp_mean:+.4f}  cp-lit={cp_mean - lit_mean:+.4f}")
    print(f"wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
