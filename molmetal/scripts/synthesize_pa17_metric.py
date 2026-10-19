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
import time
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

# Ground truth for the D-MPNN arm's splits: the per-seed files the canonical
# aggregate was built from.  The aggregate itself only gained a per-seed
# fingerprint on 2026-10-20, so older aggregates carry no split sizes.
OURS_RUN_DIR = "molmetal/reports/pa14_moleculenet_cap512_d4"


def _sizes_from_per_seed_files(
    run_dir: Path, task: str, backbone: str, seeds: List[int]
) -> List[tuple]:
    """``(n_train, n_test)`` per seed, read from the per-seed run files."""
    out = []
    for s in seeds:
        p = run_dir / f"{task}_{backbone}_seed{int(s)}.json"
        if not p.exists():
            return []
        d = json.loads(p.read_text())
        if "n_train" not in d or "n_test" not in d:
            return []
        out.append((int(d["n_train"]), int(d["n_test"])))
    return out


def _sizes_from_report(task_payload: Dict[str, object]) -> List[tuple]:
    """``(n_train, n_test)`` per seed, from a report's own payload.

    Falls back to the seed-0 scalars that pre-2026-10-20 reports carried, so a
    single-seed comparison is still possible — the caller records how many
    seeds were actually checked.
    """
    nt = task_payload.get("per_seed_n_test") or []
    ntr = task_payload.get("per_seed_n_train") or []
    if nt and len(nt) == len(ntr):
        return [(int(a), int(b)) for a, b in zip(ntr, nt)]
    if task_payload.get("n_test") is not None and task_payload.get("n_train") is not None:
        return [(int(task_payload["n_train"]), int(task_payload["n_test"]))]
    return []


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--report", default=REPORT)
    p.add_argument("--ours", default=OURS)
    p.add_argument("--literature", default=LITERATURE)
    p.add_argument("--out", default=OUT)
    p.add_argument("--ours-run-dir", default=OURS_RUN_DIR)
    p.add_argument(
        "--allow-unverified-split",
        action="store_true",
        help=(
            "Write the comparison even when the two arms' test-set sizes could "
            "not be compared.  Off by default: a 'same protocol' label that "
            "nobody verified is the defect this check exists to catch."
        ),
    )
    args = p.parse_args(argv)

    rep = json.loads((PROJECT_ROOT / args.report).read_text())
    ours = json.loads((PROJECT_ROOT / args.ours).read_text())
    lit = json.loads((PROJECT_ROOT / args.literature).read_text())

    # --- Same-protocol is a claim about the split, not a label. ---------------
    # The aligned and pre-2026-10-19 keys produce different test-set sizes
    # (MEASURED: bace seed0 299 vs 302, bbbp 412 vs 390, clintox 297 vs 298;
    # hiv unchanged at 8296 because its scaffolds carry no stereocentres).  So
    # the size vector is a discriminator, and comparing it is what turns
    # "same protocol" from an assertion into a check.
    run_dir = PROJECT_ROOT / args.ours_run_dir
    split_check: Dict[str, object] = {"per_task": {}, "seeds_compared": 0}
    problems: List[str] = []
    for t in TASKS:
        cp_sizes = _sizes_from_report(rep["tasks"][t])
        our_sizes = _sizes_from_report(ours["tasks"][t]) or _sizes_from_per_seed_files(
            run_dir, t, str(ours.get("backbone", "dmpnn")), list(ours.get("seeds") or [])
        )
        k = min(len(cp_sizes), len(our_sizes))
        split_check["per_task"][t] = {
            "chemprop_sizes": cp_sizes,
            "ours_sizes": our_sizes,
            "seeds_compared": k,
        }
        if k == 0:
            problems.append(f"{t}: no test-set sizes available on one or both arms")
        elif cp_sizes[:k] != our_sizes[:k]:
            problems.append(
                f"{t}: chemprop sizes {cp_sizes[:k]} != ours {our_sizes[:k]}"
            )
        else:
            split_check["seeds_compared"] = max(split_check["seeds_compared"], k)

    if problems and not args.allow_unverified_split:
        print(
            "[synthesize] REFUSING to write a 'same protocol' block — the two "
            "arms are not demonstrably on the same split:",
            file=sys.stderr,
        )
        for msg in problems:
            print(f"  - {msg}", file=sys.stderr)
        print(
            "  Re-run the stale arm, or pass --allow-unverified-split to write "
            "the block with split_check.verified=false.",
            file=sys.stderr,
        )
        return 2
    split_check["verified"] = not problems
    if problems:
        print(f"[synthesize] WARN: split unverified ({'; '.join(problems)})", file=sys.stderr)
    else:
        print(
            f"[synthesize] split verified: {split_check['seeds_compared']} seed(s) "
            f"per task share test-set sizes across both arms"
        )

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
            "report_source": args.report,
            "report_date": rep.get("date"),
            "report_written_at": time.strftime(
                "%Y-%m-%dT%H:%M:%S",
                time.localtime((PROJECT_ROOT / args.report).stat().st_mtime),
            ),
            "split_key": ours.get("split_key"),
            "split_check": split_check,
            "same_protocol_requires": (
                "Both arms must be measured under the same split key AND the "
                "same per-seed splits.  split_check records the test-set sizes "
                "of both arms per task; verified=true means they match, which "
                "is evidence the claim holds.  The 2026-10-12 chemprop run was "
                "on the pre-alignment (isomeric) key and is preserved as "
                "moleculenet_auc_chemprop_v2_same_protocol_pre_alignment.json — "
                "it is not comparable row-by-row with this block."
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
                "on-disk metrics, no value is asserted.  The same-protocol claim "
                "is backed by split_check, not by the file names."
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
