#!/usr/bin/env python3
"""Assemble the tmQM bond-perception reference metric from the bench reports.

The canonical ``metrics/by_metric/tmqm_bond_perception.json`` is a *synthesis* of
three runs of ``bench_tmqm_bond_perception.py``:

  molmetal/reports/tmqm_bond_perception/            (360-complex base table)
  molmetal/reports/tmqm_bond_perception_scoped/     (controlled 3-way, probe CFM)
  molmetal/reports/tmqm_bond_perception_fullcorpus/ (900-complex, production CFM)

The bench writes each run to its own report tree; the metric that Paper A/B cite
was assembled by hand. This script performs that assembly deterministically so
the numbers are reproducible from committed code. Prose blocks are carried as
constants (they are commentary, not measurements) and the numeric tables are
recomputed from the raw reports.

Usage
-----
::

    .venv/bin/python molmetal/scripts/synthesize_tmqm_bond_perception.py \
        --out-metric metrics/by_metric/tmqm_bond_perception.json
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

BASE_REPORT = "molmetal/reports/tmqm_bond_perception/tmqm_bond_perception.json"
SCOPED_REPORT = "molmetal/reports/tmqm_bond_perception_scoped/tmqm_bond_perception.json"
FULLCORPUS_REPORT = (
    "molmetal/reports/tmqm_bond_perception_fullcorpus/tmqm_bond_perception.json"
)

# Keys carried verbatim from the bench protocol block.
_PROTOCOL_KEYS = ("ground_truth", "bo_threshold", "bond_type", "metrics", "note")


def _table_from_results(results: Dict[str, object], ndigits: Optional[int]) -> Dict[str, object]:
    """Map a bench `results` block into the comparison-table shape."""
    out: Dict[str, object] = {}
    for method, r in results.items():
        row = {
            "success_rate": r["success_rate"],
            "f1_exist": r["f1_exist_of_successes"],
            "macro_f1_type": r["macro_f1_type_of_successes"],
            "exact_type_rate": r["exact_type_rate_of_successes"],
            "scope": r.get("scope"),
        }
        if ndigits is not None:
            row = {k: (round(v, ndigits) if isinstance(v, float) else v)
                   for k, v in row.items()}
        out[method] = row
    return out


def _load(rel: str) -> Dict[str, object]:
    return json.loads((PROJECT_ROOT / rel).read_text())


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Assemble tmqm_bond_perception.json from the bench reports."
    )
    p.add_argument("--base-report", default=BASE_REPORT)
    p.add_argument("--scoped-report", default=SCOPED_REPORT)
    p.add_argument("--fullcorpus-report", default=FULLCORPUS_REPORT)
    p.add_argument(
        "--out-metric",
        default="metrics/by_metric/tmqm_bond_perception.json",
    )
    args = p.parse_args(argv)

    base = _load(args.base_report)
    scoped = _load(args.scoped_report)
    full = _load(args.fullcorpus_report)

    protocol = {k: base["protocol"][k] for k in _PROTOCOL_KEYS if k in base["protocol"]}

    controlled_protocol = {
        "scope": "non-hydrogen pairs for EVERY method (--heavy-atom-scope)",
        "why_that_scope": (
            "The CFM is trained on hydrogen-free clouds, so scoring it over all "
            "atoms would measure it out of distribution while the baselines kept "
            "the hydrogens. Restricting all three to the same non-hydrogen pair "
            "set is what makes the rows comparable."
        ),
        "n_complexes": int(scoped["results"]["cfm"]["n_complexes"]),
        "sampling": (
            f"{scoped['n_samples_per_metal']} per metal, six target metals, seed 0"
        ),
        "bo_threshold": protocol.get("bo_threshold"),
        "aggregation": "per-complex metric then mean over complexes",
        "source": args.scoped_report,
    }

    report = {
        "metric": "tmqm_bond_perception",
        "description": (
            "Reference table for bond perception (3D geometry -> molecular graph) "
            "on tmQM metal complexes. Built because a 2026-10-04 literature sweep "
            "found NO published number for this task on tmQM or any "
            "transition-metal corpus."
        ),
        "unit": "ratio (0-1, higher better unless noted)",
        "scope": "tier2 / tmQM / six target metals (Pt,Ru,Ir,Rh,Os,Re)",
        "source": "molmetal/scripts/bench_tmqm_bond_perception.py",
        "synthesis_script": "molmetal/scripts/synthesize_tmqm_bond_perception.py",
        "protocol": protocol,
        "n_complexes": int(base["n_samples_per_metal"]) * len(base["metals"]),
        "n_per_metal": int(base["n_samples_per_metal"]),
        "results": base["results"],
        "comparison_status": (
            "NO PUBLISHED BASELINE EXISTS. Published bond-perception numbers are "
            "all on organic molecules (RDKit DetermineBonds 95.3% exact-type on a "
            "fixed GEOM-DRUGS cohort; OpenBabel 64.4%) and are NOT comparable: "
            "different dataset, different chemistry. RDKit DetermineBonds cannot "
            "process transition metals at all. The tmQM paper (Balcells & "
            "Skjelstad, JCIM 2020, doi 10.1021/acs.jcim.0c01041) benchmarks DFT "
            "level-of-theory only and defines no structure task, so it shares no "
            "metric with this."
        ),
        "key_finding": (
            "Connectivity is easy; bond ORDER is what fails on metals. Where each "
            "method produces a molecule at all, it recovers the bond connectivity "
            "essentially perfectly (RDKit-stripped F1(exist)=1.000, OpenBabel "
            "0.985) but assigns the wrong bond orders (F1(type) 0.358 and 0.539; "
            "exact-type 0.035 and 0.028). The organic SOTA pipeline additionally "
            "only produces anything for 31.9% of metal complexes."
        ),
        "honest_framing": {
            "measured": (
                "All four numbers per baseline are MEASURED on this host, 360 "
                "complexes, seed-0 sample, BO threshold 0.3."
            ),
            "scope_caveat": (
                "Each baseline is scored ONLY on the bond subset it can address, "
                "named in its row. rdkit_stripped is scored on non-metal bonds "
                "only (it removes the metal before running); openbabel is scored "
                "on all bonds. The two rows are therefore not directly comparable "
                "to each other."
            ),
            "protocol_choice": (
                "Ground truth bond EXISTENCE comes from tmQM's Wiberg BO >= 0.3; "
                "bond TYPE is the nearest of single(1.0)/double(2.0)/triple(3.0) "
                "by BO magnitude. A SMILES-derived ground truth would give exact "
                "orders but needs an atom-mapping step whose failures would be "
                "indistinguishable from perception failures. The threshold and the "
                "type-rounding are protocol choices, not facts about the chemistry."
            ),
            "sample": (
                "360 complexes is a seed-0 sample, not the full 33,429-complex "
                "target-metal pool. Per-metal breakdown is not reported here."
            ),
            "not_claimed": (
                "These are NOT SOTA comparisons \u2014 no SOTA exists on this "
                "corpus. xyz2mol_tm (the published metal-capable method) is NOT "
                "included: it is not installed and could not be fetched in this "
                "session."
            ),
        },
        "controlled_comparison_2026_10_04": {
            "why": (
                "The table previously held only classical perception, so nothing "
                "in the project could answer 'does the learned bond head earn its "
                "place?'. The question has a specific shape: OpenBabel already "
                "recovers bond EXISTENCE almost perfectly and its documented weak "
                "axis is bond TYPE (metal-donor orders). That is the only axis on "
                "which a learned head can justify itself."
            ),
            "protocol": controlled_protocol,
            "table": _table_from_results(scoped["results"], ndigits=4),
            "cfm_provenance": scoped.get("cfm", {}),
            "reading": (
                "MEASURED, the CFM does NOT beat OpenBabel on either axis at this "
                "training scale: existence F1 0.612 vs 0.977, type F1 0.280 vs "
                "0.505. The burden of proof is therefore on the learned head, and "
                "the number that would discharge it is the TYPE column at "
                "production scale. rdkit_stripped's F1(exist) of 1.000 is on the "
                "30.3% of complexes it can address at all \u2014 it is NOT "
                "comparable to the other two rows, which is why success_rate is "
                "printed beside every row."
            ),
            "caveats": [
                "The CFM here is a 4-epoch probe (n_train=256), NOT a production "
                "model; the production config is 20 epochs over ~946 clouds.",
                "The CFM's own training-time numbers use a POOLED aggregation "
                "(bond_type_macro_f1_pooled) while this table averages per "
                "complex. The two agree on a single complex (pinned by "
                "molmetal/tests/test_cfm_bond_type_metric.py) but are not "
                "interchangeable across a cohort.",
                "OpenBabel is a mature deterministic toolkit; the CFM is 4 epochs "
                "of gradient descent. The gap is expected and is not by itself "
                "evidence that the head is worthless \u2014 it is evidence that "
                "nothing yet shows it is worth anything.",
            ],
        },
        "corrections": [
            {
                "field": "results / scope",
                "old_value": (
                    "two classical baselines, each on its own subset, no learned "
                    "model"
                ),
                "new_value": (
                    "adds a controlled three-way comparison on one non-hydrogen "
                    "scope"
                ),
                "reason": (
                    "A reference table with no learned-model row could not support "
                    "any claim about the project's own bond head. Measured under "
                    "one scope, the head currently loses to OpenBabel on both "
                    "existence and type."
                ),
                "date": "2026-10-04",
            },
            {
                "field": "(new section) production_scale_comparison_2026_10_11",
                "old_value": (
                    "Paper B headline 'bond_type_f1 0.9194 vs OpenBabel 0.505' "
                    "(docs/PAPER_SCOPE.md)"
                ),
                "new_value": (
                    "same-protocol 900-complex table: CFM macro-F1(type) 0.4723 vs "
                    "OpenBabel 0.5051 (still behind); CFM wins F1(exist) 0.9961 vs "
                    "0.9771 and exact-type 0.0978 vs 0.0300"
                ),
                "reason": (
                    "the old pair differed in aggregation, model scale, and cohort "
                    "simultaneously; re-scored under one protocol"
                ),
                "date": "2026-10-11",
            },
        ],
        "production_scale_comparison_2026_10_11": {
            "date": "2026-10-11",
            "why": (
                "PAPER_SCOPE.md quoted 'bond_type_f1 0.9194 vs OpenBabel 0.505' as "
                "Paper B's headline. That pairs two numbers that differ in THREE "
                "variables at once. The same checkpoint was therefore re-scored "
                "under the bench's own protocol so the comparison is like-for-like."
            ),
            "variables_that_differed_in_the_old_pairing": {
                "aggregation": (
                    "trainer gate = POOLED over all scored pairs "
                    "(train_cfm_yuelbond_wasserstein.py:5010); bench = per-complex "
                    "F1 then mean (bench_tmqm_bond_perception.py:752). The "
                    "trainer's own comment at :4996 says they are 'not "
                    "interchangeable' and the field name carries the _pooled suffix "
                    "for exactly that reason."
                ),
                "model_scale": (
                    "trainer gate came from n_train=26715; the bench's CFM row on "
                    "disk came from a probe at n_train=256, epoch 3."
                ),
                "cohort": (
                    "trainer gate used its own 192-complex val split; bench samples "
                    "900 (150 x 6 metals)."
                ),
            },
            "production_checkpoint_rescored": {
                "cfm_ckpt": full.get("cfm", {}).get("path"),
                "cfm_provenance": {
                    k: full.get("cfm", {}).get(k)
                    for k in ("epoch", "best_epoch", "n_train", "n_val",
                              "decision", "decision_threshold")
                },
                "protocol": (
                    "identical to controlled_comparison_2026_10_04: 900 complexes, "
                    "150 per metal, non-hydrogen pair scope, BO threshold 0.3, "
                    "per-complex aggregation"
                ),
                "out_dir": str(Path(args.fullcorpus_report).parent),
            },
            "table_same_protocol_900_complexes": _table_from_results(
                full["results"], ndigits=None
            ),
            "verdict": {
                "the_0.9194_figure_is_withdrawn": (
                    "The same checkpoint scores macro_f1_type 0.4723 here and 0.9194 "
                    "under the trainer's pooled gate. Nearly a factor of two. 0.9194 "
                    "is a real number for its own protocol but must NOT be placed "
                    "beside OpenBabel's 0.5051, which is per-complex averaged."
                ),
                "the_head_does_beat_OpenBabel_on_two_of_three_axes": (
                    "MEASURED at production scale, same 900 complexes: bond "
                    "EXISTENCE F1 0.9961 vs 0.9771, and exact-type rate 0.0978 vs "
                    "0.0300 (3.3x). Both were losses at probe scale."
                ),
                "the_head_still_loses_on_macro_f1_type": (
                    "0.4723 vs 0.5051, i.e. 0.0328 BEHIND. This remains the "
                    "honest-negative on the axis the bond head was supposed to win."
                ),
                "scale_effect_is_real_and_large": (
                    "Same protocol, n_train 256 -> 26715: F1(exist) 0.6118 -> "
                    "0.9961, macro-F1(type) 0.2801 -> 0.4723, exact-type 0.0011 -> "
                    "0.0978. The earlier 'does not beat OpenBabel at this training "
                    "scale' verdict was true at 256 and is now only partly true."
                ),
            },
            "for_paper_b": {
                "cite": (
                    "the 900-complex same-protocol table, including the "
                    "macro_f1_type loss"
                ),
                "do_not_cite": (
                    "0.9194 as a comparison against OpenBabel 0.5051; it is a "
                    "different aggregation on a different cohort"
                ),
            },
        },
    }

    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    out = PROJECT_ROOT / args.out_metric
    atomic_write_file(out, json.dumps(report, indent=2, default=float, ensure_ascii=False))

    t = report["production_scale_comparison_2026_10_11"]["table_same_protocol_900_complexes"]
    print(f"900-complex same-protocol table ({len(t)} methods):")
    for method, row in t.items():
        print(f"  {method:16} exist={row['f1_exist']:.4f} "
              f"type={row['macro_f1_type']:.4f} exact={row['exact_type_rate']:.4f} "
              f"scope={row['scope']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
