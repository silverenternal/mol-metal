# SPDX-License-Identifier: Apache-2.0
"""Census of every CFM run already on disk, and what each one already answers.

WHY THIS EXISTS
---------------
On 2026-10-10 a full-corpus run was found on disk — ``cfm_r9_fix_v2``,
``n_train=26715``, ``bond_type_macro_f1 = 0.9470`` — while the canonical
config was running ``n_train=256`` at ``bond_type_macro_f1 = 0.3770``.  The
26715 was being printed in the loader line ("26715/26715 requested complexes
have a Wiberg BO block") during runs that were in flight at the time.  Two
things followed from not reading it:

  * a probe that scaled 256 -> 4096 and concluded data volume was not the
    constraint, when the real comparison was 256 -> 26715; and
  * a proposed batch-size A/B for which the disk already held eight variants.

This script reads every ``per_epoch_metrics.csv`` under ``molmetal/reports``
and emits one row per run, plus the config each run actually used (from the
``config`` block its own metrics JSON recorded — never re-derived, never
retyped).  The point is that the NEXT change can be checked against what has
already been measured before another run is launched.

The `already_answers` column is the load-bearing one. It maps each run to the
question its existence closes, so a proposal can be dropped rather than run.

SCOPE
-----
Only runs that wrote ``per_epoch_metrics.csv``.  Runs killed before the first
write leave no row here — that is stated in `not_counted` rather than papered
over, because two runs from 2026-10-10 died that way (the epsilon-gate
verification pair) and their logs remain as the only record of what they
proved.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional

REPO = Path(__file__).resolve().parent.parent.parent
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

REPORTS = REPO / "molmetal" / "reports"

#: Configuration keys that plausibly move bond-type quality. Pulled from each
#: run's own recorded `config` dict, so a value here is what that run used.
CFG_KEYS = (
    "n_train", "n_val", "batch_size", "hidden_dim", "n_layers", "lr",
    "epochs", "bond_predictor", "use_tmqm_init", "bond_head_warmup_epochs",
    "early_stop_metric", "use_egnn_velocity", "class_weights",
    "bond_class_weight_power", "use_fm_objective",
)

#: Question -> the runs whose existence already answers it. Each string says
#: WHAT the run measured, so a proposal that is already covered is visible as
#: such rather than having to be rediscovered.
ALREADY_ANSWERS: Dict[str, List[str]] = {
    "does the bond head keep improving with more epochs at n_train=256?": [
        "cfm_bond_head_100ep_2026_10_09 (typeF1 0.6370 @100ep vs 0.3770 @30ep, same n_train=256)",
        "cfm_bond_head_20ep_2026_10_09 (typeF1 0.3236 @19ep)",
    ],
    "does training on the FULL tmQM corpus beat the 256-cloud cohort?": [
        "cfm_r9_fix_v2 (n_train=26715 -> typeF1 0.9470, exact_type 0.3849)",
        "cfm_r11_eval256 (n_train=26715 -> typeF1 0.9456, exact_type 0.4374)",
        "cfm_r17_eval512 (n_train=26715 -> typeF1 0.7795, exact_type 0.0652; NOTE worse than r9)",
    ],
    "does flow matching help or hurt, at fixed n_train=256?": [
        "cfm_fm_ab_2026_10_09 (fm=true: decode 0.5833, gv 0.0625)",
        "cfm_fm_control_2026_10_09 (fm=false: decode 0.2500, gv 0.0000)",
    ],
    "does the assemble/connectivity term help or hurt?": [
        "cfm_frag_assemble (typeF1 0.1756) vs cfm_frag_control (typeF1 0.2703)",
        "-> NULL on bond type; both collapse gv to ~0.04",
    ],
    "is bond_exact_type_rate ALWAYS zero (i.e. is `combined` always constant)?": [
        "NO. cfm_r9_fix_v2 = 0.3849, cfm_r11_eval256 = 0.4374, "
        "cfm_bond_head_100ep = 0.1302. It is zero in the n_train=256 runs "
        "(cfm_fm_*, cfm_blk_c5_repair, cfm_frag_*) but NOT in the full-corpus runs.",
        "So the constant-0 gate is a property of the SMALL cohort, not a "
        "universal defect. cfm.yaml declares no early_stop_metric, so it "
        "inherits `combined` and inherits this.",
    ],
    "does use_tmqm_init actually load pretrained weights end-to-end?": [
        "2026-10-10 live run: '[load_tmQM_pretrained] Loaded tmQM-pretrained "
        "encoder: 47/51 params transferred' inside a full CFM trainer run.",
        "ckpt.yaml (shipped) does NOT declare it; 9 ablation configs do.",
    ],
    "does the epsilon-constraint gate wire correctly?": [
        "2026-10-10 live run logged "
        "'criterion=bond_type_f1_eps value=0.0613 (gen_validity 0.9753 >= "
        "epsilon 0.8500; bond_type_macro_f1 0.0613 scored +0.0613)'.",
        "NOT answered: which epoch it ultimately selects (both runs killed at "
        "ep5/ep2 of 30). cfm_r9_fix_v2 is the closest existing evidence "
        "(best @ ep7 under `combined`, gate_value 0.2205).",
    ],
    "yuelbond vs dmpnn bond head (isolated)?": [
        "2026-10-10 head-only A/B, same val split: yuelbond typeF1 0.7653 / "
        "exact 0.0573 vs dmpnn typeF1 0.6593 / exact 0.0169 at n_train=256; "
        "same ordering at 1024. yuelbond wins on the type axis.",
        "NOT answered: inside joint CFM training.",
    ],
}


from molmetal.eval.bond_metric_schema import (
    bond_ap_definition,
    bond_ap_legend,
    stamp_counts,
)


def _f(row: Dict[str, str], key: str) -> Optional[float]:
    try:
        v = float(row.get(key, "") or "nan")
        return None if v != v else v
    except (TypeError, ValueError):
        return None


def _gv(row: Dict[str, str]) -> Optional[float]:
    n, v = _f(row, "gen_validity_n"), _f(row, "gen_validity_val")
    return v / n if n and v is not None else None


#: Wf-A99 — one glyph per definition so the table stays narrow.
_DEF_CODE = {
    "existence_ap_v2": "v2",
    "no_bond_ap_v1": "v1",
    "undeclared": "?",
}


def load_run(d: Path) -> Optional[Dict]:
    csv_path = d / "per_epoch_metrics.csv"
    if not csv_path.is_file():
        return None
    rows = list(csv_path.open(encoding="utf-8"))
    if len(rows) < 2:
        return None
    rs = list(csv.DictReader(rows))
    cfg: Dict = {}
    try:
        cfg = json.load((d / "per_epoch_metrics.json").open(encoding="utf-8")).get("config", {})
    except Exception:
        pass
    gvs = [g for g in (_gv(r) for r in rs) if g is not None]
    last = rs[-1]
    out = {
        "run": d.name,
        "n_epochs_run": len(rs),
        "config": {k: cfg.get(k) for k in CFG_KEYS if k in cfg},
        "bond_f1": _f(last, "bond_f1"),
        "bond_type_macro_f1": _f(last, "bond_type_macro_f1_pooled"),
        "bond_exact_type_rate": _f(last, "bond_exact_type_rate"),
        "bond_pr_auc": _f(last, "bond_pr_auc"),
        # Wf-A99 (2026-10-20) — runs written before Wf-A97 carry the
        # NO_BOND AP here, runs after it carry the existence AP, and
        # nothing in the old rows said which.  Stamp per row so the
        # census cannot present the two as one column.
        "bond_pr_auc_definition": bond_ap_definition(last),
        "decode_ratio": _f(last, "decode_ratio"),
        "gen_validity_max": max(gvs) if gvs else None,
        "gen_validity_end": gvs[-1] if gvs else None,
    }
    if "n_train" in cfg and cfg.get("batch_size"):
        out["optimizer_steps_per_epoch"] = -(-cfg["n_train"] // cfg["batch_size"])
    try:
        m = json.load((d / "per_epoch_metrics.json").open(encoding="utf-8"))
        out["gate_metric"] = m.get("gate_metric")
        out["gate_value"] = m.get("gate_value")
        if isinstance(m.get("best"), dict):
            out["best_epoch"] = m["best"].get("epoch")
    except Exception:
        pass
    return out


def main() -> int:
    runs = []
    for d in sorted(REPORTS.iterdir()):
        r = load_run(d)
        if r:
            runs.append(r)

    _defs = stamp_counts(runs)
    print(
        "bond_pr_auc definitions present: "
        + ", ".join(f"{d}={n}" for d, n in sorted(_defs.items()))
    )
    for d in sorted(_defs):
        print(f"  {_DEF_CODE[d]:>4} = {d}: {bond_ap_legend(d)}")
    print()

    hdr = (
        f"{'run':<40}{'ep':>4}{'n_train':>9}{'bs':>4}{'step/ep':>9}"
        f"{'hid':>5}{'lr':>8}{'typeF1':>8}{'exactT':>8}{'gv_max':>8}{'gv_end':>8}"
        f"{'prAUC':>8}{'def':>4}"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in sorted(runs, key=lambda x: -(x["bond_type_macro_f1"] or -1)):
        c = r["config"]
        def g(k, w, f="{}"):
            v = c.get(k)
            return (f.format(v) if v is not None else " " * (w - 3) + "n/a").rjust(w)
        def num(v, w):
            return f"{v:>{w}.4f}" if isinstance(v, float) else "n/a".rjust(w)

        print(
            f"{r['run']:<40}{r['n_epochs_run']:>4}{g('n_train', 9)}{g('batch_size', 4)}"
            f"{str(r.get('optimizer_steps_per_epoch', 'n/a')).rjust(9)}"
            f"{g('hidden_dim', 5)}{g('lr', 8)}"
            f"{num(r['bond_type_macro_f1'], 8)}{num(r['bond_exact_type_rate'], 8)}"
            f"{num(r['gen_validity_max'], 8)}{num(r['gen_validity_end'], 8)}"
            f"{num(r['bond_pr_auc'], 8)}{_DEF_CODE[r['bond_pr_auc_definition']]:>4}"
        )

    out = {
        "metric": "cfm_run_census",
        "purpose": (
            "Every CFM run on disk with its own recorded config and results, so "
            "a proposal can be checked against what is already measured before "
            "another run is launched."
        ),
        "source": "each run's own per_epoch_metrics.{csv,json}; nothing re-derived",
        "n_runs": len(runs),
        "bond_pr_auc_definitions": _defs,
        "bond_pr_auc_definition_legend": {
            d: bond_ap_legend(d) for d in sorted(_defs)
        },
        "runs": runs,
        "already_answers": ALREADY_ANSWERS,
        "not_counted": (
            "Runs killed before their first per_epoch_metrics.csv write leave no "
            "row. Two exist from 2026-10-10 (wf_cfm_eps_gate_2026_10_10 and "
            "wf_cfm_eps_gate_tmqminit_2026_10_10, stopped at ep5/ep2 of 30); "
            "their master.log holds the only record of the epsilon-gate note and "
            "the tmQM weight load."
        ),
        "headline_gap": {
            "best_on_disk_typeF1": "cfm_r9_fix_v2 0.9470 (n_train=26715)",
            "canonical_config_typeF1": "cfm_fm_control_2026_10_09 0.3770 (n_train=256)",
            "ratio": 2.51,
            "cause_candidate": (
                "n_train 26715 vs 256 = 104x more complexes; with batch 16 vs 64 "
                "that is ~417x vs 4 optimizer steps per epoch. NOT yet "
                "attributed to a single variable — r9 also differs in batch "
                "size, lr and epoch budget."
            ),
        },
    }
    dest = REPO / "metrics" / "by_metric" / "cfm_run_census.json"
    from molmetal.scripts.wf386a3_atomic_write import atomic_write_file

    atomic_write_file(str(dest), json.dumps(out, indent=2))
    print(f"\nwrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())