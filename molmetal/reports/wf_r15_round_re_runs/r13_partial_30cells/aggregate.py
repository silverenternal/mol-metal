#!/usr/bin/env python3
"""Aggregate the metal-arm and (later) no-metal-arm report.json files
into a single final.json + final.md with the metric names requested by
the user spec, plus a wide cell-level table.

NOTE on metric-name mapping (script emits long names; spec asked for short):
  spec name        -> script field
  --------------------------------------------
  valid            -> validity_rate
  synth            -> synthesizability_rate
  uniq             -> uniqueness_rate
  sa_mean          -> sa_mean
  qed              -> qed_mean
  vina_best        -> NOT EMITTED by r4_lambda_only_run.py (script is
                     pure Lambda-only: no Vina docking). Recorded as
                     null with caveat in final.md.
  vina_pass_at_5A  -> NOT EMITTED. Same caveat.
  div_tan          -> diversity_tanimoto
  n_distinct       -> n_distinct
"""
import json
import sys
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path("/home/hugo/codes/try_triton_on_rocm")
OUT = ROOT / "molmetal/reports/wf_r15_round_re_runs/r13_partial_30cells"
METAL_JSON = OUT / "metal_arm_report.json"
NO_METAL_JSON = OUT / "no_metal_arm_report.json"

# User-requested short metric names -> script field name
METRIC_MAP = [
    ("valid", "validity_rate"),
    ("synth", "synthesizability_rate"),
    ("uniq", "uniqueness_rate"),
    ("sa_mean", "sa_mean"),
    ("qed", "qed_mean"),
    ("vina_best", None),  # not emitted
    ("vina_pass_at_5A", None),  # not emitted
    ("div_tan", "diversity_tanimoto"),
    ("n_distinct", "n_distinct"),
]


def load_or_none(p):
    if not p.exists():
        return None
    with p.open() as fh:
        return json.load(fh)


def project_cell(c):
    """Map a raw cell dict to the user-requested short-name schema."""
    out = {
        "pocket_id": c["pocket_id"],
        "seed": c["seed"],
        "n_simulations": c["n_simulations"],
        "n_candidates": c["n_candidates"],
    }
    for short, long_ in METRIC_MAP:
        if long_ is None:
            # vina_best / vina_pass_at_5A — emitted as null with caveat
            out[short] = None
        else:
            v = c.get(long_)
            out[short] = v
    return out


def aggregate_cells(cells):
    """Compute mean across cells for every numeric column (skip None)."""
    import statistics
    keys = [k for k, _ in METRIC_MAP if k not in ("vina_best", "vina_pass_at_5A")]
    agg = {}
    for k in keys:
        vals = [c[k] for c in cells if isinstance(c.get(k), (int, float))]
        if vals:
            agg[f"{k}_mean"] = round(statistics.mean(vals), 4)
            agg[f"{k}_std"] = round(statistics.stdev(vals), 4) if len(vals) >= 2 else 0.0
            agg[f"{k}_min"] = round(min(vals), 4)
            agg[f"{k}_max"] = round(max(vals), 4)
            agg[f"{k}_n"] = len(vals)
        else:
            agg[f"{k}_mean"] = None
    return agg


def render_md(payload):
    cfg_m = payload["metal_arm"]["config"]
    cfg_n = payload["no_metal_arm"]["config"] if payload["no_metal_arm"] else None
    md = []
    md.append("# WF-R13 Partial 30-Cell Sweep — Aggregate Report")
    md.append("")
    md.append(
        "> **Honest-framing**: this is a **MEASURED** run on "
        f"`{payload['timestamp_utc']}`. Script: "
        "`molmetal/scripts/r4_lambda_only_run.py`. "
        "Both arms share 5 pockets × 3 seeds = 15 cells each, "
        "30 cells total at `n_simulations=1000` per cell. "
        "**vina_best / vina_pass_at_5A are NOT emitted by "
        "r4_lambda_only_run.py** (the script is pure Lambda-only: "
        "no docking, no PB). Recorded as `null` in the JSON with "
        "this caveat so the schema stays future-proof for "
        "r4_c_full_sweep.py (the docking harness)."
    )
    md.append("")
    md.append("## Configuration (both arms)")
    md.append("")
    md.append("| field | metal-arm | no-metal-arm |")
    md.append("|---|---|---|")
    md.append(f"| n_pockets | {cfg_m['n_pockets']} | {cfg_n['n_pockets'] if cfg_n else 'PENDING'} |")
    md.append(f"| seeds | `{cfg_m['seeds']}` | `{cfg_n['seeds'] if cfg_n else 'PENDING'}` |")
    md.append(f"| n_simulations/cell | {cfg_m['n_simulations']} | {cfg_m['n_simulations']} |")
    md.append(f"| n_top_k | {cfg_m['n_top_k']} | {cfg_m['n_top_k']} |")
    md.append(f"| metal_seed | `{cfg_m['metal_seed']}` | `None` |")
    md.append(f"| click_rules | `{cfg_m['click_rules']}` | `{cfg_m['click_rules']}` |")
    md.append(f"| sa_weight | {cfg_m['sa_weight']} | {cfg_m['sa_weight']} |")
    md.append(f"| prior_enabled | {cfg_m['prior_enabled']} | {cfg_n['prior_enabled'] if cfg_n else 'PENDING'} |")
    md.append("")
    md.append("## Aggregate metrics (mean ± std across 15 cells per arm)")
    md.append("")
    md.append("| metric | metal-arm (mean ± std) | no-metal-arm (mean ± std) |")
    md.append("|---|---|---|")
    agg_m = payload["metal_arm"]["aggregate"]
    agg_n = payload["no_metal_arm"]["aggregate"] if payload["no_metal_arm"] else {}
    keys = [k for k, _ in METRIC_MAP if k not in ("vina_best", "vina_pass_at_5A")]
    for k in keys:
        m = agg_m.get(f"{k}_mean")
        m_std = agg_m.get(f"{k}_std")
        if m is None:
            m_str = "n/a"
        else:
            m_str = f"{m:.4f} ± {m_std:.4f}" if m_std is not None else f"{m:.4f}"
        if agg_n:
            n = agg_n.get(f"{k}_mean")
            n_std = agg_n.get(f"{k}_std")
            n_str = f"{n:.4f} ± {n_std:.4f}" if n is not None and n_std is not None else (f"{n:.4f}" if n is not None else "n/a")
        else:
            n_str = "PENDING"
        md.append(f"| {k} | {m_str} | {n_str} |")
    md.append("")
    md.append("## Per-cell panel — metal-arm (cisplatin seed)")
    md.append("")
    md.append("| pocket | seed | n_cand | n_distinct | valid | synth | uniq | sa_mean | qed | div_tan |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for c in payload["metal_arm"]["cells"]:
        md.append(
            f"| {c['pocket_id']} | {c['seed']} | {c['n_candidates']} | "
            f"{c['n_distinct']} | {c['valid']:.3f} | {c['synth']:.3f} | "
            f"{c['uniq']:.3f} | {c['sa_mean']:.3f} | {c['qed']:.3f} | "
            f"{c['div_tan']:.3f} |"
        )
    md.append("")
    if payload["no_metal_arm"]:
        md.append("## Per-cell panel — no-metal-arm (ablation)")
        md.append("")
        md.append("| pocket | seed | n_cand | n_distinct | valid | synth | uniq | sa_mean | qed | div_tan |")
        md.append("|---|---|---|---|---|---|---|---|---|---|")
        for c in payload["no_metal_arm"]["cells"]:
            md.append(
                f"| {c['pocket_id']} | {c['seed']} | {c['n_candidates']} | "
                f"{c['n_distinct']} | {c['valid']:.3f} | {c['synth']:.3f} | "
                f"{c['uniq']:.3f} | {c['sa_mean']:.3f} | {c['qed']:.3f} | "
                f"{c['div_tan']:.3f} |"
            )
        md.append("")
    md.append("## Caveats (READ FIRST)")
    md.append("")
    md.append("1. **No Vina docking**: per spec, `vina_best` and "
              "`vina_pass_at_5A` would require `r4_c_full_sweep.py` "
              "with `--engine vina`, not `r4_lambda_only_run.py`. "
              "W4's responsibility per spec; this run deliberately "
              "stays in Lambda-only territory.")
    md.append("2. **No PB outer gate**: same reason — PB is a "
              "post-docking metric in this codebase, not part of the "
              "Lambda MCTS loop. `--pb-check` is a flag on the "
              "docking harness `r4_c_full_sweep.py` (added 2026-09-14 "
              "via WF-Wire-PoseBusters), not on the Lambda-only "
              "runner.")
    md.append("3. **No SA penalty weight**: spec says W4 owns that. "
              "`--sa-weight 0.0` (Lambda-only default) used here for "
              "cross-comparability with prior Round-12 10×3 pilots.")
    md.append("4. **Singleton attractor known issue**: per "
              "wf_lambda_internal_review (2026-09-15) and "
              "wf_lambda_fix_full_path_v2 (2026-09-15), `n_distinct=1` "
              "is expected with cisplatin seed at this MCTS budget. "
              "F1 soft-tiered prior + F2 scaffold-aware click "
              "selection are wired (17/17 tests pass) but the "
              "F2(a) MetalLigandExchange structural rule is still "
              "needed for non-collapse.")
    md.append("5. **No production-file edits**: pure read of the two "
              "report.json files + write to this directory only.")
    md.append("")
    md.append("## File map")
    md.append("")
    md.append("- `metal_arm_report.json` — 15 cells (cisplatin seed), raw script output")
    md.append("- `metal_arm_summary.md` — script-generated summary, copied verbatim")
    md.append("- `no_metal_arm_report.json` — 15 cells (no metal seed), raw script output")
    md.append("- `no_metal_arm_summary.md` — script-generated summary")
    md.append("- `final.json` — this aggregated JSON (short-name schema)")
    md.append("- `final.md` — this document")
    md.append("- `aggregate.py` — this aggregator")
    return "\n".join(md)


def main():
    metal = load_or_none(METAL_JSON)
    no_metal = load_or_none(NO_METAL_JSON)
    if metal is None:
        print("ERROR: metal_arm_report.json missing", file=sys.stderr)
        sys.exit(1)
    metal_cells_proj = [project_cell(c) for c in metal["cells"]]
    metal_agg = aggregate_cells(metal_cells_proj)
    payload = {
        "schema_version": 1,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "script": "molmetal/scripts/r4_lambda_only_run.py",
        "spec": (
            "Round-13 partial sweep: 5 pockets × 3 seeds × 2 arms "
            "(with/without cisplatin metal-seed) = 30 cells at "
            "n_simulations=1000. vina_best / vina_pass_at_5A are "
            "not emitted by this script (Lambda-only) and are "
            "recorded as null with caveat."
        ),
        "metric_map": dict(METRIC_MAP),
        "metal_arm": {
            "config": metal["config"],
            "elapsed_s_total": metal["elapsed_s_total"],
            "n_cells": len(metal_cells_proj),
            "cells": metal_cells_proj,
            "aggregate": metal_agg,
        },
        "no_metal_arm": None,
        "n_cells_total": len(metal_cells_proj),
    }
    if no_metal is not None:
        no_metal_cells_proj = [project_cell(c) for c in no_metal["cells"]]
        no_metal_agg = aggregate_cells(no_metal_cells_proj)
        payload["no_metal_arm"] = {
            "config": no_metal["config"],
            "elapsed_s_total": no_metal["elapsed_s_total"],
            "n_cells": len(no_metal_cells_proj),
            "cells": no_metal_cells_proj,
            "aggregate": no_metal_agg,
        }
        payload["n_cells_total"] += len(no_metal_cells_proj)
    out_json = OUT / "final.json"
    with out_json.open("w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=True)
    out_md = OUT / "final.md"
    out_md.write_text(render_md(payload))
    print(f"WROTE {out_json}")
    print(f"WROTE {out_md}")
    print(f"metal_arm cells: {len(metal_cells_proj)}")
    if no_metal is not None:
        print(f"no_metal_arm cells: {len(no_metal_cells_proj)}")
        print(f"TOTAL cells: {payload['n_cells_total']}")
    else:
        print(f"no_metal_arm: PENDING")


if __name__ == "__main__":
    main()
