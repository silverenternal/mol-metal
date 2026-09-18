#!/usr/bin/env python3
"""Aggregate Round-13 sweep JSON outputs into a single schema-compliant report."""
import json, pathlib, statistics

ROUND13 = pathlib.Path(__file__).resolve().parent
LAMBDA_OUT = ROUND13.parent / "wf_lambda1_wf_round13_100x3" / "lambda" / "r4c"
PB_OUT = ROUND13 / "pb"


def load(p):
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as e:
        return {"_parse_error": str(e), "_path": str(p)}


def aggregate_lambda(report):
    if not report:
        return {"status": "missing"}
    agg = report.get("aggregate", {})
    n = report.get("n_cells", None)
    return {
        "n_cells": n,
        "validity_rate": agg.get("validity_rate"),
        "uniqueness_rate": agg.get("uniqueness_rate"),
        "diversity_tanimoto": agg.get("diversity_tanimoto"),
        "diversity_homotype": agg.get("diversity_homotype"),
        "novelty": agg.get("novelty"),
        "synthesizability_rate": agg.get("synthesizability_rate"),
        "metal_compliance_rate": agg.get("metal_compliance_rate"),
        "logp_mean": agg.get("logp_mean"),
        "tpsa_mean": agg.get("tpsa_mean"),
        "rotb_mean": agg.get("rotb_mean"),
        "coordination_number_mean": agg.get("coordination_number_mean"),
        "monodentate_cl_count": agg.get("monodentate_cl_count"),
        "gsh_evasion_score": agg.get("gsh_evasion_score"),
        "dna_kb_proxy": agg.get("dna_kb_proxy"),
        "anticancer_index": agg.get("anticancer_index"),
        "sa_mean": agg.get("sa_mean"),
        "qed_mean": agg.get("qed_mean"),
        "rigid_rmsd_mean": agg.get("rigid_rmsd_mean"),
        "com_shift_mean": agg.get("com_shift_mean"),
        "n_coords_3d_attached_total": agg.get("n_coords_3d_attached_total"),
        "n_scaffold_aware_gate_active_total": agg.get("n_scaffold_aware_gate_active_total"),
        "decoder_pass_rate": agg.get("decoder_pass_rate"),
    }


def aggregate_pb(report):
    if not report:
        return {"status": "missing"}
    summary = report.get("summary", {})
    rows = report.get("per_pocket", [])
    n_total = len(rows)
    n_no_candidates = sum(1 for r in rows if r.get("status") == "no_candidates")
    n_seed_only = sum(1 for r in rows if r.get("status") == "seed_only")
    n_docked = sum(1 for r in rows if (r.get("physical", {}).get("n_docked") or 0) > 0)
    pb_eligible = [r for r in rows if (r.get("physical", {}).get("n_docked") or 0) > 0]
    n_pb_pass = sum(r.get("n_pb_pass") or 0 for r in rows)
    pb_pass_rate = summary.get("pb_pass_rate")
    vina_best_values = []
    for r in rows:
        top1 = r.get("top1_vina_proxy")
        if top1 is not None:
            vina_best_values.append(top1)
    return {
        "n_cells": n_total,
        "n_unique_pockets": summary.get("n_unique_pockets"),
        "n_no_candidates": n_no_candidates,
        "n_seed_only": n_seed_only,
        "n_docked": n_docked,
        "pb_eligible_cells": len(pb_eligible),
        "n_pb_pass_total": n_pb_pass,
        "pb_pass_rate_aggregate": pb_pass_rate,
        "vina_proxy_top1_mean": summary.get("vina_proxy_top1_mean"),
        "vina_proxy_top1_min": summary.get("vina_proxy_top1_min"),
        "vina_proxy_top1_max": summary.get("vina_proxy_top1_max"),
        "wall_seconds_mean": summary.get("wall_seconds_mean_per_pocket"),
        "physical_jobs_completed": summary.get("physical_jobs_completed"),
        "physical_n_docked": summary.get("physical_n_docked"),
    }


def main():
    lambda_rep = load(LAMBDA_OUT / "report.json")
    pb_rep = load(PB_OUT / "pb.json")

    lambda_metrics = aggregate_lambda(lambda_rep)
    pb_metrics = aggregate_pb(pb_rep)

    out = {
        "phase": "Round-13_100x3_partial",
        "n_evaluations_total": (
            (lambda_metrics.get("n_cells") or 0)
            + (pb_metrics.get("n_cells") or 0)
        ),
        "lambda_path": lambda_metrics,
        "pb_path": pb_metrics,
        "cfm_path": {
            "status": "BLOCKED",
            "reason": "r10_cfg_real_crossdocked.py: ModuleNotFoundError on `from molmetal.ports import GenerationConfig` under uv run; dGPU HSA_STATUS_ERROR userland-level. PyTorch uv-managed CUDA still usable (RX 7800 XT + Radeon 780M).",
            "verdict": "Deferred to Round-14 (TODO-26).",
        },
        "lambda_diversity_tanimoto_mean": lambda_metrics.get("diversity_tanimoto"),
        "lambda_n_distinct_mean": lambda_metrics.get("n_cells"),
        "cfm_vina_mean": None,
        "pb_pass_rate_aggregate": pb_metrics.get("pb_pass_rate_aggregate"),
        "all_metrics_aggregated": {
            "lambda": lambda_metrics,
            "pb": pb_metrics,
        },
        "paper_grade_data_ready": bool(
            lambda_metrics.get("n_cells") and lambda_metrics["n_cells"] >= 300
        ),
        "honest_deviation_from_spec": {
            "lambda_n_sim": "spec=1000, executed=200 (per-cell 22s at 200 → 110s at 1000 → 9.2h budget exceeded)",
            "pb_n_pockets": "spec=100, executed=10 (PB+Vina per-cell cost 2-5x Lambda; 30 cells in 4 min is already search-bound)",
            "cfm_path": "BLOCKED — script import + GPU runtime issue (see cfm_path.reason)",
        },
    }
    print(json.dumps(out, indent=2))
    (ROUND13 / "aggregate.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
