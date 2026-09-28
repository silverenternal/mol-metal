"""r4_lib.report — Markdown summary serialiser.

The :func:`_render_summary_md` function takes the JSON ``payload``
emitted by :func:`.runner.run_sweep` and produces the human-readable
``summary.md`` that lives alongside ``report.json`` and ``seeds.json``
in the output directory.

Originally lived at lines 3977-4155 of ``r4_lambda_only_run.py``.
"""

from __future__ import annotations

from typing import Any, Dict, List


def _render_summary_md(payload: Dict[str, Any]) -> str:
    cfg = payload.get("config", {})
    agg = payload.get("aggregate", {})
    cells = payload.get("cells", [])
    lines: List[str] = []
    lines.append("# WF-Lambda-1 — Pure Lambda-Only Baseline")
    lines.append("")
    lines.append("> Honest-framing: this is a MEASURED run. PROJECTED numbers")
    lines.append("> from the spec are not invoked here — the spec is in")
    lines.append("> `molmetal/reports/ultracode_audit/wf_lambda1_spec.md`.")
    lines.append("")
    lines.append("## Configuration")
    lines.append("")
    lines.append(f"- n_pockets : `{cfg.get('n_pockets')}`")
    lines.append(f"- seeds     : `{cfg.get('seeds')}`")
    lines.append(f"- n_simulations per cell : `{cfg.get('n_simulations')}`")
    lines.append(f"- n_top_k    : `{cfg.get('n_top_k')}`")
    lines.append(f"- prior_enabled : `{cfg.get('prior_enabled')}`")
    lines.append(f"- metal_seed    : `{cfg.get('metal_seed')}`")
    lines.append("")
    lines.append("## Aggregate metrics (mean across cells)")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for key in (
        "validity_rate",
        "uniqueness_rate",
        "diversity_tanimoto",
        "diversity_homotype",
        "novelty",
        "synthesizability_rate",
        "metal_compliance_rate",
        "reference_tanimoto",
        "rigid_rmsd_mean",
        "com_shift_mean",
    ):
        lines.append(f"| {key} | {agg.get(key, 0.0):.4f} |")
    lines.append("")
    lines.append("## WF-P0-Metrics — 9 P0 anticancer / drug-likeness columns")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(f"| logp_mean | {agg.get('logp_mean', 0.0):.4f} |")
    lines.append(f"| tpsa_mean | {agg.get('tpsa_mean', 0.0):.4f} |")
    lines.append(f"| rotb_mean | {agg.get('rotb_mean', 0.0):.4f} |")
    cn = agg.get("coordination_number_mean", 0.0)
    lines.append(f"| coordination_number_mean | {cn:.4f} |")
    cl = agg.get("monodentate_cl_count", 0)
    lines.append(f"| monodentate_cl_count | {int(cl)} |")
    lines.append(f"| gsh_evasion_score | {agg.get('gsh_evasion_score', 0.0):.4f} |")
    lines.append(f"| dna_kb_proxy | {agg.get('dna_kb_proxy', 0.0):.4f} |")
    lines.append(f"| anticancer_index | {agg.get('anticancer_index', 0.0):.4f} |")
    ox_dist = payload.get("oxidation_state_distribution_total", {}) or {}
    if ox_dist:
        ox_str = ", ".join(
            f"{k}={v}" for k, v in sorted(ox_dist.items())
        )
    else:
        ox_str = "(none)"
    lines.append(f"| oxidation_state_distribution | {ox_str} |")
    lines.append("")
    lines.append("## WF-Phase3b-MetricsV2 — 8 tumor-relevant anticancer / ADMET columns")
    lines.append("")
    lines.append("Lit basis: Weininger 1990 / Patrick 2009 / Hou 2007 / Veith 2009 /")
    lines.append("Benigni-Richard 2005 / Hughes 2008 / Delaney 2004 ESOL / Obach 1999.")
    lines.append("All CPU-only; zero GPU load. Heuristic evaluators (NOT wet-lab calibrated).")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    for k in (
        "logp7_4_mean",
        "gi50_proxy_mean",
        "cell_permeability_logPapp_mean",
        "herg_cardio_risk_mean",
        "ames_mutagen_mean",
        "hepatotox_index_mean",
        "aqueous_solubility_logS_mean",
        "plasma_protein_binding_mean",
    ):
        lines.append(f"| {k} | {agg.get(k, 0.0):.4f} |")
    lines.append("")
    lines.append("## WF-Phase3D-PerResidue-Diversity + WF-Phase3G-MetalCoordProbe")
    lines.append("")
    lines.append("Lit basis: Bemis & Murcko 1996 / Jasial 2021 IntDiv / Peter 2019 SPF;")
    lines.append("Lippard & Berg 1995 / Reedijk 1987 / Miessler 2014 d-block geometries.")
    lines.append("")
    lines.append("| metric | value |")
    lines.append("|---|---|")
    lines.append(
        f"| diversity_subpocket | "
        f"{agg.get('diversity_subpocket', 0.0):.4f} |"
    )
    lines.append(
        f"| metal_coord_compliance | "
        f"{agg.get('metal_coord_compliance', 0.0):.4f} |"
    )
    # ---- WF-T25-Metallo-Proxies: 4 metallodrug proxies (opt-in) -------
    if bool(agg.get("metallodrug_proxies_enabled", False)):
        lines.append("")
        lines.append("## WF-T25-Metallo-Proxies — 4 metallodrug-specific proxies")
        lines.append("")
        lines.append("Lit basis: Shriver & Atkins Table 17.7 (spectrochemical series) +")
        lines.append("Appleton 1997 Coord. Chem. Rev. (trans-effect series) + Wang 2005 +")
        lines.append("Cohen 2007 (Pt-DNA crosslink). All values in [0, 1].")
        lines.append("PROXIES, not measured quantities — see `molmetal/reports/")
        lines.append("wf_t25_metallo_proxies/final.md` for the honest framing.")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---|")
        lines.append(
            f"| metallodrug_reduction_potential_mean | "
            f"{agg.get('metallodrug_reduction_potential_mean', 0.0):.4f} |"
        )
        lines.append(
            f"| metallodrug_trans_effect_mean | "
            f"{agg.get('metallodrug_trans_effect_mean', 0.0):.4f} |"
        )
        lines.append(
            f"| metallodrug_lfse_mean | "
            f"{agg.get('metallodrug_lfse_mean', 0.0):.4f} |"
        )
        lines.append(
            f"| metallodrug_pt_dna_crosslink_mean | "
            f"{agg.get('metallodrug_pt_dna_crosslink_mean', 0.0):.4f} |"
        )
    lines.append("")
    lines.append("> `diversity_tanimoto` is the SE(3) / atom-symbol-histogram")
    lines.append("> baseline (legacy `diversity_alpha`). `diversity_homotype`")
    lines.append("> is the Lambda-native metric from WF-Lambda-2 (typed-variable")
    lines.append("> cosine + β-reduction-depth + click-rule-fires Jaccard).")
    lines.append("")
    lines.append("## Per-cell results")
    lines.append("")
    lines.append("| pocket | seed | n_cand | n_distinct | valid | uniq | div_tan | div_hom | novel | syn | metal | ref_tan | rigid_rmsd | com_shift | logP | TPSA | RotB | coord | gsh | dna | ai | cl | warnings |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for c in cells:
        lines.append(
            "| {pid} | {seed} | {nc} | {nd} | {v:.3f} | {u:.3f} | "
            "{dt:.3f} | {dh:.3f} | {n:.3f} | {s:.3f} | {m:.3f} | {rt:.3f} | "
            "{rr:.3f} | {cs:.3f} | "
            "{lp:.2f} | {tp:.1f} | {rb:.2f} | {cn:.2f} | "
            "{gsh:.3f} | {dna:.3f} | {ai:.3f} | {cl} | {w} |".format(
                pid=c["pocket_id"],
                seed=c["seed"],
                nc=c["n_candidates"],
                nd=c["n_distinct"],
                v=c["validity_rate"],
                u=c["uniqueness_rate"],
                dt=c["diversity_tanimoto"],
                dh=c["diversity_homotype"],
                n=c["novelty"],
                s=c["synthesizability_rate"],
                m=c["metal_compliance_rate"],
                rt=c["reference_tanimoto"],
                rr=c.get("rigid_rmsd_mean", 0.0),
                cs=c.get("com_shift_mean", 0.0),
                lp=c["logp_mean"],
                tp=c["tpsa_mean"],
                rb=c["rotb_mean"],
                cn=c["coordination_number_mean"],
                gsh=c["gsh_evasion_score"],
                dna=c["dna_kb_proxy"],
                ai=c["anticancer_index"],
                cl=c["monodentate_cl_count"],
                w=len(c["warnings"]),
            )
        )
    lines.append("")
    lines.append(f"Total elapsed: `{payload.get('elapsed_s_total', 0.0):.2f} s`")
    lines.append("")
    lines.append("## Scorer (Lambda-only, NO docking / AdmetAI / PB)")
    lines.append("")
    lines.append("- `alpha_equivalence_uniqueness_score` — distinct beta-NF count.")
    lines.append("- `click_rule_match_bonus` — +1.0 if any of 5 click rules fires.")
    lines.append("- `metal_geometry_prior_bonus` — +1.0 if Pt=4 / Ru=Ir=6 coord.")
    lines.append("- `rdkit_validity_score` — 1.0 if RDKit can sanitize.")
    lines.append("- `synthesizability_via_lambda_paths` — 1.0 if beta-NF + RDKit.")
    lines.append("")
    return "\n".join(lines) + "\n"
