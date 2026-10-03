"""Template R22-P plan JSON generator (Wf-72 Phase 2 / R22-P1..P6).

CPU-only: emits per-R22-P plan JSON templates (phases + verification + MEASURED-vs-PROJECTED)
following `TODO/pending/r22_p1_p2_p3_p4_p5_p6_plan.md` structure. Each template is pure data
(JSON-serializable) and is consumed by `r22_plan_signals.json` (per-R22-P signals + estimates).

Per `party-twenty-spirit §3.5`: this script ships honest framing — MEASURED fields are populated
only when on-disk artifacts exist; PROJECTED fields are explicitly labeled as PROJECTED.

Usage:
    python molmetal/scripts/template_r22_plan.py \\
        --plan-id R22-P1 \\
        --output molmetal/reports/wf72_r22_plans/R22-P1_template.json

    # or generate all 6 R22-P templates at once:
    python molmetal/scripts/template_r22_plan.py --all \\
        --output-dir molmetal/reports/wf72_r22_plans/

Author: Wf-72 Phase 2 (ultracode single-pass, CPU-only docs)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


# --- Data structures ---------------------------------------------------------


@dataclass
class Phase:
    """A single phase in an R22-P plan (P1.1, P1.2, ...)."""

    id: str
    title: str
    wall_human: str  # e.g. "1 wk CPU" / "3-4 wk GPU-0" / "6 mo wet-lab"
    output: str
    verification: str
    sub_tasks: list[str] = field(default_factory=list)


@dataclass
class CrossSkillStrategy:
    """Cross-skill matrix for a single R22-P plan item."""

    sub_task: str
    primary_skill: str
    secondary_skill: str


@dataclass
class MeasuredVsProjected:
    """MEASURED-vs-PROJECTED targets per `party-twenty-spirit §3.5`."""

    measured: list[str] = field(default_factory=list)  # What is MEASURED on disk today
    projected_stretch: list[str] = field(default_factory=list)  # Stretch target (PROJ.)
    projected_realistic: list[str] = field(default_factory=list)  # Realistic target (PROJ.)


@dataclass
class R22PlanTemplate:
    """A complete R22-P plan template (R22-P1, R22-P2, ..., R22-P6)."""

    plan_id: str  # e.g. "R22-P1"
    lambda_gap: str  # e.g. "L13"
    title: str
    status: str  # "ACTIVE" | "RESERVED"
    priority: str  # "HIGH" | "MEDIUM" | "LOW"
    hardware: str
    wall_time: str
    depends_on: list[str] = field(default_factory=list)
    phases: list[Phase] = field(default_factory=list)
    cross_skill_strategy: list[CrossSkillStrategy] = field(default_factory=list)
    measured_vs_projected: MeasuredVsProjected = field(default_factory=MeasuredVsProjected)
    output_dir: str = ""
    tests_estimate: int = 0
    parent_refs: list[str] = field(default_factory=list)


# --- Plan content definitions -------------------------------------------------


PLAN_DEFINITIONS: dict[str, dict[str, Any]] = {
    "R22-P1": {
        "lambda_gap": "L13",
        "title": "Hierarchical multi-metal coordination geometry",
        "status": "ACTIVE",
        "priority": "HIGH",
        "hardware": "CPU-only (parser + BFS) + Blackwell sm_120 GPU-0 for 30-metal sweep",
        "wall_time": "3-4 weeks (1 wk CPU parser + 1 wk GPU-0 sweep + 1-2 wk analysis)",
        "depends_on": ["L1 (closure theorem verifier)", "L6 (closure d-block bench)", "L10 (3-way integration)"],
        "tests_estimate": 25,
        "output_dir": "molmetal/reports/wf75_r22_p1/",
        "phases": [
            Phase(
                id="P1.1",
                title="Chemistry-aware dative-bond parser",
                wall_human="1 wk CPU",
                output="molmetal/molmetal_lam/lam_chem/dative_bond_parser.py + tests/test_dative_bond_parser.py",
                verification="parser PASS on [M+]<-X SMARTS for Pt/Pd/Au/Ru/Ir",
                sub_tasks=["Dative-bond SMARTS grammar", "Valence accounting undercount fix", "PT/Pd/Au/Ru/Ir coverage"],
            ),
            Phase(
                id="P1.2",
                title="Hierarchical M-M closure BFS (depth >= 2)",
                wall_human="1 wk CPU",
                output="molmetal/molmetal_lam/lam_chem/hierarchical_bfs.py + tests/test_hierarchical_bfs.py",
                verification="reachability lift on 5 metal clusters >= +0.05",
                sub_tasks=["M-M bond graph depth>=2 BFS", "Pt-Pt / Pt-Ru / Ru-Ru / Pt-Au / Au-Au clusters"],
            ),
            Phase(
                id="P1.3",
                title="Multi-metal cluster centers (30-metal sweep)",
                wall_human="1 wk GPU-0",
                output="multimetal_cluster_30metals.json (30 metals x 1000 samples)",
                verification="per-metal reachability >= 0.85 on >= 25/30 metals",
                sub_tasks=["30-metal sweep", "per-metal reachability", "lift vs Wf-37 P2 baseline 0.5891"],
            ),
            Phase(
                id="P1.4",
                title="Trans-effect propagation capture",
                wall_human="2-3 days CPU",
                output="transeffect_propagation.json",
                verification="trans-effect ordering matches Reedijk 2008 experimental literature",
                sub_tasks=["Pt(II) trans-effect ordering Cl- > NH3 > H2O", "kinetic trans-effect capture"],
            ),
            Phase(
                id="P1.5",
                title="Wf-75 close-out + LAMBDA_ROADMAP §4.2 L13 row patch",
                wall_human="2 days CPU",
                output="TODO/completed/105_r22_p1_hierarchical_multimetal.md + LAMBDA_ROADMAP §4.2 L13 row",
                verification="INDEX §17.6 cross-ref update + party-twenty-spirit §3.5 honest framing",
                sub_tasks=["close-out file", "INDEX cross-ref", "LAMBDA_ROADMAP §4.2 patch"],
            ),
        ],
        "cross_skill_strategy": [
            CrossSkillStrategy("Chemistry-aware dative-bond parser", "mol-metal-lambda-calculus", "mol-metal-click-chemistry"),
            CrossSkillStrategy("Hierarchical M-M closure BFS", "mol-metal-lambda-calculus", "mol-metal-r18-math (M26 Score-Rep)"),
            CrossSkillStrategy("Multi-metal cluster centers", "mol-metal-r18-math (T1 SE3 SB)", "mol-metal-lambda-calculus"),
            CrossSkillStrategy("Trans-effect propagation", "mol-metal-click-chemistry", "mol-metal-lambda-calculus"),
        ],
        "measured_vs_projected": MeasuredVsProjected(
            measured=[
                "TODO/pending/r22_p1_p2_p3_p4_p5_p6_plan.md §2 (this template ship today)",
                "TODO/pending/r22_l13_l16_gap_analysis.md §1 (Wf-72 Phase 1 ship)",
                "TODO/completed/96_r20_p5_closure_catherine.md §3.2 (reachability 0.5744-0.5962 NOT MET baseline)",
                "TODO/completed/95_r20_p4_3way_100pocket.md (3-way + 100-pocket honest-negative)",
            ],
            projected_stretch=[
                "reachability >= 0.95 on >= 25/30 metals",
                "6/6 Catherine properties preserved PASS",
                "trans-effect ordering matches Reedijk 2008 within 5%",
            ],
            projected_realistic=[
                "reachability >= 0.85 on >= 25/30 metals",
                "lift +0.20 vs Wf-37 P2 baseline 0.5891",
                "multi-metal cluster centers coverage 5 clusters (Pt-Pt, Pt-Ru, Ru-Ru, Pt-Au, Au-Au)",
            ],
        ),
        "parent_refs": [
            "TODO/pending/r22_l13_l16_gap_analysis.md §1",
            "TODO/completed/96_r20_p5_closure_catherine.md §3.2",
            "TODO/completed/95_r20_p4_3way_100pocket.md",
            "TODO/03_plans/LAMBDA_ROADMAP.md §4.2 (L1/L6/L10 trajectory column)",
        ],
    },
    "R22-P2": {
        "lambda_gap": "L14",
        "title": "Graph Neural ODE for metallodrug kinetics",
        "status": "ACTIVE",
        "priority": "MEDIUM",
        "hardware": "Blackwell sm_120 GPU-1 (32 GB) — GNN-ODE retrain",
        "wall_time": "4 weeks GPU-1 (~10 days compute + 2 weeks analysis)",
        "depends_on": ["L5 (Hilbert-ASTSR)", "L7 (Multi-PH + PLL)"],
        "tests_estimate": 20,
        "output_dir": "molmetal/reports/wf76_r22_p2/",
        "phases": [
            Phase(
                id="P2.1",
                title="GNN-ODE backbone (GRU-ODE-Bayes Chen et al. 2018 NeurIPS)",
                wall_human="1 wk GPU-1",
                output="molmetal/molmetal_lam/lam_chem/gnn_ode.py + tests/test_gnn_ode.py",
                verification="backbone PASS on synthetic Pt(II)/Pd(II)/Ru(II) redox cycle",
                sub_tasks=["GRU-ODE-Bayes backbone", "dopri5 ODE solver", "synthetic redox cycle test"],
            ),
            Phase(
                id="P2.2",
                title="Ligand-exchange ODE (Cl-/H2O/GSH rate constants)",
                wall_human="1 wk GPU-1",
                output="gnn_ode_ligand_exchange.json (per-metal rate constants)",
                verification="t1/2 RMSE <= 0.5 vs Reedijk 2008 experimental",
                sub_tasks=["Cl-/H2O/GSH exchange rates", "literature kinetics data integration", "t1/2 RMSE measurement"],
            ),
            Phase(
                id="P2.3",
                title="Hydrolysis kinetics + redox cycling",
                wall_human="1 wk GPU-1",
                output="hydrolysis_t12.json + redox_cycling_potential.json",
                verification="multi-task loss: static R^2 + kinetic rate constant MSE + t1/2 classification",
                sub_tasks=["cisplatin aquation kinetics", "Pt(II)/Pt(IV) redox potential", "multi-task loss ablation"],
            ),
            Phase(
                id="P2.4",
                title="Wf-76 close-out + LAMBDA_ROADMAP §4.2 L14 row patch",
                wall_human="2 days CPU",
                output="TODO/completed/106_r22_p2_gnn_ode_kinetics.md + LAMBDA_ROADMAP §4.2 L14 row",
                verification="INDEX §17.6 cross-ref update + party-twenty-spirit §3.5 honest framing",
                sub_tasks=["close-out file", "INDEX cross-ref", "LAMBDA_ROADMAP §4.2 patch"],
            ),
        ],
        "cross_skill_strategy": [
            CrossSkillStrategy("GNN-ODE backbone (GRU-ODE-Bayes)", "mol-metal-r18-math (T1 SE3 SB)", "mol-metal-lambda-calculus"),
            CrossSkillStrategy("Ligand-exchange ODE (Cl-/H2O/GSH)", "mol-metal-click-chemistry", "mol-metal-r18-math (T7 SB-Bass)"),
            CrossSkillStrategy("Hydrolysis kinetics (cisplatin aquation)", "mol-metal-anticancer", "mol-metal-click-chemistry"),
            CrossSkillStrategy("Redox cycling ODE (Pt(II)<->Pt(IV))", "mol-metal-r18-math (T7 SB-Bass)", "mol-metal-anticancer"),
        ],
        "measured_vs_projected": MeasuredVsProjected(
            measured=[
                "TODO/pending/r22_p1_p2_p3_p4_p5_p6_plan.md §3 (this template ship today)",
                "TODO/completed/93_r20_p2_hilbert_astsr_gpu.md §1.1 (CPU smoke R^2 ~ -0.02 baseline)",
                "TODO/completed/100_wf71_p3_wf49_cross_round2.md (PlatinAI 4-cell A2780cis only R^2=+0.037)",
            ],
            projected_stretch=[
                "t1/2 RMSE <= 0.3 vs Reedijk 2008",
                "multi-task R^2 lift >= +0.10 vs static-only baseline",
                "ligand-exchange rate constants within 5% of experimental literature",
            ],
            projected_realistic=[
                "t1/2 RMSE <= 0.5 vs Reedijk 2008",
                "multi-task R^2 lift >= +0.05 vs static-only baseline",
                "redox cycling Pt(II)/Pt(IV) potential measured for 5 metals (Pt/Pd/Au/Ru/Ir)",
            ],
        ),
        "parent_refs": [
            "TODO/pending/r22_l13_l16_gap_analysis.md §2",
            "TODO/completed/93_r20_p2_hilbert_astsr_gpu.md §1.1",
            "TODO/completed/100_wf71_p3_wf49_cross_round2.md",
            "TODO/03_plans/LAMBDA_ROADMAP.md §4.2 (L5/L7 trajectory column)",
        ],
    },
    "R22-P3": {
        "lambda_gap": "L15",
        "title": "Multi-objective Pareto front over toxicity / efficacy / ADMET",
        "status": "ACTIVE",
        "priority": "MEDIUM",
        "hardware": "CPU-only (NSGA-III bench) + Blackwell sm_120 GPU-0 for 30-metal sweep",
        "wall_time": "3 weeks (1 wk CPU NSGA-III + 1 wk GPU-0 sweep + 1 wk analysis)",
        "depends_on": ["L3 (ScoreRep + VariancePUCT + MCTS)", "L8 (ASTSR <-> Hilbert <-> MCTS triangle)"],
        "tests_estimate": 20,
        "output_dir": "molmetal/reports/wf77_r22_p3/",
        "phases": [
            Phase(
                id="P3.1",
                title="m18/m19/m20 metric definition (Protein-Metal binding AUC / LOMO / Tumor-specific AUC)",
                wall_human="3 days CPU",
                output="TODO/02_metrics/METRICS_TARGETS.md §m18-m20 + tests/test_metric_definitions.py",
                verification="metric PASS on retrospective + scaffold split",
                sub_tasks=["m18 Protein-Metal binding AUC", "m19 LOMO 5-fold", "m20 Tumor-specific AUC"],
            ),
            Phase(
                id="P3.2",
                title="NSGA-III backbone + 3-objective Pareto",
                wall_human="1 wk CPU",
                output="molmetal/molmetal_lam/search_alg/nsga3_pareto.py + tests/test_nsga3_pareto.py",
                verification="NSGA-III PASS on synthetic 3-objective test (ZDT1 / DTLZ2)",
                sub_tasks=["NSGA-III reference-point backbone", "3-objective Pareto (m18 + toxicity + ADMET)", "synthetic ZDT1/DTLZ2 test"],
            ),
            Phase(
                id="P3.3",
                title="MCTS <-> NSGA-III bridge (VariancePUCT as offspring generator)",
                wall_human="2-3 days CPU",
                output="molmetal/molmetal_lam/search_alg/mcts_nsga3_bridge.py + tests/test_mcts_nsga3_bridge.py",
                verification="bridge PASS on 30-metal x 1000-sample pareto front",
                sub_tasks=["VariancePUCT as offspring generator", "MCTS-NSGA-III bridge", "30-metal pareto front"],
            ),
            Phase(
                id="P3.4",
                title="Hypervolume indicator (paper §4 main result)",
                wall_human="2-3 days GPU-0",
                output="hypervolume_indicator.json (3-objective hypervolume)",
                verification="hypervolume >= 0.6 (stretch) / >= 0.4 (realistic) on PlatinAI 4-cell",
                sub_tasks=["hypervolume metric", "PlatinAI 4-cell hypervolume", "per-cohort hypervolume breakdown"],
            ),
            Phase(
                id="P3.5",
                title="Wf-77 close-out + LAMBDA_ROADMAP §4.2 L15 row patch",
                wall_human="2 days CPU",
                output="TODO/completed/107_r22_p3_pareto_front.md + LAMBDA_ROADMAP §4.2 L15 row",
                verification="INDEX §17.6 cross-ref update + party-twenty-spirit §3.5 honest framing",
                sub_tasks=["close-out file", "INDEX cross-ref", "LAMBDA_ROADMAP §4.2 patch"],
            ),
        ],
        "cross_skill_strategy": [
            CrossSkillStrategy("NSGA-III backbone (Deb & Jain 2014)", "mol-metal-r18-math (T5 Multi-PH)", "mol-metal-lambda-calculus"),
            CrossSkillStrategy("3-objective Pareto (m18 + toxicity + ADMET)", "mol-metal-anticancer", "mol-metal-mechanism-mapping"),
            CrossSkillStrategy("Hypervolume indicator", "mol-metal-r18-math (T5 Multi-PH)", "mol-metal-anticancer"),
            CrossSkillStrategy("MCTS <-> NSGA-III bridge", "mol-metal-lambda-calculus (M26 Score-Rep)", "mol-metal-r18-math (T5)"),
        ],
        "measured_vs_projected": MeasuredVsProjected(
            measured=[
                "TODO/pending/r22_p1_p2_p3_p4_p5_p6_plan.md §4 (this template ship today)",
                "TODO/INDEX.md §17.1 (m18/m19/m20 UNDEFINED or WEAK_REAL_NEEDS_MORE_DATA)",
                "TODO/completed/85_mock_audit.md (8 critical + 23 major + 98 minor mocks)",
                "TODO/completed/95_r20_p4_3way_100pocket.md §0 (3-way on MetalCytoToxDB NOT executed)",
            ],
            projected_stretch=[
                "hypervolume >= 0.6 (3-objective)",
                "m18 Protein-Metal binding AUC >= 0.78 on PlatinAI 4-cell",
                "dominance coverage >= 80% across 30-metal sweep",
            ],
            projected_realistic=[
                "hypervolume >= 0.4 (3-objective)",
                "m18 Protein-Metal binding AUC >= 0.65 on PlatinAI 4-cell",
                "dominance coverage >= 60% across 30-metal sweep",
            ],
        ),
        "parent_refs": [
            "TODO/pending/r22_l13_l16_gap_analysis.md §3",
            "TODO/INDEX.md §17.1",
            "TODO/completed/85_mock_audit.md",
            "TODO/03_plans/LAMBDA_ROADMAP.md §4.2 (L3/L8 trajectory column)",
        ],
    },
    "R22-P4": {
        "lambda_gap": "L16",
        "title": "Wet-lab-in-the-loop Bayesian optimization",
        "status": "ACTIVE",
        "priority": "HIGH",
        "hardware": "wet-lab (ORNL/PKU/协和) + CPU-only BO",
        "wall_time": "6 months wet-lab + 2 weeks CPU BO",
        "depends_on": ["L13 (hierarchical multi-metal coord)", "L15 (Pareto front)"],
        "tests_estimate": 10,
        "output_dir": "molmetal/reports/wf78_r22_p4/",
        "phases": [
            Phase(
                id="P4.1",
                title="Wet-lab PI outreach (ORNL / PKU / 协和 LOI + NDA)",
                wall_human="1 mo wet-lab",
                output="LOI signed + NDA + 20 compounds synthesized",
                verification="LOI signed + 20/20 compounds QC-pass",
                sub_tasks=["ORNL/PKU/协和 outreach", "LOI + NDA", "20 compound synthesis"],
            ),
            Phase(
                id="P4.2",
                title="Cell culture + IC50 assay (MCF7 + A549 + A2780 + A2780cis + 5 NCI-60 cells)",
                wall_human="3 mo wet-lab",
                output="wetlab_ic50_q1.json (per-compound x per-cell IC50 +/- std, 9 cell lines)",
                verification="full panel IC50 measured",
                sub_tasks=["MCF7/A549/A2780/A2780cis", "+5 NCI-60 cells", "MTT/SRB/CellTiter-Glo assay"],
            ),
            Phase(
                id="P4.3",
                title="GP surrogate + q-EI batch BO (CPU)",
                wall_human="2 wk CPU",
                output="bayes_opt_convergence.json (top-20 candidate acquisition + 5-candidate batches per cycle)",
                verification="GP calibration + q-EI convergence >= 5 cycles",
                sub_tasks=["GP surrogate", "q-EI batch acquisition", "5-cycle convergence"],
            ),
            Phase(
                id="P4.4",
                title="In-silico vs wet-lab agreement",
                wall_human="1 mo CPU",
                output="silico_vs_wetlab_agreement.json (Pearson r on top-20 candidates)",
                verification="Pearson r >= 0.7 (stretch) / >= 0.5 (realistic)",
                sub_tasks=["D-MPNN vs wet-lab IC50", "Pearson r measurement", "per-cell breakdown"],
            ),
            Phase(
                id="P4.5",
                title="Wf-78 close-out + paper §5 final claim (gated by TODO-84)",
                wall_human="1 mo doc (after TODO-84 authorization)",
                output="TODO/completed/108_r22_p4_wetlab_bayes_opt.md + paper/sections/05_wetlab_validation.tex (gated)",
                verification="shipped after auth + Pearson r >= 0.7",
                sub_tasks=["close-out file", "paper §5 (gated)", "INDEX cross-ref"],
            ),
        ],
        "cross_skill_strategy": [
            CrossSkillStrategy("Gaussian-process surrogate", "mol-metal-r18-math (T7 SB-Bass)", "mol-metal-anticancer"),
            CrossSkillStrategy("q-EI batch acquisition", "mol-metal-r18-math (T5 Multi-PH)", "mol-metal-lambda-calculus"),
            CrossSkillStrategy("Wet-lab cycle (9 cell lines)", "mol-metal-anticancer", "mol-metal-wetlab-protocol"),
            CrossSkillStrategy("In-silico vs wet-lab agreement", "mol-metal-anticancer", "mol-metal-mechanism-mapping"),
        ],
        "measured_vs_projected": MeasuredVsProjected(
            measured=[
                "TODO/pending/r22_p1_p2_p3_p4_p5_p6_plan.md §5 (this template ship today)",
                "TODO/completed/97_r20_p6_wetlab_ic50_dryrun.md (Wf-58 P6.0+P6.1+P6.2 ship surface)",
                "TODO/pending/wetlab_ic50_analysis_pipeline.md (Wf-74 P3 IC50 analysis pipeline template)",
                "TODO/pending/97_r20_p6_wetlab_ic50.md §0 (wet-lab cooperation 6-mo deferred)",
            ],
            projected_stretch=[
                "Pearson r >= 0.7 between D-MPNN prediction and wet-lab IC50",
                "GP calibration RMSE <= 0.3 on top-20 candidates",
                "q-EI convergence >= 5 cycles with monotonic improvement",
            ],
            projected_realistic=[
                "Pearson r >= 0.5 between D-MPNN prediction and wet-lab IC50",
                "GP calibration RMSE <= 0.5 on top-20 candidates",
                "q-EI convergence >= 3 cycles with monotonic improvement",
            ],
        ),
        "parent_refs": [
            "TODO/pending/r22_l13_l16_gap_analysis.md §4",
            "TODO/pending/97_r20_p6_wetlab_ic50.md",
            "TODO/completed/97_r20_p6_wetlab_ic50_dryrun.md",
            "TODO/pending/wetlab_ic50_analysis_pipeline.md",
            "TODO/pending/84_no_paper_task_without_authorization.md (paper-gate)",
            "TODO/03_plans/LAMBDA_ROADMAP.md §4.2 (L13/L15 dependency)",
        ],
    },
    "R22-P5": {
        "lambda_gap": "L17 (reserved: ship-evidence gap)",
        "title": "[reserved] ship-evidence gap (close-out traceability + benchmark reproducibility)",
        "status": "RESERVED",
        "priority": "MEDIUM",
        "hardware": "CPU-only (audit + ledger + reproducibility bench)",
        "wall_time": "TBD by Wf-73 Phase 1",
        "depends_on": ["TBD by Wf-73"],
        "tests_estimate": 20,
        "output_dir": "molmetal/reports/wf73_p5_ship_evidence/",
        "phases": [
            Phase(
                id="P5.1",
                title="R20+R21 ledger verification (Wf-70 Phase 3)",
                wall_human="1 wk CPU",
                output="TODO/completed/r21_status_ledger_2026_09_26.md",
                verification="ledger PASS review",
                sub_tasks=["R21 ledger mirroring R20 format", "8 R21-P plan items status"],
            ),
            Phase(
                id="P5.2",
                title="Close-out traceability audit",
                wall_human="1 wk CPU",
                output="TODO/02_metrics/EVALUATORS_MIGRATED.md §audit-trail + tests/test_closeout_audit.py",
                verification="audit PASS on 50 R20 sub-tasks",
                sub_tasks=["50 R20 sub-task audit", "MEASURED-vs-PROJECTED traceability", "audit test scripts"],
            ),
            Phase(
                id="P5.3",
                title="Benchmark reproducibility bench",
                wall_human="1 wk CPU",
                output="molmetal/reports/bench_reproducibility/{per_metric_hash,per_report_sha256}.json",
                verification="hash chain PASS",
                sub_tasks=["per-metric hash", "per-report sha256", "hash chain verification"],
            ),
            Phase(
                id="P5.4",
                title="MEASURED-vs-PROJECTED audit log",
                wall_human="2-3 days CPU",
                output="TODO/02_metrics/MEASURED_VS_PROJECTED_AUDIT.md + tests/test_measured_vs_projected.py",
                verification="audit PASS on all R20+R21 close-out files",
                sub_tasks=["audit log structure", "close-out file audit", "audit test scripts"],
            ),
            Phase(
                id="P5.5",
                title="Wf-73 close-out",
                wall_human="2 days CPU",
                output="TODO/completed/109_r22_p5_ship_evidence_gap.md",
                verification="INDEX §17 cross-ref update",
                sub_tasks=["close-out file", "INDEX cross-ref"],
            ),
        ],
        "cross_skill_strategy": [
            CrossSkillStrategy("R20+R21 ledger verification", "party-twenty-spirit §3.5", "mol-metal-mechanism-mapping"),
            CrossSkillStrategy("Close-out traceability audit", "mol-metal-mechanism-mapping", "party-twenty-spirit §3.5"),
            CrossSkillStrategy("Benchmark reproducibility bench", "mol-metal-r18-math (T5 Multi-PH)", "party-twenty-spirit §3.5"),
            CrossSkillStrategy("MEASURED-vs-PROJECTED audit log", "party-twenty-spirit §3.5", "mol-metal-mechanism-mapping"),
        ],
        "measured_vs_projected": MeasuredVsProjected(
            measured=[
                "TODO/completed/r20_status_ledger_2026_09_26.md §0 (5/8 partial + 3/50 NOT MET carry)",
                "TODO/completed/100_r20_r21_consolidation.md (Wf-70 Phase 2 L1-L12 expansion)",
                ".claude/skills/party-twenty-spirit/SKILL.md §3.5 (0-tolerance honest framing)",
            ],
            projected_stretch=["TBD by Wf-73 Phase 1"],
            projected_realistic=["TBD by Wf-73 Phase 1"],
        ),
        "parent_refs": [
            "TODO/completed/r20_status_ledger_2026_09_26.md §0",
            "TODO/completed/100_r20_r21_consolidation.md",
            "TODO/02_metrics/EVALUATORS_MIGRATED.md",
            ".claude/skills/party-twenty-spirit/SKILL.md §3.5",
        ],
    },
    "R22-P6": {
        "lambda_gap": "L18 (reserved: cross-skill ecosystem)",
        "title": "[reserved] cross-skill ecosystem (L13-L16 skill matrix)",
        "status": "RESERVED",
        "priority": "MEDIUM",
        "hardware": "CPU-only (skill docs + test scripts)",
        "wall_time": "TBD by Wf-73 Phase 1",
        "depends_on": ["R22-P1", "R22-P2", "R22-P3", "R22-P4"],
        "tests_estimate": 22,
        "output_dir": "molmetal/reports/wf79_r22_p6_cross_skill/",
        "phases": [
            Phase(
                id="P6.1",
                title="mol-metal-hierarchical-multimetal SKILL.md",
                wall_human="2 wk CPU",
                output=".claude/skills/mol-metal-hierarchical-multimetal/SKILL.md + references.md + test_hierarchical_multimetal.py",
                verification="SKILL.md PASS review + 6 test scripts",
                sub_tasks=["SKILL.md >= 250 LOC", "references.md", "test scripts"],
            ),
            Phase(
                id="P6.2",
                title="mol-metal-gnn-ode-kinetics SKILL.md",
                wall_human="2 wk CPU",
                output=".claude/skills/mol-metal-gnn-ode-kinetics/SKILL.md + references.md + test_gnn_ode.py",
                verification="SKILL.md PASS review + 6 test scripts",
                sub_tasks=["SKILL.md >= 250 LOC", "references.md", "test scripts"],
            ),
            Phase(
                id="P6.3",
                title="mol-metal-pareto-front SKILL.md",
                wall_human="2 wk CPU",
                output=".claude/skills/mol-metal-pareto-front/SKILL.md + references.md + test_pareto_front.py",
                verification="SKILL.md PASS review + 6 test scripts",
                sub_tasks=["SKILL.md >= 250 LOC", "references.md", "test scripts"],
            ),
            Phase(
                id="P6.4",
                title="mol-metal-bayes-opt-wetlab SKILL.md",
                wall_human="1 wk CPU",
                output=".claude/skills/mol-metal-bayes-opt-wetlab/SKILL.md + references.md + test_bayes_opt_wetlab.py",
                verification="SKILL.md PASS review + 4 test scripts",
                sub_tasks=["SKILL.md >= 250 LOC", "references.md", "test scripts"],
            ),
            Phase(
                id="P6.5",
                title="Wf-79 close-out",
                wall_human="1 wk CPU",
                output="TODO/completed/110_r22_p6_cross_skill_ecosystem.md",
                verification="INDEX §17.6 cross-ref update",
                sub_tasks=["close-out file", "INDEX cross-ref"],
            ),
        ],
        "cross_skill_strategy": [
            CrossSkillStrategy("L13 SKILL.md", "mol-metal-lambda-calculus", "mol-metal-r18-math"),
            CrossSkillStrategy("L14 SKILL.md", "mol-metal-r18-math (T1 SE3 SB)", "mol-metal-lambda-calculus"),
            CrossSkillStrategy("L15 SKILL.md", "mol-metal-r18-math (T5 Multi-PH)", "mol-metal-anticancer"),
            CrossSkillStrategy("L16 SKILL.md", "mol-metal-anticancer", "mol-metal-r18-math (T7 SB-Bass)"),
        ],
        "measured_vs_projected": MeasuredVsProjected(
            measured=[
                "TODO/pending/99_r20_skill_ecosystem.md (8 R20 skills — L1-L12 anchor)",
                "TODO/completed/99_r20_skill_ecosystem.md (Wf-55 8 R20 skills close-out, partial SHIP)",
            ],
            projected_stretch=["TBD by Wf-73 Phase 1"],
            projected_realistic=["TBD by Wf-73 Phase 1"],
        ),
        "parent_refs": [
            "TODO/pending/99_r20_skill_ecosystem.md",
            "TODO/pending/r22_l13_l16_gap_analysis.md §6",
            "TODO/completed/99_r20_skill_ecosystem.md",
        ],
    },
}


# --- Builder ------------------------------------------------------------------


def build_r22_plan_template(plan_id: str) -> R22PlanTemplate:
    """Build a single R22-P plan template from PLAN_DEFINITIONS."""
    if plan_id not in PLAN_DEFINITIONS:
        raise ValueError(f"Unknown plan_id: {plan_id}. Valid: {sorted(PLAN_DEFINITIONS.keys())}")
    spec = PLAN_DEFINITIONS[plan_id]
    return R22PlanTemplate(
        plan_id=plan_id,
        lambda_gap=spec["lambda_gap"],
        title=spec["title"],
        status=spec["status"],
        priority=spec["priority"],
        hardware=spec["hardware"],
        wall_time=spec["wall_time"],
        depends_on=spec["depends_on"],
        phases=spec["phases"],
        cross_skill_strategy=spec["cross_skill_strategy"],
        measured_vs_projected=spec["measured_vs_projected"],
        output_dir=spec["output_dir"],
        tests_estimate=spec["tests_estimate"],
        parent_refs=spec["parent_refs"],
    )


def template_to_dict(template: R22PlanTemplate) -> dict[str, Any]:
    """Convert R22PlanTemplate to JSON-serializable dict."""
    d = asdict(template)
    # Ensure phases + cross_skill_strategy + measured_vs_projected are dicts (not dataclass repr)
    d["phases"] = [asdict(p) for p in template.phases]
    d["cross_skill_strategy"] = [asdict(c) for c in template.cross_skill_strategy]
    d["measured_vs_projected"] = asdict(template.measured_vs_projected)
    return d


# --- CLI ----------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate R22-P plan JSON templates (Wf-72 Phase 2, CPU-only docs)",
    )
    parser.add_argument(
        "--plan-id",
        type=str,
        choices=sorted(PLAN_DEFINITIONS.keys()),
        help="Single R22-P plan id to emit (e.g. R22-P1). Use --all to emit all 6.",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Emit all 6 R22-P templates (R22-P1..P6).",
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output JSON path (single-plan mode).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Output directory (--all mode). Default: molmetal/reports/wf72_r22_plans/",
    )
    args = parser.parse_args(argv)

    if not args.plan_id and not args.all:
        parser.error("Either --plan-id or --all is required.")

    if args.all:
        output_dir = Path(args.output_dir or "molmetal/reports/wf72_r22_plans/")
        output_dir.mkdir(parents=True, exist_ok=True)
        for plan_id in sorted(PLAN_DEFINITIONS.keys()):
            template = build_r22_plan_template(plan_id)
            out_path = output_dir / f"{plan_id}_template.json"
            out_path.write_text(json.dumps(template_to_dict(template), indent=2, ensure_ascii=False) + "\n")
            print(f"[OK] {out_path} ({len(template.phases)} phases, status={template.status})")
        return 0

    # Single-plan mode
    if not args.output:
        parser.error("--output is required for single-plan mode.")
    template = build_r22_plan_template(args.plan_id)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(template_to_dict(template), indent=2, ensure_ascii=False) + "\n")
    print(f"[OK] {out_path} ({len(template.phases)} phases, status={template.status})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
