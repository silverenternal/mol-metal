# Wf-299 Synthesis — CFM Follow-up Aggregation

**Date**: 2026-09-29
**Workflow**: Wf-299E (synthesis pass over Wf-299A/B/C/D)
**Honest-framing gate**: party-twenty-spirit §3.5 (MEASURED-vs-PROJECTED discipline preserved)

---

## §1 Per-Agent Summary

| agent | scope | status | key finding | artifact |
|---|---|---|---|---|
| **Wf-299A** | PlatinAI 4-cell coverage extend (MB-Finder 100 → 1000+) | **MISSING / BLOCKED** | No output directory; per Wf-299D §2.1 the agent must (1) harvest 1000+ Pt compounds, (2) scrape numeric IC50 from paper-gated DOIs, (3) apply 70/10/20 Bemis-Murcko split — none delivered. **Blocks PlatinAI cohort in Wf-299D §3 table**. | none on disk |
| **Wf-299B** | chemistry-aware metric (SA+QED+Vina on PlatinAI 100) | **PARTIAL — honest-negative** | SA+QED succeeded 97/100 (SA mean 5.05 / QED mean 0.36, drug-like in cytotoxicity space). **Vina FAILED 0/100** — RDKit UFF cannot parameterise Pt+2, every PlatinAI mol blocked at 3D conformer stage. Failures are NOT stub floor (vina_binary_ok=True, receptor staged); genuine UFF/MMFF limitation on Pt(II) coord complexes. | `wf299b_chem_metric/chem_metric_report.md`, `wf299b_sa_qed/sa_qed.json`, `wf299b_vina/vina.json` |
| **Wf-299C** | TODO-119 P1.4 launch script | **COMPLETED — READY** | `launch_wf299_cfm_main.sh` shipped: 3 waves (A=P1.4 smoke 2.0h / B=G6 no_wass smoke 0.5h / C=G6 full sweep 7.5h paper-grade subset, 12.5h full PROJECTED) × 8 sub-waves inside Wave-C × 8 abort criteria (TODO-207 v2 §7.2) × 3 pre-launch checks (pre_flight + per_wave_checklist + wave_budget_tracker) × `--dry-run` mode × per-wave `<2hr` kill_then_adapt OK gate × `per_wave_user_ok_required:true` between waves. All wall-times PROJECTED, MEASURED post-wave. | `wf299c_launch_script/launch_script_report.md`, `molmetal/scripts/launch_wf299_cfm_main.sh` |
| **Wf-299D** | cross-cohort generalization plan | **PLAN COMPLETE — no GPU work executed** | 3 cohorts identified: PlatinAI 4-cell (BLOCKED on Wf-299A), DrugOOD ECFP4 (95,236 rows, Bemis-Murcko 0-leak, organic-only negative control), MetalCytoToxDB v2 (26,632 rows, 5-metal Bemis-Murcko strict, already P1.4 primary). Per-cohort MEASURED target `decode_ratio ≥ 0.5` (current honest-negative 0.0). Honest-framing §4.2 forbids single aggregated MEASURED number until each cohort passes individually. | `wf299d_cross_cohort/plan.md` |

---

## §2 CFM Main Line Progress

The CFM main line is **Wf-124 P3 per-innovation ablation G6 + Wf-299 P1.4 sub-line** (chemprop_v2_mt backbone, h=128, n_layers=3, YuelBond bond head, joint_train, Wasserstein pIC50 reg, Catherine closure-bounded). Three Wf-299 outputs advance it, one blockers it:

| lever | status | honest framing |
|---|---|---|
| **Application-layer metric panel** (Wf-299B SA+QED+Vina) | PARTIAL | SA+QED real values (97/100) usable for chem-aware eval now. Vina → 0/100 honest-negative (Pt(II) coord complexes need Pt-stripped ligand-only path or 2D-only metric panel per Wf-299B §followup). |
| **Launch ship surface** (Wf-299C script) | SHIPPED | Ready-to-launch 3-wave orchestration with full safety net. **No CFM model MEASURED-lift yet** — script sits idle awaiting user OK. |
| **Cross-cohort generalization plan** (Wf-299D) | PLAN only | Defines 2 of 3 READY cohorts (DrugOOD + MetalCytoToxDB v2) + 1 BLOCKED (PlatinAI). No GPU work executed; no MEASURED cross-cohort decode_ratio. |
| **PlatinAI 4-cell extension** (Wf-299A) | BLOCKED | No output. Without numeric IC50 scrape + 1000+ Pt harvest, Wf-299D §3 PlatinAI column stays "TBD BLOCKED" and the in-distribution metal benchmark for P1.4 is limited to MetalCytoToxDB v2 (Ru/Ir/Os/Rh/Re). |

**CFM main line honest summary**: zero MEASURED per-lever attribution yet (60-run G6 PROJECTED, paper-grade subset 7.5h PROJECTED, smoke 2.0h PROJECTED). m14 decode_ratio remains 0.0/8 (CPU architecture ceiling) until Wf-299C's Wave-A actually launches. Wf-299B's Vina honest-negative is the **first MEASURED application-layer regression for the CFM line** (Pt coord complexes incompatible with default Vina stack).

---

## §3 Retrain Sequence (recommended)

Party-twenty-spirit §3.5 — every step below has explicit MEASURED-vs-PROJECTED marker; user-OK required between waves (Wf-299C §6).

| # | workflow | what it runs | GPU | wall (PROJECTED) | gate | honest output |
|---:|---|---|---|---|---|---|
| 1 | **Wf-299A** | PlatinAI MB-Finder 100 → 1000+ Pt harvest + numeric IC50 scrape + 70/10/20 Bemis-Murcko zero-leak split | none (data only) | 0.5 hr | CSV row-count ≥ 1000 + IC50 numeric + scaffold_split_largest_first present | PlatinAI 4-cell cohort unblocks Wf-299D §3 column 1; per-cell-line MEASURED target achievable |
| 2 | **Wf-299E** | Wave A P1.4 smoke — 4 ablations × 1 seed × 3 epochs (a_full_stack / b_yuelbond_only / c_w2_only / d_catherine_only) on **DrugOOD** + **MetalCytoToxDB v2** (2 of 3 Wf-299D cohorts, PlatinAI still BLOCKED) | `cuda:0` (PRO 6000 Blackwell) | 2.0 hr | per-wave `<2hr` kill_then_adapt; min per_epoch decode_ratio ≥ 0.05; per-seed test_auc ≥ 0.51 (TODO-207 v2 §7.2.5) | 4-ablation × 2-cohort MEASURED decode_ratio table; per-lever Δ_yuelbond/Δ_w2/Δ_catherine |
| 3 | **Wf-299F** | Wave B G6 no_wass smoke (Ru + Ir, seed 0) → Wave C G6 full sweep (60 runs × 5 metals × 3 seeds, 4 ablations, split into 8 sub-waves × `<2hr` each) | `cuda:1` (RTX 5090) | 0.5 hr smoke + 7.5 hr paper-grade (12.5 hr full PROJECTED) | per-sub-wave user OK; abort_recovery on 7.2.1-7.2.8 patterns | 5-metal × 3-seed × 4-ablation MEASURED table; per-metal × per-lever Δ; CFM main line primary attribution |
| 4 | **Wf-299G** | Write `wf299d_cross_cohort/results_table.md` + 4-ablation × N-cohort MEASURED report + per-lever attribution paragraph (Wf-124 P3 paper §3 hook) | none (analysis only) | 0.5 hr | results_table.md emitted with MEASURED columns only (TBD entries retained for PlatinAI if still BLOCKED) | honest cross-cohort report for paper §3 per-lever attribution; honest-negative preserved where cohorts missing |

**Ordering rationale** (Wf-299D §5 dependency graph): Wf-299A → Wf-299E → Wf-299F → Wf-299G. Wf-299E can launch **today** without Wf-299A (DrugOOD + MetalCytoToxDB v2 are READY), but the 3-cohort table ships only after PlatinAI extension completes.

**Total GPU budget**: Wf-299E 2.0 + Wf-299F 8.0 = **10.0 hr PROJECTED** (paper-grade subset) / **14.5 hr PROJECTED** (full 60-run G6) — fits inside the 5-seed × 50-epoch standard protocol cap per Wf-258 audit §1.2.

---

## §4 Honest-framing ledger (party-twenty-spirit §3.5)

**DO claim (post-Wf-299E)**:
- "Wf-299C orchestration script SHIPS at `molmetal/scripts/launch_wf299_cfm_main.sh` (3 waves × 8 sub-waves × 8 abort criteria × 3 pre-launch checks)."
- "Wf-299B SA+QED MEASURED on 97/100 PlatinAI mols (SA mean 5.05, QED mean 0.36) — usable for chem-aware eval."
- "Wf-299D cross-cohort PLAN COMPLETE with 3 cohorts (PlatinAI BLOCKED, DrugOOD READY, MetalCytoToxDB v2 READY)."

**DO NOT claim (until waves run)**:
- "Per-lever attribution on G6 60-run sweep" — MEASURED only after Wf-299F.
- "Cross-cohort generalization" — aggregated MEASURED only after each cohort passes individually (Wf-299D §4.2).
- "Vina docking on Pt(II) coord complexes" — honest-negative 0/100; need Pt-stripped or 2D-only path (Wf-299B §followup).

**Honest-negative catalogued today**:
- Vina 0/100 on Pt(II) coord complexes (Wf-299B) — first application-layer CFM regression.
- PlatinAI 4-cell BLOCKED on numeric IC50 scrape (Wf-299D §2.1) — data-side honest-negative.
- m14 decode_ratio 0.0/8 floor (unchanged from R16b, INDEX.md §530) — model-side honest-negative still active.

---

## §5 Cross-references

- `TODO/completed/207_unified_pre_flight_checklist_v2.md` §1/§6/§7 (BLOCKER + fail-fast + wave budget)
- `TODO/pending/119_decode_ratio_strengthen.md` (Wf-299C origin)
- `TODO/pending/182_wf124_p3_per_innovation_ablation.md` (Wf-299E/F parent TODO)
- `TODO/pending/130_drugood_official_datasets.md` (DrugOOD adapter ship)
- `molmetal/reports/wf258_audit/cfm_todo_gap_audit.md` §1.2 (P1.4 audit)
- `molmetal/reports/wf295b_platinai_loader/*.md` (PlatinAI loader ship)
- `molmetal/reports/wf261_update/*.md` (MetalCytoToxDB v2 strict scaffold split)
- `molmetal/reports/wf292d_platinai_bypass/strategy_report.md` §3 R1 (Wf-299A strategy)
- `molmetal/reports/wf295d_launch_sequence/` (parent launch-sequence readiness)
- `molmetal/reports/wf262_audit/` (4-script ship acceptance gates)

---

**Status**: SYNTHESIS COMPLETE — 3 of 4 agents delivered, 1 (Wf-299A) MISSING/BLOCKED. CFM main line has ship-ready launch surface (Wf-299C) + chem-aware eval panel (Wf-299B SA+QED only) + cross-cohort plan (Wf-299D), but ZERO MEASURED per-lever attribution yet. Retrain sequence ordered by Wf-299D §5 dependency graph: Wf-299A → Wf-299E → Wf-299F → Wf-299G, total 10.0 hr PROJECTED GPU for paper-grade subset.