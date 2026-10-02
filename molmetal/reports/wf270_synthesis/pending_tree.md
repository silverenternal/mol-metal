# Wf-270 Synthesis — Prioritized Pending-Work Tree

**Date**: 2026-09-29
**Wf**: Wf-270E (aggregate Wf-270A-D findings into prioritized pending-work tree)
**Owner**: Hugo Lee + ultracode agents
**Skill alignment**: `mol-metal-cfm-training` + `mol-metal-gpu-execution` + `party-twenty-spirit` §3.5 honest framing
**Inputs aggregated**:
- Wf-270A (G6 pre-flight): cuda_fix=undefined, smoke=undefined, issues=undefined  ⚠️ **upstream returned undefineds**
- Wf-270B (TODO-119 audit): p13_status=undefined, p14_status=undefined, cpu_hr=undefined, gpu_hr=undefined, recommend=undefined  ⚠️ **upstream returned undefineds**
- Wf-270C (chemprop docs): mismatches=undefined, undocumented=undefined  ⚠️ **upstream returned undefineds**
- Wf-270D (CFM panel tests): new_tests_added=undefined, gaps=undefined  ⚠️ **upstream returned undefineds**

**Honest framing note (party-twenty-spirit §3.5)**: All four upstream waves (Wf-270A-D) returned `undefined` for their schema fields. This is an upstream-pipeline breakage, NOT an indication that the work items are zero. The prioritized tree below is therefore reconstructed from:
- `TODO/03_plans/LAMBDA_ROADMAP.md` §9.Z (5 CFM TODOs PENDING GPU; 37.8 hr total)
- `TODO/pending/180_wf124_p3_remaining_metals_seeds.md` (G4: 14/15 PENDING)
- `TODO/pending/181_wf124_p3_50_epoch_full_sweep.md` (G5: BLOCKED GPU-ordinal)
- `TODO/pending/182_wf124_p3_per_innovation_ablation.md` (G6 / TODO-119 P1.4: NOT MEASURED)
- `molmetal/reports/wf267_final/final_status.md` (25/25 pre-flight PASS; 2 residual §3.5 hygiene violations)
- `molmetal/reports/wf269c_cfm_real/wf269c_summary.md` (YuelBond IS load-bearing; ablation delta=1.000 MEASURED)
- `TODO/completed/207_unified_pre_flight_checklist_v2.md` §9.6 (CFM application-layer honest-negatives)

---

## §0 TL;DR (FLAT schema verdict)

| Gate | Value |
|---|---|
| **G1_priority_1_actions** | 5 items (citation hygiene 2 + TODO-180 G4 GPU + Wf-269C §3 cite + TODO-119 P1.3 chemistry panel) |
| **G2_priority_2_actions** | 4 items (TODO-119 P1.4 chemprop ablation + chemprop v3-fallback DrugOOD re-measurement + Wf-264 §4 refresh + CFM largest-fragment panel scale-up) |
| **G3_priority_3_actions** | 5 items (TODO-181 G5 + L9-L12 R20-P1..P6 + clinical-trial + ORNL Summit + R22+ tier-1) |
| **G4_report_path** | `molmetal/reports/wf270_synthesis/pending_tree.md` |
| **G5_n_total_actions** | **14** |
| **honest_framing_note** | All Wf-270A-D upstream feeds returned `undefined`; tree reconstructed from sibling TODO files + recent MEASURED wave reports. |

**Headline (party-twenty-spirit §3.5)**:
- **Wf-269C just delivered decisive MEASURED signal** (YuelBond ablation 1.000 → 0.000 = full signal), which means TODO-119 P1.3 has its first-half closed and §3 algorithm description can cite the YuelBond-load-bearing finding.
- **Citation hygiene §3.5 violations dropped Wf-264 baseline 5 → Wf-267 2**; the remaining 2 are quick CPU fixes (citation-only edits).
- **TODO-180 G4 has 14 PENDING (metal, seed) pairs** at ~7.7 min mean wall-time ≈ 1.8 hr GPU — quick win for paper §3 attribution honesty.
- **TODO-181 G5 BLOCKED at GPU-ordinal layer** → not actionable until R21+ GPU ordinal reconciliation (out-of-scope for ≤8 hr).
- **TODO-119 P1.4 chemprop ablation 60 runs × 7.5 min ≈ 7.5 hr GPU** — boundary item between P1 and P2; classified P2 because it requires sustained GPU scheduling.

---

## §1 PRIORITY 1 (do next, ≤4 hr)

### P1.1 — Citation hygiene §3.5 violations (CPU only, ~30 min)
**Evidence**: `molmetal/reports/wf267_final/final_status.md` §0 G2 — 2 residual violations in §2.2 line 110 (chemprop_v2_5metal_5seed_50epoch.json) + §8 line 388 (cfm_decode_real.json). Down from Wf-264 baseline 5; 3 already FIXED.
**Action**: Edit `paper/main.tex` (or `paper/sections/section_02_*.tex` + `section_08_*.tex`) to add `TODO-XXX` cross-refs / citation-hygiene provenance lines matching the §3.5 template from `TODO/completed/207_unified_pre_flight_checklist_v2.md` §3.5.
**Owner**: Hugo Lee (paper-edit authority) + ultracode agent (locate exact line contexts).
**Why P1**: Zero GPU cost; unblocks paper §4 LEAD claim hygiene at 4 disclosure sites (§2.1, §5.1, §6.1, §7.1 L11) where Wf-219 PARTIAL disclosure is needed.

### P1.2 — TODO-180 G4: 14 PENDING (metal, seed) pairs on RTX 5090 (≈ 1.8 hr GPU)
**Evidence**: `TODO/pending/180_wf124_p3_remaining_metals_seeds.md` §1 — only Ru seed=0 MEASURED at test_auc=0.7613; 14 PENDING: Ru seeds 1-2 + Ir seeds 0-2 + Os seeds 0-2 + Rh seeds 0-2 + Re seeds 0-2.
**Wave-budget cost**: 14 × ~7.7 min mean (per `molmetal/scripts/wave_budget_tracker.py` TODO_180_G4_ACTUAL_HR_BY_METAL) ≈ 1.8 hr RTX 5090 GPU-1. Sequential is fine; parallel would require kanzi contention awareness.
**Action**: Re-launch `molmetal/scripts/train_dmpnn_multitask_wasserstein.py --metal {Ru,Ir,Os,Rh,Re} --seed {1,2}` for the 14 pending pairs; populate `molmetal/reports/wf124_p3_multitask/{metal}/per_seed_metrics_seed{1,2}.json`. Aggregate into `training_summary.json` per-metal MEAN.
**Owner**: ultracode agent (GPU runner).
**Why P1**: Hard requirement for paper §3 honest framing per `party-twenty-spirit` §3.5; aggregate lift is currently reported as 1-seed + Ru-only and cannot be honestly generalized without the 14-pair MEASURED set.

### P1.3 — Cite Wf-269C YuelBond-load-bearing finding in paper §3 algorithm description (CPU, ~30 min)
**Evidence**: `molmetal/reports/wf269c_cfm_real/wf269c_summary.md` — full config decode_ratio=1.000 (51/51), no_yuelbond=0.000 (0/51), no_wasserstein=1.000 (11/11), no_closure_bounded=1.000 (projected). **YuelBond IS the load-bearing innovation** with ablation delta=1.000 (max possible).
**Action**: Add a paragraph to `paper/sections/section_03_algorithm.tex` (or equivalent) citing `molmetal/reports/wf269c_cfm_real/wf269c_summary.md` and Wf-268A flag-audit (3/3 PASS). Cite the honest-framing caveat: Wasserstein and closure-bounded knobs do NOT differentiate on RDKit-strict decode_ratio because they are auxiliary (regularization + decode-time bias) rather than gating mechanisms; chemistry-aware ablation deferred to TODO-119 P1.3.
**Owner**: Hugo Lee (paper authority).
**Why P1**: Freshly MEASURED signal that paper §3 should cite; deferred citations risk stale/incorrect claims by R20-P1 launch.

### P1.4 — TODO-119 P1.3 chemistry-aware follow-up: SA+QED+Vina on n=100 stratified (CPU + minimal Vina, ~1 hr)
**Evidence**: `TODO/completed/207_unified_pre_flight_checklist_v2.md` §9.6.1 — SA/QED/Vina on generated set NOT YET MEASURED at scale; Wf-267-Panel covered largest-fragment subset (96/100 docked, mean −6.904 kcal/mol) but the n=100 stratified CFM-generated cohort is in-flight per Wf-267-Final §1 G1.
**Action**: Complete the Wf-267 stratified n=100 cohort sweep on the YuelBond+CFM-generated mols (per `molmetal/reports/wf267_panel/aggregate.json` partial); emit final aggregate.json + per_metal.json with SA, QED, Vina mean; cite in paper §3 alongside Wf-269C.
**Owner**: ultracode agent (CPU + Vina runner).
**Why P1**: Closes the §9.6.1 honest-negative anchor for `m14` LEAD claim; without this, any §4 LEAD citation of decode_ratio must cite the §9.6 caveat per LAMBDA_ROADMAP §4.2.

### P1.5 — Update LAMBDA_ROADMAP §9.Z to reflect Wf-269C measurement (CPU, ~15 min)
**Evidence**: `TODO/03_plans/LAMBDA_ROADMAP.md` §9.Z currently says TODO-119 P1.3 = "real CrossDocked loader"; Wf-269C delivered the first-half (YuelBond ablation) on MetalCytoToxDB v4 — not CrossDocked. The §9.Z text needs a one-paragraph update citing Wf-269C + flagging the chemistry-aware follow-up as the actual P1.3.
**Action**: Edit LAMBDA_ROADMAP.md §9.Z to add a sub-bullet: "Wf-269C (2026-09-29) MEASURED YuelBond-load-bearing on MetalCytoToxDB v4 cohort (full=1.000, no_yuelbond=0.000; report at `molmetal/reports/wf269c_cfm_real/`); remaining TODO-119 P1.3 chemistry-aware follow-up per Wf-267 stratified cohort sweep."
**Owner**: Hugo Lee.
**Why P1**: Keeps the roadmap doc honest about what was measured vs projected; prevents downstream Wf-271+ from re-deriving the same finding.

**P1 total wall-time**: ~4 hr (≈ 1.8 hr GPU + ~2.2 hr CPU; can mostly parallelize CPU + GPU; bottleneck = P1.2 GPU scheduling).

---

## §2 PRIORITY 2 (after P1, ≤8 hr)

### P2.1 — TODO-119 P1.4 per-innovation chemprop ablation (60 runs × 7.5 min ≈ 7.5 hr RTX 5090)
**Evidence**: `TODO/pending/182_wf124_p3_per_innovation_ablation.md` §3 — 4 configs × 5 metals × 3 seeds = 60 runs; per-lever attribution (multi-task %, W₂ %, L2 guard %, interaction %) NOT MEASURED.
**Wave-budget cost**: 60 × 7.5 min ≈ 7.5 hr on RTX 5090 (sequential); parallel with P1.2 if kanzi has 2 RTX 5090 slots (per LAMBDA_ROADMAP §5.1).
**Action**: Run `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` with `--no-multi-task`, `--no-wass`, `--no-l2-guard`, `--single-metal {M}` flags (need to verify flag presence per `TODO/pending/182` §3.1); emit `molmetal/reports/wf124_p3_ablation/ablation_report.json`.
**Owner**: ultracode agent (GPU runner).
**Why P2, not P1**: 7.5 hr exceeds P1 budget; boundary case. Pre-flight P0 BLOCKER gate applies per `TODO/completed/207_unified_pre_flight_checklist_v2.md` §1; run with `--no-enforce-budget` if needed.
**Honest framing**: Per `TODO/pending/182` §6, the projected attribution (multi-task 60-80%, W₂ 10-30%, L2 ~5%) is a hypothesis, not measured. Do NOT cite per-lever percentages as MEASURED until 60 runs complete.

### P2.2 — chemprop v3-fallback DrugOOD empirical re-measurement (~3 hr RTX 5090)
**Evidence**: `molmetal/reports/wf267_final/final_status.md` §0 G2 + `TODO/completed/207_unified_pre_flight_checklist_v2.md` §9.5/§9.7 — Wf-263A v3-fallback mode SHIPPED structurally; empirical re-measurement TIMED OUT at 75-min wall budget (epoch 1, 7:13 elapsed).
**Wave-budget cost**: 3 configs × 5 seeds × 20 epochs ≈ 3 hr on RTX 5090 (with `--chemprop-mode v3-fallback` per Wf-263A-Verify).
**Action**: Re-run `molmetal/scripts/train_chemprop_v2_v3.py --chemprop-mode v3-fallback` for 5-seed × 3-config DrugOOD; compare to Wf-185 baseline (assay 0.7137 / scaffold 0.6707 / size 0.6757). Emit `molmetal/reports/wf263a_verify_v3_fallback/final_aggregate.json`.
**Owner**: ultracode agent (GPU runner).
**Why P2**: Structural ship is sound; empirical ground truth is PENDING; closing it lets chemprop v3-fallback be honestly cited in paper §4 LEAD with Wf-263A v3-fallback = Wf-185 baseline parity (the §9.5 honesty gate).

### P2.3 — Wf-264 final_status refresh + paper §4 LEAD refresh (~30 min CPU)
**Evidence**: `molmetal/reports/wf264_final/final_status.md` predates Wf-267-Final 25/25 + Wf-269C; needs refresh to reflect pre-flight 25/25 + Wf-269C YuelBond-load-bearing + 2 residual §3.5 hygiene status.
**Action**: Re-emit `molmetal/reports/wf264_final/final_status.md` with updated Wf-265..269 cross-refs; sync to paper §4 LEAD table footnote (per `TODO/completed/207_unified_pre_flight_checklist_v2.md` §3.5 disclosure template).
**Owner**: Hugo Lee.
**Why P2**: Keeps the final_status aggregator honest with the freshest MEASURED set; downstream Wf-270+ synthesis depends on it.

### P2.4 — CFM largest-fragment panel scale-up to n=500 (~2 hr CPU + Vina)
**Evidence**: `molmetal/reports/wf267_panel/aggregate.json` covers 96/100 docked for largest-fragment subset; full 100-cohort Vina per `molmetal/reports/wf267_panel/vina_5pocket_n20.json` is at n=20 per pocket. Honest framing per Wf-267-Final §0 says meeko 0.8.0 multi-fragment is honest-negative.
**Action**: Scale largest-fragment Vina to n=500 across 5 pockets from Wf-179 m15 cohort; emit `molmetal/reports/wf267_panel/vina_5pocket_n500.json` with STUB_FALLBACK discipline preserved per `TODO/completed/207_unified_pre_flight_checklist_v2.md` §9.6.1.
**Owner**: ultracode agent (CPU + Vina runner).
**Why P2**: Anchors `m14` + `m15` paper §4 LEAD claim with 5× the sample size; closes the largest-fragment-only caveat from Wf-267-Final §0.

**P2 total wall-time**: ~13 hr (sequential); 8 hr with dual-RTX-5090 + CPU/Vina parallelism. Boundary wave per `TODO/completed/207_unified_pre_flight_checklist_v2.md` §7.1.

---

## §3 PRIORITY 3 (R20 deferral, weeks)

### P3.1 — TODO-181 G5 50-epoch full sweep (BLOCKED at GPU-ordinal layer)
**Evidence**: `molmetal/reports/wf267_final/final_status.md` §2.2 — wrapper PID 604379 requested `--device cuda:1` but PyTorch sees only `cuda:0`; 0/15 pairs MEASURED; deferred to R21+ GPU ordinal reconciliation per LAMBDA_ROADMAP §4.2.
**Action**: NOT actionable until kanzi GPU-ordinal mapping is reconciled (env-level CUDA_VISIBLE_DEVICES vs torch.cuda.device_count() mismatch).
**Why P3**: Hard blocker at the env layer; not a code or data issue. ETA depends on env ops, not algorithm work.

### P3.2 — L9-L12 R20-P1..P6 deferred items (12 weeks per LAMBDA_ROADMAP §1.2)
**Evidence**: `TODO/03_plans/LAMBDA_ROADMAP.md` §1.2 + §10.4 — 4 deferred items:
- L9 CFM decoder GPU retrain → R20-P2 (bridge_rmse ≤ 0.5, decode_ratio ≥ 0.95)
- L10 3-way real pIC50 → R20-P4 (3way R² ≥ +0.30 on MetalCytoTox 5-metal)
- L11 vuln_f+g → R20-P2 (MCTS tree hit_rate ≥ 0.95)
- L12 closure OOD Phase 2 → R20-P5 (reachability ≥ 0.85 on ≥25 metals, 6/6 Catherine properties PASS)
**Why P3**: Multi-week GPU retrain cycle; outside the ≤8 hr budget. Per `party-twenty-spirit` §3.5, all 4 ship rows must cite §9.6 (CFM application honest-negatives) when claiming any `m14` LEAD progress.

### P3.3 — Clinical trial + wet-lab work (TODO-97 R20-P6)
**Evidence**: `TODO/03_plans/LAMBDA_ROADMAP.md` §3.3 + §5.3 — Phase I trial registration, IND filing, 1-2 patient cohort. NOT in scope for ≤8 hr wave; 24-week cycle per R23.
**Why P3**: Wet-lab + clinical facility + IRB approval — outside algorithmic scope; flagged here for completeness of the roadmap echo.

### P3.4 — ORNL Summit integration (R22-P1..P5)
**Evidence**: `TODO/03_plans/LAMBDA_ROADMAP.md` §5.2 — ≥100 nodes, 6 PB/s bandwidth; for Phase B tier-1 wet-lab 配套 GPU + AlphaFold-Multimer integration.
**Why P3**: 24-week R22 cycle; outside the ≤8 hr budget.

### P3.5 — R22+ tier-1 metric lift (m18/m19/m20)
**Evidence**: `TODO/03_plans/LAMBDA_ROADMAP.md` §3.2 — m18 Protein-Metal AUC ≥ 0.78 (AlphaFold-Multimer + MetalSite); m19 LOMO ≥ 0.75 (9-cell × scaffold split); m20 Tumor AUC ≥ 0.72 (patient-derived organoid + TCGA).
**Why P3**: Multi-year R22-R25 trajectory (2027-07 → 2030-07); flagged here to keep the long-horizon context.

---

## §4 Cross-reference (party-twenty-spirit §3.5 honest framing)

### §4.1 Upstream-feed integrity disclosure
All four Wf-270A-D feeds returned `undefined` for their schema fields. This is an upstream-pipeline breakage (likely a wave-launcher or wave-result-emit script regression) and does NOT indicate that the corresponding work items are zero. The reconstructed tree above is grounded in:
- Sibling TODO files: `TODO/pending/180`, `TODO/pending/181`, `TODO/pending/182`, `TODO/pending/177`
- Recent wave final reports: Wf-264 (5), Wf-265 (3), Wf-266 (4), Wf-267 (4), Wf-269C (1)
- Roadmap doc: `TODO/03_plans/LAMBDA_ROADMAP.md` §9.Z + §10.4
- Pre-flight checklist: `TODO/completed/207_unified_pre_flight_checklist_v2.md` §9.6 + §9.5 + §9.7

**Action**: Wf-270 upstream pipeline breakage should itself be tracked as P1.6 (≤30 min) — re-emit the Wf-270A-D wave JSONs with their actual schema fields populated, so downstream Wf-271+ has a verified input contract. *This item is NOT counted in the 14 G5_n_total_actions because it is a pipeline-recovery task, not a pending-work action.*

### §4.2 Per-priority honest-negative risk

| Priority | Risk | Mitigation |
|---|---|---|
| P1 (≤4 hr) | P1.2 GPU contention with kanzi shared nodes | Schedule for off-hours or use --no-enforce-budget; per `TODO/completed/207_unified_pre_flight_checklist_v2.md` §1 P0 BLOCKER. |
| P2 (≤8 hr) | P2.1 7.5 hr GPU may exceed session budget | Boundary wave per §7.1; emit `wave_budget_tracker.py` JSON to confirm PLANNED vs DEFERRED before launch. |
| P3 (weeks) | L9-L12 GPU retrain may numerically fail (per LAMBDA_ROADMAP §8.1 honest-negative on Wf-25 R² = −0.443) | Honest report per-metric; do not paper §4 LEAD claim. |

### §4.3 What this tree does NOT cover
- R19 §8-12 GPU training schedule (already shipping; not pending)
- Wet-lab cooperation (TODO-97 §19-23 R19_PLAN — out of scope for this aggregate)
- Paper v2 §1-§7 polish beyond §3.5 hygiene (out of scope; tracked in `mol-metal-paper-writing` skill)
- New R20-R22 skill ecosystem (tracked in LAMBDA_ROADMAP §1.5/§2.4/§6.4)

---

## §5 Final verdict

**14 prioritized pending-work actions** across P1 (5 items, ≤4 hr) + P2 (4 items, ≤8 hr) + P3 (5 items, weeks):
- **P1 is dominated by TODO-180 G4 GPU burst** (1.8 hr, the only P1 GPU item) + 3 CPU citation/edits.
- **P2 is dominated by TODO-119 P1.4 chemprop ablation** (7.5 hr GPU, boundary item) + chemprop v3-fallback empirical re-measurement (3 hr).
- **P3 is dominated by TODO-181 G5 GPU-ordinal BLOCK** + L9-L12 R20-P1..P6 + clinical/wet-lab + R22+ tier-1.

**Headline action for next session**: launch P1.2 (TODO-180 G4 14 PENDING (metal, seed) pairs on RTX 5090) in parallel with P1.1 (citation hygiene §3.5 fixes, CPU). Total wall-time ≤ 2.2 hr to close the P1 quick-win set + unblock paper §3 attribution honesty.

**Honest framing**: The Wf-270A-D upstream `undefined` feeds are a pipeline regression; Wf-270E reconstructs the pending-work tree from sibling TODOs + recent wave final reports. The reconstruction is best-effort and grounded in MEASURED artefacts (not speculation); per `party-twenty-spirit` §3.5, downstream Wf-271+ should re-verify the Wf-270 upstream feeds before treating this tree as the authoritative input contract.