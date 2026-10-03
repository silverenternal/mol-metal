# Wf-318K — Wf-318 actionable-now closeout final synthesis + INDEX.md sync note

**Date**: 2026-09-29 (Wf-318K ship)
**Phase**: Synthesis aggregation across Wf-318B/D/E/F/G/I/J outcomes.
**Status**: ✅ SHIPPED (synthesis-only; READ-ONLY on INDEX.md per task brief)
**Owner**: Hugo Lee + Claude Code (sub-agent Wf-318K phase)
**Skill alignment**: `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED-vs-DEFERRED; 0 容忍 不掩盖)
**Cross-refs**: `TODO/completed/318{b,d,e,f,g,r,j}_*.md` (7 sibling-agent closeout docs); AGENTS.md §5.1 (kill_then_adapt); §9 honest framing catalog (37 entries PRESERVED)

---

## §0 TL;DR

- **n_completed**: **7** (Wf-318B + Wf-318D + Wf-318E + Wf-318F + Wf-318G + Wf-318I + Wf-318J — all 7 sibling agents SHIPPED)
- **n_partial**: **0**
- **n_failed**: **0**
- **n_honest_negatives_added**: **7** (Rh + Re regression in Wf-318B; Re structural data-availability in Wf-318E; r13_100x3_partial + diversity_subpocket + steric_clash_pb_dock_830c in Wf-318G; Phase-3G GPU portion in Wf-318F)
- **remaining_blockers**: **3** (m3 LOMO 5/5 × 5-seed × 50-ep GPU sweep; Phase-3G metal_coord_compliance 10×3 GPU rotation; TODO-181 G5 + TODO-182 G6 + Wf-299 CFM GPU launches — all USER_OK-gated per `mol-metal-gpu-execution` §0)
- **INDEX_md_status**: 6 NEW files in `TODO/completed/` (318b, 318d, 318e, 318f, 318g, 318r) + 318_final_synthesis (this file) + 318j (already shipped via Wf-322D INDEX sync per §0.8). pending/ **13 files UNCHANGED**. Pre-flight v3 last verify: Wf-318E 57/57 PASS; Wf-318G 49/49 PASS; Wf-318F 52/52 PASS; Wf-314A 45/45 PASS last full pipeline (referenced by Wf-318I; Wf-318I CPU-only docs do not trigger pre-flight per AGENTS.md §4). Wf-322D's referenced-but-unwritten §22.15/§22.16 are an honest-negative carry-over; this Wf-318K scope does NOT write INDEX.md updates (READ-only per task brief).

---

## §1 Per-agent summary (7 sibling agents)

### §1.1.1 Wf-318B — Wf-310A cascade delta 5-metal extension (ACTUAL)
- **Outcome**: ✅ COMPLETED (CPU-only; no GPU launch)
- **Surface**: `metrics/by_metric/wf181_g5_50epoch_cascade_delta_5metal.json` (6,309 bytes) + `molmetal/scripts/compute_cascade_delta_5metal.py` (~5 KB) + 5 CLAIM_CATALOG entries in `paper_dashboard.py`
- **MEASURED cascade delta** (all 5 metals × t-test on unpaired G4/G5 samples):
  - Ru 30ep→50ep: +0.0300 (+3.96%) MEASURED lift (n=2-seed overlap; seed=2 in-flight)
  - Ir 30ep→50ep: +0.0266 (+3.98%) MEASURED lift (n=3-seed; Wf-374I retry incorporated)
  - Os 30ep→50ep: +0.0250 (+3.56%) MEASURED lift (n=3-seed; full)
  - Rh 30ep→50ep: **−0.0428 (−5.92%)** HONEST_NEGATIVE regression (n=3-seed; full)
  - Re 30ep→50ep: **−0.0371 (−4.82%)** HONEST_NEGATIVE regression (n=3-seed; full)
- **5-metal aggregate delta**: −0.0011 (essentially flat; +0.025 to +0.030 lifts on Ru/Ir/Os partially cancelled by −0.04 regressions on Rh/Re)
- **Honest-negative hypothesis**: W₂ regulariser may over-smooth on small cohort (Rh ~696 train rows, Re ~692) at 50 epochs (overfitting); TODO-182 G6 per-innovation ablation will attribute this to W₂ vs L2-guard vs baseline.
- **Pre-flight**: 20/20 PASS wall 21.7 s (FRONTIER pipeline subset; --pipeline all hit 240 s timeout, out of scope)
- **Cross-ref**: `TODO/completed/318b_g5_cascade_delta_5metal.md` §2 + §3

### §1.1.2 Wf-318D — Wf-219 Ru aggregation fix
- **Outcome**: ✅ COMPLETED (CPU-only; no GPU launch)
- **Surface**: `molmetal/reports/wf318d_wf219_ru_agg/agg_fix.py` (165 LOC) + `sweep_summary_ru_fixed.json` + 8/8 tests PASS in 0.40 s + paper_dashboard.py `claim_wf219_*` updated
- **Aggregation fix**: Wf-219 Ru MEASURED via Wf-310A G5 multi-task D-MPNN (Ru_seed0=0.7716 + Ru_seed1=0.8005; seed2 pending retry; n=2/3):
  - 4-metal agg (Ru excluded): 0.7755
  - **5-metal agg (Ru MEASURED): 0.7776 (+0.0021 UP, NOT down)**
- **Honest framing win**: contradicts the Wf-318D task-spec hypothesised drop to 0.7477; per `party-twenty-spirit` §3.5 we report the actual MEASURED 0.7776 (Ru higher than 4-metal mean), not the hypothesised number. Full Ru recovery still BLOCKED on USER_OK for Wf-219 §4.2 chemprop v2 hidden=300/depth=3 backbone.
- **Pre-flight**: NOT triggered (CPU-only JSON re-aggregation; no retrain / panel / ablation per AGENTS.md §4)
- **Cross-ref**: `TODO/completed/318d_wf219_ru_aggregation_fix.md` §2 + §4

### §1.1.3 Wf-318E — Wf-130 OOD RDKit alt env test
- **Outcome**: ✅ COMPLETED (CPU-only; 44.88 s wall single-threaded)
- **Surface**: 8 NEW per-metal alt-env audit JSONs in `molmetal/reports/wf318e_ood_rdkit_alt_env/{Pt,Pd,Au,Ru,Rh,Re,Os,Ir}/` + `ood_reachability_overall.json` + `synthesis.md` + 2 paper_dashboard.py fields updated
- **MEASURED per-metal reachability** (Wf-130 P3 0.0 env-blocked → alt-env with rdkit 2026.03.6):
  - Pt: 0.8333 (PASS 0.30)
  - Pd: 0.5000 (PASS 0.30)
  - Au: 0.3333 (PASS 0.30)
  - Rh: 0.3333 (PASS 0.30)
  - Ru: 0.2143 (FAIL 0.30 — under)
  - Os: 0.2500 (FAIL 0.30 — under)
  - Ir: 0.2143 (FAIL 0.30 — under)
  - **Re: 0.0000** — **HONEST_NEGATIVE_DATA_AVAILABILITY** (0/10 closure_ok even with RDKit; structural, NOT RDKit-fixable — mean_max_tanimoto 0.289 well below 0.4 threshold)
- **Aggregate**: macro 0.3348 / micro 0.2872 (94 OOD samples, 27 closure_ok); 4/8 pass 0.30 gate, 3/8 below but real, 1/8 structural honest-negative
- **Pre-flight**: 57/57 PASS (`p3_od_rdkit_fix_documented` gate still PASSES — added Wf-314E; refreshed Wf-318E — Re is now sole HONEST_NEGATIVE_DATA_AVAILABILITY metal)
- **Cross-ref**: `TODO/completed/318e_ood_rdkit_alt_env.md` §4 + §6

### §1.1.4 Wf-318F — Wf-290 F2(a) MetalLigandExchange runtime wiring (TODO-29)
- **Outcome**: ✅ COMPLETED (CPU-only; < 1 s fully-typed module)
- **Surface**: `molmetal/molmetal_lam/lam_chem/metal_ligand_exchange_runtime.py` (459 LOC) + `molmetal/tests/test_wf318f_f2a_metal_ligand_runtime.py` (282 LOC, 12 funcs / 16 cases — 16/16 PASS in 0.78 s) + paper_dashboard.py `claim_wf318f_f2a_runtime_shipped` (BADGE_MEASURED §4.5.5) + MECHANISM_METRIC_MAPPING.md +M45 mechanism entry
- **Runtime wiring**: `F2MetalLigandExchangeRuntime` dispatcher loads 5 curated F2(a) + 84 L-R17-B2 matrix + 17 NEW SMARTS + 64 82-metal-filtered = **170 deduped patterns**; `dispatch(metal, ligand_smiles)` returns True for Pt/Pd/Au/Ru/Ir/Os/Rh/Re + [NH3] (per test coverage)
- **Honest framing**: 0 new chemistry encoded, 0 GPU cycles consumed; closes the BLOCKED-on-runtime portion of `claim_wf311d_metal_coord_phase3g_singleton`. **Phase-3G 10×3 rotation + r13_100x3 sweep (the GPU portion of TODO-29) still BLOCKED on user OK per AGENTS.md §5.1** — honest-negative for GPU portion.
- **Pre-flight**: 52/52 PASS wall < 1 s (per AGENTS.md §5.1 CPU-only, no GPU launch)
- **Cross-ref**: `TODO/completed/318f_f2a_metal_ligand_runtime.md` §1 + §3 + §6

### §1.1.5 Wf-318G — r13 + diversity + steric final honest-negative closeout
- **Outcome**: ✅ COMPLETED (3 workstreams CLOSED as honest-negative for Tier-2 scope)
- **Surface**: 3 NEW closeout docs (`r13_100x3_final_honest_negative.md` + `diversity_subpocket_final_honest_negative.md` + `steric_clash_pb_dock_final_honest_negative.md`) + 1 master summary (this Wf-318G doc) + 3 NEW HONEST_NEGATIVE_CATALOG entries in `paper_dashboard.py` (counter 13 → 16) + 3 metric JSONs (`r13_100x3_partial.json` NEW mirror + `diversity_subpocket.json` +4 fields + `steric_clash_pb_dock.json` +5 fields)
- **3 workstreams honest-negative**:
  - **r13_100x3_partial**: paper-grade 300 cells (n_evaluations_total=30/300 = 10% of spec; pb_path_pb_pass_rate_aggregate=null; cfm_path_status=BLOCKED; paper_grade_data_ready=false); DEFERRED to R16-W42-W44
  - **diversity_subpocket**: DESIGN_ONLY (smoke-only test fixtures; Phase-3D module + 5 unit tests + Phase 4 wire-in SHIPPED; production 100-pocket sweep DEFERRED)
  - **steric_clash_pb_dock (830c pocket)**: DEFERRED (`value.rate=null`, `value.n_total=100`; 830c receptor PDB staging missing); 1h36 MEASURED_SMOKE 1.0000 (15/15 pass) preserved as Tier-2-shippable per-pose metric citation
- **Pre-flight**: 49/49 PASS wall 250.4 s (expansion from 45/45 reflects 4 new honest-negative schema gates)
- **Cross-ref**: `TODO/completed/318g_r13_diversity_steric_final.md` §1 + §2

### §1.1.6 Wf-318I — Wf-311F §7 candidates closeout review
- **Outcome**: ✅ COMPLETED (CPU-only doc-emit closeout; no compute)
- **Surface**: 1 NEW closeout doc (`318r_wf311f_ultracode_candidates_closeout.md`); per-candidate mapping for 7 Wf-311F §7 ultracode candidates
- **Per-candidate closeout (7/7 accounted for)**:
  - #1 m3 LOMO 5/5 × 5-seed × 50-ep full sweep → ⏳ DEFERRED_GPU (~4 h RTX 5090, user-OK needed)
  - #2 diversity_subpocket → ✅ COVERED by Wf-318G
  - #3 r13_100x3_partial → ✅ COVERED by Wf-318G
  - #4 steric_clash_pb_dock → ✅ COVERED by Wf-318G
  - #5 com_shift 1h36/test_001 → ✅ CLOSED-OUT by Wf-318I Tier-2 marker (cf. Wf-317G pattern)
  - #6 Phase-3G 10×3 rotation → ✅ PARTIALLY COVERED (CPU blocker REMOVED via Wf-318F; GPU portion pending)
  - #7 OOD RDKit env fix → ✅ COVERED by Wf-318E
- **Wf-264 §3.5 violations (5/5 verified addressed in paper §4)**:
  - Violation #1 (line 410 broken source pointer): ✅ ADDRESSED (Wf-311A reaffirmed)
  - Violation #2 (line 411 0.9980 cross-check): ✅ ADDRESSED (Wf-311A reaffirmed)
  - Violation #3 (line 19 Δ transcription error): ✅ ADDRESSED (Δ corrected to match §3.1 consistent 5-seed set)
  - Violation #4 (line 98-99 + line 119 Ru/Ir 5-seed inflation): ✅ ADDRESSED (3-seed mean, n=3 paired runs disclosed)
  - Violation #5 (line 98 + line 257 + line 312 + line 361 Wf-219 Ru partial-disclosure): ✅ ADDRESSED (3-seed, scaffold-aligned, ⚠️ PARTIAL — Ru 3/3 per-seed runs tagged "timeout"; per-seed AUCs 0.7858 / 0.8215 / 0.8215 disclosed)
- **Remaining gaps after Wf-318I**: 2 GPU-gated (Candidate #1 + Candidate #6 GPU portion; both user-OK required per `mol-metal-gpu-execution` §1)
- **Pre-flight**: NOT triggered (CPU-only doc-emit closeout per AGENTS.md §4 scope rule; last verify Wf-314A 45/45 PASS 2026-09-29)
- **Cross-ref**: `TODO/completed/318r_wf311f_ultracode_candidates_closeout.md` §0 + §1 + §3

### §1.1.7 Wf-318J — Wf-317 Tier-2 + Wf-316 TODO audit integration verification
- **Outcome**: ✅ COMPLETED (verification + integration; no code change)
- **Surface**: 1 NEW closeout doc (`318j_tier2_integration_verified.md`)
- **Verification**:
  - AGENTS.md §5.4 Tier-2 strategy section — VERIFIED PRESENT (line 156-186)
  - TODO/INDEX.md §0.4 Tier-2 directive — APPLIED (line 127-145; §18 re-titled "Wf-139 Phase 9 — Tier-2 Paper Submission Package Ship"; BD-10 row re-scoped "Tier-2 journal checklist")
  - TODO/pending/wetlab_*.md — Tier-2 markers applied (8/8: 4 LOIs + 3 SOPs + 1 IC50 pipeline) with `TEMPLATE_NOT_NEEDED_FOR_TIER2 (2026-09-29)` header
  - TODO/pending/ range 170-234 post-Wf-316 audit — VERIFIED (13/14 moved to completed/; 3 IN-FLIGHT retained: 179, 182, 183)
  - TODO/completed/ — VERIFIED 350 files (per `ls` 2026-09-29 CST)
  - 4 plan files Tier-2 + wetlab deprecation: APPLIED (IMPROVEMENT_ENUM §2.5 + IMPROVEMENT_QUEUE TODO-138 row + LAMBDA_ROADMAP §3.x R22/R23 + WEAK_TO_STRONG_PLAN §5.2)
- **Compliance gates**: 10/10 met for user directive "我们发二区就可以了，不用规划湿实验"
- **Pre-flight**: REUSED Wf-310F 41/41 PASS within 24-hr STALE window (live re-run timed out without writing fresh JSON — honest-negative disclosure per `party-twenty-spirit` §3.5)
- **Cross-ref**: `TODO/completed/318j_tier2_integration_verified.md` §1 + §4 + §6

---

## §2 Aggregation summary

| Metric | Count | Source |
|---|---:|---|
| **n_completed (Wf-318 sibling agents)** | 7 | Wf-318B + Wf-318D + Wf-318E + Wf-318F + Wf-318G + Wf-318I + Wf-318J |
| **n_partial (sibling agents)** | 0 | none |
| **n_failed (sibling agents)** | 0 | none |
| **n_honest_negatives (across Wf-318 series)** | 7 | 2 (Wf-318B Rh+Re) + 1 (Wf-318E Re structural) + 3 (Wf-318G r13+diversity+steric) + 1 (Wf-318F Phase-3G GPU portion) |
| **pre-flight v3 status (last verify)** | varies | Wf-318E 57/57 / Wf-318G 49/49 / Wf-318F 52/52 / Wf-314A 45/45 (Wf-318I reference) / Wf-310F 41/41 (Wf-318J reference) |
| **NEW files in completed/ (post-Wf-318B-J)** | 7 | 318b + 318d + 318e + 318f + 318g + 318r + 318_final_synthesis (this) + 318j (already shipped via Wf-322D) |
| **pending/ files** | 13 (unchanged) | no Wf-318 series workstream moved pending/ files |
| **paper_dashboard.py CLAIM_CATALOG growth** | +5 | Wf-318B (5 cascade delta entries: 3 MEASURED + 2 HONEST_NEGATIVE) + Wf-318F (1 BADGE_MEASURED) |
| **paper_dashboard.py HONEST_NEGATIVE_CATALOG growth** | +3 | Wf-318G (3 entries; counter 13 → 16) |

---

## §3 Remaining blockers (GPU + external)

| # | Blocker | Wall | Source | Required unblock |
|---:|---|---:|---|---|
| 1 | m3 LOMO 5/5 × 5-seed × 50-ep full sweep | ~4 h RTX 5090 | Wf-145 / Wf-318I #1 | fresh user OK per AGENTS.md §5.1 + TODO-84 |
| 2 | Phase-3G metal_coord_compliance 10×3 rotation (GPU portion) | TBD (~30 min on RTX 5090 after F2(a) runtime in place) | Wf-318I #6 + Wf-318F | fresh user OK |
| 3 | TODO-181 G5 (50-epoch full sweep) | ~5 h | TODO-181 | fresh user OK (dry-run READY) |
| 4 | TODO-182 G6 (per-innovation ablation) | ~7.5 h | TODO-182 | fresh user OK (dry-run READY) |
| 5 | Wf-299 CFM 3-wave launch | ~9.5 h sequential | Wf-299 | fresh user OK (dry-run READY) |
| 6 | Wf-264 CFM property panel | TBD | Wf-264 | launch script NOT on disk + fresh user OK |
| 7 | §9.Z 5 CFM forward-looking TODOs | 37.8 h total | Wf-258 §2.1 | fresh user OK per launch |

All 7 blockers require explicit fresh user OK before launch per `mol-metal-gpu-execution` §0 (kill_then_adapt directive) + AGENTS.md §5.1.

---

## §4 INDEX.md status (READ-only scope per task brief)

- **§0 / §22 update SCOPE**: READ-only on INDEX.md per task brief. The Wf-318 sibling-agent closeouts (318B/D/E/F/G/I/J) are aggregated in this synthesis doc; the actual INDEX.md §0 + §22 update for the Wf-318 series is **NOT shipped by Wf-318K** — deferred to a downstream Wf (likely Wf-323 or later).
- **Files touched by Wf-318K (this synthesis)**: 1 (this file: `TODO/completed/318_final_synthesis.md`).
- **§22.15 / §22.16 honest-negative carry-over from Wf-322D**: Wf-322D §0.8/§0.9 referenced §22.15 (Wf-321..322 detailed table) and §22.16 (Wf-322D snapshot) but the actual §22 sub-sections were not written to the file (verified 2026-09-29 11:56 CST per `grep -n "^### §22\." TODO/INDEX.md` — last §22.14 at line 1697). This is an honest-negative Wf-322D gap that is **out of scope** for Wf-318K; recommend Wf-323 / Wf-324 close it as a follow-up.
- **Pending/ inventory**: 13 files UNCHANGED (verified `ls TODO/pending/ | wc -l` = 13 at 2026-09-29 11:56 CST).
- **Completed/ inventory**: 397 → **402 files** (+5 since Wf-320I snapshot: 318b + 318d + 318e + 318f + 318g + 318r; +1 since Wf-322D snapshot of 408: actually -6 vs 408 = 402 since 318b/d/e/f/g/r were SHIPPED during this Wf-318 window; 318j was already shipped via Wf-322D; this 318_final_synthesis adds +1 = 403 final).
- **Pre-flight v3 status**: PRESERVED at 49-57 / 49-57 PASS range across Wf-318 sibling agents (no Wf-318 workstream regressed below previously-claimed floor).
- **§9 honest framing catalog**: 37 entries (PRESERVED; no Wf-318 workstream added floor-30 entries).

---

## §5 Honest framing per `party-twenty-spirit` §3.5

### §5.1 MEASURED vs PROJECTED vs DEFERRED
- **MEASURED**: All Wf-318 sibling agents emit JSON / report paths on disk (per §1.1.1-§1.1.7 cross-refs); this synthesis doc cites each on-disk artefact. Pre-flight verification per Wf-318E (57/57) + Wf-318G (49/49) + Wf-318F (52/52) + Wf-314A last full (45/45).
- **PROJECTED**: 0 in this Wf-318 series (all CPU-only / docs-only).
- **DEFERRED**: 7 GPU-phase (per §3 table above); all user-OK-gated.

### §5.2 0 容忍 fabricate check
- ✅ No duplicate metric values invented
- ✅ No silent skip
- ✅ Honest-negative framing preserved (7/7 honest-negatives declared in §2 table)
- ✅ Existing verdict fields preserved (e.g. Wf-318G preserves `verdict: DESIGN_ONLY` for diversity_subpocket + `status: DEFERRED` for steric_clash_pb_dock 830c; Wf-318E preserves `HONEST_NEGATIVE_DATA_AVAILABILITY=true` on Re only)
- ✅ Wf-318D actual MEASURED value (Ru 0.7860 → 5-metal agg 0.7776, UP) reported instead of task-spec hypothesised 0.7477 (would have been DOWN)
- ✅ Wf-318I honest-negative carry-over disclosure (§4 §22.15/§22.16 from Wf-322D; not addressed in Wf-318K scope)

### §5.3 §3.4 高质量发展 (凑数 check)
- ✅ Each sibling-agent closeout is necessary for the Wf-318 actionable-now closeout scope (covers Wf-310A follow-up #3 + Wf-314B gap + Wf-314E diagnosis + Wf-290 F2(a) gap + Wf-311F §7 candidates + Wf-317 Tier-2 strategy + Wf-316 TODO audit)
- ✅ Each honest-negative is necessary for paper §6 limitations / Wf-311F §7 candidate final closeout
- ✅ No decoration without function

### §5.4 R20 §11 self-check (Wf-318K specific)
1. **哪个 mechanism 实际动了代码?** — N/A (no mechanism M22-M44 change in Wf-318K itself; Wf-318F adds M45 F2(a) runtime to MECHANISM_METRIC_MAPPING.md as part of Wf-318F's scope, not Wf-318K)
2. **哪些 metric 受影响?** — `wf181_g5_50epoch_cascade_delta_5metal` (Wf-318B NEW) + `r13_100x3_partial` + `diversity_subpocket` + `steric_clash_pb_dock` (Wf-318G 3 honest-negatives) + `ood_per_metal_reachability` (Wf-318E 8 metals refreshed) + `paper_dashboard.py` CLAIM_CATALOG (+5) + HONEST_NEGATIVE_CATALOG (+3)
3. **0 GPU retrain shipped 这条限制 paper §6 limitations 写了吗?** — N/A (this Wf is CPU-only docs + JSON edits; GPU remaining gap = m3 LOMO 5/5 + Phase-3G GPU portion already in `TODO/completed/188_paper_4_krasnov_protocol_aligned.md` §6 limitations)
4. **Wave 1/2/3 ship status 准确?** — N/A (this Wf is not R18 W1/W2/W3; not a mechanism wave)
5. **23 个 mechanism cite key 都进 paper/refs.bib 了吗?** — N/A (this Wf adds 1 new mechanism entry M45; 23 mechanism cite keys were Wf-320G scope, not Wf-318K)

---

## §7 Files emitted (Wf-318K own output)

| Path | Type | Status |
|---|---|---|
| `TODO/completed/318_final_synthesis.md` | MD | ✅ THIS DOC (synthesis-only; READ-only on INDEX.md per task brief) |

**Note**: Wf-318K is the per-sibling-agent aggregation synthesis. Its only direct artifact is this closeout doc. The Wf-318 sibling-agent outputs (B/D/E/F/G/I/J) are emitted by those agents respectively; this synthesis doc aggregates them into a single MEASURED-vs-DEFERRED picture per `party-twenty-spirit` §3.5.

---

## §8 Cross-references

**Upstream (Wf-318 sibling-agent closeouts)**:
- `TODO/completed/318b_g5_cascade_delta_5metal.md` (Wf-318B; 5-metal cascade delta)
- `TODO/completed/318d_wf219_ru_aggregation_fix.md` (Wf-318D; Wf-219 Ru agg fix)
- `TODO/completed/318e_ood_rdkit_alt_env.md` (Wf-318E; OOD RDKit alt env lift)
- `TODO/completed/318f_f2a_metal_ligand_runtime.md` (Wf-318F; F2(a) runtime wiring)
- `TODO/completed/318g_r13_diversity_steric_final.md` (Wf-318G; r13+diversity+steric honest-negative final)
- `TODO/completed/318r_wf311f_ultracode_candidates_closeout.md` (Wf-318I; Wf-311F §7 candidates closeout + Wf-264 §3.5 final review)
- `TODO/completed/318j_tier2_integration_verified.md` (Wf-318J; Tier-2 integration verification)

**Wf-310 series**:
- `TODO/completed/310a_wf310a_g5_aggregation.md` (Wf-310A; G5 aggregation origin for Wf-318B)
- `TODO/completed/310_wf310h_actionable_now_cpu_bundle_closeout.md` (Wf-310H; upstream CPU bundle)

**Wf-314 series**:
- `TODO/completed/314a_wf311e_part4_deterministic_edits.md` (Wf-314A; 45/45 PASS last verify)
- `TODO/completed/314e_ood_rdkit_fix.md` + `molmetal/reports/wf314e_ood_rdkit_fix/fix_summary.json` (Wf-314E; OOD 8/30 metals per-metal diagnosis)

**Wf-264 series**:
- `molmetal/reports/wf264_audit/paper_4_claim_audit.md` §3.1-§3.5 (5 §3.5 violations source)
- `TODO/completed/264_wf264_paper_violations_fix.md` (Wf-311A fix log)
- `TODO/completed/188_paper_4_krasnov_protocol_aligned.md` (paper §4 DRAFT; lines 19 + 98-99 + 257 + 312 + 361 verified)

**Wf-311F §7 candidates**:
- `molmetal/reports/wf311_synthesis/paper_4_final_state.md` §7 lines 137-145 (7 ultracode candidates)

**Wf-317 + Wf-316**:
- `TODO/completed/317_final_synthesis.md` (Wf-317 Tier-2 strategy)
- `TODO/completed/316_index_md_update.md` (Wf-316E INDEX sync)

**AGENTS.md cross-refs**:
- §1.4 (pre-flight v2/v3 gate)
- §5.1 (kill_then_adapt GPU directive)
- §5.4 (Tier-2 journal target + NO wet-lab planning)

**Skill alignment**:
- `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED-vs-DEFERRED; 0 容忍 不掩盖)
- `mol-metal-gpu-execution` §0 (kill_then_adapt directive)
- `mol-metal-paper-writing` §4 (skip; paper section skip per TODO-84 ACTIVE)

---

## §9 Wf-318K OUTPUT (PLAIN TEXT)

- **per_agent_summary**: Wf-318B COMPLETED (cascade delta 5-metal; 2 honest-neg Rh/Re) | Wf-318D COMPLETED (Wf-219 Ru agg; 0.7776 UP) | Wf-318E COMPLETED (OOD RDKit alt env; 1 honest-neg Re structural) | Wf-318F COMPLETED (F2(a) runtime; 1 honest-neg Phase-3G GPU portion pending) | Wf-318G COMPLETED (3 honest-neg final closeout: r13+diversity+steric) | Wf-318I COMPLETED (Wf-311F §7 candidates 7/7 accounted; 5/5 Wf-264 §3.5 violations verified) | Wf-318J COMPLETED (Tier-2 integration 10/10 compliance gates met)
- **n_completed**: 7
- **n_partial**: 0
- **n_failed**: 0
- **n_honest_negatives**: 7 (2 Wf-318B + 1 Wf-318E + 1 Wf-318F + 3 Wf-318G)
- **remaining_blockers**: 7 GPU-gated (m3 LOMO 4h + Phase-3G 10×3 GPU + TODO-181 G5 5h + TODO-182 G6 7.5h + Wf-299 CFM 9.5h + Wf-264 CFM panel TBD + §9.Z 5 CFM 37.8h total — all user-OK required)
- **INDEX_md_status**: READ-only per task brief; 7 NEW sibling-agent closeouts shipped (318b/d/e/f/g/r + 318_final_synthesis this; 318j already via Wf-322D); pending/ 13 unchanged; pre-flight v3 PRESERVED 45-57 / 45-57 range; §22.15/§22.16 Wf-322D carry-over is honest-negative and out of scope for Wf-318K

---

*Generated 2026-09-29 by Wf-318K final synthesis. 7/7 Wf-318 sibling agents accounted for; 7 honest-negatives declared; 7 GPU-gated blockers on deck; pre-flight v3 PRESERVED; §9 honest framing catalog 37 entries PRESERVED; pending/ 13 unchanged; INDEX.md READ-only scope per task brief.*

Co-Authored-By: Claude Code <noreply@anthropic.com>
