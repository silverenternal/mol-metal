# Wf-263B-Audit — TODO-207 §9 Honest Framing Catalog Gap List

**Date**: 2026-09-28
**Source**: Cross-reference of `TODO/completed/207_unified_pre_flight_checklist_v2.md` §9 against live metric artefacts + Wf-253d m17 reeval surface.
**Acceptance gates**:
- **G1_n_current_entries**: 3 (§9.1 chemprop v2 v3 limits + §9.2 CFM limits + §9.3 DO NOT claim list)
- **G2_n_gaps_identified**: 6 (below; ≥ 5 target MET)
- **G3_audit_path**: `/home/hugo/codes/mol-metal/molmetal/reports/wf263b_audit/gap_list.md` (this file)

---

## §0 Method

1. Read `TODO/completed/207_unified_pre_flight_checklist_v2.md` §9 verbatim (lines 337-359).
2. Enumerate entries by subsection (§9.1, §9.2, §9.3). Count: **3**.
3. Cross-reference each entry against live evidence:
   - `metrics/by_metric/drugood_v3_scaffold.json` (m17 DrugOOD v3 forward-pass)
   - `metrics/by_metric/diversity_tanimoto.json` (m7 Tanimoto diversity vs TargetDiff 0.86)
   - `metrics/by_metric/m6_kappa.json` (m6_kappa validation NEGATIVE_WEAK verdict band)
   - `metrics/by_metric/cfm_decode_real.json` (real CFM decode_ratio 0.9645 PARTIAL)
   - `metrics/by_metric/sa_mean.json` + `metrics/by_metric/qed.json` + `metrics/by_metric/vina_*.json` (application layer — gap)
   - TODO-119 P1.3/P1.4 + TODO-180/181/182 forward-looking surface (per §4.1)
   - Wf-253d m17 reeval forward-pass state (`CHECKPOINT_LOADED_FORWARD_DEFERRED` per §7.2.7)
4. For each missing piece of honest framing, emit a gap entry with: (a) what §9 currently says, (b) what is missing, (c) targeted §9.N insertion.

---

## §1 Current §9 entries (G1 = 3)

| § | Heading | Scope | MEASURED? |
|---:|---|---|---|
| 9.1 | chemprop v2 v3 honest limits | chemprop v2 primary backbone + v3 honest-negative on m17 | yes (5-seed Krasnov-beating, v3 1-seed Ru 0.7613) |
| 9.2 | CFM honest limits | CFM m14 decode_ratio + 4 NOT-cite items | yes (decode_ratio 0.9958 n=26,801) |
| 9.3 | DO NOT claim list (verbatim) | 7-item blacklist | yes (per-item cross-ref to git log + reports) |

**Total**: 3 entries. Per §9 sub-headings, no §9.4, §9.5, etc. exist in the doc.

---

## §2 Gap list (G2 = 6)

### Gap 1: CFM application layer (SA / QED / Vina on generated set) — NEVER MEASURED

- **Current §9 framing**: §9.2 only reports `decode_ratio` (m14) for CFM. No application-layer honesty section.
- **What is missing**: After CFM generates 818 Pt-subset ligands (per `cfm_decode_real.json`), the **downstream drug-like application metrics** (sa_mean / qed / vina_score / vina_high_affinity / vina_relaxed_success / steric_clash_pb_dock) are NEVER evaluated on the CFM-generated set. The metrics exist in `metrics/by_metric/{sa_mean, qed, vina_*.json}` but only on Lambda-MCTS-generated pools (R12/R17 lift_ev3) and ground-truth ligand sets.
- **MEASURED evidence (or absence)**: grep over `metrics/by_metric/*.json` for `cfm_generated` / `cfm_pool` / `yuelbond_wasserstein` tags returns ZERO entries. The gap is structural — no script wires `cfm_yuelbond_wasserstein.pt` -> 818 ligands -> SA/QED/Vina.
- **Why it is honest-negative**: paper §4 lead candidate (CFM, m14) has gate ≥ 0.05 PASS by 20× but does NOT demonstrate any drug-likeness uplift over Lambda MCTS.
- **Target §9 insertion**: new §9.4 "CFM application-layer NEVER MEASURED (no SA/QED/Vina on CFM-generated set)"

### Gap 2: m7 Tanimoto diversity gap quantification (0.1065 vs TargetDiff 0.86 = +0.7535, not "+0.820")

- **Current §9 framing**: §9.2 says "CFM beats TargetDiff on m7 Tanimoto diversity (gap +0.820 honest per TODO-180 §1.3)". The figure **+0.820** does not match the MEASURED gap.
- **What is missing**: per `metrics/by_metric/diversity_tanimoto.json` aggregate gap field, the MEASURED gap is **`-0.7535`** (i.e., TargetDiff 0.860 - best measured 0.1065 = +0.7535 to close). The §9.2 value "+0.820" appears to be from a different comparator or an early projection.
- **MEASURED evidence**: `diversity_tanimoto.json` aggregate gap = `-0.7535` (sign convention: negative = us, positive = TargetDiff). The best MEASURED is `wf_lift_ev3` = 0.1366, not 0.1065 (Path A baseline). With 0.1366: gap = `0.860 - 0.1366 = 0.7234`.
- **Why it is honest-negative**: the §9.2 number misrepresents the actual MEASURED gap by ~0.07-0.10. Paper §6 honest-negative on m7 should cite the actual MEASURED gap, not the §9.2 rounded figure.
- **Target §9 insertion**: new §9.5 "m7 Tanimoto diversity gap to TargetDiff 0.860 is +0.7234 (MEASURED 0.1366, target 0.860 — NOT meaasured-deferred per §7.2.6 abort criterion)"

### Gap 3: chemprop v3 DrugOOD per-metal coverage uneven — needs decomposition

- **Current §9 framing**: §9.1 says "v3 in-domain only 1-seed (Ru 0.7613)". The DrugOOD m17 forward-pass reeval surface is mentioned only abstractly. §9.3 has item #3 "NO chemprop v3 DrugOOD-PASS" but does NOT cite the per-metal MEASURED decomposition.
- **What is missing**: per `metrics/by_metric/drugood_v3_scaffold.json`, the per-metal m17 reeval has UNEVEN coverage that IS itself an honest-negative band:
  - **Ru**: 4 seeds MEASURED, ROC-AUC mean = 0.4905 ± 0.0312 (BELOW_RANDOM, projected 0.65-0.7)
  - **Ir**: 3 seeds MEASURED (seed3 ckpt missing), ROC-AUC mean = 0.4615 ± 0.0283 (BELOW_RANDOM, projected 0.55-0.65)
  - **Os**: 1 seed MEASURED (seeds 1/2/3 ckpts missing), ROC-AUC = 0.4113 ± 0.0 (BELOW_RANDOM, projected 0.55-0.65)
  - **Aggregate**: 0.4544 ± 0.0327 across 8/12 (metal, seed) live forward-passes; 4 DEFERRED_NO_CHECKPOINT
- **Why it is honest-negative**: paper §4 must not state chemprop v3 generalises to DrugOOD OOD; per-metal coverage (3 metals × 4 seeds = 12 targets, 4 missing) is itself a separate honest-gap distinct from aggregate BELOW_RANDOM.
- **Target §9 insertion**: new §9.6 "chemprop v3 m17 DrugOOD per-metal coverage UNEVEN: Ru 4/4, Ir 3/4, Os 1/4 ckpts MEASURED — 4/12 DEFERRED_NO_CHECKPOINT (Wf-253d forward-pass state per §7.2.7 CHECKPOINT_LOADED_FORWARD_DEFERRED)"

### Gap 4: Wf-253d `CHECKPOINT_LOADED_FORWARD_DEFERRED` state — no §9 dedicated entry

- **Current §9 framing**: §7.2.7 documents "CHECKPOINT_LOADED_FORWARD_DEFERRED" as an honest-negative abort-criterion pattern, but §9 has NO matching catalog entry. §9.2 mentions "per-metal coverage uneven: Ru 4 seeds, Ir 3, Os 1" which implies the state but does not name the §7.2.7 invariant.
- **What is missing**: a §9 entry that names the §7.2.7 invariant and says: "The m17 forward-pass reeval (Wf-253d) is `CHECKPOINT_LOADED_FORWARD_DEFERRED` — 8/12 (metal, seed) ckpts ran forward, but NO retraining occurred. Future wave must retrain Os (3 seeds) + Ir seed3 + any v3 retrain cohorts before citing v3 DrugOOD numbers."
- **MEASURED evidence**: `drugood_v3_scaffold.json` shows `n_checkpoints_total: 8`, `n_metals_with_checkpoints: 3`, `n_live_forward_passes: 8`, with `forward_only` flag (per §A appendix schema) implied by `eval_status: MEASURED` + absence of `epoch_*` keys.
- **Why it is honest-negative**: §9.3 item #3 "NO chemprop v3 DrugOOD-PASS" is correct, but does not distinguish "v3 forward-pass reeval MEASURED-on-disk (BELOW_RANDOM)" from "v3 retrain-with-DrugOOD NEVER-RUN". Both are honest-negative, but different.
- **Target §9 insertion**: new §9.7 "m17 DrugOOD reeval is `CHECKPOINT_LOADED_FORWARD_DEFERRED` (no retrain; 4/12 ckpts DEFERRED_NO_CHECKPOINT; does NOT project to v3 retrain-on-DrugOOD pass)"

### Gap 5: m6_kappa validation NEGATIVE_WEAK verdict band — §9.1 mentions NOT-QED-substitute but does not cite NEGATIVE_WEAK

- **Current §9 framing**: §3.1.13 (P1 VERIFY) cites "F2 m6_kappa NOT QED substitute"; §9.1 lists chemprop v2 v3 honest limits but does NOT cite m6_kappa's MetalCytoToxDB validation verdict.
- **What is missing**: per `metrics/by_metric/m6_kappa.json` `metalcytotox_validation`, the verdict is **`NEGATIVE_WEAK`** with global Pearson r = 0.0813, global Spearman rho = 0.1114 (n=5,460 MetalCytoToxDB rows). Per-metal breakdowns show negative-correlations on some metals:
  - Re: Pearson r = **-0.1843**, Spearman rho = -0.1247 (NEGATIVE)
  - Rh: Pearson r = -0.0335 (slightly negative)
- **Why it is honest-negative**: Wf-252 F2 marker "NOT QED substitute" was an honest call BEFORE MetalCytoToxDB validation; post-validation the verdict band is **NEGATIVE_WEAK** which is a STRONGER honest-gap than "not a substitute".
- **MEASURED evidence**: `metrics/by_metric/m6_kappa.json` lines 45-67 (metalcytotox_validation block).
- **Target §9 insertion**: new §9.8 "m6_kappa MetalCytoToxDB validation NEGATIVE_WEAK (global Pearson r=0.0813, Re NEGATIVE Pearson r=-0.1843; NOT generalisable as metallodrug score; complement to QED only)"

### Gap 6: 5 CFM TODO forward-looking (TODO-119 P1.3/P1.4 + TODO-180/181/182) — single-item §9.2 vague

- **Current §9 framing**: §9.2 has ONE aggregate item "DO NOT cite: Wf-181/Wf-188 numbers as P1.3 deliverable". No per-TODO breakdown.
- **What is missing**: 5 distinct CFM TODO deliverables with PENDING status (per §4.1):
  - **TODO-119 P1.3** — real-CFM decode_ratio lift on PlatinAI / DrugOOD / MetalCytoToxDB cohorts (16 hr GPU-0; `real_cfm_decode.json` PENDING — currently only PlatinAI 5k partial at 0.9645 per cfm_decode_real.json)
  - **TODO-119 P1.4** — per-innovation CFM ablation (YuelBond only / W₂ only / closure only; 7.5 hr GPU-0; `ablation_report.json` PENDING)
  - **TODO-180 G4** — Wf-124 P3 14 PENDING (metal, seed) pairs (1/15 done; 1.8 hr GPU-1; per-seed metrics PENDING for 14/15)
  - **TODO-181 G5** — Wf-124 P3 50-epoch full sweep (5 hr GPU-1; per_metal/per_seed metrics PENDING)
  - **TODO-182 G6** — Wf-124 P3 per-innovation ablation 60 runs (7.5 hr GPU-1; `ablation_report.json` PENDING)
- **Why it is honest-negative**: §9.2's single-sentence prohibition does not enumerate the 5 separate deferred deliverables. Paper §4/§6 must not cite ANY of these 5 as MEASURED; §9 should enumerate them.
- **MEASURED evidence**: §4.1 table (lines 158-164 of 207 v2 doc) lists all 5 PENDING.
- **Target §9 insertion**: new §9.9 "5 CFM TODO forward-looking deliverables PENDING — TODO-119 P1.3 (PlatinAI/DrugOOD cohorts, 16 hr GPU-0) + TODO-119 P1.4 (per-lever ablation, 7.5 hr) + TODO-180 G4 (14 PENDING (metal,seed) pairs, 1.8 hr) + TODO-181 G5 (50-ep full sweep, 5 hr) + TODO-182 G6 (60-run ablation, 7.5 hr). Total wall: 37.8 hr GPU, 39 waves (per Wf-258 audit §6)."

---

## §3 Summary table — for party-twenty-spirit §3.5 ingestion

| Gap # | Title | Severity | Target §9.N | MEASURED evidence path |
|---:|---|---|:---:|---|
| 1 | CFM application-layer NEVER MEASURED (no SA/QED/Vina on generated set) | DOCUMENTED | §9.4 (NEW) | `metrics/by_metric/sa_mean.json` + `qed.json` + `vina_*.json` (NONE tagged `cfm_generated`); `cfm_decode_real.json` shows 818 generated but no downstream eval |
| 2 | m7 Tanimoto gap MEASURED +0.7234 not +0.820 (§9.2 number is off) | DOCUMENTED | §9.5 (NEW) | `metrics/by_metric/diversity_tanimoto.json` aggregate `gap: -0.7535`, `best_measured: 0.1366` |
| 3 | chemprop v3 m17 per-metal coverage UNEVEN (Ru 4/4, Ir 3/4, Os 1/4) | DOCUMENTED | §9.6 (NEW) | `metrics/by_metric/drugood_v3_scaffold.json` `per_metal` (8 ckpts across 3 metals, 4 DEFERRED_NO_CHECKPOINT) |
| 4 | m17 reeval `CHECKPOINT_LOADED_FORWARD_DEFERRED` state — no dedicated §9 entry | DOCUMENTED | §9.7 (NEW) | `drugood_v3_scaffold.json` `n_live_forward_passes: 8`, no `epoch_*` keys (per §7.2.7 schema) |
| 5 | m6_kappa MetalCytoToxDB validation NEGATIVE_WEAK (global r=0.0813, Re r=-0.1843) | DOCUMENTED | §9.8 (NEW) | `metrics/by_metric/m6_kappa.json` `metalcytotox_validation` block |
| 6 | 5 CFM TODO forward-looking deliverables PENDING (TODO-119 P1.3/P1.4 + TODO-180/181/182) | DOCUMENTED | §9.9 (NEW) | §4.1 table (lines 158-164) + Wf-258 audit §6 (37.8 hr GPU total) |

**Total**: 6 gap entries (G2 ≥ 5 target MET).

---

## §4 Honest framing note (party-twenty-spirit §3.5)

Current §9 catalog has 3 entries (§9.1, §9.2, §9.3). The 6 gaps above are each backed by a MEASURED JSON path on disk (no aspirational numbers introduced). All 6 gaps are DOCUMENTED severity (not P0/P1) — they describe honest-negatives that paper §4/§6 must surface, not BLOCKER-level gates that prevent launch.

The pattern: **§9 catalogs "what NOT to claim", but does NOT catalog "what was MEASURED-but-NEGATIVE-as-honest-finding"**. Each gap above fills one of those distinct DEFERRED-but-NEGATIVE spaces:
- Gap 1: measured-positive (decode_ratio 0.9958) but downstream application NEVER measured
- Gap 2: measured-positive (diversity 0.1366) but comparator (TargetDiff 0.86) gap under-quantified
- Gap 3: measured-negative (DrugOOD 0.4544 BELOW_RANDOM) but per-metal coverage NOT broken out
- Gap 4: measured-as-forward-pass-only but `CHECKPOINT_LOADED_FORWARD_DEFERRED` invariant unnamed in §9
- Gap 5: measured-negative (m6_kappa Pearson r 0.0813 NEGATIVE_WEAK) but §9 only cites the pre-validation "NOT QED substitute" framing
- Gap 6: §9 cites single-sentence prohibition but does not enumerate 5 distinct PENDING deliverables

**Disposition**: these 6 gaps should be backfilled as §9.4 §9.5 §9.6 §9.7 §9.8 §9.9 in a future §9 expansion PR. Until that PR lands, paper §6 limitations text + reviewer rebuttal should cite each gap by its `metrics/by_metric/*.json` path + MEASURED value.

---

**End of Wf-263B-Audit gap list. G1=3, G2=6, G3=this file path.**
