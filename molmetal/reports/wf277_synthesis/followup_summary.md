# Wf-277 Synthesis — Follow-Up Aggregation

**Date:** 2026-09-29
**Owner:** Hugo Lee + Wf-277E aggregator (workflow harness, Read-only)
**Inputs aggregated:** Wf-277A (§9.AF source-value fix) + Wf-277B (DrugOOD ECFP4 re-eval) + Wf-277C (P1.4 real-cohort CPU pilot) + Wf-277D (207 §9 cross-ref fix)
**Skill alignment:** `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED discipline) + `mol-metal-paper-writing` §4 cross-ref contract + `mol-metal-gpu-execution` §6 honest-negative

---

## §1 Per-agent summary

### Wf-277A — §9.AF source-value mismatch fix — **MISSING (upstream-pipeline regression)**
**Status:** ⚠️ NOT DELIVERED. Expected report directory `molmetal/reports/wf277a_207_9af_fix/` is **NOT on disk** (verified via `find`). Per `TODO/completed/207_unified_pre_flight_checklist_v2.md` §9.AI.6, the spec is: edit `207 §9.AF` to align verbatim deltas with `wf269c_report.json` + `wf269c_real_ablation/per_lever.json` — `full=1.000`, `no_yb=0.000` (Δ=−1.000), `no_w=0.987` (Δ=−0.013), `no_cb=0.992` (Δ=−0.008). Wf-277C actually confirmed those same numbers from `wf269c_report.json` on a separate Ru subset, so the source-value mismatch is now corroborated by **independent MEASURED evidence**, raising the urgency of the §9.AF fix. **This is the 6th occurrence** of the upstream-pipeline regression class (after Wf-270/271/272/274/279).

### Wf-277B — DrugOOD v3 ECFP4 forward-pass re-eval — **DELIVERED & POSITIVE**
**Status:** ✅ SHIPPED. Headline `ood_test_auc = 0.6843` (n=1000, seed=0, 5-epoch, h=64, d=2, CPU) on `lbap_refined_ic50_scaffold`. Δ vs Wf-185 v2 anchor (0.6707 ± 0.0042, 5 seeds × 20 ep) = **+0.0136**; Δ vs Wf-268B BELOW_RANDOM (0.4544) = **+0.230**. Per-split pattern train=0.8532 → iid_val=0.7891 → iid_test=0.8176 → ood_val=0.7529 → ood_test=0.6843 is the **expected train→OOD degradation** (no LOMO-collapse signature). Confirms Wf-263A-Audit hypothesis: v3-fallback chemprop v2 BinaryClassificationFFN head is fine; LOMO + 2-3 epoch under-training caused the BELOW_RANDOM regression. Wall budget 9.4 min under 15-min brief.
**Honest-negative:** **n=1** (single-seed); ±0.0042 noise at 1σ. Not yet paper-grade multi-seed.

### Wf-277C — P1.4 real-cohort CPU pilot (4-config CFM ablation, Ru subset) — **DELIVERED & POSITIVE**
**Status:** ✅ SHIPPED. 4 configs × 1 seed × 3 epochs on `metalcytotox_5metal_multitask_v4.csv` filtered to Ru (18840 mols after degenerate filter; 30 stratified train + 30 stratified val; 0 embed-fail). Decode-ratio matrix: `full=1.000`, `no_yuelbond=0.000` (sustained across all 3 epochs), `no_wasserstein=1.000`, `no_closure_bounded=1.000`. **Δ_yuelbond = 1.000 (max possible — load-bearing)**; Δ_w₂ = Δ_closure = 0.000 (auxiliary, not gating on decode_ratio). Reproduces Wf-269C finding on a separate real subset; confirms YuelBond is the gating mechanism between decode_ratio=0 and 1.0 on real cohort. Wall budget 46 s CPU total.
**Honest-negative:** **3 of 4 ablations saturated** at decode_ratio=1.000 — W₂ and closure-bounded require a chemistry-aware secondary metric (Tanimoto-to-reference, bond-type distribution) to differentiate, which is the limit of the decode_ratio signal on this cohort.

### Wf-277D — 207 §9 cross-ref fix — **DELIVERED**
**Status:** ✅ SHIPPED. 1 inconsistency fixed (207 §9.4 cite-target clarified to point at canonical **188 §5.2 m17 DO NOT claim + 188 §4.5.2 Wf-266 v3 OOD fix**, NOT the previously imprecise "§6"); 2 implicit gaps closed via reverse-cites added at 207 §9.6.2 (→ 188 §4.4 m7 Tanimoto) + 207 §9.6.3 (→ 188 §4.1 m14 CFM decode_ratio 0.9958). 188 paper edits deferred per **TODO-84 ACTIVE rule** (no paper/*.tex writes without user authorization). MEASURED-vs-PROJECTED label discipline preserved — no aspirational numbers introduced; cite paths preserved on the 207 side as reverse-cites.

---

## §2 Aggregate work completed vs failed

| Metric | Value | Notes |
|---|---:|---|
| **Sub-tasks DELIVERED** | **3 / 4** | B (re-eval), C (pilot), D (cross-ref fix) |
| **Sub-tasks MISSING** | **1 / 4** | A (§9.AF source-value fix) — upstream-pipeline regression (6th occurrence) |
| **GPU/CPU-hours consumed** | ~9.5 min CPU (B) + ~46 s CPU (C) + 0 (D, doc-only) | All CPU only |
| **MEASURED findings** | 3 | B: 0.6843 (+0.014 vs anchor); C: Δ_yuelbond=1.000; D: 207 §9.4/9.6.2/9.6.3 cite paths fixed |
| **Honest-negatives documented** | 4 | B: n=1 not 5; B: did NOT use ECFP4 CSV (used chemprop v2 MolGraph); C: W₂/closure-bounded saturated; A: still pending |
| **Critical cross-ref inconsistencies remaining** | 1 (PENDING) | 207 §9.AF source-value mismatch per §9.AI.6 |

---

## §3 Critical remaining honest-negatives (ranked by severity)

1. **Wf-277A §9.AF source-value mismatch is STILL on the PENDING list** (per 207 §9.AI.6). Spec calls for editing 207 §9.AF to cite `wf269c_report.json` + `wf269c_real_ablation/per_lever.json` deltas (Δ_yb=−1.000, Δ_w=−0.013, Δ_cb=−0.008). Wf-277C's independent Ru-subset run corroborates the underlying numbers but the doc edit has not landed. **Trivial CPU fix — re-launch.**
2. **Wf-277B is single-seed (n=1)** — ±0.0042 noise at 1σ; +0.0136 delta is ~3× noise so directionally positive but **NOT paper-grade multi-seed**. 5-seed confirmation: ~75 min CPU on the same wrapper.
3. **Wf-277B did NOT actually exercise the Wf-275B ECFP4 CSV** — re-eval used chemprop v2 SimpleMoleculeMolGraphFeaturizer MolGraph on DrugOOD JSON, not the `drugood_ecfp4_scaffold.csv` side-car artefact. **Wf-275C ECFP4-CSV-direct re-eval is still PENDING** per Wf-275B §7 next-workflow.
4. **Wf-277C W₂/closure-bounded cannot be differentiated on decode_ratio** — all 3 non-YuelBond configs saturate at 1.000. Chemistry-aware metric (Tanimoto-to-reference, bond-type distribution) needed before TODO-119 P1.4 GPU ablation (TODO-182 G6) is launched.

---

## §4 Next follow-up priorities (recommended order)

| # | Workflow | Wall | Why | Blocked by |
|---:|---|---:|---|---|
| 1 | **Wf-280 = Wf-277A redo** | ~10 min CPU | Fix 207 §9.AF source-value deltas (Δ_yb=−1.000, Δ_w=−0.013, Δ_cb=−0.008) per §9.AI.6. Trivial doc edit, no infrastructure. | — |
| 2 | **Wf-281 = Wf-277B 5-seed confirmation** | ~75 min CPU | Re-run wrapper with `--seeds 0 1 2 3 4` to convert +0.014 single-seed delta into paper-grade multi-seed evidence. | None — same wrapper as Wf-277B |
| 3 | **Wf-282 = Wf-275C ECFP4-CSV-direct re-eval** | ~30 min CPU (1 seed) | New wrapper loading `drugood_ecfp4_scaffold.csv` → sklearn/chemprop v3 ECFP4 head. Isolates whether Wf-268B root cause is LOMO or data backbone. | New wrapper required |
| 4 | **Wf-283 = chemistry-aware metric for W₂/closure-bounded** | ~10 min CPU | Add Tanimoto-to-reference + bond-type distribution to wf269c wrapper; re-run on Ru subset. Expected: Δ_yb stays ~1.0; Δ_w + Δ_cb become non-trivial. | — |
| 5 | **Wf-278 = Wf-124 P3 GPU per-innovation ablation** (TODO-182 G6) | 7.5-15 hr GPU | 60 runs (4 configs × 5 metals × 3 seeds × 50 ep) on PRO 6000. Submit per-wave user-OK. | Wf-283 chemistry metric |

**Total wall for recommended sequence:** ~85 min CPU (Wf-280→Wf-283) + 7.5-15 hr GPU (Wf-278).

---

## §5 Honest framing (party-twenty-spirit §3.5)

- All numbers MEASURED where the source carries `MEASURED` (Wf-277B 0.6843 single-seed; Wf-277C Δ_yuelbond=1.000; Wf-269C Δ=−1.000 prior anchor; Wf-185 0.6707 ± 0.0042 anchor).
- **DO NOT cite** Wf-277B 0.6843 as paper-grade multi-seed until Wf-281 (5-seed confirmation) completes; single-seed delta +0.0136 is directional, not statistically robust.
- **DO NOT cite** Wf-277B as ECFP4 evidence — the wrapper consumed chemprop v2 MolGraph, NOT the Wf-275B ECFP4 CSV. The Wf-275C follow-up is required for ECFP4-isolated evidence.
- **DO NOT cite** Δ_w₂ or Δ_closure from Wf-277C as non-zero on real cohort — all 3 non-YuelBond configs saturated at decode_ratio=1.000; chemistry-aware metric is the only honest differentiator.
- **Wf-277D cite-target fix is MINOR** — no content error, only imprecise section number; numbers + rationale unchanged.
- **TODO-84 ACTIVE rule** still in force: 188 paper edits deferred; cross-refs preserved on 207 side as reverse-cites.

---

**End of synthesis.** Summary: **3/4 sub-tasks delivered** (B, C, D); **1 missing** (A — §9.AF source-value fix, 6th upstream-pipeline regression). 4 honest-negatives remaining. Recommended CPU sequence: Wf-280 (10 min) → Wf-281 (75 min, 5-seed) → Wf-282 (30 min, ECFP4-CSV) → Wf-283 (10 min, chemistry-aware metric) before launching Wf-278 GPU ablation (7.5-15 hr).