# Paper §6 Limitations — Draft Scaffold (docs/, NOT paper/)

**Status:** 📝 **DRAFT SCAFFOLD ONLY** — per TODO-84 + AGENTS.md §5.3, ZERO touches
to `paper/sections/*.tex` without explicit USER_OK authorization. This document
is the **pre-prepared source** that lives in `docs/` so that, when USER_OK
arrives, the actual paper §6 write is just a copy-paste + format-conversion.

**Date**: 2026-09-29
**Owner**: Hugo Lee + ultracode agent (Wf-359-B2 dispatched by workflow harness)
**Skill alignment**: `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline)
**Cross-ref**:
- AGENTS.md §5.2 (honest framing mandate)
- AGENTS.md §5.3 (no paper work without authorization, TODO-84)
- AGENTS.md §5.4 (Tier-2 journal target, NO wet-lab planning)
- Wf-357-B3 (Tier-2 honest-negative metrics verification)
- Wf-357-C3 (workflow analysis + 5 DEFERRED CFM/TODO-119 waves)
- Wf-358-C2 (speedup flags + preflight + cli-help cascade fix)
- Wf-359-A1 (m6_kappa W₂ R20-P3 verify; HONEST_NEGATIVE_PARTIAL_STRUCTURAL)
- Wf-359-A2 (m1 multi-seed scaffold for 7 DEFERRED runs)

---

## §0 TL;DR — Honest-negative framing is the §6 posture

Paper §6 Limitations is an **honest-negative disclosure** of every metric
and workflow that did NOT meet its paper-grade gate, NOT an apology. Per
`AGENTS.md` §5.2 + `party-twenty-spirit` §3.5, every numeric value in §6
MUST trace to a JSON / report path on disk; PROJECTED vs MEASURED tag is
required. **Zero tolerance for fabricated / aspirational numbers in
limitations sections** (mirroring §3 standing rule for main results).

This document is organized into 6 sections:

| § | Topic | Honest-negative posture |
|---|---|---|
| §1 | Per-metric (m1-m21) | MEASURED lift vs gate threshold + delta |
| §2 | Per-workflow (Wf-1..358) | DEFERRED / BLOCKED / STATUS_HOLD |
| §3 | Per-launcher (5 wrappers × ~37.5 hr GPU) | USER_OK-gated, pre-flight PASS required |
| §4 | Resolved entries (Wf-344..358) | CLOSED at this submission |
| §5 | Forward-looking (next 90 days) | ETA + external dependency |
| §6 | Cross-references (AGENTS.md, Wf closeouts) | file:line citations |

---

## §1 Per-metric honest-negatives (m1-m21)

### §1.1 m1 — MetalCytotoxDB Ru AUC

| Axis | MEASURED | Gate | Status |
|---|---:|---|---|
| Wf-124 P3 Ru seed=0 30-epoch test_auc | **0.7613** | Krasnov 2026 ≥ 0.86 | **MEASURED_PARTIAL_LIFT** (gate NOT YET HIT, gap −0.099) |
| Wf-131 P1 Ru 3-seed × 10-epoch mean | 0.7352 ± 0.0023 | Krasnov 2026 ≥ 0.86 | **MEASURED** (gate NOT YET HIT, gap −0.125) |
| Wf-355-B1 wrapper for 7 DEFERRED runs | launch wrapper SHIPPED | n/a | **DEFERRED** (USER_OK-gated; 5 hr GPU) |

**Source**: `metrics/by_metric/metalcytotox_ru_auc.json` + Wf-357-B3 §4 +
Wf-359-A2 scaffold (`molmetal/scripts/eval_m1_multi_seed.py`).

### §1.2 m2 — R² ceiling (metalcytotox_combined / platinai_4cell / hilbert)

| Cohort | Best R² (multi-seed CV mean) | Verdict |
|---|---:|---|
| metalcytotox_combined | **+0.2227** (lift +0.1172) | **CEILING_BREAK** (CV-mean lift ≥ +0.10) |
| platinai_4cell | −0.1251 (best) | **HONEST_NEGATIVE_R2_DEEP** |
| hilbert | −0.1268 (best) | **HONEST_NEGATIVE_R2_DEEP** |

**Source**: `molmetal/reports/wf335m2_r2/result.json` per_cohort + Wf-335-SYN §3 +
Wf-357-B3 §3. The platinai_4cell + hilbert per-cell pIC50 noise is the binding
ceiling on those cohorts; remediation requires Multi-PH + PLL per TODO-94 or
Hilbert-ASTSR joint retrain per TODO-93.

### §1.3 m6_kappa — Shape Kappa Index (druglikeness panel)

| Axis | MEASURED | Gate | Status |
|---|---:|---|---|
| Wf-325D baseline mean cohort | **0.4719** (n=449) | ≥ 0.6 druglikeness | **HONEST_NEGATIVE_FAIL** (−0.128 margin) |
| Wf-335-M1 chemprop+PLL CV-R² | **+0.6814** | n/a (structural ceiling diagnostic) | **CEILING_BREAK** |
| Wf-335-M1 augmented_mean | 0.4386 | ≥ 0.6 | **HONEST_NEGATIVE_PARTIAL_STRUCTURAL** (constrained ~baseline by unbiased-CV identity) |
| Wf-348-B1 W₂ on R20-P3 augmented_mean | NOT MEASURED | ≥ 0.6 | gate PRESERVED |
| Wf-348-B3 Hilbert R² on R20-P3 (4-cell) | −1.2335 ± 0.3998 | R² ≥ 0 | **HONEST_NEGATIVE_R2_DEEP** |

**Source**: `metrics/by_metric/m6_kappa.json` + Wf-335-M1 + Wf-348-B3 + Wf-357-B3 §2 +
Wf-359-A1. **§6 framing**: structural ceiling lifted via +0.6814 CV-R² (CEILING_BREAK
on structural axis); augmented_mean gate NOT lifted above 0.6 by chemprop+PLL,
W₂ on R20-P3 features, or Hilbert primal-dual on R20-P3 features.

### §1.4 Other Tier-2 metrics (m3-m21)

| Metric | Status | Honest-negative caveat |
|---|---|---|
| m3 Vina cohort median | **−7.782 kcal/mol** PROJECTED | passes industry −7.0 by 0.782 margin; Wf-333-V3 PROJECTED cohort lift −2.3 kcal/mol on 5-pocket smoke |
| m4 SA / QED sub-criterion | tier-2 shippable | per Wf-315A |
| m7 Multi-PH + PLL diversity | tier-2 shippable | per Wf-325G |
| m15 per-metal reachability | 0.4918-0.6682 (9 d-block) | per Wf-37 P2 bench |
| m18 per-residue protein-metal binding | tier-2 shippable | per Wf-243 |
| m19 λ-CFM coupling | GPU launch BLOCKED | on `decode_ratio=0` per Wf-291D + Wf-299 |
| m21 ASTSR ↔ Hilbert ↔ MCTS triangle | GPU launch BLOCKED | on `hilbert_r2 ≥ 0.6` honesty gate |

---

## §2 Per-workflow honest-negatives (Wf-1..358)

### §2.1 DEFERRED CFM / TODO-119 waves (per Wf-357-C3 §3)

5 GPU launchable items deferred (USER_OK-gated):

| Launch | Wall | Reason DEFERRED | Wrapper |
|---|---:|---|---|
| TODO-181 G5 (Ru seed 2 + Ir seed 0; 5-metal × 3-seed × 50-epoch) | ~5 hr | G5 retrain in flight; paused at PID 2040025 | `scripts/launch_wf124_p3_g5.sh` |
| TODO-182 G6 (per-innovation ablation 2^3 × 50-epoch) | ~7.5 hr | G6 launch wrapper READY; awaiting USER_OK | `scripts/launch_wf124_p3_g6.sh` (with `--no-cudagraphs` per Wf-335-R3) |
| TODO-119 P1.4 (chemprop v3 LOMO) | ~3 hr | CPU scaffold VERIFIED per Wf-336-X2; GPU launch BLOCKED on USER_OK | `scripts/launch_wf119_p14_chemprop_v3_lomo.sh` (NEW) |
| Wf-299 CFM 3-wave (P1.4 + G6 + G6 full) | ~9.5 hr | sequential; launch wrapper READY; awaiting USER_OK | `scripts/launch_wf299_cfm_main.sh` |
| Wf-264 CFM property panel (SA/QED/diversity) | TBD | launch wrapper SHIPPED per Wf-334-U3 | `scripts/launch_wf264_cfm_property_panel.sh` (NEW) |

### §2.2 DEFERRED Wf-131 P3 runs (7 entries)

7 (metal, seed) pairs DEFERRED (per Wf-355-B1 §1.2):

| # | Metal | Seed | Reason |
|:---:|:---:|:---:|---|
| 1-2 | Rh | 3, 4 | training stopped at epoch 23; one never started |
| 3-7 | Re | 0, 1, 2, 3, 4 | never started |

**Aggregate gap**: Wf-131 P3 LOMO 18/25 seeds present; 7/25 DEFERRED.
Multi-seed mean ± std paper-grade number requires 25/25.

### §2.3 DEFERRED Wf-146 runs (6 entries)

Per Wf-146 LOMO sweep log (`molmetal/reports/wf146_chemprop_lomo/sweep.log`):
19/25 done; 6 (metal, seed) pairs DEFERRED. Per Wf-348 / Wf-355 roadmap,
Wf-146 closure requires USER_OK-gated 5-seed × 50-epoch chemprop v2 retrain.

### §2.4 Other deferred workflows

| Workflow | Reason DEFERRED | Status |
|---|---|---|
| Wf-298 R20 §4 BindingDB integration | **DATA-HOLD** (BindingDB external data not refreshed in scope) | STATUS_HOLD |
| Wf-298 R20 §5 tmQM integration | **STATUS_HOLD** (tmQM CPU path incomplete; pending TODO-94 R20-P3 features) | STATUS_HOLD |
| Wf-74 v2 DrugOOD DANN | launch wrapper SHIPPED per Wf-336-X3; USER_OK-gated | DEFERRED |
| TODO-134 Vina meeko alt path | SHIPPED per Wf-336-X4 (CPU-only); GPU launch READY (5-7 hr) | DEFERRED |
| TODO-21 λ-CFM GPU launch | BLOCKED on `decode_ratio=0` per Wf-291D + Wf-299 | BLOCKED |
| TODO-93 Hilbert-ASTSR GPU retrain | launch wrapper SHIPPED per Wf-333-G4 (pre_featurize_dmpnn); USER_OK-gated | DEFERRED |
| TODO-94 Multi-PH + PLL feature extraction | launch wrapper SHIPPED per Wf-333-G2; USER_OK-gated | DEFERRED |

---

## §3 Per-launcher honest-negatives (5 wrappers × ~37.5 hr GPU total)

Per Wf-358-C2 §3 + Wf-357-C3 §3:

| Wrapper | Wall | USER_OK-gated? | Pre-flight PASS? |
|---|---:|:---:|:---:|
| `scripts/launch_wf124_p3_g5.sh` | ~5 hr | YES | YES (62/62) |
| `scripts/launch_wf124_p3_g6.sh --no-cudagraphs` | ~7.5 hr | YES | YES (62/62) |
| `scripts/launch_wf119_p14_chemprop_v3_lomo.sh` | ~3 hr | YES | YES (62/62) |
| `scripts/launch_wf299_cfm_main.sh` | ~9.5 hr | YES | YES (62/62) |
| `scripts/launch_wf264_cfm_property_panel.sh` | ~3 hr | YES | YES (62/62) |
| `scripts/launch_wf131_p3_deferred_runs.sh` | ~5 hr | YES | YES (62/62) |
| `scripts/launch_wf74_v2_drugood_dann.sh` | ~3 hr | YES | YES (62/62) |
| `scripts/launch_todo93_hilbert_astsr_gpu.sh` | ~3 hr | YES | YES (62/62) |
| `scripts/launch_todo94_multiph_pll_gpu.sh` | ~3 hr | YES | YES (62/62) |
| `scripts/launch_todo134_vina_meeko_alt_gpu.sh` | ~5-7 hr | YES | YES (62/62) |
| **TOTAL** | **~47 hr GPU** | | |

**Per `AGENTS.md` §5.1 standing GPU authorization** (2026-09-29 user grant
"显存够就放心大胆用"): no per-launch user OK required going forward; only
VRAM-budgeted check + pre-flight PASS + config-aligned hyperparameters.

---

## §4 Resolved entries (Wf-344 + Wf-345 + Wf-346 + Wf-356 + Wf-358 closed)

### §4.1 Wf-358 ultracode residual wave (CLI help + STALE preflight + speedup flags)

| Issue | Wf | Status |
|---|---|---|
| CLI help `TypeError: must be real number, not dict` for `+~30%` argparse help text | **Wf-358-A1** | **CLOSED** (regression test `test_wf358a1_cli_help_fix.py` PASS) |
| STALE pre-flight JSON refresh (25.21 h → 0.0008 h) | **Wf-358-A2** | **CLOSED** (2 NEW tests PASS) |
| 3 NEW speedup CLI flags wired (`--gradient-checkpointing`, `--channels-last`, `--torch-compile-dynamic`) | **Wf-358-A3** | **CLOSED** (per `train_dmpnn_multitask_wasserstein.py`) |

### §4.2 Wf-344 (m6_kappa ADMET honest-negative close)

Per `TODO/completed/344b1_admet_honest_negative.md`: ADMET 4 below threshold
deferred to Wf-356-A1/A2.

### §4.3 Wf-345 (scaffold zero-leak fix + hyperparam config + 4 anchor refresh)

Per `TODO/completed/345_final_synthesis.md`: scaffold zero-leak fix
(`molmetal/scripts/bench_chemprop_v2_v3.py` MAY_DISPATCH fix), hyperparam
config complete, 4 anchor refresh (Wf-121 + Wf-143 + Wf-170 + 4th
anchor) shipped. CLI docstring dedup + dynamic count wired.

### §4.4 Wf-346 (registration drift fix)

Per Wf-346-B1 pipeline drift audit + Wf-346-B2 orphan gate audit +
Wf-346-B3 constants runtime compare + Wf-343c4 cascade: registration drift
between `pre_flight_check.py` constants and actual gates FIXED.

### §4.5 Wf-356 (Tier-2 honest-negative catalog + 4 stale anchors)

Per Wf-356-A2: 4 stale anchors refreshed (Wf-121 + Wf-143 + Wf-170 + 4th
anchor) so pre-flight v3 24-h cap holds. Per Wf-356-A3: Zotero + tectonic
install README pending; USER_OK-gated.

---

## §5 Forward-looking honest-negatives (next 90 days)

### §5.1 USER_OK-gated GPU launches (10 wrappers × ~47 hr total)

Per `AGENTS.md` §5.1 + §3 standing GPU authorization. Wall-budgeted
launches per §3 above. No per-launch USER_OK needed (standing grant),
but VRAM check + pre-flight PASS + config-aligned hyperparameters
required.

### §5.2 Zotero + tectonic install (per Wf-356-A3)

Zotero (AGPL-3.0) + Better BibTeX (MIT) + tectonic (LaTeX engine)
binary install on host. USER_OK-gated per Wf-351-A2 audit. CSL
styles for Mol. Divers. SHIPPED at `.tools/sci-paper/styles/mol-divers.csl`.

### §5.3 Tier-2 journal submission package (TODO-138 v2)

Per AGENTS.md §5.4: paper target downgraded from JCIM Tier-1 to Tier-2.
Submission package TEMPLATE_READY; actual journal portal submission
USER_OK-gated.

### §5.4 GPU Blackwell sm_120 retune (Wf-326E follow-up B)

24 kernels × 117 configs per-kernel autotune; 2.46× speedup vs generic 288
PROJECTED. Full sm_120 GPU retune DEFERRED per `AGENTS.md` §5.1 (VRAM +
pre-flight check required before launch).

### §5.5 Bond_fdim 6→10 schema gap (Wf-333-G4 honest-negative)

`pre_featurize_dmpnn` hook wired into basic D-MPNN; `cache` ctor param +
`cache_hit_rate` property SHIPPED. Bond_fdim 6→10 schema gap surfaced
as honest-negative requiring dedicated workflow.

---

## §6 Cross-references

### §6.1 AGENTS.md standing directives honored

- **§5.1 Standing GPU authorization** (2026-09-29 user grant) — every
  §3 launcher is VRAM-budgeted + pre-flight PASSED + config-aligned.
- **§5.2 Honest framing mandate** — every numeric value in §1-§5 traces
  to a JSON / report path on disk.
- **§5.3 No paper work without authorization** (TODO-84 active) — this
  draft lives in `docs/`, NOT `paper/`. The actual paper §6 write to
  `paper/sections/06_limitations_v3.tex` requires USER_OK.
- **§5.4 Tier-2 journal target + NO wet-lab planning** — every entry in
  §1-§5 is computational-only; no wet-lab external dependency.

### §6.2 Wf closeout references

- `TODO/completed/357b3_tier2_honest_neg_metrics.md` (Wf-357-B3 §6
  Limitations source for m6_kappa + m2 + m1)
- `TODO/completed/357c3_workflow_analysis.md` (Wf-357-C3 §3 per-workflow
  honest-negatives + 5 DEFERRED CFM/TODO-119 waves)
- `TODO/completed/358a1_cli_help_fix.md` + `358a2_preflight_refresh.md`
  + `358a3_speedup_flags.md` (Wf-358 resolved entries §4)
- `TODO/completed/359a1_m6_kappa_w2_r20p3_verify.md` + `359a2_m1_multi_seed_scaffold.md`
  + `359a3_m2_m3_m15_reverify.md` (Wf-359 forward-looking entries)
- `TODO/completed/335_final_synthesis.md` (Wf-335-SYN §3 + §7 for
  platinai_4cell + hilbert HONEST_NEGATIVE_R2_DEEP)
- `TODO/completed/348b3_hilbert_r2_gate.md` (Wf-348-B3 Hilbert R² on
  R20-P3 features, aggregate −1.23)

### §6.3 Plan + skill references

- `TODO/03_plans/WEAK_TO_STRONG_PLAN.md` §3.5 (PROJECTED CV-R² band +0.05
  to +0.15; MEASURED +0.6814 on m6_kappa CEILING_BREAK)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §1.3 L5 (Hilbert R² honest-negative
  gate spec) + §2.4 R21-P4 (triangle closure GPU launch BLOCKED on
  `hilbert_r2 ≥ 0.6` honesty gate)
- `TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md` (chemprop v2 / W₂ /
  Multi-PH/PLL remediation axes)
- `.claude/skills/party-twenty-spirit` §3.5 (0-tolerance honest framing)
- `.claude/skills/mol-metal-paper-writing` (READ-only on `paper/*`)

### §6.4 Source directories

- `metrics/by_metric/m6_kappa.json` — Wf-325D canonical baseline
- `metrics/by_metric/metalcytotox_combined_r2.json` — Wf-87 Phase 2 aggregator
- `metrics/by_metric/platinai_combined_r2.json` — Wf-87 Phase 2 aggregator
- `metrics/by_metric/metalcytotox_ru_auc.json` — Wf-124 P3 anchor
- `molmetal/reports/wf335m1_chemprop_pll/result.json` — m6_kappa ceiling lift
- `molmetal/reports/wf335m2_r2/result.json` — m2 R² per-cohort breakdown
- `molmetal/reports/wf348b3_hilbert_r20p3/r2_results.json` — R20-P3 Hilbert R²
- `molmetal/reports/wf131_p3_lomo_full/` — 18/25 seeds Wf-131 P3 anchor
- `molmetal/reports/wf359a2_m1_multi_seed_scaffold/aggregate.json` — m1 multi-seed scaffold

---

## §7 Next steps for paper §6 write

When USER_OK arrives per TODO-84:

1. Read this draft scaffold at `docs/PAPER_SECTION_6_LIMITATIONS_DRAFT.md`
2. Convert §1-§5 prose into LaTeX `\section{Limitations}` blocks for
   `paper/sections/06_limitations_v4.tex`
3. Embed §6.4 source JSON paths as `\texttt{file:line}` citations
4. Add Tier-2 acceptance context per AGENTS.md §5.4 (computational-only,
   no wet-lab required)
5. Cross-link to `paper/sections/04_5_honest_negative.tex` (existing
   honest-negative section)
6. Validate via `latexmk` + `chktex` before commit
7. Update `paper/SUBMISSION_CHECKLIST.md` §6 Limitations row

**Until USER_OK arrives**: zero touches to `paper/sections/*.tex`. This
draft scaffold is the institutional pre-preparation per `AGENTS.md` §5.3.

---

**END OF §6 LIMITATIONS DRAFT SCAFFOLD** — Wf-359-B2 SHIPPED. ~290 LOC.
This doc lives in `docs/` (NOT `paper/`) per TODO-84 + AGENTS.md §5.3.
Pre-prepared for USER_OK-gated paper §6 write.