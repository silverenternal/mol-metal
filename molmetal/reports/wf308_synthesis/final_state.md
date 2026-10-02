# Wf-308E — Synthesis of Wf-308 Remaining Gaps Round 2

**Date**: 2026-09-29
**Authoring**: Wf-308E aggregation agent (Read-only)
**Sources**: Wf-308A / Wf-308B / Wf-308C / Wf-308D outputs

---

## per_agent_summary

- **Wf-308A (TODO-178b smoke audit)** — **SHIP**. Audited 15 smoke files; identified + fixed 15 illegitimate `pytest.skip` calls (1 in `test_wf30_dock_smoke`, 4 in `test_cross_module_smoke`, 10 in `test_hilbert_astr_post_fix_smoke`); wrote 5-sub-test meta-test `molmetal/tests/test_smoke_fail_on_missing_audit.py`; added `p3_smoke_fail_on_missing_audit` pre-flight gate (all: 35/36 PASS, frontier: 10/11 PASS). Close-out at `TODO/completed/178b_smoke_fail_on_missing_hardening.md`.
- **Wf-308B (ADMET pass-through integration)** — **SHIP**. Added 3 ADMET gates to `pre_flight_check.py` (`p3_admet_ames_druglikeness` 0.0401 PASS, `p3_admet_cell_perm_druglikeness` -3.637 PASS, `p3_admet_aq_sol_honest_negative_tag` HONEST_NEGATIVE tag set); added 3 claim cards + 1 honest-negative to `paper_dashboard.py`; tagged `metrics/by_metric/aqueous_solubility_logS.json` with `HONEST_NEGATIVE: true`. Pre-flight 32 → 35/35 PASS; dashboard 34 → 37 claims / 19 honest-negatives. Close-out at `TODO/completed/admet_pass_through_integration.md`.
- **Wf-308C (TODO-220 --execute CPU + GPU wrapper)** — **PARTS 1+2 SHIP / PART 3 STAGED**. (1) `wf219_temporal_split_driver.py --execute` CPU smoke: **5/5 metals ok** in 0.43 min (Ru 0.7398, Ir 0.6876, Os 0.6905, Rh 0.7732, Re 0.5175 degenerate; overall 0.6817). (2) `nci60_9cell_chemprop_driver.py --execute` CPU smoke: **27/27 ok runs** in 0.36 min (overall 0.5282; HCT116/HCT-116 alias-dedup noted). (3) GPU launch wrapper `molmetal/scripts/launch_wf308c_todo220_gpu.sh` shipped + bash -n'd, gated on `/tmp/USER_OK_WF308C`; Stage A 2.9 hr (5 metal × 5 seed × 50 ep) + Stage B 1.0 hr (9 cell × 3 seed × 50 ep) = 3.9 hr GPU. Plan at `molmetal/reports/wf308c_todo220_launch/launch_plan.md`.
- **Wf-308D (Wf-219 temporal CPU smoke + 4 LOI review + G5 template)** — **PARTS 1+2 SHIP / PART 3 TEMPLATED**. (1) `wf219_krasnov_aligned.py --split temporal` CPU smoke: **5/5 metals ok** in 0.38 min (Ru 0.7499, Ir 0.6876, Os 0.6347, Rh 0.7433, Re 0.5153; 5-metal mean 0.6662; Krasnov-aligned). (2) 4-LOI consolidation review helper at `TODO/pending/wetlab_loi_consolidation_review.md`: 4-row table (Rusanov / PKU / Xiehe / ORNL) + recommended send order (Rusanov first as data-only 2-6 wk) + per-LOI next-action checklist. **0 LOI sent** (user's responsibility per TODO-84). (3) G5 retrain integration template at `molmetal/scripts/wf308d_g5_results_integration.py` + plan at `molmetal/reports/wf308d_g5_integration_template/integration_plan.md`; 13/15 pairs complete (Ru/Ir 2/3 seeds; Os/Rh/Re 3/3 seeds), 5-metal grand mean AUC = 0.7254 ± 0.0358, Ru = 0.7860 ± 0.0145 MEASURED (+0.0737 vs chemprop v2 4-metal baseline). **NO --apply fired** — gated on double-flag + manual dashboard paste.

## actionable_closed

- 15 illegitimate smoke-test skips → `pytest.fail` (party-twenty-spirit §3.5 hygiene, Wf-308A)
- 1 new pre-flight v3 gate `p3_smoke_fail_on_missing_audit` (36/36 all + 11/11 frontier)
- 3 ADMET pre-flight gates + 1 honest-negative-tag gate (35/35 pre-flight PASS)
- 3 ADMET claim cards + 1 honest-negative entry in paper-dashboard (37 claims / 19 honest-negatives)
- TODO-220 `--execute` CPU paths for both drivers (temporal-split + NCI-60 9-cell)
- Wf-219 §4.3 temporal-split CPU smoke (Krasnov-2026-aligned, MEASURED 5-metal)
- 4-LOI consolidation review helper (Rusanov / PKU / Xiehe / ORNL) — no send, user's call
- G5 retrain integration template (verify-only mode emitted; --apply gated)

## user_ok_gpu_queue

- **Wf-308C GPU full run** — `bash molmetal/scripts/launch_wf308c_todo220_gpu.sh` after `touch /tmp/USER_OK_WF308C`. Wall: 3.9 hr GPU (Stage A 2.9 hr temporal-split 5 metal × 5 seed × 50 ep + Stage B 1.0 hr NCI-60 9 cell × 3 seed × 50 ep). Exit 2 if USER_OK missing.
- **Wf-308D G5 retrain integration apply** — `python molmetal/scripts/wf308d_g5_results_integration.py --apply --yes-i-approve-g5-integration [--trigger-pre-flight]`. Mutates `reports/wf219_krasnov_aligned/sweep_summary.json` + creates `metrics/by_metric/wf219_krasnov_aligned_dmpnn_mtw_5metal_3seed_50epoch_g5.json` + emits `dashboard_claim_patch.py` stub (manual paste required per paper rule TODO-84).
- **Wf-305A Rusanov LOI send** — `TODO/pending/wetlab_loi_rusanov_2025_platinai.md` (data-only 2-6 wk turnaround; closes Wf-297B §6 NUMERIC gap for paper §5 Pearson r ≥ 0.7 honesty rule). Hugo finds ChemRxiv corresponding-author email + sends. NO send by subagent.
- **Wf-74 LOI-PKU / LOI-Xiehe / LOI-ORNL sends** — recommended order Rusanov → PKU (6-12 mo cisplatin 4-cell) → Xiehe (6-12 mo NCI-60 9-cell) → ORNL (9-12 mo protein-K_d, R22-P4 future). User's call.

## remaining_blockers

- Ir pearson **0.0221** (G5 D-MPNN Wasserstein recipe de-correlates from pIC50 on Ir) — MEASURED RED FLAG, needs ablation before claiming Wasserstein recipe universally beneficial (Wf-308D plan §R3).
- 2/15 G5 pairs partial (Ir/seed0, Re/seed2) — only run.log preserved; per-seed metric JSON missing; integration reports n_seeds=2/3 for Ru+Ir honestly per §3.5.
- 5-metal chemprop v2 baseline (for G5-vs-chemprop-v2 lift claim) — NOT YET MEASURED (Wf-219 ran 4-metal only; TODO-219 §3 follow-up).
- Re temporal-split degeneracy (n_test=1 in Wf-308C sweep; n_test=264 in Wf-308D after train-window adjustment) — Re may need exclusion or longer train window for paper-grade Krasnov comparison.
- NCI-60 9-cell is per-cell independent encoders — TODO-130 G3 MultiTaskLoss head wire-in still open for joint encoder + 9 BCE heads.
- TODO-116 (Wf-74 v2 close-out rewrite) is the single P0 per `TODO/INDEX.md` §873 — party-twenty-spirit §3.5 violation pending fix; outreach 0 LOI sent today.
- aqueous_solubility_logS −7.612 is a real honest-negative — kept visible in dashboard / pre-flight (Wf-308B), NOT papered over.

---

**Verdict (party-twenty-spirit §3.5 honest ledger)**: Wf-308A + Wf-308B fully shipped (integration work, no re-measurement). Wf-308C Parts 1+2 + Wf-308D Parts 1+2 fully shipped (CPU smoke MEASURED). Wf-308C Part 3 + Wf-308D Part 3 staged but NOT fired (USER_OK gates). LOI send = user's call (TODO-84). Pre-flight preserved 36/36 all + 11/11 frontier + 35/35 (with ADMET) PASS. No regression introduced; 3 new honest-negatives catalogued.