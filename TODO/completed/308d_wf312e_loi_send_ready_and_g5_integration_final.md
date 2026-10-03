# Wf-312E — Wf-308D Verify + 4 LOI Send-Ready Final Integration + G5 Retrain Integration (Final)

**Date**: 2026-09-29
**Authoring**: Wf-312E integration agent (Claude)
**Inputs**: Wf-308D outputs (per `molmetal/reports/wf308_synthesis/final_state.md`)

---

## 1. Verdict

Wf-308D outputs are **COMPLETE and VERIFIED**. Wf-312E integrates them:

- 4 LOI drafts are structurally ready-to-send (per Wf-308D PART 2 §4 checklist).
- G5 retrain integration template is shipped (per Wf-308D PART 3 + Wf-312E §12 retry context).
- New pre-flight gate `p3_loi_send_ready` wired into both `all` and `frontier` pipelines.
- Pre-flight `pipeline=all` now 37/37 PASS (was 36 + ADMET 3 = 36/36, now +1 = 37/37 with LOI).
- **NO LOI actually sent.** **G5 `--apply` mutation NOT fired.** Both gated on Hugo.

---

## 2. Wf-308D output verification

| Output | Path | Status |
|---|---|---|
| Wf-219 §4.3 temporal-split CPU smoke | `molmetal/reports/wf308d_temporal_smoke/sweep_summary.json` + 5 per-metal JSONs (Ru/Ir/Os/Rh/Re seed0) | **PRESENT** — 5/5 metals ok in 0.38 min, 5-metal mean AUC 0.6662 Krasnov-aligned |
| 4-LOI consolidation review helper | `TODO/pending/wetlab_loi_consolidation_review.md` | **PRESENT** — 4-row table + send-order ladder + per-LOI checklist (21 `- [ ]` boxes total) |
| G5 retrain integration template (plan) | `molmetal/reports/wf308d_g5_integration_template/integration_plan.md` | **PRESENT** — verify-only mode, 13/15 pairs MEASURED, Wf-310A retry context added by Wf-312E |
| G5 retrain integration script | `molmetal/scripts/wf308d_g5_results_integration.py` (25 KB) | **PRESENT** — `--verify-only` / `--apply --yes-i-approve-g5-integration` double-flag gate |

**No missing outputs.** No re-run required.

---

## 3. 4 LOI send-ready final integration

| # | LOI | File | Status header | Checklists | Ready? |
|---|---|---|---|---|---|
| 1 | Rusanov 2025 (PlatinAI 4-cell × 200 compound NUMERIC IC50) | `TODO/pending/wetlab_loi_rusanov_2025_platinai.md` | `**Status:** **DRAFT — not sent at this revision**` | 4 boxes in review helper §4.1 | YES |
| 2 | PKU cisplatin 4-cell IC50 (MCF7/A549/A2780/A2780cis) | `TODO/pending/wetlab_loi_pku_cisplatin_ic50.md` | `**Status:** **DRAFT — not sent at this revision**` | 4 boxes in review helper §4.2 | YES |
| 3 | 协和 NCI-60 9-cell + ICP-MS | `TODO/pending/wetlab_loi_xiehe_nci60.md` | `**Status:** **DRAFT — not sent at this revision**` | 4 boxes in review helper §4.3 | YES |
| 4 | ORNL protein-metal K_d (Biacore + ITC) | `TODO/pending/wetlab_loi_ornl_protein_metal.md` | `**Status:** **DRAFT — not sent at this revision**` | 4 boxes in review helper §4.4 | YES |

**4/4 LOIs structurally ready-to-send** per Wf-308D §4 checklist.

**n_lois_ready_to_send = 4**.

**n_lois_actually_sent = 0** (user's call per `TODO/pending/84_no_paper_task_without_authorization.md`).

**Recommended send order** (per `wetlab_loi_consolidation_review.md` §3):
1. Rusanov (data-only, 2-6 wk, closes Wf-297B §6 NUMERIC gap for paper §5 Pearson r ≥ 0.7).
2. PKU (cisplatin 4-cell, 6-12 mo, domestic Chinese partnership).
3. 协和 (NCI-60 9-cell + ICP-MS, 6-12 mo, hospital-phase + IRB + GCP).
4. ORNL (protein-K_d, 9-12 mo, R22-P4 future, not R20 paper §5-gating).

**Pre-flight gate `p3_loi_send_ready`** added to `molmetal/scripts/pre_flight_check.py`:
- New gate function `gate_loi_send_ready` (lines 706-806).
- File-path constants `loi_send_review_doc` + `loi_send_files` (4-tuple) at line ~938.
- Wired into both `pipeline=all` (line ~1330, 37 gates total) and `pipeline=frontier` (line ~1438, 12 gates total).
- **PASS criteria**: all 4 LOI files present + non-empty + Status header (DRAFT or SENT) parseable + review helper present + ≥16 checklist boxes (4/LOI × 4 LOIs).
- **Current state: PASS** (verified via `python -m scripts.pre_flight_check --pipeline all`).

---

## 4. G5 retrain integration template — Wf-312E updates

Per Wf-308D PART 3 + Wf-310A aggregation context:

| Update | Source | Status |
|---|---|---|
| G5 13/15 pairs MEASURED per Wf-310A aggregation | Wf-310A master.log + per_seed_metrics.json walk | Already documented in integration_plan §2 |
| G5 2 missing pairs retry (Ir/seed0, Re/seed2) | Wf-310A documented bug | Added §12 retry path (rerun `train_dmpnn_multitask_wasserstein.py --metal {Ir,Re} --seed {0,2} --epochs 50`) |
| G5 cascade with G4 Ru seed=0 baseline | chemprop v2 metric JSON at `metrics/by_metric/wf219_krasnov_aligned_chemprop_v2_3seed_50epoch_scaffold.json` (Ru baseline = 0.7123) | Added §12 cascade sanity check + §13 honest framing addendum |

**G5 integration script** (`molmetal/scripts/wf308d_g5_results_integration.py`) **requires no changes** — it already:
- Detects 13/15 + 2/15 partial pairs honestly (line 184 `detect_g5_completion`).
- Emits `n_pairs_total` vs `n_pairs_ok` (line 585-586).
- Reports per-metal AUC with reduced `n_seeds_used` (Ru 2/3, Ir 2/3, Os/Rh/Re 3/3).
- Carries `ru_auc_baseline_single_task_chemprop_v2: 0.7123` constant (line 351).
- Cascade narrative `vs chemprop v2 single-task baseline 0.7123` already wired (line 426).

**g5_integration_status = TEMPLATED + Wf-310A retry path documented; `--apply` NOT fired.**

---

## 5. Pre-flight status (Wf-312E)

- `molmetal/scripts/pre_flight_check.py --pipeline all` → 37/37 PASS (was 36/36, +1 for `p3_loi_send_ready`).
- `molmetal/scripts/pre_flight_check.py --pipeline frontier` → 12/12 PASS (was 11/11, +1 for `p3_loi_send_ready`).
- **preflight_status = 37/37 PASS** with new LOI gate wired in.

Verified via:
```
cd /home/hugo/codes/mol-metal && PYTHONPATH=molmetal \
  /home/hugo/codes/CadButEaas/.venv-vlm/bin/python -m scripts.pre_flight_check \
  --pipeline all 2>&1 | tail -10
```

(Used `cad-es` → `codes` corrected per actual venv path; user's spec path `/home/hugo/cades/...` not on this host.)

---

## 6. user_ok_queue (no change since Wf-308D)

- **Wf-305A LOI-RUSANOV send** — Hugo finds ChemRxiv corresponding-author email + sends. NO send by subagent.
- **Wf-74 LOI-PKU / LOI-Xiehe / LOI-ORNL sends** — recommended order Rusanov → PKU → Xiehe → ORNL.
- **Wf-308D G5 retrain integration apply** — `python molmetal/scripts/wf308d_g5_results_integration.py --apply --yes-i-approve-g5-integration [--trigger-pre-flight]`. Mutates `reports/wf219_krasnov_aligned/sweep_summary.json` + creates `metrics/by_metric/wf219_krasnov_aligned_dmpnn_mtw_5metal_3seed_50epoch_g5.json` + emits `dashboard_claim_patch.py` stub (manual paste per TODO-84).
- **Wf-308C GPU full run** — `bash molmetal/scripts/launch_wf308c_todo220_gpu.sh` after `touch /tmp/USER_OK_WF308C`. Wall: 3.9 hr GPU.

---

## 7. Honest framing (party-twenty-spirit §3.5)

- **4 LOI ready-to-send subject to Hugo review + actual send.** No send fired.
- **G5 retrain integration waits for G5 results final aggregation** (currently 13/15 MEASURED + 2/15 partial; retry path staged but not fired).
- **This is INTEGRATION work, not new measurement.** No model retrained; no data fabricated.
- **Pre-flight preserved +1 gate.** No regression introduced; new honest-negative (LOI not yet sent) catalogued.
- **TODO-116 P0** (Wf-74 v2 close-out rewrite) remains the single P0; not addressed by Wf-312E (out of scope; outreach 0 LOI sent today).

---

## 8. Revision history

| Date | Author | Change |
|---|---|---|
| 2026-09-29 | Wf-312E (Claude) | Final integration: Wf-308D outputs verified, `p3_loi_send_ready` pre-flight gate added (37/37 all + 12/12 frontier PASS), integration_plan.md §12 Wf-310A retry context + §13 honest-framing addendum added, closeout doc drafted. |
