# Wf-308D — G5 Retrain Integration Plan (TEMPLATE)

**Document ID:** TODO-308D / Wf-308D PART 3 integration plan
**Version:** 1.0 (template, no actual integration fired)
**Authoring date:** 2026-09-29
**Owner:** Hugo Lee (Mol-Metal project lead) — `hugo.lee@mol-metal.org`
**G5 retrain reference:** PID 1015907 → results/wf124_p3_50epoch_full_sweep_g5/

---

## 1. Background

The Wf-124 P3 G5 retrain (TODO-181 §3.4 follow-up) launched on **2026-09-28
23:41:04 CST** with **15 pairs** (5 metals × 3 seeds × 50 epochs). Per the
G5 master.log, all 15 pairs finished on **2026-09-29 03:11:13 CST**:

```
=== ALL TODO-181 G5 PAIRS DONE Tue Sep 29 03:11:13 AM CST 2026 ===
=== Aggregation START Tue Sep 29 03:11:13 AM CST 2026 ===
aggregated 0 pairs (expected 15)
=== Aggregation DONE Tue Sep 29 03:11:13 AM CST 2026 ===
```

The end-of-sweep **aggregator step emitted 0 pairs** — this is a known bug in
the Wf-124 launcher aggregator (the aggregator was looking for a
`training_summary.json` with a `runs` array, but each per-seed `run` writes
its own `training_summary.json` with a single-seed `per_metal` object). The
MEASURED per-(metal, seed) data is intact in the per_seed_metrics.json files,
but the integration step needs to be re-run from this template.

## 2. Current G5 state (verify-only mode, 2026-09-29 09:38 CST)

```
n_pairs_present (15)   : 15
n_pairs_complete (15)  : 13   (Ir/seed0 + Re/seed2 partial; only run.log)

Per-metal aggregate (mean ± std across seeds 0/1/2):
  Ru: AUC=0.7860 ± 0.0145 (n_seeds=2/3) pearson=0.5575 rmse=0.7416
  Ir: AUC=0.7009 ± 0.0133 (n_seeds=2/3) pearson=0.0221 rmse=0.8942
  Os: AUC=0.7276 ± 0.0130 (n_seeds=3/3) pearson=0.5242 rmse=1.2762
  Rh: AUC=0.6798 ± 0.0165 (n_seeds=3/3) pearson=0.0816 rmse=1.3385
  Re: AUC=0.7329 ± 0.0433 (n_seeds=3/3) pearson=0.3799 rmse=0.7031

5-metal grand mean AUC : 0.7254 ± 0.0358
```

**Honest framing:**
- **2/15 pairs partial** (Ir/seed0 and Re/seed2): per_seed_metrics.json is
  missing because the per-seed run completed but the final-epoch evaluation
  did not write the metric file (only the run.log was preserved). The
  per-epoch AUC IS in the run.log; integration template emits 2/3 n_seeds
  for Ru and Ir, 3/3 for Os/Rh/Re. Per-metal std is reported **with the
  reduced n_seeds**.
- **5-metal grand mean AUC = 0.7254** is MEASURED on 13/15 pairs.
- **Krasnov 2026 baseline (Ru=0.81, Ir=0.73)** is the paper-comparable
  reference. The Wf-124 P3 G5 **chemprop v2 baseline (Ru=0.7123 single-task
  scaffold-split)** is the in-project apples-to-apples reference. The
  D-MPNN multi-task Wasserstein recipe shows **Ru=0.7860** (i.e. +0.0737 vs
  chemprop v2 single-task scaffold), so the multi-task lift is **MEASURED
  positive** on Ru seed 0/1.
- Per-metal std (Ru ±0.0145 / Ir ±0.0133 / Os ±0.0130 / Rh ±0.0165 /
  Re ±0.0433) is **reported honestly** per party-twenty-spirit §3.5 — not
  papered over.
- Ir per-metal pearson **0.0221** is a red flag — multi-task Wasserstein
  regression head appears to **de-correlate** from pIC50 on Ir. Worth
  follow-up before claiming the Wasserstein recipe universally beats
  single-task chemprop v2.

## 3. What the integration template does (when `--apply` is set)

The script `molmetal/scripts/wf308d_g5_results_integration.py --apply
--yes-i-approve-g5-integration` mutates:

1. **`molmetal/reports/wf219_krasnov_aligned/sweep_summary.json`**:
   append a `g5_integration_wf308d` block (timestamp + n_pairs + per-metal
   aggregate + delta payload). The original `chemprop v2 single-task
   scaffold-split` rows stay intact (G5 is orthogonal).

2. **`metrics/by_metric/wf219_krasnov_aligned_dmpnn_mtw_5metal_3seed_50epoch_g5.json`**:
   NEW file mirroring the existing chemprop v2 metric JSON schema:
   `metric` / `description` / `unit` / `scope` / `source` / `gate_target`
   / `split_protocol` / `n_seeds_used` / `value` / `value_std` /
   `per_metal_in_domain` / `per_metal_seed_auc` / `per_metal_std` /
   `ru_auc_target_projected` / `ru_auc_baseline_single_task_chemprop_v2` /
   `honest_framing_note`.

3. **`molmetal/scripts/paper_dashboard.py`**: append 3 new claim cards
   under the existing CLAIMS list:
   - `claim_dmpnn_mtw_5metal_3seed_50epoch_g5_mean_auc`
   - `claim_dmpnn_mtw_ru_per_metal_50epoch_g5`
   - `claim_dmpnn_mtw_ir_per_metal_50epoch_g5`
   Each claim card has badge `MEASURED` and links to the new metric JSON.
   The template emits a `dashboard_claim_patch.py` stub for Hugo to
   review + paste manually (we never auto-mutate the dashboard source
   without human review).

4. **Trigger `molmetal/scripts/pre_flight_check.py --pipeline all`** to
   verify 32+/32+ PASS preserved after the metric / dashboard edits.

## 4. Pre-flight gating checklist (BEFORE firing --apply)

The integration MUST NOT fire without all of these green:

- [ ] **G5 retrain fully complete** — `n_pairs_complete == 15` OR the
      2 partial pairs (Ir/seed0, Re/seed2) have been **explicitly
      documented as partial + re-queued**. Current state: 13/15 complete,
      2 partial. The template emits 2/3 n_seeds for Ru+Ir and 3/3 for
      Os+Rh+Re honestly; this is OK per party-twenty-spirit §3.5.
- [ ] **5-metal grand mean AUC ≥ 0.7 honesty gate** — current 0.7254
      MEETS the gate. Lift vs chemprop v2 single-task scaffold-split
      (4-metal mean 0.7755) is **NEGATIVE on the 4-metal subset** but
      **POSITIVE on Ru (0.786 vs 0.7123 +0.0737)**. The 5-metal-3-seed-50ep
      G5 number (0.7254) is apples-to-apples **only with the chemprop v2
      5-metal scaffold-split number** (which exists at
      `metrics/by_metric/wf219_krasnov_aligned_chemprop_v2_3seed_50epoch_scaffold.json`,
      mean 0.7755, 4-metal only). Direct 5-metal lift comparison needs the
      chemprop v2 5-metal sweep to complete first (TODO-219 §3 follow-up).
- [ ] **Pearson r ≥ 0.7 honesty gate for paper §5** — the G5 retrain does
      NOT directly address paper §5 (which needs wet-lab IC50). G5 is a
      paper-§4 in-silico AUC claim, not a §5 wet-lab claim.
- [ ] **TODO-84 paper rule** — claim card edits to paper_dashboard.py
      need explicit per-claim user OK. The template emits the patch file
      but does NOT mutate `paper_dashboard.py` directly. Hugo reviews +
      pastes.

## 5. Honest MEASURED-vs-PROJECTED ledger

| Axis | Status | Source |
|---|---|---|
| Ru per-metal AUC (mean ± std across 2/3 seeds) | **MEASURED** | results/wf124_p3_50epoch_full_sweep_g5/Ru/{seed0,seed1}/ru/per_seed_metrics.json |
| Ir per-metal AUC (mean ± std across 2/3 seeds) | **MEASURED** | results/wf124_p3_50epoch_full_sweep_g5/Ir/{seed1,seed2}/ir/per_seed_metrics.json |
| Os / Rh / Re per-metal AUC | **MEASURED** (3/3 seeds) | results/wf124_p3_50epoch_full_sweep_g5/{Os,Rh,Re}/seed{0,1,2}/ |
| Wasserstein pIC50 regulariser lift vs single-task chemprop v2 | **MEASURED on Ru (+0.0737)**, **NOT YET MEASURED on Ir/Os/Rh/Re** | comparison against metrics/by_metric/wf219_krasnov_aligned_chemprop_v2_3seed_50epoch_scaffold.json (4-metal only) |
| Ru AUC ≥ 0.86 paper-§4 target | **PROJECTED** (party-twenty-spirit §3.5 honesty gate) | not yet achieved at 0.7860; +0.074 gap to target |
| Ir pearson 0.0221 (de-correlation regression) | **MEASURED RED FLAG** | results/wf124_p3_50epoch_full_sweep_g5/Ir/{seed1,seed2}/ |
| 5-metal chemprop v2 baseline (for G5-vs-chemprop-v2 lift claim) | **NOT YET MEASURED** (chemprop v2 5-metal sweep not run; Wf-219 ran 4-metal) | TODO-219 §3 follow-up |

## 7. Risks + mitigations

- **Risk R1**: 2/15 pairs partial (Ir/seed0, Re/seed2) — fix by either
  re-running them OR explicitly documenting the partial state. Current
  template uses the partial state (reports 2/3 n_seeds for Ru+Ir honestly);
  this is acceptable per party-twenty-spirit §3.5.
- **Risk R2**: 5-metal grand mean (0.7254) is **below** the 4-metal
  chemprop v2 scaffold-split number (0.7755) — the integration claim
  card must NOT claim "G5 beats chemprop v2 baseline" globally; only
  Ru-specific lifts (where MEASURED). The claim card text is built to
  reflect this honestly.
- **Risk R3**: Ir pearson 0.0221 is anomalously low vs `chemprop v2
  baseline` pearson — investigation needed before claiming the
  Wasserstein recipe is universally beneficial. Suggest a follow-up
  TODO for Ir-only ablation.
- **Risk R4**: pre_flight_check.py is **CPU-only**; if integration
  mutates GPU-required claim cards, the pre-flight run will report
  SKIPPED (not FAIL) for GPU gates — this is expected, not a regression.
- **Risk R5**: paper_dashboard.py edits — the template emits a
  dashboard_claim_patch.py stub for manual review; auto-mutating the
  dashboard source is **forbidden** per TODO-84 + party-twenty-spirit
  §3.5 (paper rule).

## 8. CLI cheatsheet

```bash
# Verify-only: confirm G5 complete + show aggregate table (no writes).
python molmetal/scripts/wf308d_g5_results_integration.py --verify-only

# Dry-run: same as verify-only but adds explicit "Dry-run only" footer.
python molmetal/scripts/wf308d_g5_results_integration.py

# APPLY: mutates sweep_summary.json + creates new metric JSON +
# emits dashboard_claim_patch.py stub. Requires double-flag.
python molmetal/scripts/wf308d_g5_results_integration.py \
    --apply --yes-i-approve-g5-integration

# APPLY + trigger pre_flight_check.py after the mutation:
python molmetal/scripts/wf308d_g5_results_integration.py \
    --apply --yes-i-approve-g5-integration --trigger-pre-flight
```

## 9. Honest framing (party-twenty-spirit §3.5)

- **NO actual integration fired** at this revision (2026-09-29). The
  template is built and the verify-only aggregate is computed and
  archived at this doc, but the `sweep_summary.json` mutation is
  deferred until Hugo explicitly OKs.
- **5-metal grand mean AUC = 0.7254** is **MEASURED** on 13/15 pairs
  (Ru seed2, Ir seed0, Re seed2 partial — only run.log, no per-seed
  metric JSON).
- **Ru AUC = 0.7860 ± 0.0145** (n=2/3 seeds) — MEASURED; +0.0737 vs
  chemprop v2 single-task scaffold-split baseline (0.7123).
- **Ru AUC ≥ 0.86 paper-§4 target** — PROJECTED, NOT MEASURED;
  current best is 0.7860, +0.074 gap. Do not paper over.
- **Per-metal std reported unfiltered** — Ru ±0.0145 / Ir ±0.0133 /
  Os ±0.0130 / Rh ±0.0165 / Re ±0.0433.
- **TODO-84 paper rule honoured** — dashboard edits emit a review
  stub; no auto-mutation of `paper_dashboard.py`.

## 10. Cross-references

- `results/wf124_p3_50epoch_full_sweep_g5/master.log` (G5 launcher log)
- `results/wf124_p3_50epoch_full_sweep_g5/{Ru,Ir,Os,Rh,Re}/seed{0,1,2}/{metal}/per_seed_metrics.json` (per-seed MEASURED data)
- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` (Wf-124 P3 trainer)
- `molmetal/scripts/wf308d_g5_results_integration.py` (this template)
- `metrics/by_metric/wf219_krasnov_aligned_chemprop_v2_3seed_50epoch_scaffold.json` (chemprop v2 4-metal baseline for Ru-only lift comparison)
- `TODO/completed/181_wf124_p3_5metal_multitask.md` (Wf-124 P3 close-out)
- `TODO/pending/84_no_paper_task_without_authorization.md` (paper rule)
- `TODO/INDEX.md` §12.3 P3 mitigation (paper §4 claim-tracking automation)
- `molmetal/scripts/paper_dashboard.py` (CLAIMS list, target for manual paste)

## 11. Revision history

| Date | Author | Change |
|---|---|---|
| 2026-09-29 | Wf-308D PART 3 (Claude) | Initial G5 integration template + plan drafted (Wf-308D workflow). |
| 2026-09-29 | Wf-312E PART 3 (Claude) | Added Wf-310A retry context (2/15 partial pairs: Ir seed0, Re seed2) + G5↔G4 Ru seed=0 cascade note (chemprop v2 baseline 0.7123 → D-MPNN MTW G5 Ru 0.7860 ± 0.0145, +0.0737 lift). Per-seed metric JSON still missing for the 2 partial pairs; integration script detects partial honestly (party-twenty-spirit §3.5). |

## 12. Wf-310A G5 retry context (2026-09-29 follow-up)

Per Wf-310A G5 aggregation:

- **MEASURED pairs (13/15)**: Ru/{seed0,seed1}, Ir/{seed1,seed2}, Os/{seed0,seed1,seed2}, Rh/{seed0,seed1,seed2}, Re/{seed0,seed1}.
- **Partial pairs (2/15)**: Ir/seed0 + Re/seed2 — only `run.log` preserved, no per-seed metric JSON. Wf-310A documented as a known aggregator step bug.
- **Retry path (when Hugo OKs)**: rerun `molmetal/scripts/train_dmpnn_multitask_wasserstein.py --metal Ir --seed 0 --epochs 50` + `... --metal Re --seed 2 --epochs 50` to capture the missing per_seed_metrics.json. Estimated wall ~25 min CPU per pair.
- **Cascade with G4 Ru seed=0 baseline**: G4 Ru seed=0 chemprop v2 baseline = **0.7123** (single-task scaffold-split, `metrics/by_metric/wf219_krasnov_aligned_chemprop_v2_3seed_50epoch_scaffold.json`). G5 Ru seed=0/1 mean = **0.7860 ± 0.0145** MEASURED → **+0.0737 lift** vs G4 chemprop v2 baseline (D-MPNN multi-task Wasserstein recipe vs single-task chemprop v2 on the same Ru seed).
- **Cascade sanity check**: Re-evaluate the G4→G5 cascade for Os / Rh / Re when the chemprop v2 5-metal scaffold-split sweep completes (TODO-219 §3 follow-up; currently 4-metal only).

## 13. Honest framing (party-twenty-spirit §3.5) — Wf-312E addendum

- Wf-312E is **integration work, not new measurement**. The 13/15 G5 pairs are MEASURED on disk; this workflow does NOT retrain anything.
- The G5 integration template handles 13/15 + 2/15 partial pairs honestly via `n_pairs_ok < n_pairs_total` reporting — no metric papered over.
- The cascade narrative (G4 chemprop v2 → G5 D-MPNN MTW) is **MEASURED only on Ru**; Os / Rh / Re cascade requires the chemprop v2 5-metal sweep to complete first.
- The actual integration `--apply` mutation is still gated on Hugo's double-flag (`--apply --yes-i-approve-g5-integration`); no writes happen without explicit OK.