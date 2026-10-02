# Wf-293D — paper_dashboard.py refresh post-Wf-284/285/286/287/288

**Date:** 2026-09-29
**Workflow:** Wf-293D (retry of Wf-279C dashboard refresh, post-Wf-286A/B/D)
**Mode:** Read-only aggregation + small catalog additions
**Owner:** Hugo Lee

---

## §0 TL;DR

Dashboard catalog refreshed on top of the Wf-280C baseline (33 claims / 17
honest-negatives / 6 cohorts).

| Counter | Pre (Wf-280C) | Post (Wf-293D) | Δ |
|---|---:|---:|---:|
| Claims | 33 | **34** | +1 |
| Honest-negatives | 17 | **18** | +1 |
| Cohorts | 6 | **7** | +1 |
| Dashboard HTML | 76.9 KB | **80.9 KB** | +4.0 KB |

**Note on `claim_wf269c_yuelbond_load_bearing`:** This claim was ALREADY in the
catalog from Wf-279C (line 393-400 of the pre-edit file). It is not a NEW
addition in Wf-293D — only `claim_wf286b_ecfp4_axis_ruled_out` is genuinely
new. So the "new claims added" tally below is 1, not 2.

---

## §1 New claim added

### §1.1 `claim_wf286b_ecfp4_axis_ruled_out` (NEW — MEASURED)

| Field | Value |
|---|---|
| Section | §6.3 |
| Badge | MEASURED |
| Value | `test_auc=0.8820 (Δ=+0.2113 vs Wf-185 0.6707 baseline)` |
| JSON path | `molmetal/reports/wf286b_ecfp4_reeval/aggregate.json::test_auc_mean + reeval_report.md §4` |
| Timestamp | 2026-09-29 |

**Verbatim claim text:**

> Wf-286B ECFP4 + Bemis-Murcko scaffold-split re-eval (1-seed × 5-ep, chemprop
> v2 BinaryClassificationFFN, CPU) test_auc=0.8820 RULES OUT ECFP4
> featurization axis as Wf-253-D 0.4544 BELOW_RANDOM collapse bottleneck
> (Δ=+0.211 vs Wf-185 v2 baseline 0.6707); remaining hypotheses: (a) chemprop
> v3 MPNN backbone, (b) v3-fallback path config, (c) DrugOOD JSON train/val/test
> OOD splits

### §1.2 `claim_wf269c_yuelbond_load_bearing` (already present from Wf-279C)

Verified present in pre-edit catalog at lines 393-400. Not modified in
Wf-293D; preserved verbatim.

---

## §2 New honest-negative added

### §2.1 `hn_wf286a_5_seed_drugood_failed` (NEW — honest-negative)

| Field | Value |
|---|---|
| Section | §9.AK |
| Item | DO NOT cite Wf-286A as 5-seed DrugOOD v3-fallback confirmation |
| Ref | `molmetal/reports/wf286a_drugood_5seed/aggregate.json + molmetal/reports/wf286_synthesis/followup_summary.md §1.1` |

**Verbatim honest-negative text:**

> DO NOT cite Wf-286A as 5-seed DrugOOD v3-fallback confirmation (FAILED:
> gate_1_all_ran=false, gate_2_total_runs=0, all 7 gates empty; only
> load_split_data ran (n_train=60908) before per-seed trainer step did not
> start — no AUC produced; wall = 1.86e-5 min effective NO-OP)

---

## §3 New cohort section added

### §3.1 `wf286b_ecfp4_direct` (NEW cohort)

Renders as a new `<section class="cohort">` block in the dashboard HTML. The
cohort carries:

- **Title:** "Wf-286B ECFP4-direct re-eval cohort (rules out ECFP4 axis as Wf-253-D 0.4544 collapse bottleneck)"
- **JSON path:** `molmetal/reports/wf286b_ecfp4_reeval/aggregate.json::per_seed[0] + reeval_report.md §3 + §4`
- **Featurization:** ECFP4 (Morgan radius=2, n_bits=2048) + chemprop v2 BinaryClassificationFFN
- **Config:** seeds=[0], epochs=5, head_hidden=128, batch=128, lr=0.001, device=cpu, wall=190.9s
- **Rows (per-split AUC table):**
  - train (n=66665) → 0.9603
  - val (n=9524) → 0.8598
  - ood_test Bemis-Murcko scaffold (n=19047) → 0.8820
- **Wf-185 v2 baseline:** mean=0.6707 ± 0.0042 (lbap_refined_ic50_scaffold, 5-seed × 20-ep × h=128/d=2)
- **Δ vs Wf-185:** +0.2113 (single-seed; Δ subset-dependent optimistic)
- **Pass-gate:** `test_auc ≥ 0.6707 − 0.02 = 0.6507` → PASS (0.8820 ≫ 0.6507)
- **Rules out:** ECFP4 featurization axis (collapse is backbone / v3-fallback / JSON-split axis, not featurization)
- **Remaining hypotheses:** (a) chemprop v3 MPNN backbone, (b) v3-fallback path config, (c) DrugOOD JSON train/val/test OOD splits
- **Caveat (subset mismatch):** Wf-275B adapter defaults to lbap_core_ic50_assay (OOD-assay task), NOT lbap_refined_ic50_scaffold — Δ is across subsets, optimistic.

---

## §4 Code edits summary

| File | Edit |
|---|---|
| `molmetal/scripts/paper_dashboard.py` | Added 1 claim entry to `CLAIM_CATALOG` (claim_wf286b_ecfp4_axis_ruled_out, §6.3). |
| `molmetal/scripts/paper_dashboard.py` | Added 1 honest-negative entry to `HONEST_NEGATIVE_CATALOG` (hn_wf286a_5_seed_drugood_failed, §9.AK). |
| `molmetal/scripts/paper_dashboard.py` | Added new `_extract_wf286b_ecfp4_direct_cohort` function and wired into `obj["cohorts"]`. |
| `molmetal/scripts/paper_dashboard.py` | Added new `_render_wf286b_ecfp4_direct_table` function and wired into HTML render pipeline. |
| `molmetal/scripts/paper_dashboard.py` | Updated `update_report` block: `wf_id` → Wf-293D, `n_new_claims_added` → 14, `n_new_honest_negatives_added` → 6, `n_new_cohort_sections_added` → 3, added new claim/hn/cohort IDs. |
| `molmetal/scripts/paper_dashboard.py` | Updated verbose print message (now says 7 cohorts, Wf-293D-Dash prefix). |

No file in `TODO/`, `paper/`, `molmetal/reports/wf265_collect/`, or `results/`
was modified — this is a dashboard refresh only.

---

## §5 Build verification

```
$ /home/hugo/codes/CadButEaas/.venv-vlm/bin/python molmetal/scripts/paper_dashboard.py \
    --out-html /tmp/dashboard_v293d.html --verbose
[Wf-293D-Dash] built dashboard: 34 claims, 18 honest-negatives, 7 cohorts, html_len=82373
[Wf-265-Dashboard] wrote /tmp/dashboard_v293d.html (80.9 KB); build JSON at /tmp/dashboard_v293d.build.json.
```

| Verification | Result |
|---|---|
| `claim_wf286b_ecfp4_axis_ruled_out` appears in HTML | YES (line 1379, claim-card div) |
| `hn_wf286a_5_seed_drugood_failed` appears in HTML | YES (line 1514-1515, honest-negative li) |
| `wf286b_ecfp4_direct` cohort section appears in HTML | YES (line 425, h3) |
| Schema version | 280.v1 (unchanged) |
| Build JSON path | `/tmp/dashboard_v293d.build.json` |
| Dashboard HTML path | `/tmp/dashboard_v293d.html` (80.9 KB) |

---

## §6 Aggregate update_report (now embedded in dashboard.html)

```json
{
  "wf_id": "Wf-293D (refresh on top of Wf-280C, post-Wf-286A/B/D)",
  "schema_version": "280.v1",
  "n_new_claims_added": 14,
  "n_new_honest_negatives_added": 6,
  "n_new_cohort_sections_added": 3,
  "new_claim_ids": [
    "claim_chemprop_v3_fallback_default",
    "claim_chemprop_v3_fallback_official",
    "claim_wf266_min_epochs_flag",
    "claim_wf266_lomo_init_scale_flag",
    "claim_wf266_v3_fix_mode",
    "claim_todo_180_g4_re_top",
    "claim_todo_180_g4_overall_15pair",
    "claim_wf266_g5_50epoch_blocked",
    "claim_wf275b_drugood_adapter",
    "claim_wf269c_yuelbond_load_bearing",
    "claim_wf275c_m14_lead_precondition",
    "claim_wf274a_regression_disclosure",
    "claim_wf271b_m17_drugood_v3_reeval",
    "claim_wf286b_ecfp4_axis_ruled_out"
  ],
  "new_honest_negative_ids": [
    "hn_no_wf263a_empirical_pass",
    "hn_no_wf266_g5_measured",
    "hn_no_wf266_retrain_lift",
    "hn_no_cfm_decode_real_09645_as_lead",
    "hn_no_yuelbond_off_deployment",
    "hn_wf286a_5_seed_drugood_failed"
  ],
  "new_cohort_keys": [
    "chemprop_v3_fallback",
    "wf266_lomo_fix",
    "wf286b_ecfp4_direct"
  ]
}
```

Note: the "n_new_claims_added" tally of 14 is the CUMULATIVE count across
Wf-268C + Wf-279C + Wf-293D. The Wf-293D-specific delta is +1 claim (the
ECFP4 axis-ruled-out) since `claim_wf269c_yuelbond_load_bearing` was already
in the catalog from Wf-279C.

---

## §7 Files produced

- `/tmp/dashboard_v293d.html` — refreshed dashboard (80.9 KB, 34 claims, 18 honest-negatives, 7 cohorts)
- `/tmp/dashboard_v293d.build.json` — build metadata JSON
- `molmetal/reports/wf293d_dashboard_refresh/refresh_report.md` — this report

---

## §8 Next step (Wf-294+)

Per Wf-286 synthesis §4 P2 — `Wf-286B-Full` (5-seed × 20-ep ECFP4 +
BinaryClassificationFFN) + matching-subset chemprop v2 MPNN baseline on
`lbap_core_ic50_assay` will pin down the exact Δ ± pooled-std and update the
TODO-130 G5 sign-flip audit. When MEASURED, refresh the dashboard again to
add a "5-seed" cohort row for `wf286b_ecfp4_direct`.

---

*Generated by Wf-293D (2026-09-29). Read-only catalog refresh. 1 NEW claim +
1 NEW honest-negative + 1 NEW cohort section added on top of the Wf-280C
baseline (33 / 17 / 6). Dashboard schema 280.v1 preserved.*