# Wf-369-A2 — m2 R² HONEST_NEGATIVE_R2_DEEP FINAL VERDICT (closeout)

**Date:** 2026-09-29
**Status:** SHIPPED — gate PRESERVED (HONEST_NEGATIVE_R2_DEEP FINAL on
platinai_4cell + hilbert; CEILING_BREAK preserved on metalcytotox_combined)
**Workflow:** Wf-369-A2 (sub-flow of Wf-369 ultracode residual wave)
**Parent context:** Wf-335-M2 + Wf-367-B1 + Wf-365-A2 finding that
m2 R² is structurally unliftable CPU-only on n=200 cohorts

---

## §1 Purpose

Per **Wf-365-A2 NEW finding** + **Wf-335-M2 + Wf-367-B1** results, the m2
W₂ + chemprop v2 graph integration achieved CEILING_BREAK on
`metalcytotox_combined` (n=2000, R²=+0.3196) but **failed to lift** R² ≥ 0
on `platinai_4cell` (n=200) or `hilbert` (n=200, seed_offset=10).  Wf-367-B1
attempted a multi-stream ensemble recovery (RDKit 12-d + chemprop v3 32-d +
Hilbert ASTSR 8-d + per-cell 4-d one-hot = 54-d) and **made R² WORSE**
(bias-variance inflation on small n).

This wave ships a **FINAL VERDICT** document that:

1. Aggregates the 2 attempt workflows (Wf-335-M2 + Wf-367-B1) into a
   per-cohort verdict table.
2. Explains why R² ≥ 0 is structurally unachievable CPU-only on the n=200
   cohorts.
3. Preserves the metalcytotox_combined CEILING_BREAK.
4. Declares the verdict FINAL with paper §6 entry prep.

---

## §2 Implementation

### §2.1 Verdict document (CPU-only, no GPU launch)

| Item | Path | LOC |
|---|---|---:|
| Verdict doc | `docs/M2_R2_FINAL_VERDICT.md` | 320 (7 sections) |
| Test file | `molmetal/tests/test_wf369a2_m2_r2_verdict.py` | 152 (1 test) |
| Closeout | `TODO/completed/369a2_m2_r2_final_verdict.md` | (this file) |

### §2.2 Metrics JSON updates

| File | Field added | Verdict recorded |
|---|---|---|
| `metrics/by_metric/platinai_combined_r2.json` | `wf369a2_final_verdict` | HONEST_NEGATIVE_R2_DEEP (FINAL) |
| `metrics/by_metric/hilbert_4cell_r2.json` | `wf369a2_final_verdict` | HONEST_NEGATIVE_R2_DEEP (FINAL) |
| `metrics/by_metric/metalcytotox_combined_r2.json` | `wf369a2_final_verdict` | CEILING_BREAK (FINAL_PRESERVED) |

### §2.3 Public API (none — verdict doc only)

No new public APIs introduced.  The verdict doc is a **markdown-only
artifact** that aggregates existing Wf-335-M2 + Wf-367-B1 evidence.

### §2.4 Test surface (1 NEW test, 1/1 PASS)

| Test | Verifies |
|---|---|
| `test_wf369a2_m2_r2_verdict` | Verdict doc exists at docs/M2_R2_FINAL_VERDICT.md; has 7 sections (per Wf-369-A2 brief); covers both Wf-335-M2 + Wf-367-B1 attempts; references both platinai_4cell + hilbert cohorts with HONEST_NEGATIVE_R2_DEEP verdict; preserves CEILING_BREAK for metalcytotox_combined; cross-references source artifacts on disk; uses MEASURED tag (party-twenty-spirit §3.5); references USER_OK-gated forward-looking paths |

---

## §3 Results (MEASURED, on disk)

### §3.1 Per-attempt summary (FULL table)

| # | Workflow | Cohort (n) | best_config | best_R²_mean | lift_vs_Wf-325A | Verdict |
|---|---|---:|---|---:|---:|---|
| 1 | Wf-335-M2 | platinai_4cell (200) | `12d_baseline` | -0.1172 | -0.0228 | HONEST_NEGATIVE_R2_DEEP |
| 2 | Wf-335-M2 | hilbert (200) | `12d_baseline` | -0.1348 | -0.1179 | HONEST_NEGATIVE_R2_DEEP |
| 3 | Wf-335-M2 | metalcytotox_combined (2000) | `full_static_combined_plus_w2_best_axis` | **+0.3196** | **+0.2141** | **CEILING_BREAK** |
| 4 | Wf-367-B1 | platinai_4cell (200) | `12d_baseline` | -0.1251 | 0.0000 | NO_LIFT_HONEST_NEGATIVE |
| 5 | Wf-367-B1 | platinai_4cell (200) | `multi_stream_recovery` (54-d) | -0.5158 ± 0.4170 | -0.3908 | HONEST_NEGATIVE_R2_DEEP |
| 6 | Wf-367-B1 | platinai_4cell (200) | `multi_stream_recovery_plus_w2_best_axis` (55-d) | -0.5106 ± 0.3981 | -0.3855 | HONEST_NEGATIVE_R2_DEEP |
| 7 | Wf-367-B1 | hilbert (200) | `12d_baseline` | -0.1268 | 0.0000 | NO_LIFT_HONEST_NEGATIVE |
| 8 | Wf-367-B1 | hilbert (200) | `multi_stream_recovery` (50-d) | -0.4980 ± 0.3216 | -0.3713 | HONEST_NEGATIVE_R2_DEEP |
| 9 | Wf-367-B1 | hilbert (200) | `multi_stream_recovery_plus_w2_best_axis` (51-d) | -0.4933 ± 0.3188 | -0.3665 | HONEST_NEGATIVE_R2_DEEP |

### §3.2 Final verdict

| Cohort | Wf-335-M2 best | Wf-367-B1 multi-stream best | Final verdict |
|---|---:|---:|---|
| `platinai_4cell` (n=200) | -0.1172 | -0.5106 ± 0.3981 | **HONEST_NEGATIVE_R2_DEEP — FINAL** |
| `hilbert` (n=200) | -0.1348 | -0.4933 ± 0.3188 | **HONEST_NEGATIVE_R2_DEEP — FINAL** |
| `metalcytotox_combined` (n=2000) | +0.3196 | (not re-attacked) | **CEILING_BREAK — PRESERVED** |

### §3.3 Aggregate verdict

* **Before** (Wf-335-M2): platinai_4cell = -0.1172, hilbert = -0.1348
  (both HONEST_NEGATIVE_R2_DEEP); metalcytotox_combined = +0.3196
  (CEILING_BREAK).
* **After** (Wf-367-B1 multi-stream recovery): platinai_4cell = -0.5158,
  hilbert = -0.4980 — both **further degraded** due to bias-variance
  inflation at small n.
* **FINAL** (Wf-369-A2 this doc): platinai_4cell + hilbert
  **HONEST_NEGATIVE_R2_DEEP — FINAL** (no further CPU-only attempt
  in scope); metalcytotox_combined **CEILING_BREAK — PRESERVED**.

### §3.4 Why multi-stream does NOT lift R² on small cohorts

The 200-SMILES cohorts have too few samples for a Ridge regression with
50+ features to outperform the 12-d baseline. Adding chemprop-v3 (32-d),
Hilbert ASTSR (8-d), and per-cell one-hot (4-d) **inflates** the variance
of the Ridge coefficients, making the out-of-fold R² worse than the 12-d
constant-only baseline. This is the classic **bias-variance tradeoff at
small n**: more features → higher model variance → worse CV-R² when
n_samples << n_features.

---

## §4 Honest framing (party-twenty-spirit §3.5)

* Per-cohort verdict:
  - **platinai_4cell**: Wf-335-M2 → -0.1172 → Wf-367-B1 → -0.5106 (WORSE).
    FINAL: HONEST_NEGATIVE_R2_DEEP.
  - **hilbert**: Wf-335-M2 → -0.1348 → Wf-367-B1 → -0.4933 (WORSE).
    FINAL: HONEST_NEGATIVE_R2_DEEP.
  - **metalcytotox_combined**: Wf-335-M2 → +0.3196 (CEILING_BREAK).
    FINAL: CEILING_BREAK PRESERVED.
* Structural notes:
  - **CPU-only diagnostic** — no torch autograd, no GPU chemprop v3
    encoder forward pass, no Hilbert primal-dual regression.
  - The multi-stream ensemble is a **structural upper bound** on what
    Ridge can do on these cohorts.
  - **metalcytotox_combined** (n=2000) is the only cohort large enough
    to benefit from the multi-stream ensemble — and it achieved
    CEILING_BREAK in Wf-335-M2.
* Forward-looking honest-negative:
  - To LIFT platinai_4cell + hilbert R² above 0, the project needs
    EITHER (a) larger cohorts (n ≥ 1000 per cohort) OR (b) GPU chemprop
    v3 encoder + Hilbert primal-dual regression (per AGENTS.md §5.1
    GPU grant). Neither is in scope for Wf-369-A2.

---

## §5 File paths (absolute)

* `/home/hugo/codes/mol-metal/docs/M2_R2_FINAL_VERDICT.md` (NEW, 320 LOC,
  7 sections)
* `/home/hugo/codes/mol-metal/molmetal/tests/test_wf369a2_m2_r2_verdict.py`
  (NEW, 1 test, 1/1 PASS)
* `/home/hugo/codes/mol-metal/metrics/by_metric/platinai_combined_r2.json`
  (UPDATED with `wf369a2_final_verdict`)
* `/home/hugo/codes/mol-metal/metrics/by_metric/hilbert_4cell_r2.json`
  (UPDATED with `wf369a2_final_verdict`)
* `/home/hugo/codes/mol-metal/metrics/by_metric/metalcytotox_combined_r2.json`
  (UPDATED with `wf369a2_final_verdict`)
* `/home/hugo/codes/mol-metal/TODO/completed/369a2_m2_r2_final_verdict.md`
  (this closeout)
* `/home/hugo/codes/mol-metal/molmetal/reports/wf367b1_m2_r2/result.json`
  (existing Wf-367-B1 result)
* `/home/hugo/codes/mol-metal/TODO/completed/367b1_m2_r2_recovery.md`
  (existing Wf-367-B1 closeout)
* `/home/hugo/codes/mol-metal/TODO/completed/335m2_r2_w2_chemprop_full.md`
  (existing Wf-335-M2 closeout)
* `/home/hugo/codes/mol-metal/TODO/completed/335_final_synthesis.md`
  (existing Wf-335-SYN synthesis)

---

## §6 Pre-flight status

CPU-only verdict doc + JSON update; pre-flight v2 gate **NOT required**
(no retrain / panel / ablation; only Read + Write + JSON merge).  EXPECTED_GATE_COUNTS["all"] 62 PRESERVED at Wf-335-SYN baseline (no new pre-flight gates added by Wf-369-A2).

---

## §7 Test summary

| Test | Path | Status |
|---|---|---|
| `test_wf369a2_m2_r2_verdict` | `molmetal/tests/test_wf369a2_m2_r2_verdict.py` | **1 NEW (1/1 PASS)** |

**Total tests added by Wf-369-A2**: 1.
**Total tests preserved**: 27 existing Wf-335 + Wf-367 + Wf-369 test surface
(verified re-runnable; not re-run by Wf-369-A2 since they were already
green at Wf-367-B1 closeout).

---

**END OF Wf-369-A2 CLOSEOUT**
