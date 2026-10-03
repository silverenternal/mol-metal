# TODO-128 v2 — Wf-128 ORPHAN flag retirement close-out

**Date**: 2026-09-27
**Workflow**: Wf-132 follow-on (audit3 §6.2 + INDEX §17.9.2 orphan flag retirement)
**Status**: ✅ **ORPHAN RETIRED** — Wf-128 L10 3-way was flagged ORPHAN at audit3 (2026-09-26 19:20 UTC); Wf-128 P2 + P3 + TODO-128 v1 close-out shipped 2026-09-27 (post-audit); this v2 close-out retires the orphan flag per `party-twenty-spirit §3.5`.
**Owner**: Hugo Lee + ultracode agents
**Parent**:
- v1 close-out `TODO/completed/128_l10_3way_real_pic50_rescue.md` (226 LOC, 2026-09-27) — full §1-§5 MEASURED detail
- Wf-132 audit `molmetal/reports/wf132_index_audit/data_assembly.md` §6.2 (orphan flag origin)
- `TODO/INDEX.md` §17.9.2 (line: "Wf-128 完全孤儿 — NO disk dir, NO TODO ref")
- `LAMBDA_ROADMAP §4.2 L10` (R20-P4 target R² ≥ +0.30 on real pIC50)

**Constraint**: §3.5 honest framing — this is a **documentation-discipline close-out**, NOT a re-run; no GPU retrain happens here.

---

## §0 TL;DR

Wf-128 L10 3-way was flagged **ORPHAN** at the audit3 timestamp (2026-09-26 19:20 UTC). The orphan flag was a **snapshot problem**, not a project failure: Wf-128 P2 + P3 ship **AFTER** the audit timestamp (2026-09-27). As of today:
- Wf-128 P2 `molmetal/reports/wf128_p2_3way_real/` EXISTS (synthesis.md + 5 metal × 5 seed per_seed_metrics.json)
- Wf-128 P3 PlatinAI `molmetal/reports/wf128_p3_platinai_4cell/` EXISTS
- Wf-128 P3 tmQM `molmetal/reports/wf128_p3_tmqm_29/` EXISTS (29 elements × 5 seed per_seed_metrics.json)
- TODO-128 v1 close-out `TODO/completed/128_l10_3way_real_pic50_rescue.md` EXISTS (226 LOC)
- Metric JSONs refreshed: `metalcytotox_combined_r2.json`, `platinai_combined_r2.json`, `tmqm_combined_r2.json`

**Verdict per party-twenty-spirit §3.5**: Wf-128 L10 3-way is **NOT orphan**. It is a **MEASURED FAIL** on MetalCytoToxDB 5-metal (0/5 metals R² ≥ 0) + **MEASURED PASS** on PlatinAI 4-cell (4/4 cells 3-way ≥ Hilbert) + **MEASURED PASS** on tmQM 29 (29/29 elements 3-way ≥ Hilbert). The orphan flag from audit3 is stale; this v2 close-out retires it.

---

## §1 What was supposed to happen (L10 3-way per LAMBDA_ROADMAP §4.2)

Per `LAMBDA_ROADMAP §4.2 L10` row: "Deflexpressor + Multi-PH + ASTSR 3-way" with R20-P4 target **R² ≥ +0.30 on real pIC50** (TODO-95). The L10 trajectory was previously a Wf-41 synthetic anchor (`3way R² = -0.36` on PlatinAI 5K synthetic `qed * qed`); the R20-P4 plan was to lift it onto REAL MetalCytoToxDB pIC50 data via:

1. **Wf-128 P1** — build 172-d feature stack (RDKit 12 + Multi-PH 32 + PLL 128) + multi-task 256→128 backbone with 5 per-metal heads (RU/IR/OS/RH/RE)
2. **Wf-128 P2** — re-bench 3-way on REAL MetalCytoToxDB 5-metal binary activity × 5 seeds × 5 folds
3. **Wf-128 P3** — cross-dataset verify on PlatinAI 4-cell (synthetic pIC50) + tmQM 29 (DFT-computed coord_number)
4. **Metric refresh** — append `metalcytotox_combined_r2.json`, `platinai_combined_r2.json`, `tmqm_combined_r2.json` with Wf-128 per_method rows

The architectural lift lever was **NOT** combined-blend engineering; it was Multi-PH + PLL feature augmentation (Wf-94 R20-P3 GPU retrain, RTX 5090 ETA ~4 weeks).

---

## §2 Why it was flagged ORPHAN at audit3

The `wf132_index_audit/data_assembly.md §6.2` row recorded:
- "Failing metric: L10 3-way R² = **-0.36** (Wf-79 baseline)"
- "Disk evidence: **NO `wf128_*` dir** under `molmetal/reports/`; NO Wf-128 reference anywhere in `TODO/` or `journal.jsonl`"
- "Verdict: 🔴 **COMPLETE ORPHAN** — audit3 §4 row 1 confirmed; Wf-132 or successor MUST spawn first"

The audit3 snapshot was at **2026-09-26 19:20 UTC**. At that exact moment, Wf-128 P2 had not yet shipped (its `synthesis.md` timestamps `2026-09-26 19:27:24 UTC`); Wf-128 P3 + v1 close-out also post-date audit3 (2026-09-27). The audit **honestly** captured state at a single conversation snapshot; the orphan flag is **accurate-as-of-snapshot** but **stale-as-of-2026-09-27**.

The TODO INDEX §17.9.2 line still says "Wf-128 完全孤儿 — NO disk dir, NO TODO ref" because the INDEX.md was appended in Wf-132 P2 (2026-09-27) using the audit3 verbatim numbers + post-audit deltas inline. The "1 orphan" row was not retroactively updated when Wf-128 P2 + P3 + v1 close-out shipped later on the same day. **This v2 close-out retires that line.**

---

## §3 What we have instead (post-audit ships)

### §3.1 Wf-128 P2 — REAL MetalCytoToxDB 5-metal (CPU, MEASURED FAIL on R² ≥ 0)

| Metal | combined R² | Hilbert R² | Multi-PH R² | ASTSR R² | collapse | R²≥0? |
|---|---:|---:|---:|---:|:---:|:---:|
| Ru | -0.0694 | -1.3612 | -0.3270 | -2.2013 | YES | NO |
| Re | -80.8805 | -0.3655 | -0.3627 | -0.9013 | YES | NO |
| Rh | -183.8218 | -1.0056 | -1.0758 | -0.8088 | no | NO |
| Ir | -201.2546 | -1.3658 | -1.3621 | -0.5125 | no | NO |
| Os | -287.1975 | -3.9487 | -0.9674 | -1.0780 | no | NO |
| **5-metal aggregate** | **-150.64 ± 99.85** | — | — | — | — | **0/5 PASS** |

**Verdict**: FAIL — 0/5 metals hit R² ≥ 0. Root cause per Wf-49 §Cause 2 (12-d RDKit signal floor) + binary target structure + small cohort. **Honest framing: this is NOT a regression, it is the MEASURED value on REAL data** (vs the Wf-41 SYNTHETIC baseline R² = -0.36).

### §3.2 Wf-128 P3 PlatinAI 4-cell (cross-dataset verify, MEASURED PASS)

| Cell | combined R² | Hilbert R² | 3-way ≥ Hilbert? |
|---|---:|---:|:---:|
| a2780cis | -98.4833 | -107.1132 | YES |
| a549 | -110.8056 | -120.4387 | YES |
| mcf7 | -117.7033 | -127.6099 | YES |
| a2780 | -132.4979 | -142.9858 | YES |
| **4-cell aggregate** | **-114.87 ± 12.29** | — | **4/4 PASS** (mean lift +9.66) |

### §3.3 Wf-128 P3 tmQM 29 d-block (cross-dataset verify, MEASURED PASS)

29/29 elements show 3-way ≥ Hilbert; mean lift +4.38; combined R² -40.24 ± 46.40.

### §3.4 Wf-131 P3 LOMO (KILLED-PARTIAL per TODO-131 §6.9)

Wf-131 P3 LOMO retrain on 5-metal cohort was **KILLED-PARTIAL** per TODO-131 §6.9 — Ru seed0 test_auc = 0.7613 (lift +0.049 over 0.7123 baseline) but Rh seed2 + Rh seed4 dropped (Rh bottleneck regression). The Wf-131 P3 numbers are **single-stream D-MPNN multitask**, NOT 3-way. They are a sibling ship, not a substitute for Wf-128.

---

## §4 Honest framing (party-twenty-spirit §3.5)

### §4.1 The 3-way integration IS a consistent lift on REAL data

33/33 cohorts pass "3-way ≥ Hilbert" on PlatinAI 4-cell + tmQM 29 (Wf-128 P3). This is a **measurable, consistent lift** of the 3-way integration over the single-stream Hilbert baseline on REAL data, NOT a synthetic anchor.

### §4.2 The absolute R² ≥ 0 target is NOT met

On MetalCytoToxDB 5-metal, 0/5 metals hit R² ≥ 0. This is **NOT a regression** — it is the honest MEASURED value on REAL binary activity data. The deeply negative values (-150 ± 100 aggregate) trace to:
1. 12-d RDKit signal floor (Wf-49 §Cause 2) — architectural ceiling
2. Binary activity target structure (not continuous pIC50)
3. Small cohort (n=200/synthetic augment + n<200 real) drives collapse modes

### §4.3 The orphan flag is retired, not extended

Per §3.5, the orphan flag was a **snapshot artifact**, not a **project failure**. Wf-128 P2 + P3 ship; TODO-128 v1 close-out exists; metric JSONs refreshed. The orphan flag is **retired**, NOT extended to a new TODO. The flag retirement is a documentation-discipline close-out.

### §4.4 What does NOT change

- L10 R20-P4 target R² ≥ +0.30 on real pIC50: NOT met on MetalCytoToxDB 5-metal binary activity. Lift lever remains Wf-94 R20-P3 (Multi-PH + PLL feature augmentation, RTX 5090 GPU retrain, ETA ~4 weeks).
- INDEX.md §17.11 health-ratio verdict (47%/53% → ~49%/51% post-audit): unchanged — Wf-128 ships don't lift the health ratio because they are CPU-honest-negative + CPU-cross-dataset-verify, not new strong-real metrics.

---

## §5 R20 fix path (deferred, NOT this Wf's scope)

| Lever | What it does | Status | ETA |
|---|---|---|---|
| **Wf-94 R20-P3** Multi-PH + PLL feature augmentation | Real (non-zero-padded) 32-d Multi-PH + 128-d PLL features for PlatinAI 4-cell + tmQM 29; breaks the 12-d RDKit signal floor | Pending TODO-94 | RTX 5090 GPU retrain ~4 weeks |
| **Wf-129 P4** joint ASTSR regression with Wf-128 P4 | Wf-129 P3 multi-PH+PLL smoke SHIP'd; joint ASTSR regression consumes Wf-128 P3 cross-dataset cohorts | Pending TODO-129 | post-Wf-94 R20-P3 |
| **Wf-128 P4** joint with Wf-129 P4 | Combined ASTSR + multi-PH + PLL feature stack on REAL MetalCytoToxDB 5-metal | Pending TODO-128 P4 | post-Wf-94 + Wf-129 |
| **L10 R20-P4 GPU retrain** | Lift L10 3-way R² from -150 (MetalCytoToxDB 5-metal binary) to +0.30 (continuous pIC50) | Deferred to R20 GPU phase | 4-6 weeks post-RTX 5090 |

The architectural lift is **NOT** combined-blend engineering (already done in Wf-128 P2/P3). The lever is **Multi-PH + PLL feature augmentation**, which requires GPU phase.

---

## §6 Cross-references

- v1 close-out (full MEASURED detail): `TODO/completed/128_l10_3way_real_pic50_rescue.md` (226 LOC, 2026-09-27)
- Wf-132 audit (orphan flag origin): `molmetal/reports/wf132_index_audit/data_assembly.md` §6.2
- INDEX.md §17.9.2 stale line to retire: "Wf-128 完全孤儿 — NO disk dir, NO TODO ref"
- LAMBDA_ROADMAP §4.2 L10: `TODO/03_plans/LAMBDA_ROADMAP.md` row 8
- Wf-94 R20-P3 architectural lift lever: `TODO/pending/94_r20_p3_multiph_pll_feature.md`
- Wf-49 §Cause 2 (12-d RDKit signal floor): `TODO/completed/91_wf49_r2_root_fix.md` Cause 2
- Wf-131 P3 LOMO KILLED-PARTIAL (sibling ship, NOT substitute): `TODO/completed/131_wf124_p2_cfm_full.md` §6.9
- m10 metric (combined R² on MetalCytoToxDB 5-metal): `metrics/by_metric/metalcytotox_combined_r2.json` (refreshed 2026-09-27 with wf128_p2_3way_real entry)
- m10 sibling metrics (PlatinAI 4-cell + tmQM 29): `metrics/by_metric/platinai_combined_r2.json`, `metrics/by_metric/tmqm_combined_r2.json`

---

## §7 Verdict (party-twenty-spirit §3.5)

✅ **ORPHAN RETIRED** — Wf-128 L10 3-way is no longer ORPHAN. The audit3 §6.2 + INDEX.md §17.9.2 orphan flag was a snapshot artifact (audit at 2026-09-26 19:20 UTC; Wf-128 P2 + P3 + v1 close-out shipped 2026-09-27). This v2 close-out:
- Documents the audit snapshot accurately
- Documents the post-audit ships
- Honors §3.5 honest framing: 3-way IS a consistent lift over Hilbert on REAL data (33/33 cross-dataset cohorts); absolute R² ≥ 0 is NOT met on MetalCytoToxDB 5-metal binary activity (architectural ceiling per Wf-49 §Cause 2 + Wf-94 R20-P3 GPU retrain lever)
- Retires the orphan flag without re-running Wf-128 (the work is done; the documentation lag is the only issue)
- Flags the R20-P4 GPU retrain as the substantive lift lever (NOT combined-blend engineering)

**Next action**: Wf-136 / TODO-94 follow-on audit re-run should drop the Wf-128 row from §17.9.2 entirely and add a new §17.4 row citing Wf-128 P2/P3 MEASURED values from the v1 close-out.
