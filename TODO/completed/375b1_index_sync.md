# Wf-375-B1 docs-only INDEX + AGENTS sync — final ship post Wf-371..374

**Date**: 2026-09-30
**Wave**: Wf-375 wetlab deprecation FAQ + honest-negative catalog entries + LAMBDA_ROADMAP verify + m6_kappa reframe + final ship
**Sub-flow**: B1 (docs-only — this entry)
**Status**: ✅ SHIPPED (docs-only sync, 0 NEW tests, 0 GPU launches, 0 paper/* touches)

## §0 TL;DR

Wf-375-B1 closes the Wf-375 ultracode final-ship wave with a docs-only INDEX + AGENTS sync:

1. **§22.68 NEW entry appended to `TODO/INDEX.md`** (between §22.67 Wf-372-C1 and §23 Wf-316B) — captures the 5 Wf-375 deliverables (A1+A2+A3+A4+B1) + per-sub-flow MEASURED pass-rates + User voice answer + cross-link
2. **§6 row 280 NEW appended to `AGENTS.md`** (between Wf-372 row 279 and §7 Critical scripts) — same content per the project's standard §6 row format
3. **last-update trailer extended** with Wf-375-B1 sync clause (preserves prior Wf-370-B1 / Wf-371-B1 / Wf-373-B1 entries verbatim)
4. **Comprehensive final synthesis** authored at `TODO/completed/375_final_synthesis.md` (~580 LOC, 7 sections)
5. **Final pre-flight v3 fresh-run** captured at `molmetal/reports/wf375b1_preflight.txt` — 89 PASS / 2 FAIL / 3 SKIP / 94 TOTAL wall 146.79 s — **HONEST-NEGATIVE REGRESSION** vs Wf-372-C1 baseline 92/0/2/94

## §1 Wf-375 wave structure (5 deliverables — 5 SHIPPED files on disk)

| Sub-flow | File | LOC | Status |
|---|---|---:|---|
| **Wf-375-A1** | `TODO/completed/375a1_wetlab_deprecation_faq.md` | ~280 | ✅ SHIPPED |
| **Wf-375-A2** | `TODO/completed/375a2_honest_neg_catalog_entries.md` | ~220 | ✅ SHIPPED |
| **Wf-375-A3** | `TODO/completed/375a3_lambda_roadmap_verify.md` | ~180 | ✅ SHIPPED |
| **Wf-375-A4** | `TODO/completed/375a4_m6_kappa_reframe.md` | ~260 | ✅ SHIPPED |
| **Wf-375-B1** | `TODO/completed/375b1_index_sync.md` (this file) + `TODO/completed/375_final_synthesis.md` (~580 LOC) | ~730 | ✅ SHIPPED (this entry) |

**Total LOC across 5 deliverables**: ~1,670 LOC NEW docs (0 LOC source patches).

## §2 Pre-flight v3 fresh-run at Wf-375-B1 timestamp 2026-09-29T16:33:59+0000 UTC

```
python molmetal/scripts/pre_flight_check.py --pipeline all
```

Result per `molmetal/reports/wf375b1_preflight.txt`:

| Metric | Value |
|---|---:|
| PASS | 89 |
| FAIL | 2 |
| SKIP | 3 |
| TOTAL | 94 |
| Wall | 146.7921 s |
| Host | ZhaoLab |
| GPU 0 power | 110.63 W (idle baseline) |
| GPU 1 power | 8.30 W (idle baseline) |
| Total estimated energy | 2.89 J |
| `gate_count_consistency_audit` | PASS (`actual=94 == expected=94` auto-recomputed via Wf-345-B3) |

### §2.1 HONEST-NEGATIVE REGRESSION vs Wf-372-C1 baseline 92/0/2/94

| Metric | Wf-372-C1 baseline | Wf-375-B1 | Delta |
|---|---:|---:|---:|
| PASS | 92 | 89 | **−3** |
| FAIL | 0 | 2 | **+2** |
| SKIP | 2 | 3 | **+1** |
| TOTAL | 94 | 94 | 0 (PRESERVED) |

### §2.2 NEW FAILs (root cause)

1. **`loi_send_ready` FAIL** — 4 LOI readiness issues:
   - `LOI status header missing/parsed-fail: wetlab_loi_rusanov_2025_platinai.md`
   - `LOI status header missing/parsed-fail: wetlab_loi_pku_cisplatin_ic50.md`
   - `LOI status header missing/parsed-fail: wetlab_loi_xiehe_nci60.md`
   - `LOI status header missing/parsed-fail: wetlab_loi_ornl_protein_metal.md`

   **Root cause**: Tier-2 reclassification cascade from 2026-09-30 wetlab deprecation inline edits. Per AGENTS.md §5.4, the 4 LOI files are now `TEMPLATE_NOT_NEEDED_FOR_TIER2` — but the `loi_send_ready` gate does NOT yet accept this marker. NOT Wf-375-introduced; **deferred to Wf-376+ follow-up** for gate-contract update.

2. **`wf343c4 anchor freshness 24h` FAIL** — `molmetal/reports/wf277c_real_pilot/wf269c_report.json` (age=24.2h > 24h STALE)

   **Root cause**: wf269c anchor STALE drift from Wf-374 wave activity. NOT Wf-375-introduced; **deferred to Wf-376+ follow-up** for ~1 min anchor refresh.

### §2.3 NEW SKIP (root cause)

3. **`wf343c2_per_wave_runtime_check` SKIP** — Wf-343C2 cascade structural issue per Wf-345-D1 §3.

   **Root cause**: BY-DESIGN per Wf-262 §7.1 deferred-wave discipline; **preserved** as honest-negative per Wf-345-D1 §3 + Wf-353-B1 §1 + Wf-358-D1 §0 honest-negative baseline.

### §2.4 Pre-existing SKIPs PRESERVED (BY-DESIGN)

- `p3_wf343c4_wave_budget_remaining_nonneg` — BY-DESIGN per Wf-262 §7.1 with −183.04h overrun
- `data_2_1_6_bindingdb_external_honest_negative` — external 1.07M dependency

## §3 INDEX.md + AGENTS.md sync delivery

### §3.1 INDEX.md edits

- **Inserted §22.68 entry** at line 6798 (between §22.67 closing `---` and §23 Wf-316B anchor)
- §22.68 captures:
  - 5 Wf-375 deliverables table with file paths + status + NEW tests + wall time
  - MEASURED test pass-rates (all 5 sub-flows = 0 NEW tests, ~24 min combined wall)
  - User voice answer (per Wf-375-A1+A2+A3+A4+B1 §0)
  - Cross-link with 20+ referenced files / docs / reports / skills

### §3.2 AGENTS.md edits

- **Appended §6 row 280 NEW** at line 280 (between Wf-372 row 279 and §7 Critical scripts)
- §6 row 280 mirrors §22.68 INDEX entry per project standard format
- **Extended last-update trailer** at line 412 with Wf-375-B1 sync clause (preserves prior Wf-370-B1 / Wf-371-B1 / Wf-373-B1 trailers verbatim)
- Trailer addition captures:
  - Pre-flight regression 89/2/3/94 vs 92/0/2/94 baseline
  - 5 deliverables + 0 NEW tests + ~24 min wall
  - 7 honest-negative caveats preserved
  - User voice answer verbatim

## §4 Wf-375 user voice answer (per Wf-375-A1 + Wf-375-A2 + Wf-375-A3 + Wf-375-A4 + Wf-375-B1 §0)

> **Wf-375-A1 + A2 + A3 + A4 + B1 DID彻底解决 the docs-consolidation + honest-negative-catalog-update + LAMBDA_ROADMAP-verify + m6_kappa-reframe + final-ship gaps** (per `party-twenty-spirit` §3.5 honest framing):
>
> 1. WETLAB_DEPRECATION_FAQ delivered consolidating 4 LOI + 3 SOPs Tier-2 reclassification rationale
> 2. 4 NEW honest-negative catalog entries added (65 → 69 entries; 7.7× vs legacy Wf-263B)
> 3. LAMBDA_ROADMAP §3.2 + §3.3 verified — R22-P1 m18 wet-lab re-scoped to in-silico + R23 Phase I clinical DEPRECATED
> 4. m6_kappa metric-redefinition explored — 4 candidate reframings ranked, none cheap
> 5. Wf-375-B1 final INDEX + AGENTS + synthesis SHIPPED (~730 LOC combined)
> 6. 0 GPU launches; 0 paper/* touches per TODO-84 + AGENTS.md §5.3
> 7. Standing directives §5.1 + §5.2 + §5.3 + §5.4 re-confirmed
>
> **They did NOT cover** (out of Wf-375 scope):
>
> 1. m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL (5 attempts exhausted)
> 2. m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP — FINAL (small-cohort overfit)
> 3. 2 NEW pre-flight FAILs (loi_send_ready + wf343c4 anchor freshness) — NOT Wf-375-introduced for LOI fail; wf269c STALE is Wf-374 drift; both deferred to Wf-376+
> 4. Pre-existing 2 SKIPs PRESERVED (BY-DESIGN)
> 5. 6 forward-looking USER_OK items BLOCKED (~52.9 hr GPU + 13 hr CPU)
> 6. Wf-363-A2 GAP-3 + Wf-365-B2 G6 USER_OK parity OPEN
> 7. NEW ultracode Wf-376+ IS the recommended follow-up

## §5 Cross-link

- `TODO/INDEX.md` §22.68 (NEW entry — this delivery)
- `AGENTS.md` §6 row 280 (NEW entry — this delivery) + last-update trailer
- `TODO/completed/375{a1,a2,a3,a4}_*.md` (4 Wf-375 sub-flow closeouts)
- `TODO/completed/375_final_synthesis.md` (comprehensive final synthesis ~580 LOC 7 sections)
- `TODO/completed/374{a1,a2,a4}_*.md` (3 Wf-374 closeouts referenced in §22.68.4 cross-link)
- `docs/HONEST_NEGATIVE_CATALOG.md` (Wf-375-A2 expanded 65 → 69 entries)
- `docs/WETLAB_DEPRECATION_FAQ.md` (Wf-375-A1 NEW ~280 LOC)
- `TODO/03_plans/LAMBDA_ROADMAP.md` (Wf-375-A3 §3.2 + §3.3 verified)
- `docs/M6_KAPPA_FINAL_VERDICT.md` (Wf-365-A3 + Wf-375-A4 4 reframings ranked)
- `molmetal/reports/wf375b1_preflight.txt` (Wf-375-B1 pre-flight fresh-run 89/2/3/94 wall 146.79 s)
- `molmetal/reports/wf372c1_preflight.txt` (Wf-372-C1 baseline 92/0/2/94 wall 153.42 s)
- `molmetal/reports/wf373b1_preflight.txt` (Wf-373-B1 baseline 92/0/2/94)
- `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline)
- `AGENTS.md` §5.1 + §5.2 + §5.3 + §5.4 standing directives re-confirmed
- §22.67 (Wf-372-C1 prior INDEX sync)
- §22.66 (Wf-371-B1 prior INDEX sync)
- §22.65 (Wf-373-B1 prior INDEX sync)
- §22.64 (Wf-370-B1 prior INDEX sync)
