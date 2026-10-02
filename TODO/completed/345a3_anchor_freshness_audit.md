# Wf-345-A3 — Anchor Freshness 24h Audit (CPU-only)

**Date:** 2026-09-29
**Wave:** Wf-345-A3 (Wf-345 ultracode wave; A3 sub-flow)
**Status:** SHIPPED (audit artifact + 3 NEW tests)
**Tag discipline:** party-twenty-spirit §3.5 (MEASURED vs PROJECTED)

---

## §1 Background

Per the 2026-09-29 ultracode user directive ("基于刚才的洞察，可以启动
ultracode，覆盖所有发现的问题把残余的问题解决了，一定要全，工作流可以长"),
the :func:`gate_wf343c4_anchor_freshness_24h` pre-retrain pre-flight
gate (added per Wf-343C4) currently FAILs because 4 of 11 paper-grade
``*_anchor.json`` + ``sweep_summary.json`` files exceed the 24h freshness
budget per Wf-302A / Wf-304A AnchorGuard.

This Wf-345-A3 sub-flow produces the AUDIT artifact + 3 NEW tests. The
**REFRESH of the stale anchors themselves** is queued for Wf-345 Phase 3
C1-C4 (separate workflows — each anchor has its own re-run script
+ GPU/CPU budget). This doc is the AUDIT only.

---

## §2 Audit results (MEASURED)

**Source:** `molmetal/reports/wf345a3_anchor_audit/audit.json`

### §2.1 Top-level summary

| Field | Value | MEASURED |
|---|---|---|
| `n_anchors_total` | **11** | yes |
| `n_anchors_fresh` | **7** | yes |
| `n_anchors_stale` | **4** | yes |
| `gate_status` | **FAIL** (until Phase 3 refresh) | derived |
| `max_age_hr` | 24 | config (per Wf-302A / Wf-304A) |

### §2.2 Stale anchor set (MEASURED — full audit at audit-write time)

| # | Path | mtime | age_hr | Wall-to-budget (hr) |
|---|---|---|---:|---:|
| 1 | `molmetal/reports/wf67_p4_3_100pocket/sweep_summary.json` | 2026-09-26T17:08:46Z | **70.86** | +46.86 over |
| 2 | `molmetal/reports/wf121_cisplatin_real_vina/sweep_summary.json` | 2026-09-27T03:21:01Z | **60.65** | +36.65 over |
| 3 | `molmetal/reports/wf143_p4_chemotype_dock/sweep_summary.json` | 2026-09-27T06:16:45Z | **57.73** | +33.73 over |
| 4 | `molmetal/reports/wf170_permetal_singletask/sweep_summary.json` | 2026-09-28T10:55:14Z | **29.08** | +5.08 over |

### §2.3 Fresh anchor set (7 anchors)

| Path | mtime | age_hr |
|---|---|---:|
| `molmetal/reports/wf188_m14_full_metalcytotox/m14_anchor.json` | 2026-09-28T17:32:21Z | 14.47 |
| `molmetal/reports/wf219_krasnov_aligned/sweep_summary.json` | 2026-09-28T17:32:23Z | 14.47 |
| `molmetal/reports/wf253d_m17_drugood_reeval/m17_drugood_reeval.json` | 2026-09-28T17:32:23Z | 14.47 |
| `molmetal/reports/wf269c_cfm_real/wf269c_report.json` | 2026-09-28T17:32:14Z | 14.47 |
| `molmetal/reports/wf277c_real_pilot/wf269c_report.json` | 2026-09-28T16:21:43Z | 15.64 |
| `molmetal/reports/wf308c_temporal_split/sweep_summary.json` | 2026-09-29T01:44:42Z | 6.26 |
| `molmetal/reports/wf308d_temporal_smoke/sweep_summary.json` | 2026-09-29T01:36:41Z | 6.39 |

---

## §3 Anchor refresh scope (Phase 3 work — NOT in this audit)

Each stale anchor has its own re-run script + cohort + wall budget.
The audit does NOT refresh them. Phase 3 C1-C4 = 4 sub-workflows:

| ID | Stale anchor | Refresh workflow (planned) | Wall (PROJECTED) |
|---|---|---|---:|
| C1 | wf67_p4_3_100pocket | sweep_summary.json re-run on 100-pocket × 5-seed cohort | ~30 min CPU |
| C2 | wf121_cisplatin_real_vina | sweep_summary.json re-run on cisplatin real Vina cohort | ~15 min CPU |
| C3 | wf143_p4_chemotype_dock | sweep_summary.json re-run on chemotype dock cohort | ~15 min CPU |
| C4 | wf170_permetal_singletask | sweep_summary.json re-run on per-metal singletask cohort | ~10 min CPU |

**Phase 3 total PROJECTED = ~70 min CPU** (sequential, single-worker).
**GPU needed?** No — every Phase 3 refresh is CPU-only cohort re-run on
existing cached features (per Wf-326F + Wf-328-W10 16.83× featurization
cache + Wf-333-V4 m6_kappa chemprop+PLL integration).

**Refresh risk (HONEST-NEGATIVE per party-twenty-spirit §3.5):** if any
Phase 3 re-run produces different numbers vs the stale anchor's
numbers, that is MEASUREMENT DRIFT — the stale anchor's downstream
consumers (paper_dashboard, paper_auto_update, pre_flight_check itself)
were using stale numbers for the past 5-46 hours. Declare drift as
honest-negative rather than paraphrase as "achievable lift".

---

## §4 3 NEW tests SHIPPED (audit test surface)

**Source:** `molmetal/tests/test_wf345a3_anchor_freshness.py`

| Test | What it pins | Verdict |
|---|---|---|
| `test_wf345a3_anchor_catalog_count` | Exactly 11 paper-grade anchors catalogued under the 4 ANCHOR_GLOB_PATTERNS globs. FAILs on catalog drift (e.g. new anchor added, old anchor deleted). | PASS |
| `test_wf345a3_stale_anchor_detection` | Detects every anchor whose mtime exceeds 24h; verifies the audit's stale set is a subset of the live detection set; expects ≥ 4 stale. | PASS |
| `test_wf345a3_audit_json_schema_valid` | Verifies the audit artifact at `molmetal/reports/wf345a3_anchor_audit/audit.json` parses as JSON + carries the canonical schema fields + values. | PASS |

**Test verdict:** 3/3 NEW tests PASS (per `pytest -v` output captured
2026-09-29 16:00 UTC wall ~16 s).

---

## §5 Honest framing (party-twenty-spirit §3.5)

| Claim | Tag | Source |
|---|---|---|
| 4/11 anchors exceed 24h | MEASURED | `molmetal/reports/wf345a3_anchor_audit/audit.json` |
| Stale anchor IDs (4 paths) | MEASURED | same |
| Refresh cost ~70 min CPU total | PROJECTED | per-anchor wall estimate from prior Wf re-runs |
| 3 NEW tests PASS | MEASURED | `pytest -v` output |
| Wall 15-25 min CPU-only | PROJECTED | per Wf-345-A3 scope (audit only; refresh in Phase 3) |
| Phase 3 re-runs may shift numbers | HONEST-NEGATIVE (measurement drift) | per Wf-291H §4 explicit-error contract |
| GPU launches in this audit | 0 | CPU-only audit |

---

## §6 Files written (this audit)

| Path | LOC | Purpose |
|---|---:|---|
| `molmetal/reports/wf345a3_anchor_audit/audit.json` | ~150 | Per-anchor audit artifact (schema-locked by test 3) |
| `molmetal/tests/test_wf345a3_anchor_freshness.py` | ~210 | 3 NEW audit tests |
| `TODO/completed/345a3_anchor_freshness_audit.md` | (this) | Audit completion doc |

**Backward-compat verdict:** 3 NEW tests are ADDITIVE only — they do
NOT modify any pre-existing test / gate / runner / path. The
Wf-343C4 `EXPECTED_GATE_COUNTS["all"] == 66` invariant is unchanged
because these tests live in a new file (do NOT register any gate).

---

## §7 Pre-flight gate status

| Gate | Status | Reason |
|---|---|---|
| `p3_wf343c4_anchor_freshness_24h` | **FAIL** | 4/11 anchors STALE — this Wf-345-A3 audit + Phase 3 C1-C4 refresh queued |

Until Phase 3 C1-C4 refresh the 4 stale anchors, the pre-flight v3
gate set returns `1 FAIL` in the `p3` pipeline (the `all` pipeline
also picks up the 4 NEW pre-retrain gates from Wf-343C4 including
this one). Per AGENTS.md §5.1 standing GPU grant, pre-flight PASS is
**STILL required** before any retrain / panel / ablation launch — so
the 4 stale anchors MUST be refreshed (Phase 3 C1-C4) before any
subsequent Wf launches that depend on them.

---

## §8 Cross-references

- AnchorGuard contract: `TODO/completed/302a_*.md` + `TODO/completed/304a_*.md`
- Wf-343C4 ship: `TODO/completed/343c4_integration.md`
- `gate_wf343c4_anchor_freshness_24h` source: `molmetal/scripts/pre_flight_check.py:3166`
- `ANCHOR_GLOB_PATTERNS` constant: `molmetal/scripts/pre_flight_check.py:94-99`
- Wf-345 ultracode wave overview: `TODO/INDEX.md` §0.22 + AGENTS.md §6

---

**END OF Wf-345-A3 Audit Doc**