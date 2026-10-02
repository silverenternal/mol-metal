# Wf-322D — INDEX.md §0 + §22 + README.md Wf-309..322 sync

**Date:** 2026-09-29 11:06 CST
**Owner:** Hugo Lee + Wf-322D sub-agent
**Status:** ✅ SHIPPED (docs-only)
**Parent TODO:** TODO-322D (Wf-322D docs-sync)
**Cross-refs:** `TODO/INDEX.md` §0.8/§0.9/§22.15/§22.16 + `TODO/README.md` Wf-309..322 ship wave section + `TODO/INDEX.md` §22.16 honest framing disclaimer

---

## §0 TL;DR

| Question | Answer |
|---|---|
| What was added? | 13 NEW entries: 1 new §0.8 (Wf-321..322 ship wave timeline) + 1 new §0.9 (net effect) + 1 new §22.15 (Wf-321..322 detailed table) + 1 new §22.16 (honest framing disclaimer with 15-FAIL pre-flight disclosure) + 9 README.md Wf-321/322 lines + 1 closeout doc (this file) |
| Pre-flight v3 fresh-run result | **42 PASS / 15 FAIL / 0 SKIP / 57 TOTAL** wall 65.0s (2026-09-29 11:06 CST) — HONEST-NEGATIVE REGRESSION vs previously-claimed 52/52 PASS PRESERVED |
| Was paper work touched? | NO — `paper/*` UNTOUCHED per TODO-84 ACTIVE rule |
| Was GPU work launched? | NO — pure docs-only Wf per AGENTS.md §5.1 kill_then_adapt directive |
| Was TODO directory moved (pending → completed)? | NO — pending/ UNCHANGED at 13 files (Wf-321/322 docs-only) |
| completed/ count today | **405 files** (per `ls TODO/completed/ | wc -l` 2026-09-29 11:06 CST) |

---

## §1 Scope (Wf-322D mandate)

Per Wf-317I synthesis final closeout (this Wf = Wf-322D = docs-sync surface):

1. **Read** `TODO/INDEX.md` §0 + §22 (current ship wave entries) — DONE.
2. **Add** ship wave entries for Wf-309..322 — DONE (Wf-309..316 already in §0.4/§22.10/§22.11 per Wf-316E; Wf-317..320 already in §0.6/§22.13 per Wf-320I; Wf-321..322 ADDED in §0.8/§22.15 per this Wf).
3. **Update** `TODO/README.md` with Wf-309..322 entries — DONE (added Wf-318..320 closeout wave + Wf-321..322 ship wave + pre-flight v3 fresh-run honest-negative REGRESSION block).
4. **Verify** pre-flight: `python molmetal/scripts/pre_flight_check.py --pipeline all` — DONE (output: 42 PASS / 15 FAIL / 0 SKIP / 57 TOTAL wall 65.0s; honest-negative disclosed).
5. **Document** this closeout: `TODO/completed/322d_index_readme_wf309_322_sync.md` — DONE (this file).

---

## §2 What was actually changed

### §2.1 `TODO/INDEX.md` edits

| # | Section | Change | LOC added |
|---:|---|---|---:|
| 1 | Line 3 header | "Last updated" updated to Wf-322D 2026-09-29 11:06 CST + 15-FAIL pre-flight disclosure | ~0.5 |
| 2 | §0.7 | PRESERVED (no change); followed by NEW §0.8 Wf-321..322 timeline + NEW §0.9 net effect | ~50 |
| 3 | §22.14 | PRESERVED (no change); followed by NEW §22.15 Wf-321..322 detailed table (13 rows) + NEW §22.16 honest framing disclaimer with 15-FAIL breakdown | ~80 |
| 4 | §22 END marker | Updated to include Wf-321..322 codification references (§22.15/§22.16) + 405 file count | ~0.5 |
| 5 | **Total** | **Pure docs-only; no code change; no paper touch; no GPU launch** | **~130 LOC** |

### §2.2 `TODO/README.md` edits

| # | Section | Change | LOC added |
|---:|---|---|---:|
| 1 | Line 3 header | "Last updated" updated to Wf-322D 2026-09-29 11:06 CST + 15-FAIL pre-flight disclosure | ~0.5 |
| 2 | After "Wf-309..317 ship wave" | NEW "Wf-318..320 closeout wave" section (3 Wf entries) | ~6 |
| 3 | After "Wf-318..320 closeout wave" | NEW "Wf-321..322 ship wave" section (12 Wf entries + Wf-322D itself) | ~18 |
| 4 | After "Wf-321..322 ship wave" | NEW "Pre-flight v3 fresh-run HONEST-NEGATIVE REGRESSION disclosure" block (15-FAIL breakdown) | ~20 |
| 5 | "Archived Work" | Updated file count 351 → 405 + Wf-321/322 cross-refs | ~1 |
| 6 | **Total** | **Pure docs-only; no code change; no paper touch; no GPU launch** | **~45 LOC** |

### §2.3 NEW closeout doc

| # | File | Purpose |
|---:|---|---|
| 1 | `TODO/completed/322d_index_readme_wf309_322_sync.md` | This file — Wf-322D closeout with Wf-309..322 cross-refs + 15-FAIL pre-flight disclosure + Wf-321/322 per-file scope |

---

## §3 Pre-flight v3 fresh-run result (2026-09-29 11:06 CST, wall 65.0s)

Per AGENTS.md §4 (pre-flight v2/v3 gate) + Wf-306C (v2 → v3 upgrade 25 → 32 → 48 → 52 → **57 gates** with Wf-321D `p3_gate_count_consistency_audit` added):

```json
{
  "timestamp": "2026-09-29T03:06:51+0000",
  "schema_version": "207.v1",
  "pipeline": "all",
  "summary": {
    "pass": 42,
    "fail": 15,
    "skip": 0,
    "total": 57,
    "total_wall_sec": 65.0
  }
}
```

**HONEST-NEGATIVE disclosure per `party-twenty-spirit` §3.5**: this is a **REGRESSION from previously-claimed 52/52 PASS PRESERVED** (per Wf-316E §22.12 + Wf-320I §22.14). Wf-322D does NOT fix the 15 FAILs (docs-only Wf per AGENTS.md §5.1). The 15 FAILs are disclosed in `TODO/INDEX.md` §22.16 (NOT MEASURED today block):

1. `p1_chemprop_smoke` (P1) — chemprop `--quick` smoke FAIL in 8.1s; `molmetal/data/metal_cytotox.py:217` import error.
2. `p3_todo_119_p14_cpu_smoke_ready` (P3) — TODO-119 help missing `--chemprop-mode v3-fix --chemprop-min-epochs --chemprop-lomo-init-scale` tokens (per Wf-321H scaffold).
3. `p2_tests_pass` (P2) — pytest FAIL `ImportError: flow_matching upstream not configured; call configure_reference(<path>) before importing _reference.path`.
4. `p3_frontier_math_day_3_5_integration_tests` (P3) — same `flow_matching` ImportError.
5. `p3_smoke_fail_on_missing_audit` (P3) — same `flow_matching` ImportError.
6. `p3_r20_phase0_loaders_complete` (P3) — same `flow_matching` ImportError on R20-Phase-0 loader tests.
7. `p3_ablation_runner_tests_present` (P3) — same `flow_matching` ImportError.
8. `p3_admet_aq_sol_honest_negative_tag` (P3) — `metrics/by_metric/aqueous_solubility_logS.json: HONEST_NEGATIVE=None (expected True)`.
9. `p3_admet_hepatotox_honest_negative` (P3) — `metrics/by_metric/hepatotox_index.json: HONEST_NEGATIVE=None (expected True)`.
10. `p3_admet_logp7_4_honest_negative` (P3) — `metrics/by_metric/logp7_4.json: HONEST_NEGATIVE=None (expected True)`.
11. `p3_admet_m6_kappa_honest_negative` (P3) — `metrics/by_metric/m6_kappa.json: HONEST_NEGATIVE=None (expected True)`.
12. `p3_nci60_9cell_per_cell_aggregate` (P3) — `metrics/by_metric/nci60_precious_9cell_auc.json: wf312b_nci60_smoke_aggregate block missing`.
13. `p3_pb_packages_status_present` (P3) — `steric_clash_pb_dock.json` status='DEFERRED' (expected 'MEASURED_SMOKE').
14. `p3_pb_packages_available_or_documented` (P3) — PB packages available (rdkit=2026.03.6, posebusters=0.6.5) but `steric_clash_pb_dock.json` not DEFERRED.
15. `p3_gate_count_consistency_audit` (P3) — gate count drift for pipeline='audit': actual=0 vs expected=57 [-DRIFT (silent gate loss)].

**Standing directive**: 5 of the 15 FAILs are pytest cascade-FAILs rooted in a single `flow_matching` upstream `configure_reference(<path>)` import — fixing this upstream import would re-enable all 5 pytest gates. The other 10 FAILs are independent.

**Recommended follow-up Wf-323+ candidates** (NOT in this Wf-322D scope):
- Wf-323A: fix `flow_matching` upstream `configure_reference` import (5 cascade-FAIL recovery).
- Wf-323B: add `HONEST_NEGATIVE=True` to 4 ADMET metric JSONs (`aqueous_solubility_logS.json` + `hepatotox_index.json` + `logp7_4.json` + `m6_kappa.json`).
- Wf-323C: add `--chemprop-mode v3-fix --chemprop-min-epochs --chemprop-lomo-init-scale` argv tokens to TODO-119 P1.4 help (per Wf-321H scaffold).
- Wf-323D: add `wf312b_nci60_smoke_aggregate` block to `nci60_precious_9cell_auc.json`.
- Wf-323E: update `steric_clash_pb_dock.json` status from 'DEFERRED' to 'MEASURED_SMOKE' (with Vina 1.2.7 evidence).
- Wf-323F: fix `p3_gate_count_consistency_audit` gate-count drift (expected 57, got 0).

---

## §4 Honest framing per `party-twenty-spirit` §3.5

1. **MEASURED-vs-PROJECTED tag discipline**: every numeric value in this doc + INDEX.md §22.16 + README.md pre-flight REGRESSION block traces to either (a) the pre-flight `molmetal/reports/wf322d_preflight/pre_flight.json` JSON (42/15/0/57/wall=65.0s), (b) `ls TODO/completed/ | wc -l = 407` (2026-09-29 11:16 CST; concurrent Wf-323 closeout writes increased count from 405 to 407 between initial query and final state), or (c) the Wf-321/322 closeout files (each cited by file path).
2. **"Kill then adapt" GPU directive**: NO GPU launch was performed in this Wf-322D. The 15 FAIL pre-flight gates do NOT require GPU work to fix — they are CPU-only docs/test/import fixes. Per AGENTS.md §5.1, no GPU launch > 1 hr is needed for this Wf.
3. **No paper work without authorization**: ZERO touches to `paper/*` in this Wf-322D. Per TODO-84 ACTIVE rule, paper work remains user-authorization gated.
4. **Pre-flight v2/v3 25/25 PASS gate**: the pre-flight v3 fresh-run today **REGRESSED** to 42/57 PASS / 15 FAIL. This is HONEST-NEGATIVE disclosed in §3 above and in INDEX.md §22.16 + README.md pre-flight REGRESSION block. DO NOT paraphrase as "52/52 PRESERVED" without the 15-FAIL disclosure.
5. **Honest-negative framing**: 15 pre-flight FAILs are PENDING for future code-class Wf (Wf-323+ candidates). Wf-322D does NOT fix them.

---

## §5 What was NOT touched in this Wf-322D (per standing directives)

- `paper/*` (zero touches per TODO-84 ACTIVE)
- `molmetal/` source code (no edits; this Wf is pure docs-sync)
- `metrics/by_metric/*.json` (no metric JSON writes; pre-flight FAILs on 4 ADMET JSONs remain PENDING)
- `scripts/` (no script writes; pre-flight FAILs on `flow_matching` upstream `configure_reference` import + chemprop `--quick` smoke remain PENDING)
- `TODO/pending/` (no file moves; pending/ stays at 13 files)
- GPU launches (zero per AGENTS.md §5.1 kill_then_adapt directive)
- `molmetal/reports/wf{321,322}_*/` (no NEW report directories created)

---

## §6 Cross-references

| # | Cross-link | Section | Purpose |
|---:|---|---|---|
| 1 | `TODO/INDEX.md` §0.8 | §2.1 | Wf-321..322 ship wave timeline (NEW) |
| 2 | `TODO/INDEX.md` §0.9 | §2.1 | Wf-322D net effect (NEW) |
| 3 | `TODO/INDEX.md` §22.15 | §2.1 | Wf-321..322 detailed table (NEW) |
| 4 | `TODO/INDEX.md` §22.16 | §2.1, §3 | Honest framing disclaimer + 15-FAIL pre-flight breakdown (NEW) |
| 5 | `TODO/README.md` Wf-309..322 ship wave | §2.2 | All 14 Wf entries (Wf-309..322) reflected (NEW Wf-321/322 + Wf-318/319/320) |
| 6 | `TODO/README.md` pre-flight REGRESSION block | §2.2, §3 | 15-FAIL pre-flight honest-negative disclosure (NEW) |
| 7 | `molmetal/reports/wf322d_preflight/pre_flight.json` | §3 | Pre-flight v3 fresh-run evidence (42/15/0/57/wall=65.0s) |
| 8 | `TODO/completed/321{abcdefgh}_*.md` | §2.3 | 6 Wf-321 closeouts (321a/321b/321d/321e/321f/321h) |
| 9 | `TODO/completed/322{abcdefg}_*.md` | §2.3 | 7 Wf-322 closeouts (322a/322b/322c/322d/322e/322f/322g, including this file) |
| 10 | `TODO/completed/320_final_synthesis.md` | §2.3 | Wf-320I aggregation doc (predecessor) |
| 11 | `TODO/completed/316_todo_directory_final_state.md` | §2.3 | Wf-316F TODO directory synthesis (predecessor) |
| 12 | `.claude/skills/party-twenty-spirit` §3.5 | §4 | Honest framing mandate |
| 13 | `.claude/skills/mol-metal-gpu-execution` §0 | §4 | "Kill then adapt" GPU directive |
| 14 | `AGENTS.md` §1 standing rules | §4 | MEASURED-vs-PROJECTED + no paper + pre-flight 25/25 PASS gate |

---

**END OF §6** — Owner Hugo Lee; review `party-twenty-spirit` §3.5; Wf-322D = docs-only sync; 13 INDEX.md entries + 9 README.md lines + 1 closeout doc; pre-flight v3 fresh-run REGRESSION 42/57 PASS / 15 FAIL honest-negative disclosed in §3 + INDEX.md §22.16 + README.md pre-flight block; 15 FAIL root causes mapped to 6 Wf-323+ follow-up candidates.

**Word count**: ~900 (~50 lines). Dense by design — every claim traces to MEASURED source per `party-twenty-spirit` §3.5.
