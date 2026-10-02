# diversity_subpocket — FINAL honest-negative close-out

**Date**: 2026-09-29
**Workflow**: Wf-318G (diversity_subpocket FINAL close-out per AGENTS.md §5.4)
**Status**: ✅ **CLOSED — NEGATIVE_RESULT_HONEST**
**Owner**: Hugo Lee (per user directive relayed through harness)
**Skill alignment**: `party-twenty-spirit` §3.5 (MEASURED-vs-DEFERRED tag discipline)

---

## §0 TL;DR

**Status**: `NEGATIVE_RESULT_HONEST`. The diversity_subpocket metric is
**wired and unit-tested but production 100-pocket Round-13 sweep was NOT
executed** — closed as honest-negative for Tier-2 scope per AGENTS.md §5.4.

- **MEASURED on disk**: `metrics/by_metric/diversity_subpocket.json` carries
  `verdict: DESIGN_ONLY` since 2026-09-15; only the smoke-unit-test fixture
  result is recorded (smoke-only: empty=NaN, N=1=0.0, 2-disjoint>0).
- **Status flag**: This Wf-318G pass adds the canonical `HONEST_NEGATIVE=true`
  + `honest_negative_reason` + `honest_framing_note` + `verdict: FAIL` schema
  (mirroring Wf-308B / Wf-311E Part 4 / Wf-314A pattern) and promotes the
  metric into the paper_dashboard.py §9 honest-negative catalog.
- **What is preserved**: the module + 5 unit tests + Phase 4 wire-in are
  SHIPPED and PRODUCTION-WIRED. Only the full pilot smoke at Round-12 10×3
  on real generated pool is NOT executed.

## §1 MEASURED state — what is on disk

### §1.1 JSON source verbatim (cited from `metrics/by_metric/diversity_subpocket.json`)

| Field | Value |
|---|---|
| `metric` | `diversity_subpocket` |
| `description` | `Per-residue sub-pocket fingerprint diversity (3D-aware local-patch complement to global Tanimoto IntDiv1). Mean pairwise Tanimoto distance on Morgan-ECFP4 fingerprints aggregated over per-residue 8.0 Å neighbourhood (Bemis-Murcko 1996 + Peter 2019 SPF).` |
| `unit` | `ratio (0-1); higher = more locally diverse` |
| `measurements` (only one entry) | `wf_phase3b_metrics_v2 unit tests (test_per_residue_diversity.py); value = smoke-only: empty=NaN, N=1=0.0, 2-disjoint>0; n_samples = test fixtures; status = DESIGN; date_utc = 2026-09-15` |
| `aggregate.all_measured_lambda_diversity_subpocket` | `DESIGN only (no full sweep yet)` |
| `aggregate.lit_target_targetdiff` | `TargetDiff does NOT report per-residue; closest analog is per-pocket ligand-COM distance — TargetDiff M2 mismatch flagged in TODO-22` |
| `verdict` | `DESIGN_ONLY — Phase-3D module + Phase-4 dispatch integration shipped; full pilot smoke at Round-12 10x3 not yet executed (per workflow #666). Honest framing: this metric is wired and unit-tested but production 100-pocket Round-13 sweep is BLOCKED pending F2(a) MetalLigandExchange SMARTS rule (TODO-29).` |

### §1.2 Wired-and-tested artefacts (SHIPPED, NOT removed)

Per `verdict` field, the following are SHIPPED on disk:

- `molmetal/molmetal_lam/sbdd_env/per_residue_diversity.py` — Phase 3D module
  (NEW).
- `molmetal/tests/test_per_residue_diversity.py` — unit tests (NEW).
- `molmetal/scripts/r4_lambda_only_run.py` — Phase 4 wire-in via
  `metric_subpocket_diversity_mean`.
- `wf_parallel_tasks/phase3d_subpocket_div.md` + `wf_parallel_tasks/phase4_integration.md`
  — workflow documents.

The smoke-only result is the test-fixture smoke, not a real generated pool
result.

## §2 What remains — Phase-3G 10×3 rotation (BLOCKED)

The Round-12 10×3 full pilot smoke on real generated pool is the
**forward-pointer** for this metric:

- **R16-W44** — Lambda 100×3 sweep at n_sim=1000 with sub-fix A+C+Deflex flags
  (TODO-26 master plan).
- **Phase-3G 10×3 rotation** (per Wf-311F §7 candidate #5) — BLOCKED on
  TODO-29 F2(a) MetalLigandExchange SMARTS rule (structural singleton-attractor
  fix) + GPU pipeline Phase 3.

**Blocker chain** (per AGENTS.md §5.1):

1. TODO-29 F2(a) NOT shipped (per `metrics/by_round/r13_100x3_partial.json`
   `path_forward` field #4).
2. GPU HSA userland outage unresolved.
3. User-OK required before any 5–7.5 hr GPU launch.

## §3 Per-Tier-2 honest framing (AGENTS.md §5.4)

Per AGENTS.md §5.4 (Tier-2 journal target; pure computational submission):

- **diversity_subpocket is NOT a Tier-2 acceptance gate.** Tier-2 is
  computational-only and accepts the metric as wired + smoke-tested. The
  global Tanimoto IntDiv1 (`diversity_tanimoto`) IS the headline metric on
  the cohort (per `metrics/by_metric/diversity_tanimoto.json`); the
  per-residue 8.0 Å sub-pocket variant is a §6 honest-limitation or §7
  future-work item, not a §4 blocker.
- **What DOES gate Tier-2 acceptance**: the metric is wired + unit-tested +
  Phase-4-integrated; the verdict field honestly states `DESIGN_ONLY`. No
  fabricated MEASURED value exists.
- **What does NOT gate Tier-2 acceptance**: the full pilot smoke on real
  generated pool (R16-W44).

**Honest-negative framing for §9 catalog (this Wf-318G)**:

> DO NOT cite diversity_subpocket on production 100-pocket cohort as a real
> per-residue diversity measurement (metric is wired and unit-tested but
> production Round-12 10×3 pilot smoke on real generated pool is NOT
> executed; only test-fixture smoke runs at empty=NaN, N=1=0.0,
> 2-disjoint>0; per-residue 8.0 Å aggregation is Phase-3D unit-tested
> only; full sweep BLOCKED on F2(a) MetalLigandExchange SMARTS rule +
> GPU pipeline Phase 3).

## §4 Recovery path — TODO-29 + GPU pipeline Phase 3

Per the GPU pipeline Phase 3 queue (TODO-26 R16 master plan):

| Recovery item | Owner | Dependency | Tier-2 gate? |
|---|---|---|---|
| Ship F2(a) MetalLigandExchange SMARTS rule | TODO-29 | Structural code change | NO (computational) |
| Phase-3G 10×3 rotation | Round-16 W44 | F2(a) ship + GPU recovery | NO (§7 future work) |
| Lambda 100×3 at n_sim=1000 | Round-16 W44 | F2(a) ship + GPU recovery | NO (§7 future work) |
| Cite-on-Round-13 30-cell production sample | Round-16 W45 | Round-13 re-run after F2(a) | NO (§7 future work) |

All items are §7 future-work for Tier-2. They re-enter Tier-1 acceptance
scope only if the user upgrades the journal target.

## §5 Cross-reference

- `metrics/by_metric/diversity_subpocket.json` — source-of-truth JSON (carries
  `verdict: DESIGN_ONLY` since 2026-09-15; this Wf adds HONEST_NEGATIVE tags).
- `molmetal/molmetal_lam/sbdd_env/per_residue_diversity.py` — Phase 3D module
  (SHIPPED).
- `molmetal/tests/test_per_residue_diversity.py` — unit tests (SHIPPED).
- `TODO/completed/14_full_100pocket_paper_r13.md` — TODO-14 (Round-13 master
  + per-pocket row gap §3.2 where diversity_subpocket is deferred).
- `TODO/completed/314a_wf311e_part4_deterministic_edits.md` §7 — Wf-311F §7
  candidate #4 ("Wf-311E diversity_subpocket + steric_clash + com_shift
  staging — DESIGN_ONLY / DEFERRED / NOT-staged").
- `AGENTS.md` §5.1 (kill_then_adapt) + §5.4 (Tier-2 journal target).
- `party-twenty-spirit` §3.5 (MEASURED-vs-DEFERRED tag discipline; data真实).

## §6 Honest framing checklist (party-twenty-spirit §3.5)

- [x] Every numeric value traces to a JSON path on disk (see §1.1).
- [x] `NEGATIVE_RESULT_HONEST` flag set in JSON + dashboard + closeout doc.
- [x] No fabricated / aspirational numbers in main results (smoke-only test
  fixtures are explicitly labelled).
- [x] Recovery path documented but NOT executed in this Wf (CPU-only doc
  edit + JSON tag + paper_dashboard.py wiring).
- [x] Cross-refs to Wf-314F + Wf-311F preserved verbatim.
- [x] Wired module + 5 unit tests preserved (NOT removed).

## §7 Files touched (Wf-318G)

| Path | Type | Edit |
|---|---|---|
| `metrics/by_metric/diversity_subpocket.json` | JSON | +4 fields (`HONEST_NEGATIVE=true`, `honest_negative_reason`, `honest_framing_note`, `verdict: FAIL`) |
| `molmetal/scripts/paper_dashboard.py` | Python | +1 HONEST_NEGATIVE_CATALOG entry (`hn_wf318g_diversity_subpocket_design_only`) |
| `TODO/completed/diversity_subpocket_final_honest_negative.md` | Doc | NEW — this file |
| `TODO/completed/318g_r13_diversity_steric_final.md` | Doc | NEW — Wf-318G summary closeout |

---

*Generated 2026-09-29 by Wf-318G close-out. Workstream CLOSED as honest-negative
for Tier-2 scope. Wired module + unit tests preserved; recovery path documented
via GPU pipeline Phase 3 (R16-W44).*
