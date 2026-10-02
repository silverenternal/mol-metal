# r13_100x3_partial — FINAL honest-negative close-out

**Date**: 2026-09-29
**Workflow**: Wf-318G (r13_100x3_partial FINAL close-out per AGENTS.md §5.4 Tier-2 directive)
**Status**: ✅ **CLOSED — NEGATIVE_RESULT_HONEST**
**Owner**: Hugo Lee (per user directive relayed through harness)
**Skill alignment**: `party-twenty-spirit` §3.5 (MEASURED-vs-DEFERRED tag discipline)

---

## §0 TL;DR

**Status**: `NEGATIVE_RESULT_HONEST`. The Round-13 100×3 sweep **is not paper-grade
300 cells in this session**; the partial result is preserved on disk and the
workstream is **closed as honest-negative for Tier-2 scope** per AGENTS.md §5.4.

- **MEASURED on disk**: `metrics/by_round/r13_100x3_partial.json` (delivered
  2026-09-15, n_evaluations_total = 30/300 cells, pb_path wall_seconds_mean = 7.75,
  cfm_path_status = `BLOCKED`, lambda_path_status = `missing`).
- **Status flag**: `verdict.status = NEGATIVE_RESULT_HONEST` (already present
  in by_round JSON since 2026-09-15; this Wf-318G pass promotes it into the
  paper_dashboard.py catalog as a §9 honest-negative badge).
- **What is preserved**: path-A 10×3 diversity lift (n_distinct 1→20, div_tan
  0→0.1065) on 10 specific pockets (test_000..test_009) — does NOT generalize
  to 100 pockets when search-bound reappears on test_010..test_019.

## §1 MEASURED state — what is on disk

### §1.1 by_round file (source of truth)

`metrics/by_round/r13_100x3_partial.json` carries the round-level audit
verbatim. Key fields (cited verbatim):

| Field | Value |
|---|---|
| `phase_results.phase_2_sweep_execution` | `PARTIAL — process killed mid-run at 55% (no final report.json)` |
| `phase_results.phase_3_paper_integration` | `DELIVERED — paper §4.2/§4.3/§4.6 updated with honest framing` |
| `aggregate_metrics.n_evaluations_total` | `30` |
| `aggregate_metrics.lambda_path_evaluated` | `0` |
| `aggregate_metrics.pb_path_evaluated` | `30` |
| `aggregate_metrics.pb_path_n_no_candidates` | `15` |
| `aggregate_metrics.pb_path_n_seed_only` | `15` |
| `aggregate_metrics.pb_path_n_docked` | `0` |
| `aggregate_metrics.pb_path_pb_pass_rate_aggregate` | `null` |
| `aggregate_metrics.pb_path_wall_seconds_mean` | `7.75` |
| `aggregate_metrics.cfm_path_status` | `BLOCKED (ModuleNotFoundError + GPU HSA userland outage)` |
| `aggregate_metrics.paper_grade_data_ready` | `false` |
| `verdict.status` | `NEGATIVE_RESULT_HONEST` |

### §1.2 Honest deviations from spec

From `molmetal/reports/wf_round13_100x3/final.md` (cross-ref) and
`TODO/completed/14_full_100pocket_paper_r13.md` (TODO-14):

- `lambda_n_sim`: spec = 1000, executed = 200 (reached pocket test_055/165 cells
  before being killed).
- `pb_n_pockets`: spec = 100, executed = 10 (per-cell cost dominated by MCTS
  search, not Vina/PB; search-bound collapse).
- `cfm_path`: spec = run full sweep, BLOCKED (r10_cfg_real_crossdocked.py import
  fails + dGPU HSA userland outage).

### §1.3 Path forward (preserved verbatim, NOT executable in Wf-318G)

1. Serialize Lambda runs (kill PID 110660, launch single n_sim=200 sweep,
   ETA ~110 min).
2. Fix r10_cfg_real_crossdocked.py import (one-line sys.path or PYTHONPATH).
3. PB path needs MCTS search budget lift before re-running at scale (per
   wf_lambda_fix_full_path_v2).
4. **F2(a) MetalLigandExchange SMARTS rule is structural fix for singleton
   attractor** — currently NOT shipped. Per Wf-314A cross-ref this is the
   GPU pipeline Phase 3 dependency.

## §2 What remains — Phase-3G 10×3 rotation (BLOCKED)

The Round-13 300-cell paper-grade sweep is the **forward-pointer** for the
GPU pipeline Phase 3 workstream. Per `TODO/completed/14_full_100pocket_paper_r13.md`:

- **R16-W42** — YuelBond retrain 10000-step h=128 (target: `decode_ratio ≥ 0.5`)
- **R16-W43** — CFM path import fix + 10000-step retry
- **R16-W44** — Lambda 100×3 sweep at n_sim=1000 with sub-fix A+C+Deflex flags

**Blocker chain** (per AGENTS.md §5.1 kill_then_adapt):
1. TODO-29 F2(a) MetalLigandExchange SMARTS rule NOT shipped (structural
   singleton-attractor fix).
2. GPU HSA userland outage unresolved on ROCm 7.2 / RX 7800 XT gfx1101 wave64.
3. User-OK required before any 5–7.5 hr GPU launch per AGENTS.md §5.1.

## §3 Per-Tier-2 honest framing (AGENTS.md §5.4)

Per AGENTS.md §5.4 (Tier-2 journal target; pure computational submission):

- **r13_100x3 paper-grade 300 cells is NOT a Tier-2 acceptance gate.** Tier-2
  is computational-only; cite-only SOTA + the partial path-A 10×3 (n_distinct
  1→20) + R15 structural ship + the DESIGN cell tags in paper §4.2/§4.3/§4.6
  are sufficient for Tier-2 acceptance.
- **What DOES gate Tier-2 acceptance**: the `paper_grade_data_ready=false` cell
  tag audit (per TODO-14 §4.2/§4.3/§4.5 status table) is honest and complete.
  Every cell carries the appropriate DESIGN / SEARCHONLY / SEARCHONE / CITEDONLY
  tag; no fabricated MEASURED value exists.
- **What does NOT gate Tier-2 acceptance**: the forward-pointer to R16 GPU
  retrain (TODO-26 master plan). This is a §7 future-work item, not a §4
  blocker.

**Honest-negative framing for §9 catalog (this Wf-318G)**:

> DO NOT cite r13_100x3 as a paper-grade 300-cell sweep (delivered 30/300
> cells; 30/30 PB-path cells search-bound; CFM-path BLOCKED on r10 import +
> GPU HSA userland outage; process killed at 55% before lambda final report.json;
> paper_grade_data_ready = false). The path-A 10×3 (n_distinct 1→20, div_tan
> 0→0.1065) is the headline result on 10 specific pockets only; it does NOT
> generalize to 100 pockets when search-bound reappears on test_010..test_019.

## §4 Recovery path — TODO-29 + GPU pipeline Phase 3

Per the **GPU pipeline Phase 3 already in queue** (TODO-26 R16 master plan +
Wf-314A §7 close-out):

| Recovery item | Owner | Dependency | Tier-2 gate? |
|---|---|---|---|
| Ship F2(a) MetalLigandExchange SMARTS rule | TODO-29 | Structural code change | NO (computational) |
| GPU HSA userland recovery | TODO-22 / Round-16 W42 | Cold power cycle OR iGPU gfx1100 fallback | NO (cite-only OK) |
| Lambda 100×3 at n_sim=1000 | Round-16 W44 | F2(a) ship + GPU recovery | NO (§7 future work) |
| CFM 10000-step retrain | Round-16 W42-W43 | YuelBond decoder + sub-fix C | NO (cite-only OK) |
| PB MMFF94 production | Round-16 W44-W45 | MCTS budget lift | NO (cite-only OK) |

All 5 items are §7 future-work for Tier-2. They re-enter Tier-1 acceptance
scope only if the user upgrades the journal target.

## §5 Cross-reference

- `molmetal/reports/wf_round13_100x3/final.md` — Phase 3 audit report.
- `metrics/by_round/r13_100x3_partial.json` — source-of-truth JSON (verdict.status
  = `NEGATIVE_RESULT_HONEST` since 2026-09-15).
- `TODO/completed/14_full_100pocket_paper_r13.md` — TODO-14 (Round-13 master).
- `TODO/completed/314a_wf311e_part4_deterministic_edits.md` §7 — Wf-311F §7
  candidate #3 ("r13_100x3_partial paper-grade 300 cells — CPU-only").
- `TODO/pending/26_round13_round14_complete_plan.md` — R16 master plan
  (forward-pointer).
- `AGENTS.md` §5.1 (kill_then_adapt) + §5.4 (Tier-2 journal target; wet-lab
  planning NOT required).
- `molmetal/reports/wf_round13_100x3/final.md` — verdict + path forward.
- `party-twenty-spirit` §3.5 (MEASURED-vs-DEFERRED tag discipline; data真实).

## §6 Honest framing checklist (party-twenty-spirit §3.5)

- [x] Every numeric value traces to a JSON path on disk (see §1.1).
- [x] `NEGATIVE_RESULT_HONEST` flag set in JSON + dashboard + closeout doc.
- [x] No fabricated / aspirational numbers in main results (every cell carries
  DESIGN / SEARCHONLY / SEARCHONE / CITEDONLY tag).
- [x] Recovery path documented but NOT executed in this Wf (CPU-only doc
  edit + paper_dashboard.py wiring).
- [x] Cross-refs to Wf-314F + Wf-311F preserved verbatim.

## §7 Files touched (Wf-318G)

| Path | Type | Edit |
|---|---|---|
| `metrics/by_metric/r13_100x3_partial.json` | JSON | NEW — Tier-2 honest-negative schema (mirrors by_round source verbatim) |
| `molmetal/scripts/paper_dashboard.py` | Python | +1 HONEST_NEGATIVE_CATALOG entry (`hn_wf318g_r13_100x3_not_paper_grade`) |
| `TODO/completed/r13_100x3_final_honest_negative.md` | Doc | NEW — this file |
| `TODO/completed/318g_r13_diversity_steric_final.md` | Doc | NEW — Wf-318G summary closeout |

---

*Generated 2026-09-29 by Wf-318G close-out. Workstream CLOSED as honest-negative
for Tier-2 scope. Recovery path documented; no GPU launches, no compute, no
fabricated numbers.*
