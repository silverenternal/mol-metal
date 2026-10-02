# Wf-307E — Actionable-now batch 2 synthesis (Wf-307A→D aggregate)

**Date**: 2026-09-29
**Owner**: Hugo Lee (synthesis, Read-only)
**Upstream chain**: Wf-307A (Wf-134 P1 ref SMILES extract) / Wf-307B (TODO-178b smoke-test FAIL-on-missing) / Wf-307C (ADMET pass-through integration) / Wf-307D (TODO-234 frontier math CPU portion)
**Parent audit chain**: Wf-303A audit → Wf-303B categorize → Wf-303C implement → Wf-303D verify → Wf-303E synthesis → Wf-307 batch 2 (this doc)
**Skill alignment**: `party-twenty-spirit` §3.5 honest framing + `mol-metal-gpu-execution` §0 "kill then adapt" GPU directive.

---

## §0 TL;DR (8-line verdict)

| Field | Value |
|---|---|
| **n_agents** | **4** (Wf-307A / B / C / D) |
| **n_completed** | **1** (Wf-307D — TODO-234 frontier math CPU portion SHIPPED) |
| **n_failed** | **0** (none crashed; A/B/C simply did not produce on-disk evidence) |
| **n_partial** | **0** (no partial ships — A/B/C produced no evidence, so partial is N/A) |
| **n_not_executed** | **3** (Wf-307A, Wf-307B, Wf-307C) |
| **actionable_now_closed** | **1 of 4** (Wf-307D only) |
| **actionable_now_remaining** | **3** (Wf-307A, Wf-307B, Wf-307C) |
| **actionable_now_drained** | **NO** — bucket NOT yet empty (3 CPU items still actionable) |

---

## §1 Per-agent summary (Read-only evidence scan)

| Agent | Brief | On-disk evidence | Status |
|---|---|---|---|
| **Wf-307A** | Wf-134 Phase 1 ref SMILES extraction (5/5 pockets; ~280 LOC + ≥12 tests; CPU) | **NONE** — no `molmetal/scripts/wf134_p1_extract_ref_smiles.py`, no `wf134_p1_ref_smiles.csv`, no `molmetal/tests/test_wf134_p1_extract_ref_smiles.py`, no `TODO/completed/134_*.md` | **NOT EXECUTED** |
| **Wf-307B** | TODO-178b smoke-test FAIL-on-missing hardening (party-twenty-spirit §3.5 hygiene; 0.5 day CPU) | **NONE** — no `molmetal/tests/test_smoke_fail_on_missing_audit.py`, no `TODO/completed/178b_*.md` | **NOT EXECUTED** |
| **Wf-307C** | ADMET pass-through integration into `pre_flight_check.py` + `paper_dashboard.py` (wire-only) | **PARTIAL upstream only** — `molmetal/scripts/admet_pass_through.py` + 3 metric JSONs exist from Wf-303C T2; **NO** new ADMET gates in `pre_flight_check.py` (1388 LOC scanned — `gate_ames_mutagen` / `gate_cell_permeability_logPapp` / `gate_aqueous_solubility_logS` absent); `paper_dashboard.py` not verified | **NOT EXECUTED** |
| **Wf-307D** | TODO-234 frontier math wire-in CPU portion (4 adapter families + smoke) | ✅ `molmetal/tests/test_wf307d_frontier_adapters_smoke.py` (7 tests, all PASS, 0.19s) + `TODO/completed/234_wf234_frontier_math_cpu_portion.md` | **SHIPPED** — honest note: expected class aliases (`MultiPHScheduler` etc.) did not exist; shipped classes are `MultiPHPocketFeatures` / `SE3LipmanSolver` / `SE3SBVelocityField` / `TransportRelaxationTerm`; GPU portion deferred per R17 invariant |

---

## §2 Actionable-now closed (Wf-307 batch 2)

1. ✅ **Wf-307D / TODO-234 (CPU portion)** — 4 adapter families import + forward-pass smoke verified; 7 tests PASS in 0.19s; close-out at `TODO/completed/234_wf234_frontier_math_cpu_portion.md`. GPU portion of full TODO-234 wire-in remains BLOCKED on R17 invariant + ultracode launch authorization (NOT actionable-now).

---

## §3 Remaining actionable-now (post-Wf-307)

Per Wf-303B §3.1 + Wf-303E §1, after Wf-307 batch 2 the actionable-now bucket is **NOT drained** — 3 CPU-only items remain:

| # | ID | Task | Effort | Why still actionable |
|---:|---|---|---|---|
| 1 | **Wf-307A / Wf-134 Phase 1** | m15 RMSD ref SMILES extraction (5/5 pockets, RDKit + CSV) | 1-2 days CPU | 5/5 pockets READY per Wf-126 P1; no auth gate; no GPU; feeds paper §4.5.7 + unblocks Phase 2/3 |
| 2 | **Wf-307B / TODO-178b** | smoke-test FAIL-on-missing hardening (audit + meta-test) | 0.5 day CPU | party-twenty-spirit §3.5 hygiene; no GPU; Wf-303C verified `_metalcytotox_preflight` raises but no global audit exists |
| 3 | **Wf-307C / ADMET pass-through integration** | wire 3 ADMET gates into `pre_flight_check.py` + 3 claim cards into `paper_dashboard.py` | 0.5-1 day CPU | JSONs already MEASURED in `metrics/by_metric/`; pure integration; no new measurement; `aqueous_solubility_logS -7.612` stays visible as honest-negative |

**Plus the original 3 still-actionable from Wf-303E §5** (re-listed for completeness, none of which were touched by Wf-307 batch 2):
- TODO-138 (Paper v2 + JCIM submission) — **BLOCKED by rule 84**; not CPU-actionable without authorization.
- TODO-234b (paper §3 method-ref for 3 research-only modules) — **BLOCKED by rule 84**; not actionable-now.
- TODO-84 (sweep `pending/` for rule-84 violators) — meta-hygiene; **actionable-now but trivial**.

---

## §4 Blocked-on-gpu (unchanged from Wf-303E §4 — none actioned by Wf-307 batch 2)

| ID | Wall-time | Resource | Why blocked | Rank |
|---|---|---|---|---|
| **TODO-219 Ru re-run** | ~20 min | RTX 5090 GPU-1 | > 1 hr launch rule; 3/3 timeouts at 43/50ep | #1 (smallest GPU ask) |
| TODO-180 | 1.8 hr | RTX 5090 GPU-1 | > 1 hr launch | #2 |
| TODO-179 | 4-5 hr | RTX 5090 | > 1 hr launch | #3 |
| TODO-181 | 5 hr | RTX 5090 GPU-1 | > 1 hr launch | #4 |
| TODO-182 | 7.5 hr | RTX 5090 | > 1 hr launch | #5 |
| TODO-234 wire-in (full) | 5-7 days | CPU + GPU | R17 invariant forbids locked-`__init__.py` edit | #6 |
| TODO-135 Phase 2 | 2 weeks | RTX 5090 / Blackwell | > 1 hr launch | #7 |

---

## §5 Honest framing (party-twenty-spirit §3.5)

- **MEASURED today**: 1 of 4 Wf-307 agents shipped (Wf-307D only) — 7/7 smoke tests PASS, GPU portion explicitly DEFERRED per R17 invariant; honest note that expected class aliases did not exist (shipped via wrapper class `ChempropV2Frontier` per Wf-237, not via lazy import).
- **NOT MEASURED today**: Wf-307A / B / C produced no on-disk evidence (no scripts, no tests, no close-out docs, no pre-flight gate additions). They are classified as **NOT EXECUTED** rather than **FAILED** because no failure mode was reported.
- **PROJECTED (not MEASURED)**: Wf-307A → ref SMILES for 5/5 pockets; Wf-307B → meta-test `test_smoke_fail_on_missing_audit.py` PASS; Wf-307C → 3 ADMET gates in `pre_flight_check.py` + 3 claim cards in `paper_dashboard.py`.
- **No silent skip**: Wf-307D test file `test_wf307d_frontier_adapters_smoke.py` genuinely runs 7 tests; Wf-307A/B/C absence is reported explicitly here (not masked as PASS).
- **Standing directive**: TODO-219 Ru re-run (~20 min, smallest GPU ask) is the recommended first GPU pick when user OK arrives; actionable-now bucket on CPU side has 3 items still pending (Wf-307A / B / C).

---

## §6 Cross-references

- Wf-303A audit: `molmetal/reports/wf303a_todo_audit/audit_list.md` (11 code tasks identified)
- Wf-303B categorize: `molmetal/reports/wf303b_categorize/categorize.md` (8 actionable-now ranked)
- Wf-303C implement: `molmetal/reports/wf303c_implement/task_report.md` (TODO-178 + ADMET + TODO-219 driver)
- Wf-303D verify: `molmetal/reports/wf303d_verify/verify_report.md` (25/25 preflight PASS; actionable-now drained verdict)
- Wf-303E synthesis: `molmetal/reports/wf303_synthesis/audit_summary.md` (5 closed + 1 partial + 7 blocked)
- Wf-307D close-out: `TODO/completed/234_wf234_frontier_math_cpu_portion.md` (the 1 of 4 ship from Wf-307 batch 2)
- Wf-307D test: `molmetal/tests/test_wf307d_frontier_adapters_smoke.py` (7 tests, 0.19s)
- TODO-178 close-out: `TODO/completed/178_metalcytotox_csv_hygiene_closeout.md` (from Wf-303C)
- Pre-flight v2: `TODO/completed/207_unified_pre_flight_checklist_v2.md`
- GPU directive: `TODO/INDEX.md` BD-5
- Honest framing policy: `TODO/03_plans/R18_PLAN.md §honest` + `party-twenty-spirit` §3.5