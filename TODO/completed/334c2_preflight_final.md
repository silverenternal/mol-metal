# Wf-334-C2 — Pre-flight final verification (2026-09-29)

**Status:** SHIPPED (CPU-only measurement; pure pre-flight diff capture)
**Pipeline:** `molmetal/scripts/pre_flight_check.py` v2 (schema 207.v1)
**JSON carrier:** `/tmp/wf334c2_preflight.json`
**Reference doc:** `TODO/completed/207_unified_pre_flight_checklist_v2.md` (§3.1 chemprop, §4.2 CFM, §5 P2)

## §1 Headline (PLAIN TEXT)

- **Wf-322D baseline (2026-09-28):** 42 PASS / 15 FAIL / 0 SKIP / 57 TOTAL
- **Wf-330 final (per Wf-332-S4):** 62 PASS / 0 FAIL / 0 SKIP / 62 TOTAL
- **Wf-334-C2 final (this run, 2026-09-29T06:15:24Z):** **62 PASS / 0 FAIL / 0 SKIP / 62 TOTAL**

**Δ Wf-322D → Wf-334-C2:** +20 PASS, −15 FAIL, +5 TOTAL (5 NEW gates), 0 SKIP, 0 gates regressed.
**Δ Wf-330 → Wf-334-C2:** 0 PASS, 0 FAIL, 0 SKIP, 0 TOTAL — pre-flight STABLE since Wf-330 close.

## §2 Honest framing

- Pure measurement (no model retrain; no paper touches; no GPU).
- Pre-flight v2 25/25 PASS gate (TODO-207) is the only deliverable.
- All 62 gates hit in 120.34 s wall (`total_wall_sec = 120.3414`).
- 0 STALE flag (per §4 staleness = 24 hr rule).
- No fabricated metrics; every gate evidence field in JSON carrier is file-cited.

## §3 Per-gate attribution — 15 FAIL→PASS flips (Wf-322D → Wf-334-C2)

| Gate | Pipeline | Source Wf |
|---|---|---|
| `p1_chemprop_smoke` | chemprop_v2_v3 | Wf-263A (v3-fallback default) + Wf-323D (TODO-119 P1.4 CPU smoke) |
| `p2_tests_pass` | tests | Wf-258 → Wf-323g pytest accumulation (145 tests passing) |
| `p3_ablation_runner_tests_present` | wf319a | Wf-319A ablation runner test pack |
| `p3_admet_aq_sol_honest_negative_tag` | admet_druglikeness | Wf-316 §9 expansion (AMES / aq sol / hepato / LogP / κ honest-negative tags) |
| `p3_admet_hepatotox_honest_negative` | admet_druglikeness | Wf-316 §9 expansion |
| `p3_admet_logp7_4_honest_negative` | admet_druglikeness | Wf-316 §9 expansion |
| `p3_admet_m6_kappa_honest_negative` | admet_druglikeness | Wf-316 §9 expansion |
| `p3_frontier_math_day_3_5_integration_tests` | frontier | Wf-234 frontier-math CPU portion |
| `p3_gate_count_consistency_audit` | all | Wf-322D pre-flight gate-count audit |
| `p3_nci60_9cell_per_cell_aggregate` | nci60_9cell | Wf-220 temporal-split NCI-60 9-cell acceptance |
| `p3_pb_packages_available_or_documented` | wf315d | Wf-315D pb-packages availability audit |
| `p3_pb_packages_status_present` | wf319c | Wf-319C pb-packages status doc |
| `p3_r20_phase0_loaders_complete` | r20_phase0 | Wf-322E (TODO-148 loader API doc) + Wf-323 R20 phase0 loaders |
| `p3_smoke_fail_on_missing_audit` | smoke_hygiene | Wf-323G smoke fail-on-missing audit |
| `p3_todo_119_p14_cpu_smoke_ready` | chemprop_v2_v3 | Wf-261 + Wf-323D TODO-119 P1.4 CPU smoke |

**Total flipped FAIL → PASS = 15 (matches baseline 15 FAIL, all recovered).**

## §4 Per-gate attribution — 5 NEW gates added (Wf-322D → Wf-334-C2)

| Gate | Pipeline | Source Wf | Status |
|---|---|---|---|
| `p3_admet_4_below_documented` | admet_druglikeness | Wf-316 §9 expansion (4 below field-avg ADMET metrics) | PASS |
| `p3_catherine_30metal_closeout_documented` | catherine_30metal | Wf-322G (TODO-130 G3 closure + 30-metal closeout) | PASS |
| `p3_m8_full_pass_rate_documented` | wf323c | Wf-323C (M8 full CPU pass-rate documentation) | PASS |
| `p3_phase3g_f2a_runtime_unblock_verified` | f2a_runtime_unblock | Wf-323G (Phase3-G F2A runtime unblock) | PASS |
| `p3_wf327_cpu_num_workers_sufficient` | wf327 | Wf-327 (CPU num_workers ≥ 4 across 6 launchers) | PASS |

**Total NEW gates = 5. Total = 57 + 5 = 62. ✓**

## §5 Regression audit — 0 PASS→FAIL

- Wf-334-C2: 0 PASS→FAIL flips (zero regressions).
- Wf-330 → Wf-334-C2: 0 flips in either direction (pre-flight STABLE).

## §6 Pipeline distribution (Wf-334-C2, n=62)

| Pipeline | Gates |
|---|---:|
| chemprop_v2_v3 | 13 |
| admet_druglikeness | 9 |
| cfm | 6 |
| data | 5 |
| frontier | 3 |
| r20_phase0 | 2 |
| wf319a | 2 |
| tests / config / drivers / anchors / consumers / harness / diagnostic / ood_rdkit_fix / f2a_runtime_unblock / honest_framing / smoke_hygiene / nci60_9cell / loi_send_ready / wf315d / wf319c / wf319b / wf323c / wf321e / catherine_30metal / wf327 / wf323f / all | 1 each (= 23 × 1) |
| **Total** | **62** |

## §7 Priority distribution (Wf-334-C2, n=62)

| Priority | Gates |
|---|---:|
| P0 | 3 |
| P1 | 18 |
| P2 | 4 |
| P3 | 37 |
| **Total** | **62** |

## §8 Standing directives cross-check (AGENTS.md §1)

- **§1.4 Pre-flight v2 25/25 PASS gate (TODO-207):** ✓ 62/62 PASS — gate exceeded (62 vs minimum 25).
- **§1.2 MEASURED-vs-PROJECTED tag discipline:** ✓ Every value in this doc traces to JSON carrier `/tmp/wf334c2_preflight.json` + Wf-322D baseline `molmetal/reports/wf322d_preflight/pre_flight.json`.
- **§1.5 Honest-negative framing:** ✓ Wf-316 §9 expansion gates (AMES / aq sol / hepato / LogP / κ) carry honest-negative tags per Wf-316 §9 catalog.
- **§5.3 No paper work:** ✓ Zero `paper/*` touched. Pure pre-flight measurement.

## §9 Cross-references

- Wf-322D pre-flight JSON baseline: `/home/hugo/codes/mol-metal/molmetal/reports/wf322d_preflight/pre_flight.json`
- Wf-260 pre-flight v2 (historic reference): `/home/hugo/codes/mol-metal/molmetal/reports/wf260_preflight/pre_flight_v2.json`
- Wf-334-C2 JSON carrier: `/tmp/wf334c2_preflight.json`
- Pre-flight script: `molmetal/scripts/pre_flight_check.py` (schema_version 207.v1)
- Pre-flight reference doc: `TODO/completed/207_unified_pre_flight_checklist_v2.md`
- Wave sync context: `TODO/INDEX.md` §22.16 (Wf-322D honest-negative regression = 42/15/57)
- Per-Wf tracker: `TODO/completed/323{a..g}_*.md` (CPU closeouts) + `TODO/completed/324d_agents_refresh.md` (AGENTS.md sync)

## §10 Final verdict

```
WF-334-C2 PRE-FLIGHT v2 VERDICT:  62 PASS / 0 FAIL / 0 SKIP / 62 TOTAL  (Δ +20 / −15 / +5 vs Wf-322D)
                                         WALL: 120.34 s     STALE: false
```

**Outcome:** PRE-FLIGHT STABLE at 62/62 PASS since Wf-330. All 15 baseline FAILs recovered; 5 new gates added per Wf-316 / Wf-322G / Wf-323C / Wf-323G / Wf-327; zero regressions.

This is the **final pre-flight capture** for the Wf-309..334 ship wave.

**END Wf-334-C2** — Owner Hugo Lee; honest framing per AGENTS.md §1.5.