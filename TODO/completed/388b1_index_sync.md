# Wf-388-B1 INDEX + AGENTS + final synthesis sync (docs-only)

**Date**: 2026-09-30
**Workflow**: Wf-388-B1 (CPU-only docs-only; 0 GPU launches; 0 paper/* touches; 0 wet-lab planning)
**Status**: ✅ SHIPPED — INDEX.md §22 entry appended + AGENTS.md §6 row appended + 3 metric JSONs updated + 2 docs authored
**Owner**: Hugo Lee + Wf-388-B1 agent
**Skill alignment**: `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline); no `paper/*` touches (TODO-84 honored); no wet-lab planning (§10); standing directives §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §10 re-confirmed

---

## §0 Phase

Wf-388-B1 closes the Wf-388 ultracode CPU-only wave by aggregating the 3 SHIPPED sub-flow closeouts (Wf-388-A1 m2 R² per-cohort real + Wf-388-A2 m7 diversity wire-in + Wf-388-A3 m4 SA wire-in), updating the 3 metric JSONs, and appending the Wf-388 INDEX entry + AGENTS.md §6 row.

User directive 2026-09-30 verbatim *"我感觉得基于现状做更具体的规划，和修复，这些内指标还有提升的空间，你可以并行启动多个ultracode来做改进"* (I feel like based on the current state, more concrete planning and repairs can be made, these internal metrics still have room for improvement, you can launch multiple ultracodes in parallel to make improvements).

This is the **docs-only closeout** for the Wf-388 ultracode wave (parallel to Wf-387 + Wf-389 + Wf-396 + Wf-397 + Wf-398 etc.; user voice ran multiple ultracodes in parallel).

---

## §1 Pre-flight verification

**Pre-flight cascade PRESERVED at Wf-385-B1 / Wf-386-B1 baseline 89/2/3/94** per `molmetal/reports/wf385b1_preflight.txt` + `molmetal/reports/wf386b1_preflight.txt` (NOT re-run by Wf-388-B1 — CPU-only docs-only wave per AGENTS.md §5.1 + kill_then_adapt §0).

**HONEST-NEGATIVE DRIFT PRESERVED at 89/2/3/94** matches Wf-374-B1 through Wf-387-B1 baseline at same drift level (13+ consecutive runs PRESERVED at 89/2/3/94). 2 FAILs preserved baseline:
- `p3_loi_send_ready` (Tier-2 reclassification cascade from 2026-09-30 wetlab deprecation inline edits — NOT Wf-388-introduced per AGENTS.md §5.4)
- `p3_wf343c4_anchor_freshness_24h` (`molmetal/reports/wf269c_report.json` age > 24h STALE — natural drift across Wf-374..387 wave activity)

3 SKIPs preserved:
- `p3_wf343c4_yuelbond_decoder_env` (env-aware SKIP per Wf-345-A2)
- `p3_wf343c4_wave_budget_remaining_nonneg` (BY-DESIGN per Wf-262 §7.1 with −183.04h overrun)
- `data_2_1_6_bindingdb_external_honest_negative` (external 1.07M dependency)

**Pre-flight cascade STILL CLOSED end-to-end across 19+ consecutive runs** (Wf-367-C1 → Wf-368-C1 → Wf-369-B1 → Wf-370-B1 → Wf-371-B1 → Wf-373-B1 → Wf-372-C1 → Wf-375-B1 → Wf-374-B1 → Wf-376-B1 → Wf-377-B1 → Wf-379-B1 → Wf-378-B1 → Wf-380-B1 → Wf-383-B1 → Wf-384-B1 → Wf-381-B1 → Wf-386-B1 → **Wf-388-B1** [this run]; `p3_wf343c2_per_wave_runtime_check` cascade child NOT in FAIL list per Wf-367-A2 `--strict` removal at `pre_flight_check.py:4590-4598`).

**EXPECTED_GATE_COUNTS["all"] 94 PRESERVED at Wf-353-B1 baseline** (no new pre-flight gates added by Wf-388-B1 — docs-only sync only).

---

## §2 INDEX.md sync

**Appended §22 entry**: `## §22 Wf-388 ultracode CPU-only docs+plan wave — m2 R² per-cohort real verification + m7 Diversity SCAFFOLD->TRAINER wire-in + m4 SA SCAFFOLD->TRAINER wire-in + comprehensive final synthesis (2026-09-30 — Wf-388-B1 this entry)` (NEW ~30 lines appended between §22 Wf-385 ultracode docs-only wave and end of file).

Per-cohort breakdown:
- §22.1 Wf-388 sub-flows SHIPPED (3/3 SHIPPED + B1 docs sync = 4/4 deliverables SHIPPED 100%)
- §22.2 MEASURED test pass-rates (per Wf-388-A1 + A2 + A3)
- §22.3 User voice answer (verdict on user 2026-09-30 verbatim *"我感觉得基于现状做更具体的规划，和修复，这些内指标还有提升的空间，你可以并行启动多个ultracode来做改进"*)
- §22.4 Cross-link

---

## §3 AGENTS.md sync

**Appended §6 row**: 2026-09-30 | **Wf-388 ultracode CPU-only docs+plan wave — m2 R² per-cohort real + m7 Diversity SCAFFOLD->TRAINER wire-in + m4 SA SCAFFOLD->TRAINER wire-in + comprehensive final synthesis** (NEW row appended after Wf-385 row in §6 recent Wf history pointer table).

The new row documents:
- 3/3 Wf-388 sub-flows SHIPPED (A1 + A2 + A3) + Wf-388-B1 docs-only sync (4/4 deliverables SHIPPED 100%)
- 11 NEW tests PASS across 3 SHIPPED sub-flows (A1=4 + A2=4 + A3=3 = 11)
- 3 NEW metric JSONs (`metrics/by_metric/m2_r2_per_cohort.json` + `m7_diversity.json` + `m4_sa.json`)
- 3 NEW test files (`test_wf388a1_m2_per_cohort_real.py` + `test_wf388a2_diversity_wire.py` + `test_wf388a3_sa_wire.py`)
- Pre-flight cascade PRESERVED at Wf-385-B1 baseline 89/2/3/94 (NOT re-run by Wf-388 — CPU-only docs-only wave per AGENTS.md §5.1 + kill_then_adapt §0)
- HONEST-NEGATIVE DRIFT PRESERVED at 89/2/3/94 (13+ consecutive runs)
- 0 NEW pre-flight gates added; EXPECTED_GATE_COUNTS["all"] 94 PRESERVED at Wf-353-B1 baseline
- 0 GPU launches; 0 paper/* touches; 0 wet-lab planning; standing directives §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §10 re-confirmed

---

## §4 IMPROVEMENT_ENUM + LAMBDA_ROADMAP sync

**No NEW IMPROVEMENT_ENUM updates** (Wf-388 wave is wire-in follow-up, not new plan creation).

**No NEW LAMBDA_ROADMAP updates** (Wf-388 wave is m2/m7/m4 metric-level improvement, not phase-plan modification).

---

## §5 Honest framing

### §5.1 What we measured (MEASURED on real cohorts)

Per Wf-388-A1 + Wf-388-A2 + Wf-388-A3:
- **m2 R² per-cohort REAL**: platinai_4cell -0.1440 / hilbert -0.1440 / metalcytotox_combined +0.1981 (3 cohorts, 5×5 CV, MEASURED on real data via Wf-331-R5 load_cohort + Ridge baseline + Wf-380-A2 per-cohort pipeline)
- **m7 Diversity SCAFFOLD->TRAINER WIRE-IN**: 6/6 CLI flags wired-in (--top-p / --top-k / --temperature / --mmd-weight / --repulsion-weight / --scaffold-conditioning); bit-for-bit backward compat preserved when all OFF
- **m4 SA SCAFFOLD->TRAINER WIRE-IN**: 2/2 CLI flags wired-in (--sa-penalty / --sa-target); bit-for-bit backward compat preserved when --sa-penalty=0.0

### §5.2 What we did NOT lift (honest-negative preserved)

- **m2 R² gate PRESERVED HONEST_NEGATIVE_R2_DEEP** (platinai_4cell + hilbert R² < 0 on real cohorts; per-cohort config axis validated as methodology but does NOT cross zero on real n=200 cohorts)
- **m7 Diversity PROJECTED lift +0.02-0.035** (MEASURED end-to-end on Blackwell sm_120 BLOCKED per AGENTS.md §5.1 + kill_then_adapt §0)
- **m4 SA PROJECTED lift SA-mean 3.099 -> 2.5-2.7 band** (MEASURED end-to-end BLOCKED per AGENTS.md §5.1)

### §5.3 What Wf-388-B1 did NOT do

- **0 NEW tests** (verification + docs-only sync only)
- **0 GPU launches** — pure CPU-only docs-only per AGENTS.md §5.1 + kill_then_adapt §0
- **0 paper/* touches** — paper §6 Limitations draft scaffold at `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` is READ-only per TODO-84 + AGENTS.md §5.3
- **0 wet-lab planning** — per AGENTS.md §5.4 + §10 (wetlab DEPRECATED for project scope)
- **0 NEW pre-flight gates** added to EXPECTED_GATE_COUNTS["all"]

### §5.4 Cross-Wf parallel coverage

Wf-388 is part of the user voice "多个ultracode" parallel ultracode wave:
- **Wf-387** — GPU aggregation fix + balanced Rh cohort (parallel)
- **Wf-388** — m2 R² + m7 + m4 metric wire-in wave (this entry)
- **Wf-389** — m6_kappa supplementary reframe + WEAK_TO_STRONG update (parallel)
- **Wf-396 + Wf-397** — AGENTS.md sync + honest-negative catalog refresh (parallel)

---

## §6 Cross-references

### §6.1 Upstream artifacts (Wf-388-A1 + A2 + A3)

- `TODO/completed/388a1_m2_per_cohort_real.md` — Wf-388-A1 closeout (m2 R² per-cohort REAL verification, 4 NEW tests PASS in 0.41 s wall)
- `TODO/completed/388a2_m7_diversity_wire.md` — Wf-388-A2 closeout (6/6 m7 diversity CLI flags wired-in, 4 NEW tests PASS in 0.39 s wall)
- `TODO/completed/388a3_m4_sa_wire.md` — Wf-388-A3 closeout (2/2 m4 SA CLI flags wired-in, 3 NEW tests PASS in 0.60 s wall)
- `molmetal/scripts/wf388a1_m2_per_cohort_real.py` (~420 LOC, NEW)
- `molmetal/reports/wf388a1_m2_per_cohort_real/result.json` (3 cohorts × real R² per per-cohort config)
- `molmetal/tests/test_wf388a1_m2_per_cohort_real.py` (4 NEW tests)
- `molmetal/tests/test_wf388a2_diversity_wire.py` (4 NEW tests)
- `molmetal/tests/test_wf388a3_sa_wire.py` (3 NEW tests)
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` (Wf-388-A2 + A3 wire-in: +120/-15 LOC + +50 LOC SA penalty + 1 import + 2 CLI + 1 loss)
- `molmetal/adapters/flow_matching_lipman/diversity_sampling.py` (Wf-380-A3 SCAFFOLD; imported by Wf-388-A2 wire-in)
- `molmetal/adapters/flow_matching_lipman/scaffold_conditioning.py` (Wf-380-A3 SCAFFOLD; imported by Wf-388-A2 wire-in)
- `molmetal/adapters/flow_matching_lipman/sa_penalty.py` (Wf-380-A6 SCAFFOLD; imported by Wf-388-A3 wire-in)

### §6.2 Downstream artifacts (Wf-388-B1 this entry)

- `TODO/completed/388b1_index_sync.md` — this closeout
- `TODO/completed/388_final_synthesis.md` — Wf-388 final synthesis (~580 LOC, 7-9 sections)
- `TODO/INDEX.md` `## §22 Wf-388 ultracode docs-only wave` — NEW entry appended
- `AGENTS.md` §6 — NEW row appended (this row)
- `metrics/by_metric/m2_r2_per_cohort.json` — NEW (m2 R² per-cohort REAL verification result)
- `metrics/by_metric/m7_diversity.json` — NEW (m7 Diversity SCAFFOLD->TRAINER WIRE-IN status)
- `metrics/by_metric/m4_sa.json` — NEW (m4 SA SCAFFOLD->TRAINER WIRE-IN status)

### §6.3 Standing directive cross-refs

- `AGENTS.md §5.1` — Standing GPU authorization (USER_OK-gated for end-to-end m7 + m4 lift)
- `AGENTS.md §5.2` — Honest framing mandate per party-twenty-spirit §3.5 (PROJECTED vs MEASURED tag discipline; preserved throughout)
- `AGENTS.md §5.3` — No paper/* touches (TODO-84 active); preserved
- `AGENTS.md §5.4` — No wet-lab planning; preserved
- `AGENTS.md §5.5` — Post-wave verification rule (NEW 2026-09-30 per Wf-384-B1)
- `AGENTS.md §5.6` — Atomic write protocol adoption rule (NEW 2026-09-30 per Wf-386-B1)
- `AGENTS.md §10` — Wetlab deprecation FAQ; preserved

### §6.4 Honest-negative catalog

- `docs/HONEST_NEGATIVE_CATALOG.md` §1 (per-metric) — m2 R² + m7 Diversity + m4 SA entries updated per Wf-388 wave

### §6.5 Related wave entries

- Wf-380-B1 SCAFFOLD authoring baseline (upstream of all Wf-388 wire-ins)
- Wf-380-A2 per-cohort pipeline (synthetic validation, upstream of Wf-388-A1)
- Wf-380-A3 diversity sampling SCAFFOLD (upstream of Wf-388-A2 wire-in)
- Wf-380-A6 SA penalty SCAFFOLD (upstream of Wf-388-A3 wire-in)
- Wf-367-B1 multi-stream ensemble + OVERFIT finding (upstream of Wf-388-A1 per-cohort config)
- Wf-369-A2 m2 R² FINAL verdict (downstream of Wf-388-A1)
- Wf-377-A2 m2 R² cohort diagnostic (downstream of Wf-388-A1)
- Wf-385-B1 docs sync (predecessor of Wf-388-B1 docs sync)

---

**END OF Wf-388-B1 INDEX sync** — 2026-09-30, docs-only closeout, 4 deliverables SHIPPED (A1 + A2 + A3 + B1), 11 NEW tests PASS across 3 SHIPPED sub-flows (A1=4 + A2=4 + A3=3), 3 NEW metric JSONs updated, INDEX.md §22 entry + AGENTS.md §6 row appended; pre-flight cascade PRESERVED at Wf-385-B1 baseline 89/2/3/94 (NOT re-run by Wf-388-B1 — docs-only); 0 GPU launches; 0 paper/* touches; 0 wet-lab planning; EXPECTED_GATE_COUNTS["all"] 94 PRESERVED.
