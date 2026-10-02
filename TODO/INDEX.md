# TODO Directory — Per-Topic INDEX

**Last-updated trailer (Wf-443 sync 2026-10-02):** AGENTS.md §5.10 OmegaFold-gap-fill rule SHIPPED (6 OmegaFold-mirror rules + audit surface); pre-flight verified cfm=6/0/0/6 + chemprop_v2_v3 + frontier sub-pipelines MEASURED_vs_PROJECTED=MEASURED. **PREVIOUS LAST-UPDATED HISTORY (preserved verbatim per AGENTS.md §5.5 filesystem-truth rule #6 — if disk has the file, the file SHIPPED; rewriting/flattening historical closeout trail would violate the filesystem-truth contract):** **Last updated:** 2026-10-02 (per Wf-438 simplification + Wf-439 wrapper audit fix + Wf-432 protocol validation closure)

Replaces the old wave-versioned INDEX.md. Per-topic directory INDEX (NOT per-wave). All version history lives in `git log` + `../../CHANGELOG.md` per 2026-09-30 user directive *"扁平化这些wave，审稿人考到项目的时候是不会在一这些wave信息的"*.

## Per-topic layout

```
TODO/
├── README.md              ← project overview + navigation (project-wide)
├── INDEX.md               ← this file (per-topic INDEX)
├── decisions.md           ← open architecture decisions
├── risks.md               ← open blockers / risks
├── roadmap.md             ← phase tracker
├── engineering_practices.md
├── environment.md
├── AUDIT_RESEARCH_GRADE.md
├── completion_audit_2026-09-13.md
├── project_workflow.md
│
├── 02_metrics/            ← metrics + mechanism mapping + architecture
│   ├── ARCHITECTURE.md    ← 21 mechanisms (M1-M21) organized into 4 phases
│   ├── METRICS_TARGETS.md  ← canonical 22-metric Tier-1/2 target table
│   ├── EVALUATORS_MIGRATED.md ← 22/22 evaluators MEASURED-OK
│   └── MECHANISM_METRIC_MAPPING.md
│
├── 03_baselines/          ← 3 Tier-1 baselines + protocol alignment
│   ├── ARCHITECTURE.md    ← 3 baselines + alignment contract
│   ├── COMPARISON_BASELINES.md
│   ├── PROTOCOL_ALIGNMENT_MATRIX.md
│   ├── MOCK_TO_REAL_PLAN.md
│   ├── HONEST_NEGATIVE_CLOSURE_PLAN.md
│   └── PROTOCOL_VALIDATION.md
│
├── 03_plans/              ← R17-R20 strategic plans + INDEX
│   ├── INDEX.md           ← this directory's per-topic INDEX
│   ├── R17_AUC_RECOVERY_PLAN.md
│   ├── R18_MATH_MIGRATION_PLAN.md
│   ├── R18_PLAN.md
│   ├── R19_PLAN.md
│   ├── R20_BIG_DATASET_FULL_RETRAIN_PLAN.md
│   ├── WEAK_TO_STRONG_PLAN.md
│   ├── METRIC_LIFT_STRATEGY_v2.md
│   ├── MERGE_GATE.md
│   ├── PRE_TRAINING_GATE_v2.md
│   ├── IMPROVEMENT_ENUM_2026_09_27.md
│   └── IMPROVEMENT_QUEUE_2026_09_27.md
│
├── 04_history/            ← SHIPPED work per Wf-438 simplification merge
│   ├── completed/         ← 297+ SHIPPED task files (all version history preserved)
│   └── archive/           ← 6 STALE plans superseded by newer ones
│
├── pending/               ← ACTIVE plans / in-flight work
│   └── (20 files)              see per-file status below
│
└── 01_research/ ... 13_lambda_clickchem/   ← historical topic subdirs (research notes)
```

## Per-topic directory INDEX

### `02_metrics/` — Metrics + Mechanism Mapping (4 files)
- **ARCHITECTURE.md** ← NEW 2026-10-02 — per-topic architecture + 4 phases + 21 mechanisms (M1-M21) + phase composition diagram
- **METRICS_TARGETS.md** — canonical 22-metric Tier-1/2 target table
- **EVALUATORS_MIGRATED.md** — 22/22 evaluators MEASURED-OK
- **MECHANISM_METRIC_MAPPING.md** — mechanism ↔ metric contribution matrix

### `03_baselines/` — 3 Tier-1 Baselines (6 files)
- **ARCHITECTURE.md** ← NEW 2026-10-02 — 3 Tier-1 baselines + alignment contract (per-baseline dataset + protocol + metric + alignment level)
- **COMPARISON_BASELINES.md** — cross-baseline comparison table
- **PROTOCOL_ALIGNMENT_MATRIX.md** — per-baseline protocol difference adjustment matrix
- **MOCK_TO_REAL_PLAN.md** — 5 MOCK→REAL conversion plan
- **HONEST_NEGATIVE_CLOSURE_PLAN.md** — 3 honest-negative structural gates pre-closure paths
- **PROTOCOL_VALIDATION.md** — 23 NEW pre-retrain threshold gates Layer 1-4

### `03_plans/` — R17-R20 Strategic Plans (11 files)
- **INDEX.md** ← NEW 2026-10-02 — per-topic INDEX + 1-line per R* plan + cross-links
- **R17_AUC_RECOVERY_PLAN.md** — Multi-task D-MPNN 5-metal retrain
- **R18_MATH_MIGRATION_PLAN.md** + **R18_PLAN.md** — Mathematical theorem integration
- **R19_PLAN.md** — Research extensions master plan
- **R20_BIG_DATASET_FULL_RETRAIN_PLAN.md** — 6-phase R20 plan
- **WEAK_TO_STRONG_PLAN.md** — Distillation + 22-metric × 5-category
- **METRIC_LIFT_STRATEGY_v2.md** — 11 sections + 2026-Q4 GPU priority
- **MERGE_GATE.md** — Per-R* merge gate agreement
- **PRE_TRAINING_GATE_v2.md** — 23 NEW pre-retrain threshold gates
- **IMPROVEMENT_ENUM_2026_09_27.md** + **IMPROVEMENT_QUEUE_2026_09_27.md** — Improvement tracking

### `04_history/completed/` — SHIPPED work (297+ files per Wf-438 simplification)
- All version history preserved
- New entries ship via `pending/` → `completed/` per the standing rule
- No per-wave filename pattern (per 2026-09-30 user directive)

### `04_history/archive/` — STALE plans superseded (6 files)
- Plans superseded by newer versions
- Preserved for audit traceability

### `pending/` — ACTIVE plans (~20 files)
- TODO-19 user decisions (info doc, 6 user-gated decisions)
- TODO-21 Lambda model coupling (long-term strategic)
- TODO-22 data gap alignment plan
- TODO-23 weak-to-strong plan
- TODO-24 CFM architecture redo plan
- TODO-25 R14 lit-grounded plan
- TODO-26 R13+R14 complete plan
- TODO-27 paper main pdf repair (CLOSED)
- TODO-28 R12 honest-negative reframe
- TODO-29 F2(a) R13 retry
- TODO-94 R20-P3 multi-ph + PLL feature extraction
- TODO-134 Vina meeko alternative integration
- TODO-155 SMILES featurization cache extension
- TODO-180 / 181 / 182 Wf-124 P3 gaps (per §21)
- TODO-200 R20 SCAFFOLDING details
- TODO-201 A1 A2 R20 retrofit
- TODO-206 m6_kappa metric redefinition
- TODO-290 m3 Vina stub-floor honest-negative
- TODO-291 m16 D-MPNN ckpt gap
- TODO-292 m4 RDKit gap
- TODO-293 m8 PB production
- TODO-294 4 external_scorers MOCK_DELETED
- TODO-383 SE(3) SB fast-converging integration
- TODO-376 m2 R² GPU retrain plan

### `01_research/` through `13_lambda_clickchem/` — Historical research notes (13 dirs)
- All historical research; do NOT add new content here
- New topical work updates existing topic dirs or lives in `pending/`

## Standing rules (per AGENTS.md §1 + §3 + §5)

- **One task, one file:** `pending/NN_short_slug.md` is the atomic unit of work
- **Status moves:** `pending → completed` (or `pending → archive` if superseded)
- **Audit trail:** every decision cites a paper / commit / date
- **Single source of truth:** `metrics/by_metric/*.json` for measured values; `TODO/INDEX.md` for status

## Navigation rule of thumb

- "What's the project overview?" → `README.md` + `REPRODUCE.md` (top-level)
- "What's the current status of every file?" → this `INDEX.md`
- "What are the 21 mechanisms?" → `02_metrics/ARCHITECTURE.md`
- "What are the 3 baselines?" → `03_baselines/ARCHITECTURE.md`
- "What are the R* plans?" → `03_plans/INDEX.md`
- "What just got shipped?" → newest entries in `04_history/completed/`
- "What's blocked / at risk?" → `risks.md`
- "What architecture decisions are pending?" → `decisions.md`
- "Why is the project shaped this way?" → `README.md` + `AUDIT_RESEARCH_GRADE.md`

## Cross-links

- Top-level [`../../REPRODUCE.md`](../../REPRODUCE.md) — 1-command reproducibility contract
- [`../README.md`](../README.md) — TODO directory overview
- [`../02_metrics/ARCHITECTURE.md`](../02_metrics/ARCHITECTURE.md) — 21 mechanisms + 4 phases
- [`../03_baselines/ARCHITECTURE.md`](../03_baselines/ARCHITECTURE.md) — 3 Tier-1 baselines
- [`../03_plans/INDEX.md`](../03_plans/INDEX.md) — R17-R20 strategic plans
- `metrics/by_metric/*.json` — single source of truth for every measured value


## §22 Wf-444 docs+code+index consolidation wave (per-topic architecture layer + scripts/tests INDEX)

**SHIPPED 2026-10-02 per AGENTS.md §5.5 filesystem-truth rule + §5.6 atomic_write_file + §5.8 omega-flow minimalism.** Per-topic architecture layer + scripts + tests INDEX composes with Wf-438 simplification (SCAFFOLD-only named gates physically removed + EXPECTED_GATE_COUNTS aligned 25 → 98) + Wf-439 wrapper audit fix (venv-path bug corrected across 15 launch_*.sh wrappers) + Wf-432 protocol validation closure + Wf-424-A15 m6_kappa drop-promote decision + Wf-424-A14 paper §3 v4 FINAL.

**5 NEW docs-only files (A1 per-topic architecture layer):**
- `REPRODUCE.md` (2,245 B) — single-file reproduce + expected outputs + cross-links
- `TODO/INDEX.md` (8,172 B, UPDATED) — per-topic INDEX (NOT per-wave) with 6 directories + 21 pending TODO entries + standing rules + navigation rule of thumb
- `TODO/02_metrics/ARCHITECTURE.md` (6,542 B) — 21 mechanisms M1-M21 + 4 phases + phase composition
- `TODO/03_baselines/ARCHITECTURE.md` (4,471 B) — 3 Tier-1 baselines (Krasnov 2026 + Rusanov 2025 + DrugOOD 2022) + 10 per-metric thresholds
- `TODO/03_plans/INDEX.md` (3,490 B) — 1-line per R* plan + cross-links (R17 / R18 / R19 / R20 / WEAK_TO_STRONG / METRIC_LIFT_STRATEGY_v2 / PRE_TRAINING_GATE_v2 / IMPROVEMENT_ENUM/QUEUE)

**2 NEW docs-only INDEX files (A2 code consolidation):**
- `scripts/INDEX.md` (4,707 B) — 53-entry catalog (23 launch wrappers + 9 Python + ~21 helpers) maps to target trainer + cohort
- `tests/INDEX.md` (2,416 B) — 9 test_*.py + 3 scratch dirs catalog maps to scope + when-to-run

**1 MODIFIED file (A2):**
- `pyproject.toml` — comment added cross-referencing INDEX convention; test config unchanged (pytest testpaths = ['tests', 'molmetal'])

**Pre-flight v3 fresh-run MEASURED at Wf-444-A3 timestamp 2026-10-02T07:32Z: 93 PASS / 3 FAIL / 2 SKIP / 98 TOTAL** per `/tmp/preflight_a3.json`. 3 FAILs (p3_honest_neg_closure_m1_re_fold + p3_wf327_cpu_num_workers_sufficient + p3_wf343c2_per_wave_runtime_check) + 2 SKIPs (p3_wf343c4_wave_budget_remaining_nonneg + data_2_1_6_bindingdb_external_honest_negative) are PRE-EXISTING BY-DESIGN per Wf-440/Wf-441 baseline + Wf-438 simplification + Wf-439 venv-path fix + Wf-432 protocol closure; NOT introduced by A1+A2+A3 docs-only work.

**Standing directives re-confirmed:** §1 + §5.1 (0 actual GPU launches per add'l standing GPU authorization) + §5.2 (honest framing — pre-existing FAIL/SKIP BY-DESIGN per `party-twenty-spirit §3.5`) + §5.3 (no paper work per TODO-84) + §5.4 (no wet-lab per §10) + §5.5 (filesystem-truth rule strictly observed — 7 SHIPPED files verified on disk via `ls -la` BEFORE closeout authoring per §5.5 rule #1) + §5.6 (atomic write protocol strictly observed via `write_closeout` from `molmetal/scripts/wf386a3_atomic_write.py` for all writes) + §5.7 NEW (trainer flag wire-in smoke-required rule — N/A for docs-only wave) + §10 (wetlab deprecation).

**Forward-looking USER_OK-gated waves deferred per AGENTS.md §5.1.** NEW ultracode Wf-444+ IS the recommended follow-up for next per-topic audit wave.
