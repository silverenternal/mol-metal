# Wf-321B: R20 + R21 Skill Ecosystem Audit — TODO-99 (8 NEW skill audit)

**Date**: 2026-09-29 (Wf-321B ship)
**Status**: ✅ **SHIPPED** (4 NEW skill stubs + 16/16 stub tests PASS + pre-flight 21/21 PASS)
**Parent doc**: `TODO/03_plans/LAMBDA_ROADMAP.md` §1.5 (R20 4 NEW skill 候选) + §2.4 (R21 4 NEW skill 候选)
**Author**: ultracode Wf-321B subagent
**Cross-ref**: AGENTS.md §3 (skill catalog) + TODO-99 (8 NEW skills, S1-S8)

---

## §0 Why this audit

Per `LAMBDA_ROADMAP.md` §1.5 + §2.4:
- **R20 (2027-01 → 2027-04)**: 4 NEW skill candidates (`variance-puct`, `hilbert-astsr-e2e`, `closure-theorem`, `multiph-pll`)
- **R21 (2027-04 → 2027-07)**: 4 NEW skill candidates (`score-rep-variance-mcts`, `mcts-replay-learner`, `closure-dblock-bench`, `triangle-closure`)

**TODO-99 = 8 NEW skills (S1-S8) covering all Lambda gaps.**

**Audit scope**: identify which candidates already exist in `.claude/skills/` (post-Wf-296 ship wave) vs which need STUB creation.

---

## §1 Audit results

### §1.1 Existing skills inventory (24 skills verified)

| # | Skill | Type | Created | Source |
|---|---|---|---|---|
| 1 | `mol-metal-anticancer` | domain | 2026-09-26 | hot-path |
| 2 | `mol-metal-catherine-dblock-bench` | R20-P5 axis | 2026-09-28 | Wf-256/257 |
| 3 | `mol-metal-cfm-training` | domain (hot-path) | 2026-09-26 | hot-path |
| 4 | `mol-metal-click-chemistry` | domain | 2026-09-26 | hot-path |
| 5 | `mol-metal-closure-catherine` | domain | 2026-09-26 | Wf-43 ship |
| 6 | `mol-metal-closure-theorem` | R20 NEW | 2026-09-26 | R17 + Wf-37 ship |
| 7 | `mol-metal-gpu-execution` | hot-path | 2026-09-26 | hot-path |
| 8 | `mol-metal-hierarchical-multimetal` | R20-P4 | 2026-09-26 | Wf-42 ship surface |
| 9 | `mol-metal-hilbert-astsr-e2e` | R20 NEW | 2026-09-26 | R17 + Wf-37 ship |
| 10 | `mol-metal-hilbert-astsr-r2` | R20-P2 axis | 2026-09-28 | Wf-256/257 |
| 11 | `mol-metal-lambda-calculus` | domain | 2026-09-26 | hot-path |
| 12 | `mol-metal-mcts-replay-learner` | R21 NEW | 2026-09-26 | R17 + Wf-37 ship |
| 13 | `mol-metal-mechanism-mapping` | domain | 2026-09-26 | hot-path |
| 14 | `mol-metal-multiph-pll` | R20 NEW | 2026-09-26 | R17 + Wf-37 ship |
| 15 | `mol-metal-multi-ph-pll-feature` | R20-P3 axis | 2026-09-28 | Wf-256/257 |
| 16 | `mol-metal-neural-ode-kinetics` | R20-P4 | 2026-09-26 | Wf-42 ship surface |
| 17 | `mol-metal-paper-writing` | hot-path | 2026-09-26 | hot-path |
| 18 | `mol-metal-pareto-multiobj` | R20-P4 | 2026-09-26 | Wf-77 R22-P3 |
| 19 | `mol-metal-r18-math` | domain | 2026-09-26 | hot-path |
| 20 | `mol-metal-triangle-closure` | R21 NEW | 2026-09-26 | R17 + Wf-37 ship |
| 21 | `mol-metal-variance-puct-default` | R20-P1 axis | 2026-09-28 | Wf-256/257 |
| 22 | `mol-metal-wetlab-bayesian` | domain | 2026-09-26 | hot-path |
| 23 | `mol-metal-wetlab-protocol` | domain | 2026-09-26 | hot-path |
| 24 | `party-twenty-spirit` | hot-path (mandatory) | 2026-09-26 | hot-path |

**n_existing_verified = 24** (full audit on disk + in SKILL.md frontmatter).

### §1.2 R20 §1.5 candidates — coverage matrix

| §1.5 candidate | Existing skill | Status |
|---|---|---|
| `mol-metal-variance-puct` (NEW R20) | `mol-metal-variance-puct-default` (Sep 28 stub) | ✅ COVERED |
| `mol-metal-hilbert-astsr-e2e` (NEW R20) | `mol-metal-hilbert-astsr-e2e` (Sep 26 full) | ✅ COVERED |
| `mol-metal-closure-theorem` (NEW R20) | `mol-metal-closure-theorem` (Sep 26 full) | ✅ COVERED |
| `mol-metal-multiph-pll` (NEW R20) | `mol-metal-multiph-pll` (Sep 26 full) | ✅ COVERED |

**R20 coverage = 4/4 (100%)** — no NEW R20 skill STUB needed for §1.5 candidates.

### §1.3 R21 §2.4 candidates — coverage matrix

| §2.4 candidate | Existing skill | Status |
|---|---|---|
| `mol-metal-score-rep-variance-mcts` (NEW R21) | (none) | ❌ **MISSING — STUB created** |
| `mol-metal-mcts-replay-learner` (NEW R21) | `mol-metal-mcts-replay-learner` (Sep 26 full) | ✅ COVERED |
| `mol-metal-closure-dblock-bench` (NEW R21) | `mol-metal-catherine-dblock-bench` (Sep 28 stub) | ✅ COVERED |
| `mol-metal-triangle-closure` (NEW R21) | `mol-metal-triangle-closure` (Sep 26 full) | ✅ COVERED |

**R21 coverage = 3/4 (75%)** — 1 missing: `mol-metal-score-rep-variance-mcts` (NEW STUB).

### §1.4 Task-list NEW stubs (4 written)

Per Wf-321B task spec, 4 NEW STUB skills written (R20 + R21 summary packs + R20-P4 3way multi-metal + R21-P1 score-rep):

| NEW skill | Status | Purpose |
|---|---|---|
| `mol-metal-3way-multimetal` | ✅ STUB | R20-P4 multi-metal 3-way ship surface (TODO-95) |
| `mol-metal-r20-skill-pack` | ✅ STUB | R20 summary skill pack (4 NEW + 16 R17-R19) |
| `mol-metal-r21-skill-pack` | ✅ STUB | R21 summary skill pack (4 NEW + 16 R17-R20) |
| `mol-metal-score-rep-variance-mcts` | ✅ STUB | R21-P1 3-way joint ablation (L3 closure) |

**n_new_skill_stubs = 4** (all 4 written with full STUB contract + acceptance gate + MEASURED-vs-PROJECTED ledger).

---

## §2 Stub tests (16/16 PASS)

Each NEW skill stub includes 4 sanity-check tests:

| Skill | Tests | Status |
|---|---|---|
| `mol-metal-3way-multimetal` | 4 | ✅ 4/4 PASS |
| `mol-metal-r20-skill-pack` | 4 | ✅ 4/4 PASS |
| `mol-metal-r21-skill-pack` | 4 | ✅ 4/4 PASS |
| `mol-metal-score-rep-variance-mcts` | 4 | ✅ 4/4 PASS |
| **TOTAL** | **16** | **✅ 16/16 PASS** |

Run command: `pytest .claude/skills/{mol-metal-3way-multimetal,mol-metal-r20-skill-pack,mol-metal-r21-skill-pack,mol-metal-score-rep-variance-mcts}/tests/ -v` — **0.08s wall, 16/16 PASS**.

---

## §3 Pre-flight v2 check (21/21 PASS)

Per AGENTS.md §4 critical workflows, pre-flight v2 (TODO-207) is mandatory before any retrain/panel/ablation launch. This audit is **CPU-only documentation** (no retrain/panel launch), so pre-flight was run as a sanity check (not as a launch gate).

**Pre-flight frontier pipeline** (Wf-306C subset, CPU-only):

```
PYTHONPATH=. /usr/bin/python molmetal/scripts/pre_flight_check.py --pipeline frontier --json /tmp/wf321b_preflight.json

Result: 21/21 PASS, 0 FAIL, 0 SKIP
```

**preflight_status = PASS (21/21)**.

---

## §4 Honest framing (per game accounting §3.5)

Per AGENTS.md §5.2 honest framing mandate + `party-twenty-spirit` §3.5, every numeric value below traces to a JSON / report path on disk OR is marked PROJECTED:

| Value | Tag | Source |
|---|---|---|
| 24 existing skills (n_skills_audited) | MEASURED | `.claude/skills/` directory listing |
| 4 NEW skill STUBs written | MEASURED | this audit |
| 16/16 stub tests PASS | MEASURED | pytest output |
| 21/21 pre-flight PASS | MEASURED | `/tmp/wf321b_preflight.json` |
| R20 4/4 candidate coverage | MEASURED | LAMBDA_ROADMAP §1.5 cross-ref |
| R21 3/4 candidate coverage | MEASURED | LAMBDA_ROADMAP §2.4 cross-ref |
| 4 NEW skill STUB full ship | PROJECTED (Wf-42/Wf-45) | R20-P4 + R21-P1 ship surface |
| R20 + R21 4+4 = 8 NEW skill ship | PROJECTED | LAMBDA_ROADMAP §1.5 + §2.4 |

**Honest-negative anchor**: All 4 NEW skill STUBs carry **STUB note** + **NOT YET CREATED** markers on `bench_3way_multimetal.py` + `bench_score_rep_variance_mcts.py`. Full contract ships after Wf-42 (R20-P4) + Wf-45 (R21-P1) close.

---

## §5 Cross-ref

| File | Purpose |
|---|---|
| `TODO/03_plans/LAMBDA_ROADMAP.md` §1.5 | R20 4 NEW skill 候选 (source of truth) |
| `TODO/03_plans/LAMBDA_ROADMAP.md` §2.4 | R21 4 NEW skill 候选 (source of truth) |
| `AGENTS.md` §3 | skill catalog (24 skills pre-this-audit) |
| `TODO/INDEX.md` §22 | recent Wf timeline (Wf-321B ship wave) |
| `TODO/completed/207_unified_pre_flight_checklist_v2.md` §3.1 + §4.2 + §5 | pre-flight v2 25/25 PASS gate |

---

## §6 Summary metrics (per OUTPUT spec)

| Metric | Value |
|---|---|
| **n_skills_audited** | 24 (existing) + 4 (NEW STUB) = **28** |
| **n_new_skill_stubs** | **4** (mol-metal-3way-multimetal, mol-metal-r20-skill-pack, mol-metal-r21-skill-pack, mol-metal-score-rep-variance-mcts) |
| **n_existing_verified** | **24** (full audit, all frontmatter + SKILL.md verified) |
| **preflight_status** | **PASS (21/21)** (frontier pipeline, CPU-only sanity check) |
| **stub_tests** | **16/16 PASS** (4 tests per new skill) |

---

## §7 Honest gaps (NOT closed by this audit)

- **Wf-321B did NOT create** `mol-metal-3way-multimetal` **full content** — only STUB. Full contract ships after Wf-42 (R20-P4 100-pocket × 10-seed).
- **Wf-321B did NOT create** `mol-metal-score-rep-variance-mcts` **full content** — only STUB. Full contract ships after Wf-45 (R21-P1 joint ablation).
- **Wf-321B did NOT create** `bench_3way_multimetal.py` or `bench_score_rep_variance_mcts.py` — both marked **NOT YET CREATED** in respective SKILL.md §3 reference impl pointer.
- **Wf-321B did NOT close** any R20-P4 or R21-P1 ship wave — only staged the STUB skill surface for downstream Wf-42 / Wf-45 owners.

These gaps are intentional — Wf-321B is **skill ecosystem audit + STUB creation only**. Full ship is downstream Wf-42 / Wf-45 / Wf-50 work.

---

**END OF Wf-321B R20+R21 Skill Ecosystem Audit** — Owner Hugo Lee; cross-ref `party-twenty-spirit` §3.5; last update 2026-09-29.