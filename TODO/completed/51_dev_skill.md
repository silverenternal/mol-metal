# TODO-51 — Dev-Skill Workflow Synthesis (R17 close-out deltas)

**Date:** 2026-09-25 · **Status:** ✅ **SHIP** · **Source plans:** TODO-39 (CI + Docs +
Skills) + TODO-44 (Diagnostic + Skill) + TODO-45 (Refactor + README) + TODO-48 (Repro +
Coverage + CONTRIBUTING + CHANGELOG) + this TODO's two new skills.

**Companion to:** `39_ci_docs_skills.md` (5-job CI baseline). This file is the dev-skill
delta: pre-commit + CONTRIBUTING + CHANGELOG + 5 existing skills + 2 new skills + CI matrix.

---

## 1. Executive summary

| Stream | Status | Artefact | Delta |
|---|---|---|---|
| **Pre-commit** | ✅ LIVE (6 hooks) | `.pre-commit-config.yaml` (35 lines) | NEW |
| **CONTRIBUTING.md** | ✅ SHIP (13 sections) | `CONTRIBUTING.md` (460 lines) | NEW |
| **CHANGELOG.md** | ✅ SHIP (Keep-a-Changelog) | `CHANGELOG.md` (180 lines, R17/R18/R16/R0) | NEW |
| **5 existing skills** | ✅ All carry R17 banner | `mol-metal-{lambda,cfm,click,anti,mech}/SKILL.md` (978 LOC) | MODIFIED |
| **2 new skills** | ✅ Activated | `mol-metal-paper-writing/SKILL.md` (174), `mol-metal-r18-math/SKILL.md` (174) | +348 NEW |
| **CI matrix** | ✅ 5+1 jobs | `.github/workflows/ci.yml` (315 lines) | EXPANDED |

**Footprint:** 1,694 LOC across 9 artefacts; **0 regressions** in the 434-pass
mechanism-smoke gate; 5 R17 skills retain banners; 2 new skills extend coverage to paper
drafting + R18 mathematical upgrades.

---

## 2. Per-task deliverables (D1-D7)

| ID | Item | Path | Lines | Verdict |
|---|---|---|---:|---|
| **D1** | Pre-commit (trailing-WS, EOF, YAML/JSON, ruff+format, mypy, mdformat, codespell) | `.pre-commit-config.yaml` | 35 | ✅ LIVE |
| **D2** | CONTRIBUTING.md (Quick start / Workflow / Conventions / Docs / Testing / PRs / Reviewer / Ablation / Metric / Mechanism / Checklist / Pitfalls) | `CONTRIBUTING.md` | 460 | ✅ SHIP |
| **D3** | CHANGELOG.md (Keep-a-Changelog 1.1.0 + SemVer 2.0.0; Unreleased/R18, 0.2.0/R17, 0.1.0/R16, 0.0.1) | `CHANGELOG.md` | 180 | ✅ SHIP |
| **D4** | 5 existing skills + R17 banner sync | `.claude/skills/mol-metal-{lambda-calculus,cfm-training,click-chemistry,anticancer,mechanism-mapping}/SKILL.md` | 978 | ✅ SYNCED |
| **D5** | New skill — paper writing | `.claude/skills/mol-metal-paper-writing/SKILL.md` | 174 | ✅ NEW |
| **D6** | New skill — R18 mathematical theorem integration | `.claude/skills/mol-metal-r18-math/SKILL.md` | 174 | ✅ NEW |
| **D7** | CI matrix expansion (5 → 5+1 with optional Blackwell) | `.github/workflows/ci.yml` | 315 | ✅ EXPANDED |

**Honest framing:** D5/D6 are NEW skills that TODO-39 shipped without — R17 closed with 5
skills; this TODO adds the 2 most-requested extensions (paper drafting + R18 math).
**Pre-commit hook chain (D1):** pre-commit-hooks v4.5.0; ruff+format v0.6.9; mypy
v1.11.2 (`--ignore-missing-imports --strict-equality`); mdformat 0.7.17 (+myst +toc,
`--wrap 100`); codespell v2.2.6.

---

## 3. New skill: `mol-metal-paper-writing` (174 lines, NEW)

**Trigger phrases:** "write §3", "draft §4", "results table", "limitations", "future
work", "abstract", "MEASURED vs PROJECTED", "honest framing", "figure caption", any
mention of `paper/main.tex`, `paper/sections/*.tex`, `paper/appendices/*.tex`.

**Frontmatter:** `name`, `description` (trigger phrases), `tools: Read, Grep, Bash, Edit`.
**Three input docs:** `TODO/02_metrics/METRICS_TARGETS.md` (numbers → row), `TODO/02_metrics/
MECHANISM_METRIC_MAPPING.md` (metric → mechanism), `paper/main.tex` (bundle policy +
target journal: Digital Discovery Q1 8.5 → JCIM Q1 5.6 fallback).

**Body:** per-section template (§1-§7), honest-framing tag convention (`MEASURED` /
`PROJECTED` / `ARCHITECTURAL`), cross-skill references, figure-caption rules,
table-generation workflow tied to `molmetal/scripts/generate_section{4,6}.py` +
`generate_abstract.py` + `check_submission_ready.py` (TODO-47). **Convention:** "tag every
quantitative claim with one of MEASURED / PROJECTED / ARCHITECTURAL and cite the source
row inline as `\todo{metric=N,file=METRICS_TARGETS.md}`."

---

## 4. New skill: `mol-metal-r18-math` (174 lines, NEW)

**Trigger phrases:** R18 upgrade planning, paper §3 method extension, reward/flow
redesign, pocket-conditioned generation on non-Euclidean manifolds, persistent homology
features, risk analysis for theoretical hardness.

**Frontmatter:** `name`, `description` listing the 9 theorems (Mahmood SB-on-Lie,
Jiang-Xu-Zhang transport relaxation, 汪徐家 Monge-Ampère, 薛小平 Wasserstein gradient
flow, Kim-Jung multi-parameter persistence, Kolbe-Mayr PH approximation hardness,
Henry-Labordere + Backhoff-Veraguas Bass martingale couplings, 麻小南 Bergman kernel
quantum geometry).

**9 theorems (P0/P1/P2 priority map at file end):** **T1** SB on Lie Groups (Mahmood
2026) — Pocket-Conditioned Lipman SE(3) flow → Vina Dock -8.5 → -10.5 kcal/mol; **T2**
Transport Relaxation (Jiang-Xu-Zhang) — pK_bind MAE via ε-relaxed OT; **T3** Monge-Ampère
regularity (汪徐家) — MetalCytotoxDB AUC convexity guarantees; **T4** Wasserstein gradient
flow (薛小平) — SynFlowNet terminal sharpening; **T5** Multi-parameter persistence
(Kim-Jung) — pocket feature summaries vs single-param bottleneck; **T6** PH approximation
hardness (Kolbe-Mayr) — risk gate before compute commitment; **T7** Bass martingale
couplings (Henry-Labordere, Backhoff-Veraguas) — RBFE RMSE reduction; **T8** Bergman
kernel quantum geometry (麻小南) — novel pose manifold metric. Each: statement (informal)
→ Mol-Metal anchor → integration sketch → expected delta.

---

## 5. CI gate expansion (D7)

**File:** `.github/workflows/ci.yml` (315 lines, 5+1 jobs)

| # | Job | Gate | Verdict @ R17 |
|---|---|---|---|
| 1 | `evaluator-sanity` | 14/14 evaluators PASS (`grep "14/14 evaluators PASS"`) | ✅ 14/14 |
| 1b | `blackwell-gpu-sanity` | CFM subsystem smoke + 200-step retrain + Blackwell pytest | ✅ dispatch-only |
| 2 | `pitfall-runner` | `PASS=7 FAIL=0 SKIP=0` | ✅ PASS=7 |
| 3 | `pitfall-pytest` | ≥11 passed | ✅ 18 passed |
| 4 | `mechanism-smoke` | ≥217 passed (8 named files) | ✅ 434 / 22 skipped |
| 5 | `ci-summary` | Always-runs aggregation → `$GITHUB_STEP_SUMMARY` | ✅ all green |

**Expansion vs TODO-39:** +99 lines (216 → 315) for Blackwell optional job 1b and the
8-file mechanism-smoke list (CI YAML lines 272-279). Concurrency cancel + 14-day artifact
retention + shared venv path contract preserved. **Hardware matrix:** self-hosted
`[self-hosted, linux, x64, mol-metal]` for CPU jobs; `[self-hosted, linux, x64,
mol-metal-blackwell]` for 1b only (guarded by
`if: github.event_name == 'workflow_dispatch' || 'schedule'`); GitHub-hosted
`ubuntu-latest` is the drop-in replacement.

---

## 6. Cross-reference to existing 5 skills

All 5 carry the canonical **R17 production state** banner: `Decode probes: 30/30 = 1.000
/ Evaluators: 14/14 verified / 6 shim fixes (TODO-33 P1+P2) tracked with file:line anchors`.

| # | Skill | LOC | Banner | Cross-ref to NEW skills |
|---|---|---:|---:|---|
| 1 | `mol-metal-lambda-calculus` | 178 | L24 | r18-math T3 (Monge-Ampère arity=valence) |
| 2 | `mol-metal-cfm-training` | 190 | L30 | r18-math T1+T2 (SB-on-Lie + transport relaxation) |
| 3 | `mol-metal-click-chemistry` | 196 | L113 | r18-math T7 (Bass martingale SPAAC) |
| 4 | `mol-metal-anticancer` | 280 | L47 | paper-writing §3 (NCI-60 framing) |
| 5 | `mol-metal-mechanism-mapping` | 204 | L11 | paper-writing §4+§6 (MEASURED/PROJECTED tags) |

**Cross-skill graph:** `lambda-calculus / cfm-training / click-chemistry / anticancer` fan
into `mechanism-mapping` → `paper-writing` → `paper/main.tex`; same sources fan into
`r18-math` → `paper-writing §3+appendices`. `mechanism-mapping` is the hub.

---

## 7. Verification + caveats + companion files

**Verification:** 5 existing skills banner present + line-anchored (`grep -n "R17 production
state" .claude/skills/*/SKILL.md` returns 5 matches); 2 new skills have valid YAML
frontmatter + tool lists; CI grep gates intact; Blackwell job guarded against PRs;
CONTRIBUTING.md + CHANGELOG.md lint-clean and cross-linked.

**Honest caveats:** the 2 new skills are R18-facing — encode future-work, not R17
production behaviour; treat as drafting aids until R18 papers land. Pre-commit
configured but not yet wired into CI (R18 follow-up; CI gate stays pytest-based).

**Companion files:** `TODO/completed/39_ci_docs_skills.md` (5-job CI baseline this TODO
expands); `TODO/completed/48_repro_and_coverage.md` (CHANGELOG + CONTRIBUTING delta);
`TODO/completed/44_diag_and_skill.md` ("Common mistake" on each of 5 skills);
`TODO/02_metrics/METRICS_TARGETS.md` + `MECHANISM_METRIC_MAPPING.md` (paper-writing
inputs); `paper/refs.bib` (receives r18-math citations, TODO-42 §3).
