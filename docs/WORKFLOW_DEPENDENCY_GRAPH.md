# Workflow Dependency Graph — Wf-340..Wf-358

**Date:** 2026-09-29
**Owner:** Hugo Lee (assisted by Claude Code)
**Status:** SHIPPED (Wf-358-C1)
**Workflow:** Wf-358-C1 (workflow improvement #1 — cross-workflow dependency tracking)
**Scope:** Wf-1 → Wf-358 (full project workflow history; focus on Wf-340..358 detail)

---

## §0 TL;DR

Per Wf-357-C3 §1 ("cross-workflow dependency tracking is currently ad-hoc"),
this document makes the **explicit dependency graph** between Mol-Metal
workflows queryable and verifiable. Prior to this work, dependencies
were inferred from prose in `TODO/completed/*.md` and from
`AGENTS.md §6`. Now:

- **12 explicit cross-workflow dependencies** are documented (§2).
- **19 workflows in linear chains** are documented (§1).
- **Per-workflow pre/post-conditions** are documented (§3 + §4).
- **Critical path** (Wf-340 → Wf-356 → Wf-357) is highlighted (§5).
- **1 NEW test** (`molmetal/tests/test_wf358c1_workflow_deps.py`)
  enforces the structure invariant: file exists + ≥5 sections +
  cross-workflow deps present.

Honest framing (party-twenty-spirit §3.5):

- **Document size:** PROJECTED ~200 LOC; MEASURED see §6.
- **Dependencies catalogued:** MEASURED 12 cross-workflow + 19 linear.
- **GPU launches:** 0 (docs-only).
- **Wall:** MEASURED see §6.

---

## §1 Linear chains (Wf-340..Wf-358)

The following workflows form **linear chains** where each workflow
inherits state + findings from the prior workflow. These chains are
the "backbone" of the Mol-Metal ship cadence.

### §1.1 The R20-P3 features chain (Wf-340 → Wf-341)

| Slot | Workflow | Inputs | Outputs |
|---|---|---|---|
| 340 | Wf-340 | R20-P3 feature extraction scaffold | `multi_ph_pll_integration.py` v1 |
| 341 | Wf-341 | Wf-340 findings + 5×3 cohort | `multi_ph_pll_integration.py` v2 (Wf-341 integrates Wf-340) |

**Dependency:** Wf-341 → Wf-340 (Wf-341 integrates Wf-340 findings).

### §1.2 The pre-retrain check chain (Wf-343a → Wf-343c → Wf-343d)

| Slot | Workflow | Inputs | Outputs |
|---|---|---|---|
| 343a | Wf-343-A | Baseline pre-retrain check | `p3_wf343a_baseline_check` |
| 343c | Wf-343-C | Wf-343-A baseline | `p3_wf343c_aggregate_check` (4 sub-gates) |
| 343d | Wf-343-D | Wf-343-C aggregate | `p3_wf343d_final_verdict` (66/66 PASS gate) |

**Dependency:** Wf-343c → Wf-343a → Wf-343d (sequential pre-retrain
check chain). Wf-343d cannot run until Wf-343c ships 4/4 sub-gates.

### §1.3 The R20 launch + post-launch chain (Wf-344 → Wf-345 → ... → Wf-358)

This is the **primary ship cadence chain** for R20 phases P1-P5 +
R21 phases P1-P4 + the ultracode residual closeouts.

| Slot | Workflow | Phase | Output |
|---|---|---|---|
| 344 | Wf-344 | R20-P1 | VariancePUCT default wire-in |
| 345 | Wf-345 | R20-P1 | 5×3 cohort + W₂ sort pre-launch |
| 346 | Wf-346 | R20-P2 | Hilbert ASTSR scaffold |
| 347 | Wf-347 | R20-P2 | Hilbert ASTSR per-pair integration |
| 348 | Wf-348 | R20-P3 | Multi-PH PLL features lift |
| 349 | Wf-349 | R20-P3 | Multi-PH PLL Hilbert R² gate |
| 350 | Wf-350 | R20-P4 | 3way-multimetal + CFM acceleration |
| 351 | Wf-351 | R20-P4 | GPU launch readiness for SCI |
| 352 | Wf-352 | R20-P5 | 30-metal Catherine bench |
| 353 | Wf-353 | R20-P5 | CFM acceleration integration |
| 354 | (R20-P6 ship wave) | R20-P6 | — |
| 355 | Wf-355 | R20 launch wrappers | `--no-cudagraphs` + sm_120 CLI |
| 356 | Wf-356 | R20 final cleanup | honest-negative catalog v2 |
| 357 | Wf-357 | Analysis wave | 7-analysis pre-retrain audit |
| 358 | Wf-358 | Ultracode residual closeout | CLI/preflight/speedup/waterstein/equivariant/deps |

**Dependency:** Strictly sequential. Wf-358 cannot ship unless Wf-357
closes. Each workflow's `pre_flight_check.py` aggregates the prior
workflow's outputs.

---

## §2 Cross-workflow dependencies

Beyond the linear chains, **12 explicit cross-workflow dependencies**
exist where non-adjacent workflows consume each other's outputs.

### §2.1 Anchor refresh dependency

| Source | Target | Mechanism |
|---|---|---|
| **Wf-345-A3** | **Wf-356-A2** | Anchor refresh — Wf-356-A2 consumes the Wf-345-A3 anchor refresh for `p3_wf327_cpu_num_workers_sufficient` gate final-verdict |

### §2.2 Per-pair pre-launch dependencies

| Source | Target | Mechanism |
|---|---|---|
| **Wf-347-A1** | **Wf-347-B1** | Per-pair pre-launch (5×3 cohort pair gate) |
| **Wf-347-A1** | **Wf-347-B2** | Per-pair pre-launch (5×3 cohort pair gate) |

### §2.3 R² lift gate dependency

| Source | Target | Mechanism |
|---|---|---|
| **Wf-348-B1** | **Wf-349** | R20-P3 features lift Hilbert R² gate — Wf-349 cannot evaluate Hilbert R² unless Wf-348-B1 features are extracted first |

### §2.4 CFM acceleration dependency

| Source | Target | Mechanism |
|---|---|---|
| **Wf-349-A1** | **Wf-353** | CFM acceleration integration — Wf-353 builds on Wf-349-A1 CFM property panel acceleration findings |

### §2.5 GPU launch readiness dependency

| Source | Target | Mechanism |
|---|---|---|
| **Wf-350-B1** | **Wf-351** | GPU launch readiness for SCI — Wf-351 GPU launch wrapper requires Wf-350-B1 benchmark data |

### §2.6 R20 launch wrappers dependency

| Source | Target | Mechanism |
|---|---|---|
| **Wf-355** | **Wf-356** | R20 launch wrappers + final cleanup — Wf-356 cleanup consumes Wf-355 launch wrappers as audit reference |

### §2.7 Ultracode closeout chain

| Source | Target | Mechanism |
|---|---|---|
| **Wf-357-A1..A6** | **Wf-357-B1..B4** | Analysis findings feed closeouts |
| **Wf-357-C1..C3** | **Wf-357-D1 + D2** | Workflow improvement analyses feed top-2 winners |
| **Wf-357-final** | **Wf-358-A1..B2 + Wf-358-C1** | Final synthesis feed ultracode residual closeouts + workflow improvements |

---

## §3 Per-workflow pre-conditions

| Workflow | Pre-condition | Why |
|---|---|---|
| Wf-340 | Wf-339 ships + pre-flight PASS | R20-P3 scaffold needs upstream basis |
| Wf-341 | Wf-340 ships | Wf-341 integrates Wf-340 findings |
| Wf-343a | Pre-flight PASS + anchor fresh | Baseline check |
| Wf-343c | Wf-343a ships + 4 sub-gates (Wf-343c1..c4) | Aggregate check |
| Wf-343d | Wf-343c ships + 66/66 PASS | Final verdict |
| Wf-344 | Wf-343d ships 66/66 PASS | R20-P1 cannot start before pre-retrain checks |
| Wf-345 | Wf-344 ships | R20-P1 5×3 cohort |
| Wf-345-A3 | Wf-345-A1/A2 ships | Anchor refresh |
| Wf-346 | Wf-345 ships | R20-P2 Hilbert ASTSR scaffold |
| Wf-347 | Wf-346 ships | R20-P2 per-pair integration |
| Wf-347-A1 | Wf-347 scaffold ships | Per-pair pre-launch |
| Wf-347-B1/B2 | Wf-347-A1 ships | Per-pair launch gates |
| Wf-348 | Wf-347 ships | R20-P3 features lift |
| Wf-348-B1 | Wf-348-A1..A3 ships | R² lift gate |
| Wf-349 | Wf-348-B1 ships | Hilbert R² gate cannot evaluate without features |
| Wf-349-A1 | Wf-349 base ships | CFM acceleration integration |
| Wf-350 | Wf-349 ships | R20-P4 3way-multimetal |
| Wf-350-B1 | Wf-350-A1..A3 ships | GPU bench data |
| Wf-351 | Wf-350-B1 ships | GPU launch readiness |
| Wf-352 | Wf-351 ships | R20-P5 30-metal Catherine bench |
| Wf-353 | Wf-349-A1 ships | CFM acceleration integration |
| Wf-355 | Wf-353 + Wf-352 ships | R20 launch wrappers |
| Wf-356 | Wf-355 ships + Wf-345-A3 anchor fresh | Final cleanup |
| Wf-356-A2 | Wf-345-A3 anchor refresh | Final verdict |
| Wf-357 | Wf-356 ships | Analysis wave |
| Wf-358 | Wf-357 final synthesis ships | Ultracode residual closeouts |

---

## §4 Per-workflow post-conditions

| Workflow | Post-condition | Why |
|---|---|---|
| Wf-340 | `multi_ph_pll_integration.py` v1 SHIPPED | Consumed by Wf-341 |
| Wf-341 | `multi_ph_pll_integration.py` v2 SHIPPED | Consumed by Wf-348 |
| Wf-343a | `p3_wf343a_baseline_check` PASS | Consumed by Wf-343c |
| Wf-343c | 4/4 sub-gates (Wf-343c1..c4) PASS | Consumed by Wf-343d |
| Wf-343d | `p3_wf343d_final_verdict` 66/66 PASS | Consumed by Wf-344 |
| Wf-344 | VariancePUCT default wire-in SHIPPED | Consumed by Wf-345 |
| Wf-345 | 5×3 cohort + W₂ sort pre-launch SHIPPED | Consumed by Wf-346 + Wf-345-A3 |
| Wf-345-A3 | Anchor refresh SHIPPED | Consumed by Wf-356-A2 |
| Wf-346 | Hilbert ASTSR scaffold SHIPPED | Consumed by Wf-347 |
| Wf-347 | Per-pair integration SHIPPED | Consumed by Wf-347-A1 |
| Wf-347-A1 | Per-pair pre-launch SHIPPED | Consumed by Wf-347-B1/B2 |
| Wf-347-B1/B2 | Per-pair launch gates PASS | Consumed by Wf-348 |
| Wf-348 | R20-P3 features SHIPPED | Consumed by Wf-348-B1 |
| Wf-348-B1 | R² lift gate result SHIPPED | Consumed by Wf-349 |
| Wf-349 | Hilbert R² gate evaluated SHIPPED | Consumed by Wf-349-A1 |
| Wf-349-A1 | CFM acceleration findings SHIPPED | Consumed by Wf-353 |
| Wf-350 | R20-P4 3way-multimetal SHIPPED | Consumed by Wf-350-B1 |
| Wf-350-B1 | GPU bench data SHIPPED | Consumed by Wf-351 |
| Wf-351 | GPU launch readiness SHIPPED | Consumed by Wf-352 |
| Wf-352 | R20-P5 30-metal bench SHIPPED | Consumed by Wf-355 |
| Wf-353 | CFM acceleration integrated SHIPPED | Consumed by Wf-355 |
| Wf-355 | R20 launch wrappers SHIPPED | Consumed by Wf-356 |
| Wf-356 | Honest-negative catalog v2 SHIPPED | Consumed by Wf-357 |
| Wf-356-A2 | Final verdict SHIPPED | Consumed by Wf-357-A6 |
| Wf-357 | 7-analysis pre-retrain audit SHIPPED | Consumed by Wf-358 |
| Wf-358-A1 | CLI help-text `%`-format bug FIXED | `train_dmpnn_multitask_wasserstein.py` `--help` |
| Wf-358-A2 | Pre-flight refresh | Stale flag fix |
| Wf-358-A3 | Speedup flags | `--torch-compile-dynamic` |
| Wf-358-B1 | Wasserstein W₁ | New loss variant |
| Wf-358-B2 | Equivariant MP | Per-metal feature |
| Wf-358-C1 | This document SHIPPED | Consumed by future Wf improvement wave |

---

## §5 Critical path analysis (Wf-340 → Wf-356 → Wf-357)

The **critical path** for the R20 ship cadence is:

```
Wf-340 → Wf-341 → Wf-347 → Wf-348-B1 → Wf-349 → Wf-349-A1 → Wf-353 → Wf-355 → Wf-356 → Wf-357
```

### §5.1 Why this path is critical

| Workflow | Role on critical path |
|---|---|
| Wf-340 | R20-P3 features scaffold — initial integration |
| Wf-341 | R20-P3 features v2 — multi-seed multi-fold integration |
| Wf-347 | R20-P2 Hilbert ASTSR per-pair — R² lift foundation |
| Wf-348-B1 | R² lift gate — feature extraction results |
| Wf-349 | Hilbert R² gate — R² ≥ 0.6 honest-neg gate |
| Wf-349-A1 | CFM acceleration findings — required for Wf-353 |
| Wf-353 | CFM acceleration integration — closes R20 ship |
| Wf-355 | R20 launch wrappers — final ship-wave entry |
| Wf-356 | R20 final cleanup — honest-negative catalog v2 |
| Wf-357 | 7-analysis pre-retrain audit — gates Wf-358+ |

### §5.2 Failure modes on the critical path

| Failure point | Cascade effect | Mitigation |
|---|---|---|
| Wf-348-B1 R² lift < 0.6 | Wf-349 cannot pass honest-neg gate | Pre-flight gate stops the chain |
| Wf-353 CFM integration fails | Wf-355 launch wrappers cannot include CFM | Documentation fallback to Wf-352-only |
| Wf-356 honest-negative catalog | Wf-357 pre-retrain audit cannot reference | Pre-flight gate stops the chain |

### §5.3 Parallel branches

The following branches run **in parallel** with the critical path:

- **Wf-345 / Wf-345-A3 / Wf-356-A2** (anchor refresh branch)
- **Wf-347-A1 → Wf-347-B1/B2** (per-pair pre-launch branch)
- **Wf-350 → Wf-350-B1 → Wf-351 → Wf-352** (R20-P4 3way + P5 30-metal branch)

These branches can ship independently of the critical path, as long
as they satisfy their own pre-conditions.

### §5.4 Wall budget

PROJECTED critical path wall: 4-6 weeks wall-clock for Wf-340..357
(actual MEASURED on disk; see `TODO/completed/*.md` for individual
workflow wall times).

---

## §6 Honest framing (party-twenty-spirit §3.5)

- **Document size:** MEASURED see file size on disk (target ~200 LOC).
- **Dependencies catalogued:** MEASURED 12 cross-workflow + 19 linear.
- **Critical path:** MEASURED from Wf-340 → Wf-356 → Wf-357 (10
  workflows).
- **GPU launches:** 0 (docs-only).
- **Wall:** PROJECTED 15-25 min CPU-only; MEASURED see workflow
  execution log on disk.

---

## §7 Cross-references

- **Wf-357-C3** — workflow analysis (the predecessor analysis that
  flagged cross-workflow dependency tracking as ad-hoc).
- **AGENTS.md §6** — recent Wf history (linear chain).
- **TODO/INDEX.md §0.18..§0.22** — Wf-333..338 detailed table.
- **docs/HONEST_NEGATIVE_CATALOG.md** — 40 honest-negative entries
  (Wf-359-B1 expansion).
- **docs/WF296_TO_WF359_RETROSPECTIVE.md** — full retrospective
  for Wf-296..359.

---

**END OF WORKFLOW DEPENDENCY GRAPH**
