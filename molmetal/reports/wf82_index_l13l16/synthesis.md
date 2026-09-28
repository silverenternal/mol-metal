# Wf-82 — INDEX.md §17.6 L13-L16 candidate row + party-twenty-spirit cross-ref (synthesis)

**Date**: 2026-09-26
**Workflow**: Wf-82 (CPU-only docs — INDEX.md patch + 1-page synthesis + per-section presence audit)
**Status**: ✅ SHIP
**Constraint**: paper-gate (TODO-84) honored · CPU-only docs · no GPU retrain executed · `party-twenty-spirit` §3.5 honest framing applied throughout

---

## §1 What Wf-82 ships

| Deliverable | Path | Status | Size |
|---|---|:---:|---|
| **§17.6 patch** | `TODO/INDEX.md` (new section after §17.5) | ✅ NEW | ~80 lines |
| **Synthesis (this file)** | `molmetal/reports/wf82_index_l13l16/synthesis.md` | ✅ NEW | 1 page |
| **Per-section presence audit** | `molmetal/reports/wf82_index_l13l16/audit.json` | ✅ NEW | JSON |

---

## §2 §17.6 content (4 candidates + 6 plan items)

### §2.1 L13-L16 candidate Lambda gaps (4 rows)

| Candidate | Title | Priority | R first close | Signal count |
|---|---|:---:|---|---:|
| **L13** | Hierarchical multi-metal coordination geometry | **HIGH** | R22-P4 (Wf-75) | **188** (57%) |
| **L14** | Graph Neural ODE for metallodrug kinetics | MEDIUM | R23-P2 | **20** (6%) |
| **L15** | Multi-objective Pareto front (toxicity / efficacy / ADMET) | MEDIUM | R22-P3 (Wf-77) | **52** (16%) |
| **L16** | Wet-lab-in-the-loop Bayesian optimization | **HIGH** | R23-P3 | **70** (21%) |
| **Total** | 4 candidates | 2 HIGH + 2 MEDIUM | R22-R23 | **330** |

### §2.2 R22-P1..P6 plan items (6 rows)

| R22-P | Lambda gap | Wall | Hardware | Status |
|---|---|---|---|:---:|
| **R22-P1** | L13 | 3-4 wk | CPU + GPU-0 | 🆕 ACTIVE |
| **R22-P2** | L14 | 4 wk | GPU-1 | 🆕 ACTIVE |
| **R22-P3** | L15 | 3 wk | CPU + GPU-0 | 🆕 ACTIVE |
| **R22-P4** | L16 | 6 mo wet-lab + 2 wk CPU BO | wet-lab + CPU | 🆕 ACTIVE |
| **R22-P5** | [reserved] ship-evidence gap | TBD by Wf-73 | CPU | 🆕 RESERVED |
| **R22-P6** | [reserved] cross-skill ecosystem | TBD by Wf-73 | CPU | 🆕 RESERVED |

---

## §3 Cross-references (per task spec)

| Reference | Path | Used in §17.6 |
|---|---|---|
| Wf-72 Phase 1 L13-L16 gap analysis | `TODO/pending/r22_l13_l16_gap_analysis.md` (311 lines) | §17.6.1 + §17.6.3 |
| Wf-72 Phase 2 R22-P1..P6 plan | `TODO/pending/r22_p1_p2_p3_p4_p5_p6_plan.md` (~440 lines) | §17.6.2 |
| Wf-72 Phase 3 5-cat audit JSON | `molmetal/reports/wf72_5cat_extend/audit_22_to_25.json` | §17.6.5 + header |
| LAMBDA_ROADMAP §4.2 12-row trajectory | `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 | §17.6.3 (proposed 4-row extension) |
| LAMBDA_ROADMAP §3 R22-R25 long-term | `TODO/03_plans/LAMBDA_ROADMAP.md` §3 | §17.6.5 |
| party-twenty-spirit §3.5 honest framing | `.claude/skills/party-twenty-spirit/SKILL.md` §3.5 | §17.6.4 (REQUIRED) |
| Paper-gate | `TODO/pending/84_no_paper_task_without_authorization.md` | §17.6.4 + §17.6.5 |

---

## §4 Honest framing (per `party-twenty-spirit` §3.5)

| Honest finding | Evidence |
|---|---|
| **L13-L16 are CANDIDATE shapes, not committed gaps** | Per `r22_l13_l16_gap_analysis.md` §0 — R22+ plan owners re-evaluate before adopting |
| **330 scan signals = regex density ≠ quality** | 188 L13 hits ≠ "L13 is the right shape"; pattern-match counts are NOT quality measures |
| **0 GPU retrain executed today** | CPU-only docs only — R22-P1 GPU sweep + R22-P2 GNN-ODE + R22-P3 GPU-0 + R22-P4 wet-lab reserved for R22-R23 |
| **0 LAMBDA_ROADMAP §4.2 patch executed** | §17.6.3 proposes 4 new rows; Wf-72 Phase 4 (next cycle) applies the patch |
| **No paper work touched** | Per TODO-84 paper-gate — no `paper/sections/*.tex`, no `paper/main.tex`, no reviewer rebuttal |
| **R20 honest-negative carry-forward** | 0/8 R20-P fully MEASURED + 3/50 NOT MET (reachability 0.5962 < 0.95; 0.5744 < 0.95; Wf-44 P2 empty dir) + 30/50 DEFERRED. L13-L16 do NOT resolve R20 honest-negative |
| **Wet-lab cooperation 6-month deferred** | ORNL/PKU/协和 outreach 0 progress today; L16 cannot close until R23-P4 wet-lab cycle starts |

---

## §5 VERIFY checklist (per task spec)

- [x] **INDEX.md §17.6 references 4 L13-L16 candidates** (L13/L14/L15/L16 with 188/20/52/70 = 330 signals)
- [x] **INDEX.md §17.6 references 6 R22-P plan items** (R22-P1/P2/P3/P4 active + R22-P5/P6 reserved)
- [x] **INDEX.md §17.6 references audit_22_to_25.json** (`molmetal/reports/wf72_5cat_extend/audit_22_to_25.json` cited in §17.6.5)
- [x] **party-twenty-spirit §3.5 cross-ref applied** (§17.6.4 honest framing table + conclusion)
- [x] **paper-gate honored** (§17.6.4 honest finding "No paper work touched" cites TODO-84)
- [x] **CPU-only docs** (no GPU retrain; per §4 honest finding)

---

## §6 One-line verdict

Wf-82 SHIPs **§17.6 L13-L16 candidate row** in `TODO/INDEX.md` referencing 4 candidates (L13/L14/L15/L16, 330 scan signals) + 6 R22-P plan items (P1-P4 active + P5/P6 reserved) + `audit_22_to_25.json` + LAMBDA_ROADMAP §4.2 proposed 4-row extension + `party-twenty-spirit` §3.5 honest framing — CPU-only docs, paper-gate honored, no GPU retrain executed today.