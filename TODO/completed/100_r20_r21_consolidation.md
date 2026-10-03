# Wf-70 — R20/R21 Plan Status Consolidation Close-out

**Date**: 2026-09-26
**Workflow**: Wf-70 (Phase 2: R20 plan status ledger verification + LAMBDA_ROADMAP §4.2 update + INDEX.md §17 cross-ref sweep)
**Status**: ✅ **SHIP** — R20 status ledger SHIP + LAMBDA_ROADMAP §4.2 expanded L1-L8 → L1-L12 + INDEX.md §17 cross-ref sweep SHIP
**Priority**: HIGH (consolidates Wf-65/66/67/68/69 + Wf-55 ship surface for R21 launch readiness)
**Owner**: Hugo Lee + ultracode agents

**Parent**:
- `TODO/completed/r20_status_ledger_2026_09_26.md` (Wf-70 Phase 1 — per-R20-P ledger)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 (L1-L8 trajectory, R20-R25)
- `TODO/INDEX.md` §17.3 (R20 plan table + 5-category audit)
- `TODO/pending/92_r20_p1_*.md` ... `TODO/pending/99_r20_skill_ecosystem.md` (8 active R20 plans)
- `TODO/pending/84_no_paper_task_without_authorization.md` (paper-gate)

---

## §1 TL;DR

Wf-70 Phase 2 合并 Wf-65..69 + Wf-55 ship surface into 3 个 consolidated artefacts:

| Deliverable | Status | Source |
|---|---|---|
| **R20 status ledger** | ✅ SHIP — 7/8 R20-P verified + 1 IN FLIGHT (R20-P1) | `completed/r20_status_ledger_2026_09_26.md` (Wf-70 Phase 1, 308 lines, 50 sub-tasks audited) |
| **LAMBDA_ROADMAP §4.2 update** | ✅ SHIP — L1-L8 → L1-L12 (added L9 Lambda×CFM e2e / L10 3-way / L11 MCTS tree / L12 OOD); 12 columns × 8 stages table | `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 (rev 2026-09-26) |
| **INDEX.md §17 cross-ref sweep** | ✅ SHIP — §17.3 R20 plan table + §17.4 cross-references + §17.5 honest framing (NEW) cross-ref to 8 active TODO + 7 close-out files | `TODO/INDEX.md` §17 (rev 2026-09-26) |

**Aggregate**: **3 NEW consolidated artefacts ship** (R20 ledger + roadmap update + INDEX sweep) + **8 active R20 plan cross-references** (TODO-92..99) + **7 R20 close-out files cross-referenced** (TODO-93..99) + **8 R20 NEW skill cross-references** (S1-S8). R21 launch readiness: ✓.

---

## §2 Per-deliverable

### §2.1 R20 status ledger (`completed/r20_status_ledger_2026_09_26.md`, Wf-70 Phase 1)

**Status**: ✅ SHIP (308 lines, 50 sub-tasks audited; 50 JSON files cross-referenced).

| R20-P | Plan item | Wf | Close-out file | Verification |
|---|---|---|---|---|
| **R20-P1** | VariancePUCT default + Lazy RDKit | Wf-65 | (NOT YET) | ⚠️ IN FLIGHT (sign-flip audit SHIP; lazy RDKit partial) |
| **R20-P2** | Hilbert-ASTSR + Multi-PH + PLL + Triangle | Wf-56 | `completed/93_*.md` (16 KB) | 🟡 HONEST PARTIAL (3/7 VERIFIED, 4/7 DEFERRED GPU) |
| **R20-P3** | Multi-PH Fourier 32-bin + PLL 128-d + Score-Rep | Wf-66 | `completed/94_*.md` (25 KB) | 🟡 HONEST PARTIAL (2/6 VERIFIED, 3/6 DEFERRED GPU) |
| **R20-P4** | 3-way MetalCytoToxDB + Closure d-block + 100-pocket | Wf-67 | `completed/95_*.md` (22 KB) | 🟡 HONEST-NEGATIVE (1/6 VERIFIED, 3/6 DEFERRED, 2/6 NOT MET) |
| **R20-P5** | Closure + Catherine 6/6 | Wf-68+Wf-57+Wf-67 | `completed/96_*.md` (30 KB) | 🟡 HONEST PARTIAL (3/5 VERIFIED, 2/5 NOT MET) |
| **R20-P6** | Wet-lab IC50 cooperation | Wf-58 | `completed/97_*.md` (25 KB) | 🟡 PARTIAL (3/6 VERIFIED, 3/6 DEFERRED wet-lab) |
| **Wf-44 P2** | Closure OOD Phase 2 (L12) | Wf-57 | `completed/98_*.md` (24 KB) | ✅ PHASE 2 PARTIAL (4/5 VERIFIED, 1/5 DEFERRED) |
| **R20 skills** | 8 NEW skills (4 R20 + 4 R21) | Wf-55 | `completed/99_*.md` (19 KB) | 🟡 PARTIAL (3/9 VERIFIED; 2/8 FULL + 4/8 stub + 2/8 NOT) |

**Aggregate**: **15/50 sub-tasks VERIFIED**; **30/50 DEFERRED** (GPU / wet-lab / paper-gate); **3/50 NOT MET** (R20-P4 P4.2 reachability 0.5962 < 0.95; R20-P5 P5.2 reachability 0.5744 < 0.95; Wf-44 P2 per-domain empty dir); **2/50 IN FLIGHT** (R20-P1 P1.2 + close-out TODO file).

**0/8 R20-P fully MEASURED on disk** — all 7 close-out files honestly report GPU / wet-lab phases as deferred per `party-twenty-spirit §3.5`.

### §2.2 LAMBDA_ROADMAP §4.2 update (`TODO/03_plans/LAMBDA_ROADMAP.md`)

**Status**: ✅ SHIP — §4.2 row count expanded L1-L8 (8 rows) → L1-L12 (12 rows); L9-L12 NEW rows added:

| Gap | Description | Added column | Source |
|---|---|---|---|
| **L9** | Lambda × CFM e2e pipeline | R20-P2 GPU retrain cfm_to_lambda ≥ 0.95 | `completed/83h_lambda_cfm_pipeline.md` + Wf-40 P1+P2 (`bridge_rmse=1.183`) |
| **L10** | Deflexpressor + Multi-PH + ASTSR 3-way | R20-P4 real pIC50 R² ≥ +0.30 | `completed/91_wf41_3way_integration.md` (`3way R² = -0.3595`) |
| **L11** | MCTS tree per-node statistics | R20-P2 hit_rate ≥ 0.95 | `completed/83a_lambda_core.md` Wf-36 P3 (`vuln_f/g CRITICAL`) |
| **L12** | Closure theorem cross-domain OOD | R20-P5 + Wf-44 P2 | `completed/92_wf41_phase2_closure_ood.md` (OOD 0.5581 / 0.0300) |

**Before**: 8-gap table L1-L8 only. **After**: 12-gap table L1-L12, **8 stages × 12 rows = 96 cell grid** with R17/R18 ship / R19 ship / Wf-36/37/40/41/44 R20 bench / R20 ship / R21 ship / R22+ ship columns. R20 close 9/12 gap (L1/L2/L5/L6/L7/L9/L10/L11/L12) + L4 retrain; R21 close 3/12 (L3/L4/L8) for **12/12 全 close**.

### §2.3 INDEX.md §17 cross-ref sweep (`TODO/INDEX.md` §17)

**Status**: ✅ SHIP — §17.3 R20 plan table extended; §17.4 cross-references cross-link all 8 active pending TODO + 7 close-out files + 8 NEW skill directories; **NEW §17.5 honest framing** added.

**§17.3 R20 plan table** (8 rows × 5 cols):
- R20-P1 → TODO-92 → 2-3 wk GPU-0
- R20-P2 → TODO-93 → 4 wk GPU-0 (CRITICAL)
- R20-P3 → TODO-94 → 3 wk GPU-1
- R20-P4 → TODO-95 → 4 wk GPU-1 + 2 wk GPU-0
- R20-P5 → TODO-96 → 2 wk CPU + 1 wk GPU-0
- R20-P6 → TODO-97 → wet-lab 6 mo
- L12 P2 → TODO-98 → 2 wk CPU (Q4 2026)
- R20 Skills → TODO-99 → 4 wk CPU docs

**§17.4 cross-references**: 16 files cross-linked (8 active pending TODO + 7 close-out files + 1 ROUTING_POLICY).

**§17.5 honest framing (NEW)**: 4 行 honest ledger per `party-twenty-spirit §3.5`:
1. 0/8 R20-P fully MEASURED on disk
2. 3/8 partial MEASURED not gate-meeting (R20-P4 / R20-P5 / Wf-44 P2)
3. 5/8 code+skill+smoke SHIP, GPU phases unblocked
4. Structural reachability 0.5744-0.5962 ≪ 0.95 = NOT bug, structural ceiling

---

## §3 MEASURED vs PROJECTED

### §3.1 MEASURED (on disk today, 2026-09-26)

| Item | Value | Source |
|---|---:|---|
| R20-P close-out files SHIP | **7 / 8** | `completed/{93,94,95,96,97,99}_r20_p*.md` + `completed/98_wf44_*.md` |
| R20 NEW skills SHIP (FULL) | **2 / 8** (S2 + S3) | `.claude/skills/mol-metal-{hilbert-astsr-e2e,multiph-pll}/` |
| R20 NEW skills stub-shell | **4 / 8** (S5-S8) | `.claude/skills/mol-metal-{closure-catherine,wetlab-protocol,triangle-closure,mcts-replay-learner}/` |
| R20 NEW skills NOT created | **2 / 8** (S1 + S4) | n/a |
| LAMBDA_ROADMAP §4.2 row count | **12** (L1-L12) | `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 (this rev) |
| INDEX §17 cross-ref count | **16** files (8 active TODO + 7 close-out + 1 routing policy) | `TODO/INDEX.md` §17.4 (this rev) |
| Sub-tasks VERIFIED | **15 / 50** | `completed/r20_status_ledger_2026_09_26.md` §5 |
| Sub-tasks DEFERRED | **30 / 50** | GPU + wet-lab + paper-gate |
| Sub-tasks NOT MET | **3 / 50** | R20-P4/P5 reachability, Wf-44 P2 empty dir |
| Sub-tasks IN FLIGHT | **2 / 50** | R20-P1 P1.2 + close-out TODO |

### §3.2 PROJECTED (R20-P GPU retrain completion)

| Item | Target | Required | Block |
|---|---|---|---|
| R20-P fully MEASURED | 0 → 6 / 8 (R20-P2..P6 + Wf-44 P2) | 16 wk GPU + 6 wk CPU | Wf-23 GPU ultracode queue exit + Blackwell sm_120 single-card |
| R20 reachability ≥ 0.95 | 0.5744-0.5962 → 0.95 | SMARTS coverage expansion + max_depth ≥ 2 | Structural (not bug) + GPU retrain |
| S1 / S4 NEW skill creation | stub → FULL | 2 wk CPU docs each | n/a (CPU-only docs) |
| Per-domain reachability (Wf-44 P2) | empty → ≥ 0.85 (PlatinAI 5K + MetalCytoToxDB + tmQM) | 1 wk GPU-0 | Q4 2026 GPU slot |

---

## §4 Before vs After

| Aspect | Before (Wf-65..69 fragmented) | After (Wf-70 consolidated) |
|---|---|---|
| **R20 ledger** | 7 separate ship reports (Wf-65/66/67/68/69 + Wf-55), each standalone | 1 consolidated ledger (`completed/r20_status_ledger_2026_09_26.md`, 50 sub-tasks × 8 R20-P × 3 verdicts) |
| **LAMBDA_ROADMAP §4.2** | 8 gap L1-L8 row table | 12 gap L1-L12 row table (L9-L12 added) with 96 cell grid |
| **INDEX §17 cross-ref** | 8 active TODO + 5 close-out files cross-linked | 8 active TODO + **7 close-out files** + 8 NEW skill directories cross-linked + **NEW §17.5 honest framing** |
| **R21 launch readiness** | ⏳ unclear (5 ship reports + Wf-55 isolated) | ✓ ready (consolidated ledger shows R20 base surface for R21-P1..P4) |
| **Honest framing** | implicit in 7 ship reports | **explicit** in §17.5 (4-row ledger per `party-twenty-spirit §3.5`) |
| **Reachability gap diagnostic** | scattered across Wf-67/68 reports | **centralized** in §1.4-1.5 + §2.4 (structural ceiling, not bug) |

**Net effect**: Wf-70 collapses 7 fragmented ship reports into 3 consolidated artefacts (1 ledger + 1 roadmap update + 1 INDEX sweep) — R21 launch readiness goes from ⏳ to ✓.

---

## §5 Honest framing per `party-twenty-spirit §3.5`

**0 容忍 + 不掩盖**:

| Honest finding | Evidence |
|---|---|
| **0 / 8 R20-P fully MEASURED on disk** | Every R20-P close-out file honestly reports GPU retrain phases as deferred. No fabricated MEASURED numbers. |
| **3 / 8 R20-P NOT MEETING gate** | R20-P4 P4.2 reachability **0.5962** < gate **0.95** (gate failed); R20-P5 P5.2 reachability **0.5744** < gate **0.95** (gate failed); Wf-44 P2 per-domain reachability **DEFERRED** (empty dir, not yet measured). |
| **5 / 8 R20-P code + skill + smoke SHIP only** | GPU retrain phases unblocked but pending Wf-23 GPU ultracode queue exit + Blackwell sm_120 single-card constraint. |
| **Structural reachability issue is NOT a bug** | Per-periodic-block ceiling (3d=0.6682 / 4d=0.5813 / 5d_La=0.4918) + SMARTS coverage gaps + `max_depth=1` ceiling. Lifting requires SMARTS coverage expansion + `max_depth ≥ 2` + GPU retrain. |
| **R20-P1 is the ONLY R20-P without close-out file** | `completed/92_r20_p1_*.md` NOT yet written — sign-flip audit SHIP, lazy RDKit import partial, GPU trajectory + close-out pending. |
| **Wet-lab cooperation 6-month deferred** | ORNL / PKU / 协和 outreach 0 progress today; in-silico dry-run + D-MPNN prediction layer SHIP; cell culture + IC50 assay + in-silico-vs-wet-lab agreement = future work. |
| **2 / 8 R20 NEW skills NOT created** | S1 `mol-metal-variance-puct` (R20-P1) + S4 `mol-metal-3way-integration` (R20-P4) deferred; 4 skills stub-shell only (S5-S8). |
| **Paper work forbidden** | Per `pending/84_no_paper_task_without_authorization.md` — no `paper/sections/*.tex`, no `paper/main.tex` rebuilds, no reviewer rebuttal without explicit user OK. |

**Per `party-twenty-spirit §3.5`**: All 7 R20 close-out files (TODO-93..99) + this Wf-70 ledger ship **honest framing** — no MEASURED numbers fabricated, no PROJECTED numbers labeled as MEASURED, no GPU/wet-lab phases silently claimed ship.

---

## §6 Cross-reference

### §6.1 Wf-65..69 ship reports (R20 close-out surface)

- `TODO/completed/93_r20_p2_hilbert_astsr_gpu.md` (16 017 bytes, Wf-56 R20-P2 close-out) — L4+L5+L7+L8+L11
- `TODO/completed/94_r20_p3_multiph_pll_feature.md` (25 004 bytes, Wf-66 R20-P3 close-out) — L7+L3
- `TODO/completed/95_r20_p4_3way_100pocket.md` (22 048 bytes, Wf-67 R20-P4 close-out) — L6+L10
- `TODO/completed/96_r20_p5_closure_catherine.md` (30 368 bytes, Wf-68+Wf-57+Wf-67 R20-P5 close-out) — L1+L6+L12
- `TODO/completed/97_r20_p6_wetlab_ic50_dryrun.md` (25 417 bytes, Wf-58 R20-P6 close-out) — m18/m19/m20
- `TODO/completed/98_wf44_closure_ood_phase2.md` (24 180 bytes, Wf-57 Wf-44 P2 close-out) — L12 Phase 2
- `TODO/completed/99_r20_skill_ecosystem.md` (19 483 bytes, Wf-55 R20 skill ecosystem close-out) — 8 NEW skills (S1-S8)

### §6.2 INDEX.md §17 (5-category audit + R20 plan table + honest framing)

- `TODO/INDEX.md` §17.1 — 22 conceptual metrics × 5 categories (7 STRONG_REAL / 11 WEAK_REAL / 5+ MOCK / ~6 UNDEFINED / R20 PROJECTED)
- `TODO/INDEX.md` §17.2 — per-question status (Q1-Q5 user audit)
- `TODO/INDEX.md` §17.3 — R20 plan items table (8 rows × 5 cols)
- `TODO/INDEX.md` §17.4 — cross-references (16 files: 8 active TODO + 7 close-out + 1 routing policy)
- `TODO/INDEX.md` §17.5 — honest framing (NEW this rev, per `party-twenty-spirit §3.5`)

### §6.3 LAMBDA_ROADMAP §4.2 (12 gap L1-L12 trajectory)

- `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 (rev 2026-09-26) — 12 rows × 8 stages table (R17/R18 ship / R19 ship / Wf-36/37/40/41/44 R20 bench / R20 ship / R21 ship / R22+ ship)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §1.2 — R20-P1..P6 plan items (6 phases × 12 weeks)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §1.5 — R20 skill ecosystem (4 NEW skills)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §2.4 — R21 skill ecosystem (4 NEW skills)
- `TODO/03_plans/LAMBDA_ROADMAP.md` §4.3 — Paper trajectory R20-R25

### §6.4 TODO/pending/92..99 (8 active R20 plans)

- `TODO/pending/92_r20_p1_variance_puct_lazy_rdkit.md` (R20-P1, 2-3 wk GPU-0)
- `TODO/pending/93_r20_p2_hilbert_astsr_gpu.md` (R20-P2, 4 wk GPU-0, CRITICAL)
- `TODO/pending/94_r20_p3_multiph_pll_feature.md` (R20-P3, 3 wk GPU-1)
- `TODO/pending/95_r20_p4_3way_100pocket.md` (R20-P4, 4 wk GPU-1 + 2 wk GPU-0)
- `TODO/pending/96_r20_p5_closure_catherine.md` (R20-P5, 2 wk CPU + 1 wk GPU-0)
- `TODO/pending/97_r20_p6_wetlab_ic50.md` (R20-P6, wet-lab 6 mo)
- `TODO/pending/98_wf44_closure_ood_phase2.md` (Wf-44 P2 L12 Phase 2, 2 wk CPU Q4 2026)
- `TODO/pending/99_r20_skill_ecosystem.md` (R20 skill ecosystem, 4 wk CPU docs)

### §6.5 Supporting docs (parent anchors)

- `TODO/completed/r20_status_ledger_2026_09_26.md` — Wf-70 Phase 1 (308 lines, 50 sub-tasks × 8 R20-P × 3 verdicts)
- `TODO/pending/84_no_paper_task_without_authorization.md` — paper-gate (CRITICAL)
- `TODO/completed/83j_lambda_closeout.md` §3 + §5 (TODO-83 umbrella, L1-L12 + R20-P1..P6 routing)
- `TODO/completed/89_wf43_routing_policy.md` (Wf-43 routing policy — 27 routed metrics + 4 MOCK_DELETED)
- `TODO/completed/85_mock_audit.md` (Wf-42 mock inventory: 8 critical + 23 major + 98 minor)
- `.claude/skills/party-twenty-spirit/SKILL.md` §3.5 — 0 容忍 + 不掩盖 (always apply)
- `.claude/skills/mol-metal-gpu-execution/SKILL.md` §5 — kanzi contention policy
- `molmetal/reports/wf{65,66,67,68,57,58}_*/` — Wf-65/66/67/68/57/58 raw bench JSONs

---

## §7 Final verdict

**Wf-70 Phase 2 one-liner**: **3 consolidated artefacts ship today** (R20 status ledger + LAMBDA_ROADMAP §4.2 L1-L12 update + INDEX.md §17 cross-ref sweep); **7/8 R20-P close-out files ship** (R20-P1 still in flight); **15/50 sub-tasks VERIFIED + 30/50 DEFERRED + 3/50 NOT MET + 2/50 IN FLIGHT**; **8 NEW skill cross-references** (2 FULL + 4 stub + 2 NOT); **0/8 R20-P fully MEASURED on disk** (honest framing per `party-twenty-spirit §3.5`).

**R21 launch readiness**: ✓ — LAMBDA_ROADMAP §4.2 L1-L12 table provides complete trajectory view for R21-P1 (L3 ScoreRep+VariancePUCT+MCTS ablation), R21-P2 (L4 AlphaGo-style replay learner), R21-P3 (L6 closure 30 metal), R21-P4 (L8 triangle ship → paper §3.6).

**R20 GPU retrain readiness**: ⏳ — pending Wf-23 GPU ultracode queue exit + Blackwell sm_120 single-card constraint + RTX 5090 GPU-0/1 contention policy per `mol-metal-gpu-execution §5`. Cumulative wall-time ~16 weeks GPU (serialized) or ~6 weeks GPU (parallelized across GPU-0 + GPU-1).

**Paper work readiness**: 🚫 FORBIDDEN per `pending/84_no_paper_task_without_authorization.md` — all `paper/*` modifications require explicit user OK.

**Date**: 2026-09-26 (Wf-70 Phase 2 ship — R20/R21 plan status consolidation close-out)
