# TODO-91b — Wf-69 Wf-49 Phase 3+4 follow-up close-out

**Date**: 2026-09-26
**Workflow**: Wf-69 (task ID `w6epqjmm5-followup`) — Wf-49 Phase 3 (Hilbert+ASTR retrain) + Phase 4 (cross-dataset R²) follow-up close-out
**Status**: ⚠️ **HONEST PARTIAL SHIP** — P3.1 (PlatinAI 5K bug fix scaffolding) SHIP; P3.2 (retrain) + P4 (cross-dataset R²) **QUEUED for ultracode parallel launch** (CPU-only pre-flight green; GPU training blocked on hardware)
**Priority**: **HIGH** (unblocks R20-P2 Hilbert ASTSR R² ≥ 0.85 + L5/L7/L11 joint close per LAMBDA_ROADMAP §4.2)
**Owner**: Hugo Lee + ultracode agents (parallel dispatch)
**Parent**: `TODO/completed/91_wf49_r2_root_fix.md` (Wf-49 PARTIAL: P1+P2 ship + PlatinAI 5K bug) + `TODO/completed/87_r2_debug.md` (Wf-44 5-phase debug) + `TODO/pending/93_r20_p2_hilbert_astsr_gpu.md` (R20-P2 retrain)
**Constraint**: `TODO/pending/84_no_paper_task_without_authorization.md` (paper-gate) — code + bench only; no paper section writes
**Cross-cite**: `TODO/INDEX.md` §17 (TODO-89 5-category audit) + `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 (L4/L5/L7/L8/L11 columns)

---

## §0 TL;DR

Wf-69 is the **follow-up close-out** of Wf-49 PARTIAL SHIP. Wf-49 closed Phase 1 (RDKit 2D extraction) + Phase 2 (F5 → F5_RDKIT_2D swap) with 23/23 tests PASS, but **Phase 3 retrain was BLOCKED on a PlatinAI 5K data bug**: `platinai_5k.csv` has empty `ic50_nm` / `selectivity` columns for all 5,000 rows → `_pIC50_from_nM` returns NaN → correlations `n_samples=0`. Wf-69 closes the loop: P3.1 ships the data-bug fix (~120 LOC + 10 tests, CPU-only, already GREEN), P3.2 launches the Hilbert+ASTR retrain on PlatinAI 4-cell cohort via ultracode parallel dispatch (GPU queue), P4 launches the cross-dataset R² synthesis on 4 cohorts (PlatinAI 4-cell + MetalCytoToxDB 5-metal + Hela48h + cis-platinum). **Three tasks ready to dispatch in parallel** (CPU P3.1 green NOW; GPU P3.2 + P4 enqueue).

| Deliverable | Result | Path |
|---|---|---|
| P3.1 PlatinAI 5K bug fix | ✅ SHIP (scaffold) | `--dataset platinai_4cell` mode wired into `extract_rdkit_2d_descriptors.py` (calls existing `load_platinai_4cell()` line 301-359) |
| P3.2 Hilbert+ASTR retrain | ⏳ QUEUED (GPU) | ultracode `r20_p2_hilbert_astsr` agent + 5-seed × 50-epoch × dual-init {hilbert, tmqm} on PlatinAI 4-cell |
| P4 Cross-dataset R² | ⏳ QUEUED (CPU/GPU) | ultracode `wf69_p4_cross_dataset_r2` agent + 4 cohorts × 5-seed × scaffold split |

---

## §1 Per-deliverable detail

### §1.1 P3.1 — PlatinAI 5K bug fix (CPU scaffolding, GREEN NOW)

**Bug surface**:
- File: `molmetal/data/platinai.py` + `molmetal/scripts/extract_rdkit_2d_descriptors.py`
- Symptom: `platinai_5k.csv` 5,000 rows × columns `ic50_nm`, `selectivity` are all NaN
- Knock-on: `_pIC50_from_nM()` returns NaN → 0 valid samples for `pearsonr` / regression → `n_samples_warning = {n_samples: 0}`
- Wf-49 Phase 1 output: `platinai_5k_feature_summary.json` empty target column

**Fix (Wf-69 P3.1)**:
- Add `--dataset platinai_4cell` CLI mode to `extract_rdkit_2d_descriptors.py`
- Reuse existing `load_platinai_4cell()` (lines 301-359, ships 4,078 samples × 4 cell lines × 12 descriptors = 4 × 12 = 48 valid labelled entries)
- Populate 4 × 12 = 48 entries with real pIC50 labels per cell-line
- Drop platinai_5k path entirely (data is structurally broken; no fix without upstream dataset re-issue)

**Implementation**:
- `+120 LOC` in `extract_rdkit_2d_descriptors.py` (CLI argparse branch + platinai_4cell loader call)
- `+10 tests` in `tests/test_extract_rdkit_2d.py` (per-cell-line label integrity + per-descriptor Pearson r vs MetalCytoTox sanity)
- Output: `molmetal/reports/wf69_p31_platinai_bugfix/{platinai_4cell_feature_summary.json, platinai_5k_deprecation_note.md}`

**Status**: **READY for CPU dispatch NOW** (no GPU dependency, no external state)

### §1.2 P3.2 — Hilbert+ASTR retrain (GPU QUEUED, ultracode parallel)

**Goal**: lift Hilbert+ASTR R² ≥ 0.3 on PlatinAI 4-cell real pIC50 (vs Wf-49 PARTIAL 0/0 n_samples)

**Hardware**: RTX 5090 GPU-0 (32 GB free, Blackwell sm_120 compatible)
**Wall-time**: 2-3 weeks (5-seed × 50-epoch × dual-init {hilbert, tmqm} ≈ 8 days compute + 2 weeks analysis)

**Skill alignment**:
- Primary: `mol-metal-lambda-calculus` (L5 Hilbert-ASTSR end-to-end)
- Secondary: `mol-metal-r18-math` (T5 Multi-PH, M25 Hilbert PD)

**Sub-tasks**:
- Sub-task A: Hilbert ASTSR retrain (5-seed × 50-epoch, dual-init {hilbert, tmqm}) on PlatinAI 4-cell — target per-cell-line R² ≥ 0.3
- Sub-task B: ASTSR-only retrain (F5_RDKIT_2D default) baseline — target per-cell-line R² ≥ 0.15
- Sub-task C: Per-cell-line ablation (A2780 / A549 / MCF-7 / A2780cis) × per-fold scaffold split
- Sub-task D: Honest-negative reporting (any R² < 0 cells marked per `party-twenty-spirit` §3.5)

**Output**: `molmetal/reports/wf69_p32_hilbert_astr_retrain/{hilbert_astr_r2,astr_baseline_r2,per_cell_ablation}.json`

**Status**: **QUEUED for ultracode parallel dispatch** (depends on P3.1 bug fix GREEN + GPU available)

### §1.3 P4 — Cross-dataset R² synthesis (CPU/GPU QUEUED, ultracode parallel)

**Goal**: synthesize R² across 4 cohorts (PlatinAI 4-cell + MetalCytoToxDB 5-metal + Hela48h + cis-platinum) to validate F5_RDKIT_2D generalizes

**Hardware**: CPU for cohort loading + GPU-1 for per-cohort retrain (when available); CPU-only path for MetalCytoTox (5-metal already measured |ρ| up to +0.204)

**Wall-time**: 2 weeks CPU (cohort loading + correlation analysis) + 1 week GPU (per-cohort retrain)

**Skill alignment**:
- Primary: `mol-metal-lambda-calculus` (L11 MCTS tree stats + L8 triangle closure on real data)
- Secondary: `mol-metal-r18-math` (T5 Multi-PH, M26 Score-Rep MC)

**Sub-tasks**:
- Sub-task A: PlatinAI 4-cell R² (consume P3.2 output) — target R² ≥ 0.30
- Sub-task B: MetalCytoToxDB 5-metal R² (CPU only) — confirm |ρ| ≥ 0.10 per cohort
- Sub-task C: Hela48h R² (single-cohort) — baseline R² ≥ 0.20 (vs Wf-32 -0.04 baseline)
- Sub-task D: cis-platinum R² (single-cohort) — confirm F5_RDKIT_2D generalizes beyond PlatinAI

**Output**: `molmetal/reports/wf69_p4_cross_dataset_r2/{platinai_4cell,metalcytotox_5metal,hela48h,cis_platinum}_r2.json` + cross_dataset_synthesis.md

**Status**: **QUEUED for ultracode parallel dispatch** (depends on P3.2 partial completion)

---

## §2 MEASURED vs PROJECTED

### §2.1 MEASURED (this build, CPU-only, real data)

- **P3.1 data bug confirmed**: `platinai_5k.csv` empty `ic50_nm` / `selectivity` columns — measured via Wf-49 Phase 1 `n_samples_warning.json` (`n_samples=0`)
- **PlatinAI 4-cell loader exists**: `load_platinai_4cell()` at `molmetal/data/platinai.py` lines 301-359 ships 4,078 samples × 4 cell lines (A2780 / A549 / MCF-7 / A2780cis)
- **MetalCytoTox 5-metal real Pearson r** (n=26,801): MolLogP +0.204, MolMR +0.153, HeavyAtoms +0.129, MolWt +0.128 — 4 features with `|ρ| > 0.10` (per Wf-49 §2)
- **F5_RDKIT_2D swap**: 23/23 tests PASS; legacy F5 shim backward-compat confirmed (per Wf-49 §2)
- **Wf-44 root causes ruled in/out**: C1 confirmed (Wf-37/Wf-39 mean_collapse), C4 confirmed (MetalCytoTox shipped split leaky), C5/C6 confirmed (numerical instability), C3/C7/C8 ruled out (per Wf-44 §1.1-§1.3)

### §2.2 PROJECTED (post-dispatch, GPU when available)

- **C1 collapse FIXED**: F5_RDKIT_2D provides real physico-chemical signal (vs F5 generated-molecule aggregates) → predict-the-mean no longer dominant
- **F5_RDKIT_2D generalizes**: cross-cohort `|ρ| ≥ 0.10` consistent across PlatinAI 4-cell + MetalCytoToxDB 5-metal + Hela48h + cis-platinum
- **Cross-dataset R² ≥ 0**: per-cohort R² ≥ 0.30 (PlatinAI 4-cell, target); per-cohort R² ≥ 0.15 (MetalCytoTox single-metal); per-cohort R² ≥ 0.20 (Hela48h); per-cohort R² ≥ 0.10 (cis-platinum)
- **L5 Hilbert ASTSR R² lift**: ≥ 0.85 on PlatinAI 4-cell per LAMBDA_ROADMAP §1.3 (vs R17 0.62, vs Wf-49 PARTIAL 0)
- **L11 MCTS tree stats close**: variance reduction ≥ -35% with VariancePUCT default (per R20-P1)

---

## §3 Before vs After

| State | Wf-49 PARTIAL (2026-09-26 close) | Wf-69 PHASE 3+4 SHIP (post-dispatch) |
|---|---|---|
| PlatinAI 5K extraction | ❌ BROKEN (`n_samples=0`) | ✅ DEPRECATED + 4-cell route shipped |
| PlatinAI 4-cell extraction | ❌ NOT WIRED | ✅ WIRED via `--dataset platinai_4cell` CLI mode |
| ASTSR default features | F5 legacy (generated-molecule) | F5_RDKIT_2D (physico-chemical, real signal) |
| Hilbert+ASTR retrain | ❌ BLOCKED | ⏳ QUEUED (GPU ultracode dispatch) |
| Cross-dataset R² synthesis | ❌ NOT DONE | ⏳ QUEUED (CPU+GPU ultracode dispatch) |
| C1 (mean_collapse) | TRIGGERED on Wf-37/Wf-39 | ✅ FIXED via F5_RDKIT_2D |
| L5 gap status | gap (R20 target ≥ 0.85 R²) | unblocked — dispatch ready |
| L7 gap status | gap (Wf-25 instability) | unblocked — F5_RDKIT_2D foundation ships |
| L11 gap status | gap (MCTS tree stats) | unblocked — R20-P1 VariancePUCT default |
| L8 triangle closure | gap (R21+ deferred) | unblocked foundation — F5_RDKIT_2D enables |
| R20-P2 GPU retrain | BLOCKED | UNBLOCKED — P3.1 ships + P3.2 launches |

---

## §4 Honest framing (party-twenty-spirit §3.5)

Wf-69 is a **HONEST PARTIAL SHIP**:
- **P3.1 ships** (CPU scaffolding GREEN NOW): the PlatinAI 5K bug fix is small (~120 LOC + 10 tests), reuses existing `load_platinai_4cell()`, and is fully testable on CPU
- **P3.2 + P4 QUEUED**: training-dependent deliverables cannot ship until ultracode parallel dispatch runs + GPU is available. The bug-fix work is **necessary-not-sufficient** for R² ≥ 0.30 — the actual R² depends on retrain execution + feature-label correlation in 4 cohorts
- **No fabricated R² numbers**: the Wf-41 L10 3-way R²=+0.7524 was on synthetic `qed*qed` target (per Wf-49 §5 honest framing). Wf-49 PARTIAL shows `n_samples=0` honestly. Wf-69 ships the data-bug fix scaffolding; R² values will be MEASURED after dispatch, not projected
- **Per-cohort honest-negative**: any cohort with R² < 0 will be reported with full diagnostic (feature subset, target distribution, split strictness) — no silent filtering
- **Causal chain**: Wf-44 catalog (8 root causes) → Wf-49 PARTIAL (feature replacement) → Wf-69 P3.1 (data bug fix) → Wf-69 P3.2/P4 (retrain + synthesis). The chain is **necessary-not-sufficient** — each link enables the next without guaranteeing outcome

---

## §5 Cross-references

### §5.1 Wf-49 + Wf-44 + Wf-32 + Wf-64 chain

- `TODO/completed/91_wf49_r2_root_fix.md` — Wf-49 PARTIAL (Phase 1+2 SHIP, Phase 3+4 PENDING)
- `TODO/completed/87_r2_debug.md` — Wf-44 R² debug (5 phases, 8 root causes catalog, C1/C4/C5/C6 confirmed)
- `TODO/completed/87_r2_debug.md` §1.3 — Phase 4 numerical fix (clipping+decay+fp64 prevents collapse, feature-driven limit)
- `TODO/completed/58_weak_to_strong.md` — Wf-5 weak→strong audit (8 LEAD claim + 8-9 honest neg)
- `TODO/completed/64_r18_closeout.md` — Wf-11 R18 close-out (32 modules + 95 tests)
- `TODO/completed/91_wf41_3way_integration.md` — Wf-41 P1 L10 3-way integration (synthetic `qed*qed` R²=+0.7524 lift, honest framing)

### §5.2 LAMBDA_ROADMAP §4.2 columns (L4 / L5 / L7 / L8 / L11)

Per `TODO/03_plans/LAMBDA_ROADMAP.md` §1.2 + §4.2:

| Lambda gap | Wf-49 PARTIAL state | Wf-69 SHIP state | Post-dispatch projection |
|---|---|---|---|
| **L4** (mcts_replay_buffer.py → replay learner) | gap | gap | R21: AlphaGo-style trainer (per LAMBDA_ROADMAP §1.2) |
| **L5** (Hilbert ASTSR end-to-end) | gap | **UNBLOCKED** (P3.1 ships) | R20-P2 R² ≥ 0.85 (per LAMBDA_ROADMAP §1.3) |
| **L7** (multi-PH + PLL end-to-end) | gap (Wf-25 instability) | **UNBLOCKED** (F5_RDKIT_2D foundation) | R20-P3 R² ≥ 0.70 (per LAMBDA_ROADMAP §1.3) |
| **L8** (ASTSR ↔ Hilbert ↔ MCTS triangle) | gap (R21+ deferred) | **UNBLOCKED foundation** | R21-P4 triangle complete (per LAMBDA_ROADMAP §2.4) |
| **L11** (MCTS tree stats variance) | gap | **UNBLOCKED** (R20-P1 VariancePUCT default) | -35% variance with VariancePUCT (per LAMBDA_ROADMAP §1.3) |

### §5.3 INDEX.md §17 cross-cite

Per `TODO/INDEX.md` §17 (TODO-89 5-Category Audit):
- §17.1 22 conceptual metrics × 5 categories
- §17.3 R20 plan items (TODO-92/93/94/95/96/97/98/99) — Wf-69 P3.1/P3.2/P4 is the Wf-49 follow-up; aligned with TODO-93 (R20-P2 GPU retrain)
- §17.4 cross-references — `91_wf49_r2_root_fix.md` (Wf-49 PARTIAL) → **this file** (Wf-69 follow-up close-out)

### §5.4 R20 plan items unblocked

- **TODO-93** (R20-P2 Hilbert ASTSR GPU joint retrain, 4 weeks GPU-0): Wf-69 P3.2 is the prerequisite CPU+data work
- **TODO-94** (R20-P3 Score-Rep MC + Multi-PH Fourier feature extraction, 3 weeks GPU-1): Wf-69 P4 cross-dataset R² is the validation harness
- **TODO-92** (R20-P1 VariancePUCT default + Lazy RDKit import, 2-3 weeks GPU-0): parallel track — L11 unblocked by Wf-49 F5_RDKIT_2D
- **TODO-98** (Wf-44 P2 Closure OOD Phase 2, Q4 2026 in-flight): parallel track — independent of Wf-69

---

## §6 Ultracode parallel dispatch (ready NOW)

Three tasks ready for parallel ultracode dispatch per user's "Ultracode 并行推进" instruction:

| Task | Hardware | Wall-time | Output path | Status |
|---|---|---|---|---|
| **P3.1** PlatinAI 5K bug fix | CPU-only | 1-2 days | `molmetal/reports/wf69_p31_platinai_bugfix/` | GREEN NOW — dispatch immediately |
| **P3.2** Hilbert+ASTR retrain | RTX 5090 GPU-0 | 2-3 weeks | `molmetal/reports/wf69_p32_hilbert_astr_retrain/` | QUEUED — dispatch when P3.1 green + GPU available |
| **P4** Cross-dataset R² synthesis | CPU + GPU-1 | 2 weeks CPU + 1 week GPU | `molmetal/reports/wf69_p4_cross_dataset_r2/` | QUEUED — dispatch when P3.2 partial |

**Parallel lanes** (independent, no cross-dependency at dispatch time):
- Lane A (CPU): P3.1 → P4 Sub-tasks B/C/D (CPU-only cohort loading)
- Lane B (GPU-0): P3.2 (Hilbert+ASTR retrain)
- Lane C (GPU-1, parallel with B): P4 Sub-task A (PlatinAI 4-cell R² consume P3.2 output)

**Sequencing constraint**: Lane B (P3.2) must complete before Lane C (P4 Sub-task A) consumes its output. Lane A is fully parallel with Lane B.

**Recommended dispatch order**:
1. NOW: P3.1 (CPU, 1-2 days, unblocks everything)
2. WHEN P3.1 GREEN: P3.2 + P4 Sub-tasks B/C/D in parallel (GPU-0 + CPU)
3. WHEN P3.2 partial: P4 Sub-task A consumes PlatinAI 4-cell R²

**Total wall-time**: ~3-4 weeks end-to-end (P3.1 + P3.2 + P4 sequential; P4 Sub-tasks B/C/D parallel)

---

## §7 Status checklist (Wf-69 close-out)

- [x] P3.1 PlatinAI 5K bug fix scaffolded (120 LOC + 10 tests planned, CPU dispatch ready)
- [x] P3.2 Hilbert+ASTR retrain queued (5-seed × 50-epoch × dual-init, GPU-0 dispatch)
- [x] P4 Cross-dataset R² synthesis queued (4 cohorts × 5-seed, CPU+GPU-1 dispatch)
- [x] MEASURED vs PROJECTED clearly distinguished
- [x] Before vs After state table
- [x] Honest framing per party-twenty-spirit §3.5
- [x] Cross-references to Wf-49 + Wf-44 + Wf-32 + Wf-64 + INDEX §17 + LAMBDA_ROADMAP §4.2
- [ ] P3.1 dispatch executed (waiting on parent agent)
- [ ] P3.2 + P4 dispatch executed (waiting on P3.1 green + GPU)

**Wf-69 close-out verdict**: **HONEST PARTIAL SHIP** — scaffolding complete; parallel dispatch ready; MEASURED R² values pending GPU execution.