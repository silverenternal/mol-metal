# Wf-303B — Code Task Categorization (actionable-now vs blocked)

**Date**: 2026-09-29
**Owner**: Hugo Lee (categorize) — ultracode execution to follow
**Trigger**: user said "现在看一下todo目录里面还有没有说明之前规划过的但是从来没做过的任务（代码类），抓紧时间做起来"
**Source**: `TODO/pending/` (17 files) + `TODO/03_plans/IMPROVEMENT_ENUM_2026_09_27.md` (4 P0 tasks) + `TODO/03_plans/IMPROVEMENT_QUEUE_2026_09_27.md` (3 P0 remaining) + `TODO/03_plans/R18_MATH_MIGRATION_PLAN.md` (P0-1..P1-7) + `TODO/03_plans/R18_PLAN.md` (Phase 0-4) + `TODO/03_plans/LAMBDA_ROADMAP.md` §1 (R20-P1..P6).
**Skill alignment**: `party-twenty-spirit` §3.5 honest framing + `mol-metal-gpu-execution` §0 "kill then adapt" GPU directive + `mol-metal-anticancer` Tier-1 SOTA comparison.

---

## §1 Categorization rubric

| Code category | Definition | Gate |
|---|---|---|
| **actionable-now** | Dependencies met; can be implemented immediately (CPU-only or low-contention GPU) | No external authorization / wet-lab / other-TODO blocker |
| **blocked-on-gpu** | Needs explicit fresh user OK + GPU launch > 1 hr wall (per `mol-metal-gpu-execution` §0 "kill then adapt" standing directive) | User must authorize each launch |
| **blocked-on-external** | Wet-lab LOI / external API / cloudflare / data download — NOT code-executable today | Wet-lab PI signoff + 6-month timeline |
| **blocked-on-other-todo** | Depends on another TODO still pending | Halt until blocker ships |

---

## §2 Master inventory of code tasks with plans but no implementation

Cross-referenced from `TODO/pending/` + `TODO/03_plans/` (excluding files already shipped to `TODO/completed/`).

### §2.1 IMPROVEMENT_ENUM / QUEUE P0 tasks (3-4 tasks)

| ID | Task | Resources | ETA | Source |
|---|---|---|---|---|
| TODO-134 | m15 Docking RMSD redesign (3 phases: ref extract → Vina redocking → metric JSON) | CPU + GPU-0 | 5-8 days | `pending/134_*.md` |
| TODO-135 Phase 2 | m19 LOMO 5-metal × 5-seed × 50-epoch re-bench | GPU-1 (or CPU fallback) | 2 weeks | `pending/135_*.md` §2 |
| TODO-138 | Paper v2 + JCIM submission package | CPU-only docs | 5-7 days | `pending/138_*.md` |

### §2.2 Wf-123/Wf-124 P3 gap TODOs (6 tasks — see `183_wf123_wf124_p3_gap_routing_plan.md`)

| ID | Task | Resources | ETA | Status |
|---|---|---|---|---|
| TODO-177 | M1+M2 downstream metric lift (composite of G3/G4/G5/G6 consumer-side) | Blackwell + RTX 5090 dual | 6 weeks staggered | 🟡 OPEN |
| TODO-178 | metalcytotox 5-metal CSV on disk + schema audit + FAIL-on-missing | CPU-only | 0.5-1 day | 🟢 PARTIAL (CSV exists; schema MISSING pic50/cell_line/source) |
| TODO-179 | W₂ alignment on PlatinAI 4-cell (L5 R² lift) | RTX 5090 | 4-5 hr wall | 🟡 OPEN |
| TODO-180 | 14/15 (metal,seed) PENDING (only Ru seed=0 complete) | RTX 5090 GPU-1 | 1.8 hr wall | 🔴 OPEN |
| TODO-181 | 50-epoch full sweep (current 30-epoch undertraining) | RTX 5090 GPU-1 | 5 hr wall | 🔴 OPEN |
| TODO-182 | Per-innovation ablation (W₂ only / closure only / YuelBond only) | RTX 5090 | 7.5 hr wall | 🔴 OPEN |

### §2.3 Wf-234 frontier math integration (1 task)

| ID | Task | Resources | ETA | Status |
|---|---|---|---|---|
| TODO-234 | Wf-234 frontier math wire-in (4 of 7 modules → main CFM pipeline) | CPU + GPU for retrain | 3-4 weeks | Audit SHIPPED; integration gated on user OK + ultracode launch |

### §2.4 Wf-219 Krasnov-aligned (1 task)

| ID | Task | Resources | ETA | Status |
|---|---|---|---|---|
| TODO-219 | Wf-219 Ru + Ir pending (3-metal Re/Os/Rh ✅ SHIPPED; Ru + Ir NOT run) | RTX 5090 GPU-1 | ~10 min wall | Pending fresh user authorization |

### §2.5 Wetlab planning artifacts (7 files — all DRAFT/TEMPLATE, no wet-lab partner LOI)

| ID | Task | Resources | ETA | Status |
|---|---|---|---|---|
| wetlab_ic50_analysis_pipeline | IC50 analysis CPU pipeline (template only) | CPU-only | template ready | 🟡 Active pipeline doc |
| wetlab_loi_ornl_protein_metal | DRAFT LOI letter — ORNL protein-metal binding | — | — | 🚫 NOT sent |
| wetlab_loi_pku_cisplatin_ic50 | DRAFT LOI letter — PKU cisplatin IC50 | — | — | 🚫 NOT sent |
| wetlab_loi_xiehe_nci60 | DRAFT LOI letter — 协和 NCI-60 | — | — | 🚫 NOT sent |
| wetlab_sop_binding_affinity_assay | TEMPLATE SOP | — | — | 🚫 TEMPLATE only |
| wetlab_sop_cellular_uptake_assay | TEMPLATE SOP | — | — | 🚫 TEMPLATE only |
| wetlab_sop_cytotoxicity_assay | TEMPLATE SOP | — | — | 🚫 TEMPLATE only |

### §2.6 Active rules (no implementation work — meta rule)

| ID | Task | Status |
|---|---|---|
| TODO-84 | `84_no_paper_task_without_authorization.md` | 🚫 Active rule — applies to all workflows |

---

## §3 Categorization (the verdict)

### §3.1 actionable-now (8 tasks — CPU-only or low-contention GPU)

| Rank | ID | Task | Why actionable | Impact |
|---:|---|---|---|---|
| 1 | **TODO-178** | metalcytotox 5-metal CSV audit + FAIL-on-missing | CSV exists at `.data/molmetal/metalcytotox_5metal_multitask.csv` (26801 rows); smoke test currently SKIPs; needs FAIL-on-missing + schema gap fill (`pic50`, `cell_line`, `source`) + audit JSON. Pure CPU RDKit + file ops. ~0.5-1 day. | **HIGH** — closes hard-dependency smoke test for Wf-123 P3 + Wf-124 P3 + Wf-145 + Wf-176 (5+ downstream consumers) |
| 2 | **TODO-134** | m15 Docking RMSD redesign Phase 1 (ref SMILES extraction) | Phase 1 = ref-ligand SMILES extraction + InChIKey check (CPU-only RDKit); unblocks Phase 2 Vina redocking. 5/5 pockets READY per Wf-126 P1. ~2 days Phase 1 alone. | **HIGH** — ref data feeds paper §4.5.7 + Phase 2 RMSD bench |
| 3 | **TODO-138** | Paper v2 + JCIM submission package | CPU-only docs work; weakly depends on TODO-134 (m15 data placeholder acceptable). Paper builds clean + cover letter + highlights + reviewers + `submission_package.zip`. ~5-7 days. | **HIGH** — gates paper submission; today's blockers are docs not code |
| 4 | **TODO-234** (CPU portions) | Wf-234 frontier math wire-in (4 wire points × CPU work) | Audit DONE; 4 of 7 modules wire-ready (W2.1 SE(3) SB trio + W2.3 multi-PH already shipped as adapters; only "drop into `flow_matching_lipman`" wire step remains). Pure CPU import wire + glue. | **MEDIUM-HIGH** — closes R18 P1-1..P1-3 wire tasks; paper §3 honest-framing ready |
| 5 | **TODO-178b** | test_metalcytotox_csv_integration_smoke FAIL-on-missing hardening | Test gate: `pytest ≥12 tests PASS` (per IMPROVEMENT_ENUM §1.1); CPU-only | **MEDIUM** — hygiene per `party-twenty-spirit` §3.5 (silent skip ≠ measured) |
| 6 | **wetlab_ic50_analysis_pipeline** (template part) | IC50 analysis pipeline template scaffold | CPU-only NumPy/SciPy template ready per `molmetal/scripts/template_ic50_pipeline.py`; scaffold only. ~1-2 days. | **MEDIUM** — gates paper §5 wet-lab claim (but wet-lab gate deferred 6 months) |
| 7 | **TODO-234b** | Wf-234 production-wire vs research-only tag for 3 research-only modules (`bass_existence`, `sb_bass_martingale`, `monge_ampere_regularity`) | CPU-only paper §3 cross-ref writing (1 week per Wf-234 §1) | **MEDIUM** — paper §3 method-reference completeness |
| 8 | **TODO-84 (active rule update)** | Sweep `TODO/pending/` for any file violating rule 84 (paper task without authorization) | CPU-only audit + archive | **LOW** — meta-hygiene |

### §3.2 blocked-on-gpu (8 tasks — all need explicit fresh user OK)

| ID | Task | Wall-time | Why blocked | Notes |
|---|---|---|---|---|
| TODO-219 | Wf-219 Ru + Ir pending | ~10 min wall (RTX 5090 GPU-1) | Fresh user authorization required | Smallest GPU ask — could be in same session as next launch |
| TODO-180 | 14/15 (metal,seed) PENDING (Wf-124 P3 gap G4) | 1.8 hr wall | RTX 5090 GPU-1 | 14 runs × ~7.7 min mean |
| TODO-181 | 50-epoch full sweep (Wf-124 P3 gap G5) | 5 hr wall | RTX 5090 GPU-1 | 5 metal × 3 seed × 50ep |
| TODO-179 | W₂ alignment on PlatinAI 4-cell (Wf-123 P3 gap G3) | 4-5 hr wall | RTX 5090 | Biggest single-metric L5 R² lift |
| TODO-182 | Per-innovation ablation (Wf-124 P3 gap G6) | 7.5 hr wall | RTX 5090 | 4 configs × 5 metal × 3 seed = 60 runs |
| TODO-135 Phase 2 | m19 LOMO 5×5×50 re-bench | 2 weeks wall | RTX 5090 or Blackwell | 5 metal × 5 seed × 50ep = 1250 runs |
| TODO-134 Phase 2/3 | m15 RMSD redesign (Phase 2 = Vina redocking + metric JSON) | 5-8 days wall | CPU + GPU-0 | Phase 1 (CPU) unblocks Phase 2/3 |
| TODO-177 | M1+M2 downstream metric lift (composite) | 6 weeks staggered | Blackwell + RTX 5090 dual | Composite of G3/G4/G5/G6 consumer-side lifts |

### §3.3 blocked-on-external (6 tasks — wet-lab LOI / SOP)

| ID | Task | Blocker | ETA |
|---|---|---|---|
| wetlab_loi_ornl_protein_metal | ORNL protein-metal binding LOI | ORNL PI signoff | 6-month timeline (no LOI sent) |
| wetlab_loi_pku_cisplatin_ic50 | PKU cisplatin IC50 LOI | PKU PI signoff | 6-month timeline |
| wetlab_loi_xiehe_nci60 | 协和 NCI-60 LOI | 协和 PI signoff | 6-month timeline |
| wetlab_sop_binding_affinity_assay | SOP template | Wet-lab PI review | TBD |
| wetlab_sop_cellular_uptake_assay | SOP template | Wet-lab PI review | TBD |
| wetlab_sop_cytotoxicity_assay | SOP template | Wet-lab PI review | TBD |

### §3.4 blocked-on-other-todo (1 task — already DONE upstream)

| ID | Task | Blocker | Status |
|---|---|---|---|
| TODO-135 Phase 1 | Rh balanced cohort generation (250 active + 250 inactive) | (Phase 1) | ✅ DONE 2026-09-27 (Wf-139 P7); Phase 2 unblocked but itself blocked-on-gpu |

---

## §4 Top 5 actionable-now ranked by impact

Per `party-twenty-spirit` §3.5 + `TODO/completed/172_comprehensive_lift_plan.md` + paper §5 gate priority:

| Rank | ID | Why this is top | Effort | Output |
|---:|---|---|---|---|
| **1** | **TODO-178 metalcytotox 5-metal CSV audit + FAIL-on-missing** | Closes hard-dependency for 5+ downstream consumers (Wf-123 P3, Wf-124 P3, Wf-145, Wf-176, Wf-187); schema gap fill is small + audit JSON honest-framing per `party-twenty-spirit` §3.5 | 0.5-1 day CPU | `molmetal/.data/molmetal/metalcytotox_5metal_multitask.csv` (schema-enriched) + audit JSON + 12+ tests PASS |
| **2** | **TODO-134 Phase 1 m15 Docking RMSD ref extraction** | Unblocks the largest unstarted code task in IMPROVEMENT_ENUM (Phase 2/3 require GPU but Phase 1 is pure CPU); feeds paper §4.5.7; Vina redocking design already 280-LOC spec'd | 2-3 days CPU | `molmetal/scripts/bench_docked_pose_rmsd.py` (ref extract part) + `molmetal/reports/wf134_p2_redocking/ref_smiles/` (5 pockets) |
| **3** | **TODO-138 Paper v2 + JCIM submission package** | CPU-only docs; weakly depends on TODO-134 (placeholder acceptable); closes the longest-standing P0 in IMPROVEMENT_ENUM (TODO-138 is the meta-gate for JCIM submission) | 5-7 days CPU | `paper/` (clean pdflatex 4 passes) + `cover_letter.tex` + `submission_package.zip` |
| **4** | **TODO-234 frontier math wire-in (CPU portions)** | Audit DONE (2026-09-28); 4 of 7 modules can be wired without GPU (W2.1 SE(3) SB trio + W2.3 multi-PH already shipped as adapters); pure CPU import wire + glue | 3-5 days CPU | `molmetal/adapters/flow_matching_lipman/__init__.py` + `w_flow.py` wire (4 modules); paper §3 cross-refs (3 research-only tagged) |
| **5** | **TODO-178b smoke-test FAIL-on-missing hardening** | Hygiene per `party-twenty-spirit` §3.5 (silent skip ≠ measured); ensures downstream test gates fail loudly when CSV missing | 0.5 day CPU | `molmetal/tests/test_wasserstein_closure_integration.py::test_metalcytotox_csv_integration_smoke` patched to FAIL-on-missing |

---

## §5 Net counts

| Category | n | Top 5 actionable |
|---|---:|---|
| **actionable-now** | **8** | TODO-178 → TODO-134 P1 → TODO-138 → TODO-234 (CPU) → TODO-178b |
| **blocked-on-gpu** | **8** | TODO-219 (smallest) → TODO-180 → TODO-181 → TODO-179 → TODO-182 |
| **blocked-on-external** | **6** | (all wetlab LOI/SOP; 6-month timeline; non-code today) |
| **blocked-on-other-todo** | **1** | (already DONE upstream; effectively zero today) |
| **Total tracked** | **23** | — |

---

## §6 Honest framing per `party-twenty-spirit` §3.5

- **MEASURED today**: 8 actionable-now code tasks identified (all CPU-only or low-contention; do NOT block on user OK).
- **NOT measured**: actual wall-time / lift of any blocked-on-gpu task; we have estimates from each pending/ file §2 "Wall (est)" column.
- **PROJECTED lift**:
  - TODO-178 → unblocks 5+ downstream consumers (Wf-123 P3, Wf-124 P3, Wf-145, Wf-176, Wf-187) — no lift itself, but enables measured lift downstream.
  - TODO-134 Phase 1 → feeds Phase 2/3 (Vina redocking) — Phase 2/3 PROJECTED per-pocket median RMSD ≤ 2.5 Å + ≥ 1 pocket ≤ 2.0 Å (SOTA success).
  - TODO-138 → CPU-only docs; gates paper submission — no metric lift.
  - TODO-234 (CPU) → paper §3 method-reference completeness; no metric lift.
- **Standing "kill then adapt" GPU directive**: explicit fresh user OK required before any of the 8 blocked-on-gpu tasks launch.
- **NOT actioned today**: TODO-219 (Ru + Ir ~10 min, smallest GPU ask) — keep as first GPU pick when user OKs.

---

## §7 Recommended execution order (actionable-now, serial)

Per IMPROVEMENT_ENUM §0 dependency graph + the actionable-now ranking:

1. **Wf-303C**: TODO-178 metalcytotox 5-metal CSV audit + FAIL-on-missing (0.5-1 day CPU; highest downstream-unblock ROI)
2. **Wf-303D**: TODO-134 Phase 1 m15 RMSD ref extraction (2-3 days CPU; unblocks Phase 2/3)
3. **Wf-303E**: TODO-234 frontier math wire-in (CPU portions, 3-5 days; 4 modules + paper §3 cross-refs)
4. **Wf-303F**: TODO-138 Paper v2 + JCIM submission package (5-7 days CPU; gates submission)
5. **Wf-303G**: TODO-178b smoke-test FAIL-on-missing hardening (0.5 day CPU; hygiene)

**Total actionable CPU work**: ~12-17 days serial (with no GPU launches).
**Total blocked-on-gpu work** (pending user OK): ~30 hr wall + 2 weeks (TODO-135 P2).

When user OKs any GPU launch, the recommended first pick per smallest-estimate is **TODO-219 Ru + Ir (~10 min wall)** since it completes the 5-metal × 3-seed × 50-epoch Wf-219 sweep that already shipped Re/Os/Rh.