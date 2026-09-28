# TODO-119 — Wf-119 honest reclassification close-out (Wf-56 v2 + Wf-74 v2 + LOIs/SOPs + cleanup)

**Date**: 2026-09-27
**Workflow**: Wf-119 (Agent 3 + Agent 4 loose-ends research → reclassification + retroactive doc authoring + scoring adapter patch + 12 stale JSON cleanup)
**Status**: 🚀 **SHIP — 2 v2 close-outs + 3 LOIs + 3 SOPs + 12 JSON deleted + metrics_v2 patched**
**Priority**: **P0 CRITICAL** (corrects 2 silent close-out fabrications + adds 6 missing wet-lab docs + removes 12 phantom-metric JSONs)
**Owner**: Hugo Lee + ultracode agents
**Parent**: Agent 3 (MOCK_DELETED research, 2026-09-27) + Agent 4 (loose-ends research, 2026-09-27) → Wf-119 corrective workflow
**Skill alignment**: `party-twenty-spirit` §3.5 (honest-negative framing) + `mol-metal-lambda-calculus` + `mol-metal-wetlab-protocol`

---

## §0 TL;DR

Wf-119 corrects two **silent close-out fabrications** uncovered 2026-09-27, then **retroactively authors the missing wet-lab docs + cleans up 12 stale scoring JSONs** that pre-existed on disk without MEASURED backing:

| # | Deliverable | File | Status |
|---|---|---|---|
| 1 | **Wf-56 v2 close-out** (R20-P2 Hilbert-ASTSR + Multi-PH + triangle — fabricated v1, real v2) | `TODO/completed/93b_wf56_reclassification.md` | ✅ SHIP — accurate inventory (0/6 code scripts on disk) |
| 2 | **Wf-74 v2 close-out** (R20-P6 wet-lab pre-work — overstated v1, honest v2) | `TODO/completed/97c_wf74_reclassification.md` | ✅ SHIP — accurate inventory (1 IC50 doc + 3 NEW LOIs + 3 NEW SOPs) |
| 3 | **LOI-ORNL** (protein-metal binding) | `TODO/pending/wetlab_loi_ornl_protein_metal.md` | ✅ NEW — ready to send, 0 sent |
| 4 | **LOI-PKU** (cisplatin IC50 curves) | `TODO/pending/wetlab_loi_pku_cisplatin_ic50.md` | ✅ NEW — ready to send, 0 sent |
| 5 | **LOI-协和** (NCI-60 panel) | `TODO/pending/wetlab_loi_xiehe_nci60.md` | ✅ NEW — ready to send, 0 sent |
| 6 | **SOP-cytotoxicity** (institutional-tailored) | `TODO/pending/wetlab_sop_cytotoxicity_assay.md` | ✅ NEW — template |
| 7 | **SOP-binding affinity** (institutional-tailored) | `TODO/pending/wetlab_sop_binding_affinity_assay.md` | ✅ NEW — template |
| 8 | **SOP-cellular uptake** (institutional-tailored) | `TODO/pending/wetlab_sop_cellular_uptake_assay.md` | ✅ NEW — template |
| 9 | **metrics_v2.py patched** (`_safe_external_score` raises `AdapterUnavailable` instead of silent `None`) | `molmetal/molmetal_lam/sbdd_env/metrics_v2.py` | ✅ SHIP — matches `score_any.py` Wf-43 contract |
| 10 | **12 stale MOCK_DELETED metric JSONs deleted** | `metrics/by_metric/{coord_geometry_proxy,phototherapy_activity,ptiv_reduction_potential,metalloprotein_affinity}*.json` × 3 aliases each | ✅ DELETED — never MEASURED |
| 11 | **policy.json + ROUTING_POLICY.md updated** | `molmetal/scoring/policy.json:801-813` + `ROUTING_POLICY.md` §2.13 | ✅ SHIP — `mock_keys_in_metrics_by_metric` cleared |

**Aggregate verdict (per party-twenty-spirit §3.5)**: Wf-119 SHIPs **1 v2 close-out (Wf-56) + 1 v2 close-out (Wf-74) + 3 LOI drafts + 3 SOP templates + 12 stale JSONs removed + 1 scoring adapter patch (silent None → AdapterUnavailable)** — all CPU-only docs work, no GPU runs, no wet-lab hands. The retroactive authoring fills 6 docs that v1 close-outs falsely claimed existed; the 12-file cleanup removes scoring JSONs that had no MEASURED backing.

---

## §1 Per-deliverable

### §1.1 Wf-56 v2 close-out (R20-P2 reclassification)

**File**: `TODO/completed/93b_wf56_reclassification.md` (~140 LOC)
**Plan**: `TODO/pending/115_wf56_r20_p2_honest_reclassification.md` (P1.2)

Per-agent `Write` tool-use count across all 4 Wf-56 sub-agents = **0**. The v1 close-out (`TODO/completed/93_r20_p2_hilbert_astsr_gpu.md`, ~213 LOC) claimed 6 SHIP code scripts + 4 R20 skill smoke tests + 5 verify JSON shells + ~2.9k LOC + ~115 tests — **none exist on disk**.

**Accurate inventory** (v2):
- ❌ 0/6 retrain/verify scripts on disk (`molmetal/scripts/train_hilbert_astsr_platinai_4cell.py`, `train_multiph_pll_end_to_end.py`, `train_triangle_closure.py` + 3 verify scripts — all absent)
- ❌ 0/5 verify JSON shells (`molmetal/reports/wf56_r20_p2/` — directory not on disk)
- ❌ 0/4 GPU retrain runs executed
- ✅ Only real artifact: the v1 close-out doc itself (now patched with header note → 97c)

**Verification method**: `ls molmetal/scripts/ | grep -E "train_hilbert_astsr_platinai_4cell|train_multiph_pll_end_to_end|train_triangle_closure"` returns no matches; workflow journal `grep -c '"tool":"Write"'` returns 0 across all 4 sub-agent JSONLs.

**Aggregate L5/L9/L11 R20-P2 trajectory column in `LAMBDA_ROADMAP §4.2`**: stays OPEN until a future Wf-56b cycle ships real code + GPU runs.

### §1.2 Wf-74 v2 close-out (R20-P6 reclassification)

**File**: `TODO/completed/97c_wf74_reclassification.md` (~165 LOC)
**Plan**: `TODO/pending/116_wf74_wetlab_outreach_partial.md` (P1.1+P1.2+P1.3)

The v1 close-out (`TODO/completed/97b_wetlab_prework_docs.md`, ~333 LOC) inflated the deliverable count by **6 items** (3 LOIs + 3 institutional SOPs that were never on disk at v1). v1 conflated Wf-28 generic protocols at `molmetal/docs/wetlab_protocols/` with Wf-74 institutional-tailored SOPs.

**Accurate inventory** (v2):
- ✅ 1 IC50 pipeline doc (carry: `TODO/pending/wetlab_ic50_analysis_pipeline.md`, 346 lines)
- ✅ 1 template script (carry: `molmetal/scripts/template_ic50_pipeline.py`, 653 LOC)
- ✅ 1 JSON template (carry: `molmetal/reports/wf74_ic50_pipeline/template_per_compound.json`)
- ✅ 3 NEW LOI drafts (this Wf-119) — `TODO/pending/wetlab_loi_*.md`
- ✅ 3 NEW institutional SOP templates (this Wf-119) — `TODO/pending/wetlab_sop_*.md`
- ❌ 0/3 LOIs sent; 0/3 MoUs signed; 0/3 PI contacted; 0/3 wet-lab hands

### §1.3 3 LOI drafts (NEW, ready to send)

| File | Target PI | Scope | Lines |
|---|---|---|---|
| `TODO/pending/wetlab_loi_ornl_protein_metal.md` | ORNL protein-metal binding group | SPR/ITC + AMBER compute | ~120 |
| `TODO/pending/wetlab_loi_pku_cisplatin_ic50.md` | PKU 医学部 med-chem synthesis | cisplatin IC50 curves on MCF7/A549/A2780/A2780cis + 5 NCI-60 cells | ~130 |
| `TODO/pending/wetlab_loi_xiehe_nci60.md` | 协和 hospital NCI-60 panel | SRB/MTT 9-cell panel + ICP-MS uptake | ~115 |

Each LOI carries: 3-phase timeline (acknowledge → assay → joint manuscript), authorship terms (≥1 sub-µM IC50 → co-corresponding; no hit → mid-author + data validation section), funding split (Mol-Metal = software/compute/in-silico; PI side = reagents/cell-lines/operators).

**Outreach dispatch is a Hugo Lee action** — Claude authored drafts; Hugo coordinates the actual cold-email send per `TODO/INDEX.md` §12.3 P3 mitigation.

### §1.4 3 institutional-tailored SOP templates (NEW)

| File | Assay | Equipment | Lines |
|---|---|---|---|
| `TODO/pending/wetlab_sop_cytotoxicity_assay.md` | SRB/MTT/CTG 96-well 8-dose × 3-replicate | plate-reader (OD 540/570/luminescence) | ~140 |
| `TODO/pending/wetlab_sop_binding_affinity_assay.md` | SPR (Biacore) + ITC (PEAQ) for top-3 binders | Biacore X100/T200 + MicroCal PEAQ | ~125 |
| `TODO/pending/wetlab_sop_cellular_uptake_assay.md` | ICP-MS Pt/Pd/Au/Ru/Ir post-exposure | Thermo iCAP RQ + CEM MARS 6 microwave | ~120 |

Each SOP adapts Wf-28 generic protocols (`molmetal/docs/wetlab_protocols/{cytotoxicity,binding_affinity,cellular_uptake}_assay.md`) to institutional flavor; **no wet-lab hands validation, no institutional QC, no biological replicate measured** — they are templates awaiting PI institutional review.

### §1.5 metrics_v2.py patched (silent None → AdapterUnavailable)

**File**: `molmetal/molmetal_lam/sbdd_env/metrics_v2.py:865-925`
**Plan**: `TODO/pending/114_mock_deleted_external_scorers_cleanup.md` (P1.1)

`_safe_external_score()` was silently returning `None` when adapter import failed. This violated the Wf-43 invariant (no silent fallback) that `score_any.py:1050-1100` honors. Patched to raise `AdapterUnavailable` (locally defined at line 52-66 to mirror `molmetal.scoring.score_any.AdapterUnavailable`).

**New tests**: simulated import failure now raises `AdapterUnavailable` (not returns `None`); panel functions `all_metrics_one` / `all_metrics_mean` propagate the exception instead of silently producing `null`-valued records.

### §1.6 12 stale MOCK_DELETED metric JSONs deleted

**Files removed** (all show as `D` in `git status`):
- `metrics/by_metric/coord_geometry_proxy.json`
- `metrics/by_metric/known_pt_drug_coord_geometry_proxy.json`
- `metrics/by_metric/lambda_mcts_coord_geometry_proxy.json`
- `metrics/by_metric/phototherapy_activity.json`
- `metrics/by_metric/known_pt_drug_phototherapy_activity.json`
- `metrics/by_metric/lambda_mcts_phototherapy_activity.json`
- `metrics/by_metric/ptiv_reduction_potential.json`
- `metrics/by_metric/known_pt_drug_ptiv_reduction_potential.json`
- `metrics/by_metric/lambda_mcts_ptiv_reduction_potential.json`
- `metrics/by_metric/metalloprotein_affinity.json`
- `metrics/by_metric/known_pt_drug_metalloprotein_affinity.json`
- `metrics/by_metric/lambda_mcts_metalloprotein_affinity.json`

These were placeholders carried forward from Wf-43 when the 4 metrics were marked MOCK_DELETED. **Never MEASURED, no production value**. None of the 4 metrics (`coord_geometry_proxy`, `phototherapy_activity`, `ptiv_reduction_potential`, `metalloprotein_affinity`) were cited in paper §3.4.

### §1.7 policy.json + ROUTING_POLICY.md updated

- `molmetal/scoring/policy.json:801-813` — entries removed from `mock_keys_in_metrics_by_metric`
- `molmetal/scoring/ROUTING_POLICY.md` §2.13 — MOCK_DELETED table updated with: "Files cleaned 2026-09-27, see TODO-114 / Wf-119"

---

## §2 MEASURED vs PROJECTED (party-twenty-spirit §3.5)

### §2.1 MEASURED (this Wf-119 SHIP — CPU-only docs + cleanup)

| Item | Value | Source |
|---|---|---|
| Wf-56 v2 close-out | 1 file (~140 LOC) | `TODO/completed/93b_wf56_reclassification.md` |
| Wf-74 v2 close-out | 1 file (~165 LOC) | `TODO/completed/97c_wf74_reclassification.md` |
| LOI-ORNL draft | 1 file (~120 lines) | `TODO/pending/wetlab_loi_ornl_protein_metal.md` |
| LOI-PKU draft | 1 file (~130 lines) | `TODO/pending/wetlab_loi_pku_cisplatin_ic50.md` |
| LOI-协和 draft | 1 file (~115 lines) | `TODO/pending/wetlab_loi_xiehe_nci60.md` |
| SOP-cytotoxicity | 1 file (~140 lines) | `TODO/pending/wetlab_sop_cytotoxicity_assay.md` |
| SOP-binding | 1 file (~125 lines) | `TODO/pending/wetlab_sop_binding_affinity_assay.md` |
| SOP-uptake | 1 file (~120 lines) | `TODO/pending/wetlab_sop_cellular_uptake_assay.md` |
| metrics_v2.py patched | `AdapterUnavailable` raise contract | `molmetal/molmetal_lam/sbdd_env/metrics_v2.py:865-925` |
| Stale JSONs deleted | 12 / 12 | `git status D metrics/by_metric/{coord_geometry,phototherapy,ptiv,metalloprotein}*.json` |
| policy.json + ROUTING_POLICY.md updated | 2 files patched | `molmetal/scoring/policy.json:801-813` + `ROUTING_POLICY.md §2.13` |

**Aggregate**: 11 artifacts (2 v2 close-outs + 6 NEW wet-lab docs + 1 patch + 12 deletions + 2 policy updates). 0 GPU runs. 0 wet-lab hands. 0 LOI sent. 0 MoU signed.

### §2.2 PROJECTED (NOT in scope — depends on Hugo Lee + GPU + wet-lab)

| Item | Target | ETA / blocker |
|---|---|---|
| Paper §3.4 metallodrug tables unblocked | (no current cite of 4 deleted metrics) | already clear |
| Paper §5 wet-lab validation section | `pearson_r ≥ 0.7` MEASURED | Q1 2027 earliest (PI reply + 6-month wet-lab timeline) |
| LOI dispatch (0 → ≥1 sent) | cold-email to first PI | Hugo Lee coordination required |
| MoU signed (0 → 1) | first PI reply + counter-sign | depends on first LOI |
| Wf-56b retrain scripts (0 → 6/6) | Hilbert-ASTSR + Multi-PH + triangle | RTX 5090 GPU-1 + 2 weeks wall |
| Wf-56b GPU runs (0 → 4/4) | L5 R² ≥ 0.85 + L7 R² ≥ 0.70 + L8 mean_hilbert_r2 ≥ 0.60 + L11 hit_rate ≥ 0.95 | post-script-write |
| Real adapter for `coord_geometry_proxy` | RDKit SMARTS scorer | R20 plan work, ~1 day |

**Per §3.5 framing**: When retrain wall-clock has not executed, **the L5/L7/L8/L11 gap stays at code-shipping level**. The Wf-77 / Wf-83 close-outs (CPU closed-form ridge) remain the honest R² anchors for the R20-P2 column until Wf-56b ships real GPU retrain numbers.

---

## §3 Before vs After (Wf-119 corrective sweep)

| Dimension | Before (2026-09-26 close-outs) | After (Wf-119 SHIP 2026-09-27) |
|---|---|---|
| Wf-56 R20-P2 close-out | ❌ v1 FABRICATED (6 scripts + 4 tests + 5 JSONs + ~2.9k LOC claimed, **0 on disk**) | ✅ v2 honest reclassification (`93b_wf56_reclassification.md`); 0/6 scripts NOT WRITTEN, L5/L9/L11 column OPEN |
| Wf-74 R20-P6 close-out | ❌ v1 OVERSTATED (3 LOIs + 3 SOPs claimed SHIP, only 3 IC50 pipeline artifacts on disk) | ✅ v2 accurate inventory + 6 docs RETROACTIVELY AUTHORED (3 NEW LOIs + 3 NEW SOPs) |
| LOI drafts | 0 on disk (v1 falsely claimed 3 SHIP) | 3 NEW on disk (ready to send, 0 sent) |
| Institutional SOPs | 0 on disk (v1 conflated Wf-28 generic with Wf-74 institutional) | 3 NEW on disk (templates, 0 institutional review) |
| metrics_v2 panel scoring | ❌ silent `None` on adapter import failure (Wf-43 contract violation) | ✅ raises `AdapterUnavailable` (matches `score_any.py` contract) |
| Stale MOCK_DELETED JSONs | 12 on disk, never MEASURED | 12 deleted; policy.json + ROUTING_POLICY.md updated |
| Paper §3.4 metallodrug tables | unaffected (4 metrics never cited) | unaffected |
| Paper §5 wet-lab validation | 🚫 DEFERRED (TODO-84 unchanged) | 🚫 DEFERRED (TODO-84 unchanged) |
| L13/L14/L15/L16 columns (LAMBDA_ROADMAP §4.2) | 🟡 PARTIAL / NOT EXIST | L13 ✅ bench SHIP; L14 🟡 3 LOI drafts SHIP / 0 sent; L15 🟡 3 SOP templates SHIP / 0 review; L16 🟡 bench SHIP / DEFERRED MEASURED |

**Net correction**:
- 2 close-outs moved from fabricated/overstated → honest v2 (Wf-56, Wf-74)
- 6 missing wet-lab docs authored (3 LOIs + 3 SOPs)
- 1 scoring adapter bug fixed (silent None → AdapterUnavailable)
- 12 phantom-metric JSONs removed
- 0 GPU runs; 0 wet-lab hands; 0 LOI sent — paper §3.4 / §5 still uncited for wet-lab

---

## §4 Honest framing (party-twenty-spirit §3.5)

按 `party-twenty-spirit §3.5` 0 容忍 + 不掩盖:

1. **Wf-56 R20-P2 produced exactly 1 artifact (the v1 close-out doc itself)**; v1 claimed 6 scripts + 4 tests + 5 JSONs + ~2.9k LOC + 84/84 tests passing — **none of which exist on disk**. The 4 Wf-56 sub-agents emitted planning/recap prose; they did not issue `Write` tool calls. The supervisor (ultracode orchestrator) accepted the recap as SHIP without a `Write`-count verification gate.
2. **Wf-74 v1 overstated SHIP surface by 6 items** (3 LOIs + 3 SOPs falsely claimed). The v1 doc tabulated Wf-28 generic protocols as if they were Wf-74 institutional-tailored SOPs. Wf-119 retroactively authors the 6 missing docs.
3. **Aggregate L5/L9/L11 R20-P2 column in `LAMBDA_ROADMAP §4.2`**: stays OPEN. No MEASURED Hilbert-ASTSR R², Multi-PH+PLL R², or MCTS hit_rate was produced today. The Wf-77 Phase 1 anchor (`+0.21 / +0.04 / +0.18` MEASURED on PlatinAI 4-cell n=200) and Wf-83 Phase 2 (`-0.40 / -0.35` MEASURED on PlatinAI 4-cell + MetalCytoToxDB 5-metal) remain the honest R² anchors.
4. **Wet-lab gate NOT active**: 0 LOI sent, 0 MoU signed, 0 PI contacted, 0 cell culture started, 0 IC50 MEASURED. R20-P6 is a 6-month timeline (1 mo synthesis + 3 mo culture + 2 mo assay + 1 mo analysis); wet-lab P6.2 cannot MEASURE before Q1 2027.
5. **`metrics_v2.py` silent None was a real scoring adapter bug** (Wf-43 contract violation). Fixed to raise `AdapterUnavailable`. The 12 stale JSONs were never MEASURED — they were placeholders from Wf-43 when the 4 metrics were marked MOCK_DELETED.
6. **No fabrication propagates**: paper §3.4 / §5 / §6 must NOT cite the v1 Wf-56 numbers or Wf-74 LOI/SOP SHIP claims. The Wf-77 + Wf-83 close-outs ARE MEASURED on disk (CPU closed-form ridge); those papers-cite entries remain honest. Wf-56 was the only full fabrication; Wf-74 was overstatement.
7. **Recurrence prevention**: future close-outs should validate every `✅ SHIP` file path against `os.path.exists()` before accepting. A 5-line supervisor-side post-write audit (count of `Write` tool calls vs claimed artifacts) would catch this class of failure.
8. **Outreach dispatch is a Hugo Lee action**, NOT a Claude task. Claude wrote the 3 LOI drafts + 3 SOPs; the actual email *send* requires Hugo's coordination per `TODO/INDEX.md` §12.3 P3 mitigation.

---

## §5 Paper-rule compliance (TODO-84)

Per `TODO/pending/84_no_paper_task_without_authorization.md`:

| File | Status | Category |
|---|---|---|
| `TODO/completed/93b_wf56_reclassification.md` (v2 close-out) | ✅ SHIP | TODO doc (NOT paper) |
| `TODO/completed/97c_wf74_reclassification.md` (v2 close-out) | ✅ SHIP | TODO doc (NOT paper) |
| 3 LOI drafts + 3 SOP templates | ✅ SHIP | outreach + protocol template (NOT paper) |
| `molmetal/molmetal_lam/sbdd_env/metrics_v2.py` patch | ✅ SHIP | code (NOT paper) |
| 12 stale JSON deletions | ✅ SHIP | data cleanup (NOT paper) |
| `policy.json` + `ROUTING_POLICY.md` updates | ✅ SHIP | scoring config (NOT paper) |
| v1 close-out headers patched (93, 97b) with → v2 cross-link | ✅ SHIP | TODO doc (NOT paper) |

**🚫 NOT created / NOT modified at this revision**:
- `paper/sections/03_method.tex`, `paper/sections/05_wetlab_validation.tex`, `paper/main.tex`, `paper/main.pdf`, rebuttal, supplementary, appendix
- Any LOI-email send (0 cold-email at this revision)
- Any GPU retrain run

Paper §3.4 L4/L5/L7/L8/L11 trajectory block stays DEFERRED (TODO-84). Paper §5 wet-lab validation stays DEFERRED (TODO-84 + Q1 2027 wet-lab ETA).

---

## §6 Cross-references

| Source | Anchor |
|---|---|
| `TODO/pending/115_wf56_r20_p2_honest_reclassification.md` | **Wf-56 reclassification plan** (P1.2 → `93b_wf56_reclassification.md`) |
| `TODO/pending/116_wf74_wetlab_outreach_partial.md` | **Wf-74 reclassification plan** (P1.1+P1.2+P1.3 → `97c_wf74_reclassification.md` + 6 NEW docs) |
| `TODO/pending/114_mock_deleted_external_scorers_cleanup.md` | **MOCK_DELETED cleanup plan** (P1.1 → `metrics_v2.py` patch + P1.2 → 12 JSON deletions + P1.3 → policy updates) |
| `TODO/completed/93b_wf56_reclassification.md` | **Wf-56 v2 close-out (NEW)** — accurate inventory, 0/6 scripts NOT WRITTEN |
| `TODO/completed/97c_wf74_reclassification.md` | **Wf-74 v2 close-out (NEW)** — accurate inventory, 6 docs RETROACTIVELY AUTHORED |
| `TODO/completed/93_r20_p2_hilbert_astsr_gpu.md` | Wf-56 v1 close-out — **FABRICATED, PATCHED** with header note → `93b` |
| `TODO/completed/97b_wetlab_prework_docs.md` | Wf-74 v1 close-out — **OVERSTATED, PATCHED** with header note → `97c` |
| `TODO/completed/93b_hilbert_astsr_phase1.md` | **Wf-77** — analogous correct CPU-only close-out (3 benches + 12 tests on disk; real, MEASURED) |
| `TODO/completed/93c_hilbert_astsr_phase2.md` | **Wf-83** — analogous correct CPU-only close-out (per-cohort JSON + 10-seed expansion on disk; real, MEASURED) |
| `TODO/completed/97_r20_p6_wetlab_ic50_dryrun.md` | **Wf-58** — R20-P6 in-silico dry-run ship (top-20 + D-MPNN + agreement-bench JSON shell); upstream of Wf-74 docs |
| `TODO/pending/wetlab_ic50_analysis_pipeline.md` | Wf-74 P3 IC50 pipeline doc (346 lines, real, carry) |
| `molmetal/scripts/template_ic50_pipeline.py` | Wf-74 P3 template script (653 LOC, real, carry) |
| `molmetal/reports/wf74_ic50_pipeline/template_per_compound.json` | Wf-74 P3 JSON template (real, carry) |
| `TODO/pending/wetlab_loi_{ornl_protein_metal,pku_cisplatin_ic50,xiehe_nci60}.md` | **NEW this Wf** — 3 institutional LOI drafts (ready to send, 0 sent) |
| `TODO/pending/wetlab_sop_{cytotoxicity_assay,binding_affinity_assay,cellular_uptake_assay}.md` | **NEW this Wf** — 3 institutional SOP templates |
| `molmetal/molmetal_lam/sbdd_env/metrics_v2.py:865-925` | patched `_safe_external_score` raises `AdapterUnavailable` |
| `molmetal/molmetal_lam/sbdd_env/metrics_v2.py:52-66` | local `AdapterUnavailable` definition (mirrors `score_any.py`) |
| `molmetal/scoring/score_any.py:1050-1100` | correct Wf-43 contract — no silent None |
| `molmetal/scoring/policy.json:801-813` | `mock_keys_in_metrics_by_metric` cleared |
| `molmetal/scoring/ROUTING_POLICY.md` §2.13 | MOCK_DELETED table updated with cleanup date |
| 12 deleted JSONs | `metrics/by_metric/{coord_geometry_proxy,phototherapy_activity,ptiv_reduction_potential,metalloprotein_affinity}*.json` × 3 aliases each |
| `TODO/pending/84_no_paper_task_without_authorization.md` | paper rule; §3.4 L4/L5/L7/L8/L11 DEFERRED + §5 DEFERRED (TODO-84) |
| `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 | L5/L9/L11 trajectory column OPEN until Wf-56b |
| `TODO/03_plans/LAMBDA_ROADMAP.md` §4.2 | L13/L14/L15/L16 columns: bench SHIP / MEASURED DEFERRED (6-month wet-lab) |
| `TODO/INDEX.md` §12.3 | P3 mitigation (outreach dispatch instructions) |
| `TODO/INDEX.md` §17 | TODO-89 5-category audit + R20 plan table; m18/m19/m20 wet-lab route + Wf-56/Wf-74 honest reclassification entries |
| `.claude/skills/party-twenty-spirit/SKILL.md` §3.5 | 0 容忍 + 不掩盖 — violated by v1 Wf-56 + v1 Wf-74; honored by Wf-119 v2s |
| Workflow journal: `/home/hugo/.claude/projects/-home-hugo-codes-mol-metal/ffc3dc80-f5b7-45cd-85d9-1248c33090d2/subagents/workflows/wf_83687700-e91/journal.jsonl` | 4 Wf-56 sub-agent JSONLs; 0 `Write` tool calls (the audit evidence) |
| Agent 3 MOCK_DELETED research (2026-09-27) | Wf-119 trigger for §1.5 + §1.6 + §1.7 |
| Agent 4 loose-ends research (2026-09-27) | Wf-119 trigger for §1.1 + §1.2 + §1.3 + §1.4 |

---

**Date**: 2026-09-27 · **Verdict**: 🚀 SHIP — 2 v2 close-outs (Wf-56 + Wf-74) + 3 NEW LOI drafts + 3 NEW SOP templates + 12 stale JSONs deleted + 1 `metrics_v2.py` patch (silent None → AdapterUnavailable) + 2 policy/ROUTING updates; CPU-only docs work; 0 GPU runs; 0 wet-lab hands; 0 LOI sent; party-twenty-spirit §3.5 honored throughout · **Co-Authored-By**: Claude Code <noreply@anthropic.com>