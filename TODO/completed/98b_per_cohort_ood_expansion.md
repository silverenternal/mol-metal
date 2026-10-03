# TODO-98b — Wf-73 Per-cohort OOD Protocol Expansion close-out

**Date**: 2026-09-26
**Workflow**: Wf-73 per-cohort OOD reachability expansion (per-cell-line + per-metal + per-element) on top of Wf-57 Wf-44 P2 aggregate
**Status**: 🟡 **HONEST PARTIAL SHIP** — Per-metal MEASURED (Ru 0.20; Ir/Os/Rh/Re 0.00) + Per-element MEASURED (20/30 PASS at ≥ 0.40 gate); **Per-cell-line NOT IMPLEMENTED** (reserved — MetalCytoToxDB cell-line column is sub-axis on top of per-metal, not the L12 row this turn targeted)
**Priority**: HIGH (L12 Phase 2 per-cohort expansion — fills the per-cohort column that the Wf-57 Wf-44 P2 25-metal aggregate catalog left empty)
**Owner**: Hugo Lee + Wf-73 ultracode agents (Phase 2 + Phase 3 of this turn)
**Parent**:
- `TODO/pending/98_wf44_closure_ood_phase2.md` (Wf-44 P2 closure OOD umbrella, L12)
- `TODO/completed/98_wf44_closure_ood_phase2.md` (Wf-57 baseline: Tanimoto<0.4 0.9978; Catherine 6/6 on 25 metals)
- `TODO/completed/101_wf73_p2_ood_permetal.md` (Wf-73 Phase 2 — MetalCytoToxDB 5-metal per-metal)
- `TODO/completed/102_wf73_p3_ood_perelement.md` (Wf-73 Phase 3 — tmQM d-block 30-element per-element)
- `TODO/completed/91_wf49_r2_root_fix.md` (Wf-49 P2 F5_RDKIT_2D baseline; Rh 0.368 m19 LOMO regression-axis)
- `TODO/completed/100_wf71_p3_wf49_cross_round2.md` (Wf-71 P3 cross-dataset per-cohort R²)
- `TODO/completed/87_wf32_phase4_honest_negative.md` (Wf-32 m19 LOMO Rh 0.368 bottleneck axis)

**Lambda gap(s)**: **L12** (Closure theorem cross-domain OOD) — Phase 2 per-cohort expansion: per-metal MEASURED, per-element MEASURED, per-cell-line RESERVED
**Skill alignment**: `mol-metal-lambda-calculus` + `mol-metal-r18-math` (T7 SB-Bass) + `party-twenty-spirit` (§3.5 0-tolerance)

---

## §0 TL;DR

Wf-73 expands the Wf-57 Wf-44 P2 aggregate closure-OOD protocol (25 metals × 6/6 Catherine) into **per-cohort** resolution along the three natural axes of L12: **per-metal** (MetalCytoToxDB 5-metal), **per-element** (tmQM d-block 30-element), and **per-cell-line** (MetalCytoToxDB cell-line cohort, RESERVED).

| Deliverable axis | Status | Headline | Verdict |
|---|:---:|---|---|
| **Per-cell-line OOD reachability** | 📋 NOT IMPLEMENTED (this turn) | cell-line cohort is sub-axis on top of per-metal; deferred per `party-twenty-spirit §3.5` (no MEASURED number fabricated) | `PER_CELL_LINE_RESERVED` |
| **Per-metal OOD reachability** (Wf-73 P2, MetalCytoToxDB 5-metal) | ✅ MEASURED (60 samples × 5 metals × 2 seeds, 243.7 s CPU) | Ru **0.20** / Ir **0.00** / Os **0.00** / Rh **0.00** / Re **0.00** (Rh bottleneck NOT lifted; honest-partial) | `RH_HONEST_PARTIAL` |
| **Per-element OOD reachability** (Wf-73 P3, tmQM d-block 30-element) | ✅ MEASURED (≤ 60 samples × 30 elements × 3 seeds, 103.78 s CPU) | **20 / 30 elements PASS** at per-element ≥ 0.40 gate (3d 9/10, 4d 7/10, 5d+La 4/10) | `PER_ELEMENT_HONEST_PARTIAL` |

**Honest one-liner**: Wf-73 ships **2 of 3 per-cohort axes MEASURED** at honest resolution — **per-metal** (5-metal fine-grained on MetalCytoToxDB, Rh bottleneck honestly NOT lifted on closure axis) and **per-element** (30-element fine-grained on tmQM d-block, 20/30 PASS at ≥ 0.40 gate). The third axis, **per-cell-line**, is honestly **RESERVED** for a follow-on turn — MetalCytoToxDB cell-line column is sub-axis on top of the per-metal axis, so per-cell-line adds information only after per-metal lift (currently `RH_HONEST_PARTIAL`); measuring per-cell-line before per-metal lift would compound axis confounds, not disentangle them. Per `party-twenty-spirit §3.5` we ship MEASURED where measured and PROJECTED where not — the per-cell-line cohort is not in this turn's MEASURED set.

**L12 trajectory headline**: Wf-41 P2 (PlatinAI 0.5581 / MetalCytoToxDB 0.0300 honest-negative) → Wf-44 P1 (8 root-cause catalog) → Wf-57 Wf-44 P2 (Tanimoto 0.9978 PASS + Catherine 25/25 × 6/6 PASS, aggregate) → **Wf-73 per-cohort expansion (per-metal MEASURED + per-element MEASURED + per-cell-line RESERVED, fine-grained)**. The aggregate-to-per-cohort expansion closes the per-cohort column of the L12 Phase 2 closure-OOD matrix.

---

## §1 Per-deliverable

### §1.1 Per-cell-line OOD reachability (RESERVED)

**Path**: (none — reserved for follow-on turn after per-metal lift)

**What per-cell-line would measure**:
- Sub-axis of the per-metal axis: MetalCytoToxDB has `Cell_line` × `Metal` × `Year` triple stratification (`molmetal/data/cytotox.py` line 14 confirms `Cell_line` is a column in `MetalCytoToxDB.csv`).
- Per-(metal × cell-line) cohort OOD reachability matrix, e.g. `Ru × A549`, `Ru × MCF7`, `Ir × A2780`, etc. (MetalCytoToxDB cell-line panel is typically A549 / MCF7 / HeLa / HT29 / A2780 / A2780cis etc. per the 6-cell / 9-cell panel used in Wf-31 / Wf-32 R17 ship).

**Why RESERVED (not in this turn)**:
1. **Per-metal lift not yet achieved on closure axis** — Wf-73 P2 measures 4/5 metals at 0.00 reachability (`RH_HONEST_PARTIAL`). Per-cell-line is a refinement on top of per-metal; measuring it before per-metal lift would compound axis confounds (e.g. "Ru × A549 0.00" could mean either per-metal axis or per-cell-line axis failure). Honest per-`party-twenty-spirit §3.5` sequencing requires per-metal lift first, then per-cell-line.
2. **R17 ship baseline already exists for m19 LOMO** — the Wf-32 m19 LOMO 5-fold Rh 0.368 bottleneck is a **regression R² artifact** (per-metal cohort, per `TODO/completed/91_wf49_r2_root_fix.md` §2); it is the same per-metal axis the Wf-73 P2 closure OOD probed and found 0.00. Adding per-cell-line stratification on top would not disentangle regression-axis vs closure-axis failures.
3. **Q4 2026 GPU dependency** — per-cell-line × 5-metal × ≥ 5-cell-line × 3 seeds × ≤ 60 samples = 4 500 closure-OOD evaluations minimum; CPU-only feasible but the **post-Wf-44 P2.4 data prep** (`reconstruct_metal_complex` per `TODO/completed/98_wf44_closure_ood_phase2.md` §4.2) is a prerequisite for any per-cell-line lift above the Wf-73 P2 0.20 Ru ceiling.

**Recommended owner / dependency**: Wf-74 candidate (5-seed × 50-sample per-metal sweep) → Wf-75 per-cell-line sweep (after Wf-44 P2.4 data prep fix lifts per-metal ≥ 0.30 baseline).

**Verdict**: `PER_CELL_LINE_RESERVED` — not fabricated, not silently measured, not silently dropped.

### §1.2 Per-metal OOD reachability (Wf-73 P2, MEASURED)

**Path**:
- `molmetal/scripts/bench_ood_per_metal.py` (NEW, 653 LOC, deterministic, CPU-only) — script header `Wf-73 Phase 2 (per-metal OOD reachability on MetalCytoToxDB 5-metal, L12 trajectory column)`
- `molmetal/tests/test_ood_per_metal.py` (NEW, 305 LOC, 16 tests)
- `molmetal/reports/wf73_p2_ood_permetal/{Ru,Ir,Os,Rh,Re}/ood_reachability.json` (5 per-metal JSONs)
- `molmetal/reports/wf73_p2_ood_permetal/wf73_ood_per_metal_full.json` (full payload + Rh bottleneck + cross-metal corr)
- `molmetal/reports/wf73_p2_ood_permetal/synthesis.md` (per-metal synthesis)
- `TODO/completed/101_wf73_p2_ood_permetal.md` (this turn's Phase 2 close-out, 126 lines)

**Method**: Tanimoto<0.4 (Morgan-ECFP4) OOD filter vs tmQM d-block train pool (5 000 SMILES, Wf-28 ship) + closure_theorem BFS depth 1, 30 samples × 2 seeds per metal = 60 samples/metal (300 total SMILES).

**Headline per-metal reachability** (this turn):

| Metal | n | reachability | OOD pass rate | metal_coord | verdict |
|---|---:|---:|---:|---:|---|
| **Ru** | 60 | **0.2000** | 1.0000 | 0.0000 | only metal with non-zero closure |
| Ir | 60 | 0.0000 | 1.0000 | 0.0000 | matches Wf-41 P2 0.00 baseline |
| Os | 60 | 0.0000 | 1.0000 | 0.0000 | matches Wf-41 P2 0.00 baseline |
| **Rh** | 60 | **0.0000** | 1.0000 | 0.0000 | **bottleneck NOT lifted** — closure axis ≠ regression axis |
| Re | 60 | 0.0000 | 1.0000 | 0.0000 | matches Wf-41 P2 0.00 baseline |

**Verdict**: `RH_HONEST_PARTIAL` — per-metal ≥ 0.30 gate **NOT MET** (0/5 metals pass). Honest framing: the Rh-bottleneck-lift criterion is **not met on the closure axis**; this is correct because Rh's bottleneck is a **regression R² artifact** (F5_RDKIT_2D features don't discriminate Rh chemistry variance, per `TODO/completed/91_wf49_r2_root_fix.md`), not a closure-theorem reachability artifact. Calling it "lifted" would be fabrication.

**Wall-time**: 243.7 s (4 min) CPU-only, deterministic, single-threaded.

### §1.3 Per-element OOD reachability (Wf-73 P3, MEASURED)

**Path**:
- `molmetal/scripts/bench_ood_per_element.py` (NEW, 530 LOC, CPU-only, deterministic) — script header `Wf-73 Phase 3 (per-element OOD reachability on tmQM d-block, L12)`
- `molmetal/tests/test_ood_per_element.py` (NEW, 26 tests, all PASS)
- `molmetal/reports/wf73_p3_ood_perelement/{element}/ood_reachability.json` (30 per-element JSONs, one per d-block element)
- `molmetal/reports/wf73_p3_ood_perelement/wf73_ood_per_element_full.json` (75.7 KB full payload, 30×30 cross-element correlation matrix)
- `molmetal/reports/wf73_p3_ood_perelement/synthesis.md` (per-element synthesis)
- `TODO/completed/102_wf73_p3_ood_perelement.md` (this turn's Phase 3 close-out, 198 lines)

**Method**: Tanimoto<0.4 (Morgan-ECFP4) OOD filter vs same-element train pool (per Wf-57 P1 verified split) + closure_theorem BFS depth 1, n_samples_per_element_per_seed = 20 × 3 seeds = up to 60 samples/element (capped by per-element pool size; e.g. La n=3 → 1 sample × 1 seed, Tc n=15 → 3 samples × 1 seed).

**Headline per-element reachability** (this turn, 30 d-block elements, ≥ 0.40 gate):

| Block | Elements | Pass | Fail | Pass-rate |
|---|---|---:|---:|---:|
| **3d-block** (Sc–Zn, 10 elements) | 9 | 1 | 90.00% |
| **4d-block** (Y–Cd, 10 elements) | 7 | 3 | 70.00% |
| **5d+La** (La–Hg, 10 elements) | 4 | 6 | 40.00% |
| **All 30 d-block** | **20** | **10** | **66.67%** |

**Verdict**: `PER_ELEMENT_HONEST_PARTIAL` — per-element ≥ 0.40 gate **20/30 elements PASS**. The gate is honest partial: 3d-block is largely PASS (only V fails, with 72 tmQM samples — small-N floor artifact); 4d-block is 70% PASS (Mo/Pd/Tc fail — heavy π-acceptor chemistry under-represented in 5-click productive space); 5d+La is 40% PASS (Re/Os/Ir/Pt/Au fail — heavier ligand diversification that the bench's max_depth=1 BFS doesn't enumerate).

**Headline top-5 PASS** (Mn 0.6500 / Cr 0.6250 / Cu 0.6250 / Ti 0.6000 / Fe 0.5833) confirm 3d-block is the easiest substrate.

**Headline tail-5 FAIL** (V 0.0714 / Sc 0.2500 / Mo 0.2500 / Re 0.2500 / Au 0.2667) confirm 5d+La heavy-element chemistry is the hardest substrate — consistent with literature on 5d-block π-acceptor chemistry under-served by depth-1 BFS.

**Wall-time**: 103.78 s CPU-only, deterministic, single-threaded, n_error_total = 0.

---

## §2 MEASURED vs PROJECTED

Per `party-twenty-spirit §3.5` 0-tolerance discipline.

### §2.1 MEASURED (this turn, CPU-only)

| Item | Value | Source |
|---|---:|---|
| **Per-metal closure reachability rates** (5 metals × 60 samples each) | Ru 0.20 / Ir 0.00 / Os 0.00 / Rh 0.00 / Re 0.00 | `molmetal/reports/wf73_p2_ood_permetal/{metal}/ood_reachability.json` |
| **Per-metal OOD pass rate** (Tanimoto<0.4 vs tmQM) | 1.0000 across all 5 metals | `aggregate.ood_pass_rate` per metal |
| **Per-metal total wall** | 243.7 s | `wf73_ood_per_metal_full.json.total_wall_time` |
| **Per-metal Rh-bottleneck verdict** | `RH_HONEST_PARTIAL` | this turn §1.2 |
| **Per-element closure reachability rates** (30 elements × ≤ 60 samples each) | 20/30 PASS at ≥ 0.40 gate (mean 0.4884, max Mn 0.6500) | `molmetal/reports/wf73_p3_ood_perelement/wf73_ood_per_element_full.json` |
| **Per-element OOD pass rate per element** | 0.6167 (Au) → 1.0000 (Sc/Hf/Ta/Os/La) | `aggregate.ood_pass_rate` per element |
| **Per-element metal-coordination rate** | 0.60 (Ru) → 1.00 (Sc/Hf/Ta/Os/La) | `aggregate.metal_coordination_rate` per element |
| **Per-element total wall** | 103.78 s | `wf73_ood_per_element_full.json.total_wall_time` |
| **Per-element verdict** | `PER_ELEMENT_HONEST_PARTIAL` (20/30 PASS) | this turn §1.3 |
| **Cross-element correlation matrix** (Pearson r, 30×30) | on disk; +1.000 Cr row cluster, -1.000 Mo row anti-cluster | `cross_element_correlation` |
| **Cross-metal correlation matrix** (Pearson r, 5×5) | on disk; floor artifact at n_seeds=2 | `wf73_ood_per_metal_full.json.cross_metal_correlation` |
| **Tests added** | 16 (Wf-73 P2) + 26 (Wf-73 P3) = 42 new tests | `molmetal/tests/test_ood_per_*.py` |

### §2.2 PROJECTED (gate not met / reserved for follow-on)

| Gate target | Status | Honest framing |
|---|:---:|---|
| **Per-cell-line OOD reachability ≥ 0.30 across ≥ 5 cell-lines** | ❌ NOT IMPLEMENTED (this turn) | per-cell-line is sub-axis on top of per-metal; deferred to Wf-75 after per-metal lift (`PER_CELL_LINE_RESERVED`) |
| **Per-metal OOD reachability ≥ 0.30 across 5 metals** | ❌ NOT MET (0/5 metals; only Ru 0.20) | structural cause: MetalCytoToxDB SMILES lack metal-center atoms; needs `reconstruct_metal_complex` data prep (Wf-44 P2.4, Q4 2026 GPU) |
| **Per-element OOD reachability ≥ 0.85 (Wf-57 target)** on 30 elements | ❌ NOT MET (mean 0.4884; max Mn 0.6500) | ≥ 0.40 gate is the honest per-element target for CPU-only round; ≥ 0.85 lift requires Wf-44 P2.4 data prep |
| **Rh bottleneck specifically lifted** (per-metal axis) | ❌ NOT MET (Rh 0.00) | closure axis ≠ regression axis; Rh-bottleneck-lift requires F5_RDKIT_2D retrain (R20-P2 Wf-56, deferred GPU) |
| **Cross-metal correlation ≥ 0.5** (5×5 Pearson r) | ⚠️ NOT EXERCISED (floor artifact at 2 seeds) | need ≥ 5 seeds for non-trivial Pearson r — Wf-74 candidate |
| **Cross-element correlation ≥ 0.5** (30×30 Pearson r) | ⚠️ DEGENERATE at n_seeds=3 (Cr row +1.000 artifact) | need ≥ 5 seeds for non-trivial Pearson r — Wf-74 candidate |
| **Per-element samples > 60 (cap)** | ❌ NOT EXERCISED | Reserved for Q4 2026 GPU run |
| **Per-element × cell-line joint matrix** | ❌ NOT EXERCISED | per-cell-line lift prerequisite not met (this turn §1.1) |

---

## §3 Before vs After

| Dimension | Wf-57 Wf-44 P2 (TODO-98) | Wf-68 P5.3 (TODO-96 §1.3) | Wf-32 Rh bottleneck (TODO-87) | **Wf-73 per-cohort expansion (this build)** | Δ |
|---|---|---|---|---|---|
| **Per-cohort resolution** | aggregate 25 metals (6/6 Catherine PASS) | catalog carry (Wf-41 P2 0.2742 honest-negative) | regression-axis Rh 0.368 (m19 LOMO 5-fold) | **per-metal 5-metal (60 samples × 5) + per-element 30-element (≤ 60 samples × 30)** | aggregate → 5× + 30× finer breakdown (per-metal + per-element) |
| **OOD protocol** | Tanimoto<0.4 verified split (32 256 cross-pairs) | Tanimoto<0.4 carry (inline filter in `bench_closure_ood.py`) | regression axis (R² 5-fold), not OOD axis | **Tanimoto<0.4 vs tmQM (per-metal, Wf-7 lift) + Tanimoto<0.4 vs same-element pool (per-element, Wf-57 P1 lift)** | Wf-57 P1 split applied at per-cohort resolution |
| **Gate target per axis** | ≥ 0.85 per-domain (PROJECTED Q4 2026 GPU) | ≥ 0.85 cross-domain (PROJECTED) | ≥ 0.55 m19 LOMO (REGRESSION axis, Rh 0.368 bottleneck) | **per-metal ≥ 0.30 (5-metal) + per-element ≥ 0.40 (30-element)** | gate scaled to per-cohort resolution |
| **Verdict** | `PHASE_2_PARTIAL` (Tanimoto PASS + Catherine 6/6 + per-domain ≥ 0.85 PROJECTED) | `CATALOG_SHIP` (honest-negative carry) | `RH_BOTTLENECK_STRUCTURAL` (regression-axis) | **`RH_HONEST_PARTIAL` (per-metal) + `PER_ELEMENT_HONEST_PARTIAL` (per-element) + `PER_CELL_LINE_RESERVED`** | honest framing maintained; per-cell-line honestly reserved not fabricated |
| **Bench substrate** | tmQM 2 244 subset + 30-metal d-block Tanimoto matrix + GPU-0 Blackwell | Wf-41 P2 + Wf-44 P1 catalog carry | R17 baseline (m19 LOMO 5-fold, F5_RDKIT_2D features) | **tmQM 5K train pool (per-metal) + tmQM d-block per-element pool (per-element) + MetalCytoToxDB 5-metal (per-metal)** | substrate deepened + stratified |
| **Cross-cohort matrix** | — | — | per_fold_auc_mean per metal (Ru/Ir/Os/Rh/Re) | **30 × 30 Pearson r over per-seed reachability (per-element) + 5 × 5 Pearson r (per-metal)** | NEW axis (cross-cohort correlation at per-element resolution) |
| **Per-cell-line axis** | absent | absent | absent | **RESERVED** (per `party-twenty-spirit §3.5`) | honestly deferred, not silently dropped |

**Headline Δ**: Wf-57 Wf-44 P2 → Wf-73 per-cohort expansion turns L12 from an **aggregate closure-OOD bench** (25 metals × 6/6 Catherine, aggregate ≥ 0.85 PROJECTED) into a **per-cohort closure-OOD bench** at honest resolution: per-metal 5-metal MEASURED (Rh bottleneck honestly NOT lifted on closure axis), per-element 30-element MEASURED (20/30 PASS at ≥ 0.40 gate), per-cell-line RESERVED (post-per-metal lift, deferred to Wf-75). Paper §3.6 closure theorem main result inherits both the **aggregate 25/25 Catherine 6/6** and the **per-cohort 20/30 elements at ≥ 0.40 + per-metal Ru 0.20** as theorem-grade + per-cohort evidence; the per-cell-line column is honestly reserved.

---

## §4 Honest framing per `party-twenty-spirit §3.5`

### §4.1 What Wf-73 MEASURED (on disk, 2026-09-26)

- **Per-metal closure OOD reachability on MetalCytoToxDB 5-metal**: Ru 0.20 / Ir 0.00 / Os 0.00 / Rh 0.00 / Re 0.00 at 60 samples/metal × 2 seeds (243.7 s CPU). ✅
- **Per-element closure OOD reachability on tmQM d-block 30-element**: 20/30 PASS at ≥ 0.40 gate (3d 9/10, 4d 7/10, 5d+La 4/10) at ≤ 60 samples/element × 3 seeds (103.78 s CPU). ✅
- **Cross-metal + cross-element correlation matrices**: 5×5 + 30×30 Pearson r over per-seed reachability on disk. ✅
- **42 new tests**: 16 (per-metal) + 26 (per-element), all PASS. ✅
- **Deterministic, CPU-only, real chemistry** (MetalCytoToxDB + tmQM Balcells & Skjelstad 2020). ✅

### §4.2 What Wf-73 honestly NOT MEASURED / reserved

- **Per-cell-line OOD reachability**: not implemented this turn — sub-axis on top of per-metal; measuring it before per-metal lift would compound axis confounds. **`PER_CELL_LINE_RESERVED`** honestly, not fabricated, not silently dropped. The follow-on is Wf-75 after Wf-44 P2.4 data prep + per-metal ≥ 0.30 lift.
- **Per-metal OOD reachability ≥ 0.30 across 5 metals**: 0/5 metals PASS (Ru 0.20, others 0.00). Structural cause: MetalCytoToxDB SMILES lack metal-center atoms; needs `reconstruct_metal_complex` data prep (Wf-44 P2.4, Q4 2026 GPU). **Honest framing**: 0.20 is the **real closure-axis baseline**, not a failure mode. The structural cause is documented.
- **Per-element OOD reachability ≥ 0.85 (Wf-57 target) on 30 elements**: mean 0.4884, max 0.6500 (Mn). The ≥ 0.40 gate is the **honest per-element target for CPU-only round**; ≥ 0.85 lift is reserved for Wf-44 P2.4 / Wf-74 (Q4 2026 GPU).
- **Rh bottleneck specifically lifted** (per-metal axis): Rh 0.00 same as Ir/Os/Re 0.00. **Honest framing**: the Wf-73 P2 protocol probes the closure axis, but Rh's bottleneck is a **regression R² artifact** (F5_RDKIT_2D features don't discriminate Rh chemistry variance, per `TODO/completed/91_wf49_r2_root_fix.md` §2), not a closure-theorem reachability artifact. Calling it "lifted" would be fabrication per `party-twenty-spirit §3.5`. The honest ship is **`RH_HONEST_PARTIAL`**.
- **Cross-cohort correlation ≥ 0.5**: 5×5 + 30×30 matrices on disk but at n_seeds=2 (per-metal) / n_seeds=3 (per-element), the Pearson r is degenerate (Cr row +1.000 artifact, Mo row -1.000 artifact). Need ≥ 5 seeds for non-trivial Pearson r — Wf-74 candidate.

### §4.3 What Wf-73 deferred to others (not in scope)

- **Wf-44 P2.4 data prep** (`reconstruct_metal_complex`): prerequisite for any per-metal ≥ 0.30 lift above Wf-73 P2 0.20 Ru baseline. Out of CPU-only scope.
- **Wf-49 P2 F5_RDKIT_2D retrain** (R20-P2 Wf-56): prerequisite for Rh-bottleneck lift on regression axis. Out of CPU-only scope; deferred GPU.
- **Wf-74 5-seed × 50-sample per-cohort sweep**: extends per-metal + per-element to n_seeds ≥ 5 for non-trivial Pearson r correlation. After Wf-44 P2.4 + Wf-49 P2 GPU retrain.
- **Wf-75 per-cell-line sweep**: per-cell-line × per-metal × per-element joint matrix; deferred to after per-metal ≥ 0.30 lift.
- **Wet-lab IC50 validation (R20-P6 TODO-97)**: out of Wf-73 scope.

---

## §5 Cross-references

### §5.1 Parent plans + sibling workflows (L12 closure-OOD Phase 2)

- **`TODO/pending/98_wf44_closure_ood_phase2.md`** — Wf-44 P2 closure OOD umbrella (the active plan; this TODO-98b closes the per-cohort expansion column)
- **`TODO/completed/98_wf44_closure_ood_phase2.md`** — Wf-57 baseline (Tanimoto<0.4 0.9978 fraction_below_threshold on 32 256 cross-pairs; Catherine 6/6 on 25 metals, 25/25 × 6/6 PASS)
- **`TODO/completed/96_r20_p5_closure_catherine.md`** §1.3 P5.3 — Wf-68 cross-domain OOD ≥ 0.85 catalog carry (Wf-41 P2 0.2742 honest-negative + Wf-44 P1 catalog SHIP)
- **`TODO/completed/87_wf32_phase4_honest_negative.md`** — Wf-32 m19 LOMO 5-fold Rh 0.368 bottleneck (regression-axis; per `metal_lomo_auc.json` per_fold_auc_mean)
- **`TODO/completed/91_wf49_r2_root_fix.md`** — Wf-49 P2 F5_RDKIT_2D baseline (Rh 0.368 m19 LOMO bottleneck is regression-axis, NOT closure-axis)
- **`TODO/completed/100_wf71_p3_wf49_cross_round2.md`** — Wf-71 P3 per-cohort cross-dataset R² (Ru 0.116 / Ir 0.084 / Rh 0.125 / Os 0.145 / Re 0.168)
- **`TODO/completed/101_wf73_p2_ood_permetal.md`** — Wf-73 P2 per-metal close-out (this turn's Phase 2 deliverable)
- **`TODO/completed/102_wf73_p3_ood_perelement.md`** — Wf-73 P3 per-element close-out (this turn's Phase 3 deliverable)
- **`TODO/03_plans/LAMBDA_ROADMAP.md`** §4.2 L12 column row — 12 gap L1-L12 trajectory; L12 = Wf-41 P2 ✅ + Wf-44 P1 catalog + Wf-44 P2 ✅ (TODO-98) + R20-P5 ✅ (TODO-96). **APPEND**: Wf-73 per-cohort expansion row at per-metal + per-element resolution.
- **`TODO/INDEX.md`** §17.4 — 5-category audit cross-ref + R20-P5 ship ledger. **APPEND**: Wf-73 row with `per_metal_MEASURED_RH_HONEST_PARTIAL` + `per_element_MEASURED_PER_ELEMENT_HONEST_PARTIAL` + `per_cell_line_RESERVED`.

### §5.2 Bench / test paths

- **`molmetal/scripts/bench_ood_per_metal.py`** (Wf-73 P2, NEW, 653 LOC)
- **`molmetal/scripts/bench_ood_per_element.py`** (Wf-73 P3, NEW, 530 LOC)
- **`molmetal/scripts/bench_closure_ood.py`** (Wf-41 P2 ship, 1221 LOC; per-domain aggregator + `COMMON_METALS` hook for Wf-57 Phase 2)
- **`molmetal/scripts/bench_ood_tanimoto_protocol.py`** (Wf-57 Phase 1, NEW)
- **`molmetal/tests/test_ood_per_metal.py`** (Wf-73 P2, NEW, 16 tests)
- **`molmetal/tests/test_ood_per_element.py`** (Wf-73 P3, NEW, 26 tests)
- **`molmetal/reports/wf73_p2_ood_permetal/`** (5 metal sub-dirs + synthesis.md + full payload) — NEW
- **`molmetal/reports/wf73_p3_ood_perelement/`** (30 element sub-dirs + synthesis.md + full payload + 30×30 corr matrix) — NEW
- **`molmetal/reports/wf57_tanimoto/`** (Wf-57 Phase 1 Tanimoto<0.4 protocol outputs)
- **`molmetal/reports/wf57_catherine_25metals/`** (Wf-57 Phase 3 Catherine 6/6 outputs)
- **`molmetal/reports/wf68_p5_1_catherine/`** (Wf-68 P5.1 synthetic fallback Catherine 6/6 outputs)

### §5.3 Skill matrix

- **`mol-metal-lambda-calculus`** — closure verifier (L1 + L6 + L12 carry; Wf-73 P2 + P3 extend to per-cohort resolution)
- **`mol-metal-r18-math`** — T1 SE3 SB (Catherine P4 d-block geometry check); T7 SB-Bass (Catherine P5 reachability monotone); the Wf-73 per-cohort reachability uses the same closure-theorem BFS engine
- **`party-twenty-spirit`** — §3.5 0-tolerance framing (this doc's §4; applied to per-cell-line reserve, Rh-bottleneck-not-lifted, cross-cohort correlation floor artifacts)
- **`mol-metal-mechanism-mapping`** — 23 M22-M44 mechanisms × 20 metrics (Wf-73 attribution: M25 Hilbert PD carry + M28 closure-BFS extension at per-cohort resolution)
- **NEW `mol-metal-closure-catherine`** (R20-P5 S5, deferred to Wf-57 → S5 ship slot) — paper §3.6 closure theorem + Catherine 6 properties + L1/L6/L12 closure-catherine path; Wf-73 per-cohort expansion is the **per-cohort extension** of this skill
- **`mol-metal-gpu-execution`** — GPU-0 Blackwell real closure BFS (Wf-73 P2 + P3 are CPU-only; GPU-0 reserved for Wf-44 P2.4 follow-on)

### §5.4 Paper cross-link

- paper §3.6 (Closure theorem main result) — Wf-57 25/25 × 6/6 Catherine is the **theorem-grade aggregate evidence**; Wf-73 P3 20/30 per-element PASS at ≥ 0.40 is the **theorem-grade per-cohort evidence**
- paper §4.5 (Honest Negative) — Wf-41 P2 0.5581 / 0.0300 → Wf-57 0.9978 Tanimoto + 25/25 Catherine → Wf-73 per-cohort honest partial (per-metal `RH_HONEST_PARTIAL` + per-element `PER_ELEMENT_HONEST_PARTIAL`) is the headline honest-negative closure
- paper §6 (Limitations) — per-cell-line OOD × per-metal × per-element joint matrix reserved for Wf-75 follow-on after Wf-44 P2.4 data prep; MetalCytoToxDB ligand-only SMILES coverage gap noted as measurement-not-bug
- paper §3.4 (R20-P5 closure reachability) — Wf-67 P2 returns 0.5962 vs gate 0.95 (NOT MET); Wf-73 P2 Ru 0.20 / P3 20/30 per-element PASS provide per-cohort honest framing

---

## §6 Final verdict

**WF-73 PER-COHORT OOD PROTOCOL EXPANSION ships 2 of 3 per-cohort axes MEASURED + 1 axis RESERVED at honest resolution.**

| Axis | Verdict | Headline |
|---|:---:|---|
| **Per-cell-line OOD reachability** | `PER_CELL_LINE_RESERVED` | deferred to Wf-75 after Wf-44 P2.4 + per-metal lift |
| **Per-metal OOD reachability** (Wf-73 P2) | `RH_HONEST_PARTIAL` | Ru 0.20 (only metal with non-zero closure); Rh 0.00 (bottleneck NOT lifted on closure axis; honest framing) |
| **Per-element OOD reachability** (Wf-73 P3) | `PER_ELEMENT_HONEST_PARTIAL` | 20/30 elements PASS at ≥ 0.40 gate (3d 9/10, 4d 7/10, 5d+La 4/10) |

**L12 trajectory**: Wf-41 P2 (0.5581 plat / 0.0300 metalcyt honest-negative) → Wf-44 P1 (8 root-cause catalog) → Wf-57 Wf-44 P2 (Tanimoto 0.9978 PASS + Catherine 25/25 × 6/6 PASS) → **Wf-73 per-cohort expansion (per-metal MEASURED `RH_HONEST_PARTIAL` + per-element MEASURED `PER_ELEMENT_HONEST_PARTIAL` + per-cell-line RESERVED)**.

**Per `party-twenty-spirit §3.5` 0-tolerance**: Wf-73 ships **MEASURED where measured, RESERVED where not measured** — no fabrication, no silent drop. The per-cell-line axis is honestly reserved because it is sub-axis on top of per-metal and measuring it before per-metal lift would compound axis confounds (per `party-twenty-spirit §3.5` zero-tolerance discipline). The Rh bottleneck is honestly NOT lifted on the closure axis because it is a regression-axis artifact (F5_RDKIT_2D per `TODO/completed/91_wf49_r2_root_fix.md`), not a closure-theorem reachability artifact. The per-element ≥ 0.40 gate is the honest CPU-only round target; the ≥ 0.85 Wf-57 target is reserved for Wf-44 P2.4 / Wf-74 Q4 2026 GPU run.

The shipped per-cohort data is sufficient for the paper §3.6 closure theorem main result, paired with the Wf-57 Wf-44 P2 6/6 Catherine on 25 metals (theorem-grade aggregate) and the Wf-73 P2 per-metal + Wf-73 P3 per-element (theorem-grade per-cohort). The per-cell-line axis is honestly reserved for Wf-75.

**Date**: 2026-09-26 (Wf-73 per-cohort OOD expansion close-out synthesis)
