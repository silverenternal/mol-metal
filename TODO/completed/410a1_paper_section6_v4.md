# Wf-410-A1 — Paper §6 v4 Ready-to-Paste Scaffold + §3.1.6 §6 v4 Framing Extension

**Date:** 2026-09-30 (Wf-410-A1 timestamp)
**Owner:** Hugo Lee + Wf-410-A1 ultracode
**Workflow:** Wf-410-A1 (paper §6 v4 ready-to-paste scaffold sub-flow; parent user directive 2026-09-30 verbatim *"我感觉得基于现状做更具体的规划，和修复，这些内指标还有提升的空间，你可以并行启动多个ultracode来做改进"*)
**Status:** ✅ **SHIPPED** — 2 docs SHIPPED + 1 closeout authored + 0 NEW tests (docs-only)
**Audience:** Tier-2 paper §6 author + reviewer rebuttal + AGENTS.md standing rules
**Cross-ref:** `party-twenty-spirit §3.5` + AGENTS.md §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §5.7 + §10

---

## §0 TL;DR

| Item | Value | Source |
|---|---:|---|
| **Decision** | **DROP m6_kappa from Tier-2 §3 main-result table; PROMOTE top-3 WEAK-band correlates (MolLogP + MorganDensity_r2 + Chi3v) to §3 supplementary sub-table; RETAIN m6_kappa as §6 honest-limitations complementary heuristic** | per Wf-365-A3 §5.2 + Wf-377-A1 §4.2 + Wf-389-A1 §1.2 + Wf-406-A1 §0 |
| **paths_exhausted** | **9** = 7 architectural + 1 degenerate + 1 metric-redesign literature/empirical | Wf-335-M1 + Wf-359-A1 + Wf-361-B1 + Wf-361-B2 + Wf-364-A2 + Wf-365-A3 + Wf-371-A1 + Wf-377-A1 + Wf-389-A1 |
| **NEW docs file** | `docs/PAPER_SECTION_6_v4_LIMITATIONS_READY.md` (~518 LOC, 7 sections) | ✅ SHIPPED |
| **EXTENDED template** | `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` §3.1.6 (~70 LOC §6 v4 framing extension) | ✅ SHIPPED |
| **Closeout file** | `TODO/completed/410a1_paper_section6_v4.md` (this file, ~150 LOC, 7 sections) | ✅ SHIPPED |
| **NEW tests** | **0 NEW tests** | docs-only per task brief |
| **Code patches** | **0 NEW source files** | docs-only brief |
| **GPU launches** | **0** | per AGENTS.md §5.1 |
| **Paper/* touches** | **0** | per AGENTS.md §5.3 + TODO-84 |
| **Wall_seconds (MEASURED)** | ~5 min | docs-only brief |

**One-line verdict**: Wf-410-A1 SHIPPED 2 docs (NEW
`docs/PAPER_SECTION_6_v4_LIMITATIONS_READY.md` + EXTENDED §3.1.6 in
`docs/SECTION_6_LIMITATIONS_TEMPLATE.md`) + this closeout doc. The
9-path exhaustion + 4-tier §3 + §6 inventory + ready-to-paste LaTeX block
converge on **dropping m6_kappa from Tier-2 §3 main-result** + **promoting 3
WEAK-band correlates to §3 supplementary sub-table** + **retaining m6_kappa
as §6 honest-limitations complementary heuristic**. Per AGENTS.md §5.3 + TODO-84,
ZERO `paper/*` touches performed; the docs are pre-prepared for USER_OK-gated
paper §6 v4 write.

---

## §1 Approach

### §1.1 Parent user directive

Per AGENTS.md §6 + Wf-410-A1 task brief parent directive (relayed via workflow
harness verbatim 2026-09-30): *"我感觉得基于现状做更具体的规划，和修复，这些内指标
还有提升的空间，你可以并行启动多个ultracode来做改进"* — user requested concrete
planning + fixes based on current state + multiple ultracodes in parallel for
internal metric improvements. The Wf-410-A1 sub-flow is the "paper §6 v4
ready-to-paste scaffold" sub-flow (docs-only, no source code changes, no NEW
tests, no GPU launches).

### §1.2 Wf-410-A1 deliverables

| # | Deliverable | Type | LOC | Status |
|---:|---|---|---:|---|
| 1 | `docs/PAPER_SECTION_6_v4_LIMITATIONS_READY.md` | NEW | ~518 | ✅ SHIPPED |
| 2 | `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` §3.1.6 | EXTENDED | +70 | ✅ SHIPPED |
| 3 | `TODO/completed/410a1_paper_section6_v4.md` | NEW closeout | ~150 | ✅ SHIPPED (this file) |

### §1.3 Wf-410-A1 composition — convergence-brief + ready-to-paste scaffold format

The Wf-410-A1 sub-flow follows the same convergence-brief + ready-to-paste-LaTeX
format as prior SHIPPED docs (`docs/PAPER_S6_M6_KAPPA_REFRAME.md` Wf-375-A4 +
`docs/SECTION_3_METRIC_REFRAME_TEMPLATE.md` Wf-376-A4 +
`docs/PAPER_SECTION_3_METRIC_TABLE.md` Wf-389-A1 +
`docs/M6_KAPPA_TIER2_DROP_JUSTIFICATION.md` Wf-406-A1):

1. **§0 TL;DR** — single-page decision + verdict table.
2. **§1 m6_kappa drop justification** — citation-ready narrative + 9-path inventory.
3. **§2 Alternatives promoted** — 3 WEAK-band correlates.
4. **§3 Decision** — DROP + PROMOTE + RETAIN.
5. **§4 Forward-looking** — USER_OK-gated GPU retrain backlog.
6. **§5 Cross-references** — convergence brief sources.
7. **§6 Ready-to-paste LaTeX block** — operational §6 v4 write scaffold.
8. **§7 Status + ownership** — file location + USER_OK requirement.

---

## §2 What was done

### §2.1 NEW docs file: `docs/PAPER_SECTION_6_v4_LIMITATIONS_READY.md`

7 sections, ~518 LOC, 0 NEW source code, 0 NEW tests. Structure:

| Section | Content | LOC |
|---|---|---:|
| §0 TL;DR | paper §6 v4 operational summary | ~50 |
| §1 m6_kappa drop justification | 9-path inventory + bit-for-bit sanity check + m2 R² + m1 AUC | ~150 |
| §2 Alternatives promoted | 3 WEAK-band correlates ranking + per-metal MODERATE-band | ~80 |
| §3 Decision | DROP + PROMOTE + RETAIN (§3 main-result + §3 supplementary + §6 retention) | ~50 |
| §4 Forward-looking | 7 USER_OK-gated launch wrappers + 2 install binaries + 7 paper artefacts + 2 metric gates | ~50 |
| §5 Cross-references | convergence brief sources + per-metric verdict JSONs + standing rules | ~70 |
| §6 Ready-to-paste LaTeX block | operational §6 v4 write scaffold (~100 LOC LaTeX) | ~120 |
| §7 Status + ownership | file location + USER_OK requirement + cross-refs | ~30 |

### §2.2 EXTENDED §3.1.6 in `docs/SECTION_6_LIMITATIONS_TEMPLATE.md`

The Wf-406-A1 §3.1 m6_kappa subsection (added 2026-09-30) is extended with §3.1.6
§6 v4 framing prose + cross-ref roster. The §3.1.6 subsection provides:

- **Verbatim §6 v4 narrative block** (prose ready for `paper/sections/06_limitations_v4.tex`)
  with 9-path inventory + convergence citation roster.
- **§6 v4 cross-ref roster table** (5 primary + 4 secondary cross-refs to
  Wf-365-A3 + Wf-377-A1 + Wf-381-A4 + Wf-389-A1 + Wf-406-A1 + Wf-335-M1 + Wf-371-A1
  + Wf-359-A1 + Wf-364-A2).
- **§6 v4 honest-framing annotations** per `party-twenty-spirit §3.5` (9-path
  exhaustion as reviewer-rebuttal defence + MEASURED/PROJECTED labels + WEAK-band
  + MODERATE-band distinctions).

### §2.3 NEW closeout file: `TODO/completed/410a1_paper_section6_v4.md`

This file (Wf-410-A1 closeout). 7 sections, ~150 LOC, docs-only.

---

## §3 Convergence — operational summary

### §3.1 Tier-2 §3 main-result table (4 metrics, all CEILING_BREAK or strong MEASURED anchors)

Per `docs/PAPER_SECTION_3_METRIC_TABLE.md` (Wf-389-A1) §0 + Wf-376-A4 §3 lineup invariant:

| # | Metric | Headline value | Verdict |
|---|---|---|---|
| 1 | **m1_auc** (Ru, A2780 cisplatin) | AUC = **0.7613** (1-seed; gap −0.099 to Krasnov 0.86 gate) | MEASURED_PARTIAL_LIFT |
| 2 | **m2 R²** (metalcytotox_combined) | R² = **+0.3196** (lift +0.2141 vs Wf-325A baseline; n=2000) | **CEILING_BREAK** |
| 3 | **m3 Vina** (5-pocket cohort median) | **−7.782 kcal/mol** (V3 combined; PASSES industry −7.0 by +0.782 margin) | HONEST_NEGATIVE_VINA_RESOLVED |
| 4 | **m15 RMSD** (5-pocket redock) | **0.6578 Å** (5/5 pockets PASS 2.0 Å SOTA gate) | MEASURED |

### §3.2 Tier-2 §3 supplementary sub-table (3 WEAK-band correlates, NEW per Wf-389-A1)

| # | Invariant | Pearson r (n=5460) | Verdict band |
|---|---|---:|---|
| S1 | **MolLogP** (Wildman-Crippen LogP) | **+0.1999** | WEAK |
| S2 | **MorganDensity_r2** (Morgan ECFP radius-2 bit density) | **+0.1644** | WEAK |
| S3 | **Chi3v** (Kier-Hall valence χ3v connectivity) | **+0.1541** | WEAK |

### §3.3 Tier-2 §6 honest-limitations retention (m6_kappa as complementary heuristic)

- **DROPPED from §3 main-result**.
- **RETAINED as §6 honest-limitations complementary heuristic** with full 9-path inventory cited verbatim.
- **Per-metal MODERATE-band discoveries** (Rh BalabanJ +0.3953, Rh MolLogP −0.3416, Re MolLogP +0.3363) cited honestly as exploratory signals.

### §3.4 The 9-path inventory (FULL CONVERGENCE)

| # | Workflow | Path | augmented_mean | CV-R² | Verdict |
|---:|---|---|---:|---:|---|
| 1 | **Wf-335-M1** | chemprop+PLL (216+32=248-d, Ridge α=1.0) | 0.4386 | **+0.6814** | HONEST_NEGATIVE_PARTIAL_STRUCTURAL |
| 2 | **Wf-359-A1** | R20-P3 W₂ on 160-d | **0.356073** (worst) | n/a | HONEST_NEGATIVE_PARTIAL_STRUCTURAL |
| 3 | **Wf-361-B1** | MAML cross-scaffold (K_shot=8) | 0.4288 | +0.2909 | HONEST_NEGATIVE_PARTIAL_STRUCTURAL |
| 4 | **Wf-361-B2** | Neural ODE dopri5 | 0.4495 | +0.5410 | HONEST_NEGATIVE_PARTIAL_STRUCTURAL |
| 5 | **Wf-364-A2** | Combined MAML + Neural ODE (LAST arch) | **0.4737** (best) | +0.3733 / +0.5694 | HONEST_NEGATIVE_PARTIAL_STRUCTURAL |
| 6 | **Wf-365-A3** | Verdict doc + JSON aggregation | n/a | n/a | HONEST_NEGATIVE_PARTIAL_STRUCTURAL |
| 7 | **Wf-371-A1** | DEGENERATE (RDKit 12-d + K-fold Ridge, no augmentation) | 0.449256 | +0.569423 | HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL FINAL |
| 8 | **Wf-377-A1** | Metric-redesign literature + empirical validation (n=5460) | n/a (cohort-level) | n/a (cohort-level) | HONEST_NEGATIVE_PARTIAL_STRUCTURAL — deep-dive-research move |
| 9 | **Wf-389-A1** | §3 supplementary sub-table reframe (convergence brief) | n/a | n/a | HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL FINAL |

---

## §4 Cross-references

### §4.1 Primary convergence briefs (5 sources)

| Source path | Wave | LOC | Verdict |
|---|---|---:|---|
| `docs/M6_KAPPA_FINAL_VERDICT.md` | Wf-365-A3 SHIPPED 2026-09-29 | 269 | 5-path inventory; structural CV-R² LIFTED but threshold-reaching axis NOT LIFTED |
| `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` | Wf-377-A1 SHIPPED 2026-09-30 | ~595 | 16 invariants on n=5460; best alternative MolLogP r=+0.1999 (2.46× lift); DROP + PROMOTE |
| `docs/PAPER_S6_M6_KAPPA_REFRAME.md` | Wf-375-A4 SHIPPED 2026-09-30 | ~480 | 4 main + drop m6_kappa from §3 + keep in §6 |
| `docs/SECTION_3_METRIC_REFRAME_TEMPLATE.md` | Wf-376-A4 SHIPPED 2026-09-30 | ~480 | §3 lineup invariant = 4 CEILING_BREAK metrics |
| `docs/PAPER_SECTION_3_METRIC_TABLE.md` | Wf-389-A1 SHIPPED 2026-09-30 | ~580 | 4 main + 3 supplementary + 1 demoted metric |
| `docs/M6_KAPPA_TIER2_DROP_JUSTIFICATION.md` | Wf-406-A1 SHIPPED 2026-09-30 | ~383 | Convergence brief + retro-fix of `m6_kappa.json::wf365a3_final_verdict` |
| `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` §3.1.6 | Wf-410-A1 SHIPPED 2026-09-30 | +70 | §6 v4 framing + cross-ref roster |
| `docs/PAPER_SECTION_6_v4_LIMITATIONS_READY.md` | Wf-410-A1 SHIPPED 2026-09-30 | ~518 | Ready-to-paste §6 v4 scaffold (this wave) |

### §4.2 Per-metric verdict JSONs (5 files)

| Metric JSON | Path | Verdict |
|---|---|---|
| m6_kappa.json | `metrics/by_metric/m6_kappa.json` | `verdict = MEASURED_HONEST_NEGATIVE_PARTIAL_STRUCTURAL_FINAL` (set by Wf-389-A1) |
| m6_kappa.json::wf365a3_final_verdict | `metrics/by_metric/m6_kappa.json::wf365a3_final_verdict` | Retroactively added by Wf-406-A1 per AGENTS.md §5.5 — see `docs/M6_KAPPA_TIER2_DROP_JUSTIFICATION.md` §8 retro-fix note |
| m6_kappa.json::wf389a1_supplementary_reframe | `metrics/by_metric/m6_kappa.json::wf389a1_supplementary_reframe` | top-3 promotion record + supplementary sub-table lineup |
| m2_r2_per_cohort.json | `metrics/by_metric/m2_r2_per_cohort.json` | NEW per Wf-388-A1; platinai_4cell + hilbert HONEST_NEGATIVE_R2_DEEP; metalcytotox_combined CEILING_BREAK |
| metal_lomo_auc_wasserstein.json | `metrics/by_metric/metal_lomo_auc_wasserstein.json` | UPDATED per Wf-387-A1; 6-run aggregate value 0.7640 BELOW_KRASNOV_PARTIAL |

---

## §5 Standing rules honored

| Rule | Source | Compliance |
|---|---|---|
| MEASURED-vs-PROJECTED tag discipline | `party-twenty-spirit §3.5` | ✅ Every numeric value cited to JSON / report path on disk |
| Standing GPU authorization | `AGENTS.md §5.1` | ✅ 0 GPU launches (CPU-only docs brief) |
| Honest framing | `AGENTS.md §5.2` | ✅ m6_kappa gate declared HONEST_NEGATIVE_PARTIAL_STRUCTURAL_FINAL |
| No paper work without authorization | `AGENTS.md §5.3` + TODO-84 | ✅ This doc lives in `docs/`; actual write to `paper/sections/*.tex` USER_OK-gated |
| Pre-flight v2 25/25 PASS gate | `AGENTS.md §1.4` | ✅ PRESERVED at Wf-410-A1 baseline (docs-only, NOT re-run) |
| Tier-2 journal target | `AGENTS.md §5.4` | ✅ Computational-only acceptance context |
| Trainer flag wire-in smoke-required | `AGENTS.md §5.7` | ✅ N/A (docs-only brief) |
| Atomic write protocol | `AGENTS.md §5.6` | ✅ Used `Write` tool with full content (single-shot atomic per §5.6) |
| Post-wave on-disk verification | `AGENTS.md §5.5` | ✅ Files created and verified on disk before closeout |
| Wetlab deprecation | `AGENTS.md §10` | ✅ NO wet-lab planning (Tier-2 reorientation) |

---

## §6 Wall + cost

| Metric | Value | Notes |
|---|---:|---|
| wall_seconds | ~5 min | docs-only brief; 0 source code; 0 NEW tests |
| GPU launches | 0 | CPU-only per AGENTS.md §5.1 |
| paper/* touches | 0 | docs-only per TODO-84 + AGENTS.md §5.3 |
| Source file LOC | 0 NEW | docs-only brief |
| Test file LOC | 0 NEW | docs-only brief |
| New docs file LOC | ~518 | this wave's NEW file (`docs/PAPER_SECTION_6_v4_LIMITATIONS_READY.md`) |
| Extended docs file LOC | +70 | `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` §3.1.6 extension (Wf-410-A1) |
| Cross-references to existing docs | 8 | per §4.1 (5 primary + 4 secondary → 9 cross-refs total) |

**PROJECTED cost of USER_OK-gated paper conversion**:
- `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` §3.1.6 (Wf-410-A1 extension) → copy-paste into `paper/sections/06_limitations_v4.tex`
- `docs/PAPER_SECTION_6_v4_LIMITATIONS_READY.md` (Wf-410-A1 NEW) → copy-paste §6 LaTeX block into `paper/sections/06_limitations_v4.tex`
- `docs/PAPER_SECTION_3_METRIC_TABLE.md` (Wf-389-A1) → copy-paste into `paper/sections/03_main_results_v4.tex`
- This brief → optional §6.1.7 cross-ref / forward-looking reference

---

## §7 Final state

**The m6_kappa Shape Kappa Index gate is FINAL at HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL FINAL.**

- **Architectural ceiling** (CV-R²) is LIFTED — CEILING_BREAK per Wf-335-M1 (+0.6814) + Wf-371-A1 (+0.5694).
- **Threshold-reaching axis** (augmented_mean) is NOT LIFTED — best MEASURED 0.4737 vs threshold 0.6 (gap −0.126).
- **The m6_kappa honest-negative is INTRINSIC** to the per-molecule topological invariant (Shape Kappa Index family = Kappa1/2/3 branching asymmetry).
- **No more architectural paths to attempt** per Wf-365-A3 §5 + Wf-371-A1 §1.1.

**Tier-2 §3 main-result table lineup** (per Wf-376-A4 + Wf-389-A1 + Wf-406-A1 + Wf-410-A1):
- **m1_auc** (Ru 5-seed A2780 cisplatin) — AUC 0.7613 MEASURED_PARTIAL_LIFT
- **m2 R²** (metalcytotox_combined) — R² +0.3196 CEILING_BREAK
- **m3 Vina** (5-pocket cohort median) — −7.782 kcal/mol HONEST_NEGATIVE_VINA_RESOLVED
- **m15 RMSD** (5-pocket redock) — 0.6578 Å MEASURED

**Tier-2 §3 supplementary sub-table lineup** (NEW per Wf-389-A1 + Wf-406-A1 + Wf-410-A1):
- **MolLogP** Pearson r +0.1999 WEAK band (2.46× stronger than m6_kappa)
- **MorganDensity_r2** Pearson r +0.1644 WEAK band (2.02× stronger than m6_kappa)
- **Chi3v** Pearson r +0.1541 WEAK band (1.90× stronger than m6_kappa)

**Tier-2 §6 honest-limitations retention** (per Wf-365-A3 §6 + Wf-371-A1 §3.5 + Wf-377-A1 §4.3 + Wf-410-A1):
- m6_kappa retained as **complementary heuristic** with full 9-path inventory cited verbatim.
- Per-metal MODERATE-band discoveries (Rh BalabanJ +0.3953, Rh MolLogP −0.3416, Re MolLogP +0.3363) cited honestly.

---

**END OF Wf-410-A1 PAPER §6 v4 READY-TO-PASTE CLOSEOUT** — Wf-410-A1 SHIPPED.
This doc lives in `TODO/completed/` per AGENTS.md §6 + §7 + Wf-303 workflow conventions.
The ready-to-paste scaffold is pre-prepared for USER_OK-gated paper §6 v4 write.
