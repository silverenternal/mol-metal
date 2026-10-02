# Wf-380-A7 — TODO-206 m6_kappa metric redefinition finalization

**Status:** ✅ **SHIPPED** — 2026-09-30
**Workflow:** Wf-380-A7 (per `TODO/pending/206_m6_kappa_metric_redefinition.md` source
brief + Wf-380-A7 task brief)
**Owner:** Hugo Lee + Wf-380-A7 ultracode
**Type:** DOCS-ONLY + 1 NEW TEST (CPU-only per AGENTS.md §5.1)
**Cross-ref:** `party-twenty-spirit §3.5` + AGENTS.md §5.1 + §5.2 + §5.3 + §5.4

---

## §0 TL;DR

| Item | Value | Status |
|---|---:|---|
| **4 docs verified SHIPPED** | `docs/PAPER_S6_M6_KAPPA_REFRAME.md` (Wf-375-A4) + `docs/SECTION_3_METRIC_REFRAME_TEMPLATE.md` (Wf-376-A4) + `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` (Wf-377-A1) + `docs/M6_KAPPA_FINAL_VERDICT.md` (Wf-365-A3) | ✅ all 4 SHIPPED |
| **Section counts per task brief** | 8 / 7 / 8 / 9 = **32 total sections** | ✅ all match (PAPER_S6 §0-§7 = 8 sections; SECTION_3 §0-§6 = 7 sections; REDESIGN §0-§7 = 8 sections; FINAL_VERDICT §1-§9 = 9 sections) |
| **Gaps filled** | **4 cross-linkage additions** (Wf-380-A7 self-reference in all 4 docs + Wf-376-A4 + Wf-377-A1 cross-refs in PAPER_S6; Wf-377-A1 cross-ref in SECTION_3; Wf-375-A4 + Wf-376-A4 + Wf-377-A1 + Wf-380-A7 cross-refs in FINAL_VERDICT; Wf-380-A7 self-reference in REDESIGN) | ✅ all gaps filled |
| **NEW tests** | **11 tests PASS** in 0.13 s wall | ✅ `molmetal/tests/test_wf380a7_m6kappa_finalize.py` |
| **Code patches** | 0 NEW source files; 4 docs cross-link augmentation only | ✅ docs-only |
| **GPU launches** | 0 | ✅ CPU-only per AGENTS.md §5.1 |
| **Paper/* touches** | 0 | ✅ per AGENTS.md §5.3 + TODO-84 |
| **Pre-flight v3 status** | 92/0/2/94 PRESERVED at Wf-370-B1 baseline (NOT re-run by Wf-380-A7 docs-only sync) | ✅ PRESERVED |
| **m6_kappa gate verdict** | HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL FINAL (8 paths exhausted; no more architectural candidates) | ✅ PRESERVED per Wf-377-A1 |

**One-line verdict**: The 4 m6_kappa reframe docs are SHIPPED with correct section counts
matching the task brief (8/7/8/9 = 32 sections total). 4 cross-linkage gaps were filled by
adding Wf-380-A7 self-reference + post-Wf-375-A4 cross-refs to Wf-376-A4 / Wf-377-A1. The
NEW test verifies docs exist + sections + cross-references + 8-path inventory + §3 lineup
invariant + LaTeX template presence. m6_kappa remains HONEST_NEGATIVE_PARTIAL_STRUCTURAL
FINAL FINAL (7 architectural paths + 1 metric-redesign layer per Wf-377-A1).

---

## §1 Per-doc verification status

### §1.1 `docs/PAPER_S6_M6_KAPPA_REFRAME.md` (Wf-375-A4 SHIPPED)

- **Size on disk:** 30,892 bytes (~480 LOC).
- **Section count:** 8 (§0 TL;DR + §1 Why fails + §2 Recommended framing + §3 LaTeX §6
  snippet + §4 Cross-references + §5 Wall + §6 Standing rules + §7 Final state).
- **Original cross-refs:** 59 Wf-mentions (Wf-371-A1 + Wf-365-A3 + Wf-325D + Wf-335-M1 +
  Wf-359-A1 + Wf-361-B1 + Wf-361-B2 + Wf-364-A2 + Wf-369-A3 + Wf-364-A4 + Wf-341-T1 +
  Wf-370-A3).
- **Wf-380-A7 augmentation:** added §4.8 "Wf-380-A7 finalization cross-linkage" subsection
  referencing Wf-376-A4 + Wf-377-A1 + Wf-380-A7 + the 8-path inventory aggregate.
- **LaTeX §3 block:** ~95 LOC verbatim ready-to-paste LaTeX for `paper/sections/06_limitations_v4.tex`
  §6.1.6 with 7-path inventory verbatim.
- **Status:** ✅ SHIPPED + Wf-380-A7 cross-linkage gap closed.

### §1.2 `docs/SECTION_3_METRIC_REFRAME_TEMPLATE.md` (Wf-376-A4 SHIPPED)

- **Size on disk:** 29,579 bytes (~480 LOC).
- **Section count:** 7 (§0 TL;DR + §1 Why fails + §2 Recommended §3 lineup + §3 LaTeX §3
  snippet + §4 USER_OK-gate status + §5 Cross-references + §6 How to use template).
- **Original cross-refs:** 24 Wf-mentions (Wf-365-A3 + Wf-371-A1 + Wf-335-M1 + Wf-335-M2 +
  Wf-333-V3 + Wf-341-T1 + Wf-355-B1 + Wf-359-A1 + Wf-361-B1 + Wf-361-B2 + Wf-362-A1/A2/A3 +
  Wf-364-A2 + Wf-364-A4 + Wf-365-A3 + Wf-369-A2).
- **Wf-380-A7 augmentation:** added §5.4 "Wf-380-A7 finalization cross-linkage" subsection
  referencing Wf-375-A4 + Wf-377-A1 + Wf-380-A7 + the §3 lineup invariant (4 headline + 3
  supplementary).
- **LaTeX §3 block:** ~150 LOC verbatim ready-to-paste LaTeX for `paper/sections/03_main_results_v4.tex`
  with 4-metric lineup (m1_auc + m2 + m3 + m15) + m6_kappa demoted.
- **Status:** ✅ SHIPPED + Wf-380-A7 cross-linkage gap closed.

### §1.3 `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` (Wf-377-A1 SHIPPED)

- **Size on disk:** 35,609 bytes (~590 LOC).
- **Section count:** 8 (§0 TL;DR + §1 Why fails (deeper fix) + §2 Literature review +
  §3 Empirical validation + §4 Recommended replacement + §5 Cross-references + §6 Final
  state + §7 Standing rules).
- **Original cross-refs:** 53 Wf-mentions (Wf-371-A1 + Wf-365-A3 + Wf-335-M1 + Wf-359-A1 +
  Wf-361-B1 + Wf-361-B2 + Wf-364-A2 + Wf-375-A4).
- **Wf-380-A7 augmentation:** added Wf-380-A7 self-reference to §5.1 source files list.
- **Empirical validation:** 16 invariants × n=5460 MetalCytoToxDB labelled pIC50; top-3
  alternatives (MolLogP +0.1999, MorganDensity_r2 +0.1644, Chi3v +0.1541) — all WEAK band
  per Cohen 1988; sanity-check anchor bit-identical to Wf-325D canonical at 15 significant
  digits (Δ = 2e-17).
- **§6.1.7 LaTeX:** ~30 LOC LaTeX extending Wf-375-A4 §6.1.6 entry.
- **Status:** ✅ SHIPPED + Wf-380-A7 self-reference added.

### §1.4 `docs/M6_KAPPA_FINAL_VERDICT.md` (Wf-365-A3 SHIPPED)

- **Size on disk:** 17,028 bytes (~270 LOC).
- **Section count:** 9 (§1 Per-attempt + §2 Why 0.6 fails + §3 What lifted + §4 What remains +
  §5 Forward-looking + §6 Paper §6 entry prep + §7 Cross-references + §8 Standing rules +
  §9 Wall).
- **Original cross-refs:** 8 Wf-mentions (Wf-335-M1 + Wf-359-A1 + Wf-361-B1 + Wf-361-B2 +
  Wf-364-A2 + Wf-365-A3 + Wf-365-A1 superseded by Wf-361-B2).
- **Wf-380-A7 augmentation:** added Wf-375-A4 + Wf-376-A4 + Wf-377-A1 + Wf-380-A7 rows to
  §7 cross-references table.
- **Verdict (per §4):** m6_kappa gate PRESERVED as HONEST_NEGATIVE_PARTIAL_STRUCTURAL
  FINAL state; 5 architectural attempts at Wf-365-A3 timestamp; extended to 7 architectural
  by Wf-371-A1 + 8 total by Wf-377-A1 metric-redesign layer.
- **Status:** ✅ SHIPPED + Wf-380-A7 cross-linkage gap closed.

---

## §2 Gaps filled (per task brief §3 conditional gaps)

### §2.1 PROJECTED lift envelope (per Wf-377-A1 §3)

- **Status:** ✅ ALREADY FILLED — `docs/M6_KAPPA_REDESIGN_PROPOSAL.md §3.2` contains the
  cohort-level ranking of 16 invariants with concrete Pearson r + Spearman ρ values. Top-3
  alternatives (MolLogP +0.1999, MorganDensity_r2 +0.1644, Chi3v +0.1541) are quantified
  with bit-for-bit reproducibility (sanity-check anchor vs Wf-325D canonical at 15 significant
  digits).
- **Gap-fill action:** None required (Wf-377-A1 already filled).

### §2.2 LaTeX template (per task brief gap-fill spec)

- **Status:** ✅ ALREADY FILLED — 3 of 4 docs contain ready-to-paste LaTeX with
  `\begin{itemize}` blocks:
  - `docs/PAPER_S6_M6_KAPPA_REFRAME.md §3` (~95 LOC) for `paper/sections/06_limitations_v4.tex`
    §6.1.6 (7-path inventory verbatim)
  - `docs/SECTION_3_METRIC_REFRAME_TEMPLATE.md §3` (~150 LOC) for
    `paper/sections/03_main_results_v4.tex` (4-metric headline lineup)
  - `docs/M6_KAPPA_REDESIGN_PROPOSAL.md §4.4` (~30 LOC) for `paper/sections/06_limitations_v4.tex`
    §6.1.7 (metric-redesign layer extending §6.1.6)
- **Gap-fill action:** None required (3 docs already filled). `docs/M6_KAPPA_FINAL_VERDICT.md §6`
  contains an earlier draft snippet (~25 LOC) that is preserved as a baseline.

### §2.3 Cross-references (per task brief gap-fill spec)

- **Status:** ⚠️ PARTIAL — cross-refs were present within each doc but the 4 docs did NOT
  all cross-reference each other (because they were created at different times).
- **Gap-fill action:** ✅ FILLED by Wf-380-A7:
  - Added `§4.8 "Wf-380-A7 finalization cross-linkage"` to PAPER_S6 referencing Wf-376-A4 +
    Wf-377-A1 + Wf-380-A7.
  - Added `§5.4 "Wf-380-A7 finalization cross-linkage"` to SECTION_3 referencing Wf-375-A4 +
    Wf-377-A1 + Wf-380-A7.
  - Added Wf-380-A7 self-reference to `§5.1 source files` in REDESIGN.
  - Added Wf-375-A4 + Wf-376-A4 + Wf-377-A1 + Wf-380-A7 rows to `§7 cross-references` table
    in FINAL_VERDICT.
- **Total cross-linkage additions:** 4 (one per doc, totaling ~50 LOC).

---

## §3 Test status

### §3.1 NEW test file: `molmetal/tests/test_wf380a7_m6kappa_finalize.py`

- **Lines:** ~310 LOC
- **Test class:** `TestW380A7M6KappaFinalize` (11 test methods)
- **Wall time:** 0.13 s (CPU-only)
- **Pass rate:** 11/11 = **100% PASS**

### §3.2 Per-test verdict

| # | Test | Verifies | Result |
|---|---|---|---|
| 1 | `test_all_4_docs_exist` | All 4 docs on disk + size > 5000 bytes | ✅ PASS |
| 2 | `test_paper_s6_sections` | PAPER_S6 §0-§7 = 8 section headers present | ✅ PASS |
| 3 | `test_section_3_sections` | SECTION_3 §0-§6 = 7 section headers present | ✅ PASS |
| 4 | `test_redesign_sections` | REDESIGN §0-§7 = 8 section headers present | ✅ PASS |
| 5 | `test_final_verdict_sections` | FINAL_VERDICT §1-§9 = 9 section headers present | ✅ PASS |
| 6 | `test_cross_refs_present` | 14 cross-reference invariants (PAPER_S6 → Wf-376-A4/Wf-377-A1/Wf-365-A3/Wf-371-A1; SECTION_3 → Wf-377-A1/Wf-365-A3/Wf-371-A1/Wf-375-A4; REDESIGN → Wf-375-A4/Wf-371-A1/Wf-365-A3; FINAL_VERDICT → Wf-375-A4/Wf-376-A4/Wf-377-A1) | ✅ PASS |
| 7 | `test_8_path_inventory_aggregates` | 8 unique Wf paths present across all 4 docs (Wf-335-M1 + Wf-359-A1 + Wf-361-B1 + Wf-361-B2 + Wf-364-A2 + Wf-365-A3 + Wf-371-A1 + Wf-377-A1) | ✅ PASS |
| 8 | `test_section_3_lineup_invariant` | Each doc contains DROP/demote m6_kappa recommendation + m1_auc + m2 + m3 + m15 headline references | ✅ PASS |
| 9 | `test_top3_supplementary_promoted` | REDESIGN contains MolLogP + MorganDensity_r2 + Chi3v (top-3 alternatives) | ✅ PASS |
| 10 | `test_w380a7_self_reference` | All 4 docs reference Wf-380-A7 (added by this closeout) | ✅ PASS |
| 11 | `test_latex_template_present` | PAPER_S6 + SECTION_3 contain `\begin{itemize}` + `\MEASURED` tag discipline | ✅ PASS |

### §3.3 Test file conventions followed

- Path: `molmetal/tests/test_wf380a7_m6kappa_finalize.py` (per Wf-377-A1 + Wf-371-A1 + Wf-365-A3
  convention).
- Style: `unittest.TestCase` (matches `test_wf377a1_m6kappa_redesign.py` baseline).
- Docstring header with date + author + CPU-only + AGENTS.md §5.3 marker.
- Imports: `pathlib`, `unittest`, `re` (no torch required).
- `REPO_ROOT = Path(__file__).resolve().parents[2]` (matches project convention).

---

## §4 Cross-references (Wf-380-A7 finalization inventory)

### §4.1 Aggregated 4-doc lineage (m6_kappa reframe surface)

| # | Doc | Workflow | Section count | Role | Wall |
|---:|---|---|---:|---|---:|
| 1 | `docs/M6_KAPPA_FINAL_VERDICT.md` | **Wf-365-A3** | 9 | ORIGINAL VERDICT — 5 architectural paths aggregated | ~7 min CPU |
| 2 | `docs/PAPER_S6_M6_KAPPA_REFRAME.md` | **Wf-375-A4** | 8 | §6.1.6 LaTeX + 7-path inventory | ~15-20 min CPU |
| 3 | `docs/SECTION_3_METRIC_REFRAME_TEMPLATE.md` | **Wf-376-A4** | 7 | §3 main-result LaTeX + 4-metric headline | ~5 min CPU |
| 4 | `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` | **Wf-377-A1** | 8 | §6.1.7 LaTeX + 8th path (metric redesign) | ~15 min CPU |
| 5 | `TODO/completed/380a7_m6kappa_finalize.md` | **Wf-380-A7** (this) | 5 | FINALIZATION — aggregates 4 docs + 1 NEW test | ~20 min CPU |

### §4.2 8-path m6_kappa attempt inventory (post-Wf-377-A1)

| # | Workflow | Path | aug_mean | CV-R² | Verdict |
|---:|---|---|---:|---:|---|
| 1 | **Wf-335-M1** | chemprop+PLL (248-d) | 0.4386 | **+0.6814** | HONEST_NEGATIVE_PARTIAL_STRUCTURAL |
| 2 | **Wf-359-A1** | R20-P3 W₂ (160-d) | 0.356073 | n/a | HONEST_NEGATIVE_PARTIAL_STRUCTURAL |
| 3 | **Wf-361-B1** | MAML (K_shot=8) | 0.4288 | +0.2909 | HONEST_NEGATIVE_PARTIAL_STRUCTURAL |
| 4 | **Wf-361-B2** | Neural ODE dopri5 | 0.4495 | **+0.5410** | HONEST_NEGATIVE_PARTIAL_STRUCTURAL |
| 5 | **Wf-364-A2** | Combined MAML+ODE | **0.4737** (best) | +0.3733 / +0.5694 | HONEST_NEGATIVE_PARTIAL_STRUCTURAL |
| 6 | **Wf-365-A3** | Verdict aggregation | n/a | n/a | HONEST_NEGATIVE_PARTIAL_STRUCTURAL |
| 7 | **Wf-371-A1** | DEGENERATE (RDKit 12-d) | 0.449256 | +0.569423 | HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL FINAL |
| 8 | **Wf-377-A1** | METRIC REDESIGN (16 inv × n=5460) | n/a | n/a | NO MODERATE-band correlate; top-3 WEAK alternatives |

**Verdict aggregate:** m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL FINAL.
No architectural attempt lifted aug_mean to 0.6 (best +0.0018 above baseline). No metric
redesign surfaced a MODERATE-band cohort-level correlate (best MolLogP +0.1999 = WEAK).
3 per-metal MODERATE-band discoveries (Rh BalabanJ +0.3953, Rh MolLogP −0.3416, Re MolLogP
+0.3363) cited honestly but do not save m6_kappa as §3 main-result metric.

### §4.3 Tier-2 paper §3 main-result line-up (final, per Wf-380-A7)

| Section | Metric | Status | Source |
|---|---|---|---|
| §3 headline (4) | **m1_auc (MetalCytotoxDB Ru 5-seed A2780 cisplatin)** | **KEEP** | `metrics/by_metric/metalcytotox_ru_auc.json` MEASURED_PARTIAL_LIFT |
| §3 headline (4) | **m2 R² (metalcytotox_combined)** | **KEEP** | `metrics/by_metric/metalcytotox_combined_r2.json::wf335m2_ceiling_break` CEILING_BREAK R²=+0.3196 lift +0.2141 |
| §3 headline (4) | **m3 Vina (5-pocket cohort median)** | **KEEP** | `metrics/by_metric/vina_kcal_per_mol.json::wf333_v3_combined_median` −7.782 kcal/mol |
| §3 headline (4) | **m15 RMSD (5-pocket redock)** | **KEEP** | `metrics/by_metric/docked_pose_rmsd.json::aggregate_median` 0.6578 Å 5/5 PASS |
| §3 supplementary (3) | **MolLogP (Wildman-Crippen LogP)** | **PROMOTE** per Wf-377-A1 | Pearson r=+0.1999 (n=5460) WEAK band |
| §3 supplementary (3) | **MorganDensity_r2 (Morgan ECFP r=2 bit density)** | **PROMOTE** per Wf-377-A1 | Pearson r=+0.1644 WEAK band |
| §3 supplementary (3) | **Chi3v (Kier-Hall valence χ3v)** | **PROMOTE** per Wf-377-A1 | Pearson r=+0.1541 WEAK band |
| §3 main (dropped) | **m6_kappa (Shape Kappa Index)** | **DROP from §3 main-result** — keep §6 honest-limitations | per Wf-371-A1 §3.5 + Wf-365-A3 §5.2 + Wf-377-A1 §4.2 |

### §4.4 Honest-negative catalog entry (m6_kappa, final-final)

Per `docs/HONEST_NEGATIVE_CATALOG.md §1 per-metric` (Wf-369-A3 65 entries):

> **m6_kappa Shape Kappa Index**: HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL FINAL.
> 7 architectural paths + 1 metric-redesign path exhausted (Wf-335-M1 + Wf-359-A1 + Wf-361-B1 +
> Wf-361-B2 + Wf-364-A2 + Wf-365-A3 + Wf-371-A1 + Wf-377-A1). Best augmented_mean 0.4737
> (Wf-364-A2 joint) vs 0.6 threshold = -0.1263 gap. Structural CV-R² ceiling LIFTED to
> +0.6814 (chemprop+PLL MEASURED). MetalCytoToxDB NEGATIVE_WEAK (Pearson r=0.081, Spearman
> ρ=0.111). Drop from Tier-2 §3 main-result table; keep as §6 honest-limitations
> complementary heuristic. Per Wf-371-A1 §3.5 + Wf-365-A3 §5.2 + Wf-377-A1 §4.2.

---

## §5 Final state (Wf-380-A7 verdict)

| Item | Value | Status |
|---|---|---|
| **4 docs verified SHIPPED** | PAPER_S6 + SECTION_3 + REDESIGN + FINAL_VERDICT | ✅ |
| **Section counts match task brief** | 8 / 7 / 8 / 9 = 32 sections | ✅ |
| **Cross-linkage gaps filled** | 4 docs × 1 augmentation each = 4 cross-link additions | ✅ |
| **NEW tests** | 11/11 PASS in 0.13 s | ✅ |
| **8-path inventory aggregated** | 7 architectural + 1 metric-redesign = 8 paths | ✅ |
| **§3 main-result line-up invariant** | m1 + m2 + m3 + m15 RETAIN + top-3 PROMOTE + m6 DROP | ✅ |
| **m6_kappa gate verdict** | HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL FINAL (8 paths) | ✅ |
| **Code patches** | 0 NEW source files | ✅ docs-only |
| **GPU launches** | 0 | ✅ per AGENTS.md §5.1 |
| **Paper/* touches** | 0 | ✅ per AGENTS.md §5.3 + TODO-84 |
| **Pre-flight v3 status** | 92/0/2/94 PRESERVED (NOT re-run by Wf-380-A7) | ✅ |
| **Wall (this Wf-380-A7 docs-only closeout)** | ~20 min CPU-only | MEASURED |

**The m6_kappa Tier-2 paper reframe is now COMPLETE + FINALIZED.** Wf-380-A7 closes the
TODO-206 m6_kappa metric redefinition task by aggregating the 4 SHIPPED docs
(Wf-375-A4 + Wf-376-A4 + Wf-377-A1 + Wf-365-A3) into a single 8-path inventory with
cross-linkage table. The 1 NEW test verifies all 4 docs exist with correct sections +
cross-references + 8-path inventory + §3 line-up invariant + LaTeX template presence.
Paper §3 + §6 v4 conversion requires USER_OK per TODO-84 + AGENTS.md §5.3.

---

**END OF Wf-380-A7 CLOSEOUT** — Owner Hugo Lee + Wf-380-A7 ultracode; cross-ref
`party-twenty-spirit §3.5`; 4 docs SHIPPED + 4 cross-link augmentations + 11/11 tests PASS
in 0.13 s; 0 GPU launches; 0 paper/* touches; 0 NEW source code; m6_kappa gate PRESERVED
HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL FINAL (8 paths exhausted).