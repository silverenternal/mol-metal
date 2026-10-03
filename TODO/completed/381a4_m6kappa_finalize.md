# Wf-381-A4 — TODO-206 m6_kappa metric redefinition finalization

**Status:** ✅ **SHIPPED** — 2026-09-30
**Workflow:** Wf-381-A4 (per `TODO/pending/206_m6_kappa_metric_redefinition.md` source brief + Wf-381-A4 task brief)
**Owner:** Hugo Lee + Wf-381-A4 ultracode
**Type:** DOCS-ONLY + 1 NEW TEST (CPU-only per AGENTS.md §5.1)
**Cross-ref:** `party-twenty-spirit §3.5` + AGENTS.md §5.1 + §5.2 + §5.3 + §5.4

---

## §0 TL;DR

| Item | Value | Status |
|---|---:|---|
| **Canonical TODO-206 source brief** | `TODO/pending/206_m6_kappa_metric_redefinition.md` (5 sections, ~10 KB) | ✅ NEW (created by Wf-381-A4) |
| **4 docs verified SHIPPED** | `docs/PAPER_S6_M6_KAPPA_REFRAME.md` (Wf-375-A4) + `docs/SECTION_3_METRIC_REFRAME_TEMPLATE.md` (Wf-376-A4) + `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` (Wf-377-A1) + `docs/M6_KAPPA_FINAL_VERDICT.md` (Wf-365-A3) | ✅ all 4 SHIPPED |
| **Section counts per task brief** | 8 / 7 / 8 / 9 = **32 total sections** | ✅ all match |
| **Gaps filled** | **0 NEW** (all 3 gap conditions already FILLED by Wf-380-A7 predecessor wave: PROJECTED lift envelope in REDESIGN §3.2 + LaTeX template in 3 docs + cross-references in FINAL_VERDICT §7) | ✅ NO-OP gap-fill |
| **NEW tests** | **14 tests PASS** in 0.002 s wall | ✅ `molmetal/tests/test_wf381a4_m6kappa_finalize.py` |
| **Code patches** | 0 NEW source files; 1 NEW docs (pending brief) + 1 NEW test file | ✅ docs-only |
| **GPU launches** | 0 | ✅ CPU-only per AGENTS.md §5.1 |
| **Paper/* touches** | 0 | ✅ per AGENTS.md §5.3 + TODO-84 |
| **Pre-flight v3 status** | 92/0/2/94 PRESERVED at Wf-370-B1 baseline (NOT re-run by Wf-381-A4 docs-only sync) | ✅ PRESERVED |
| **m6_kappa gate verdict** | HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL FINAL (8 paths exhausted) | ✅ PRESERVED per Wf-377-A1 + Wf-371-A1 |

**One-line verdict**: Wf-381-A4 finalizes TODO-206 m6_kappa metric redefinition by
**creating the canonical source brief** at `TODO/pending/206_m6_kappa_metric_redefinition.md`
(this was the missing canonical pending file — `TODO/03_plans/METRIC_LIFT_STRATEGY_v2.md`
§6 row 381 listed it as "already partial" but no on-disk source brief existed before
Wf-381-A4) + verifying the 4 SHIPPED reframe docs match the task brief §1 spec (8/7/8/9 =
32 total sections) + confirming all 3 gap conditions already FILLED by Wf-380-A7 + adding
14 NEW tests that validate this aggregation. m6_kappa remains
HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL FINAL (7 architectural + 1 metric-redesign =
8 paths).

---

## §1 Per-doc verification status

### §1.1 `TODO/pending/206_m6_kappa_metric_redefinition.md` (Wf-381-A4 NEW)

- **Size on disk:** ~10 KB (~165 LOC).
- **Section count:** 5 (§0 TL;DR + §1 Per-doc status + §2 Gaps filled + §3 Final verdict
  + §4 Cross-references).
- **Status:** ✅ NEW — created by Wf-381-A4 as the canonical TODO-206 source brief.

### §1.2 `docs/PAPER_S6_M6_KAPPA_REFRAME.md` (Wf-375-A4 SHIPPED)

- **Size on disk:** 32,428 bytes (~520 LOC).
- **Section count:** 8 (§0 TL;DR + §1 Why fails + §2 Recommended framing + §3 LaTeX §6
  snippet + §4 Cross-references + §5 Wall + §6 Standing rules + §7 Final state).
- **LaTeX §3 block:** ready-to-paste LaTeX for `paper/sections/06_limitations_v4.tex`
  §6.1.6 with 7-path inventory verbatim.
- **Status:** ✅ SHIPPED + Wf-380-A7 cross-linkage gap closed.

### §1.3 `docs/SECTION_3_METRIC_REFRAME_TEMPLATE.md` (Wf-376-A4 SHIPPED)

- **Size on disk:** 30,757 bytes (~490 LOC).
- **Section count:** 7 (§0 TL;DR + §1 Why fails + §2 Recommended §3 lineup + §3 LaTeX §3
  snippet + §4 USER_OK-gate status + §5 Cross-references + §6 How to use template).
- **LaTeX §3 block:** ready-to-paste LaTeX for `paper/sections/03_main_results_v4.tex`
  with 4-metric lineup (m1_auc + m2 + m3 + m15) + m6_kappa demoted.
- **Status:** ✅ SHIPPED + Wf-380-A7 cross-linkage gap closed.

### §1.4 `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` (Wf-377-A1 SHIPPED)

- **Size on disk:** 36,022 bytes (~600 LOC).
- **Section count:** 8 (§0 TL;DR + §1 Why fails (deeper fix) + §2 Literature review +
  §3 Empirical validation + §4 Recommended replacement + §5 Cross-references + §6 Final
  state + §7 Standing rules).
- **Empirical validation:** 16 invariants × n=5460 MetalCytoToxDB labelled pIC50;
  top-3 alternatives (MolLogP +0.1999, MorganDensity_r2 +0.1644, Chi3v +0.1541) — all
  WEAK band per Cohen 1988.
- **Status:** ✅ SHIPPED + Wf-380-A7 self-reference added.

### §1.5 `docs/M6_KAPPA_FINAL_VERDICT.md` (Wf-365-A3 SHIPPED)

- **Size on disk:** 17,987 bytes (~285 LOC).
- **Section count:** 9 (§1 Per-attempt + §2 Why 0.6 fails + §3 What lifted + §4 What
  remains + §5 Forward-looking + §6 Paper §6 entry prep + §7 Cross-references + §8
  Standing rules + §9 Wall).
- **§7 cross-references:** includes Wf-335-M1 + Wf-359-A1 + Wf-361-B1 + Wf-361-B2 +
  Wf-364-A2 + Wf-365-A3 + Wf-375-A4 + Wf-376-A4 + Wf-377-A1 + Wf-380-A7 (all 4 reframe
  Wfs cross-referenced via Wf-380-A7).
- **Status:** ✅ SHIPPED + Wf-380-A7 cross-linkage gap closed.

---

## §2 Gaps filled (per Wf-381-A4 task brief §3)

### §2.1 PROJECTED lift envelope (per Wf-377-A1 §3)

- **Status:** ✅ ALREADY FILLED — `docs/M6_KAPPA_REDESIGN_PROPOSAL.md §3.2` contains the
  cohort-level ranking of 16 invariants with concrete Pearson r + Spearman ρ values.
  Top-3 alternatives (MolLogP +0.1999, MorganDensity_r2 +0.1644, Chi3v +0.1541) are
  quantified with bit-for-bit reproducibility.
- **Gap-fill action:** None required (Wf-377-A1 itself filled).

### §2.2 LaTeX template (per task brief gap-fill spec)

- **Status:** ✅ ALREADY FILLED — 3 of 4 docs contain ready-to-paste LaTeX with
  `\begin{itemize}` blocks:
  - `docs/PAPER_S6_M6_KAPPA_REFRAME.md §3` for `paper/sections/06_limitations_v4.tex`
    §6.1.6 (7-path inventory verbatim)
  - `docs/SECTION_3_METRIC_REFRAME_TEMPLATE.md §3` for
    `paper/sections/03_main_results_v4.tex` (4-metric headline lineup)
  - `docs/M6_KAPPA_REDESIGN_PROPOSAL.md §4.4` for
    `paper/sections/06_limitations_v4.tex` §6.1.7 (metric-redesign layer extending §6.1.6)
- **Gap-fill action:** None required (3 docs already filled).

### §2.3 Cross-references (per task brief gap-fill spec)

- **Status:** ✅ ALREADY FILLED by Wf-380-A7:
  - `§4.8 "Wf-380-A7 finalization cross-linkage"` added to PAPER_S6 referencing Wf-376-A4
    + Wf-377-A1 + Wf-380-A7.
  - `§5.4 "Wf-380-A7 finalization cross-linkage"` added to SECTION_3 referencing Wf-375-A4
    + Wf-377-A1 + Wf-380-A7.
  - Wf-380-A7 self-reference added to `§5.1 source files` in REDESIGN.
  - Wf-375-A4 + Wf-376-A4 + Wf-377-A1 + Wf-380-A7 rows added to `§7 cross-references`
    table in FINAL_VERDICT.
- **Gap-fill action:** None required (Wf-380-A7 already filled).

---

## §3 Test status

### §3.1 NEW test file: `molmetal/tests/test_wf381a4_m6kappa_finalize.py`

- **Lines:** ~290 LOC
- **Test class:** `TestW381A4M6KappaFinalize` (14 test methods)
- **Wall time:** 0.002 s (CPU-only)
- **Pass rate:** 14/14 = **100% PASS**

### §3.2 Per-test verdict

| # | Test | Verifies | Result |
|---|---|---|---|
| 1 | `test_pending_file_exists` | TODO-206 pending file exists + size > 5000 bytes | ✅ PASS |
| 2 | `test_pending_file_sections` | TODO-206 pending file has 5 section headers (§0-§4) | ✅ PASS |
| 3 | `test_pending_file_user_voice_relay` | TODO-206 contains user-voice-answer relay + METRIC_LIFT_STRATEGY cross-ref | ✅ PASS |
| 4 | `test_all_4_docs_exist` | All 4 docs on disk + size > 5000 bytes | ✅ PASS |
| 5 | `test_paper_s6_sections` | PAPER_S6 §0-§7 = 8 section headers present | ✅ PASS |
| 6 | `test_section_3_sections` | SECTION_3 §0-§6 = 7 section headers present | ✅ PASS |
| 7 | `test_redesign_sections` | REDESIGN §0-§7 = 8 section headers present | ✅ PASS |
| 8 | `test_final_verdict_sections` | FINAL_VERDICT §1-§9 = 9 section headers present | ✅ PASS |
| 9 | `test_8_path_inventory_aggregates` | 8 unique Wf paths present across all 4 docs + pending brief (Wf-335-M1 + Wf-359-A1 + Wf-361-B1 + Wf-361-B2 + Wf-364-A2 + Wf-365-A3 + Wf-371-A1 + Wf-377-A1) | ✅ PASS |
| 10 | `test_cross_refs_present` | 19 cross-reference invariants (PAPER_S6 → Wf-376-A4/Wf-377-A1/Wf-365-A3/Wf-371-A1; SECTION_3 → Wf-377-A1/Wf-365-A3/Wf-371-A1/Wf-375-A4; REDESIGN → Wf-375-A4/Wf-371-A1/Wf-365-A3; FINAL_VERDICT → Wf-375-A4/Wf-376-A4/Wf-377-A1/Wf-380-A7; TODO-206 → all 4 + Wf-381-A4) | ✅ PASS |
| 11 | `test_section_3_lineup_invariant` | Each doc contains DROP/demote m6_kappa recommendation + m1_auc + m2 + m3 + m15 headline references | ✅ PASS |
| 12 | `test_top3_supplementary_promoted` | REDESIGN contains MolLogP + MorganDensity_r2 + Chi3v (top-3 alternatives) | ✅ PASS |
| 13 | `test_latex_template_present` | PAPER_S6 + SECTION_3 contain `\begin{itemize}` LaTeX template blocks | ✅ PASS |
| 14 | `test_no_paper_touches` | Wf-381-A4 deliverables do NOT live under `paper/` (per AGENTS.md §5.3 + TODO-84) | ✅ PASS |

### §3.3 Test file conventions followed

- Path: `molmetal/tests/test_wf381a4_m6kappa_finalize.py` (per Wf-380-A7 +
  Wf-377-A1 + Wf-371-A1 + Wf-365-A3 convention).
- Style: `unittest.TestCase` (matches `test_wf380a7_m6kappa_finalize.py` baseline).
- Docstring header with date + author + CPU-only + AGENTS.md §5.3 marker.
- Imports: `re`, `unittest`, `pathlib` (no torch required).
- `REPO_ROOT = Path(__file__).resolve().parents[2]` (matches project convention).
- Runnable via `python -m unittest molmetal.tests.test_wf381a4_m6kappa_finalize`
  (bypasses conftest torch requirement — mirrors Wf-380-A7 test approach).

---

## §4 Cross-references (Wf-381-A4 finalization inventory)

### §4.1 Aggregated 5-doc lineage (m6_kappa reframe surface + source brief)

| # | Doc | Workflow | Sections | Role | Wall |
|---:|---|---|---:|---|---:|
| 1 | `docs/M6_KAPPA_FINAL_VERDICT.md` | **Wf-365-A3** | 9 | ORIGINAL VERDICT — 5 architectural paths aggregated | ~7 min CPU |
| 2 | `docs/PAPER_S6_M6_KAPPA_REFRAME.md` | **Wf-375-A4** | 8 | §6.1.6 LaTeX + 7-path inventory | ~15-20 min CPU |
| 3 | `docs/SECTION_3_METRIC_REFRAME_TEMPLATE.md` | **Wf-376-A4** | 7 | §3 main-result LaTeX + 4-metric headline | ~5 min CPU |
| 4 | `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` | **Wf-377-A1** | 8 | §6.1.7 LaTeX + 8th path (metric redesign) | ~15 min CPU |
| 5 | `TODO/pending/206_m6_kappa_metric_redefinition.md` | **Wf-381-A4** (NEW) | 5 | SOURCE BRIEF — canonical pending file | ~15-20 min CPU |
| 6 | `TODO/completed/380a7_m6kappa_finalize.md` | **Wf-380-A7** (predecessor) | 5 | PRIOR FINALIZATION — 4-doc aggregation + 1 NEW test | ~20 min CPU |
| 7 | `TODO/completed/381a4_m6kappa_finalize.md` | **Wf-381-A4** (this) | 5 | FINALIZATION — 4-doc + pending brief + 14 NEW tests | ~15 min CPU |

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

**Verdict aggregate:** m6_kappa gate PRESERVED HONEST_NEGATIVE_PARTIAL_STRUCTURAL
FINAL FINAL. No architectural attempt lifted aug_mean to 0.6 (best +0.0018 above
baseline). No metric redesign surfaced a MODERATE-band cohort-level correlate (best
MolLogP +0.1999 = WEAK). 3 per-metal MODERATE-band discoveries (Rh BalabanJ +0.3953, Rh
MolLogP −0.3416, Re MolLogP +0.3363) cited honestly but do not save m6_kappa as §3
main-result metric.

### §4.3 Tier-2 paper §3 main-result line-up (final, per Wf-381-A4)

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
> 7 architectural paths + 1 metric-redesign path exhausted (Wf-335-M1 + Wf-359-A1 +
> Wf-361-B1 + Wf-361-B2 + Wf-364-A2 + Wf-365-A3 + Wf-371-A1 + Wf-377-A1). Best
> augmented_mean 0.4737 (Wf-364-A2 joint) vs 0.6 threshold = −0.1263 gap. Structural
> CV-R² ceiling LIFTED to +0.6814 (chemprop+PLL MEASURED). MetalCytoToxDB NEGATIVE_WEAK
> (Pearson r=0.081, Spearman ρ=0.111). Drop from Tier-2 §3 main-result table; keep as
> §6 honest-limitations complementary heuristic. Per Wf-371-A1 §3.5 + Wf-365-A3 §5.2 +
> Wf-377-A1 §4.2.

### §4.5 User voice answer relay

Per user verbatim 2026-09-29 (Wf-381-A4 task brief relay):

> todo目录里没有被刚才启动的workflow覆盖的任务也启动额外的ultracode去做
> (launch additional ultracode for TODO items not covered by the workflows just launched)

**Wf-381-A4 covers TODO-206 m6_kappa metric redefinition** which was listed as
"already partial" in `TODO/03_plans/METRIC_LIFT_STRATEGY_v2.md` §6 row 381 but did NOT
have a canonical `TODO/pending/206_*.md` source brief on disk. This wave closes that
gap by creating the canonical pending file + verifying the 4 SHIPPED reframe docs.

---

## §5 Final state (Wf-381-A4 verdict)

| Item | Value | Status |
|---|---|---|
| **Canonical TODO-206 source brief** | `TODO/pending/206_m6_kappa_metric_redefinition.md` (5 sections, ~10 KB) | ✅ NEW |
| **4 docs verified SHIPPED** | PAPER_S6 + SECTION_3 + REDESIGN + FINAL_VERDICT | ✅ |
| **Section counts match task brief** | 8 / 7 / 8 / 9 = 32 sections | ✅ |
| **Gaps filled** | 0 NEW (all 3 gap-conditions already FILLED by Wf-380-A7) | ✅ |
| **NEW tests** | 14/14 PASS in 0.002 s | ✅ |
| **8-path inventory aggregated** | 7 architectural + 1 metric-redesign = 8 paths | ✅ |
| **§3 main-result line-up invariant** | m1 + m2 + m3 + m15 RETAIN + top-3 PROMOTE + m6 DROP | ✅ |
| **m6_kappa gate verdict** | HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL FINAL (8 paths) | ✅ |
| **Code patches** | 0 NEW source files; 1 NEW docs (pending brief) + 1 NEW test file | ✅ docs-only |
| **GPU launches** | 0 | ✅ per AGENTS.md §5.1 |
| **Paper/* touches** | 0 | ✅ per AGENTS.md §5.3 + TODO-84 |
| **Pre-flight v3 status** | 92/0/2/94 PRESERVED (NOT re-run by Wf-381-A4) | ✅ |
| **Wall (this Wf-381-A4 docs-only closeout)** | ~15 min CPU-only | MEASURED |

**The m6_kappa Tier-2 paper reframe is now COMPLETE + FINALIZED + CANONICAL SOURCE BRIEF
CREATED.** Wf-381-A4 closes the TODO-206 m6_kappa metric redefinition task by:
(1) creating the canonical `TODO/pending/206_*.md` source brief that was missing on disk,
(2) verifying the 4 SHIPPED reframe docs (Wf-375-A4 + Wf-376-A4 + Wf-377-A1 + Wf-365-A3)
all match the task brief §1 spec (8/7/8/9 = 32 sections total),
(3) confirming all 3 gap conditions already FILLED by Wf-380-A7 predecessor wave
(PROJECTED lift envelope + LaTeX template + cross-references),
(4) adding 14 NEW tests verifying all 4 docs exist + 5-section canonical structure +
cross-references + 8-path inventory + §3 line-up invariant + LaTeX template presence +
0 paper/* touches.
Paper §3 + §6 v4 conversion requires USER_OK per TODO-84 + AGENTS.md §5.3.

---

**END OF Wf-381-A4 CLOSEOUT** — Owner Hugo Lee + Wf-381-A4 ultracode; cross-ref
`party-twenty-spirit §3.5` + AGENTS.md §5.1 + §5.2 + §5.3 + §5.4.
