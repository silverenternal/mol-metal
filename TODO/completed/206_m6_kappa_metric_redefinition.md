# 206 — m6_kappa Metric Redefinition (Drop from Tier-2 §3 Main-Result; Keep as §6 Honest-Limitations)

**Document ID:** TODO-206 / Wf-381-A4-M6KAPPA-PENDING
**Version:** 1.0
**Authoring date:** 2026-09-30
**Owner:** Hugo Lee + Wf-381-A4 ultracode
**Trigger:** Per `TODO/03_plans/METRIC_LIFT_STRATEGY_v2.md` §4.7 + §6 row 381 — m6_kappa metric redefinition (drop as §3 main-result; Wf-375-A4 partial)
**Status:** SHIPPED-PARTIAL → FINALIZED via Wf-381-A4 (this wave)

---

## §0 TL;DR

**问题**: m6_kappa Shape Kappa Index remains **HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL FINAL**
across **8 architectural + metric-redesign paths** (Wf-335-M1 + Wf-359-A1 + Wf-361-B1 +
Wf-361-B2 + Wf-364-A2 + Wf-365-A3 + Wf-371-A1 + Wf-377-A1). Best augmented_mean MEASURED
**0.4737** (Wf-364-A2 joint) vs 0.6 threshold = **−0.1263 gap** preserved. Structural
CV-R² ceiling **LIFTED** to +0.6814 CEILING_BREAK (chemprop+PLL Wf-335-M1) but threshold-
reaching axis NOT lifted. Drop m6_kappa from Tier-2 paper §3 main-result; promote 4
CEILING_BREAK anchors (m1_auc + m2 + m3 + m15); keep m6_kappa as §6 honest-limitations
complementary heuristic.

**Solution**: Aggregate 4 SHIPPED m6_kappa reframe docs into single canonical surface:

| # | Doc | Workflow | Sections | Role |
|---:|---|---|---:|---|
| 1 | `docs/PAPER_S6_M6_KAPPA_REFRAME.md` | Wf-375-A4 | 8 | §6.1.6 LaTeX + 7-path inventory |
| 2 | `docs/SECTION_3_METRIC_REFRAME_TEMPLATE.md` | Wf-376-A4 | 7 | §3 main-result LaTeX + 4-metric headline lineup |
| 3 | `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` | Wf-377-A1 | 8 | §6.1.7 LaTeX + 8th path (metric redesign) |
| 4 | `docs/M6_KAPPA_FINAL_VERDICT.md` | Wf-365-A3 | 9 | ORIGINAL verdict — 5 architectural paths aggregated |
| 5 | `TODO/pending/206_*.md` (this) | Wf-381-A4 | 5 | SOURCE BRIEF + canonicalization |

**Wall**: ~15-20 min CPU-only (no GPU; no paper/* touches).

---

## §1 Per-doc status (verification)

### §1.1 `docs/PAPER_S6_M6_KAPPA_REFRAME.md` (Wf-375-A4 SHIPPED)

- **Size on disk:** 32,428 bytes (~520 LOC).
- **Section count:** 8 (§0 TL;DR + §1 Why fails + §2 Recommended framing + §3 LaTeX §6
  snippet + §4 Cross-references + §5 Wall + §6 Standing rules + §7 Final state).
- **Wf-381-A4 verification:** ✅ All 8 section headers present; ready-to-paste LaTeX §6.1.6
  snippet present with `\begin{itemize}` block.
- **Status:** SHIPPED + Wf-380-A7 cross-linkage gap closed.

### §1.2 `docs/SECTION_3_METRIC_REFRAME_TEMPLATE.md` (Wf-376-A4 SHIPPED)

- **Size on disk:** 30,757 bytes (~490 LOC).
- **Section count:** 7 (§0 TL;DR + §1 Why fails + §2 Recommended §3 lineup + §3 LaTeX §3
  snippet + §4 USER_OK-gate status + §5 Cross-references + §6 How to use template).
- **Wf-381-A4 verification:** ✅ All 7 section headers present; ready-to-paste LaTeX §3
  main-result lineup present with 4-metric headline + m6_kappa demoted.
- **Status:** SHIPPED + Wf-380-A7 cross-linkage gap closed.

### §1.3 `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` (Wf-377-A1 SHIPPED)

- **Size on disk:** 36,022 bytes (~600 LOC).
- **Section count:** 8 (§0 TL;DR + §1 Why fails (deeper fix) + §2 Literature review +
  §3 Empirical validation + §4 Recommended replacement + §5 Cross-references + §6 Final
  state + §7 Standing rules).
- **Wf-381-A4 verification:** ✅ All 8 section headers present; 16 invariants × n=5460
  MetalCytoToxDB validation table; top-3 alternatives (MolLogP +0.1999, MorganDensity_r2
  +0.1644, Chi3v +0.1541) all WEAK band per Cohen 1988.
- **Status:** SHIPPED + Wf-380-A7 self-reference added.

### §1.4 `docs/M6_KAPPA_FINAL_VERDICT.md` (Wf-365-A3 SHIPPED)

- **Size on disk:** 17,987 bytes (~285 LOC).
- **Section count:** 9 (§1 Per-attempt + §2 Why 0.6 fails + §3 What lifted + §4 What remains +
  §5 Forward-looking + §6 Paper §6 entry prep + §7 Cross-references + §8 Standing rules +
  §9 Wall).
- **Wf-381-A4 verification:** ✅ All 9 section headers present; §7 cross-references table
  includes Wf-335-M1 + Wf-359-A1 + Wf-361-B1 + Wf-361-B2 + Wf-364-A2 + Wf-365-A3 +
  Wf-375-A4 + Wf-376-A4 + Wf-377-A1 + Wf-380-A7 (all 4 reframe Wfs cross-referenced).
- **Status:** SHIPPED + Wf-380-A7 cross-linkage gap closed.

---

## §2 Gaps filled (per Wf-381-A4 task brief §3)

### §2.1 PROJECTED lift envelope (per Wf-377-A1 §3)

- **Status:** ✅ ALREADY FILLED — `docs/M6_KAPPA_REDESIGN_PROPOSAL.md §3.2` contains the
  cohort-level ranking of 16 invariants with concrete Pearson r + Spearman ρ values.
  Top-3 alternatives (MolLogP +0.1999, MorganDensity_r2 +0.1644, Chi3v +0.1541) are
  quantified with bit-for-bit reproducibility (sanity-check anchor vs Wf-325D canonical
  at 15 significant digits).
- **Gap-fill action:** None required (Wf-377-A1 itself filled).

### §2.2 LaTeX template (begin-itemize blocks)

- **Status:** ✅ ALREADY FILLED — 3 of 4 docs contain ready-to-paste LaTeX with
  `\begin{itemize}` blocks:
  - `docs/PAPER_S6_M6_KAPPA_REFRAME.md §3` for `paper/sections/06_limitations_v4.tex`
    §6.1.6 (7-path inventory verbatim)
  - `docs/SECTION_3_METRIC_REFRAME_TEMPLATE.md §3` for
    `paper/sections/03_main_results_v4.tex` (4-metric headline lineup)
  - `docs/M6_KAPPA_REDESIGN_PROPOSAL.md §4.4` for
    `paper/sections/06_limitations_v4.tex` §6.1.7 (metric-redesign layer extending §6.1.6)
- **Gap-fill action:** None required (3 docs already filled). `docs/M6_KAPPA_FINAL_VERDICT.md
  §6` contains an earlier draft snippet preserved as baseline.

### §2.3 Cross-references (Wf-371-A1 + 7-path inventory table)

- **Status:** ✅ ALREADY FILLED by Wf-380-A7:
  - Wf-375-A4 row added to `docs/M6_KAPPA_FINAL_VERDICT.md §7` cross-references table.
  - Wf-376-A4 row added to FINAL_VERDICT §7.
  - Wf-377-A1 row added to FINAL_VERDICT §7.
  - Wf-380-A7 row added to FINAL_VERDICT §7 (Wf-381-A4 inherits this).
  - 8-path inventory aggregated across the 4 docs (per Wf-380-A7 closeout §4.2).
- **Gap-fill action:** None required (Wf-380-A7 already filled).

---

## §3 Final verdict

| Item | Value | Status |
|---|---|---|
| **4 docs verified SHIPPED** | PAPER_S6 + SECTION_3 + REDESIGN + FINAL_VERDICT | ✅ |
| **Section counts match task brief** | 8 / 7 / 8 / 9 = 32 sections | ✅ |
| **Gaps filled** | 0 NEW (all 3 gap-conditions already FILLED by Wf-380-A7) | ✅ |
| **TODO-206 source brief** | THIS FILE — created by Wf-381-A4 | ✅ NEW |
| **NEW test** | 1 NEW test PASS at `molmetal/tests/test_wf381a4_m6kappa_finalize.py` | ✅ NEW |
| **8-path inventory** | 7 architectural + 1 metric-redesign = 8 paths | ✅ |
| **§3 main-result line-up invariant** | m1 + m2 + m3 + m15 RETAIN + top-3 PROMOTE + m6 DROP | ✅ |
| **m6_kappa gate verdict** | HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL FINAL (8 paths) | ✅ |
| **Code patches** | 0 NEW source files | ✅ docs-only |
| **GPU launches** | 0 | ✅ per AGENTS.md §5.1 |
| **Paper/* touches** | 0 | ✅ per AGENTS.md §5.3 + TODO-84 |
| **Pre-flight v3 status** | 92/0/2/94 PRESERVED at Wf-370-B1 baseline (NOT re-run by Wf-381-A4) | ✅ |

**One-line verdict**: Wf-381-A4 finalizes TODO-206 m6_kappa metric redefinition by
creating the canonical source brief (this pending file) + verifying the 4 SHIPPED docs
match the task brief §1 spec (8/7/8/9 sections = 32 total) + confirming all 3 gap
conditions already FILLED by Wf-380-A7 (PROJECTED lift envelope + LaTeX template + cross-
references) + adding 1 NEW test that validates this aggregation. m6_kappa remains
HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL FINAL — 8 paths exhausted.

---

## §4 Cross-references

### §4.1 Source brief lineage (m6_kappa reframe surface)

| # | Source | Type | Sections | Wall | Status |
|---:|---|---|---:|---|---|
| 1 | `TODO/03_plans/METRIC_LIFT_STRATEGY_v2.md` §6 row 381 | Canonical TODO list | n/a | n/a | SHIPPED |
| 2 | `docs/M6_KAPPA_FINAL_VERDICT.md` | Wf-365-A3 verdict doc | 9 | ~7 min CPU | SHIPPED |
| 3 | `docs/PAPER_S6_M6_KAPPA_REFRAME.md` | Wf-375-A4 paper §6 LaTeX | 8 | ~15-20 min CPU | SHIPPED |
| 4 | `docs/SECTION_3_METRIC_REFRAME_TEMPLATE.md` | Wf-376-A4 paper §3 LaTeX | 7 | ~5 min CPU | SHIPPED |
| 5 | `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` | Wf-377-A1 metric redesign | 8 | ~15 min CPU | SHIPPED |
| 6 | `TODO/pending/206_*.md` (this) | Wf-381-A4 source brief | 5 | ~15-20 min CPU | NEW |
| 7 | `TODO/completed/380a7_m6kappa_finalize.md` | Wf-380-A7 prior finalization | 5+ | ~20 min CPU | SHIPPED |
| 8 | `TODO/completed/381a4_m6kappa_finalize.md` | Wf-381-A4 closeout | 5 | ~10 min CPU | NEW |

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

### §4.3 Tier-2 paper §3 main-result line-up (final, per Wf-381-A4)

| Section | Metric | Status | Source |
|---|---|---|---|
| §3 headline (4) | **m1_auc (MetalCytotoxDB Ru 5-seed A2780 cisplatin)** | **KEEP** | `metrics/by_metric/metalcytotox_ru_auc.json` MEASURED_PARTIAL_LIFT |
| §3 headline (4) | **m2 R² (metalcytotox_combined)** | **KEEP** | `metrics/by_metric/metalcytotox_combined_r2.json::wf335m2_ceiling_break` CEILING_BREAK R²=+0.3196 lift +0.2141 |
| §3 headline (4) | **m3 Vina (cisplatin reference)** | **KEEP** | `metrics/by_metric/vina_kcal_per_mol.json::cisplatin_reference_mean_kcal_per_mol` −3.103 kcal/mol MEASURED (positive control). The −7.782 "V3 cohort median" is **PROJECTED** arithmetic, not a docking run — no such key has ever existed in the JSON (corrected 2026-10-20). |
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

**END OF TODO-206 SOURCE BRIEF** — Owner Hugo Lee + Wf-381-A4 ultracode; cross-ref
`party-twenty-spirit §3.5` + AGENTS.md §5.1 + §5.2 + §5.3 + §5.4 + TODO-84. The 4
m6_kappa reframe docs are SHIPPED with correct section counts (8/7/8/9 = 32 sections
total). All 3 gap conditions from Wf-381-A4 task brief are already FILLED by Wf-380-A7
(predecessor wave). Wf-381-A4 contributes: (1) canonical `TODO/pending/206_*.md` source
brief + (2) 1 NEW test at `molmetal/tests/test_wf381a4_m6kappa_finalize.py` + (3)
`TODO/completed/381a4_m6kappa_finalize.md` closeout. m6_kappa remains
HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL FINAL across 8 paths.
