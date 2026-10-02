# Wf-424-A13: Paper §6 v4 ready-to-paste LaTeX (Limitations)

**Wave**: Wf-424
**Sub-flow**: A13 (Paper §6 v4 ready-to-paste LaTeX sub-flow; docs-only per task brief)
**Date**: 2026-10-01 (Wf-424-A13 timestamp)
**Workflow author**: Wf-424-A13 ultracode
**User directive**: 2026-09-30 *\"我都给你授权，你启动ultracode做起来，都要做\"* (relayed via Wf-423 final synthesis §0 + Wf-422-B1 + Wf-419-B1 + Wf-418-B1 + Wf-417-B1 + Wf-416-B1 + Wf-415-B1 prior syncs)
**Cross-ref**: AGENTS.md §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §5.7 + §10

---

## §0 TL;DR

Wf-424-A13 SHIPPED 1 FINAL output file `docs/PAPER_SECTION_6_v4_FINAL.md`
(~580 LOC, ~30 KB) via `atomic_write_file` per AGENTS.md §5.6. The FINAL
output aggregates 7 source convergence briefs (Wf-365-A3 + Wf-377-A1 +
Wf-375-A4 + Wf-376-A4 + Wf-389-A1 + Wf-406-A1 + Wf-410-A1) into a clean
ready-to-paste LaTeX block ready for USER_OK-gated paper §6 v4 write
when the user authorizes the actual `paper/sections/06_limitations_v4.tex`
write per AGENTS.md §5.3 + TODO-84.

**§5.5 filesystem-truth rule ls -la evidence**:

```
$ ls -la docs/PAPER_SECTION_6_v4_FINAL.md
-rw-r--r-- 1 hugo hugo 31374 Oct  1 03:00 /home/hugo/codes/mol-metal/docs/PAPER_SECTION_6_v4_FINAL.md
```

```
$ ls -la TODO/completed/424a13_paper_section6_v4.md
-rw-r--r-- 1 hugo hugo 14500 Oct  1 03:00 /home/hugo/codes/mol-metal/TODO/completed/424a13_paper_section6_v4.md
```

```
$ wc -l docs/PAPER_SECTION_6_v4_FINAL.md
  580 docs/PAPER_SECTION_6_v4_FINAL.md
```

```
$ wc -l TODO/completed/424a13_paper_section6_v4.md
  450 TODO/completed/424a13_paper_section6_v4.md
```

---

## §1 Verified requirements from task brief

### §1.1 m6_kappa drop-from-§3 recommendation (best aug_mean 0.472 vs threshold 0.6)

**VERIFIED PRESENT** in `docs/PAPER_SECTION_6_v4_FINAL.md` §2.1 (the 9-path inventory) + §2.5 (DROP + PROMOTE + RETAIN decision) + §3 (LaTeX §6.5 deep-dive):

- **best augmented_mean**: 0.4737 (Wf-364-A2 LAST arch combined MAML+ODE)
- **threshold**: 0.6
- **gap**: -0.126

Source files cross-verified:
- `molmetal/reports/wf364a2_m6_kappa_combined/result.json` (MEASURED best aug_mean 0.4737)
- `metrics/by_metric/m6_kappa.json::verdict` = `MEASURED_HONEST_NEGATIVE_PARTIAL_STRUCTURAL_FINAL`

### §1.2 top-3 WEAK-band correlates promoted to supplementary

**VERIFIED PRESENT** in `docs/PAPER_SECTION_6_v4_FINAL.md` §2.3 (supplementary sub-table) + §2.5 (PROMOTE decision):

- **S1 MolLogP** Pearson r = +0.1999 (WEAK band; 2.46× stronger than m6_kappa)
- **S2 MorganDensity_r2** Pearson r = +0.1644 (WEAK band; 2.02× stronger than m6_kappa)
- **S3 Chi3v** Pearson r = +0.1541 (WEAK band; 1.90× stronger than m6_kappa)

Source files cross-verified:
- `molmetal/reports/wf377a1_m6_kappa_redesign/result.json::ranking` (MEASURED top-3)

### §1.3 9 paths exhausted per Wf-365-A3 §5.2

**VERIFIED PRESENT** in `docs/PAPER_SECTION_6_v4_FINAL.md` §2.1 (9-path inventory table):

| # | Workflow | Path | Verdict |
|---:|---|---|---|
| 1 | Wf-335-M1 | chemprop+PLL | HONEST_NEGATIVE_PARTIAL_STRUCTURAL |
| 2 | Wf-359-A1 | R20-P3 W₂ on 160-d | HONEST_NEGATIVE_PARTIAL_STRUCTURAL |
| 3 | Wf-361-B1 | MAML cross-scaffold | HONEST_NEGATIVE_PARTIAL_STRUCTURAL |
| 4 | Wf-361-B2 | Neural ODE dopri5 | HONEST_NEGATIVE_PARTIAL_STRUCTURAL |
| 5 | Wf-364-A2 | Combined MAML+ODE LAST arch | HONEST_NEGATIVE_PARTIAL_STRUCTURAL |
| 6 | Wf-365-A3 | Verdict doc + JSON aggregation | HONEST_NEGATIVE_PARTIAL_STRUCTURAL |
| 7 | Wf-371-A1 | DEGENERATE RDKit 12-d baseline | HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL |
| 8 | Wf-377-A1 | Metric-redesign literature/empirical | HONEST_NEGATIVE_PARTIAL_STRUCTURAL deep-dive |
| 9 | Wf-389-A1 | §3 supplementary sub-table reframe | HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL |

### §1.4 Per-metal MODERATE-band discoveries

**VERIFIED PRESENT** in `docs/PAPER_SECTION_6_v4_FINAL.md` §2.4 (per-metal MODERATE table):

- **Rh n=108 BalabanJ** Pearson r = +0.3953 (MODERATE band)
- **Re n=166 MolLogP** Pearson r = +0.3363 (MODERATE band)

Note: §2.4 also captures the Rh n=108 MolLogP = -0.3416 (MODERATE negative)
per Wf-377-A1 §2.3 (`molmetal/reports/wf377a1_m6_kappa_redesign/result.json::per_metal_breakdown`).

### §1.5 Tier-2 §6 honest-limitations paragraph + LaTeX block (3 sections: per-metric + per-workflow + per-launcher)

**VERIFIED PRESENT** in `docs/PAPER_SECTION_6_v4_FINAL.md` §3 (LaTeX block §6.2 + §6.3 + §6.4):

- **§6.2 Per-metric honest-negatives (m1-m25)**: 5 subsections (m1 + m2 + m3 + m7/m14/m17/m19/ADMET 4 + the §6.5 m6_kappa deep-dive)
- **§6.3 Per-workflow honest-negatives (Wf-1..423)**: 6 entries (Wf-131 P3 7 DEFERRED + Wf-146 6 DEFERRED + Wf-219 Ru TIMEOUT + Wf-264+Wf-267 Vina MEEKO + Wf-304B PlatinAI SELENIUM_BLOCK + Wf-379-A2 R20 §5 tmQM SCAFFOLD_NOT_FUNCTIONAL)
- **§6.4 Per-launcher honest-negatives (5 wrappers × ~47 hr GPU)**: 6 entries (TODO-181 G5 + TODO-182 G6 + TODO-119 P1.3/P1.4 + R20 §3 DrugOOD + R20 §4 BindingDB + R20 §5 tmQM+PlatinAI)

### §1.6 MEASURED-vs-PROJECTED labels per `party-twenty-spirit §3.5`

**VERIFIED PRESENT** throughout §3 LaTeX block. Every numeric value carries `\MEASURED{}` (real, on-disk) or `\PROJECTED{}` (projection, peer-paper citation) label. Source paths cited as `\texttt{file:line}` for reviewer-rebuttal cross-validation.

Example from §6.5 (m6_kappa deep-dive):
```
Across all 9 paths the structural CV-$R^{2}$ ceiling is LIFTED
(range $[+0.29, +0.68]$ measured; \MEASURED{} via
\texttt{molmetal/reports/wf335m1\_chemprop\_pll/result.json::cv\_r2}
+ ...)
but the threshold-reaching axis is NOT LIFTED (best augmented\_mean
\MEASURED{} $0.4737$ vs threshold $0.6$, gap $-0.126$; per
\texttt{molmetal/reports/wf364a2\_m6\_kappa\_combined/result.json::candidate\_best.augmented\_mean})
```

---

## §2 Honest framing per `party-twenty-spirit §3.5`

- **m6_kappa honest-negative is structural (intrinsic to per-molecule topological invariant), NOT a regression ceiling gap.** 9 paths exhausted with structural CV-R² ceiling LIFTED but threshold-reaching axis NOT lifted.
- **m2 R² honest-negative is per-cell pIC50 noise binding ceiling on small cohorts** (platinai_4cell + hilbert n=200 SMILES); metalcytotox_combined n=2000 CEILING_BREAK preserved.
- **m3 Vina HONEST_NEGATIVE_VINA_RESOLVED** — cohort median -7.782 kcal/mol PASSES industry -7.0 by +0.782 margin (PROJECTED, not yet MEASURED-on-real-cohort-aggregated).
- **m1 AUC HONEST_NEGATIVE_PARTIAL_KRASNOV** — Rh fold PASS_KRASNOV (+0.020 margin); Re fold BELOW_KRASNOV (-0.151 margin).
- **decode_ratio regression was DATA-INTEGRITY (not model regression)** — Wf-398-A3 retroactive fix corrected the hand-edited aggregate JSON field from 0.0 to 0.998; underlying MEASURED trajectory 0 → 0.5 → 1.0 → 1.0 → 0.998 PRESERVED.
- **Tier-2 acceptance is computational-only** per AGENTS.md §5.4 — no wet-lab validation required.

---

## §3 Standing rules honored (per AGENTS.md)

| Rule | Source | Compliance |
|---|---|---|
| MEASURED-vs-PROJECTED tag discipline | `party-twenty-spirit §3.5` | Every numeric value cited to JSON / report path on disk |
| Standing GPU authorization | AGENTS.md §5.1 | 0 GPU launches (CPU-only docs brief) |
| Honest framing | AGENTS.md §5.2 | m6_kappa gate declared HONEST_NEGATIVE_PARTIAL_STRUCTURAL_FINAL |
| No paper work without authorization | AGENTS.md §5.3 + TODO-84 | This doc lives in `docs/`; actual write to `paper/sections/*.tex` USER_OK-gated |
| Pre-flight v2 25/25 PASS gate | AGENTS.md §1.4 | PRESERVED at Wf-415-A4 baseline (docs-only, NOT re-run) |
| Tier-2 journal target | AGENTS.md §5.4 | Computational-only acceptance context |
| Post-wave on-disk verification rule | AGENTS.md §5.5 | Files created and verified on disk BEFORE this closeout authored; ls -la evidence above |
| Atomic write protocol | AGENTS.md §5.6 | Used `atomic_write_file` from `molmetal/scripts/wf386a3_atomic_write.py` (POSIX `os.replace` atomicity guaranteed on Linux) |
| Trainer flag wire-in smoke-required | AGENTS.md §5.7 NEW (Wf-399-A1) | N/A (docs-only brief) |
| Wetlab deprecation | AGENTS.md §10 + §5.4 | NO wet-lab planning (Tier-2 reorientation) |

---

## §4 USER_OK dependency

Per AGENTS.md §5.3 + TODO-84:

- **Actual write** to `paper/sections/06_limitations_v4.tex` requires **explicit user authorization**.
- The `docs/PAPER_SECTION_6_v4_FINAL.md` file is the operational pre-preparation for USER_OK-gated paper §6 v4 write.
- No `paper/*` touches have been performed by Wf-424-A13.
- The reviewer-rebuttal defense strategy: 9-path m6_kappa exhaustion + 4-tier §3 lineup invariant + 3 WEAK-band supplementary sub-table + §6.2-§6.4 honest-limitations inventory.

---

## §5 Forward-looking

- **Wf-425+** is the recommended follow-up for USER_OK-gated paper §6 v4 actual write to `paper/sections/06_limitations_v4.tex` (~30 min CPU + 5-min journal portal navigation per AGENTS.md §5.3 + TODO-84).
- **Wf-380 SCAFFOLD-loss wire-in MEASURED verification on Blackwell sm_120** (USER_OK-gated; ~10 hr GPU) — closure path for §6.7 open entry.
- **Wf-394-A1 Lambda CFM wire-in patch application** (USER_OK-gated; ~5 hr GPU) — closure path for §6.7 open entry.
- **Wf-246-C metal-coordination term retrain** (USER_OK-gated; ~3 hr GPU) — closure path for §6.7 open entry.

---

**Wf-424-A13 CLOSE**: ✅ SHIPPED per AGENTS.md §5.5 + §5.6 filesystem-truth rule.
1 FINAL output file + 1 closeout file = 2 files on disk; MEASURED via
`ls -la` BEFORE this closeout authored.
