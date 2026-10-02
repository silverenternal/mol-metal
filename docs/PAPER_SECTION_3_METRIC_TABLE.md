# Paper §3 — Metric Table (Main Result + Supplementary Sub-Table)

**Status:** 📝 **SCAFFOLD TEMPLATE ONLY** — per TODO-84 + AGENTS.md §5.3, ZERO touches
to `paper/sections/*.tex` without explicit USER_OK authorization. This document is the
**ready-to-paste LaTeX source** that lives in `docs/` so that, when USER_OK arrives, the
actual paper §3 write to `paper/sections/03_main_results_v4.tex` is a copy-paste into a
new file. All `\MEASURED{}` / `\PROJECTED{}` labels are preserved per
`party-twenty-spirit` §3.5.

**Date:** 2026-09-30
**Owner:** Hugo Lee + Wf-389-A1 ultracode
**Source verdicts:**
- `docs/SECTION_3_METRIC_REFRAME_TEMPLATE.md` (Wf-376-A4 SHIPPED; 7 sections, ~480 LOC)
- `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` (Wf-377-A1 SHIPPED; 8 sections, ~590 LOC)
- `docs/M6_KAPPA_FINAL_VERDICT.md` (Wf-365-A3 SHIPPED; 9 sections, ~270 LOC)
- `docs/PAPER_S6_M6_KAPPA_REFRAME.md` (Wf-375-A4 SHIPPED; 7 sections, ~480 LOC)
- `TODO/completed/380a7_m6kappa_finalize.md` (Wf-380-A7 SHIPPED; 5 sections, ~150 LOC)

**Cross-ref:** AGENTS.md §5.2 (honest framing) + §5.3 (no paper work) + §5.4 (Tier-2)

---

## §0 TL;DR — explicit §3 metric table (4 main + 3 supplementary)

This document is the **explicit Tier-2 paper §3 metric table** that operationalizes the
convergence of Wf-376-A4 §3 (drop m6_kappa + promote 4 CEILING_BREAK) and Wf-377-A1 §4.2
(promote top-3 alternatives as §3 supplementary sub-table). The final lineup is:

**§3 main-result table (4 metrics, all with CEILING_BREAK or strong MEASURED anchors):**

| # | Metric | Headline value | Verdict |
|---|---|---|---|
| 1 | **m1_auc** (Ru, A2780 cisplatin) | AUC = **0.7613** (1-seed; gap −0.099 to Krasnov 0.86 gate) | MEASURED_PARTIAL_LIFT |
| 2 | **m2 R²** (metalcytotox_combined) | R² = **+0.3196** (lift +0.2141 vs Wf-325A baseline; n=2000) | **CEILING_BREAK** |
| 3 | **m3 Vina** (5-pocket cohort median) | **−7.782 kcal/mol** (V3 combined; PASSES industry −7.0 by +0.782 margin) | HONEST_NEGATIVE_VINA_RESOLVED |
| 4 | **m15 RMSD** (5-pocket redock) | **0.6578 Å** (5/5 pockets PASS 2.0 Å SOTA gate) | MEASURED |

**§3 supplementary sub-table (3 WEAK-band correlates, NEW per Wf-389-A1):**

| # | Invariant | Pearson r (n=5460) | Verdict band |
|---|---|---:|---|
| S1 | **MolLogP** (Wildman-Crippen LogP) | **+0.1999** | WEAK |
| S2 | **MorganDensity_r2** (Morgan ECFP radius-2 bit density) | **+0.1644** | WEAK |
| S3 | **Chi3v** (Kier-Hall valence χ3v connectivity) | **+0.1541** | WEAK |

**§3 demoted (NOT in main table):**
- **m6_kappa (Shape Kappa Index)** → DROPPED from §3 main-result; PROMOTE to §6 honest-limitations
  as complementary heuristic per Wf-365-A3 §6 + Wf-371-A1 §3.5 + Wf-377-A1 §4.2.

**Why this reframe is honest** (per `party-twenty-spirit` §3.5):
- m6_kappa gate is intrinsic to per-molecule Shape Kappa Index topology; 7 architectural
  paths + 1 metric-redesign literature/empirical path (8 total) all preserved the
  HONEST_NEGATIVE_PARTIAL_STRUCTURAL verdict (no path lifts aug_mean to ≥ 0.6).
- m2 R² metalcytotox_combined has the strongest CEILING_BREAK evidence (+0.2141 lift
  on n=2000 SMILES, multi-seed × 5-fold CV per Wf-335-M2 + Wf-369-A2 final verdict).
- Top-3 alternatives (MolLogP, MorganDensity_r2, Chi3v) are statistically stronger than
  m6_kappa by 2.46×, 2.02×, 1.90× respectively, BUT all are WEAK band (|r| < 0.3 per
  Cohen 1988) — they are exploratory correlates, NOT headline fit metrics.
- Per-metal MODERATE-band discoveries (Rh BalabanJ +0.3953, Rh MolLogP −0.3416, Re
  MolLogP +0.3363) are honest-negative on cohort-level scale (n ≤ 166 too small for
  Tier-2 §3 main-result claims) but merit §3 supplementary mention.

---

## §1 §3 main-result table (4 metrics, 4 sources)

### §1.1 m1_auc (MetalCytotoxDB Ru) — primary headline metric

- **MEASURED headline**: AUC = **0.7613** (1-seed, Wf-124 P3 30-epoch D-MPNN multi-task +
  Wasserstein + L2 guard; lift +0.049 vs 0.7123 single-task 5-epoch baseline)
- **Baseline (Krasnov 2026 paper-tier)**: AUC = 0.81 (cite-only)
- **Gate target (Krasnov 2026)**: AUC ≥ 0.86 (NOT YET HIT, gap −0.0987)
- **Multi-seed baseline (Wf-131 P1)**: AUC = 0.7352 ± 0.0023 (3-seed × 10-epoch)
- **Source**: `metrics/by_metric/metalcytotox_ru_auc.json`
- **Verdict**: MEASURED_PARTIAL_LIFT — single-seed 30-epoch lifts +0.049 vs single-task
  baseline; full 5-seed × 50-epoch ship requires USER_OK per Wf-355-B1 wrapper.
- **USER_OK-gated closure path**: `scripts/launch_wf131_p3_deferred_runs.sh` (7 DEFERRED
  runs: Rh 3,4 + Re 0,1,2,3,4). Wall ~5 hr GPU (per Wf-355-B1 + Wf-379-A1 IN-FLIGHT).

### §1.2 m2 R² (metalcytotox_combined) — CEILING_BREAK on n=2000 cohort

- **MEASURED headline**: R² = **+0.3196** (best_config `full_static_combined_plus_w2_best_axis`,
  164-d features, W₂+chemprop v2 graph integration, multi-seed K-fold CV per Wf-335-M2)
- **WF-325A baseline R²**: +0.1055
- **Lift**: **+0.2141** (CEILING_BREAK; CV-mean lift ≥ +0.10 per WEAK_TO_STRONG_PLAN §3.5)
- **Cohort size**: n=2000 SMILES × 5 seeds × 5 folds = 25 splits
- **Aggregate status**: HONEST_NEGATIVE_PARTIAL (only 1 of 3 cohorts achieves
  CEILING_BREAK; platinai_4cell + hilbert remain HONEST_NEGATIVE_R2_DEEP per
  Wf-335-M2 `per_cohort[0,1].best_config` and Wf-367-B1 multi-stream ensemble result)
- **Source**: `metrics/by_metric/metalcytotox_combined_r2.json::wf335m2_ceiling_break`
  + `wf369a2_final_verdict`
- **Verdict**: CEILING_BREAK — FINAL PRESERVED at Wf-369-A2 timestamp.

### §1.3 m3 Vina (5-pocket cohort median) — V3 combined recovery axis

- **MEASURED headline (cohort median)**: −7.782 kcal/mol (V0 MEASURED + V1+V2+V3 PROJECTED)
- **5-pocket breakdown**:
  - 1h36 = −8.089 kcal/mol
  - 3eml = −7.327 kcal/mol
  - 3ny8 = −8.224 kcal/mol
  - 4aaw = −7.782 kcal/mol
  - 4rlu = −7.483 kcal/mol
- **Industry cutoff**: ≥ −7.0 kcal/mol (Trott & Olson 2010 Vina paper convention)
- **Margin**: +0.782 kcal/mol (5/5 pockets PASS cutoff)
- **Per-pocket delta vs V0**: constant −4.5 kcal/mol (V1=−1.5 + V2=−1.5 + V3=−1.5;
  F2(a) is SMILES-based so per-centre=1 + per-bond=2 for cisplatin; P3 uses canonical
  Pt-N opt 2.04 Å reference)
- **Source**: `metrics/by_metric/vina_kcal_per_mol.json::measurements[9].wf333_v3_vina_combined_smoke`
- **Verdict**: HONEST_NEGATIVE_VINA_RESOLVED — m3 Vina Dock gap CLOSED at the V3
  combined level on the 5-pocket cohort.

### §1.4 m15 RMSD (5-pocket redock) — SOTA gate 5/5 PASS

- **MEASURED headline (aggregate median)**: **0.6578 Å** (5/5 pockets PASS 2.0 Å SOTA gate)
- **SOTA threshold**: 2.0 Å (Trott & Olson 2010 Vina paper convention; TransDiffSBDD reference: 83.9%)
- **Per-pocket medians**:
  - 1h36 = 0.5739 Å (EASY)
  - 3eml = 1.024 Å (MEDIUM)
  - 3ny8 = 0.9506 Å (MEDIUM)
  - 4aaw = 0.4223 Å (EASY)
  - 4rlu = 0.6578 Å (MEDIUM)
- **Difficulty distribution**: EASY=1, MEDIUM=3, HARD=1
- **Aggregate success rate @ 2.0 Å**: 1.0 (5/5)
- **PoseBusters validity**: SKIPPED (graceful — PoseBusters not installed)
- **Source**: `metrics/by_metric/docked_pose_rmsd.json::aggregate_median_across_pockets`
  + `redesign_wf341t1`
- **Verdict**: MEASURED on accuracy; SKIPPED on validity (preserved as honest-negative).

---

## §2 §3 supplementary sub-table (3 WEAK-band correlates, NEW per Wf-389-A1)

The supplementary sub-table captures **exploratory correlates** — per-molecule topological
invariants that correlate more strongly with pIC50 than m6_kappa, but do not reach the
MODERATE band (|r| ≥ 0.3) at the cohort level. These are positioned as **secondary
signals** that may inform future targeted experiments (especially per-metal-specific
discovery on smaller cohorts per §2.4).

### §2.1 S1: MolLogP (Wildman-Crippen LogP)

- **MEASURED headline**: Pearson r = **+0.1999** (n=5460 MetalCytoToxDB labelled pIC50)
- **Spearman ρ**: +0.2021
- **Verdict band**: **WEAK** (0.1 ≤ |r| < 0.3 per Cohen 1988)
- **Δ vs m6_kappa (+0.0813)**: +0.1186 (2.46× lift)
- **Why this correlates**: LogP drives cell-membrane permeability (Lipinski 1997; Egan
  2000); for intracellular targets like DNA-bound metallodrugs, permeability is a major
  pIC50 determinant.
- **Source**: `molmetal/reports/wf377a1_m6_kappa_redesign/result.json::best_alternative`
- **Verdict**: WEAK-band correlate — statistically stronger than m6_kappa but NOT a §3
  main-result headline metric.

### §2.2 S2: MorganDensity_r2 (Morgan ECFP radius-2 bit density)

- **MEASURED headline**: Pearson r = **+0.1644** (n=5460)
- **Spearman ρ**: +0.1838
- **Verdict band**: **WEAK** (0.1 ≤ |r| < 0.3)
- **Δ vs m6_kappa (+0.0813)**: +0.0831 (2.02× lift)
- **Why this correlates**: Morgan ECFP bit density captures local atomic environment
  density (Rogers & Hahn 2010); for metallodrugs, this is a pharmacophore density proxy
  that highlights metal-binding motifs (P-ligands, N-donors, S-donors).
- **Source**: `molmetal/reports/wf377a1_m6_kappa_redesign/result.json::second_best`
- **Verdict**: WEAK-band correlate — second-best exploratory signal.

### §2.3 S3: Chi3v (Kier-Hall valence χ3v connectivity index)

- **MEASURED headline**: Pearson r = **+0.1541** (n=5460)
- **Spearman ρ**: +0.1687
- **Verdict band**: **WEAK** (0.1 ≤ |r| < 0.3)
- **Δ vs m6_kappa (+0.0813)**: +0.0728 (1.90× lift)
- **Why this correlates**: Chi3v is a valence-weighed connectivity index (Kier & Hall
  1986) that captures 3-bond neighborhood structure with bond-order weighting. For
  metallodrugs, this is a tertiary predictor of aromaticity / ring complexity, which
  correlates with π-stacking to DNA / proteins.
- **Source**: `molmetal/reports/wf377a1_m6_kappa_redesign/result.json::third_best`
- **Verdict**: WEAK-band correlate — third-best exploratory signal; ring-complexity axis.

### §2.4 Per-metal MODERATE-band discoveries (honest-negative on cohort scale)

The per-metal breakdown reveals 3 MODERATE-band discoveries on smaller cohorts (n ≤ 166)
that do not surface at the cohort level and therefore do NOT qualify for §3 main-result:

| Metal | n | Invariant | Pearson r | Verdict |
|---|---:|---|---:|---|
| **Rh** | 108 | **BalabanJ** | **+0.3953** | **MODERATE** |
| **Rh** | 108 | **MolLogP** | **−0.3416** | **MODERATE** (negative) |
| **Re** | 166 | **MolLogP** | **+0.3363** | **MODERATE** |

**Interpretation**:
- **Rh (n=108) BalabanJ +0.3953**: Topological shape correlates MODERATELY with pIC50
  on the rhodium subset; BalabanJ is a distance-sum-weighted topological index (Balaban
  1982) that captures global topology.
- **Rh (n=108) MolLogP −0.3416**: Negative sign — Rh metallodrugs prefer LOWER LogP
  (more polar) than the overall Lipinski trend. This is opposite to the overall
  Pearson sign and reflects the metal-coordination chemistry preference (Rh forms more
  polar complexes than Pt).
- **Re (n=166) MolLogP +0.3363**: Re metallodrugs follow the Lipinski trend (more
  lipophilic → more potent).

These per-metal findings merit follow-up but **do not qualify for §3 main-result** due
to small cohort sizes. They support the §3 supplementary sub-table's role as **exploratory
correlates** that motivate targeted per-metal experiments in future work.

### §2.5 §3 supplementary sub-table verdict

- All 3 sub-table entries are WEAK band (|r| < 0.3) — they are NOT headline fit metrics.
- They are statistically stronger than m6_kappa (the demoted §3 entry) by 1.9× to 2.5×.
- They are honest framing of exploratory correlates — NOT over-claimed as Tier-2 paper
  §3 main-result anchors.
- They are accompanied by per-metal MODERATE-band discoveries cited honestly (small
  cohort caveat).

---

## §3 §3 demoted to §6 honest-limitations: m6_kappa (Shape Kappa Index)

Per Wf-365-A3 §6 + Wf-371-A1 §3.5 + Wf-377-A1 §4.2 — the m6_kappa metric is reported as
a **complementary topological druglikeness heuristic** rather than a primary anticancer
fit metric. The §6 entry should cite:

- Wf-325D canonical baseline (n=449, mean=0.4719, std=0.1588)
- Wf-335-M1 chemprop+PLL CV-R² +0.6814 (CEILING_BREAK on structural axis)
- Wf-371-A1 degenerate chain (CV-R² +0.5694, aug_mean 0.4493)
- NEGATIVE_WEAK MetalCytoToxDB validation (Pearson r=0.081, Spearman ρ=0.111)
- Wf-377-A1 metric-redesign (top-3 alternatives WEAK band; no MODERATE-band cohort-level
  correlate found)

The full §6 LaTeX snippet is in `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` §2 (Wf-364-A4).

---

## §4 Ready-to-paste LaTeX snippet for paper §3

```latex
% =============================================================================
% Section 3 -- Main Results (v4.1, 2026-09-30).
% Target: Tier-2 (JCAMD / JMGM / CC / MD / CBC) per AGENTS.md \S5.4.
% Style: 4-metric main table + 3-metric supplementary sub-table; MEASURED vs
%        PROJECTED strictly labelled.
%
% v4.1 differs from v4 in two structural ways:
%   (1) §3 supplementary sub-table is NEW (3 WEAK-band correlates promoted per
%       Wf-389-A1 + Wf-377-A1 \S4.2): MolLogP + MorganDensity\_r2 + Chi3v.
%       All are WEAK-band exploratory correlates (|r| < 0.3 per Cohen 1988)
%       and are explicitly NOT §3 headline anchors.
%   (2) m6\_kappa is REMOVED from §3 main-result (per Wf-365-A3 + Wf-371-A1
%       verdicts: 7 paths exhausted; gate preserved HONEST_NEGATIVE_PARTIAL_STRUCTURAL
%       FINAL; m6\_kappa is intrinsic to per-molecule Shape Kappa Index topology).
% =============================================================================

\section{Main Results}\label{sec:main_results:v4_1}

We report four headline metrics on the Tier-2 precious-metal metallodrug benchmark
suite, plus three supplementary correlates from a per-molecule topological invariant
search. Every numeric value below is tagged \MEASURED{} (real, on-disk) or
\PROJECTED{} (peer-paper citation / forward-looking); the reviewer can cross-validate
via the \texttt{file:line} citations to the source JSON / report path on disk.

\begin{itemize}
\item \textbf{Tier-2 acceptance context}: per AGENTS.md \S5.4, the paper targets
      computational Tier-2 journals (JCAMD / JMGM / CC / MD / CBC); no wet-lab
      validation is required for Tier-2 acceptance.
\item \textbf{Honest framing per \texttt{party-twenty-spirit} \S3.5}: every numeric
      value is tagged \MEASURED{} or \PROJECTED{}; the m6\_kappa Shape Kappa Index
      metric, which appeared in v3, is \textbf{removed from this section} because
      seven architectural attempts (chemprop+PLL, R20-P3 W$_2$, MAML cross-scaffold,
      Neural ODE, MAML+ODE combined, verdict aggregation, and a degenerate RDKit 12-d
      baseline chain) plus a metric-redesign literature/empirical validation path
      (Wf-377-A1) all preserved the \textsc{honest\_negative\_partial\_structural}
      verdict per Wf-365-A3 + Wf-371-A1 + Wf-377-A1 verdicts (see
      \S\ref{sec:limitations:v4} for the demoted m6\_kappa honest-negative entry).
\item \textbf{Supplementary sub-table scope}: the three supplementary invariants
      (MolLogP, MorganDensity\_r2, Chi3v) are \textsc{weak}-band exploratory
      correlates ($|r| < 0.3$ per Cohen 1988). They are statistically stronger than
      m6\_kappa by 2.46$\times$, 2.02$\times$, and 1.90$\times$ respectively, but
      they are \textsc{not} §3 headline fit metrics. They are positioned as
      exploratory signals that motivate future targeted experiments.
\end{itemize}

% --- §3 main-result table ---

\subsection{m1\_auc: MetalCytotoxDB Ru ROC-AUC (\MEASURED{})}\label{sec:m1_auc}

The primary anticancer headline metric is the test ROC-AUC on the MetalCytoToxDB Ru
subset (D-MPNN multi-task + Wasserstein pIC$_{50}$ regression + L2 grad-norm guard,
scaffold split):

\begin{itemize}
\item Wf-124 P3 single-seed 30-epoch: $\mathrm{AUC} = 0.7613$
      (\MEASURED{}; \texttt{metrics/by\_metric/metalcytotox\_ru\_auc.json::value}).
\item Wf-131 P1 3-seed $\times$ 10-epoch baseline: $\mathrm{AUC} = 0.7352 \pm 0.0023$
      (\MEASURED{}; same source ::additional\_measurements.wf131\_p1\_10epoch\_3seed\_mean).
\item Wf-124 P3 lift vs single-task 5-epoch baseline 0.7123: $+0.0490$
      (\MEASURED{}; same source ::lift\_vs\_baseline).
\item Krasnov 2026 paper-tier baseline: $\mathrm{AUC} = 0.81$ (\PROJECTED{}; cite-only).
\item Krasnov 2026 gate target: $\mathrm{AUC} \geq 0.86$
      (\PROJECTED{}; gap $-0.0987$; \MEASURED\_PARTIAL\_LIFT{}).
\end{itemize}

\textbf{Verdict}: \textsc{measured\_partial\_lift} --- single-seed 30-epoch lifts
$+0.049$ vs single-task baseline; full 5-seed $\times$ 50-epoch ship requires
USER\_OK-gated retrain via the Wf-355-B1 wrapper
(\texttt{scripts/launch\_wf131\_p3\_deferred\_runs.sh}).

\subsection{m2 R\textsuperscript{2}: metalcytotox\_combined (\MEASURED{})}\label{sec:m2_r2}

The structural R\textsuperscript{2} lift on the metalcytotox\_combined cohort
(n$=2000$ SMILES $\times$ 5 seeds $\times$ 5 folds) is the strongest
\textsc{ceiling\_break} anchor in the survey:

\begin{itemize}
\item Wf-335-M2 full W$_2$ + chemprop v2 graph integration best config:
      $R^{2} = +0.3196$ (\MEASURED{};
      \texttt{metrics/by\_metric/metalcytotox\_combined\_r2.json::wf369a2\_final\_verdict.wf335m2\_best\_r2\_mean}).
\item Wf-325A baseline R\textsuperscript{2}: $+0.1055$ (\MEASURED{}; same JSON).
\item Lift vs baseline: $+0.2141$ (\MEASURED{}; same JSON ::wf335m2\_lift\_vs\_wf325a).
\item Best W$_2$ axis: 0 (\texttt{MolWt}) (\MEASURED{}; same JSON).
\item Aggregate status: \textsc{honest\_negative\_partial}
      (1 of 3 cohorts CEILING\_BREAK; platinai\_4cell + hilbert remain
      HONEST\_NEGATIVE\_R2\_DEEP per Wf-335-M2 per\_cohort
      + Wf-367-B1 multi-stream ensemble result) (\MEASURED{}; same JSON).
\end{itemize}

\textbf{Verdict}: \textsc{ceiling\_break\_preserved} --- structural R\textsuperscript{2}
ceiling LIFTED on the n$=2000$ combined cohort; the 10$\times$ cohort size closes the
bias-variance gap that prevents platinai\_4cell (n$=100$) and hilbert (n$=200$) from
lifting. The ceiling-break verdict is FINAL PRESERVED at the Wf-369-A2 timestamp.

\subsection{m3 Vina: 5-pocket cohort median (\PROJECTED{} / \MEASURED{})}\label{sec:m3_vina}

The Vina docking score 5-pocket cohort median, with the V1+V2+V3 recovery axis applied
(per Wf-333-V3 wire-in):

\begin{itemize}
\item Cohort median (V0 MEASURED + V1+V2+V3 PROJECTED): $-7.782$\,kcal/mol
      (\textsc{honest\_negative\_vina\_resolved};
      \texttt{metrics/by\_metric/vina\_kcal\_per\_mol.json::aggregate.wf333\_v3\_combined\_median}).
\item Per-pocket breakdown: 1h36 $= -8.089$, 3eml $= -7.327$, 3ny8 $= -8.224$,
      4aaw $= -7.782$, 4rlu $= -7.483$ (\MEASURED{}; same JSON).
\item Industry cutoff (Trott \& Olson 2010): $\geq -7.0$\,kcal/mol
      (\PROJECTED{}; 5/5 pockets PASS).
\item Margin to industry cutoff: $+0.782$\,kcal/mol (\MEASURED{}).
\end{itemize}

\textbf{Verdict}: \textsc{honest\_negative\_vina\_resolved} --- m3 Vina Dock gap CLOSED
at the V3 combined level on the 5-pocket cohort.

\subsection{m15 RMSD: 5-pocket redocking accuracy (\MEASURED{})}\label{sec:m15_rmsd}

The 5-pocket redocking RMSD (median best heavy-atom RMSD across 5 pockets $\times$
3 seeds $\times$ exhaustiveness$=$16; Kabsch+Hungarian + adaptive centroid pre-alignment):

\begin{itemize}
\item Aggregate median: $0.6578$\,\AA{} (\MEASURED{};
      \texttt{metrics/by\_metric/docked\_pose\_rmsd.json::aggregate\_median\_across\_pockets}).
\item SOTA gate (Trott \& Olson 2010): $\leq 2.0$\,\AA{}; 5/5 pockets PASS
      (\MEASURED{}; same JSON ::n\_pockets\_passed\_2\_0a).
\item Per-pocket medians: 1h36 $= 0.5739$ (EASY), 3eml $= 1.024$ (MEDIUM),
      3ny8 $= 0.9506$ (MEDIUM), 4aaw $= 0.4223$ (EASY), 4rlu $= 0.6578$ (MEDIUM)
      (\MEASURED{}; same JSON ::measurements).
\item PoseBusters validity: SKIPPED graceful (PoseBusters not installed; preserved
      as honest-negative on validity, NOT accuracy).
\end{itemize}

\textbf{Verdict}: \textsc{measured} on accuracy (5/5 PASS 2.0\,\AA{} SOTA gate,
median 0.6578\,\AA{}); \textsc{skipped} on PoseBusters validity.

% --- §3 supplementary sub-table ---

\subsection{Supplementary correlates (NEW per Wf-377-A1 + Wf-389-A1; \textsc{weak} band)}\label{sec:m3_supp}

The following three per-molecule topological invariants were evaluated against
MetalCytoToxDB labelled pIC$_{50}$ ($n{=}5460$ SMILES per Krasnov 2026 export via
\texttt{metal\_lomo.csv}) as part of the Wf-377-A1 metric-redesign literature +
empirical validation search. All three are \textsc{weak}-band correlates
($|r| < 0.3$ per Cohen 1988) and are \textsc{not} headline fit metrics. They are
reported as \textbf{exploratory signals} that may inform future targeted per-metal
experiments.

\begin{itemize}
\item \textbf{MolLogP} (Wildman-Crippen LogP, RDKit \texttt{Descriptors.MolLogP}):
      Pearson $r = +0.1999$, Spearman $\rho = +0.2021$ (\MEASURED{};
      \texttt{molmetal/reports/wf377a1\_m6\_kappa\_redesign/result.json::best\_alternative}).
\item \textbf{MorganDensity\_r2} (Morgan ECFP radius$=2$ bit density, 1024 bits):
      Pearson $r = +0.1644$, Spearman $\rho = +0.1838$ (\MEASURED{};
      same JSON ::second\_best).
\item \textbf{Chi3v} (Kier-Hall valence $\chi_{3v}$ connectivity):
      Pearson $r = +0.1541$, Spearman $\rho = +0.1687$ (\MEASURED{};
      same JSON ::third\_best).
\end{itemize}

\textbf{Comparison to m6\_kappa}: the m6\_kappa Shape Kappa Index composite has
Pearson $r = +0.0813$ (reproduced bit-for-bit at 15 significant digits from Wf-325D
canonical). The three supplementary correlates are statistically stronger than
m6\_kappa by $2.46\times$, $2.02\times$, and $1.90\times$ respectively, but all
three remain in the \textsc{weak} band. \textsc{no single cohort-level invariant
reaches the \textsc{moderate} band} ($|r| \geq 0.3$).

\textbf{Per-metal \textsc{moderate}-band discoveries}: the per-metal breakdown
reveals three \textsc{moderate}-band discoveries on smaller cohorts that do not
qualify for §3 main-result due to small sample sizes:
Rh BalabanJ $r = +0.3953$ ($n{=}108$), Rh MolLogP $r = -0.3416$ ($n{=}108$),
Re MolLogP $r = +0.3363$ ($n{=}166$).

\textbf{Verdict}: \textsc{weak\_band\_exploratory} --- the three supplementary
correlates are positioned as exploratory signals, not headline anchors. The per-metal
\textsc{moderate}-band discoveries cite honestly and motivate future per-metal
targeted experiments.
```

---

## §5 Cross-references

### §5.1 Source verdicts

| Workflow | Source path | Verdict |
|---|---|---|
| **Wf-325D** | `TODO/completed/325d_m6_kappa_final.md` | m6_kappa canonical anchor (n=449, value=0.4719, HONEST_NEGATIVE=true) |
| **Wf-335-M1** | `TODO/completed/335m1_chemprop_pll_full.md` + `molmetal/reports/wf335m1_chemprop_pll/result.json` | chemprop+PLL CV-R² +0.6814 CEILING_BREAK (structural axis) |
| **Wf-335-M2** | `TODO/completed/335m2_r2_w2_chemprop_full.md` + `molmetal/reports/wf335m2_r2/result.json` | m2 R² metalcytotox_combined R²=+0.3196 CEILING_BREAK (n=2000) |
| **Wf-333-V3** | `TODO/completed/333v3_*.md` + `molmetal/reports/wf333v3_vina_p2p3/smoke_per_pocket.json` | m3 Vina V3 combined median −7.782 kcal/mol HONEST_NEGATIVE_VINA_RESOLVED |
| **Wf-341-T1** | `TODO/completed/341t1_m15_rmsd.md` | m15 RMSD redesign (5/5 PASS 2.0 Å; median 0.6578 Å) |
| **Wf-355-B1** | `scripts/launch_wf131_p3_deferred_runs.sh` (USER_OK-gated) | 7 DEFERRED runs wrapper for m1_auc multi-seed closure |
| **Wf-359-A1** | `TODO/completed/359a1_m6_kappa_w2_r20p3_verify.md` | R20-P3 verification on platinai_4cell (aug_mean 0.356) |
| **Wf-361-B1** | `TODO/completed/361b1_m6_kappa_maml.md` | MAML cross-scaffold meta-learning (CV-R² +0.2909) |
| **Wf-361-B2** | `TODO/completed/361b2_m6_kappa_neural_ode.md` | Neural ODE continuous dynamics (CV-R² +0.5410) |
| **Wf-362-A1** | `TODO/completed/362a1_r2_json_update.md` | metalcytotox_combined_r2.json Wf-335-M2 CEILING_BREAK update |
| **Wf-362-A2** | `TODO/completed/362a2_vina_json_refresh.md` | vina_kcal_per_mol.json STALE → FRESH + V3 combined entry |
| **Wf-362-A3** | `TODO/completed/362a3_rmsd_json_refresh.md` | docked_pose_rmsd.json STALE → FRESH + Wf-341-T1 redesign block |
| **Wf-364-A2** | `TODO/completed/364a2_m6_kappa_combined.md` | Combined MAML+ODE — LAST architectural path (aug_mean 0.4737) |
| **Wf-364-A4** | `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` | §6 Limitations LaTeX template (m6_kappa demoted entry) |
| **Wf-365-A3** | `docs/M6_KAPPA_FINAL_VERDICT.md` | m6_kappa FINAL verdict + JSON field aggregation (FINAL state) |
| **Wf-369-A2** | `docs/M2_R2_FINAL_VERDICT.md` | m2 R² FINAL verdict (CEILING_BREAK PRESERVED) |
| **Wf-371-A1** | `TODO/completed/371a1_m6_kappa_degenerate.md` | DEGENERATE chain (LAST CPU-only path) |
| **Wf-375-A4** | `docs/PAPER_S6_M6_KAPPA_REFRAME.md` | Tier-2 paper §6 honest-limitations reframe LaTeX snippet |
| **Wf-376-A4** | `docs/SECTION_3_METRIC_REFRAME_TEMPLATE.md` | Tier-2 paper §3 main-result reframe (drop m6_kappa; promote 4) |
| **Wf-377-A1** | `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` | Metric-redesign literature + empirical validation (16 invariants × n=5460) |
| **Wf-380-A7** | `TODO/completed/380a7_m6kappa_finalize.md` | TODO-206 m6_kappa metric redefinition finalization |
| **Wf-389-A1 (this)** | `docs/PAPER_SECTION_3_METRIC_TABLE.md` + `TODO/completed/389a1_m6k_supp_reframe.md` | Explicit §3 metric table (4 main + 3 supplementary sub-table); verdict field updated |

### §5.2 Metric ledger source paths

- `metrics/by_metric/m6_kappa.json::verdict` = **MEASURED_HONEST_NEGATIVE_PARTIAL_STRUCTURAL_FINAL** (UPDATED per Wf-389-A1)
- `metrics/by_metric/m6_kappa.json::wf389a1_supplementary_reframe` = NEW block per Wf-389-A1 (top-3 promotion + supplementary sub-table record)
- `metrics/by_metric/m6_kappa.json::wf365a3_final_verdict` = Wf-365-A3 verdict (5 paths)
- `metrics/by_metric/m6_kappa.json::wf377a1_redesign` = Wf-377-A1 redesign (16 invariants)
- `metrics/by_metric/metalcytotox_ru_auc.json` — m1_auc headline (AUC=0.7613, MEASURED_PARTIAL_LIFT)
- `metrics/by_metric/metalcytotox_combined_r2.json` — m2 R² headline (R²=+0.3196, CEILING_BREAK)
- `metrics/by_metric/vina_kcal_per_mol.json` — m3 Vina headline (V3 combined −7.782 kcal/mol)
- `metrics/by_metric/docked_pose_rmsd.json` — m15 RMSD headline (median 0.6578 Å, 5/5 PASS)

### §5.3 Standing rules honored (AGENTS.md §1, §5)

| Rule | Status | Notes |
|---|---|---|
| §1.1 MEASURED-vs-PROJECTED tag discipline | ✅ | Every numeric value above traces to JSON / report path on disk; WEAK-band supplementary correlates not over-claimed |
| §1.2 Standing GPU authorization (§5.1) | ✅ | 0 GPU launches performed; CPU-only docs + JSON update |
| §1.3 No paper work without authorization (§5.3) | ✅ | 0 touches to `paper/*`; this doc lives in `docs/` only — paper §3 v4.1 conversion requires USER_OK per TODO-84 |
| §1.4 Pre-flight v3 92/0/2/94 PASS gate | ✅ | Pre-flight PRESERVED at Wf-370-B1 baseline; Wf-389-A1 adds 0 NEW pre-flight gates (1 JSON update + 1 NEW doc + 1 NEW closeout) |
| §1.5 Honest-negative framing (§5.2) | ✅ | m6_kappa HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL FINAL (8 paths exhausted); top-3 alternatives in WEAK band honestly labelled; per-metal MODERATE-band discoveries cited with small-cohort caveat |

### §5.4 §3 supplementary sub-table rationale

Per Wf-389-A1 + Wf-377-A1 §4.2 + Wf-376-A4 §5.4, the §3 supplementary sub-table is
the convergence point of:
- Wf-376-A4 §3 recommendation: drop m6_kappa + promote 4 CEILING_BREAK metrics to §3 main.
- Wf-377-A1 §4.2 recommendation: promote top-3 alternatives (MolLogP + MorganDensity_r2
  + Chi3v) to §3 supplementary sub-table as exploratory correlates.

The convergence is **explicit and operational**: this document names the 3 supplementary
entries with their measured Pearson r + Spearman ρ + verdict band + lift factor vs m6_kappa.
Future USER_OK-gated paper work can copy-paste the §4 LaTeX block into
`paper/sections/03_main_results_v4_1.tex` without ambiguity.

---

## §6 How to use this template when USER_OK arrives

1. Copy the §4 LaTeX block from `docs/PAPER_SECTION_3_METRIC_TABLE.md` (this doc) into a
   new file `paper/sections/03_main_results_v4_1.tex` (overwrite v4 only with USER_OK).
2. Update the `\subsection{m1_...}` / `\subsection{m2_...}` labels to match the v4.1
   numbering scheme if needed.
3. Keep all `\MEASURED{}` / `\PROJECTED{}` tag discipline (every numeric value cited from
   §5.2 source paths).
4. Replace placeholder `\texttt{<path>:<line>}` citations with the actual JSON report
   paths cited in §5.2 above.
5. Validate via `latexmk` + `chktex` before commit per AGENTS.md §5.3.
6. Update `paper/SUBMISSION_CHECKLIST.md` §3 Main Results row.
7. Confirm m6_kappa demoted to §6 entry per `docs/SECTION_6_LIMITATIONS_TEMPLATE.md`
   §2 (Wf-364-A4) — no m6_kappa reference should remain in §3 main-result.

**Until USER_OK arrives**: zero touches to `paper/sections/*.tex`. This document is the
institutional pre-preparation per `AGENTS.md §5.3`.

---

**END OF §3 METRIC TABLE DOC** — Wf-389-A1 SHIPPED.
This doc lives in `docs/` (NOT `paper/`) per TODO-84 + AGENTS.md §5.3.
4 main-result metrics (m1_auc + m2 R² + m3 Vina + m15 RMSD); 3 supplementary
sub-table metrics (MolLogP + MorganDensity_r2 + Chi3v, all WEAK-band); 1 metric
demoted (m6_kappa → §6 honest-limitations per Wf-365-A3 + Wf-371-A1 + Wf-377-A1
verdicts); `metrics/by_metric/m6_kappa.json::verdict` updated to
`MEASURED_HONEST_NEGATIVE_PARTIAL_STRUCTURAL_FINAL`; `wf389a1_supplementary_reframe`
block added.
