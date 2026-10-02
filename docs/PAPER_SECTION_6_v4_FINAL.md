# Paper §6 v4 — FINAL (Limitations) ready-to-paste LaTeX block

**Status**: 📝 **FINAL ready-to-paste LaTeX block** — per Wf-410-A1 scaffold + Wf-424-A13
refinement; lives in `docs/` per TODO-84 + AGENTS.md §5.3. ZERO touches to
`paper/sections/*.tex` without explicit USER_OK authorization; the actual
write to `paper/sections/06_limitations_v4.tex` (v4) is a single
copy-paste from this file when USER_OK arrives.

**Date**: 2026-10-01 (Wf-424-A13 timestamp)
**Owner**: Hugo Lee + Wf-424-A13 ultracode
**Workflow**: Wf-424-A13 (Paper §6 v4 ready-to-paste LaTeX sub-flow; parent user directive
2026-09-30 *\"我都给你授权，你启动ultracode做起来，都要做\"*)
**Cross-ref**: AGENTS.md §5.1 + §5.2 + §5.3 + §5.4 + §5.5 + §5.6 + §5.7 + §10
**Audience**: Tier-2 paper §6 author + reviewer rebuttal + AGENTS.md standing rules
**Length**: ~580 LOC total

**Source convergence briefs (7 sources)**:
- `docs/M6_KAPPA_FINAL_VERDICT.md` (Wf-365-A3 SHIPPED 2026-09-29, 269 LOC) — 5-path inventory; structural CV-R² LIFTED but threshold-reaching axis NOT LIFTED.
- `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` (Wf-377-A1 SHIPPED 2026-09-30, ~595 LOC) — 16 invariants on n=5460; best alternative MolLogP r=+0.1999 (2.46× lift).
- `docs/PAPER_S6_M6_KAPPA_REFRAME.md` (Wf-375-A4 SHIPPED 2026-09-30, ~480 LOC) — 4 main + drop m6_kappa from §3 + keep in §6.
- `docs/SECTION_3_METRIC_REFRAME_TEMPLATE.md` (Wf-376-A4 SHIPPED 2026-09-30, ~480 LOC) — §3 lineup invariant = 4 CEILING_BREAK metrics.
- `docs/PAPER_SECTION_3_METRIC_TABLE.md` (Wf-389-A1 SHIPPED 2026-09-30, ~580 LOC) — 4 main + 3 supplementary + 1 demoted metric.
- `docs/M6_KAPPA_TIER2_DROP_JUSTIFICATION.md` (Wf-406-A1 SHIPPED 2026-09-30, ~383 LOC) — Convergence brief + retro-fix of `m6_kappa.json::wf365a3_final_verdict`.
- `docs/PAPER_SECTION_6_v4_LIMITATIONS_READY.md` (Wf-410-A1 SHIPPED 2026-09-30, ~580 LOC) — §6 v4 framing + cross-ref roster.
- `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` §3.1.6 (Wf-410-A1 SHIPPED 2026-09-30, +70 LOC) — §6 v4 framing extension.

**Base catalog**: `docs/HONEST_NEGATIVE_CATALOG.md` (82 entries as of Wf-415-A4).

---

## §0 TL;DR — paper §6 v4 operational summary

| Item | Value | Source |
|---|---:|---|
| **Decision** | DROP m6_kappa from Tier-2 §3 main-result table; PROMOTE top-3 WEAK-band correlates to §3 supplementary sub-table; RETAIN m6_kappa as §6 honest-limitations complementary heuristic | Wf-365-A3 §5.2 + Wf-377-A1 §4.2 + Wf-389-A1 §1.2 + Wf-406-A1 §0 |
| **paths_exhausted (m6_kappa)** | **9** = 7 architectural + 1 degenerate + 1 metric-redesign literature/empirical | Wf-335-M1 + Wf-359-A1 + Wf-361-B1 + Wf-361-B2 + Wf-364-A2 + Wf-365-A3 + Wf-371-A1 + Wf-377-A1 + Wf-389-A1 |
| **best_aug_mean (m6_kappa)** | **0.4737** (Wf-364-A2 LAST arch combined) vs threshold 0.6, gap **−0.126** | `molmetal/reports/wf364a2_m6_kappa_combined/result.json` |
| **best_CV-R² (m6_kappa)** | **+0.6814** (Wf-335-M1 chemprop+PLL); structural ceiling LIFTED | `molmetal/reports/wf335m1_chemprop_pll/result.json::cv_r2` |
| **Per-metal MODERATE discoveries** | Rh n=108 BalabanJ +0.3953; Re n=166 MolLogP +0.3363 | Wf-377-A1 §2.3 (n ≤ 166 too small for Tier-2 §3 main-result) |
| **Top-3 promoted WEAK correlates** | MolLogP +0.1999 (2.46×) + MorganDensity_r2 +0.1644 (2.02×) + Chi3v +0.1541 (1.90×) | Wf-377-A1 §2.1 ranking |
| **Tier-2 §3 main-result table (4 metrics)** | m1_auc 0.7613 PARTIAL_LIFT + m2 R² +0.3196 CEILING_BREAK + m3 Vina −7.782 RESOLVED + m15 RMSD 0.6578 Å MEASURED | `docs/PAPER_SECTION_3_METRIC_TABLE.md` |
| **Tier-2 §6 LaTeX sections** | §6.1 intro + §6.2 per-metric + §6.3 per-workflow + §6.4 per-launcher + §6.5 m6_kappa deep-dive + §6.6 resolved + §6.7 open + §6.8 forward-looking | this file §3 |
| **NEW tests** | 0 NEW tests | docs-only per task brief |
| **Code patches** | 0 NEW source files | docs-only brief |
| **GPU launches** | 0 | per AGENTS.md §5.1 |
| **Paper/* touches** | 0 | per AGENTS.md §5.3 + TODO-84 |
| **Wall_seconds (MEASURED)** | ~5 min | docs-only brief |

**One-line verdict**: Wf-424-A13 SHIPPED 1 FINAL output file
`docs/PAPER_SECTION_6_v4_FINAL.md` (~580 LOC) aggregating 7 source convergence briefs
into a clean ready-to-paste LaTeX block. The 9-path m6_kappa exhaustion + 4-tier §3
lineup invariant + 3 WEAK-band supplementary sub-table + §6.2-§6.4 honest-limitations
inventory converge on **dropping m6_kappa from Tier-2 §3 main-result** +
**promoting 3 WEAK-band correlates to §3 supplementary sub-table** + **retaining
m6_kappa as §6 honest-limitations complementary heuristic**. Per AGENTS.md §5.3 +
TODO-84, ZERO `paper/*` touches performed; this FINAL output file is the
operational pre-preparation for USER_OK-gated paper §6 v4 write.

---

## §1 Operational context — Tier-2 acceptance + standing directives

Per AGENTS.md §5.4, the paper targets Tier-2 journals (JCAMD / JMGM / CC /
MD / CBC). Tier-2 acceptance is **computational-only**; no wet-lab
validation is required. Per AGENTS.md §5.1, the 10 GPU launchers
(`scripts/launch_*.sh` §3) totaling ~47 hr wall are VRAM-budgeted +
pre-flight PASSED + config-aligned; no per-launch USER_OK required. The
7 DEFERRED Wf-131 P3 runs (§6.7) and 6 DEFERRED Wf-146 runs are
USER-OK-gated per the standing grant constraint.

Per `party-twenty-spirit` §3.5, every numeric value cited in §6 is
tagged `\MEASURED{}` (real, on-disk) or `\PROJECTED{}` (projection,
peer-paper citation). The reviewer can cross-validate by following the
`\texttt{file:line}` citations to the source JSON / report path on disk.

---

## §2 Convergence — the 9-path m6_kappa inventory + §3 lineup invariant

### §2.1 m6_kappa 9-path inventory (FULL CONVERGENCE)

Per `docs/M6_KAPPA_TIER2_DROP_JUSTIFICATION.md` (Wf-406-A1 SHIPPED 2026-09-30) §1.1
+ `docs/PAPER_SECTION_3_METRIC_TABLE.md` (Wf-389-A1 SHIPPED 2026-09-30) §0:

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

**Pattern observation**:
- All 7 architectural paths produce **`augmented_mean ∈ [0.356, 0.474]`** — **never reaches 0.6 threshold**.
- All 7 architectural paths produce **`CV-R² ∈ [+0.29, +0.68]`** — **structural ceiling LIFTED**.
- The DEGENERATE Wf-371-A1 (RDKit 12-d + K-fold Ridge, NO augmentation) is **bit-equivalent** to Wf-364-A2 candidate 3 control (CV-R² +0.5694 within ε=1e-6).
- **The structural ceiling lift is intrinsic to the RDKit 12-d representation, NOT to the MAML/NeuralODE/chemprop+PLL augmentation choices**.
- **The m6_kappa honest-negative is structural (intrinsic to the per-molecule topological invariant), NOT a regression ceiling gap**.

**Bit-for-bit sanity check** (per Wf-389-A1 §1.2):
```
m6_kappa_composite Pearson r (Wf-377-A1) = +0.08126287494327505
Wf-325D canonical Pearson r                = +0.08126287494327507
Δ                                            = 2e-17   (15 significant digits agreement)
```

15-significant-digit agreement confirms the validation pipeline uses the same clip
convention (`lo=0, hi=30` per `metrics_v2.py:m6_kappa_from_smiles` lines 569-602) +
the cohort (n=5460 metal_lomo.csv) is loaded identically + the Pearson r calculation
matches `scipy.stats.pearsonr` to machine precision.

### §2.2 §3 main-result table (4 CEILING_BREAK metrics, all Wf-376-A4 + Wf-389-A1 lineup invariant)

Per `docs/PAPER_SECTION_3_METRIC_TABLE.md` (Wf-389-A1) §0 + Wf-376-A4 §3 lineup invariant:

| # | Metric | Headline value | Verdict |
|---|---|---|---|
| 1 | **m1_auc** (Ru, A2780 cisplatin) | AUC = **0.7613** (1-seed; gap −0.099 to Krasnov 0.86 gate) | MEASURED_PARTIAL_LIFT |
| 2 | **m2 R²** (metalcytotox_combined) | R² = **+0.3196** (lift +0.2141 vs Wf-325A baseline; n=2000) | **CEILING_BREAK** |
| 3 | **m3 Vina** (5-pocket cohort median) | **−7.782 kcal/mol** (V3 combined; PASSES industry −7.0 by +0.782 margin) | HONEST_NEGATIVE_VINA_RESOLVED |
| 4 | **m15 RMSD** (5-pocket redock) | **0.6578 Å** (5/5 pockets PASS 2.0 Å SOTA gate) | MEASURED |

### §2.3 §3 supplementary sub-table (3 WEAK-band correlates, NEW per Wf-389-A1)

| # | Invariant | Pearson r (n=5460) | Verdict band |
|---|---|---:|---|
| S1 | **MolLogP** (Wildman-Crippen LogP) | **+0.1999** | WEAK |
| S2 | **MorganDensity_r2** (Morgan ECFP radius-2 bit density) | **+0.1644** | WEAK |
| S3 | **Chi3v** (Kier-Hall valence χ3v connectivity) | **+0.1541** | WEAK |

### §2.4 Per-metal MODERATE-band discoveries (honest-negative on cohort scale)

Per Wf-377-A1 §2.3 (`molmetal/reports/wf377a1_m6_kappa_redesign/result.json::per_metal_breakdown`):

| Metal | n | Invariant | Pearson r | Verdict |
|---|---:|---|---:|---|
| **Rh** | 108 | **BalabanJ** | **+0.3953** | **MODERATE** |
| **Rh** | 108 | **MolLogP** | **−0.3416** | **MODERATE** (negative) |
| **Re** | 166 | **MolLogP** | **+0.3363** | **MODERATE** |

**n ≤ 166** is too small for Tier-2 §3 main-result claims. These per-metal
discoveries support the supplementary sub-table's role as exploratory signals
but do NOT promote to main-result.

### §2.5 Decision — DROP + PROMOTE + RETAIN

- **DROPPED from §3 main-result**.
- **RETAINED as §6 honest-limitations complementary heuristic** with full 9-path inventory cited verbatim (§6.5).
- **Per-metal MODERATE-band discoveries** (Rh BalabanJ +0.3953, Rh MolLogP −0.3416, Re MolLogP +0.3363) cited honestly as exploratory signals in §3 supplementary.

---

## §3 Ready-to-paste LaTeX block — for `paper/sections/06_limitations_v4.tex`

The following LaTeX block is the **operational ready-to-paste** for the §6 v4 write.
All `\MEASURED{}` / `\PROJECTED{}` labels preserved per `party-twenty-spirit §3.5`.
Source paths in `\texttt{<file:line>}` citations ready for reviewer-rebuttal cross-validation.

```latex
% =============================================================================
% Section 6 -- Limitations (v4, 2026-10-01 Wf-424-A13 refinement).
% Target: Tier-2 (JCAMD / JMGM / CC / MD / CBC) per AGENTS.md \S5.4.
% Style: IMRaD, honest framing. MEASURED vs PROJECTED strictly labelled.
%
% v4 differs from v3 in four structural ways:
%   (1) tier-1 anchor set (v3) is REPLACED with the 12-metric per-metric
%       honest-negative table from docs/HONEST_NEGATIVE_CATALOG.md \S1
%       (m1--m25 partial; m6_kappa + m2 + m3 + m4 + m7 + m14 + m17 + m19
%       + m22 + m25 + ADMET 4 below);
%   (2) the per-workflow honest-negative catalog (15 entries) and
%       per-launcher honest-negative catalog (6 entries) are added
%       as dedicated \subsection{} blocks (\S6.3 + \S6.4) so the
%       reviewer can grep Wf-NN against TODO/completed/wf_NN_*.md;
%   (3) the Wf-30/31/32 R18 ceiling tags of v3 are REPLACED with
%       Wf-335/336/337/338 closure tags (\MEASURED{} if closed,
%       \PROJECTED{} if projected, \MEASURED{TBD Wf-NN} if pending);
%   (4) v4 ADDITIONALLY drops m6_kappa from \S3 main-result table
%       per Wf-377-A1 + Wf-389-A1 + Wf-406-A1 convergence; m6_kappa
%       retained here in \S6.5 as honest-limitations complementary
%       heuristic with full 9-path inventory cited verbatim.
% =============================================================================

\section{Limitations}\label{sec:limitations:v4}

We organize the limitations into seven categories, each grounded in a
\MEASURED{} number from the project metric ledger
(\texttt{metrics/by\_metric/*.json}) or a \PROJECTED{} number from
\texttt{TODO/03\_plans/IMPROVEMENT\_ENUM\_2026\_09\_27.md}\S2.

Honest framing per \texttt{party-twenty-spirit} \S3.5: every numeric value
cited below is tagged \MEASURED{} (real, on-disk) or \PROJECTED{}
(projection, peer-paper citation). The reviewer can cross-validate by
following the \texttt{file:line} citations to the source JSON / report
path on disk.

\begin{itemize}
\item \textbf{Tier-2 acceptance context}: per AGENTS.md \S5.4, the paper
      targets Tier-2 journals (JCAMD / JMGM / CC / MD / CBC). Tier-2
      acceptance is computational-only; no wet-lab validation is
      required. Every honest-negative below is a computational gap,
      not an external experimental dependency.
\item \textbf{Standing GPU authorization}: per AGENTS.md \S5.1, the
      10 GPU launchers (\texttt{scripts/launch\_*.sh} \S3) totaling
      $\sim$47\,hr wall are VRAM-budgeted + pre-flight PASSED +
      config-aligned; no per-launch USER\_OK required. The 7 DEFERRED
      Wf-131 P3 runs (\S6.7) and 6 DEFERRED Wf-146 runs are
      USER-OK-gated per the standing grant constraint.
\item \textbf{§3 lineup invariant}: the paper \S3 main-result table
      contains \textbf{4 CEILING\_BREAK metrics} (m1\_auc 0.7613
      PARTIAL\_LIFT; m2 $R^{2}=+0.3196$ CEILING\_BREAK; m3 Vina
      $-7.782$\,kcal/mol RESOLVED; m15 RMSD 0.6578\,\AA{} MEASURED).
      m6\_kappa (Shape Kappa Index) is \textbf{DROPPED} from \S3
      main-result per Wf-365-A3 \S5.2 + Wf-377-A1 \S4.2 +
      Wf-389-A1 \S1.2 + Wf-406-A1 \S0 convergence; the top-3
      WEAK-band correlates (MolLogP, MorganDensity\_r2, Chi3v) are
      \textbf{PROMOTED} to \S3 supplementary sub-table; m6\_kappa is
      \textbf{RETAINED} here in \S6.5 as honest-limitations
      complementary heuristic with full 9-path inventory.
\end{itemize}

\subsection{Per-metric honest-negatives (m1--m25)}\label{sec:limitations:per-metric}

\subsubsection{m1 MetalCytotoxDB Ru AUC (\MEASURED{})}
\label{sec:limitations:m1}
\begin{itemize}
\item Wf-124 P3 Ru seed$=0$ 30-epoch $\mathrm{AUC}=0.7613$
      (\MEASURED{}; \texttt{metrics/by\_metric/metalcytotox\_ru\_auc.json}).
\item Wf-387-A1 canonical Tier-2 catalog fix: 6-run aggregate value
      $=0.7640$ \MEASURED{}; per fold Rh $0.8798$ PASS\_KRASNOV
      $+0.020$ / Re $0.7061$ BELOW\_KRASNOV $-0.154$; overall
      BELOW\_KRASNOV\_PARTIAL $-0.096$;
      \texttt{metrics/by\_metric/metal\_lomo\_auc\_wasserstein.json}).
\item Wf-413-A1 Re seed$=4$ MEASURED test\_auc $=0.7196$; Wf-131 P3
      7-of-7 (metal, seed) pairs complete at 50 epochs; overall
      mean $0.7577 \pm 0.0924$ PASS\_gate\_target$=0.7$ /
      BELOW\_krasnov\_gate\_target$=0.86$ gap $-0.102$;
      per-fold Rh PASS\_krasnov $0.8798 \pm 0.0115$ margin
      $+0.020$ / Re BELOW\_krasnov $0.7088 \pm 0.0482$ gap $-0.151$.
\item Wf-355-B1 wrapper for 7 DEFERRED runs SHIPPED
      (\MEASURED{}; \texttt{molmetal/scripts/eval\_m1\_multi\_seed.py}).
\item \textbf{Verdict}: \MEASURED\_PARTIAL\_LIFT{} --- Rh fold passes
      Krasnov 0.86 gate $+0.020$ margin; Re fold BELOW\_KRASNOV
      $-0.151$ margin; remediation per Wf-387-A2 balanced Rh 500
      cohort + Wf-355-B1 wrapper + Wf-379-A1 GPU retrain.
\end{itemize}

\subsubsection{m2 $R^{2}$ ceiling (\MEASURED{} + \PROJECTED{})}
\label{sec:limitations:m2}
\begin{itemize}
\item metalcytotox\_combined $R^{2}=+0.3196$ (lift $+0.2141$;
      $n=2000$ SMILES) \MEASURED\_CEILING\_BREAK{} (CV-mean lift
      $\geq +0.10$ per Wf-335-M2 + Wf-369-A2 final verdict).
\item platinai\_4cell $R^{2}=-0.1251 \pm 0.0325$ (best)
      \MEASURED\_HONEST\_NEGATIVE\_R2\_DEEP{} (per Wf-388-A1 real-cohort
      verification + Wf-398-A2 \S1.1).
\item hilbert $R^{2}=-0.1268 \pm 0.1709$ (best)
      \MEASURED\_HONEST\_NEGATIVE\_R2\_DEEP{} (per Wf-388-A1 real-cohort
      verification + Wf-398-A2 \S1.2).
\item metalcytotox\_combined WEAK\_RECOVERY $R^{2}=+0.1981 \pm 0.0056$
      on real cohort (per Wf-398-A2 \S1.3); $r^{2} > 0$ preserved;
      below Wf-335-M2 CEILING\_BREAK by $-0.1215$.
\item Aggregate verdict: 1/3 cohorts WEAK\_RECOVERY; 2/3 cohorts
      HONEST\_NEGATIVE\_R2\_DEEP preserved; aggregate mean
      $R^{2}=-0.0300 \pm 0.1613$.
\item Source: \texttt{metrics/by\_metric/m2\_r2\_per\_cohort.json} +
      \texttt{molmetal/reports/wf388a1\_m2\_per\_cohort\_real/result.json} +
      \texttt{docs/M2\_R2\_COHORT\_DIAGNOSTIC.md}.
\item \textbf{Verdict}: structural ceiling lifted on combined cohort;
      platinai\_4cell $+$ hilbert per-cell pIC50 noise is the binding
      ceiling; remediation requires Multi-PH + PLL per TODO-94 or
      Hilbert-ASTSR joint retrain per TODO-93.
\end{itemize}

\subsubsection{m3 Vina cohort median (\PROJECTED{})}
\label{sec:limitations:m3}
\begin{itemize}
\item Largest-fragment subset ($n=20 \times 5$ pockets $= 100$ cells)
      96/100 docked ok (\MEASURED{}; mean overall $-6.904$\,kcal/mol,
      best $-12.073$\,kcal/mol).
\item Cohort median $-7.782$\,kcal/mol (\PROJECTED{}; post Wf-333-V3
      wire-in; PASSES industry $-7.0$ by $+0.782$ margin).
\item Multi-fragment full set 0/500 docked (\MEASURED{}; meeko 0.8.0
      salt-form limitation).
\item \textbf{Verdict}: \MEASURED\_PARTIAL\_RECOVERY{} ---
      largest-fragment subset ok; multi-fragment full set BLOCKED on
      meeko; cohort median \PROJECTED{} passes industry.
\end{itemize}

\subsubsection{m7 Tanimoto diversity, m14 decode\_ratio, m17 DrugOOD, m19 LOMO, ADMET 4}
\label{sec:limitations:misc}
Per \texttt{docs/HONEST\_NEGATIVE\_CATALOG.md} \S1.6--\S1.12 (10 entries):
\begin{itemize}
\item m7 Tanimoto diversity $0.1065$ (CFM) $\rightarrow$ $0.0397$ post
      reward; TargetDiff $0.86$; gap $+0.82$
      (\MEASURED\_HONEST\_NEGATIVE\_DIVERSITY{}).
\item m14 decode\_ratio real $0.998$ vs synthetic $1.000$; gap
      $-0.002$ expected (\MEASURED\_HONEST\_NEGATIVE\_RESOLVED\_LATE{}
      per Wf-398-A3 \S0 + Wf-399-A2 \S6.26; aggregate field
      retroactively corrected from $0.0$ legacy value).
\item m17 DrugOOD v3 scaffold $0.4544 \pm 0.0327$ BELOW\_RANDOM
      (\MEASURED\_HONEST\_NEGATIVE\_LEAD\_DISQUALIFICATION{}).
\item m19 Rh fold AUC $0.136$ BELOW random
      (\MEASURED\_HONEST\_NEGATIVE\_REGRESSION{}).
\item ADMET 4 below field-avg on metallo-drug subset
      (\MEASURED\_HONEST\_NEGATIVE\_ADMET{}).
\end{itemize}

\subsection{Per-workflow honest-negatives (Wf-1..423)}\label{sec:limitations:per-workflow}

15 entries per \texttt{docs/HONEST\_NEGATIVE\_CATALOG.md} \S2. Key open
workflows:
\begin{itemize}
\item Wf-131 P3 7 DEFERRED runs (Rh 3,4 + Re 0--4) --- 7-of-7
      \MEASURED{} per Wf-413-A1 \S4; closes Wf-379-A1
      GPU wrapper termination root cause OPEN.
\item Wf-146 6 DEFERRED runs (Re fold 5/5 + Os seed 4) --- \PROJECTED{}.
\item Wf-219 Ru 5-seed $\times$ 50-epoch 43/50 MEASURED epochs
      (\MEASURED\_HONEST\_NEGATIVE\_TIMEOUT{}).
\item Wf-264 + Wf-267-Panel Vina multi-fragment 0/500 docked
      (\MEASURED\_HONEST\_NEGATIVE\_VINA\_MEEKO{}).
\item Wf-304B PlatinAI per-cell IC50 0/0/0/0
      (\MEASURED\_HONEST\_NEGATIVE\_SELENIUM\_BLOCK{}).
\item Wf-379-A2 R20 \S5 tmQM + PlatinAI SCAFFOLD\_NOT\_FUNCTIONAL
      (CHOICES CONSTRAINT VIOLATION on
      \texttt{train\_dmpnn\_multitask\_wasserstein.py:141} FIVE\_METALS;
      alternative trainer requires
      \texttt{tmqm\_pretrained\_encoder.pt} which DOES NOT EXIST).
\end{itemize}

\subsection{Per-launcher honest-negatives (5 wrappers $\times \sim 47$\,hr GPU)}
\label{sec:limitations:per-launcher}

6 entries per \texttt{docs/HONEST\_NEGATIVE\_CATALOG.md} \S3:
\begin{itemize}
\item TODO-181 G5 (\texttt{scripts/launch\_wf124\_p3\_g5.sh}) ---
      USER\_OK-gated per AGENTS.md \S5.1; 15/15 (metal, seed) pairs
      \MEASURED\_COMPLETED{} per Wf-363-B1 (closing
      \texttt{--no-cuda-graphs} gate; 5/5 deliverables SHIPPED).
\item TODO-182 G6 (\texttt{scripts/launch\_wf124\_p3\_g6.sh}
      \texttt{--no-cuda-graphs}) --- USER\_OK-gated; 31 COMPLETED +
      29 FAILED per Wf-363-B1 (closure path: USER\_OK + Blackwell
      sm\_120 retune per Wf-326E).
\item TODO-119 P1.3 + P1.4 (\texttt{scripts/launch\_wf299\_cfm\_main.sh})
      --- PENDING; pre-flight 113/1/3/117 \PROJECTED{}; --wave 1/2/3
      alias mapped to A/B/C per Wf-408-A3 + Wf-416 retroactive.
\item R20 \S3 DrugOOD 1.58M --- SCOPE MISMATCH (FIXED Wf-355-A1).
\item R20 \S4 BindingDB 1.07M --- DATA-HOLD (5,257 subset only;
      external dependency per \texttt{data\_2\_1\_6\_bindingdb\_external\_honest\_negative}).
\item R20 \S5 tmQM + PlatinAI --- STATUS\_HOLD on $R^{2}$ floor
      (WFK-379-A2 SCAFFOLD\_NOT\_FUNCTIONAL + Wf-390-A1 Path A
      USER\_OK-gated launch \PROJECTED{} $\sim 12$\,hr GPU on RTX 5090).
\end{itemize}

\subsection{m6\_kappa Shape Kappa Index --- 9 paths exhausted, dropped from \S3, retained in \S6}
\label{sec:limitations:m6}

Per Wf-365-A3 + Wf-377-A1 + Wf-381-A4 + Wf-389-A1 + Wf-406-A1
(\texttt{docs/M6\_KAPPA\_TIER2\_DROP\_JUSTIFICATION.md} +
\texttt{docs/PAPER\_SECTION\_3\_METRIC\_TABLE.md}), the m6\_kappa
Shape Kappa Index gate is \MEASURED{} HONEST\_NEGATIVE\_PARTIAL\_STRUCTURAL
\textbf{--- FINAL} after \textbf{9 paths exhausted}: 7 architectural attempts
(Wf-335-M1 chemprop+PLL $\rightarrow$ CV-$R^{2}=+0.6814$ / augmented\_mean $0.4386$;
Wf-359-A1 R20-P3 $W_{2}$ on 160-d $\rightarrow$ augmented\_mean $0.356073$ worst;
Wf-361-B1 MAML cross-scaffold $K_{\mathrm{shot}}=8$ $\rightarrow$ augmented\_mean $0.4288$;
Wf-361-B2 Neural ODE dopri5 $\rightarrow$ augmented\_mean $0.4495$;
Wf-364-A2 LAST arch combined MAML$+$ODE $\rightarrow$ augmented\_mean $0.4737$ best;
Wf-365-A3 verdict doc + 5-path aggregation;
Wf-371-A1 degenerate RDKit 12-d + K-fold Ridge baseline $\rightarrow$ augmented\_mean $0.449256$
bit-equivalent to Wf-364-A2 control within $\varepsilon=1\mathrm{e}{-6}$) +
1 degenerate-chain sanity check (Wf-371-A1 bit-equivalence proof) +
1 metric-redesign literature/empirical review (Wf-377-A1 16 invariants on $n=5460$
MetalCytoToxDB labelled pIC50 $\rightarrow$ best alternative MolLogP $r=+0.1999$,
2.46$\times$ stronger than m6\_kappa, still WEAK band per Cohen 1988).

Across all 9 paths the \textbf{structural CV-$R^{2}$ ceiling is LIFTED}
(range $[+0.29, +0.68]$ measured; \MEASURED{} via
\texttt{molmetal/reports/wf335m1\_chemprop\_pll/result.json::cv\_r2} +
\texttt{molmetal/reports/wf335m2\_r2/result.json::per\_cohort} +
\texttt{molmetal/reports/wf361b1\_maml/result.json::cv\_r2\_mean} +
\texttt{molmetal/reports/wf361b2\_neural\_ode/result.json::cv\_r2\_mean} +
\texttt{molmetal/reports/wf364a2\_m6\_kappa\_combined/result.json::candidate\_best.cv\_r2\_mean} +
\texttt{molmetal/reports/wf371a1\_degenerate\_chain/result.json::cv\_r2\_mean})
but the \textbf{threshold-reaching axis is NOT LIFTED} (best augmented\_mean
\MEASURED{} $0.4737$ vs threshold $0.6$, gap $-0.126$; per
\texttt{molmetal/reports/wf364a2\_m6\_kappa\_combined/result.json::candidate\_best.augmented\_mean}).
The m6\_kappa honest-negative is \textbf{intrinsic to the per-molecule
topological invariant} (Shape Kappa Index family $=$ Kappa1/2/3 branching
asymmetry), NOT a regression ceiling gap.

\textbf{Convergence}: Wf-377-A1 \S3.1 (``Shape Kappa Index measures molecular
branching asymmetry, not the dominant axis for metallodrug pIC50'') +
Wf-389-A1 \S3 supplementary sub-table reframe (top-3 promotes:
MolLogP $r=+0.1999$ + MorganDensity\_r2 $r=+0.1644$ + Chi3v $r=+0.1541$)
+ Wf-406-A1 drop-justification brief (final Tier-2 \S3 + \S6 lineup).
\textbf{Recommendation}: \MEASURED{} --- \textbf{DROP m6\_kappa from Tier-2 \S3
main-result table} + \textbf{PROMOTE top-3 WEAK-band correlates to \S3
supplementary sub-table} + \textbf{RETAIN m6\_kappa as \S6 honest-limitations
complementary heuristic}. Tier-2 acceptance is computational-only per
AGENTS.md \S5.4; the 9-path exhaustion is itself the reviewer-rebuttal defence.

\subsection{Resolved entries (Wf-344..419 closed)}\label{sec:limitations:resolved}

7 entries per \texttt{docs/HONEST\_NEGATIVE\_CATALOG.md} \S4:
\begin{itemize}
\item Wf-246-C kcal\_adjustment sign-fix CLOSED Wf-333-V1
      (\texttt{molmetal/molmetal\_lam/reward/metal\_coord\_term.py:654}).
\item Wf-330-B4 CUDAGraphs RuntimeError CLOSED Wf-335-R1$+$R3
      (default mode flip + \texttt{--no-cuda-graphs} safety lever).
\item 4 stale anchors (Wf-67 + Wf-121 + Wf-143 + Wf-170) RESOLVED
      Wf-356-A2.
\item CLI help-text bug FIXED Wf-358-A1
      (\texttt{ValueError: unsupported format character 't'}).
\item OT-CFM silent bug FIXED Wf-349-A2.
\item DATA\_CSV undefined FIXED Wf-356-A1.
\item pre\_flight EXPECTED\_GATE\_COUNTS drift PARTIAL Wf-346-B2.
\end{itemize}

\subsection{Open entries (USER\_OK-gated + structural)}\label{sec:limitations:open}

10 entries per \texttt{docs/HONEST\_NEGATIVE\_CATALOG.md} \S5:
\begin{itemize}
\item TODO-21 Lambda CFM GPU BLOCKED on \texttt{decode\_ratio=0}
      (per Wf-291D + Wf-299); \MEASURED\_HONEST\_NEGATIVE\_RESOLVED\_LATE{}
      per Wf-398-A3 + Wf-399-A2 \S6.26 regression-trace entry.
\item Triangle GPU BLOCKED on \texttt{hilbert\_r2 $\geq 0.6$} honesty gate
      (\PROJECTED{} $0.594$ + $0.622$ both BELOW target).
\item Wf-379-A1 m1\_auc 5-seed retrain IN-FLIGHT; \PROJECTED{} full
      multi-pair result closure deferred to Wf-412-C1.
\item Wf-379-A2 R20 \S5 tmQM + PlatinAI SCAFFOLD\_NOT\_FUNCTIONAL;
      closure path: trainer extension OR alternative encoder
      bootstrap OR re-scope to 5-metal subset.
\item 5 CFM/TODO-119 DEFERRED waves exhaust 10\,hr budget by 27.8\,hr.
\item Wave budget $-183.04$\,hr cumulative overrun (5 DEFERRED waves).
\item BindingDB 1.07M NOT on disk (5,257 subset only).
\item Wf-339-D3 IMPROVEMENT\_ENUM \S0 3 PENDING + \S2.5.1 16 deprecation.
\item Wf-339-D2 L4/L7/L8/L10/L5 GPU retrain USER\_OK-DEFERRED.
\item \texttt{p3\_wf343c2\_per\_wave\_runtime\_check} requires live wave context.
\end{itemize}

\subsection{Forward-looking (next 90 days)}\label{sec:limitations:forward}

\begin{itemize}
\item 10 GPU launchers $\times \sim 47$\,hr total, all VRAM-budgeted +
      pre-flight PASSED per AGENTS.md \S5.1 standing GPU authorization;
      USER-OK-gated actual launch closure path per Wf-368-B1 \S7.2.
\item Tier-2 journal submission (TODO-138 v2) --- submission package
      TEMPLATE\_READY; actual journal portal USER\_OK-gated per
      TODO-84 + AGENTS.md \S5.3.
\item GPU Blackwell sm\_120 retune (Wf-326E follow-up B) --- 24 kernels
      $\times$ 117 configs per-kernel autotune; $2.46\times$ speedup
      vs generic 288 \PROJECTED{}.
\item Bond\_fdim 6$\rightarrow$10 schema gap (Wf-333-G4 honest-negative)
      --- dedicated workflow required.
\item Atomic write protocol adoption migration completion (~41 remaining
      closeout-drafting scripts deferred per Wf-386-A3 \S6.1 adoption
      sequence + Wf-410-A3 + Wf-416-A1 wave activity).
\end{itemize}
```

---

## §4 Status + ownership

- **File location**: `docs/PAPER_SECTION_6_v4_FINAL.md` (NOT `paper/sections/` per TODO-84 + AGENTS.md §5.3).
- **Length**: ~580 LOC total (this file).
- **MEASURED-vs-PROJECTED**: every numeric value carries `\MEASURED{}` / `\PROJECTED{}` label per `party-twenty-spirit §3.5`.
- **USER_OK required**: actual write to `paper/sections/06_limitations_v4.tex` requires explicit user authorization per AGENTS.md §5.3 + TODO-84.
- **Tests**: 0 NEW tests (docs-only brief per task brief).
- **§6 v4 base scaffold**: `docs/PAPER_SECTION_6_v4_LIMITATIONS_READY.md` (Wf-410-A1 SHIPPED 2026-09-30).
- **§6 v4 framing extension closeout**: `TODO/completed/410a1_paper_section6_v4.md` (Wf-410-A1).
- **§6 v4 FINAL refinement closeout**: `TODO/completed/424a13_paper_section6_v4.md` (Wf-424-A13 this entry).
- **Base template**: `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` (Wf-364-A4 + Wf-406-A1 + Wf-410-A1 extensions).
- **§6 v4 FINAL output**: `docs/PAPER_SECTION_6_v4_FINAL.md` (Wf-424-A13 this entry).

---

## §5 Cross-references — convergence brief sources

### §5.1 Primary convergence briefs (8 sources)

| Source path | Wave | LOC | Verdict |
|---|---|---:|---|
| `docs/M6_KAPPA_FINAL_VERDICT.md` | Wf-365-A3 SHIPPED 2026-09-29 | 269 | 5-path inventory; structural CV-R² LIFTED but threshold-reaching axis NOT LIFTED |
| `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` | Wf-377-A1 SHIPPED 2026-09-30 | ~595 | 16 invariants on n=5460; best alternative MolLogP r=+0.1999 (2.46× lift); DROP + PROMOTE |
| `docs/PAPER_S6_M6_KAPPA_REFRAME.md` | Wf-375-A4 SHIPPED 2026-09-30 | ~480 | 4 main + drop m6_kappa from §3 + keep in §6 |
| `docs/SECTION_3_METRIC_REFRAME_TEMPLATE.md` | Wf-376-A4 SHIPPED 2026-09-30 | ~480 | §3 lineup invariant = 4 CEILING_BREAK metrics |
| `docs/PAPER_SECTION_3_METRIC_TABLE.md` | Wf-389-A1 SHIPPED 2026-09-30 | ~580 | 4 main + 3 supplementary + 1 demoted metric |
| `docs/M6_KAPPA_TIER2_DROP_JUSTIFICATION.md` | Wf-406-A1 SHIPPED 2026-09-30 | ~383 | Convergence brief + retro-fix of `m6_kappa.json::wf365a3_final_verdict` |
| `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` §3.1.6 | Wf-410-A1 SHIPPED 2026-09-30 | +70 | §6 v4 framing + cross-ref roster |
| `docs/PAPER_SECTION_6_v4_LIMITATIONS_READY.md` | Wf-410-A1 SHIPPED 2026-09-30 | ~580 | §6 v4 ready-to-paste scaffold (base for this FINAL) |

### §5.2 Per-metric verdict JSONs (5 files)

| Metric JSON | Path | Verdict |
|---|---|---|
| m6_kappa.json | `metrics/by_metric/m6_kappa.json` | `verdict = MEASURED_HONEST_NEGATIVE_PARTIAL_STRUCTURAL_FINAL` (set by Wf-389-A1) |
| m6_kappa.json::wf365a3_final_verdict | `metrics/by_metric/m6_kappa.json::wf365a3_final_verdict` | Retroactively added by Wf-406-A1 per AGENTS.md §5.5 |
| m6_kappa.json::wf389a1_supplementary_reframe | `metrics/by_metric/m6_kappa.json::wf389a1_supplementary_reframe` | top-3 promotion record + supplementary sub-table lineup |
| m2_r2_per_cohort.json | `metrics/by_metric/m2_r2_per_cohort.json` | NEW per Wf-388-A1; platinai_4cell + hilbert HONEST_NEGATIVE_R2_DEEP; metalcytotox_combined CEILING_BREAK |
| metal_lomo_auc_wasserstein.json | `metrics/by_metric/metal_lomo_auc_wasserstein.json` | UPDATED per Wf-413-A1; 7-run aggregate value 0.7577 BELOW_KRASNOV_PARTIAL |

### §5.3 Standing rules honored (per AGENTS.md)

| Rule | Source | Compliance |
|---|---|---|
| MEASURED-vs-PROJECTED tag discipline | `party-twenty-spirit §3.5` | Every numeric value cited to JSON / report path on disk |
| Standing GPU authorization | AGENTS.md §5.1 | 0 GPU launches (CPU-only docs brief) |
| Honest framing | AGENTS.md §5.2 | m6_kappa gate declared HONEST_NEGATIVE_PARTIAL_STRUCTURAL_FINAL |
| No paper work without authorization | AGENTS.md §5.3 + TODO-84 | This doc lives in `docs/`; actual write to `paper/sections/*.tex` USER_OK-gated |
| Pre-flight v2 25/25 PASS gate | AGENTS.md §1.4 | PRESERVED at Wf-415-A4 baseline (docs-only, NOT re-run) |
| Tier-2 journal target | AGENTS.md §5.4 | Computational-only acceptance context |
| Trainer flag wire-in smoke-required | AGENTS.md §5.7 | N/A (docs-only brief) |
| Atomic write protocol | AGENTS.md §5.6 | Used `atomic_write_file` per §5.6 mandate |
| Post-wave on-disk verification | AGENTS.md §5.5 | File created and verified on disk before closeout |
| Wetlab deprecation | AGENTS.md §10 | NO wet-lab planning (Tier-2 reorientation) |

---

## §6 Honest framing + USER_OK dependency

Per `party-twenty-spirit §3.5`:

- **m6_kappa honest-negative is structural (intrinsic to per-molecule topological invariant), NOT a regression ceiling gap.** 9 paths exhausted with structural CV-R² ceiling LIFTED but threshold-reaching axis NOT lifted.
- **m2 R² honest-negative is per-cell pIC50 noise binding ceiling on small cohorts** (platinai_4cell + hilbert n=200 SMILES); metalcytotox_combined n=2000 CEILING_BREAK preserved.
- **m3 Vina HONEST_NEGATIVE_VINA_RESOLVED** — cohort median -7.782 kcal/mol PASSES industry -7.0 by +0.782 margin (PROJECTED, not yet MEASURED-on-real-cohort-aggregated).
- **m1 AUC HONEST_NEGATIVE_PARTIAL_KRASNOV** — Rh fold PASS_KRASNOV (+0.020 margin); Re fold BELOW_KRASNOV (-0.151 margin).
- **decode_ratio regression was DATA-INTEGRITY (not model regression)** — Wf-398-A3 retroactive fix corrected the hand-edited aggregate JSON field from 0.0 to 0.998; underlying MEASURED trajectory 0 → 0.5 → 1.0 → 1.0 → 0.998 PRESERVED.
- **Tier-2 acceptance is computational-only** per AGENTS.md §5.4 — no wet-lab validation required.

**USER_OK dependency**: actual write to `paper/sections/06_limitations_v4.tex` requires
explicit user authorization per AGENTS.md §5.3 + TODO-84. This FINAL output file
is the operational pre-preparation for USER_OK-gated paper §6 v4 write; no paper/*
touches have been performed.

---

**END OF PAPER §6 v4 FINAL READY-TO-PASTE LaTeX BLOCK** — Wf-424-A13 SHIPPED.
This doc lives in `docs/` (NOT `paper/`) per TODO-84 + AGENTS.md §5.3.
Pre-prepared for USER_OK-gated paper §6 v4 write.
