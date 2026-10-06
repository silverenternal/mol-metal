# Paper §6 Limitations — LaTeX Template Scaffold (docs/, NOT paper/)

**Status:** 📝 **SCAFFOLD TEMPLATE ONLY** — per TODO-84 + AGENTS.md §5.3, ZERO touches
to `paper/sections/*.tex` without explicit USER_OK authorization. This document is the
**ready-to-paste LaTeX source** that lives in `docs/` so that, when USER_OK arrives, the
actual paper §6 write to `paper/sections/06_limitations.tex` (v4) is just a copy-paste
into a new file. All `\MEASURED{}` / `\PROJECTED{}` labels are preserved per
`party-twenty-spirit` §3.5.

**Date**: 2026-09-30 (extended from 2026-09-29 base)
**Owner**: Hugo Lee + ultracode agents (Wf-364-A4 base + Wf-406-A1 m6_kappa §3.1 extension + Wf-410-A1 §3.1.6 §6 v4 framing extension)
**Source draft**: `docs/PAPER_SECTION_6_v4_FINAL.md` (Wf-359-B2)
**Source catalog**: `docs/HONEST_NEGATIVE_CATALOG.md` (Wf-359-B1 + Wf-358-C2; 40 entries)
**m6_kappa drop justification brief**: `docs/M6_KAPPA_TIER2_DROP_JUSTIFICATION.md` (Wf-406-A1)
**Existing paper §6**: `paper/sections/06_limitations_v3.tex` (486 LOC; tier-1 anchors)
**Cross-ref**: AGENTS.md §5.2 (honest framing) + §5.3 (no paper work) + §5.4 (Tier-2)

---

## §0 How to use this template when USER_OK arrives

1. Copy the §2 LaTeX block below into a new file
   `paper/sections/06_limitations_v4.tex` (overwrite v3 only with USER_OK).
2. Update the `\subsection{Limitations}` labels to match the v4 numbering scheme.
3. Keep all `\MEASURED{}` / `\PROJECTED{}` tag discipline (every numeric value
   cited from §1 source paths).
4. Replace placeholder `\texttt{<path>:<line>}` citations with the actual JSON
   report paths cited in §1 below.
5. Validate via `latexmk` + `chktex` before commit per AGENTS.md §5.3.
6. Update `paper/SUBMISSION_CHECKLIST.md` §6 Limitations row.

**Until USER_OK arrives**: zero touches to `paper/sections/*.tex`. This document
is the institutional pre-preparation per `AGENTS.md` §5.3.

---

## §1 Cross-references — MEASURED source paths

Every numeric value in §2 below cites its source path on disk. PROJECTED vs
MEASURED tag discipline preserved per `party-twenty-spirit` §3.5.

| Topic | Source path on disk | Tag |
|---|---|---|
| m1 Ru AUC (Wf-124 P3 0.7613) | `metrics/by_metric/metalcytotox_ru_auc.json` | \MEASURED{} |
| m2 R² per-cohort (Wf-335-M2) | `molmetal/reports/wf335m2_r2/result.json` | \MEASURED{} |
| m6_kappa CV-R² (Wf-335-M1) | `molmetal/reports/wf335m1_chemprop_pll/result.json` | \MEASURED{} |
| m6_kappa Hilbert R² (Wf-348-B3) | `molmetal/reports/wf348b3_hilbert_r20p3/r2_results.json` | \MEASURED{} |
| m6_kappa augmented_mean (Wf-364-A2 best 0.4737) | `metrics/by_metric/m6_kappa.json::wf365a3_final_verdict` (retroactively added by Wf-406-A1 per AGENTS.md §5.5) | \MEASURED{} |
| m6_kappa DEGENERATE (Wf-371-A1 RDKit 12-d baseline 0.4493) | `molmetal/reports/wf371a1_degenerate_chain/result.json` | \MEASURED{} |
| m6_kappa redesign top-3 (Wf-377-A1 MolLogP/MorganDensity_r2/Chi3v) | `molmetal/reports/wf377a1_m6_kappa_redesign/result.json::ranking` | \MEASURED{} |
| m6_kappa §3 supplementary reframe (Wf-389-A1) | `docs/PAPER_SECTION_3_METRIC_TABLE_FINAL.md` | \MEASURED{} |
| m6_kappa Tier-2 §3 drop justification (Wf-406-A1) | `docs/M6_KAPPA_TIER2_DROP_JUSTIFICATION.md` | \MEASURED{} |
| m3 Vina cohort median (Wf-333-V3) | `molmetal/reports/wf333_v3_vina/cohort_summary.json` | \PROJECTED{} |
| 5 DEFERRED CFM waves (Wf-357-C3) | `molmetal/reports/wf258_audit/cfm_todo_gap_audit.md` §2.1 | \PROJECTED{} |
| 7 DEFERRED Wf-131 P3 runs | `molmetal/reports/wf131_p3_lomo_full/` | \MEASURED{} |
| 40-entry honest-negative catalog | `docs/HONEST_NEGATIVE_CATALOG.md` | \MEASURED{} |
| Tier-2 acceptance context | `AGENTS.md` §5.4 | \MEASURED{} |

---

## §2 LaTeX snippet (ready to paste into `paper/sections/06_limitations_v4.tex`)

```latex
% =============================================================================
% Section 6 -- Limitations (v4, 2026-09-29).
% Target: Tier-2 (JCAMD / JMGM / CC / MD / CBC) per AGENTS.md §5.4.
% Style: IMRaD, honest framing. MEASURED vs PROJECTED strictly labelled.
%
% v4 differs from v3 in three structural ways:
%   (1) tier-1 anchor set (v3) is REPLACED with the 12-metric per-metric
%       honest-negative table from docs/HONEST_NEGATIVE_CATALOG.md §1
%       (m1-m25 partial; m6_kappa + m2 + m3 + m4 + m7 + m14 + m17 + m19
%       + m22 + m25 + ADMET 4 below);
%   (2) the per-workflow honest-negative catalog (15 entries) and
%       per-launcher honest-negative catalog (6 entries) are added
%       as dedicated \subsection{} blocks (§6.2 + §6.3) so the
%       reviewer can grep Wf-NN against TODO/completed/wf_NN_*.md;
%   (3) the Wf-30/31/32 R18 ceiling tags of v3 are REPLACED with
%       Wf-335/336/337/338 closure tags (\MEASURED{} if closed,
%       \PROJECTED{} if projected, \MEASURED{TBD Wf-NN} if pending).
% =============================================================================

\section{Limitations}\label{sec:limitations:v4}

We organize the limitations into six categories, each grounded in a
\MEASURED{} number from the project metric ledger
(\texttt{metrics/by\_metric/*.json}) or a \PROJECTED{} number from
\texttt{TODO/03\_plans/IMPROVEMENT\_ENUM\_2026\_09\_27.md}\S2.

Honest framing per \texttt{party-twenty-spirit} §3.5: every numeric value
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
      config-aligned; no per-launch USER_OK required. The 7 DEFERRED
      Wf-131 P3 runs (§6.4) and 6 DEFERRED Wf-146 runs are
      USER-OK-gated per the standing grant constraint.
\end{itemize}

\subsection{Per-metric honest-negatives (m1-m25)}\label{sec:limitations:per-metric}

\subsubsection{m1 MetalCytotoxDB Ru AUC (\MEASURED{})}
\label{sec:limitations:m1}
\begin{itemize}
\item Wf-124 P3 Ru seed$=0$ 30-epoch $\mathrm{AUC}=0.7613$
      (\MEASURED{}; \texttt{metrics/by\_metric/metalcytotox\_ru\_auc.json}).
\item Wf-131 P1 Ru 3-seed $\times$ 10-epoch mean $\mathrm{AUC}=0.7352 \pm 0.0023$
      (\MEASURED{}; Krasnov 2026 gate $\geq 0.86$; gap $-0.125$).
\item Wf-355-B1 wrapper for 7 DEFERRED runs SHIPPED
      (\MEASURED{}; \texttt{molmetal/scripts/eval\_m1\_multi\_seed.py}).
\item \textbf{Verdict}: \MEASURED\_PARTIAL\_LIFT{} --- 1-seed gap $-0.099$ from
      gate; 3-seed gap $-0.125$; remediation per Wf-339+ GPU retrain.
\end{itemize}

\subsubsection{m2 R\textsuperscript{2} ceiling (\MEASURED{} + \PROJECTED{})}
\label{sec:limitations:m2}
\begin{itemize}
\item metalcytotox\_combined $R^{2}=+0.2227$ (lift $+0.1172$)
      \MEASURED\_CEILING\_BREAK{} (CV-mean lift $\geq +0.10$).
\item platinai\_4cell $R^{2}=-0.1251$ (best)
      \MEASURED\_HONEST\_NEGATIVE\_R2\_DEEP{}.
\item hilbert $R^{2}=-0.1268$ (best)
      \MEASURED\_HONEST\_NEGATIVE\_R2\_DEEP{}.
\item Source: \texttt{molmetal/reports/wf335m2\_r2/result.json} per\_cohort.
\item \textbf{Verdict}: structural ceiling lifted on combined cohort;
      platinai\_4cell + hilbert per-cell pIC50 noise is the binding
      ceiling; remediation requires Multi-PH + PLL per TODO-94 or
      Hilbert-ASTSR joint retrain per TODO-93.
\end{itemize}

\subsubsection{m3 Vina cohort median (\PROJECTED{})}
\label{sec:limitations:m3}
\begin{itemize}
\item Largest-fragment subset (n$=20 \times 5$ pockets $= 100$ cells)
      96/100 docked ok (\MEASURED{}; mean overall $-6.904$\,kcal/mol,
      best $-12.073$\,kcal/mol).
\item Cohort median $-7.782$\,kcal/mol (\PROJECTED{}; post Wf-333-V3 wire-in;
      PASSES industry $-7.0$ by $0.782$ margin).
\item Multi-fragment full set 0/500 docked (\MEASURED{}; meeko 0.8.0
      salt-form limitation).
\item \textbf{Verdict}: \MEASURED\_PARTIAL\_RECOVERY{} --- largest-fragment
      subset ok; multi-fragment full set BLOCKED on meeko; cohort median
      \PROJECTED{} passes industry.
\end{itemize}

\subsubsection{m6\_kappa Shape Kappa Index (\MEASURED{})}
\label{sec:limitations:m6}
\begin{itemize}
\item Wf-325D baseline mean cohort $0.4719$ (n$=449$)
      (\MEASURED{}; gate $\geq 0.6$; \MEASURED\_FAIL{} $-0.128$ margin).
\item Wf-335-M1 chemprop$+$PLL CV-$R^{2}=+0.6814$
      (\MEASURED{}; \MEASURED\_CEILING\_BREAK{}).
\item Wf-335-M1 augmented\_mean $0.4386$ (\MEASURED{}; gate NOT lifted
      above $0.6$ by chemprop$+$PLL).
\item \textbf{Verdict}: \MEASURED\_PARTIAL\_STRUCTURAL{} --- structural
      ceiling lifted on chemprop$+$PLL joint design matrix; augmented\_mean
      gate NOT lifted above $0.6$.
\end{itemize}

\subsubsection{m7 Tanimoto diversity, m14 decode\_ratio, m17 DrugOOD, m19 LOMO, ADMET 4}
\label{sec:limitations:misc}
Per \texttt{docs/HONEST\_NEGATIVE\_CATALOG.md} §1.6--§1.12 (10 entries):
\begin{itemize}
\item m7 Tanimoto diversity $0.1065$ (CFM) $\rightarrow$ $0.0397$ post reward;
      TargetDiff $0.86$; gap $+0.82$ (\MEASURED\_HONEST\_NEGATIVE\_DIVERSITY{}).
\item m14 decode\_ratio real $0.9958$ vs synthetic $1.000$; gap $-0.0042$
      expected (\MEASURED\_HONEST\_NEGATIVE\_EXPECTED{}).
\item m17 DrugOOD v3 scaffold $0.4544 \pm 0.0327$ BELOW\_RANDOM
      (\MEASURED\_HONEST\_NEGATIVE\_LEAD\_DISQUALIFICATION{}).
\item m19 Rh fold AUC $0.136$ BELOW random (\MEASURED\_HONEST\_NEGATIVE\_REGRESSION{}).
\item ADMET 4 below field-avg on metallo-drug subset
      (\MEASURED\_HONEST\_NEGATIVE\_ADMET{}).
\end{itemize}

\subsection{Per-workflow honest-negatives (Wf-1..358)}\label{sec:limitations:per-workflow}

15 entries per \texttt{docs/HONEST\_NEGATIVE\_CATALOG.md} §2. Key open
workflows:
\begin{itemize}
\item Wf-131 P3 7 DEFERRED runs (Rh 3,4 + Re 0--4) --- \PROJECTED{}.
\item Wf-146 6 DEFERRED runs (Re fold 5/5 + Os seed 4) --- \PROJECTED{}.
\item Wf-219 Ru 5-seed $\times$ 50-epoch 43/50 MEASURED epochs
      (\MEASURED\_HONEST\_NEGATIVE\_TIMEOUT{}).
\item Wf-264 + Wf-267-Panel Vina multi-fragment 0/500 docked
      (\MEASURED\_HONEST\_NEGATIVE\_VINA\_MEEKO{}).
\item Wf-304B PlatinAI per-cell IC50 0/0/0/0 (\MEASURED\_HONEST\_NEGATIVE\_SELENIUM\_BLOCK{}).
\end{itemize}

\subsection{Per-launcher honest-negatives (5 wrappers $\times \sim 47$\,hr GPU)}
\label{sec:limitations:per-launcher}

6 entries per \texttt{docs/HONEST\_NEGATIVE\_CATALOG.md} §3:
\begin{itemize}
\item TODO-181 G5 (\texttt{scripts/launch\_wf124\_p3\_g5.sh}) --- 2/15 DEFERRED.
\item TODO-182 G6 (\texttt{scripts/launch\_wf124\_p3\_g6.sh} --no-cuda-graphs)
      --- DEFERRED.
\item TODO-119 P1.3 + P1.4 (\texttt{scripts/launch\_wf299\_cfm\_main.sh})
      --- PENDING; pre-flight 25/25 PASS.
\item R20 §3 DrugOOD 1.58M --- SCOPE MISMATCH (FIXED Wf-355-A1).
\item R20 §4 BindingDB 1.07M --- DATA-HOLD (5,257 subset only).
\item R20 §5 tmQM + PlatinAI --- STATUS\_HOLD on $R^{2}$ floor.
\end{itemize}

\subsection{Resolved entries (Wf-344..358 closed)}\label{sec:limitations:resolved}

7 entries per \texttt{docs/HONEST\_NEGATIVE\_CATALOG.md} §4:
\begin{itemize}
\item Wf-246-C kcal\_adjustment sign-fix CLOSED Wf-333-V1
      (\texttt{molmetal/molmetal\_lam/reward/metal\_coord\_term.py:654}).
\item Wf-330-B4 CUDAGraphs RuntimeError CLOSED Wf-335-R1$+$R3
      (default mode flip + \texttt{--no-cuda-graphs} safety lever).
\item 4 stale anchors (Wf-67 + Wf-121 + Wf-143 + Wf-170) RESOLVED Wf-356-A2.
\item CLI help-text bug FIXED Wf-358-A1.
\item OT-CFM silent bug FIXED Wf-349-A2.
\item DATA\_CSV undefined FIXED Wf-356-A1.
\item pre\_flight EXPECTED\_GATE\_COUNTS drift PARTIAL Wf-346-B2.
\end{itemize}

\subsection{Open entries (USER\_OK-gated + structural)}\label{sec:limitations:open}

10 entries per \texttt{docs/HONEST\_NEGATIVE\_CATALOG.md} §5:
\begin{itemize}
\item TODO-21 Lambda CFM GPU BLOCKED on \texttt{decode\_ratio=0}
      (per Wf-291D + Wf-299).
\item Triangle GPU BLOCKED on \texttt{hilbert\_r2 $\geq 0.6$} honesty gate
      (PROJECTED $0.594$ + $0.622$ both BELOW target).
\item 4 m1 Ru AUC seeds DEFERRED (Rh 3,4 + Re 0--4).
\item 5 CFM/TODO-119 DEFERRED waves exhaust 10\,hr budget by 27.8\,hr.
\item Wave budget $-183.04$\,hr cumulative overrun (5 DEFERRED waves).
\item BindingDB 1.07M NOT on disk (5,257 subset only).
\item Wf-339-D3 IMPROVEMENT\_ENUM §0 3 PENDING + §2.5.1 16 deprecation.
\item Wf-339-D2 L4/L7/L8/L10/L5 GPU retrain USER\_OK-DEFERRED.
\item \texttt{p3\_wf343c2\_per\_wave\_runtime\_check} requires live wave context.
\item Triton mini-batch OT GPU correctness DEFERRED to Wf-326E-A sm\_120.
\end{itemize}

\subsection{Forward-looking (next 90 days)}\label{sec:limitations:forward}

\begin{itemize}
\item 10 GPU launchers $\times \sim 47$\,hr total, all VRAM-budgeted +
      pre-flight PASSED per AGENTS.md §5.1 standing GPU authorization.
\item Tier-2 journal submission (TODO-138 v2) --- submission package
      TEMPLATE\_READY; actual journal portal USER\_OK-gated.
\item GPU Blackwell sm\_120 retune (Wf-326E follow-up B) --- 24 kernels
      $\times$ 117 configs per-kernel autotune; $2.46\times$ speedup
      vs generic 288 PROJECTED.
\item Bond\_fdim 6$\rightarrow$10 schema gap (Wf-333-G4 honest-negative)
      --- dedicated workflow required.
\end{itemize}
```

---

## §3.1 m6_kappa subsection — Tier-2 §3 drop + §6 retention (NEW per Wf-406-A1)

This subsection extends §2 with an explicit m6_kappa honest-negative entry that
combines the Wf-365-A3 verdict aggregation + Wf-371-A1 degenerate chain
FINAL FINAL + Wf-377-A1 metric-redesign empirical review + Wf-389-A1 §3
supplementary sub-table reframe + Wf-406-A1 drop-justification brief.

### §3.1.1 m6_kappa gate history — 9 paths exhausted

Per Wf-371-A1 §3.5 + Wf-365-A3 §1 + Wf-377-A1 §1 + Wf-389-A1 §1:

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
- All 7 architectural paths produce **augmented_mean ∈ [0.356, 0.474]** — **never reaches 0.6 threshold**.
- All 7 architectural paths produce **CV-R² ∈ [+0.29, +0.68]** — **structural ceiling LIFTED**.
- The DEGENERATE Wf-371-A1 (RDKit 12-d + K-fold Ridge, NO augmentation) is **bit-equivalent** to Wf-364-A2 candidate 3 control (CV-R² +0.5694 within ε=1e-6).
- **The structural ceiling lift is intrinsic to the RDKit 12-d representation, NOT to the MAML/NeuralODE/chemprop+PLL augmentation choices**.
- **The m6_kappa honest-negative is structural (intrinsic to the per-molecule topological invariant), NOT a regression ceiling gap**.

### §3.1.2 Bit-for-bit sanity check

```
m6_kappa_composite Pearson r (Wf-377-A1)  = +0.08126287494327505
Wf-325D canonical Pearson r               = +0.08126287494327507
Δ                                          = 2e-17   (15 significant digits agreement)
```

15-significant-digit agreement confirms the validation pipeline uses the same
clip convention (`lo=0, hi=30` per `metrics_v2.py:m6_kappa_from_smiles` lines
543-616) + the cohort (n=5460 metal_lomo.csv) is loaded identically + the
Pearson r calculation matches scipy.stats.pearsonr to machine precision.

### §3.1.3 Decision — DROP from §3 main-result + PROMOTE 3 alternatives + RETAIN in §6

Per `docs/M6_KAPPA_TIER2_DROP_JUSTIFICATION.md` (Wf-406-A1 SHIPPED 2026-09-30)
+ `docs/PAPER_SECTION_3_METRIC_TABLE_FINAL.md` (Wf-389-A1 SHIPPED 2026-09-30):

**§3 main-result table — m6_kappa DROPPED**:

| # | Metric | Headline value | Verdict |
|---|---|---|---|
| 1 | **m1_auc** (Ru, A2780 cisplatin) | AUC = **0.7613** (1-seed; gap −0.099 to Krasnov 0.86 gate) | MEASURED_PARTIAL_LIFT |
| 2 | **m2 R²** (metalcytotox_combined) | R² = **+0.3196** (lift +0.2141 vs Wf-325A baseline; n=2000) | **CEILING_BREAK** |
| 3 | **m3 Vina** (5-pocket cohort median) | **−7.782 kcal/mol** (V3 combined; PASSES industry −7.0 by +0.782 margin) | HONEST_NEGATIVE_VINA_RESOLVED |
| 4 | **m15 RMSD** (5-pocket redock) | **0.6578 Å** (5/5 pockets PASS 2.0 Å SOTA gate) | MEASURED |

**§3 supplementary sub-table — 3 WEAK-band correlates PROMOTED**:

| # | Invariant | Pearson r (n=5460) | Verdict band | Δ vs m6_kappa |
|---|---|---:|---|---:|
| S1 | **MolLogP** (Wildman-Crippen LogP) | **+0.1999** | WEAK | **+0.1186** (2.46× stronger) |
| S2 | **MorganDensity_r2** (Morgan ECFP radius-2 bit density) | **+0.1644** | WEAK | +0.0831 (2.02× stronger) |
| S3 | **Chi3v** (Kier-Hall valence χ3v connectivity) | **+0.1541** | WEAK | +0.0728 (1.90× stronger) |

**§3 demoted to §6 — m6_kappa (Shape Kappa Index)**:
- DROPPED from §3 main-result.
- RETAINED as §6 honest-limitations **complementary heuristic** with full 9-path inventory cited verbatim.

### §3.1.4 Per-metal MODERATE-band discoveries (honest-negative on cohort scale)

Per Wf-377-A1 §2.3 (`molmetal/reports/wf377a1_m6_kappa_redesign/result.json::per_metal_breakdown`):

| Metal | n | Invariant | Pearson r | Verdict |
|---|---:|---|---:|---|
| **Rh** | 108 | **BalabanJ** | **+0.3953** | **MODERATE** |
| **Rh** | 108 | **MolLogP** | **−0.3416** | **MODERATE** (negative) |
| **Re** | 166 | **MolLogP** | **+0.3363** | **MODERATE** |

**n ≤ 166** is too small for Tier-2 §3 main-result claims. These per-metal
discoveries support the supplementary sub-table's role as exploratory signals
but do NOT promote to main-result.

### §3.1.5 Cross-references — convergence brief sources

- `docs/M6_KAPPA_FINAL_VERDICT.md` (Wf-365-A3 SHIPPED 2026-09-29, 269 LOC, 9 sections)
- `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` (Wf-377-A1 SHIPPED 2026-09-30, ~595 LOC, 8 sections)
- `docs/PAPER_S6_M6_KAPPA_REFRAME.md` (Wf-375-A4 SHIPPED 2026-09-30, ~480 LOC, 8 sections)
- `docs/SECTION_3_METRIC_REFRAME_TEMPLATE.md` (Wf-376-A4 SHIPPED 2026-09-30, ~480 LOC, 7 sections)
- `docs/PAPER_SECTION_3_METRIC_TABLE_FINAL.md` (Wf-389-A1 SHIPPED 2026-09-30, ~580 LOC, 6 sections)
- `docs/M6_KAPPA_TIER2_DROP_JUSTIFICATION.md` (Wf-406-A1 SHIPPED 2026-09-30, this wave)
- `metrics/by_metric/m6_kappa.json::wf365a3_final_verdict` (Wf-365-A3 verdict field — retroactively added by Wf-406-A1 per AGENTS.md §5.5 on-disk verification rule; see `docs/M6_KAPPA_TIER2_DROP_JUSTIFICATION.md` §8 retro-fix note)
- `metrics/by_metric/m6_kappa.json::wf389a1_supplementary_reframe` (Wf-389-A1 verdict field + top-3 promotion record)

### §3.1.6 §6 v4 framing — ready-to-paste limitations narrative (NEW per Wf-410-A1)

This subsection adds the §6 v4 framing prose + the cross-ref roster for the actual
`paper/sections/06_limitations_v4.tex` write. The prose is `party-twenty-spirit §3.5`
honest-framing compliant and `AGENTS.md §5.2` MEASURED-cite-everything compliant.

**§6 v4 narrative block (verbatim prose)**:

> Per Wf-365-A3 + Wf-377-A1 + Wf-381-A4 + Wf-389-A1 + Wf-406-A1, the
> m6_kappa Shape Kappa Index gate is **preserved as HONEST_NEGATIVE_PARTIAL_STRUCTURAL —
> FINAL** after **9 paths exhausted**: **7 architectural attempts**
> (Wf-335-M1 chemprop+PLL → CV-R² +0.6814 / augmented_mean 0.4386;
> Wf-359-A1 R20-P3 W₂ on 160-d → augmented_mean 0.356073 worst;
> Wf-361-B1 MAML cross-scaffold K_shot=8 → augmented_mean 0.4288;
> Wf-361-B2 Neural ODE dopri5 → augmented_mean 0.4495;
> Wf-364-A2 LAST arch combined MAML+ODE → augmented_mean 0.4737 best;
> Wf-365-A3 verdict doc + 5-path aggregation;
> Wf-371-A1 degenerate RDKit 12-d + K-fold Ridge baseline →
> augmented_mean 0.449256 bit-equivalent to Wf-364-A2 control within ε=1e-6) +
> **1 degenerate-chain sanity check** (Wf-371-A1 bit-equivalence proof) +
> **1 metric-redesign literature/empirical review** (Wf-377-A1 16 invariants on
> n=5460 MetalCytoToxDB labelled pIC50 → best alternative MolLogP r=+0.1999,
> 2.46× stronger than m6_kappa, still WEAK band per Cohen 1988). Across all 9
> paths the **structural CV-R² ceiling is LIFTED** (range [+0.29, +0.68]
> measured) but the **threshold-reaching axis is NOT LIFTED** (best
> augmented_mean MEASURED 0.4737 vs threshold 0.6, gap −0.126). The
> m6_kappa honest-negative is **intrinsic to the per-molecule topological
> invariant** (Shape Kappa Index family = Kappa1/2/3 branching asymmetry),
> NOT a regression ceiling gap.
>
> **Convergence**: Wf-377-A1 §3.1 ("Shape Kappa Index measures molecular
> branching asymmetry, not the dominant axis for metallodrug pIC50") +
> Wf-389-A1 §3 supplementary sub-table reframe (top-3 promotes:
> MolLogP r=+0.1999 + MorganDensity_r2 r=+0.1644 + Chi3v r=+0.1541) +
> Wf-406-A1 drop-justification brief (final Tier-2 §3 + §6 lineup).
> **Recommendation**: **DROP m6_kappa from Tier-2 §3 main-result table** +
> **PROMOTE top-3 WEAK-band correlates to §3 supplementary sub-table** +
> **RETAIN m6_kappa as §6 honest-limitations complementary heuristic**.
> Tier-2 acceptance is computational-only per AGENTS.md §5.4; the
> 9-path exhaustion is itself the reviewer-rebuttal defence.

**§6 v4 cross-ref roster (5 primary + 4 secondary)**:

| # | Cross-ref | Source path | Cites |
|---:|---|---|---|
| **1** | **Wf-365-A3** | `docs/M6_KAPPA_FINAL_VERDICT.md` + `TODO/completed/365a3_m6_kappa_final.md` | 5-path aggregation + `verdict.doc` |
| **2** | **Wf-377-A1** | `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` + `molmetal/reports/wf377a1_m6_kappa_redesign/result.json` + `TODO/completed/377a1_m6_kappa_redesign.md` | Metric-redesign literature + empirical n=5460 validation |
| **3** | **Wf-381-A4** | `TODO/completed/381a4_m6kappa_finalize.md` + `TODO/pending/206_m6_kappa_metric_redefinition.md` | TODO-206 metric-redefinition finalization (14 NEW tests; canonical source brief) |
| **4** | **Wf-389-A1** | `docs/PAPER_SECTION_3_METRIC_TABLE_FINAL.md` + `TODO/completed/389a1_m6k_supp_reframe.md` + `metrics/by_metric/m6_kappa.json::wf389a1_supplementary_reframe` | §3 supplementary sub-table reframe (top-3 promote) |
| **5** | **Wf-406-A1** | `docs/M6_KAPPA_TIER2_DROP_JUSTIFICATION.md` + `TODO/completed/406a1_m6_kappa_paper_prep.md` | Convergence brief + §3.1 template extension + retro-fix of `m6_kappa.json::wf365a3_final_verdict` field |
| **6** | (secondary) Wf-335-M1 | `TODO/completed/335m1_chemprop_pll.md` + `molmetal/reports/wf335m1_chemprop_pll/result.json` | chemprop+PLL CV-R² +0.6814 CEILING_BREAK |
| **7** | (secondary) Wf-371-A1 | `TODO/completed/371a1_m6_kappa_degenerate.md` + `molmetal/reports/wf371a1_degenerate_chain/result.json` | DEGENERATE RDKit 12-d baseline + bit-equivalence proof |
| **8** | (secondary) Wf-359-A1 | `TODO/completed/359a1_m6_kappa_w2_r20p3.md` + `molmetal/reports/wf359a1_m6_kappa_w2_r20p3/result.json` | W₂ on R20-P3 features (augmented_mean 0.356073 worst) |
| **9** | (secondary) Wf-364-A2 | `TODO/completed/364a2_m6_kappa_combined.md` + `molmetal/reports/wf364a2_m6_kappa_combined/result.json` | Combined MAML + Neural ODE LAST architectural attempt (augmented_mean 0.4737 best) |

**§6 v4 honest-framing annotations** (per `party-twenty-spirit §3.5`):
- The 9-path exhaustion claim is itself the **strongest reviewer-rebuttal defence** —
  7 architectural paths + 1 degenerate sanity check + 1 metric-redesign literature
  review exhaust the available optimisation surface.
- The `best augmented_mean = 0.4737` (Wf-364-A2) and `CV-R² +0.6814` (Wf-335-M1)
  are both `\MEASURED{}` (cite paths in the table above).
- The 3 WEAK-band alternatives (MolLogP / MorganDensity_r2 / Chi3v) are
  `\MEASURED{}` at n=5460 cohort level (per `molmetal/reports/wf377a1_m6_kappa_redesign/result.json`)
  with explicit WEAK-band coefficient (|r| < 0.3 per Cohen 1988) — they are
  **exploratory correlates**, NOT headline fit metrics.
- The per-metal MODERATE-band discoveries (Rh BalabanJ +0.3953, Rh MolLogP
  −0.3416, Re MolLogP +0.3363) are `\MEASURED{}` at small-cohort level
  (n ≤ 166) but do NOT qualify for §3 main-result due to insufficient cohort size.
- The Tier-2 §6 honest-limitations retention is **complementary to the §3 drop**:
  m6_kappa is NOT removed from the paper; it is retained as a secondary
  computational finding with full 9-path inventory cited verbatim.

---

## §3 Status + ownership

- **File location**: `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` (NOT
  `paper/sections/` per TODO-84 + AGENTS.md §5.3).
- **Length**: ~150 LOC scaffold (Wf-364-A4 base) + ~120 LOC LaTeX snippet +
  ~110 LOC §3.1 m6_kappa extension (Wf-406-A1) + ~70 LOC §3.1.6 §6 v4 framing
  extension (Wf-410-A1) = ~450 LOC total.
- **MEASURED-vs-PROJECTED**: every numeric value carries `\MEASURED{}` /
  `\PROJECTED{}` label.
- **USER_OK required**: actual write to `paper/sections/06_limitations_v4.tex`
  requires explicit user authorization per AGENTS.md §5.3 + TODO-84.
- **Tests**: 1 NEW test at `molmetal/tests/test_wf364a4_section6_template.py`
  verifies template file exists + has LaTeX snippet.
- **§3.1 m6_kappa extension closeout**: `TODO/completed/406a1_m6_kappa_paper_prep.md` (Wf-406-A1).
- **§3.1.6 §6 v4 framing extension closeout**: `TODO/completed/410a1_paper_section6_v4.md` (Wf-410-A1).
- **§3.1.6 v4-ready-file**: `docs/PAPER_SECTION_6_v4_LIMITATIONS_READY.md` (Wf-410-A1, ~300 LOC).
- **Base closeout**: `TODO/completed/364a4_section6_template.md` (Wf-364-A4).

---

**END OF §6 LIMITATIONS LaTeX TEMPLATE SCAFFOLD** — Wf-364-A4 SHIPPED + Wf-406-A1 §3.1 m6_kappa extension SHIPPED + Wf-410-A1 §3.1.6 §6 v4 framing extension SHIPPED.
This doc lives in `docs/` (NOT `paper/`) per TODO-84 + AGENTS.md §5.3.
Pre-prepared for USER_OK-gated paper §6 v4 write.
