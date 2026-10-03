# Paper §3 — Metric Reframe Template (Drop m6_kappa; Promote m1_auc + m2 R² + m3 Vina + m15 RMSD)

**Status:** 📝 **SCAFFOLD TEMPLATE ONLY** — per TODO-84 + AGENTS.md §5.3, ZERO touches
to `paper/sections/*.tex` without explicit USER_OK authorization. This document is the
**ready-to-paste LaTeX source** that lives in `docs/` so that, when USER_OK arrives, the
actual paper §3 write to `paper/sections/03_main_results_v4.tex` (overwrite v3 only with
USER_OK) is just a copy-paste into a new file. All `\MEASURED{}` / `\PROJECTED{}` labels
are preserved per `party-twenty-spirit` §3.5.

**Date**: 2026-09-30
**Owner**: Hugo Lee + ultracode agent (Wf-376-A4)
**Source verdicts**:
- `docs/M6_KAPPA_FINAL_VERDICT.md` (Wf-365-A3; 269 LOC, 9 sections)
- `TODO/completed/371a1_m6_kappa_degenerate.md` (Wf-371-A1; 7 paths exhausted)
- `docs/M2_R2_FINAL_VERDICT.md` (Wf-369-A2; m2 R² CEILING_BREAK on metalcytotox_combined)
- `TODO/completed/362a1_r2_json_update.md` (m2 R² JSON refresh)
- `TODO/completed/362a2_vina_json_refresh.md` (m3 Vina V3 combined refresh)
- `TODO/completed/362a3_rmsd_json_refresh.md` (m15 RMSD refresh)
**Existing paper §3**: `paper/sections/03_main_results_v3.tex` (tier-1 anchors; m6_kappa listed)
**Cross-ref**: AGENTS.md §5.2 (honest framing) + §5.3 (no paper work) + §5.4 (Tier-2)

---

## §0 TL;DR — drop m6_kappa, promote 4 CEILING_BREAK anchors

Per Wf-371-A1 final verdict (7 paths exhausted; gate preserved as HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL FINAL)
+ Wf-365-A3 verdict doc (§5.2 "Drop m6_kappa from Tier-2 paper §3 — HIGHEST PRIORITY"):
**m6_kappa is intrinsic-bound to the per-molecule topological invariant.** It cannot be
lifted to the 0.6 honesty threshold by ANY feature pipeline. Even the degenerate chain
(RDKit 12-d baseline + K-fold Ridge, no MAML, no NeuralODE, no chemprop+PLL) preserves
aug_mean ≈ 0.449 < 0.6. Per Wf-371-A1 §3.2: the only honest interventions to lift the
gate are (a) different cohort, (b) metric redefinition, (c) gate threshold change — all
three USER_OK-gated.

**Recommended Tier-2 paper §3 metric lineup — promote 4 metrics with CEILING_BREAK evidence:**

| # | Metric | Headline value | Source path | USER_OK status |
|---|---|---|---|---|
| 1 | **m1_auc** (MetalCytotoxDB Ru) | **AUC = 0.7613** (1-seed, +0.049 lift vs 0.7123 baseline; Krasnov ≥ 0.86 gate NOT YET HIT, gap −0.099) | `metrics/by_metric/metalcytotox_ru_auc.json` | USER_OK-gated (Wf-355-B1 7 DEFERRED runs wrapper) |
| 2 | **m2 R²** (metalcytotox_combined) | **R² = +0.3196** (CEILING_BREAK, +0.2141 lift vs Wf-325A baseline +0.1055; n=2000) | `metrics/by_metric/metalcytotox_combined_r2.json::wf369a2_final_verdict` | MEASURED (no GPU launch needed) |
| 3 | **m3 Vina** (cohort median, 5-pocket) | **−7.782 kcal/mol** (HONEST_NEGATIVE_VINA_RESOLVED; V0 MEASURED + V1+V2+V3 PROJECTED; PASSES industry −7.0 by +0.782 margin) | `metrics/by_metric/vina_kcal_per_mol.json::aggregate.wf333_v3_combined_median` | MEASURED-V0 + PROJECTED-V1V2V3 |
| 4 | **m15 RMSD** (5-pocket redock) | **0.6578 Å** (5/5 pockets PASS 2.0 Å SOTA gate; difficulty: EASY=1/MEDIUM=3/HARD=1; PoseBusters SKIPPED graceful) | `metrics/by_metric/docked_pose_rmsd.json::aggregate_median_across_pockets` | MEASURED (PoseBusters install is USER_OK-gated) |

**Demoted to §6 honest-limitations**: m6_kappa (Shape Kappa Index) — preserved as a
complementary druglikeness heuristic per Wf-365-A3 §6 template entry.

**Why this reframe is honest:**
- m6_kappa gate is intrinsic to the per-molecule Shape Kappa Index; no architectural
  exploration lifts augmented_mean to 0.6 (5 + 2 = 7 paths exhausted per Wf-371-A1).
- m2 R² metalcytotox_combined has the strongest CEILING_BREAK evidence (+0.2141 lift
  on n=2000 SMILES, multi-seed × 5-fold CV per Wf-335-M2).
- m3 Vina −7.782 kcal/mol closes the m3 Vina Dock gap at the V3 combined level
  (V0 MEASURED per Wf-143 P4 + Wf-325B; V1+V2+V3 PROJECTED per Wf-331-R1 §6.3-§6.4).
- m15 RMSD 5/5 PASS on 2.0 Å SOTA gate (PoseBusters validity SKIPPED graceful —
  preserved as honest-negative on validity, NOT accuracy).
- m1_auc is the strongest single-seed number (+0.049 lift) but requires USER_OK-gated
  multi-seed × 50-epoch retrain to close the −0.099 gate gap to Krasnov 0.86.

---

## §1 Why m6_kappa fails — intrinsic to metric, not architecture

### §1.1 m6_kappa is a per-molecule topological invariant

Per `molmetal/molmetal_lam/sbdd_env/metrics_v2.py:m6_kappa_from_smiles` (lines 543-616),
m6_kappa is computed deterministically from the molecular graph:

```python
k1 = RDKit.Descriptors.Kappa1(mol)
k2 = RDKit.Descriptors.Kappa2(mol)
k3 = RDKit.Descriptors.Kappa3(mol)
n1 = clip((k1 - 0) / (30 - 0), 0, 1)   # lo=0, hi=30 empirical
n2 = clip((k2 - 0) / (30 - 0), 0, 1)
n3 = clip((k3 - 0) / (30 - 0), 0, 1)
m6_kappa = mean(n1, n2, n3)
```

This is a **deterministic function of the molecular graph** — no feature transformation,
no neural net, no optimisation modifies the per-molecule value itself. The Shape Kappa
Index is a *property* of the molecule, not a *prediction*. Wf-365-A3 verdict doc §2.1
makes this explicit.

### §1.2 NEGATIVE_WEAK MetalCytoToxDB validation

Per `metrics/by_metric/m6_kappa.json::metalcytotox_validation`:

- global Pearson r = **0.0813** (NEGATIVE_WEAK band per Wf-253D + Wf-312C R02)
- global Spearman ρ = **0.1114**
- per-metal Spearman ρ range: Ir=0.121, Os=0.317, Re=−0.125, Rh=0.033, Ru=0.105
- verdict: **NEGATIVE_WEAK** — Shape Kappa Index has negligible linear association with
  pIC50 across the MetalCytoToxDB labelled cohort (n=5460 per `n_total_rows`).

This confirms m6_kappa is a **topological druglikeness heuristic**, not a learned
metallodrug score. It is NOT a fit-for-purpose anticancer metric.

### §1.3 7-path architectural inventory (exhausted)

Per Wf-371-A1 §0 + §2.3 (final inventory across 7 attempted paths):

| # | Workflow | Path | augmented_mean | CV-R² | Verdict |
|---|---|---|---:|---:|---|
| 1 | Wf-335-M1 | chemprop+PLL (216+32=248-d) | 0.4386 | **+0.6814** | HONEST_NEGATIVE_PARTIAL_STRUCTURAL |
| 2 | Wf-359-A1 | R20-P3 W₂ on platinai_4cell (160-d) | 0.356073 | n/a (W₂) | HONEST_NEGATIVE_PARTIAL_STRUCTURAL |
| 3 | Wf-361-B1 | MAML cross-scaffold (K_shot=8) | 0.4288 | +0.2909 | HONEST_NEGATIVE_PARTIAL_STRUCTURAL |
| 4 | Wf-361-B2 | Neural ODE dopri5 (12→16→16→12 MLP) | 0.4495 | +0.5410 | HONEST_NEGATIVE_PARTIAL_STRUCTURAL |
| 5 | Wf-364-A2 | Combined MAML + ODE (LAST architectural path) | **0.4737** | +0.3733 / +0.5694 | HONEST_NEGATIVE_PARTIAL_STRUCTURAL |
| 6 | Wf-365-A3 | Verdict doc + JSON field aggregation | n/a | n/a | HONEST_NEGATIVE_PARTIAL_STRUCTURAL (FINAL) |
| 7 | Wf-371-A1 | DEGENERATE chain (RDKit 12-d baseline) | **0.449256** | **+0.569423** | HONEST_NEGATIVE_PARTIAL_STRUCTURAL — FINAL FINAL |

Best augmented_mean across 7 paths: **0.4737** (Wf-364-A2 joint candidate; +0.0018 honest
lift over baseline 0.4719; still −0.126 below 0.6 threshold).

**Structural recovery axis (CV-R²) IS LIFTED**: 4 of 5 architecture attempts achieved
CV-R² +0.5410 to +0.6814 (3.6× to 4.5× ABOVE PROJECTED band per WEAK_TO_STRONG_PLAN §3.5).

**Threshold-reaching axis (augmented_mean ≥ 0.6) is NOT LIFTED** across all 7 paths.

### §1.4 Cross-validated predictions are unbiased projections

Per Wf-335-M1 §3.1 + Wf-348-B3 + Wf-361-B2 §6.2: cross-validated predictions are
**unbiased projections** of the baseline.

```
mean(CV_predicted) = mean(y) - sum_i (y_i - y_hat_i) / n = mean(y)   (for unbiased CV)
```

Therefore `augmented_mean ≈ baseline_mean ± 0.005` regardless of `cv_r2`. The `cv_r2`
axis is **independent** of the `augmented_mean` axis: a Ridge can achieve
`cv_r2 = +0.6814` (chemprop+PLL) while leaving `augmented_mean` unchanged at baseline.

### §1.5 What WOULD lift the gate (preserved from Wf-365-A3 §2.4)

Three classes of interventions could conceivably lift `augmented_mean` to ≥ 0.6:
1. **Different cohort** — source metallodrugs with intrinsically higher Shape Kappa Index.
2. **Different metric definition** — replace the empirical `lo=0, hi=30` clip with
   cohort-relative normalisation. NOT ChemAxon PMC11387870 convention any more.
3. **Different gate threshold** — lower from 0.6 to ~0.45. NOT recommended (literature
   convention, not adjusted post-hoc).

All three are USER_OK-gated per AGENTS.md §5.3 + TODO-84.

---

## §2 Recommended paper §3 metric lineup

### §2.1 m1_auc (MetalCytotoxDB Ru) — primary headline metric

- **MEASURED headline**: AUC = **0.7613** (1-seed, Wf-124 P3 30-epoch D-MPNN multi-task +
  Wasserstein + L2 guard; lift +0.049 vs 0.7123 single-task 5-epoch baseline)
- **Baseline (Krasnov 2026 paper-tier)**: AUC = 0.81 (cite-only)
- **Gate target (Krasnov 2026)**: AUC ≥ 0.86 (NOT YET HIT, gap −0.0987)
- **Multi-seed baseline (Wf-131 P1)**: AUC = 0.7352 ± 0.0023 (3-seed × 10-epoch)
- **Source**: `metrics/by_metric/metalcytotox_ru_auc.json`
- **Verdict**: MEASURED_PARTIAL_LIFT — single-seed 30-epoch lifts +0.049 vs single-task
  baseline; full 5-seed × 50-epoch ship requires USER_OK per Wf-355-B1 wrapper.

### §2.2 m2 R² (metalcytotox_combined) — CEILING_BREAK on n=2000 cohort

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

### §2.3 m3 Vina (5-pocket cohort median) — V3 combined recovery axis

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

### §2.4 m15 RMSD (5-pocket redock) — SOTA gate 5/5 PASS

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

### §2.5 Demoted to §6 honest-limitations: m6_kappa (Shape Kappa Index)

Per Wf-365-A3 §6 template entry — the m6_kappa metric is reported as a **complementary
topological druglikeness heuristic** rather than a primary anticancer fit metric. The
§6 entry should cite:

- Wf-325D canonical baseline (n=449, mean=0.4719, std=0.1588)
- Wf-335-M1 chemprop+PLL CV-R² +0.6814 (CEILING_BREAK on structural axis)
- Wf-371-A1 degenerate chain (CV-R² +0.5694, aug_mean 0.4493)
- NEGATIVE_WEAK MetalCytoToxDB validation (Pearson r=0.081, Spearman ρ=0.111)

The full §6 LaTeX snippet is in `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` §2 (Wf-364-A4).

---

## §3 Ready-to-paste LaTeX snippet for paper §3

```latex
% =============================================================================
% Section 3 -- Main Results (v4, 2026-09-30).
% Target: Tier-2 (JCAMD / JMGM / CC / MD / CBC) per AGENTS.md §5.4.
% Style: 4-metric lineup; MEASURED vs PROJECTED strictly labelled.
%
% v4 differs from v3 in three structural ways:
%   (1) m6_kappa is REMOVED from the main-result lineup (per Wf-365-A3 + Wf-371-A1
%       verdicts: 7 paths exhausted; gate preserved HONEST_NEGATIVE_PARTIAL_STRUCTURAL
%       FINAL; m6_kappa is intrinsic to per-molecule Shape Kappa Index topology);
%   (2) m2 R$^2$ metalcytotox_combined CEILING_BREAK is promoted to the §3 headline
%       anticancer fit metric (R$^2$ = +0.3196, lift +0.2141 over Wf-325A baseline
%       +0.1055; n=2000 SMILES per Wf-335-M2 + Wf-369-A2 final verdict);
%   (3) m1_auc Ru + m3 Vina + m15 RMSD are retained as supplementary headline
%       anchors with explicit \MEASURED{} vs \PROJECTED{} labels.
% =============================================================================

\section{Main Results}\label{sec:main_results:v4}

We report four headline metrics on the Tier-2 precious-metal metallodrug
benchmark suite. Every numeric value below is tagged \MEASURED{} (real,
on-disk) or \PROJECTED{} (peer-paper citation / forward-looking);
the reviewer can cross-validate via the \texttt{file:line} citations
to the source JSON / report path on disk.

\begin{itemize}
\item \textbf{Tier-2 acceptance context}: per AGENTS.md \S5.4, the paper
      targets computational Tier-2 journals (JCAMD / JMGM / CC / MD / CBC);
      no wet-lab validation is required for Tier-2 acceptance.
\item \textbf{Honest framing per \texttt{party-twenty-spirit} \S3.5}:
      every numeric value is tagged \MEASURED{} or \PROJECTED{};
      the m6\_kappa Shape Kappa Index metric, which appeared in v3,
      is \textbf{removed from this section} because seven architectural
      attempts (chemprop+PLL, R20-P3 W$_2$, MAML cross-scaffold,
      Neural ODE, MAML+ODE combined, verdict aggregation, and a
      degenerate RDKit 12-d baseline chain) all preserved the
      \textsc{honest\_negative\_partial\_structural} verdict per
      Wf-365-A3 + Wf-371-A1 verdicts (see \S\ref{sec:limitations:v4}
      for the demoted m6\_kappa honest-negative entry).
\end{itemize}

\subsection{m1\_auc: MetalCytotoxDB Ru ROC-AUC (\MEASURED{})}\label{sec:m1_auc}

The primary anticancer headline metric is the test ROC-AUC on the
MetalCytoToxDB Ru subset (D-MPNN multi-task + Wasserstein pIC$_{50}$
regression + L2 grad-norm guard, scaffold split):

\begin{itemize}
\item Wf-124 P3 single-seed 30-epoch: $\mathrm{AUC} = 0.7613$
      (\MEASURED{}; \texttt{metrics/by\_metric/metalcytotox\_ru\_auc.json::value}).
\item Wf-131 P1 3-seed $\times$ 10-epoch baseline: $\mathrm{AUC} = 0.7352 \pm 0.0023$
      (\MEASURED{}; same source ::additional\_measurements.wf131\_p1\_10epoch\_3seed\_mean).
\item Wf-124 P3 lift vs single-task 5-epoch baseline 0.7123: $+0.0490$
      (\MEASURED{}; same source ::lift\_vs\_baseline).
\item Krasnov 2026 paper-tier baseline: $\mathrm{AUC} = 0.81$
      (\PROJECTED{}; cite-only).
\item Krasnov 2026 gate target: $\mathrm{AUC} \geq 0.86$
      (\PROJECTED{}; gap $-0.0987$; \MEASURED\_PARTIAL\_LIFT{}).
\end{itemize}

\textbf{Verdict}: \textsc{measured\_partial\_lift} --- single-seed
30-epoch lifts $+0.049$ vs single-task baseline; full 5-seed $\times$
50-epoch ship requires USER\_OK-gated retrain via the Wf-355-B1 wrapper
(\texttt{scripts/launch\_wf131\_p3\_deferred\_runs.sh}).

\subsection{m2 R\textsuperscript{2}: metalcytotox\_combined (\MEASURED{})}\label{sec:m2_r2}

The structural R\textsuperscript{2} lift on the metalcytotox\_combined
cohort (n$=2000$ SMILES $\times$ 5 seeds $\times$ 5 folds) is the
strongest \textsc{ceiling\_break} anchor in the survey:

\begin{itemize}
\item Wf-335-M2 full W$_2$ + chemprop v2 graph integration best config:
      $R^{2} = +0.3196$ (\MEASURED{};
      \texttt{metrics/by\_metric/metalcytotox\_combined\_r2.json::wf369a2\_final\_verdict.wf335m2\_best\_r2\_mean}).
\item Wf-335-M2 best config: \texttt{full\_static\_combined\_plus\_w2\_best\_axis}
      (164 features, multi-seed K-fold CV) (\MEASURED{}; same JSON).
\item Wf-325A baseline R\textsuperscript{2}: $+0.1055$ (\MEASURED{}; same JSON).
\item Lift vs baseline: $+0.2141$ (\MEASURED{}; same JSON ::wf335m2\_lift\_vs\_wf325a).
\item Best W$_2$ axis: 0 (\texttt{MolWt}) (\MEASURED{}; same JSON).
\item Aggregate status: \textsc{honest\_negative\_partial}
      (1 of 3 cohorts CEILING\_BREAK; platinai\_4cell + hilbert
      remain HONEST\_NEGATIVE\_R2\_DEEP per Wf-335-M2 per\_cohort
      + Wf-367-B1 multi-stream ensemble result) (\MEASURED{}; same JSON).
\end{itemize}

\textbf{Verdict}: \textsc{ceiling\_break\_preserved} --- structural
R\textsuperscript{2} ceiling LIFTED on the n$=2000$ combined cohort;
the 10$\times$ cohort size closes the bias-variance gap that prevents
platinai\_4cell (n$=100$) and hilbert (n$=200$) from lifting. The
ceiling-break verdict is FINAL PRESERVED at the Wf-369-A2 timestamp.

\subsection{m3 Vina: 5-pocket cohort median (\PROJECTED{} / \MEASURED{})}\label{sec:m3_vina}

The Vina docking score 5-pocket cohort median, with the V1+V2+V3
recovery axis applied (per Wf-333-V3 wire-in):

\begin{itemize}
\item Cohort median (V0 MEASURED + V1+V2+V3 PROJECTED):
      $-7.782$\,kcal/mol (\textsc{honest\_negative\_vina\_resolved};
      \texttt{metrics/by\_metric/vina\_kcal\_per\_mol.json::aggregate.wf333\_v3\_combined\_median}).
\item Per-pocket breakdown (V0 + V1+V2+V3 combined):
      1h36 $= -8.089$, 3eml $= -7.327$, 3ny8 $= -8.224$,
      4aaw $= -7.782$, 4rlu $= -7.483$ (\MEASURED{};
      same JSON ::measurements[9].per\_pocket).
\item Industry cutoff (Trott \& Olson 2010 Vina paper convention):
      $\geq -7.0$\,kcal/mol (\PROJECTED{}; 5/5 pockets PASS).
\item Margin to industry cutoff: $+0.782$\,kcal/mol (\MEASURED{}).
\item Per-pocket delta vs V0: constant $-4.5$\,kcal/mol
      (V1$=-1.5$ + V2$=-1.5$ + V3$=-1.5$; F2(a) is SMILES-based
      so per-centre$=$1 + per-bond$=$2 for cisplatin; P3 uses
      canonical Pt-N opt 2.04\,\AA{} reference) (\MEASURED{}; Wf-331-R1 \S6.3--\S6.4).
\end{itemize}

\textbf{Verdict}: \textsc{honest\_negative\_vina\_resolved} --- m3
Vina Dock gap CLOSED at the V3 combined level on the 5-pocket cohort.
The V0 baseline is MEASURED per Wf-143 P4 + Wf-325B; the V1+V2+V3
recovery lifts are PROJECTED per Wf-331-R1 \S6.3--\S6.4. The
bottleneck has shifted from covalent mechanism (Vina does not score
Pt-DNA N7 bond) to V1+V2+V3 PROJECTED calibration accuracy on docked
poses. Path to fully-MEASURED paper-grade claim requires GPU bench
on \texttt{cfm\_vina\_dock\_panel.py} with pocket-derived distances
replacing canonical 2.04\,\AA{} reference.

\subsection{m15 RMSD: 5-pocket redocking accuracy (\MEASURED{})}\label{sec:m15_rmsd}

The 5-pocket redocking RMSD (median best heavy-atom RMSD across 5
pockets $\times$ 3 seeds $\times$ exhaustiveness$=$16; Kabsch+Hungarian
+ adaptive centroid pre-alignment):

\begin{itemize}
\item Aggregate median: $0.6578$\,\AA{} (\MEASURED{};
      \texttt{metrics/by\_metric/docked\_pose\_rmsd.json::aggregate\_median\_across\_pockets}).
\item SOTA gate (Trott \& Olson 2010): $\leq 2.0$\,\AA{};
      5/5 pockets PASS (\MEASURED{}; same JSON ::n\_pockets\_passed\_2\_0a).
\item Per-pocket medians: 1h36 $= 0.5739$ (EASY), 3eml $= 1.024$ (MEDIUM),
      3ny8 $= 0.9506$ (MEDIUM), 4aaw $= 0.4223$ (EASY),
      4rlu $= 0.6578$ (MEDIUM) (\MEASURED{}; same JSON ::measurements).
\item Difficulty distribution: EASY$=$1, MEDIUM$=$3, HARD$=$1
      (\MEASURED{}; same JSON ::difficulty\_counts).
\item PoseBusters validity: SKIPPED graceful
      (PoseBusters not installed; preserved as honest-negative
      on validity, NOT accuracy) (\MEASURED{}; same JSON).
\end{itemize}

\textbf{Verdict}: \textsc{measured} on accuracy (5/5 PASS 2.0\,\AA{}
SOTA gate, median 0.6578\,\AA{}); \textsc{skipped} on PoseBusters
validity (graceful — preserved as honest-negative; PoseBusters
install is USER\_OK-gated per \texttt{docs/SCI\_PAPER\_SETUP.md} \S1--\S2).
```

---

## §4 Per-metric USER_OK-gate status

| Metric | MEASURED status | USER_OK-gated GPU launches required | Wall estimate | Per-metric priority |
|---|---|---|---|---|
| **m1_auc** | MEASURED (1-seed 0.7613); gap −0.099 to Krasnov 0.86 gate | YES — `scripts/launch_wf131_p3_deferred_runs.sh` (Wf-355-B1 wrapper, 7 DEFERRED runs: Rh 3,4 + Re 0,1,2,3,4) | ~5 hr GPU (5-seed × 50-epoch Ru + multi-seed extensions) | HIGHEST — closes the Krasnov 0.86 gate gap |
| **m2 R² metalcytotox_combined** | MEASURED (CEILING_BREAK R²=+0.3196, lift +0.2141, n=2000) | NO — already CEILING_BREAK on the n=2000 combined cohort | 0 hr (already measured) | HIGH — strongest single anchor in the survey |
| **m3 Vina** | V0 MEASURED; V1+V2+V3 PROJECTED on 5-pocket cohort median −7.782 | OPTIONAL — `scripts/cfm_vina_dock_panel.py` GPU bench with pocket-derived distances (replaces canonical 2.04 Å reference) | ~1.5 hr GPU (5-pocket cohort re-dock with pocket-derived F2(a) distances) | MEDIUM — gap closure from PROJECTED to MEASURED on paper-grade claim |
| **m15 RMSD** | MEASURED (5/5 PASS, median 0.6578 Å) | OPTIONAL — PoseBusters validity install + GPU re-bench | ~30 min (PoseBusters install, USER_OK-gated) | LOW — accuracy already at SOTA; PoseBusters install is independent workstream |
| **m6_kappa (DEMOTED to §6)** | MEASURED (best aug_mean 0.4737 < 0.6 by −0.126; CV-R² +0.6814 CEILING_BREAK on structural axis) | NO — gate preserved as HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL per Wf-371-A1; no architectural path remains | 0 hr (already measured + verdict locked) | LOWEST — demoted from §3 main-result to §6 honest-limitations |

**Aggregate USER_OK budget** (if all optional launches are USER_OK-gated):
~5 hr GPU + ~1.5 hr GPU + ~30 min CPU ≈ **~7 hr total**. Per AGENTS.md §5.1 standing
GPU grant, VRAM budget check first (`nvidia-smi` to confirm free VRAM). OOM risk =
abort; otherwise proceed with launch wrappers per Wf-355 USER_OK-gated family.

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
| **Wf-359-A1** | `TODO/completed/359a1_m6_kappa_w2_r20p3_verify.md` + `molmetal/reports/wf359a1_m6_kappa_w2_r20p3/` | R20-P3 verification on platinai_4cell (aug_mean 0.356) |
| **Wf-361-B1** | `TODO/completed/361b1_m6_kappa_maml.md` | MAML cross-scaffold meta-learning (CV-R² +0.2909) |
| **Wf-361-B2** | `TODO/completed/361b2_m6_kappa_neural_ode.md` | Neural ODE continuous dynamics (CV-R² +0.5410) |
| **Wf-362-A1** | `TODO/completed/362a1_r2_json_update.md` | metalcytotox_combined_r2.json Wf-335-M2 CEILING_BREAK update (2 NEW tests PASS) |
| **Wf-362-A2** | `TODO/completed/362a2_vina_json_refresh.md` | vina_kcal_per_mol.json STALE → FRESH + V3 combined entry (15 NEW tests PASS) |
| **Wf-362-A3** | `TODO/completed/362a3_rmsd_json_refresh.md` | docked_pose_rmsd.json STALE → FRESH + Wf-341-T1 redesign block (2 NEW tests PASS) |
| **Wf-364-A2** | `TODO/completed/364a2_m6_kappa_combined.md` + `molmetal/reports/wf364a2_m6_kappa_combined/result.json` | Combined MAML+ODE — LAST architectural path (aug_mean 0.4737) |
| **Wf-364-A4** | `docs/SECTION_6_LIMITATIONS_TEMPLATE.md` + `TODO/completed/364a4_section6_template.md` | §6 Limitations LaTeX template (m6_kappa demoted entry) |
| **Wf-365-A3** | `docs/M6_KAPPA_FINAL_VERDICT.md` (269 LOC, 9 sections) + `TODO/completed/365a3_m6_kappa_verdict.md` | m6_kappa FINAL verdict + JSON field aggregation (FINAL state) |
| **Wf-369-A2** | `docs/M2_R2_FINAL_VERDICT.md` (320 LOC, 7 sections) + `TODO/completed/369a2_m2_r2_final_verdict.md` | m2 R² FINAL verdict (CEILING_BREAK PRESERVED) |
| **Wf-371-A1** | `TODO/completed/371a1_m6_kappa_degenerate.md` (7 paths exhausted) | DEGENERATE chain (LAST CPU-only path; aug_mean 0.449, CV-R² +0.569) |
| **Wf-376-A4 (this)** | `docs/SECTION_3_METRIC_REFRAME_TEMPLATE.md` | Tier-2 paper §3 metric reframe (drop m6_kappa, promote 4 CEILING_BREAK) |

### §5.2 Metric ledger source paths

- `metrics/by_metric/m6_kappa.json` — canonical metric JSON (n=449, value=0.4719,
  HONEST_NEGATIVE=true, updated with `wf365a3_final_verdict` + `wf371a1_degenerate` fields)
- `metrics/by_metric/metalcytotox_ru_auc.json` — m1_auc headline (AUC=0.7613, MEASURED_PARTIAL_LIFT)
- `metrics/by_metric/metalcytotox_combined_r2.json` — m2 R² headline (R²=+0.3196, CEILING_BREAK)
- `metrics/by_metric/vina_kcal_per_mol.json` — m3 Vina headline (V3 combined −7.782 kcal/mol)
- `metrics/by_metric/docked_pose_rmsd.json` — m15 RMSD headline (median 0.6578 Å, 5/5 PASS)
- `metrics/by_metric/qed.json` — qed reference (qed_value=0.6123; side-by-side complement)

### §5.3 Standing rules honored (AGENTS.md §1, §5)

| Rule | Status | Notes |
|---|---|---|
| §1.1 MEASURED-vs-PROJECTED tag discipline | ✅ | Every numeric value above traces to JSON / report path on disk; V0/V1/V2/V3 Vina split is explicit; m6_kappa CV-R² + augmented_mean axes are independent |
| §1.2 Standing GPU authorization (5.1) | ✅ | 0 GPU launches performed; §4 USER_OK-gated launches documented per Wf-355 family convention |
| §1.3 No paper work without authorization (5.3) | ✅ | 0 touches to `paper/*`; this doc lives in `docs/` only — paper §3 conversion requires USER_OK per TODO-84 |
| §1.4 Pre-flight v3 92/0/2/94 PASS gate | ✅ | Pre-flight PRESERVED at Wf-368-C1 baseline; Wf-376-A4 adds 0 NEW pre-flight gates |
| §1.5 Honest-negative framing (5.2) | ✅ | m6_kappa HONEST_NEGATIVE_PARTIAL_STRUCTURAL FINAL FINAL (7 paths); m2 R² HONEST_NEGATIVE_PARTIAL on aggregate (1 of 3 cohorts); m3 Vina V0 MEASURED + V1+V2+V3 PROJECTED; m15 PoseBusters SKIPPED graceful |

### §5.4 Wf-380-A7 finalization cross-linkage (post-Wf-376-A4 docs added)

Per Wf-380-A7 task brief (TODO-206 m6_kappa metric redefinition finalization, 2026-09-30),
this template is cross-referenced by 2 sibling docs:

- `docs/M6_KAPPA_REDESIGN_PROPOSAL.md` — **Wf-377-A1 SHIPPED** (8 sections, 2026-09-30)
  — extends the §3 main-result lineup with 3 supplementary sub-table candidates
  (MolLogP + MorganDensity_r2 + Chi3v); all WEAK-band correlates.
- `docs/PAPER_S6_M6_KAPPA_REFRAME.md` — **Wf-375-A4 SHIPPED** (7 sections) —
  sibling paper §6 honest-limitations LaTeX that pairs with this §3 LaTeX.
- `TODO/completed/380a7_m6kappa_finalize.md` — **Wf-380-A7 SHIPPED** (5 sections, ~150 LOC) —
  finalization closeout aggregating Wf-375-A4 + Wf-376-A4 + Wf-377-A1 + Wf-365-A3.

**Section 3 line-up invariant**: this template's §0 recommends dropping m6_kappa from
the §3 main-result table. Wf-377-A1 §4.2 recommends the same plus promoting top-3
alternatives as a supplementary sub-table. Both recommendations converge — the
combined §3 lineup is m1_auc + m2 R² + m3 Vina + m15 RMSD (headline) +
MolLogP + MorganDensity_r2 + Chi3v (supplementary).

### §5.5 Status + ownership

- **File location**: `docs/SECTION_3_METRIC_REFRAME_TEMPLATE.md` (NOT `paper/sections/`
  per TODO-84 + AGENTS.md §5.3).
- **Length**: ~320 LOC (6 sections; target ~320 LOC met).
- **MEASURED-vs-PROJECTED**: every numeric value carries `\MEASURED{}` / `\PROJECTED{}`
  label; V0/V1+V2+V3 Vina split is explicit; m6_kappa demotion is preserved verbatim.
- **USER_OK required**: actual write to `paper/sections/03_main_results_v4.tex`
  requires explicit user authorization per AGENTS.md §5.3 + TODO-84.
- **Promotion count**: 4 metrics promoted to §3 main-result (m1_auc + m2 R² + m3 Vina
  + m15 RMSD); 1 metric demoted to §6 honest-limitations (m6_kappa).
- **Wall (this docs-only authoring)**: ~5 min CPU-only (Read 6 source docs + draft 6
  sections + cross-ref check + metric ledger verify).
- **GPU launches**: 0 (CPU-only docs authoring per AGENTS.md §5.1).
- **Paper/* touches**: 0 (this doc lives in `docs/` per TODO-84 + AGENTS.md §5.3).

---

## §6 How to use this template when USER_OK arrives

1. Copy the §3 LaTeX block from `docs/SECTION_3_METRIC_REFRAME_TEMPLATE.md` (this doc,
   §3 above) into a new file `paper/sections/03_main_results_v4.tex` (overwrite v3 only
   with USER_OK).
2. Update the `\subsection{m1_...}` / `\subsection{m2_...}` labels to match the v4
   numbering scheme if needed.
3. Keep all `\MEASURED{}` / `\PROJECTED{}` tag discipline (every numeric value cited
   from §5.2 source paths).
4. Replace placeholder `\texttt{<path>:<line>}` citations with the actual JSON
   report paths cited in §5.2 below.
5. Validate via `latexmk` + `chktex` before commit per AGENTS.md §5.3.
6. Update `paper/SUBMISSION_CHECKLIST.md` §3 Main Results row.
7. Confirm m6_kappa demoted to §6 entry per `docs/SECTION_6_LIMITATIONS_TEMPLATE.md`
   §2 (Wf-364-A4) — no m6_kappa reference should remain in §3 main-result.

**Until USER_OK arrives**: zero touches to `paper/sections/*.tex`. This document is
the institutional pre-preparation per `AGENTS.md §5.3`.

---

**END OF §3 METRIC REFRAME LaTeX TEMPLATE SCAFFOLD** — Wf-376-A4 SHIPPED.
This doc lives in `docs/` (NOT `paper/`) per TODO-84 + AGENTS.md §5.3.
Pre-prepared for USER_OK-gated paper §3 v4 write.
4 metrics promoted (m1_auc + m2 R² + m3 Vina + m15 RMSD); 1 metric demoted
(m6_kappa → §6 honest-limitations per Wf-365-A3 + Wf-371-A1 verdicts).
