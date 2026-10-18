# PAPER-SYNC proposal — within-metal capacity ablation (+ a ceiling-attribution defect it exposed)

**Status:** PROPOSAL ONLY. Nothing has been written to `paper_a/*` (AGENTS.md §5.3,
TODO-84). Every number below is MEASURED; every LaTeX block is offered for review
before it is applied.

**Source artifacts (all committed):**

| artifact | what it holds |
|---|---|
| `metrics/by_metric/dmpnn_within_metal_cap128_pf_gpu0.json` | h128 leg, cuda:0, deterministic, 5 seeds |
| `metrics/by_metric/dmpnn_within_metal_cap512_pf_gpu0.json` | h512 leg, same device/folds |
| `metrics/by_metric/dmpnn_within_metal_capacity_pair_gpu0.json` | paired deltas, invariant + fold-identity checks, Holm |
| `metrics/by_metric/dmpnn_within_metal_capacity_foldwise.json` | fold-level decomposition, size-adaptive policy number |

Producer scripts: `molmetal/scripts/wfA_capacity_pair_analysis.py`,
`molmetal/scripts/wfA_capacity_foldwise_analysis.py`. Commits `57116ff3`,
`cd01ed7e`.

---

## 1. Proposed new results subsection

Insert after `\subsection{The within-metal axis and the published protocol}`
(currently `paper_a/sections/04_results.tex:186-248`), before the section break.

### 1a. Body text (proposed hunk)

```latex
\subsection{Width is a narrow lever, and only on large cohorts}
\label{sec:results:capacity}

The within-metal column above is measured at a fixed width. To ask whether that
choice is leaving performance on the table, we re-trained the same model at
$hidden{=}128$ and $hidden{=}512$ on the \emph{same} device, with the same folds,
the same five split seeds and the same optimisation budget; width is the only
free variable. Both runs reproduce the reference XGBoost column bit-for-bit on
all $25$ (metal, seed) cells, which is the check that the pairing is real.

Widening helps on two of five metals and only there: Ir gains $+0.0828$
($5/5$ seeds, Holm $p{=}0.0017$) and Ru gains $+0.0183$ ($5/5$ seeds, Holm
$p{=}0.0042$). Re is positive but does not survive correction ($+0.0114$,
$4/5$, Holm $0.114$); Rh and Os are null ($-0.0027$ and $-0.0029$). On the two
metals where it is significant, width alone recovers $43\%$ (Ru) and $48\%$
(Ir) of the gap to the same-fold XGBoost reference. It does not close it: at
$hidden{=}512$ the D-MPNN remains below XGBoost on \emph{all five} metals, so
the residual gap is not a capacity artefact.

The gain is not uniform inside a metal. On Ir it appears on $25/25$ folds and is
$4.8\times$ larger on the worst quintile of folds ($+0.226$ versus $+0.047$
elsewhere), i.e.\ width mainly rescues the folds where the narrow model
collapsed; the across-fold standard deviation halves, $0.078$ to $0.035$. On Ru
the gain is broad ($+0.018$ on average, $23/25$ folds positive) and the spread
does not shrink. The two significant metals are also the two largest cohorts
($n{=}4793$ and $1281$); the three cohorts below $n{=}400$ give point estimates
between $-0.003$ and $+0.011$, none significant. We therefore report the effect
per metal and do not average it.
```

### 1b. Table (proposed hunk)

```latex
\begin{table}[t]
\centering
\caption{Device-matched capacity ablation on the within-metal column.
Both widths were trained on the same GPU with deterministic kernels; the
deduplicated cohort, the five folds and the five split seeds are identical, and
the XGBoost reference column is bit-identical across the two runs on all 25
(metal, seed) cells. $\Delta$ is the paired per-seed difference
$h512-h128$; $p$ is a one-sided paired $t$-test, Holm-corrected across the five
metals. \emph{Gap closed} is the fraction of the $h128$ deficit against the
same-fold XGBoost reference removed by widening; negative values mean the gap
grew. \MEASURED:
\texttt{metrics/by\_metric/dmpnn\_within\_metal\_capacity\_pair\_gpu0.json}.}
\label{tab:capacity}
\small
\begin{tabular}{lrrrrrr}
\toprule
Metal & $n$ & $h128$ & $h512$ & $\Delta$ & seeds $>0$ & $p_{\mathrm{Holm}}$ \\
\midrule
Ru & $4793$ & $0.8192$ & $0.8375$ & $+0.0183$ & $5/5$ & $0.0042$ \\
Ir & $1281$ & $0.6543$ & $0.7371$ & $+0.0828$ & $5/5$ & $0.0017$ \\
Rh &  $334$ & $0.7775$ & $0.7748$ & $-0.0027$ & $1/5$ & $1.000$ \\
Os &  $333$ & $0.6908$ & $0.6878$ & $-0.0029$ & $2/5$ & $1.000$ \\
Re &  $250$ & $0.8056$ & $0.8169$ & $+0.0114$ & $4/5$ & $0.114$ \\
\bottomrule
\end{tabular}
\end{table}
```

### 1c. Why the table must report per metal

Ru is $64\%$ of the deduplicated cohort. A pooled mean would be a Ru mean with
four small cohorts attached to it, and the sign of the effect differs across
metals, so a pooled number is not interpretable. The artifact carries this as
`honest_framing.do_not_claim[1]`.

---

## 2. Defect exposed while preparing the above — the "within-metal ceiling" is chemprop v2, labelled as the D-MPNN

This is **not** part of the capacity hunk, but the capacity numbers made it
visible, and it changes what the capacity result means for the paper's headline
claim. It is reported here rather than acted on. It is the same defect class as
the abstract sync (chemprop-v2 numbers attributed to the in-house model) and it
falls inside the already-open **#172 Table 1 in-house migration**; what is new
here is that the capacity measurement tells us *which width* the migrated table
should use.

**The evidence.** `molmetal/reports/l3_within_pf_ceiling/metrics.json` is the
artifact that supplies the within-metal ceiling. Its own `model` field reads:

> `chemprop v2 AtomMessagePassing (d_h=300, depth=3) + BinaryClassificationFFN
> (single pooled head); metal_mode=none`

Its `within_metal.Ru.mean_auc = 0.854533391154804` — the exact value the paper
prints as "This work (D-MPNN)". The same value is the denominator in
`metrics/by_metric/fsma_paper_stats.json → recovery_to_ceiling`
(`ceiling_within_metal_mean_auc.Ru = 0.854533391154804`), so the recovery
percentages inherit the same denominator.

**Affected locations in `paper_a/sections/04_results.tex`:**

| line | current text | issue |
|---|---|---|
| 29–30 | `Ru & 0.8545`, `Ir & 0.7570` in Table 1 | attributed to the D-MPNN; source is chemprop v2 |
| 177–178 | recovery table, ceiling column | denominator is chemprop v2 |
| 191 | "the D-MPNN reaches $0.8545$ (Ru) and $0.7570$ (Ir)" | same attribution |
| 215 | `This work (D-MPNN) & Ru 0.8545 / Ir 0.7570 & ... & D-MPNN` | same attribution |

**The in-house D-MPNN on the same cohort and folds (MEASURED, 5 seeds):**

| metal | $h128$ | $h512$ | Krasnov published |
|---|---|---|---|
| Ru | $0.8192$ (CPU published column: $0.8173$) | $0.8375$ | $0.81$ |
| Ir | $0.6543$ (CPU: $0.6649$) | $0.7371$ | $0.73$ |

**Consequence for the claim.** The sentence at line 191 — "it exceeds both" —
is true for Ru at either width, but for Ir it is true **only at $h512$**
($0.7371 > 0.73$); at $h128$ the in-house model is $0.065$ \emph{below} the
published $0.73$. So the capacity result is not a cosmetic ablation: it decides
whether the paper's within-metal win statement is true on both metals. That is
also the strongest argument for adopting the size-adaptive policy as the
reported configuration rather than $h128$ uniformly.

**Recovery percentages under each candidate denominator** (adapted AUC at
$K{=}80$ unchanged at Ru $0.6397$, Ir $0.6675$; only the denominator moves):

| ceiling used | Ru recovery | Ir recovery |
|---|---|---|
| chemprop v2 (current): $0.8545$ / $0.7570$ | $74.9\%$ | $88.2\%$ |
| in-house $h128$: $0.8192$ / $0.6649$ | $78.1\%$ | $100.4\%$ (incoherent) |
| in-house $h512$: $0.8375$ / $0.7371$ | $76.4\%$ | $90.6\%$ |

The $100.4\%$ cell is diagnostic: the adapted model cannot exceed its own
within-metal ceiling, so $h128$ is the wrong denominator for Ir. That points to
the $h512$ (or size-adaptive) ceiling as the coherent choice.

**Three options — the user must pick one. Nothing has been changed.**

- **A. Relabel only.** Keep the chemprop numbers but relabel the row as the
  reference backbone, and add an explicit in-house D-MPNN row. Smallest edit;
  leaves the recovery denominators foreign to the paper's model.
- **B. Re-base the ceiling on the in-house D-MPNN at $h128$.** Recompute all
  recovery percentages. Coherent for Ru, incoherent for Ir ($>100\%$).
- **C. Re-base on the size-adaptive width ($h512$ for Ru/Ir, $h128$ elsewhere).**
  Recompute recovery percentages and relabel the within-metal row to the
  policy width. This is the only option where every claim is internally
  consistent with the capacity result, and it matches
  `configs/dmpnn_within_metal_sizeadaptive_pf_gpu0.yaml`.

My recommendation is **C**, but it touches Table 1, the recovery table and two
sentences, so it needs an explicit decision before any edit.

---

## 3. Honest framing the capacity text must keep

1. **No pooled mean.** Report the five metals; Ru is $64\%$ of the cohort.
2. **Widening is not a fix for the gap.** At $h512$ the D-MPNN is still below
   same-fold XGBoost on $5/5$ metals.
3. **Re is not established.** $+0.0114$, $4/5$ seeds, Holm $0.114$.
4. **The policy is post-hoc.** The $n{=}1000$ cut was chosen after seeing these
   five metals; it sits in the empty interval between $334$ and $1281$ but has
   not been validated on a held-out cohort.
5. **Small cohorts are not harmed.** Rh/Os are null, not negative-and-significant;
   say "not detectable", not "hurts".

---

## 4. Verification (re-run before applying any hunk)

```bash
.venv/bin/python molmetal/scripts/wfA_capacity_pair_analysis.py
.venv/bin/python molmetal/scripts/wfA_capacity_foldwise_analysis.py
.venv/bin/python -m pytest tests/test_width_policy.py -q
```

Both producer scripts re-derive from the two MEASURED JSONs and refuse to emit a
delta if the invariant check fails (only `model.hidden` may differ) or if the
XGBoost fold-identity check fails.
