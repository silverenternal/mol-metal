# P3 plan — rewrite `paper_a/` as the FSMA method paper

**Status: PROPOSAL ONLY — no `paper_a/*` file has been touched (§5.3).**
Awaiting explicit user OK. User decision 2026-10-13: *"先给 P3 详细方案再动笔"*.

## 0. The one-paragraph repositioning

The current `paper_a` is an **analysis paper** ("metal-specific activity does not
transfer; here is a within-metal win and several honest negatives"). P3 turns it
into a **method paper**: the cross-metal collapse is the *problem statement*, and
**FSMA (Few-Shot Metal Adaptation)** is the contribution — a labelled-support
fine-tuning protocol that recovers the collapse, with a complete
control/ablation/backbone story and no significant negative in the main line.

## 1. Headline claims (frozen C1–C6) and their artifacts

Every number below is MEASURED and traces to a JSON on disk. Consolidated in
`metrics/by_metric/fsma_paper_stats.json` (16 runs, per-family Holm-Bonferroni).

| # | Claim | Headline number | Artifact |
|---|---|---|---|
| C1 | FSMA recovers cross-metal collapse | **head-only K-curve all-positive**; K=40 **+0.0379**, K=80 **+0.0685** (Holm-sig, 50/50 cells) | `fsma_paper_stats.json::per_run.head_k*` |
| C2 | It is the *adaptation*, not extra labels/compute | K=20/40 kshot **+0.0171/+0.0424** vs from_scratch −0.0948/−0.0763 vs concat_retrain −0.1619/−0.1619 (step-matched) | `molmetal/reports/fsma_c2_matched_k{20,40}/metrics.json` |
| C3 | Support curation is *not* a hidden knob (robustness) | 4 samplers, none Holm-sig; ordering random ≥ stratified ≥ scaffold_diverse > activity_scaffold | `molmetal/reports/fsma_c3_*_k20/metrics.json` |
| C4 | Adaptation depth is the one real knob | head-only flips K=5 −0.0171→**+0.0097**; K=40 head still Holm-sig | `fsma_c4_head_k{5,10,20,40,80}` |
| C5 | Not a D-MPNN artifact | XGBoost warm-start +0.0029/+0.0197/+0.0435/+0.0786 (K=10/20/40/80; K=80 p=2e-7, 25/25) | `metrics/by_metric/fsma_xgb_warmstart_kshot.json` |
| C6 | Second strong metric: recovery-to-ceiling | head@K=80 recovers **74.9–98.5%** of the within-metal ceiling | `fsma_paper_stats.json::recovery_to_ceiling` |

## 2. Per-section changes

### `01_intro.tex` — restructure to problem → method → contributions
- **Keep** the cross-metal collapse as the opening problem (it is the motivation).
- **Replace** the "we report a within-metal win and honest negatives" framing with
  the FSMA contribution: *"we show a simple, backbone-agnostic adaptation
  protocol recovers 75–98% of the within-metal ceiling from K=80 labelled
  support ligands of the held-out metal."*
- **Add** an explicit contributions list C1–C6 (table above) — 6 bullets, each
  with its headline number.
- **Remove** any sentence that frames a negative as a headline.

### `02_related.tex` — mostly keep
- Keep §2.1 (ligand-based transfer), §2.2 (ML for TM complexes), §2.3
  (evaluation protocols). §2.3 is now load-bearing: it motivates the leak
  controls and the W2 protocol finding (random 10-fold **0.9182** vs DOI-grouped
  **0.7511** — `metrics/by_metric/metalcytotox_ru_paper_lightgbm_cv.json`).
- **Add** a short paragraph on few-shot / meta-learning for molecular property
  prediction to position FSMA against MAML/Reptile-style adaptation.

### `03_method.tex` — rewrite §3.4 into the FSMA formalisation
Current §3.4 is "Metal conditioning and few-shot adaptation" (2 paragraphs).
Replace with four formal sub-parts:
1. **Source model** — LOMO source-pool pretraining, `metal_mode: none`.
2. **Support set** — K labelled ligands from the held-out metal,
   ligand-disjoint from the query set; label-stratified by default (C3 shows the
   choice is not load-bearing).
3. **Adaptation operator** — freeze encoder, fine-tune **head only** for
   `ft_epochs=20` at `lr=1e-3` (C4 justifies head-only; full-scope is the
   ablation).
4. **Query evaluation** — paired AUC delta vs the *same run's* zero-shot query AUC.
- Keep §3.1 cohort/dedup, §3.5 tmQM A/B, §3.7 statistics (paired one-sided t +
  per-family Holm; bootstrap reference-only), §3.8 repro.

### `04_results.tex` — rebuild the main table around the K-curve
- **New Table 1 (main)**: FSMA head-only K-curve, per-metal + pooled,
  K ∈ {5,10,20,40,80}, with paired deltas, CIs, n-positive, Holm verdict.
- **New Figure 1**: the K-curve (delta vs K), per-metal lines + pooled.
- **New Table 2**: recovery-to-ceiling at K=40/80 (ceiling AUC + recovery %).
- **Keep** the transfer matrix (current Claim 1) as the *problem* figure, moved
  to §4.1.
- **Keep** the within-metal vs Krasnov win (current Claim 3) — but demote the
  "cross-cohort honest negative" half to SI.
- **Move to SI**: current Claim 2 (tmQM pretraining honest negative), the
  MoleculeNet control, the bace/clintox gaps.

### `05_ablation.tex` — rebuild as the four FSMA ablations
1. **Adaptation depth** (C4): full / head / head+metal at matched K — this is the
   table that explains *why* head-only is the default.
2. **Support-sampler** (C3): 4 arms, none significant → robustness claim.
3. **Controls** (C2): step-matched from_scratch / concat_retrain.
4. **Backbone** (C5): XGBoost warm-start reproduces the curve.
- Keep the tie-rule / dedup / scorer sensitivity analyses (they are leak-control
  evidence, now supporting §2.3 and §6).

### `06_limitations.tex` — trim to a short SI pointer
- Main text keeps **only**: pooled-mean is Ru-dominated; small per-metal n.
- Everything else (tmQM negative, bace/clintox gap, cross-cohort caveat, the
  cohort-construction trap) → **SI**, each with its JSON pointer.
- Per user directive, the main line carries **no honest negative**.

### Figures to regenerate (`paper_a/figures/`)
1. `fig1_kshot_curve.pdf` — delta vs K, per-metal + pooled, head-only.
2. `fig2_recovery_ceiling.pdf` — bar: kshot AUC vs ceiling per metal, K=80.
3. `fig3_scope_ablation.pdf` — full vs head vs head+metal across K.
4. `fig4_backbone.pdf` — D-MPNN vs XGBoost K-curves side by side.
5. `fig0_transfer.pdf` — the existing transfer matrix (reused, moved to §4.1).

## 3. What P3 will NOT touch
- `paper/*` (Paper B / CFM) — out of scope.
- `metrics/*` — read-only inputs.
- Any number not traceable to a JSON above (§5.2).

## 4. Proposed execution order (after OK)
1. `03_method.tex` FSMA formalisation (no new numbers).
2. `04_results.tex` main table + Fig 1/2.
3. `05_ablation.tex` four ablations + Fig 3/4.
4. `01_intro.tex` contributions list.
5. `02_related.tex` few-shot paragraph.
6. `06_limitations.tex` trim + build SI.
7. Rebuild `main.pdf`, verify **0 undefined refs** (P4.4).

## 5. Open items that may still move numbers
- **W4** (D-MPNN within-metal cap512/d4 + ensemble) — if it closes the gap to
  GBDT, §4.1's "GBDT by protocol" caveat strengthens; running.
- **W7** (tmQM A/B 5→15 seeds) — if Ir crosses p<0.05 it moves from SI back to a
  main-text sentence; running.
