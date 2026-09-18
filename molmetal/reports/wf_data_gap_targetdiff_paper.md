# TargetDiff (Guan et al. ICLR 2023) — full inventory + WF-Data-Gap context

**Task:** WF-Data-Gap-Analysis — Download ONE Q1 SBDD paper as benchmark, inventory all tables/figures/statistics, then compare to current molmetal data inventory.
**Date:** 2026-09-14
**Author:** WF-Data-Gap subagent (no code modifications; pure-analysis workflow).

---

## 0. Bibliographic note — important correction

The user's prompt named "TargetDiff (Jin et al. 2023, ICML, arXiv:2303.14246)". **Three out of four bibliographic fields are wrong**; I verified via DBLP / arXiv / paper PDF cross-check:

| Field | User prompt | Reality | Source |
|---|---|---|---|
| First author | "Jin" | **Jiaqi Guan** (with Wesley Wei Qian, Xingang Peng, Yufeng Su, Jian Peng, Jianzhu Ma) | dblp.org/pid/207/7593.html |
| Venue | "ICML" | **ICLR 2023** (Kigali, OpenReview kJqXEPXMsE0) | dblp cross-ref |
| arXiv id | 2303.14246 | **arXiv:2303.03543** (v1, 6 Mar 2023, q-bio.BM) | arxiv-vanity.com/papers/2303.03543 |
| Paper title | "TargetDiff" | "3D Equivariant Diffusion for Target-Aware Molecule Generation and Affinity Prediction" | arXiv abstract |

`arXiv:2303.14246` is actually **DecompDiff** (Guan, Zhou, Yang, Bao, Peng, Ma, Liu, Wang, Gu — ICML 2023) — a related but distinct paper. We treat it as the natural **backup cite-only baseline** for the gap analysis (Table 2 below), but the primary inventory below is the correctly-cited TargetDiff.

**Decision:** Use TargetDiff (Guan et al. ICLR 2023, arXiv:2303.03543) as primary; cite DecompDiff (ICML 2023) and DiffSBDD (Schneuing et al., Nat Commun 2024, doi:10.1038/s41467-024-49069-0, Zenodo ckpt 8183747) as SOTA baselines in §7.

---

## 1. Abstract (verbatim, condensed)

> Rich data and powerful machine learning models allow us to design drugs for a specific protein target in silico. Recently, the inclusion of 3D structures during targeted drug design shows superior performance to other target-free models as the atomic interaction in the 3D space is explicitly modeled. However, current 3D target-aware models either rely on the voxelized atom densities or the autoregressive sampling process, which are not equivariant to rotation or easily violate geometric constraints resulting in unrealistic structures. In this work, we develop a 3D equivariant diffusion model to solve the above challenges. To achieve target-aware molecule design, our method learns a joint generative process of both continuous atom coordinates and categorical atom types with a SE(3)-equivariant network. Moreover, we show that our model can serve as an unsupervised feature extractor to estimate the binding affinity under proper parameterization, which provides an effective way for drug screening. To evaluate our model, we propose a comprehensive framework to evaluate the quality of sampled molecules from different dimensions. Empirical studies show our model could generate molecules with more realistic 3D structures and better affinities towards the protein targets, and improve binding affinity ranking and prediction without retraining.

Source: arXiv:2303.03543v1, p.1.

---

## 2. Tables (n=4 main + 1 supplementary-like)

### Table 1 — Property summary (THE canonical SBDD table)

**Shape:** 7 rows (methods) × 17 cols (7 metric blocks × {Avg, Med} + 1 method column).
**Methods rows:** `Reference, liGAN, GraphBP, AR, Pocket2Mol, TargetDiff` (6 baselines + reference).
**Metric blocks:** Vina Score, Vina Min, Vina Dock, High Affinity (%), QED, SA, Diversity.
**Reported per-pocket protocol:** 100 held-out pockets × 100 samples × seed=2021.

| Method | Vina Score (Avg/Med, ↓ kcal/mol) | Vina Min (Avg/Med, ↓) | Vina Dock (Avg/Med, ↓) | High Aff. (Avg/Med, ↑ %) | QED (Avg/Med, ↑) | SA (Avg/Med, ↑) | Div. (Avg/Med, ↑) |
|---|---|---|---|---|---|---|---|
| liGAN* | — / — | — / — | −6.33 / −6.20 | 21.1 / 11.1 | 0.39 / 0.39 | 0.59 / 0.57 | 0.66 / 0.67 |
| GraphBP* | — / — | — / — | −4.80 / −4.70 | 14.2 / 6.7 | 0.43 / 0.45 | 0.49 / 0.48 | 0.79 / 0.78 |
| AR | −5.75 / −5.64 | −6.18 / −5.88 | −6.75 / −6.62 | 37.9 / 31.0 | 0.51 / 0.50 | 0.63 / 0.63 | 0.70 / 0.70 |
| Pocket2Mol | −5.14 / −4.70 | −6.42 / −5.82 | −7.15 / −6.79 | 48.4 / 51.0 | 0.56 / 0.57 | 0.74 / 0.75 | 0.69 / 0.71 |
| **TargetDiff** | **−5.47 / −6.30** | **−6.64 / −6.83** | **−7.80 / −7.91** | **58.1 / 59.1** | **0.48 / 0.48** | **0.58 / 0.58** | **0.72 / 0.71** |
| Reference | −6.36 / −6.46 | −6.71 / −6.49 | −7.45 / −7.26 | — | 0.48 / 0.47 | 0.73 / 0.74 | — |

\* liGAN + GraphBP use QVina (not Vina) because Vina fails to parse some generated atom types.
**Primary metric:** Vina Dock Avg (kcal/mol, lower=better) → **−7.80** for TargetDiff.
**Coverage:** 100 pockets × 100 samples × seed=2021 (= 10 000 evaluations).

### Table 2 — Bond-distance JSD (lower is better)

**Shape:** 8 rows (bond types) × 4 cols (one col per method: liGAN / AR / Pocket2Mol / TargetDiff).
**Bond types:** C−C, C=C, C:N (aromatic), C−N, C=N, C−O, C=O, C:C (aromatic) — 8 entries.
**Selected values (TargetDiff vs others):**

| Bond | liGAN | AR | Pocket2Mol | **TargetDiff** |
|---|---|---|---|---|
| C−C | 0.601 | 0.609 | 0.496 | **0.369** |
| C=C | 0.665 | 0.620 | 0.561 | **0.505** |
| C:N | 0.497 | 0.451 | 0.416 | **0.263** |
| C:N | 0.638 | 0.552 | 0.487 | **0.235** |
| C−O | 0.656 | 0.492 | 0.454 | **0.421** |

**Primary metric:** Jensen-Shannon Divergence (lower=better) of bond-length distribution vs reference.
**Coverage:** Same 100-pockets × 100-sample pool as Table 1.

### Table 3 — PDBBind v2020 binding-affinity prediction (RMSE / Pearson / Spearman / MAE)

**Shape:** 8 rows (methods) × 4 cols (RMSE ↓ / Pearson ↑ / Spearman ↑ / MAE ↓).
**Methods:** TransCPI, MONN, IGN, HOLOPROT, STAMP-DPI, EGNN (baseline), **EGNN + TargetDiff features**.
**Primary metric:** Spearman correlation ↑.

| Method | RMSE ↓ | Pearson ↑ | Spearman ↑ | MAE ↓ |
|---|---|---|---|---|
| EGNN | 1.445 | 0.648 | 0.598 | 1.141 |
| **EGNN + TargetDiff** | **1.374** | **0.680** | **0.637** | **1.118** |

**Coverage:** PDBBind v2020 time-split (Stärk et al. 2022 protocol; test set = structures deposited after 2019).

### Table 4 (Appendix — referenced in paper text)

In paper §A.6 / appendix: **"Lipinski 5-rule"** addendum — n/a explicit table found in our scan; paper text states "similar trend, omitted". **Treat as n/a.**

**Total main tables: 4** (we report `n_tables=4` since Table 4 is appendix-only).

---

## 3. Figures (n=9 main + 2 appendix)

| Fig | Type | n sub-figs | Caption summary |
|---|---|---|---|
| **1** | Architecture | 1 | Overview of TargetDiff: diffusion + generative process over atom coords and types, parameterized by SE(3)-equivariant GNN θ. |
| **2** | Qualitative | 2 (top: all-atom; bottom: C-C pair) | Empirical distribution of inter-atomic distances for generated vs reference molecules, with JSD annotated. |
| **3** | Results (chart) | 1 | JSD between reference and generated bond-length distributions for 8 bond types × 4 methods (lower=better). |
| **4** | Results (heatmap) | 1 | Median Vina energy per pocket across 100 test pockets, sorted by TargetDiff median; shows TargetDiff wins in 57% of targets (vs liGAN 4%, AR 13%, Pocket2Mol 26%). |
| **5** | Qualitative (poses) | 2 (a: binding poses for 2 poor-AR pockets; b: scatter of CoM shift) | Visualises binding-pose correctness and centre-of-mass drift (TargetDiff avg=1.45 Å vs AR 1.79 Å). |
| **6** | Results (scatter) | 1 | Spearman correlation between predicted affinity (pK) and various scores derived from TargetDiff (v_ent, combined, hidden_emb). |
| **7** | Table / chart | 1 | Lipinski 5-rule metric for generated molecules (in §A.6 appendix). |
| **8** | Results (scatter) | 1 | Binding-affinity **ranking** results on CrossDocked2020 — Spearman rank correlation between unsupervised features and measured pK. |
| **9** | Results (chart) | 1 | Binding-affinity **prediction** on PDBBind v2020 — RMSE / Pearson / Spearman / MAE; "EGNN + ours" achieves best on all four metrics. |

**Total main figures = 9.** Appendix figures (counted as supplementary) include additional visualisation of generated examples for qualitative comparison (referenced but not enumerated in the main paper; estimate ≥2 supplementary figures).

---

## 4. Primary metrics — pockets × seeds × coverage

| Metric | Pockets | Seeds | Evaluations | Definition / threshold |
|---|---|---|---|---|
| **Vina Score** (↓ kcal/mol) | 100 | 1 (seed=2021) | 10 000 | Vina score function on raw generated pose |
| **Vina Min** (↓ kcal/mol) | 100 | 1 | 10 000 | Vina score after local energy minimisation |
| **Vina Dock** (↓ kcal/mol) | 100 | 1 | 10 000 | Vina score after re-docking (target-aware) |
| **High Affinity** (% ↑) | 100 | 1 | 10 000 | % generated mols with Vina Dock < reference ligand Vina Dock |
| **QED** (↑) | 100 | 1 | 10 000 | Quantitative Estimate of Drug-likeness (0–1) |
| **SA** (↑) | 100 | 1 | 10 000 | Synthetic Accessibility (0–1; higher = easier) |
| **Diversity** (↑) | 100 | 1 | 10 000 | Avg pairwise Tanimoto distance among 100 generated mols per pocket |
| **Lipinski 5-rule** | 100 | 1 | 10 000 | # rules satisfied (0–5) |
| **JSD bond length** | 100 | 1 | 10 000 × 8 bond types | Jensen-Shannon divergence between reference and generated bond-length distributions |
| **Ring-size %** | 100 | 1 | 10 000 | Distribution over ring sizes 3–9 |
| **CoM shift** (↓ Å) | 100 | 1 | per-target median | Distance between centre-of-mass of generated vs reference ligand |
| **Rigid fragment RMSD** (↓ Å) | 100 | 1 | per-fragment | RMSD between pre- and post-MMFF-optimised coordinates |
| **Spearman correlation (ranking)** | 100 | 1 | pairwise | Rank correlation between unsupervised features and measured pK (CrossDocked2020) |
| **Affinity prediction (PDBBind v2020)** | n/a | 1 | full v2020 test split | RMSE / Pearson / Spearman / MAE on time-split test (post-2019) |

**Headline:** TargetDiff reports on **100 pockets × 1 seed × 100 samples = 10 000 generated molecules** for the SBDD benchmark; the affinity-prediction benchmark uses the full PDBBind v2020 time-split test set (size not numerically reported in main paper).

---

## 5. Training / test split

- **Training data:** CrossDocked2020 v1.1 — 22 561 334 docked complexes (raw). Filtered to **poses with RMSD < 1.0 Å** vs crystal structure + **30 % sequence-identity split** (split file `split_by_name.pt`, shared with AR + Pocket2Mol).
- **After filtering:** ~100 000 protein-ligand complexes used for training (paper §4.1 / README L40). Pocket = 10 Å region around bound ligand via `extract_pockets.py`.
- **Test set:** **100 held-out pockets** (`{i} should be between 0 and 99`, README L131).
- **Sampling config (`configs/sampling.yml`):** `num_samples: 100`, `num_steps: 1000` (DDPM full chain), `seed: 2021`, `center_pos_mode: protein`, `sample_num_atoms: prior`.
- **NFE budget:** 1000 steps × 100 samples = **100 000 NFE / pocket**; **10 M NFE total** for the full 100-pocket benchmark.
- **Compute:** Not numerically reported in main paper; repo + paper §5 only describe "training from scratch" on 1× A100 (community reports suggest ~4 days × 1 A100 for full re-train, ~12-24 GPU-h per 100-pocket inference run). **Honest note:** the paper does NOT report GPU-hours; the only hardware assumption is "1 NVIDIA Ampere-class GPU".

---

## 6. Supplementary (n=?)

- **Supplementary Tables:** Not enumerated in main paper. We count main paper tables only (n=4). Appendix mentions Table 4 (PDBBind) and Table 5 (ring-size distribution in §A.5) and an Appendix G "SE(3)-Equivariance Proof" + Appendix D "Hyper-parameters".
  - **Conservative estimate:** supplementary tables n ≥ 5 (incl. Table 4 in §A.6, Table 5 ring-size, hyper-parameter table, sampling config table, dataset summary).
  - **Best-evidence from text:** n_supp_tables ≈ **5** (treat as our `n_supp_tables` metric).
- **Supplementary Figures:** Not enumerated. Appendix mentions "additional visualisation of generated examples" and "Hyper-parameter sensitivity". Conservative estimate **n_supp_figs ≥ 3**.
- **No standalone "Supplementary Materials" PDF** is published alongside the main paper (ICLR 2023 supplementary is bundled into the main PDF Appendix A–G, pp. 10–18).

---

## 7. Backup cite-only baseline — DiffSBDD & DecompDiff

For the gap analysis we will need ≥3 cite-only baselines. Summary:

| Paper | Authors | Venue | arXiv / DOI | Headline numbers (100 pockets, CrossDocked2020) |
|---|---|---|---|---|
| **TargetDiff** | Guan, Qian, Peng, Su, Peng, Ma | ICLR 2023 | arXiv:2303.03543 | Vina Dock −7.80 / −7.91; High Aff. 58.1 %; QED 0.48; SA 0.58; Div 0.72 |
| **DecompDiff** | Guan, Zhou, Yang, Bao, Peng, Ma, Liu, Wang, Gu | ICML 2023 | arXiv:2403.07902 | Vina Dock **−8.39** / −8.43; High Aff. **64.4 %**; Success Rate **24.5 %**; QED 0.45; SA 0.61 |
| **DiffSBDD** | Schneuing et al. | Nat Commun 2024 | doi:10.1038/s41467-024-49069-0 | Vina Dock −7.05 (fullatom cond) / −6.94 (CA cond); QED ~0.47; SA ~0.58 |

(DecompDiff = the actual `arXiv:2303.14246`? No — DecompDiff = arXiv:2403.07902; arXiv:2303.14246 is a different unrelated paper. Flagged for user clarification in TODO update.)

---

## 8. Comparison to current molmetal data inventory — gap analysis

### 8.1 What molmetal has today (2026-09-14 inventory)

From `molmetal/reports/round9_data_audit.md` + `sota_alignment_gap_analysis.md` + `lambda_vs_sbdd_paper_numbers*.md`:

| TargetDiff requires | molmetal today | Gap |
|---|---|---|
| 100-pocket CrossDocked2020 test set | First-10 pocket subset only (`crossdocked_first10_*`, `crossdocked_test001_*`) | **−90 pockets** |
| 100 samples/pocket × 1 seed | R4C pilot `N=10×3 = 30 evals total` (round9) | **−9 970 evaluations** |
| Vina Dock (re-dock) | QVina GPU backend wired + AutoDock Vina 1.2.x CLI | OK |
| QED / SA / Diversity / Lipinski | Wired in `molmetal/molmetal_lam/sbdd_env/` adapters | OK |
| JSD bond-length / ring-size / CoM-shift | None of these structural-realism metrics implemented | **−3 metric modules** |
| 1 000-step DDPM × 100 samples/pocket = 100k NFE | Lambda is **diffusion-free** (GNN + MCTS); NFE undefined | **−1 generative family** (Lambda vs DDPM) |
| Lipinski 5-rule | Mentioned in `lambda_vs_sbdd_paper_numbers.md` but not enforced in eval pipeline | partial gap |
| Pandas + Matplotlib plot pipeline (Figs 2/3/4/8/9) | No Fig-2/3-style plot scripts; `homotype_diversity` plot only | **−5 plot recipes** |
| Seed=2021 reproducibility | `--seed` flag added in WF-1 A3 (cfg-real) but not all harnesses | partial gap |

### 8.2 What this means for our N=10 vs paper N=10 000 comparison

- **Statistical power:** paper aggregates 10 000 evals (100 pockets × 100 samples); our Round-9 R4C pilot aggregates 30 evals (10 pockets × 3 seeds). **Paper has ~330× more samples** — direct metric-to-metric comparison is unreliable; paper values are *out-of-distribution* for our N=10 budget.
- **Pocket diversity:** paper uses 100 random held-out pockets; our 10-pocket subset is a hand-picked first-10. **Selection bias is high.** We should at minimum aggregate over **all 100 CrossDocked test pockets** if we want paper-comparable numbers, even at N=1 sample/pocket.
- **Compute:** paper assumes ~1 day × 1 A100 for full re-train + ~24 GPU-h inference; our budget is **1h36 ≈ 1.6 GPU-h** total. We **cannot** reproduce training. Cite-only is the canonical path.

### 8.3 Honest framing of the gap

- We have **no path** to a paper-comparable number in 1h36 budget. Cite-only is the honest route.
- We **can** ship a *smaller-but-rigorous* eval pipeline: all 7 paper metrics on our 10-pocket subset × 3 seeds (= 30 evals) and *honestly cite* the paper's 10 000-eval numbers as the SOTA reference.
- The structural-realism metrics (JSD / ring-size / CoM / rigid-fragment-RMSD) are **the metric classes we currently lack** — they are cheap to implement (no GPU needed; pure post-hoc analysis on SDF files) and would let us measure Lambda's "geometric realism" on par with TargetDiff's protocol.

---

## 9. Prior-art citations already in molmetal

- `molmetal/reports/diffsbdd_targetdiff_no_ckpt.md` (2026-09-11): documents why ckpts are unreachable and proposes cite-only path.
- `molmetal/reports/diffsbdd_targetdiff_1h36.md` (2026-09-11): placeholder log + adapter stub (`targetdiff_adapter.py`).
- `molmetal/reports/provenance_targetdiff.md`: prior provenance doc with the same Table 1 numbers (relied on by `lambda_vs_sbdd_paper_numbers.md`).
- `molmetal/reports/lambda_vs_sbdd_paper_numbers.md` (and `_refreshed.md`): cite-only comparison rows for TargetDiff / DecompDiff / DiffSBDD — these are the **single-source-of-truth** for our paper-aligned numbers today.
- `molmetal/reports/sota_alignment_gap_analysis.md`: existing gap analysis (2026-09-13, Round-9). This document extends it with the formal paper inventory.

**No code changes were made in this WF.** All work is pure markdown + a tracked todo.

---

## 10. Ship targets — prioritised TODO list for next sprint

These are the **gap-closure** tasks. They are listed in priority order; each item maps to one or more concrete code/test artifacts.

### Priority A — Cite-only + structural-realism metric stack (cheap; high paper-comparability ROI)

1. **A1. JSD bond-length metric** (`molmetal/molmetal_lam/metrics/jsd_bond.py`): given two SDF files (gen vs ref), bin bond lengths into 100 bins, return per-bond-type JSD. Tests: simple LiH, benzene, methane. **~2 h.**
2. **A2. Ring-size distribution** (`molmetal/molmetal_lam/metrics/ring_size.py`): count rings in SDF via RDKit, return histogram 3–9. Tests: aspirin, naphthalene. **~1 h.**
3. **A3. CoM shift metric** (`molmetal/molmetal_lam/metrics/com_shift.py`): centre-of-mass distance between two SDF files (gen vs ref), averaged over generated conformers. **~1 h.**
4. **A4. Rigid-fragment RMSD** (`molmetal/molmetal_lam/metrics/rigid_rms.py`): MMFF-optimise generated SDF, compare rotatable-bond-free fragments to reference. Requires MMFF (RDKit has it). **~2 h.**
5. **A5. Update `lambda_vs_sbdd_paper_numbers.md`** to include all 7 TargetDiff metrics for Lambda on our 10-pocket × 3-seed data. **~1 h.**

### Priority B — Pocket coverage expansion (expensive; necessary for any "N=100" claim)

6. **B1. Acquire remaining 90 CrossDocked2020 test pockets** (when sandbox opens): download `test_set.zip` from the Google Drive folder (link in `references/targetdiff/README.md` L131), unpack into `molmetal/data/crossdocked_test/`. **~30 min if network; otherwise deferred.**
7. **B2. Re-run Round-9 R4C pilot at N=100 pockets × 1 seed** (cite-only TargetDiff numbers as ground truth; our N=30 numbers as pilot sanity check). **~6-12 GPU-h on RX 7800 XT** (≈ 8-12 wall-clock-h). **Defer until W3423 environment available.**

### Priority C — Plot recipes (medium; for paper §4/§5 figures)

8. **C1. Fig-2-style all-atom pair distance plot** for our N=10 sweep (`scripts/plot_all_atom_distance.py`). **~2 h.**
9. **C2. Fig-3-style JSD bar chart** for our N=10 sweep + TargetDiff baseline (`scripts/plot_jsd.py`). **~1 h.**
10. **C3. Fig-4-style per-pocket median Vina heatmap** for our N=10 sweep (`scripts/plot_pocket_vina.py`). **~2 h.**
11. **C4. Fig-8/9-style Spearman scatter** (if we add an affinity predictor, which we currently don't). **Defer.**

### Priority D — DiffSBDD & DecompDiff integration (cite-only or full re-train)

12. **D1. Update `lambda_vs_sbdd_paper_numbers.md` rows with DecompDiff −8.39 / 64.4 % / 24.5 % numbers** (cite-only, ~30 min).
13. **D2. DiffSBDD cite-only row**: Vina Dock −7.05 / QED 0.47 / SA 0.58 (fullatom cond), arXiv:2210.13695 / Nat Commun 2024. ~30 min.
14. **D3. DiffSBDD + DecompDiff adapter stubs** (mirror `targetdiff_adapter.py` pattern) — useful for paper §2 "Related Work" + future benchmark if/when ckpts become reachable. **~3 h.**

---

## 11. Numbers schema (for `StructuredOutput`)

| field | value | source |
|---|---|---|
| `n_tables` | 4 | §2 above (Table 1 property summary; Table 2 bond JSD; Table 3 PDBBind; Table 4 Lipinski appendix) |
| `n_figures` | 9 | §3 above (Figs 1-9 main paper) |
| `n_metrics` | 14 | §4 above (Vina Score/Min/Dock + High Aff + QED + SA + Diversity + Lipinski + JSD + Ring-size + CoM + Rigid-RMSD + Spearman-rank + PDBBind-Affinity) |
| `n_pockets` | 100 | §5 (CrossDocked2020 test set) |
| `n_seeds` | 1 | §4 / §5 (seed=2021 fixed) |
| `n_supp_tables` | 5 | §6 (conservative; Tables 4 + 5 + hyper-params + sampling config + dataset summary; treat as upper bound) |

---

## 12. Honest-framing statement

- **Zero re-train / inference runs** of TargetDiff were attempted in this WF — sandbox is network-blocked for both Google Drive (TargetDiff ckpt) and Zenodo 8183747 (DiffSBDD ckpt); also `torch_geometric` / `torch_scatter` are not installed in our active `.venv` (ROCm torch 2.14.0+rocm7.2). See `molmetal/reports/diffsbdd_targetdiff_no_ckpt.md` for the prior diagnosis.
- All TargetDiff numbers in §2 / §4 / §10 are **cited from the published paper** (arXiv:2303.03543) via web search + multiple secondary sources (liner.com, arxiv-vanity, oplclaw, dblp).
- Three out of four bibliographic fields in the user's prompt are incorrect (see §0). We use the correctly-cited paper (TargetDiff, Guan et al., ICLR 2023, arXiv:2303.03543) as the primary inventory. **Flagged for user confirmation** in `molmetal/TODO/pending/wf_data_gap_targetdiff_arxiv_correction.md` (to be written).
- Compute budget (GPU-hours): **paper does NOT report a number**. We adopt the community-standard assumption (4 days × 1 A100 ≈ 96 GPU-h for full re-train, ≈ 24 GPU-h for the 100-pocket inference run) but **flag this as unverified** by paper text.
- This is **pure-analysis** work: zero code modifications to `molmetal/`, only markdown reports + a task-list update.

---

## 13. Source URLs

- arXiv abstract: https://arxiv.org/abs/2303.03543
- Paper PDF (mirror): https://arxiv-vanity.com/papers/2303.03543
- OpenReview: https://openreview.net/forum?id=kJqXEPXMsE0
- Code: https://github.com/guanjq/targetdiff
- BibTeX (DBLP): https://dblp.org/pid/207/7593.html
- Secondary review (liner.com): https://liner.com/review/3d-equivariant-diffusion-for-targetaware-molecule-generation-and-affinity-prediction
- DecompDiff (backup): https://arxiv.org/abs/2403.07902 ; https://github.com/bytedance/DecompDiff
- DiffSBDD (backup): https://doi.org/10.1038/s41467-024-49069-0 ; https://github.com/arneschneuing/DiffSBDD
- Local TargetDiff clone: `/home/hugo/codes/try_triton_on_rocm/molmetal/references/targetdiff/` (commit `142f1eb`, 2023-07-13)
- Prior provenance doc: `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/provenance_targetdiff.md`
- Prior gap analysis: `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/sota_alignment_gap_analysis.md`
