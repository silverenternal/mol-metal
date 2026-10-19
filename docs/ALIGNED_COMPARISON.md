# Aligned Comparison — Same-Protocol Head-to-Head vs Baselines

**Date:** 2026-10-08
**Purpose:** Answer one question — *under same dataset + same protocol + same metric, how do our
numbers actually compare to the baselines?* The per-metric view in
[`PAPER_SECTION_3_METRIC_TABLE_FINAL.md`](PAPER_SECTION_3_METRIC_TABLE_FINAL.md) and the gate view in
[`../TODO/02_metrics/METRICS_TARGETS.md`](../TODO/02_metrics/METRICS_TARGETS.md) mix in cross-protocol
SOTA targets, which over-states how far behind we are. This file separates the two.
**Status:** READ-only analysis. Every number traces to a `metrics/by_metric/*.json` on disk (MEASURED).
**Principle** (COMPARISON_BASELINES.md §0): *"我们不需要 reproduce 别人的数字 — same dataset + same
protocol + same metric 才 comparable."*

---

## §0 TL;DR — the honest split

- **Protocol-aligned (same dataset+task+metric) — competitive or winning:** m1 Ru AUC (paper-protocol
  replication 0.7717 vs Krasnov 0.810, cross-cohort), m21 NCI-60 9-cell (0.8007 vs 0.81), Krasnov-aligned
  chemprop 3-metal (0.7812), m15 RMSD (0.6578 Å, 5/5 pass 2.0 Å), decode_ratio (0.998), n_distinct (20 vs
  gate 3), RDKit validity (1.000), m1 vs our own single-task baseline (+0.049), MoleculeNet mean 0.8149
  (D-MPNN backbone, 5-seed, passes the 0.80 gate).
- **Protocol-aligned — genuinely behind:** MoleculeNet clintox (0.8209 vs chemprop-D-MPNN 0.894, −0.073),
  hiv (0.7900 vs 0.823, −0.033), bbbp (0.9065 vs 0.940, −0.034), bace (0.7421 vs 0.876, −0.134 — a real
  architecture-level miss, not seed noise), m19 LOMO real-only (0.5575 vs target 0.70; 0.5845 with
  per-metal centering).
- **Readout finding, NOT a feature gap:** pearson r pIC50 — the bottleneck is the neural head, not the
  multi-task α (3-seed ridge probe on the frozen encoder embedding 0.403 ± 0.004 vs head 0.198 ± 0.023;
  we already beat the same-split Morgan ridge 0.3232). A random encoder scores nearly as well (0.378),
  so the encoder learned little pIC50 structure — the readable signal is generic. See §2b.
- **Not protocol-aligned (do NOT put in a comparison table):** m17 on the fallback cohort; m2 R² on n=200
  cohorts (structural overfit, not a protocol gap); the pearson **0.572** ridge (different cohort/split —
  see §2b); the Tier-2 SOTA column in METRICS_TARGETS (100-pocket / Uni-Mol-pretrained numbers we are
  explicitly not competing with).

---

## §1 Aligned + competitive / winning

| Metric | Ours (MEASURED) | Baseline (same protocol) | Verdict | Source JSON |
|---|---|---|---|---|
| **m1 Ru AUC — paper-protocol replication** | **0.7717 ± 0.0380** | Krasnov 2026 LightGBM **0.810 ± 0.024** (DOI-split 10-fold CV) | −0.038, **cross-cohort** (our 591 vs paper 23 cell lines); D-MPNN vs LightGBM | `metalcytotox_ru_auc.json::paper_protocol_measurement` |
| **m1 Ru AUC — our own single-task baseline** | 0.7613 | 0.7123 (single-task, 5-epoch) | **+0.049** apples-to-apples internal | `metalcytotox_ru_auc.json` |
| **Krasnov-aligned chemprop v2 (3-metal)** | **0.7812** (Re 0.8027 / Os 0.7884 / Rh 0.7525) | Krasnov per-metal | scaffold-split (Bemis-Murcko), 9/9 runs | `wf219_krasnov_aligned_chemprop_v2_3seed_50epoch_scaffold.json` |
| **m21 NCI-60 9-cell AUC** | **0.8007 ± 0.0160** | Krasnov 0.81 (single-cell-line avg) | ≈ tie (−0.009); MCF7 0.846 / A549 0.870 / A2780cis 0.721 | `nci60_precious_9cell_auc.json` |
| **m15 docking RMSD** | **0.6578 Å** (5/5 pockets ≤ 2.0 Å) | DiffDock 38.2% < 2 Å | **100% vs 38.2%** pass-rate (self-built 5-pocket, no external table) | `docked_pose_rmsd.json` |
| **m14 decode_ratio (real)** | **0.998** | TargetDiff lit ≥ 0.5 | **~2× lit** | `decode_ratio.json` |
| **m13 RDKit validity** | **1.000** | SOTA 0.99–1.00 | met | `rdkit_validity.json` |
| **n_distinct metal-ligand pairs** | **20** | gate ≥ 3 | **6.7× gate** | `n_distinct.json` |

**Why m1 is reportable:** the replication matches the paper's *criteria* (DOI-split 10-fold CV, dedup +
non-cancer funnel, release-identical v2 file, md5-verified) — only the cohort *scope* differs (591 vs 23
cell lines). It is a cross-cohort estimate, labelled as such, not a reproduction.

---

## §2 Aligned + genuinely behind

| Metric | Ours (MEASURED) | Baseline (same protocol) | Gap | Source JSON |
|---|---|---|---|---|
| **MoleculeNet clintox** | 0.7016 (Morgan-MLP) → **0.8209 ± 0.0122** (D-MPNN, 5-seed) | chemprop D-MPNN **0.894** | −0.192 → **−0.073** | `moleculenet_auc.json::tasks.clintox` / `moleculenet_auc_dmpnn_5seed.json::tasks.clintox` |
| **MoleculeNet hiv** | 0.7470 → **0.7900 ± 0.0109** (D-MPNN, 5-seed) | chemprop D-MPNN **0.823** | −0.076 → **−0.033** | `moleculenet_auc_dmpnn_5seed.json::tasks.hiv` |
| **MoleculeNet bbbp** | 0.8720 → **0.9065 ± 0.0227** (D-MPNN, 5-seed) | chemprop D-MPNN **0.940** | −0.068 → **−0.034** | `moleculenet_auc_dmpnn_5seed.json::tasks.bbbp` |
| **MoleculeNet bace** | 0.8609 → **0.7421 ± 0.0446** (D-MPNN, 5-seed) | chemprop D-MPNN 0.876 | **−0.134** (D-MPNN worse; 5-seed std 0.045 — a REAL regression, not seed noise) | `moleculenet_auc_dmpnn_5seed.json::tasks.bace` |
| **m19 LOMO (real-only)** | 0.5575 (0.5845 w/ per-metal centering) | target 0.70 | −0.14 (−0.116) | `metal_lomo_auc_wasserstein.json` |

**Caveat on MoleculeNet:** the original run was **Morgan-FP + MLP** (`morgan_fp_2048_r2`); the aligned
D-MPNN backbone (same Bemis-Murcko scaffold split, same ROC-AUC) closes most of the gap — Morgan mean
0.7954 → D-MPNN **0.8149** (5-seed; 3-seed was 0.8161), passing the 0.80 gate. The residual clintox
(−0.073) / hiv (−0.033) / bbbp (−0.034) gaps are now tuning, not architecture. **`bace` is the one real
architecture-level miss**: 5 seeds give 0.7421 ± 0.0446 (tightening the 3-seed ± 0.0687 without moving the
mean), so the −0.134 vs chemprop is a genuine D-MPNN regression and is reported as an honest-negative.

**Split-key caveat (MEASURED 2026-10-19, `docs/DATA_LEAKAGE_AUDIT.md` §5):** our split key
(`train_moleculenet._bemis_murcko_scaffold`) keeps stereochemistry, while the reference
implementation — `astartes`' `Scaffold.generate_bemis_murcko_scaffold`, which chemprop moved to — defaults
to `include_chirality=False`. A chirality-retaining key splits less, so **our MoleculeNet numbers are
measured on a marginally easier split than the chemprop column they are compared against**. The audit
quantifies it: re-keying stereo-blind leaves up to 23 shared scaffolds (bace) covering 32.34 % of test
rows; desalting adds a counterion component (hiv: 0 → 52 scaffolds), with a row-level maximum of 32.68 %
(bbbp). The bias runs **in our favour**: the gap to chemprop is understated until the key is aligned and
the four tasks are re-run. This is an open item for the model session; no number in the table above has
been re-measured under the aligned key.

**bace rescue attempt (MEASURED, honest-negative — 2026-10-08):** the hypothesis that the
unconditional `pos_weight=neg/pos` class weighting (applied even to near-balanced bace, pos_rate≈0.45)
depressed AUC was tested with a `--class-weight balanced` + dropout 0.1 + wd 1e-5 + patience 20 rescue
(`moleculenet_auc_dmpnn_5seed_rescue.json`). It did **not** pay off: bace went **0.7421 → 0.6805**
(−0.062, WORSE); clintox **0.8209 → 0.8332** (+0.012, inside its ±0.0135 std). A no-op **bbbp control**
(same config, no reg) reproduced **0.8939 vs 0.9065** baseline (−0.013, inside ±0.018 std), bounding
harness drift. **Confound to state:** the baseline was GPU, the rescue was CPU (`--device` defaults to
cuda:0 when free; both GPUs were busy); the bbbp control bounds the device effect at ~±0.015 but a
clean GPU re-run with a same-device control is the confirmation still owed. The class-weighting is
therefore **not** the cause of the bace gap — do not re-claim it as a fix.

**Caveat on m19:** the older 0.7577 was half-synthetic (500 fabricated Rh rows); the real-only 25-run
value is 0.5575 and is the honest number. Per-metal pIC50 centering lifts the real-only aggregate to
0.5845 ± 0.0764 (inside noise) but moves the worst fold **Rh 0.3812 → 0.5022** decisively.
**Stale-file flag (2026-10-08):** the canonical `metrics/by_metric/metal_lomo_auc_wasserstein.json` still
carries the older **0.5575 / Rh 0.3812** (ts 06:51) while the newer run report
`molmetal/reports/m19_per_metal_centering_2026_10_08/metrics.json` carries **0.5845 / Rh 0.5022**
(ts 21:17). Both are real, different runs; refresh the canonical file after the next re-measure rather
than citing both. The newer centering run had W₂ **off** (`wasserstein_weight=0.0`), a double class
correction (`pos_weight_mode=auto` + `balanced_sampler=true`), and no scheduler (`lr_scheduler=none`,
`weight_decay=0.0`) — the untested levers for any further lift. Note `--wasserstein-real` alone is a
silent no-op: `w2_enabled` requires `wasserstein_real AND wasserstein_weight > 0.0`.

### §2b Readout finding — pearson r pIC50 (NOT a feature gap)

3-seed means (seeds 0/1/2, same split, n_test=994), from the per-seed JSONs in
`molmetal/reports/wf_pearson_diag_3seed/`:

| Probe (same split, mean ± std over seeds 0/1/2) | test Pearson | Source |
|---|---|---|
| ridge on **trained** D-MPNN encoder embedding (α=0.2) | **0.4033 ± 0.0036** | `wf_pearson_diag_3seed/seed{0,1,2}.json` |
| ridge on **trained** encoder embedding (α=0.5) | 0.4019 ± 0.0058 | same |
| ridge on **random** (untrained) encoder embedding | 0.3780 ± 0.0116 | same |
| ridge on **Morgan FP** | 0.3232 (split-fixed, no seed var) | same |
| neural pIC50 head (default α=0.5) | 0.1978 ± 0.0233 | same |
| neural head, regression-weighted (α=0.2) | 0.1710 ± 0.0057 | same |

The bottleneck is the **readout**, not the multi-task α: a linear probe on the frozen encoder
embedding extracts **~2×** the signal the trained neural head does (0.403 vs 0.198; the ridge's
worst seed 0.396 is above the head's best seed 0.219 — no overlap), and re-weighting the loss
toward regression makes the head *worse* (0.171 < 0.198, refuting the α-starvation hypothesis).

**Honest caveat:** the probe on a **random** encoder is nearly as good as on the trained one
(0.378 vs 0.403). So most of the readable signal lives in "the pooled embedding has a linear
pIC50 direction at all", not in what training added to the representation — consistent with §2's
L5 finding that the input features carry almost no linear signal. The fix is the readout, but do
not claim the encoder learned a strong pIC50 representation; on this split we already exceed the
same-split Morgan ridge (0.403 vs 0.323).

---

## §3 Not protocol-aligned — excluded from any comparison table

| Metric | Value | Why excluded |
|---|---|---|
| m17 OOD AUC | 0.7479 assay | measured on `metacytotox_fallback` (5,207 rows), **not** official DrugOOD IC50 splits; iMoLD margin is gate arithmetic only. Regen pending (official CSVs on disk). |
| m2 R² platinai/hilbert | −0.144 | n=200 cohorts; structural overfit with 50+-feature Ridge, not a protocol gap. |
| pearson ridge 0.572 | 0.572 | measured on the `pic50_margin_sweep` cohort (HeLa48h/dark 702 formulations, 383 scaffolds, seeds 42/0/1234) — a **different split** from the 2026-10-08 readout diagnostic (seeds 0/1/2, n_test=994). Same-split Morgan ridge is 0.3232. Do NOT cite as a same-protocol gap; see §2b. |
| m3 Vina median | −3.103 (cisplatin) / −7.782 (organic) | no external paper baseline exists on our 5-pocket subset. |
| Tier-2 SOTA column | e.g. TransDiffSBDD −9.37 | 100-pocket / Uni-Mol-pretrained — explicitly out of scope per COMPARISON_BASELINES §2. |

---

## §4 Stale artifacts found during this analysis (doc-lag-disk)

- `metrics/by_metric/tier1_summary.json` + `tier2_summary.json` are **dated 2026-09-25**; their
  `metric_1_ru_subset_auc = 0.7123` predates the 0.7613 (09-27) and 0.7717 (10-04) measurements. Their
  `moleculenet_mean_auc = 0.7954` / verdicts are from the same stale snapshot — superseded by the
  per-task numbers in §2.
- `TODO/03_baselines/COMPARISON_BASELINES.md` §6.2 itself flags that §1.1's *"Paper split: 5-fold CV"*
  line is superseded by the verified 10-fold CV reading — the §1.1 line was not edited.

---

## §5 Cross-references

- `PAPER_SECTION_3_METRIC_TABLE_FINAL.md` — per-metric §3 table (main + supplementary), the
  metric-centric companion to this protocol-centric view.
- `TODO/03_baselines/PROTOCOL_ALIGNMENT_MATRIX.md` — alignment-level legend per Tier-1 metric.
- `TODO/03_baselines/COMPARISON_BASELINES.md` — full baseline detail + §6 honest limitations.
- `TODO/02_metrics/METRICS_TARGETS.md` — target values (mixes cross-protocol SOTA; use with §3 above).
