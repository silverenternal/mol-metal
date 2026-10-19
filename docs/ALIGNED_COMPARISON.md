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
  gate 3), RDKit validity (1.000), m1 vs our own single-task baseline (+0.049), MoleculeNet mean 0.8095
  (D-MPNN backbone, 5-seed, **aligned split key**, passes the 0.80 gate), MoleculeNet bbbp/hiv vs
  same-protocol chemprop v2 (+0.033 / +0.028).
- **Protocol-aligned — genuinely behind:** MoleculeNet bace (0.7422 vs same-protocol chemprop v2 0.8279,
  −0.086; **and vs Morgan-MLP 0.8500, −0.108** — an architecture-level miss, not seed noise), clintox
  (0.7957 vs 0.8652, −0.070), m19 LOMO real-only (0.5575 vs target 0.70; 0.5845 with per-metal
  centering). The literature chemprop column (0.8832 mean) sits 0.050 above its own reproduction, so
  gaps quoted against it overstate the real one.
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
| **MoleculeNet clintox** | **0.7957 ± 0.1075** (D-MPNN 512/4, 5-seed, aligned key) | chemprop v2 **0.8652 ± 0.0236** | **−0.070** | `moleculenet_auc.json::tasks.clintox` / `moleculenet_auc_chemprop_v2_same_protocol.json::tasks.clintox` |
| **MoleculeNet bace** | **0.7422 ± 0.0828** (D-MPNN 512/4, 5-seed, aligned key) | chemprop v2 0.8279 / **Morgan-MLP 0.8500** | **−0.086** vs chemprop, **−0.108** vs Morgan | `moleculenet_auc.json::tasks.bace` / `moleculenet_auc_morgan_mlp_aligned.json::tasks.bace` |
| **m19 LOMO (real-only)** | 0.5575 (0.5845 w/ per-metal centering) | target 0.70 | −0.14 (−0.116) | `metal_lomo_auc_wasserstein.json` |

**Caveat on MoleculeNet:** under the aligned key the D-MPNN **wins bbbp (+0.033) and hiv (+0.028)** and
**loses bace (−0.086) and clintox (−0.070)** against same-protocol chemprop v2, for a net −0.024 on the
mean (0.8095 vs 0.8330). The earlier "behind on 4/4" reading came from the pre-alignment key and from
comparing against the literature constant rather than a reproduction. **`bace` is the one real
architecture-level miss**, and the sharper form of it is that Morgan-MLP (0.8500) beats our D-MPNN
(0.7422) by 0.108: this is not "we lose to message passing", it is a task where our architecture
underperforms a fixed fingerprint MLP. Reported as an honest-negative.

**Split-key caveat (MEASURED 2026-10-19, `docs/DATA_LEAKAGE_AUDIT.md` §5):** the split key
(`train_moleculenet._bemis_murcko_scaffold`) used to keep stereochemistry, while the reference
implementation — `astartes`' `Scaffold.generate_bemis_murcko_scaffold`, which chemprop moved to — defaults
to `include_chirality=False`. A chirality-retaining key splits less, so **the MoleculeNet numbers in the
table above were measured on a marginally easier split than the chemprop column they are compared
against**. The key is now bit-exact with `MurckoScaffoldSmiles(includeChirality=False)` on every row of
all four tasks. Re-keying turns over 63.6-76.5 % (bbbp), 59.3-83.5 % (bace) and 39.3-87.0 % (clintox) of
the test molecules across the five seeds; **hiv is bit-identical** because none of its scaffolds carry
stereocentres (`metrics/by_metric/moleculenet_scaffold_key_alignment.json`).

**Aligned-key re-run (MEASURED 2026-10-19).** The four-task, five-seed protocol at hidden 512 / depth 4
was re-run end-to-end under the aligned key (`configs/pa14_moleculenet_cap512_d4.yaml`, 40 min on GPU 0,
pre-flight PASS, all 20 per-seed outputs fresh):

| Task | Aligned key (512/4, 5-seed) | Pre-alignment 300/3 (table above) | chemprop D-MPNN |
|---|---:|---:|---:|
| bbbp | **0.9025 ± 0.0350** | 0.9065 ± 0.0227 | 0.940 |
| bace | **0.7422 ± 0.0828** | 0.7421 ± 0.0446 | 0.876 |
| hiv | **0.7976 ± 0.0170** | 0.7900 ± 0.0109 | 0.823 |
| clintox | **0.7957 ± 0.1075** | 0.8209 ± 0.0122 | 0.894 |
| mean | **0.8095** | 0.8149 | 0.883 |

Source: `metrics/by_metric/moleculenet_auc.json` (`split_key: bemis_murcko_stereo_blind`). The
pre-alignment 512/4 artifact is retained as `moleculenet_auc_dmpnn_5seed_cap512_d4.json`, explicitly
annotated as superseded. **Do not mix the two columns**: the test sets differ (59-87 % turnover), so a
row-by-row delta is not a paired comparison. bbbp and hiv are flat within seed noise (hiv's split is
unchanged, so its +0.008 is training nondeterminism); bace (−0.051) and clintox (−0.035) move down, but
both have 5-seed standard deviations of 0.08-0.11, so a 5-seed mean cannot resolve a shift of that size
as significant. What is certain is that all four numbers are now on the reference key.

**Same-protocol baselines, now re-measured on the aligned key (MEASURED 2026-10-19/20).** Both baseline
arms have been re-run under the aligned key, so the comparison is now genuinely same-protocol rather
than "same splitter, different key":

| Task | Ours D-MPNN 512/4 | chemprop v2 | Morgan-MLP | chemprop D-MPNN (lit.) | ours−cp | cp−lit |
|---|---:|---:|---:|---:|---:|---:|
| bbbp | **0.9025 ± 0.0350** | 0.8693 ± 0.0441 | 0.8610 ± 0.0527 | 0.940 | **+0.0332** | −0.0707 |
| bace | 0.7422 ± 0.0828 | 0.8279 ± 0.0118 | 0.8500 ± 0.0334 | 0.876 | **−0.0857** | −0.0481 |
| hiv | **0.7976 ± 0.0170** | 0.7698 ± 0.0258 | 0.7600 ± 0.0368 | 0.823 | **+0.0278** | −0.0532 |
| clintox | 0.7957 ± 0.1075 | 0.8652 ± 0.0236 | 0.7250 ± 0.0471 | 0.894 | **−0.0695** | −0.0288 |
| mean | 0.8095 | **0.8330** | 0.7990 | 0.8832 | −0.0235 | −0.0502 |

Sources: `moleculenet_auc.json`, `moleculenet_auc_chemprop_v2_same_protocol.json`,
`moleculenet_auc_morgan_mlp_aligned.json`. All three carry
`split_key: bemis_murcko_stereo_blind (includeChirality=False)`.

**The alignment is verified, not asserted.** Each seed re-splits (`_scaffold_split` seeds its RNG with
the training seed), so the five seeds are five different test sets and a single size cannot stand in
for the rest. Comparing the full per-seed size vector:

- Morgan-MLP vs D-MPNN: **5/5 seeds × 4/4 tasks identical** (e.g. bace 299/305/304/303/308).
- chemprop v2 vs D-MPNN: **seed 0 identical on 4/4 tasks** (bbbp 412, bace 299, hiv 8296,
  clintox 297). That report predates the per-seed vector, so only seed 0 is comparable; the
  synthesizer records `split_check.seeds_compared = 1` rather than implying more.

The check is decisive because the two keys give different sizes — bbbp 390→412, bace 302→299,
clintox 298→297, hiv 8296→8296 (unchanged, no stereocentres). A match is evidence; a mismatch aborts
the write (`synthesize_pa17_metric.py` returns 2 and writes nothing unless
`--allow-unverified-split` is passed, which stamps `split_check.verified = false`).

**What the three-way table says.** chemprop v2 is 0.0502 *below* the literature constant it is
usually quoted as — the reference column was best-of-search. Against the reproduced baseline we win
2/4 (bbbp +0.033, hiv +0.028) and lose 2/4 (bace −0.086, clintox −0.070), netting −0.024 on the
mean. The bace loss is the sharper one: **Morgan-MLP at 0.8500 beats our D-MPNN at 0.7422 by 0.108**,
so bace is not "we lose to message passing" — it is a task where our architecture underperforms even
a fixed fingerprint MLP. That remains an open honest-negative, not a tuning opportunity already
exploited.

**Superseded columns.** The pre-alignment chemprop artifact is preserved as
`moleculenet_auc_chemprop_v2_same_protocol_pre_alignment.json` and the pre-alignment Morgan artifact
as `moleculenet_auc_morgan_mlp.json`, both annotated `split_key: bemis_murcko_isomeric`. Do not read
either against the aligned table: the test sets differ by 39-87 %.

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
