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
  gate 3), RDKit validity (1.000), m1 vs our own single-task baseline (+0.049).
- **Protocol-aligned — genuinely behind:** MoleculeNet clintox (0.7016 vs chemprop-D-MPNN 0.894, −0.19),
  hiv (0.7470 vs 0.823, −0.08), bbbp (0.8720 vs 0.940, −0.07), m19 LOMO real-only (0.5575 vs target 0.70),
  pearson r pIC50 (0.309 vs ridge 0.572).
- **Not protocol-aligned (do NOT put in a comparison table):** m17 on the fallback cohort; m2 R² on n=200
  cohorts (structural overfit, not a protocol gap); the Tier-2 SOTA column in METRICS_TARGETS (100-pocket
  / Uni-Mol-pretrained numbers we are explicitly not competing with).

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
| **MoleculeNet clintox** | 0.7016 | chemprop D-MPNN **0.894** | **−0.192** | `moleculenet_auc.json::tasks.clintox` |
| **MoleculeNet hiv** | 0.7470 | chemprop D-MPNN **0.823** | −0.076 | `moleculenet_auc.json::tasks.hiv` |
| **MoleculeNet bbbp** | 0.8720 | chemprop D-MPNN **0.940** | −0.068 (we beat chemprop RF 0.728) | `moleculenet_auc.json::tasks.bbbp` |
| **MoleculeNet bace** | 0.8609 | chemprop D-MPNN 0.876 | −0.015 | `moleculenet_auc.json::tasks.bace` |
| **m19 LOMO (real-only)** | 0.5575 | target 0.70 | −0.14 | `metal_lomo_auc_wasserstein.json` |
| **pearson r pIC50** | 0.3091 | ridge 0.572 | −0.263 | `pearson_r_pic50.json` |

**Caveat on MoleculeNet:** our run uses **Morgan-FP + MLP** (`morgan_fp_2048_r2`), not a D-MPNN
backbone. The −0.19 on clintox is therefore substantially a **representation/architecture** gap, not a
"method doesn't work" gap — a D-MPNN backbone on the same splits is the natural test.
**Caveat on m19:** the older 0.7577 was half-synthetic (500 fabricated Rh rows); the real-only 25-run
value is 0.5575 and is the honest number.

---

## §3 Not protocol-aligned — excluded from any comparison table

| Metric | Value | Why excluded |
|---|---|---|
| m17 OOD AUC | 0.7479 assay | measured on `metacytotox_fallback` (5,207 rows), **not** official DrugOOD IC50 splits; iMoLD margin is gate arithmetic only. Regen pending (official CSVs on disk). |
| m2 R² platinai/hilbert | −0.144 | n=200 cohorts; structural overfit with 50+-feature Ridge, not a protocol gap. |
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
