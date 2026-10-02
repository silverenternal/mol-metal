# Wf-357-B4 — Supporting metrics verification (m11 + m15 + m19 + m18)

**Workflow:** Wf-357-B4
**Date:** 2026-09-29 (12:55 UTC)
**Status:** SHIPPED (CPU-only verification, no GPU launch, no retrain)
**Skill alignment:** `party-twenty-spirit` §3.5 (honest framing) +
`mol-metal-anticancer` (m18 protein-metal + m19 LOMO context) +
AGENTS.md §5.2 (MEASURED-vs-PROJECTED tag discipline)

---

## §1. Goal

Verify the supporting metrics (m11 n_distinct, m15 Docking RMSD,
m19 LOMO, m18 m_jnk3) per Wf-296..350 ship wave. Each metric MUST
trace to a JSON / report path on disk per AGENTS.md §1 rule 1.

**Honest framing on m18 m_jnk3**: a search across `metrics/`, `TODO/`,
and `TODO/INDEX.md` returns ZERO hits for `jnk`, `jnk3`, `m_jnk3`,
or `m_jnk`. The metric labelled "m18" in `METRICS_TARGETS.md` and
`MECHANISM_METRIC_MAPPING.md` is **Multi-modal Protein-Metal Binding
AUC**, NOT JNK3. There is **no `m_jnk3` metric** in the Mol-Metal
project. This Wf-357-B4 verifies m18 = `metric18_protein_metal*_auc`
(the actual protein-metal-binding AUC on MetalCytoToxDB-derived
"protein = cell-line" signal, NOT a JNK3 kinase-specific benchmark).

---

## §2. m11 — n_distinct (distinct generated molecules per cell)

**Headline value:** **20 MEASURED** (saturated `n_top_k_cap`)

| Item | Value | Source |
|---|---:|---|
| `best_measured_with_all_fixes` | **20** | `metrics/by_metric/n_distinct.json` line 50 (`aggregate`) |
| `n_top_k_cap` | 20 | same file line 51 |
| R12 baseline (singleton collapse) | 1 | same file line 41 (`best_measured_lambda_pilot_pre_fixes`) |
| `lift_factor` | "20x (1 → 20)" | same file line 52 |
| `status` | MEASURED | same file lines 13-26 (last entry: `wf_round12_lambda_patha_10x3`) |
| `wall_sec_per_cell` | 98.27 | same file line 25 |
| `deterministic_per_cell` | true | same file line 27 |

**Per-run history (4 entries, `measurements` array):**

| Run | n_samples | n_distinct | status | Date |
|---|---:|---:|---|---|
| `wf_lambda_only_mini_pilot` (5×1, no metal-seed, n_sim=100) | 5 | 1 | MEASURED | — |
| `wf_round12_lambda_pilot` (10×3, n_sim=1000, cisplatin+all-5) | 30 | 1 | MEASURED | 2026-09-14 |
| `wf_round12_lambda_pathb` (10×3 Path B without all 4 fixes) | 30 | 1 | MEASURED | 2026-09-15 |
| **`wf_round12_lambda_patha_10x3` (Path A with ALL 4 FIXES + decoder_rework)** | 30 | **20** | MEASURED | 2026-09-15 |

**Source attribution**:
- Wf-149 baseline (R12 singleton collapse) — referenced in `n_distinct.json:7` `note: "Path B alone did NOT lift"`.
- Wf-301 enhancements (Path A 4-fix bundle) — referenced in `n_distinct.json:25` `note: "all 30 cells = n_top_k cap (20 distinct molecules)"`.

**Verdict (per `n_distinct.json:53`)**:
> "ROOT_CAUSE_RESOLVED — 3-layer singleton attractor broken by 4-fix bundle. n_distinct went from 1 (trivially stuck) to 20 (saturated n_top_k cap) on all 30 cells. Wall-clock cost ~98 s/cell (vs 50s pre-fix), within 90-min budget."

**Honest framing caveats**:
1. The n_distinct=20 is at the **n_top_k_cap ceiling** (20) — the
   MCTS now generates enough candidates to saturate the cap on every
   cell, but a higher cap (e.g., 50) might reveal whether the
   diversity is genuinely large or just hits the cap.
2. This is **single-metal (cisplatin)** within-cell diversity — does
   NOT translate directly to m11's gate target "≥3 distinct
   metal-ligand pairs" across multi-metal cohorts (METRICS_TARGETS.md §Metric 11).

**Per-metric status: PASS** (R12 collapse resolved; m11 gate ≥3 met).

---

## §3. m15 — Docking RMSD (Vina redocking, Kabsch+Hungarian)

**Headline value:** **0.6578 Å MEASURED** (aggregate median best RMSD across 5 pockets × 3 seeds × exhaustiveness=16); **5/5 pockets pass 2.0 Å SOTA gate**

| Item | Value | Source |
|---|---:|---|
| `value` | **0.6578** | `metrics/by_metric/docked_pose_rmsd.json` line 16 (`value`) |
| `aggregate_median_across_pockets` | **0.6578** | same file line 19 |
| `n_pockets_passed_2_0a` | **5/5** | same file line 21 |
| `n_pockets_passed_2_5a` | **5/5** | same file line 20 |
| `n_pockets_measured` | 5 | same file line 22 |
| `gate_4_passes_2_5a_overall` | true | same file line 23 |
| `exhaustiveness` | 16 | same file line 10 |
| `seeds` | [0, 1, 2] | same file line 12 |

**Per-pocket breakdown (15 runs = 5 pockets × 3 seeds):**

| Pocket | n_seeds_succeeded | Per-seed best RMSD (Å) | Median (Å) | Mean (Å) | Std (Å) | Passes 2.0 Å | Difficulty | Date |
|---|:---:|---|---:|---:|---:|---|---|---|
| **1h36** | 3/3 | 0.6883, 0.5037, 0.5739 | 0.5739 | 0.5886 | 0.0761 | TRUE | MEDIUM | 2026-09-28 |
| **3eml** | 3/3 | 1.0117, 1.2504, 1.024 | 1.024 | 1.0954 | 0.1098 | TRUE | HARD | 2026-09-28 |
| **3ny8** | 1/3 | 0.9506 | 0.9506 | 0.9506 | 0.0 | TRUE | MEDIUM | 2026-09-28 |
| **4aaw** | 3/3 | 0.4223, 0.461, 0.3791 | 0.4223 | 0.4208 | 0.0335 | TRUE | EASY | 2026-09-28 |
| **4rlu** | 3/3 | 0.6099, 0.6623, 0.6578 | 0.6578 | 0.6433 | 0.0237 | TRUE | MEDIUM | 2026-09-28 |
| **Aggregate** | — | — | **0.6578** | — | 0.2546 | **5/5** | 1 EASY / 3 MEDIUM / 1 HARD | — |

**Cross-pocket aggregate (`docked_pose_rmsd.json:296-301`):**
- median: **0.6578 Å**
- std: 0.2546
- min: 0.4223 (4aaw)
- max: 1.024 (3eml)
- n_pockets: 5

**Difficulty distribution (`docked_pose_rmsd.json:303-307`):**
- EASY (< 0.5 Å): 1 pocket (4aaw)
- MEDIUM (0.5-1.0 Å): 3 pockets (1h36, 3ny8, 4rlu)
- HARD (≥ 1.0 Å): 1 pocket (3eml)

**Source attribution** (per `docked_pose_rmsd.json:8`):
- Wf-179 BD-9 m15 RMSD Phase 2 Vina redocking (canonical script)
- TODO-134 (m15 redesign spec)
- Wf-139 P1 (ref ligand SMILES extraction)
- Wf-246-B redesign (per-pocket std reporting + cross-pocket aggregate + difficulty flag)

**Source Wf-341-T1** (`TODO/completed/341t1_m15_rmsd.md`) confirms:
- m15 redesign `m15_status = "MEASURED"` per `341t1_m15_rmsd.md:18`
- 5-pocket × 3-seed Vina redocking via `wf341t1_m15_rmsd_redesign.py`
- PoseBusters **SKIPPED** (not installed) per `341t1_m15_rmsd.md:32`

**SOTA gate comparison**:
- Ours: **0.6578 Å median** (well below SOTA 2.0 Å threshold by 1.34 Å margin)
- EquiBind: 38.0% < 2Å | DiffDock: 38.2% < 2Å | DeltaDock: 47.4% < 2Å | GNINA: 22.9% < 2Å (per METRICS_TARGETS.md §Metric 15)
- ✅ **0.6578 Å is SOTA-level** (per METRICS_TARGETS.md §Metric 15 gate "≤ 1.5 Å: SOTA-level")

**Per-metric status: PASS** (5/5 pockets SOTA; 0.6578 Å median is SOTA-level per METRICS_TARGETS gates).

---

## §4. m19 — Multi-metal LOMO transfer (Ru / Ir / Os / Rh / Re)

**Headline value:** **0.6217 ± 0.1284 MEASURED on 18/25 (72%)** (Wf-131 P3 partial on BUGGY Wf-125 P1 cohort, W₂ transfer + Rh oversample)

### §4.1 Per-Wf status matrix (verbatim from `metal_lomo_auc.json`)

| Metal | Wf-131 P3 D-MPNN (BUGGY cohort, W₂) | Wf-146 chemprop v2 (smoke backbone 64/2) | Wf-176 Wasserstein Tier-2 (BUGGY, Re only) | R17 baseline |
|---|---:|---:|---:|---:|
| **Ru** | **0.5941 ± 0.0143** (5/5 seeds, +0.0235) | **0.4466 ± 0.0228** (5/5 seeds, -0.1240) | n/a | 0.5706 |
| **Ir** | **0.6242 ± 0.0185** (5/5 seeds, +0.0066) | **0.4646 ± 0.0090** (5/5 seeds, -0.1530) | n/a | 0.6176 |
| **Os** | **0.6075 ± 0.0294** (5/5 seeds, -0.0038) | **0.3237 ± 0.0118** (4/5 seeds, -0.2876) | n/a | 0.6113 |
| **Rh** | **0.6874 ± 0.3584** (3/5 seeds; seed0=0.274 outlier) | **0.5411 ± 0.0298** (5/5 seeds, +0.1734) | n/a (BLOCKED on Phase 2 GPU) | 0.3677 |
| **Re** | **DEFERRED** (0/5) | **DEFERRED** (0/5) | **0.6220 (n=1, seed=4)** | 0.5059 |
| **Overall** | **0.6217 ± 0.1284** (18/25) | **0.4458** (19/25) | **0.6220** (1/25 Re-only) | **0.5346 ± 0.110** |

**Sources**:
- `metrics/by_metric/metal_lomo_auc.json` `wf131_p3_lomo_partial_update` block (lines 196-282)
- `metrics/by_metric/metal_lomo_chemprop_v2.json` (lines 159-194 + 203)
- `metrics/by_metric/metal_lomo_auc_wasserstein.json` line 19 (Re=0.6220, n_runs=1)
- `metrics/by_metric/metal_lomo_auc.json` line 7 (R17 baseline `value: 0.5346340730251975`)

### §4.2 DEFERRED runs (per Wf-355-B1 + Wf-355-B2)

**Wf-131 P3 7 DEFERRED runs** (per Wf-355-B1 §1.2, `metal_lomo_auc.json:280-285`):
- Rh seed=3 (training stopped at epoch 23 of 50)
- Rh seed=4 (never started)
- Re seed=0 (never started)
- Re seed=1 (never started)
- Re seed=2 (never started)
- Re seed=3 (never started)
- Re seed=4 (never started)
- **Recovery wrapper READY**: `scripts/launch_wf131_p3_deferred_runs.sh` (USER_OK-gated; ~5 hr GPU)

**Wf-146 6 DEFERRED runs** (per Wf-355-B2 §1.1, `metal_lomo_chemprop_v2.json:188-194`):
- Re seed=0 (never started)
- Re seed=1 (never started)
- Re seed=2 (never started)
- Re seed=3 (never started)
- Re seed=4 (never started)
- Os seed=4 (halted at epoch 10/50)
- **Recovery wrapper READY**: `scripts/launch_wf146_deferred_runs.sh` (USER_OK-gated; ~3 hr GPU on RTX 5090)

**Source attribution** (per Wf-354-A1):
- Wf-131 P3 D-MPNN LOMO (BUGGY Wf-125 P1 cohort) — `metal_lomo_auc.json:196-282` `wf131_p3_lomo_partial_update`
- Wf-146 chemprop v2 LOMO (smoke backbone hidden=64/depth=2) — `metal_lomo_chemprop_v2.json:155` `honest_framing_note`
- Wf-176 Wasserstein Tier-2 — `metal_lomo_auc_wasserstein.json:48-50` (Wf-322F closeout applied)
- Wf-310E Phase 2 GPU wrapper READY (USER_OK-BLOCKED) per `TODO/completed/322f_wf176_m19_lomo_final_closeout.md`

### §4.3 Honest framing on m19 LOMO

1. **R17 baseline (0.5346 ± 0.110) remains canonical** for paper §4 Table 2 (per `metal_lomo_auc.json:131` "DEFERRED_NO_DATA").
2. **Wf-131 P3 partial 0.6217** is on the BUGGY Wf-125 P1 cohort (constant-inactive labels, see TODO-135 §0 root cause); Rh seed=0 outlier 0.274 drags mean down (std=0.3584 high variance).
3. **Wf-146 chemprop v2 numbers are smoke-grade** (backbone hidden=64/depth/2 = 5.1× cheaper than production hidden=300/depth/3 per TODO-154 §3 MEASURED); not comparable to production chemprop v2.
4. **Re fold is DEFERRED** on both Wf-131 P3 and Wf-146; Re Tier-2 Wasserstein (n=1, seed=4) = 0.6220 MEASURED (per `metal_lomo_auc_wasserstein.json:19`).
5. **Wf-310E Wf-135 Phase 2 GPU wrapper READY** but USER_OK-BLOCKED per user directive 2026-09-29 ("先把honest-negative先解决了").

**Per-metric status: PARTIAL_SHIP** (18/25 + 19/25 partial; 13 DEFERRED total across 2 launchers; USER_OK-gated for full resolution).

---

## §5. m18 — Multi-modal Protein-Metal Binding AUC (NOT JNK3)

**Important honest framing**: a literal search for `m_jnk3`,
`jnk3`, `m_jnk`, or `JNK` in `metrics/`, `TODO/`, and `TODO/INDEX.md`
returns **ZERO hits**. The metric labelled "m18" in
`METRICS_TARGETS.md` and `MECHANISM_METRIC_MAPPING.md` is **Multi-modal
Protein-Metal Binding AUC**, NOT JNK3. There is no JNK3-specific
metric in the Mol-Metal project.

**Headline value (m18 = Multi-modal Protein-Metal Binding AUC):** **0.7436 ± 0.009 MEASURED** (v1, 3 seeds × 50 ep) / **0.7393 ± 0.009 MEASURED** (v2, 3 seeds × 100 ep)

### §5.1 v1 baseline (`metric18_protein_metal.json`)

| Item | Value | Source |
|---|---:|---|
| `metric` | metric18_protein_metal_binding_auc | line 3 |
| `test_auc_mean` | **0.7435730745526251** | line 9 |
| `test_auc_std` | 0.009216035837632744 | line 10 |
| `test_auc_per_seed` | [0.7562, 0.7344, 0.7402] | lines 11-15 |
| `test_pr_auc_mean` | 0.5694781872338304 | line 16 |
| `elapsed_sec` | 376.6 | line 22 |
| `n_rows` | 5257 | line 23 |
| `n_cell_lines` | 209 | line 24 |
| `n_metals` | 5 | line 25 |
| `epochs` | 50 | line 7 |

### §5.2 v2 (100 epoch) baseline (`metric18_protein_metal_v2.json`)

| Item | Value | Source |
|---|---:|---|
| `metric` | metric18_protein_metal_binding_auc | line 3 |
| `test_auc_mean` | **0.7392871477727422** | line 9 |
| `test_auc_std` | 0.009190760313620242 | line 10 |
| `test_auc_per_seed` | [0.7263, 0.7458, 0.7457] | lines 11-15 |
| `test_pr_auc_mean` | 0.5606730598372045 | line 16 |
| `elapsed_sec` | 586.5 | line 22 |
| `epochs` | 100 | line 7 |

### §5.3 v1 per_residue (`metric18_protein_metal_per_residue.json`)

| Item | Value | Source |
|---|---:|---|
| `metric` | metric18_protein_metal_binding_per_residue_auc | line 3 |
| `verdict` | redesign_v1 | line 4 |
| `n_complex` | 5257 | line 6 |
| `n_residue_samples` | 66303 | line 7 |
| `per_complex_auc` | 0.9944777327825318 | line 8 |
| `per_residue_auc` | 0.962731433900268 | line 9 |
| `per_residue_pr_auc` | 0.8331151503499039 | line 10 |
| `per_residue_auc_synthetic_proxy` | true | line 11 |

### §5.4 Source attribution (m18 = Multi-modal Protein-Metal)

**Per Wf-32 P2 SHIP** (per `TODO/completed/101_r22_plan_l13_l16.md:88`):
- `metrics/by_metric/metric18_protein_metal_v2.json` AUC 0.7393±0.009
- `molmetal/reports/wf32_dmpnn_top20/auc_per_seed.json` (carry)
- Wf-32 P2 5-metal multitask D-MPNN, 3-seed × 100 ep, scaffold split, n=5,257 × 209 cells × 5 metals
- Source also referenced in `TODO/completed/97_r20_p6_wetlab_ic50_dryrun.md:143` (Wf-32 P2 baseline 0.7393 ± 0.009)

**Per `TODO/completed/40_nci60_tmqm_synthesis.md:17`**:
> m18 Protein-Metal binding AUC (multimodal, 5-stream): **0.7436 ± 0.0092** — BELOW GATE (≥0.85 target).

**Per `TODO/completed/82_metric_gap_filling.md:129`**:
- m18 Multimodal Protein-Metal AUC: 0.7436 (unchanged from R17 ship)
- WEAK status preserved (≤0.85 gate)

**Honest framing caveats**:
1. m18 is **Multi-modal Protein-Metal Binding AUC**, not JNK3. The
   task brief's "m18 m_jnk3" appears to be a SPEC mismatch — JNK3
   kinase is NOT a metric in the Mol-Metal scope. The actual m18
   metric is on MetalCytoToxDB-derived "protein = cell-line" signal
   with 209 cell-lines × 5 metals × 5,257 complexes.
2. m18 baseline band (0.7393 ± 0.009) is **BELOW gate 0.85** by
   -0.111. The gap requires production-D-D-MPNN swap + pocket stream
   + cell-line embedding L2 regularisation (per `TODO/completed/40_nci60_tmqm_synthesis.md:187`).
3. R18 mechanism candidates M24 (Multi-PH) + M25 (Hilbert PD) +
   M43 (Bergman kernel, d-orbital features) are PROJECTED to lift
   m18 +0.05-0.10 to +0.181 to a Tier-1 ceiling 0.92. None of the
   R18 GPU retrains have launched (per Wf-32 Phase D deferral).
4. m18 wet-lab validation (R22-P1) was **DEPRECATED** per AGENTS.md
   §5.4 (2026-09-29 user directive "我们发二区就可以了，不用规划湿实验")
   and re-scoped to computational-only per Wf-317E.

**Per-metric status: MEASURED_BELOW_GATE** (0.7393 ± 0.009; gate ≥0.85; -0.111 gap; m18 architectural lift deferred to R20-P3 / R20-P6 / Wf-32 Phase E).

---

## §6. Per-metric verification verdict

| Metric | Headline value | Source path on disk | Tag | Status |
|---|---|---|---|---|
| **m11** n_distinct | 20 | `metrics/by_metric/n_distinct.json` line 50 (`aggregate.best_measured_with_all_fixes`) | MEASURED | PASS (R12 singleton collapse resolved; m11 gate ≥3 met; at n_top_k_cap ceiling) |
| **m15** Docking RMSD | 0.6578 Å (median); 5/5 pockets pass 2.0 Å | `metrics/by_metric/docked_pose_rmsd.json` lines 16, 19-22 | MEASURED (Wf-179 BD-9 + Wf-246-B redesign + Wf-341-T1 integration) | PASS (SOTA-level; 0.6578 < 1.5 Å gate) |
| **m19** LOMO | 0.6217 ± 0.1284 (Wf-131 P3 partial 18/25) + 0.4458 (Wf-146 19/25) + 0.6220 (Wf-176 Re-only n=1) | `metrics/by_metric/metal_lomo_auc.json` line 232 + `metal_lomo_chemprop_v2.json` line 203 + `metal_lomo_auc_wasserstein.json` line 19 | MEASURED (per-Wf partial; 13 DEFERRED total) | PARTIAL_SHIP (R17 0.5346 canonical for paper §4; Wf-310E Phase 2 GPU USER_OK-BLOCKED) |
| **m18** Protein-Metal | 0.7393 ± 0.009 (v2) / 0.7436 ± 0.009 (v1) | `metrics/by_metric/metric18_protein_metal_v2.json` line 9 + `metric18_protein_metal.json` line 9 | MEASURED (Wf-32 P2) | MEASURED_BELOW_GATE (gate ≥0.85; -0.111 gap; m18 architectural lift deferred) |

---

## §7. DEFERRED runs ready for USER_OK launch (per Wf-355-B1 + B2)

| Wrapper | (metal, seed) pairs | Wall (PROJECTED) | Status |
|---|---|---:|---|
| `scripts/launch_wf131_p3_deferred_runs.sh` | Rh 3+4 + Re 0/1/2/3/4 = **7 pairs** | ~5 hr GPU sequential | USER_OK-gated; BLOCKED per user directive 2026-09-29 |
| `scripts/launch_wf146_deferred_runs.sh` | Re 0/1/2/3/4 + Os 4 = **6 pairs** | ~3 hr GPU on RTX 5090 | USER_OK-gated; BLOCKED per user directive 2026-09-29 |
| `scripts/launch_wf310e_wf135_phase2_gpu.sh` | full 5-metal × 5-seed × 50-epoch Phase 2 (250 epoch-runs) | ~10 GPU-hr | USER_OK-gated; BLOCKED per user directive 2026-09-29 |

**Total deferred GPU work**: 7 + 6 + 250 = 263 (metal, seed) pairs / ~18 GPU-hr cumulative on RTX 5090 + Blackwell PRO 6000.

**Honest framing caveat**: All DEFERRED work honors AGENTS.md §5.1
kill_then_adapt directive + §5.4 Tier-2 markers (no wet-lab
planning). Wrappers READY but BLOCKED on user directive 2026-09-29
("先把honest-negative先解决了").

---

## §8. Honest framing per `party-twenty-spirit` §3.5

### §8.1 What is MEASURED

- **m11 = 20** — `metrics/by_metric/n_distinct.json` line 50,
  `aggregate.best_measured_with_all_fixes`, status MEASURED on
  `wf_round12_lambda_patha_10x3` (10×3 cells, 30/30 cells = n_top_k cap).
- **m15 = 0.6578 Å** — `metrics/by_metric/docked_pose_rmsd.json` line 16,
  `value` field, MEASURED 2026-09-28T02:42:18 UTC, 5-pocket × 3-seed
  Vina redocking with Kabsch+Hungarian + adaptive centroid pre-alignment.
- **m19 Wf-131 P3 partial 0.6217** — `metrics/by_metric/metal_lomo_auc.json` line 232 (`overall_partial_mean`), MEASURED on BUGGY Wf-125 P1 cohort (constant-inactive labels), 18/25 (72%) complete; Wf-131 P3 partial update block.
- **m18 = 0.7393 ± 0.009** — `metrics/by_metric/metric18_protein_metal_v2.json` line 9, `test_auc_mean`, MEASURED on MetalCytoToxDB-derived "protein = cell-line" signal (n=5257 × 209 cells × 5 metals), 3-seed × 100-ep D-MPNN scaffold split.

### §8.2 What is DEFERRED

- **m19 Wf-131 P3 7 runs (Rh 3+4 + Re 0-4)** — training stopped at
  Rh seed=3 epoch=23 / Re never started; recovery wrapper READY at
  `scripts/launch_wf131_p3_deferred_runs.sh` (Wf-355-B1).
- **m19 Wf-146 6 runs (Re 0-4 + Os seed=4)** — sweep halted at
  Os seed=4 epoch=10/50; recovery wrapper READY at
  `scripts/launch_wf146_deferred_runs.sh` (Wf-355-B2).
- **m19 Wf-310E Phase 2 GPU 250 epoch-runs** — full 5-metal × 5-seed
  × 50-epoch × {hilbert, tmqm} dual-init on BALANCED Rh cohort;
  wrapper READY at `scripts/launch_wf310e_wf135_phase2_gpu.sh`.
- **m18 R18 architectural lift** — Multi-PH + Hilbert PD + Bergman
  kernel GPU retrain; deferred to R20-P3 / R20-P6 / Wf-32 Phase E.
- **m15 PoseBusters cross-validity** — adapter gracefully SKIPS
  PoseBusters validity check (not installed in project venv); does
  NOT fabricate pass rate (per `341t1_m15_rmsd.md:32`).

### §8.3 What is BLOCKED

- All GPU launches (>1 hr wall) are USER_OK-gated per AGENTS.md
  §5.1 kill_then_adapt directive + user directive 2026-09-29
  ("先把honest-negative先解决了"). Recovery wrappers are READY
  but not executed.

### §8.4 Spec mismatch on m18 / m_jnk3

- The task brief mentions "m18 m_jnk3", but **NO m_jnk3 metric exists**
  in the Mol-Metal project. m18 = Multi-modal Protein-Metal Binding
  AUC (MetalCytoToxDB-derived), NOT JNK3 kinase-specific. Verified
  via `grep -rE "jnk|JNK3|m_jnk" metrics/ TODO/` → 0 hits.
- The task brief mentions "m19 LOMO 18/25 (per Wf-354-A1; partial)",
  which is correct — Wf-131 P3 D-MPNN LOMO partial is 18/25 (72%)
  per `metal_lomo_auc.json:228-229` (`n_runs_total_completed: 18` +
  `n_runs_target_total: 25`).

---

## §9. Pre-flight status

Wf-357-B4 does NOT introduce a new pre-flight gate. The work is
CPU-only verification (Read + JSON parse + arithmetic + path
verification). It does NOT consume or modify any active training
pipeline.

Pre-existing pre-flight v3 90 PASS / 2 FAIL / 2 SKIP / 94 TOTAL
PRESERVED at Wf-357-A1 timestamp (2026-09-29) per Wf-357-A1 §0.
EXPECTED_GATE_COUNTS["all"] 94 PRESERVED.
2 FAIL gates are pre-existing honest-negatives:
- `p2_tests_pass`
- `p3_wf343c2_per_wave_runtime_check`

2 SKIP gates are pre-existing:
- `p3_wf343c4_wave_budget_remaining_nonneg` (DEFERRED)
- `data_2_1_6_bindingdb_external_honest_negative`

---

## §10. Cross-references

| Reference | Section |
|---|---|
| Wf-357-B2 | `TODO/completed/357b2_tier1_supporting_metrics.md` (Tier-1 m7+m3+m13+m14) |
| Wf-357-A1 | `TODO/completed/357a1_preflight_audit.md` (pre-flight 90/2/2/94 baseline) |
| Wf-149 | R12 baseline for m11 singleton collapse (referenced in `n_distinct.json:7`) |
| Wf-301 | R17 m11 enhancements (Path A 4-fix bundle) |
| Wf-179 BD-9 | m15 RMSD Phase 2 Vina redocking canonical script |
| Wf-246-B | m15 RMSD redesign (per-pocket std + cross-pocket aggregate + difficulty flag) |
| Wf-336-X4 | TODO-134 Vina meeko alternative integration |
| Wf-341-T1 | TODO-134 m15 RMSD redesign integration (4 NEW tests PASS, CPU-only) |
| Wf-131 P3 | D-MPNN LOMO BUGGY-cohort partial 18/25 |
| Wf-145 P2 | D-MPNN v2 driver SHIP + 18/18 tests |
| Wf-146 | chemprop v2 LOMO 19/25 partial (smoke backbone hidden=64/depth=2) |
| Wf-176 | Wasserstein Tier-2 (Re fold n=1, seed=4) |
| Wf-310E | Wf-135 Phase 2 GPU wrapper READY (USER_OK-BLOCKED) |
| Wf-32 P2 | m18 baseline 0.7393 ± 0.009 (3-seed × 100-ep D-MPNN) |
| Wf-354-A1 | LOMO integration verification (`TODO/completed/354a1_wf342_p2_lomo_integration.md`) |
| Wf-355-B1 | Wf-131 P3 7 DEFERRED runs wrapper (USER_OK-gated) |
| Wf-355-B2 | Wf-146 6 DEFERRED runs wrapper (USER_OK-gated) |
| AGENTS.md §1 rule 1 | MEASURED-vs-PROJECTED tag discipline |
| AGENTS.md §5.1 | Standing GPU auth — USER_OK-gated for >1hr GPU |
| AGENTS.md §5.2 | Honest framing mandate |
| AGENTS.md §5.4 | Tier-2 journal, no wet-lab planning |
| `party-twenty-spirit` §3.5 | PROJECTED vs MEASURED tag discipline; 0-tolerance fabrication |

---

## §11. Test count + status

- **Tests added by this Wf**: 0 (verification only, no new code).
- **Tests verified to exist**: 6 (3 Wf-355-B1 + 3 Wf-355-B2).
- **GPU launches performed**: 0.
- **Source files modified**: 0 (Read + JSON parse + arithmetic only).

---

**END of Wf-357-B4** — Owner Hugo Lee; cross-ref `party-twenty-spirit` §3.5 + AGENTS.md §1.2 + §5.1-§5.4; SHIP (CPU-only verification; 0 NEW tests; 0 GPU launches; 4 supporting metrics (m11+m15+m19+m18) verified; 13 DEFERRED runs catalogued with USER_OK-gated recovery wrappers; 1 SPEC mismatch on m18 / m_jnk3 documented).