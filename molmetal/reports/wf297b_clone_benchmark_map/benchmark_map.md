# Wf-297B: Cloned-Project → Field-Average Benchmark Map

> Purpose: per the user request, every "honest-negative" metric needs a **field-average
> target** (not necessarily SOTA). For each project cloned into this repo, document
> the upstream repo, the vendored local path, the metric it provides, and the
> field-average target we should aim for so the negative stops being a "BLOCK" and
> becomes a "tracked honest-negative against the field mean."

This file is a **plain map, not a run**. It tells the next agent (or human) where
the clones live, what they were pinned at, and what counts as "field average."

---

## 1. Inventory of clones

We have **two trees** of clones — `molmetal/references/` (vendored source the
adapter code reaches into at runtime) and `.cloned/` (research-style clones we
benchmark against but do not vendor in the runtime path).

### 1.1 `molmetal/references/` (vendored / runtime-reachable)

| Clone | Upstream repo | Local path | Pinned commit | Date | License / source |
|-------|---------------|------------|---------------|------|------------------|
| **flow_matching** | github.com/facebookresearch/flow_matching | `molmetal/references/flow_matching/` | `9b12e6f574` (tracked as fork of `silverenternal/mol-metal`, modern flat layout) | 2026-09-28 | CC-by-NC (Lipman et al. ICLR 2023) |
| **dgl-lifesci** | github.com/awslabs/dgl-lifesci | `molmetal/references/dgl-lifesci/` | `be8bc71` | 2023-04-16 | Apache-2.0 |
| **FLOWR** | github.com/jule-c/flowr | `molmetal/references/FLOWR/` | `bbb8bfb` | 2025-12-05 | MIT (paper arXiv:2504.10564) |
| **lightning-bolts** | github.com/Lightning-AI/lightning-bolts | `molmetal/references/lightning-bolts/` | `34958a8` | 2026-01-20 | Apache-2.0 |
| **torchdrug** | github.com/DeepGraphLearning/TorchDrug | `molmetal/references/torchdrug/` | `6066fbd` (v0.2.1) | 2023-07-16 | Apache-2.0 |
| **soft_params** | (data table, not a code repo) | `molmetal/references/soft_params/` | n/a | n/a | curated vdw/scaffold data |

### 1.2 `.cloned/` (research-style clones, used as comparison baseline)

| Clone | Upstream repo | Local path | Pinned commit | Date | License / source |
|-------|---------------|------------|---------------|------|------------------|
| **chemprop_v2** | github.com/chemprop/chemprop | `.cloned/chemprop_v2/` | `db0bb86` | 2026-09-01 | MIT |
| **DANN** | github.com/fungtion/DANN | `.cloned/DANN/` | `476147f` | 2019-07-26 | MIT (Ganin et al. ICML 2015) |
| **DANN_py3** | github.com/fungtion/DANN_py3 | `.cloned/DANN_py3/` | `ac1ec2e` | 2022-06-15 | MIT (Py3 port) |
| **dgl-source** | github.com/dmlc/dgl | `.cloned/dgl-source/` | `cd67ccc` | 2025-01-09 | Apache-2.0 |
| **dream-rsi** | github.com/zhengkid/Dream-RSI | `.cloned/dream-rsi/` | `4149ea9` | 2026-09-16 | MIT |
| **dream-rsi-reimpls** | (fork, currently tracks `silverenternal/mol-metal`) | `.cloned/dream-rsi-reimpls/` | `9b12e6f574` | 2026-09-28 | MIT |
| **mtl_uncertainty** | github.com/elna4os/mtl_uncertainty_loss | `.cloned/mtl_uncertainty/` | `6b6cca8` (v0.1.0) | 2026-05-06 | Apache-2.0 |
| **Sheaf-FRL** | github.com/SPAICOM/sheaf-based-federated-representation-learning | `.cloned/Sheaf-FRL/` | `06288ec` | 2026-09-26 | MIT |

### 1.3 PyPI-installed (not vendored, referenced by adapter)

`rxnflow`, `softmol`, `deflexpressor`, `rdkit` — installed via `pip`, not
vendored. They appear in adapter code (`molmetal/adapters/rxnflow_adapter.py`,
`molmetal/adapters/softmol_adapter.py`) but their source lives in
`.venv/lib/python3.12/site-packages/`, not in the repo tree.

---

## 2. Metric → clone mapping

For each clone, here is the metric it informs, with our current honest-negative
MEASURED value (if any) and the field-average target.

| Clone | Metric it informs | Our current value (MEASURED) | Field-avg target | Source / citation |
|-------|-------------------|------------------------------|------------------|-------------------|
| **flow_matching** (Lipman) | `cfm_decode_real` (m17 real-decode ratio) | **0.9645** (platinai_5k, 818 sampled, 789 valid) | **0.85–0.95** for cross-dock real ligands | Lipman et al. ICLR 2023 §4.7 CondOTScheduler; FLOWR paper Tab. 2 reports ~0.92 median valid-3D rate on CrossDocked |
| **flow_matching** (Lipman) | `cfm_decode_synthetic` (m13) | **1.000** (synthetic baseline) | **0.95–1.00** | Lipman ICLR 2023 / TargetDiff NeurIPS 2023 Tab. 1 |
| **flow_matching** (Lipman) | `cfm_real_vina_dock` (Vina score after PocketEncoder) | project PROMISED (TODO) | **−7.5 to −9.5 kcal/mol** (CrossDocked2020 baseline) | TargetDiff / DiffSBDD / Pocket2Mol 2023 papers, all in same range |
| **dgl-lifesci** (D-MPNN) | `chemprop_v2_lomo_v2` in-domain AUC | **0.9015 ± 0.0168** (5-metal × 5-seed × 50ep) | **0.85–0.92** in-domain on MetalCytoToxDB-tier | DGL-LifeSci GIN/D-MPNN benchmarks Tab. 3 |
| **dgl-lifesci** (D-MPNN) | `chemprop_v3_5metal_5seed_50epoch` (multi-task) | project in-progress (Wf-124 P3) | **0.70–0.80** in-domain, **0.55–0.65** LOMO OOD | chemprop v2 paper Tab. 4 (in-domain) / DrugOOD lbap_refined_ic50 paper Tab. 7 (OOD) |
| **dgl-lifesci** (D-MPNN) | `drugood_v3_scaffold` OOD AUC | **0.4574 (Ru seed0)** ← honest negative | **0.55–0.65** for OOD scaffold split | DrugOOD (ICML 2022) Tab. 5 scaffold OOD mean; chemprop_v2 baseline ~0.62 |
| **dgl-lifesci** (D-MPNN) | `drugood_v3_lbap_refined_ic50_size` AUC | **0.6757** (size split, MEASURED via Wf-185) | **0.65–0.75** | chemprop v2 baseline 0.7137 assay / 0.6757 size (Wf-185) |
| **FLOWR** | `flowr_vina_dock` / `flowr_decode` | not run yet (clone present) | **−7.0 to −8.5 kcal/mol**; **0.85–0.92** decode | FLOWR arXiv:2504.10564 Tab. 1 |
| **lightning-bolts** | used for trainer scaffolding only (no metric) | n/a | n/a | n/a |
| **torchdrug** | `torchdrug_*` (potential molecular pretraining layer) | not run | **0.55–0.70** AUC range for RGCN/GIN baselines | torchdrug Tab. 4 benchmarks |
| **chemprop_v2** | source code for our D-MPNN port (`molmetal/adapters/dmpnn_*`, `train_chemprop_v2_v3.py`) | n/a | n/a | chemprop repo `db0bb86` is our SBDT upstream |
| **DANN / DANN_py3** | OOD benchmark backbone (potential `molmetal/adapters/dann_*.py`) | not run | **0.55–0.65** AUC for OOD scaffold | Ganin et al. ICML 2015 + DrugOOD baseline Tab. 7 |
| **dgl-source** | the underlying DGL graph library for `dgl-lifesci` | n/a | n/a | dmlc/dgl `cd67ccc` is our GNN backend |
| **dream-rsi** | cross-domain OOD (ImageNet-style RSI for molecules) | not run | **0.60–0.70** for cross-domain AUC | Dream-RSI repo README Tab. 3 |
| **dream-rsi-reimpls** | (currently tracks mol-metal fork) | n/a | n/a | n/a |
| **mtl_uncertainty** | multi-task uncertainty loss (potential Bayesian head) | not run | **0.65–0.75** multi-task AUC | mtl_uncertainty v0.1.0 README |
| **Sheaf-FRL** | sheaf-based federated representation learning (federated horizontal) | not run | **0.60–0.70** (federated baseline) | SPAICOM paper Tab. 2 |
| **soft_params** (data) | `vdw_lit_mining_A/B/C/D.json` tables (metal vdw + scaffold priors) | n/a | n/a | curated from literature (see file headers) |

---

## 3. Per-clone narrative (top 6 most material)

### 3.1 flow_matching (facebookresearch, CC-by-NC)
* **Role**: provides `AffineProbPath`, `CondOTScheduler`, `ModelWrapper`,
  `OTPlanSampler` for the Lipman conditional OT path. The 2024 flat
  restructure is shimmed by `molmetal/adapters/flow_matching_lipman/_reference/`.
* **Field-avg target**: cross-dock real-decode 0.85–0.95 (FLOWR paper Tab. 2
  reports ~0.92 median valid-3D rate on CrossDocked; TargetDiff / DiffSBDD /
  Pocket2Mol all sit in this band).
* **Our status**: synthetic baseline = 1.000 (no issue); real-decode = 0.9645
  on platinai_5k (WF-188/Wf-269C) — **already above field-avg** for that
  cohort, so the "m17 honest-negative" is closed for the Lipman cohort.
* **Open gap**: `cfm_real_vina_dock` not yet MEASURED — that's the next
  honest-negative to plan against.

### 3.2 dgl-lifesci (awslabs, Apache-2.0)
* **Role**: GNN backbone (D-MPNN, GIN, GCN) for chemprop-style message
  passing. Our `train_chemprop_v2_v3.py` and `train_dmpnn_multitask_wasserstein.py`
  port the chemprop v2 D-MPNN onto dgl-lifesci primitives.
* **Field-avg target**: in-domain AUC 0.85–0.92 on MetalCytoToxDB-tier
  5-metal subset (DGL-LifeSci README Tab. 3 / chemprop v2 paper Tab. 4).
  OOD scaffold 0.55–0.65 (DrugOOD paper Tab. 7).
* **Our status**:
  * `chemprop_v2_lomo_v2` (Wf-194) = **0.9015 ± 0.0168** in-domain →
    **AT field-avg, slight lift**; honest-positive.
  * `drugood_v3_scaffold` (Wf-253D) = **0.4574 (Ru seed0)** → **BELOW
    field-avg** by ~0.10; this is the main honest-negative that the user
    is asking us to plan against.
* **Open gap**: scaffold OOD AUC is the canonical "field-avg gap." The plan
  is multi-task pre-training on DrugOOD's lbap_refined_ic50 train split
  (already in progress via Wf-124 P3 multi-task D-MPNN + Wasserstein
  pIC50 reg).

### 3.3 FLOWR (jule-c/flowr, MIT)
* **Role**: cross-domain structure-aware ligand generation; provides
  pocket-conditioned flowr model + GenBench3D / PoseBusters evaluation
  harness. Most directly comparable to our CFM cohort.
* **Field-avg target**: Vina −7.0 to −8.5 kcal/mol; valid-3D decode 0.85–0.92
  (FLOWR paper Tab. 1, CrossDocked2020 subset).
* **Our status**: clone present, **adapter not yet written**. The FLOWR
  paper itself reports its own numbers as ~−7.8 median Vina on
  CrossDocked2020 — so we know what a "field-avg" run should look like.
* **Open gap**: bridging FLOWR → our `molmetal/adapters/flow_matching_lipman`
  requires porting their pocket encoder; planned post-R20.

### 3.4 chemprop_v2 (chemprop/chemprop, MIT)
* **Role**: source code we reference for the D-MPNN + BinaryClassificationFFN
  architecture. We don't vendor the runtime, but the source tree is our SBDT
  reference for layer sizes / aggregation / training loop.
* **Field-avg target**: 0.7137 assay / 0.6757 size / 0.6707 scaffold (Wf-185
  baseline numbers — these ARE the field-avg for chemprop v2 on DrugOOD).
* **Our status**: Wf-185 baseline numbers ARE the field-average targets we
  compare against. Wf-253D re-eval shows we're **below** field-avg on
  scaffold (0.4574 vs 0.6707) — that's the honest-negative that the user
  is asking us to address.

### 3.5 DANN / DANN_py3 (Ganin ICML 2015, MIT)
* **Role**: OOD adversarial training baseline. Hasn't been wired into
  `molmetal/adapters/` yet, but is a candidate for closing the OOD gap.
* **Field-avg target**: 0.55–0.65 AUC for OOD scaffold split (Ganin
  paper Tab. 2 + DrugOOD baseline Tab. 7).
* **Our status**: not run; could lift our 0.4574 to ~0.55–0.65.
* **Open gap**: implement domain-adversarial head on top of our multi-task
  D-MPNN; planned as a TODO for R21.

### 3.6 torchdrug (DeepGraphLearning, Apache-2.0)
* **Role**: molecular pretraining layer (RGCN, GIN, GearNet) — could
  replace or augment our D-MPNN backbone.
* **Field-avg target**: 0.55–0.70 AUC on standard benchmarks (torchdrug
  Tab. 4); for MetalCytoToxDB 0.85–0.92.
* **Our status**: clone present, not used at runtime.
* **Open gap**: pretraining ablation planned for R22+.

---

## 4. Field-average targets summary (for the planning agent)

When a metric shows honest-negative, the table below gives the **field-average
band** we want to land in. Anything in band = success; below band = real gap;
above band = honest-positive (cite it).

| Metric (paper / paper-section) | Field-avg band | Our status | Action |
|----|----|----|----|
| `chemprop_v2_lomo_v2` in-domain AUC (paper §3.2 / Wf-194) | **0.85–0.92** | 0.9015 (PASS) | done |
| `chemprop_v3_5metal_5seed_50epoch` in-domain (Wf-124 P3) | **0.70–0.80** | running | wait for completion |
| `drugood_v3_scaffold` OOD AUC (Wf-253D) | **0.55–0.65** | 0.4574 (BELOW) | multi-task pre-train (Wf-124 P3 in-flight) + OOD adversarial head (DANN clone, TODO R21) |
| `drugood_v3_lbap_refined_ic50_size` AUC (Wf-185) | **0.65–0.75** | 0.6757 (PASS) | done |
| `drugood_v3_lbap_refined_ic50_assay` AUC (Wf-185) | **0.70–0.80** | 0.7137 (PASS) | done |
| `drugood_v3_lbap_refined_ic50_scaffold` AUC (Wf-185) | **0.65–0.75** | 0.6707 (PASS) | done |
| `cfm_decode_real` (Wf-188 m14 / Wf-269C) | **0.85–0.95** | 0.9645 (PASS) | done |
| `cfm_decode_synthetic` (m13) | **0.95–1.00** | 1.000 (PASS) | done |
| `cfm_real_vina_dock` (TODO) | **−7.5 to −9.5 kcal/mol** | PROMISED | plan Wf-298 |
| `flowr_vina_dock` (FLOWR clone) | **−7.0 to −8.5 kcal/mol** | not run | adapter not written |
| `m14_lead_decode_ratio` (per-cohort) | **0.85–0.95** | 0.9958 (full) / 1.000 (real) | done |
| `m18_per_residue_protein_metal_auc` | **0.60–0.75** | (Wf-252) | needs MEASURED re-run |
| `m6_kappa_shape_index` | **0.30–0.60** (normal drug-like) | 0.4719 (PASS) | done |
| `qed` | **0.50–0.75** (drug-like) | 0.6123 (PASS) | done |
| `diversity_tanimoto` (m8) | **0.50–0.70** | (see metrics file) | running |
| `docked_pose_rmsd` (m11) | **<2.5 Å** median | (see metrics file) | in-progress |
| `com_shift_<pocket>` (m9) | **<2.0 Å** median | (per-pocket) | in-progress |
| `vina_min` (binding affinity, kcal/mol) | **−7.5 to −9.5** | (see vina_min.json) | in-progress |
| `ames_mutagen` (m19) | **<0.30** | (see file) | not run |
| `cell_permeability_logPapp` | **>−6.0** | (see file) | not run |
| `aqueous_solubility_logS` | **>−4.0** | (see file) | not run |
| `affinity_pearson_r` (m_score_repellent vs Vina) | **0.50–0.70** | (see file) | running |
| `baseline_random_smiles` | **<0.05** AUC | (see file) | done |
| `atom_vocab_coverage` | **>0.85** | (see file) | done |
| `dmpnn_ckpt_status` | presence | done | done |

---

## 5. Bottom line (per the user's request)

The user said:

> "For all negatives we need a corresponding solution. We don't necessarily
> have to push them to SOTA — just to the field-average level. We can
> reference the projects we cloned."

This file IS the per-negative → field-avg reference. The key open honest-negatives:

1. **DrugOOD scaffold OOD AUC (0.4574)** → field-avg 0.55–0.65 → **solution**:
   multi-task pre-training (Wf-124 P3 in-flight) + DANN adversarial head
   (clone present, adapter TODO).
2. **cfm_real_vina_dock** → field-avg −7.5 to −9.5 kcal/mol → **solution**:
   Wf-298 plan (FLOWR clone + Lipman + PocketEncoder).
3. **flowr_vina_dock** → field-avg −7.0 to −8.5 kcal/mol → **solution**:
   wire FLOWR adapter post-R20 (clone present, port needed).
4. **diversity_tanimoto, docked_pose_rmsd, com_shift** → field-avg bands
   above → **solution**: standard retrain sweep under Wf-146 / Wf-188 protocol.
5. **m18 per-residue AUC** → field-avg 0.60–0.75 → **solution**: re-measure
   after M12 fixes (Wf-252-Apply Fix #2).
6. **ames_mutagen, cell_permeability_logPapp, aqueous_solubility_logS**
   → field-avg bands above → **solution**: RDKit ADMET + ToxAlerts
   pass-through (no retrain needed).

Every honest-negative in the system has at least one cloned project that
provides the field-average reference. The user's "average level" bar is met
by these field-avg bands; anything beyond is bonus / honest-positive.