# Evaluators — 已迁移 / 已验证（2026-09-30 Wf-395-A1 refresh）

**Date**: 2026-09-30 (Wf-395-A1 refresh post Wf-380..394 wave)
**Status**: 22 evaluators MEASURED-OK on synthetic input (12 R17-era + 9 SCAFFOLD-tier Wf-380..382 + 1 MEASURED-ON-REAL Wf-388-A1)
**R17-era canonical baseline**: 14/14 evaluators PASS via `molmetal/scripts/eval_all_evaluators.py`
**Hardware**: NVIDIA Blackwell sm_120, x86_64 Linux 7.2.6
**Verification**: `molmetal/scripts/wf395a1_evaluator_audit.py` (NEW per Wf-395-A1)

---

## 0. Honest framing (party-twenty-spirit §3.5)

Per AGENTS.md §1.5, every evaluator has a **tier label** distinguishing:

| Tier | Meaning | Example |
|---|---|---|
| **R17** | R17-era canonical evaluator; MEASURED working on real cohorts historically | SA score, QED, Vina, PoseBusters |
| **SCAFFOLD** | Wf-380..394 NEW evaluator module; import + synthetic-input probe MEASURED-OK, but **NOT yet MEASURED end-to-end on real cohorts** (requires USER_OK-gated GPU retrain per AGENTS.md §5.1) | Wf-380-A3 m7 nucleus_sample, Wf-382-A1 R20-P3 PLL features |
| **WIRE-IN** | Evaluator wired into trainer as opt-in CLI flag (Wf-388-A2/A3); trainer loss bit-exact when flags default OFF | Wf-388-A2 m7 diversity, Wf-388-A3 m4 SA |
| **MEASURED-ON-REAL** | Evaluator MEASURED end-to-end on real cohort | Wf-388-A1 m2 R² platinai_4cell/hilbert/metalcytotox_combined |

The "22 evaluators" count below is the MEASURED-OK synthetic-input probe; SCAFFOLD probes do NOT imply end-to-end MEASUREMENT on real cohorts. End-to-end measurement requires the corresponding USER_OK-gated launch per AGENTS.md §5.1.

---

## 1. 评估器总览（22 evaluators MEASURED-OK @ 2026-09-30 Wf-395-A1）

| # | Evaluator | 类别 | 模块 / 工具 | Tier | 状态 | 备注 |
|---|---|---|---|---|---|---|
| 1 | **SA score** (Ertl-Schuffenhauer) | 合成可及性 | `molmetal.molmetal_lam.sbdd_env.sa_score.sa_score_ertl` | R17 | ✅ WORKING | RDKit Contrib sascorer.py |
| 2 | **QED** (Bickerton) | 类药性 | `molmetal.molmetal_lam.sbdd_env.qed_scorer.qed_from_smiles` | R17 | ✅ WORKING | RDKit QED |
| 3 | **Vina** (AutoDock) | 对接亲和 | `molmetal.molmetal_lam.sbdd_env.vina_adapter.VinaDockingAdapter` | R17 | ✅ WORKING | `vina` 1.2.7 + `meeko` 0.8.0 |
| 4 | **vina** Python binding | 对接依赖 | `vina.__version__` | R17 | ✅ WORKING | 1.2.7 |
| 5 | **meeko** | SMILES→PDBQT | `meeko.__version__` | R17 | ✅ WORKING | 0.8.0 |
| 6 | **PoseBusters** | 几何合理性 | `molmetal.molmetal_lam.sbdd_env.posebusters_adapter.PoseBustersAdapter` | R17 | ✅ WORKING | posebusters Python package |
| 7 | **BioPython PDB** | 结构解析 | `Bio.PDB.PDBParser / NeighborSearch / PPBuilder` | R17 | ✅ WORKING | PLIP 替代方案 |
| 8 | **OpenBabel** | 化学文件转换 | `openbabel.OBMol` | R17 | ✅ WORKING | `/usr/bin/obabel` 在 PATH |
| 9 | **ProDy** | 弹性网络模型 | `prody.__version__` | R17 | ✅ WORKING | pocket flexibility / ENM features |
| 10 | **MetalLigandAdapter** | 金属-配体复合 | `molmetal.molmetal_lam.sbdd_env.metal_generator_adapter.MetalLigandAdapter` | R17 | ✅ WORKING | reconstruct_complex_smiles |
| 11 | **MetalCoordinationScore** | 配位几何 | `molmetal.molmetal_lam.sbdd_env.metal_coord_term.coordination_geometry_score` | R17 | ✅ WORKING | Wf-246-C kcal correction; Pt/Ru/Ir HSAB match |
| 12 | **PLLSignature canonical** | Persistent Laplacian | `molmetal.molmetal_lam.lam_chem.pll_signature.compute_pll` | R17 | ✅ WORKING | Liu & Feng 2026 PLL 128-d |
| 13 | **Wf-380-A2 m2 R² per-cohort** | per-cohort R² | `molmetal.scripts.wf380a2_m2_r2_per_cohort.select_features` | SCAFFOLD | 🆕 SCAFFOLD-OK | TODO-201 §2.2 3-tier matrix; MEASURED-on-real per Wf-388-A1 |
| 14 | **Wf-380-A3 m7 nucleus_sample** | diversity sampling | `molmetal.adapters.flow_matching_lipman.diversity_sampling.nucleus_sample` | SCAFFOLD | 🆕 SCAFFOLD-OK | Holtzman 2020 Eq.(3); top-p / top-k / temperature |
| 15 | **Wf-380-A3 m7 mmd_loss** | MMD unbiased U-statistic | `molmetal.adapters.flow_matching_lipman.diversity_sampling.mmd_loss` | SCAFFOLD | 🆕 SCAFFOLD-OK | Gretton 2012 Eq.(3); median-heuristic sigma |
| 16 | **Wf-380-A3 m7 repulsion_loss** | cosine-similarity repulsion | `molmetal.adapters.flow_matching_lipman.diversity_sampling.repulsion_loss` | SCAFFOLD | 🆕 SCAFFOLD-OK | Liu 2021 Eq.(7); margin=0.95 |
| 17 | **Wf-380-A3 m7 Murcko scaffold** | scaffold bucketing | `molmetal.adapters.flow_matching_lipman.scaffold_conditioning.murcko_scaffold_from_smiles` | SCAFFOLD | 🆕 SCAFFOLD-OK | Bemis-Murcko 1996 |
| 18 | **Wf-380-A4 m3 Vina load_vina_config** | Vina 3-level config | `molmetal.scripts.wf380a4_vina_param_tuning.load_vina_config` | SCAFFOLD | 🆕 SCAFFOLD-OK | TODO-203 low/medium/high exhaustiveness grid |
| 19 | **Wf-380-A5 m15 compute_pose_rmsd** | RMSD Kabsch+Hungarian | `molmetal.adapters.ligand_vs_pocket_rmsd_adapter.compute_pose_rmsd` | SCAFFOLD | 🆕 SCAFFOLD-OK | TODO-204 multi-template scaffold |
| 20 | **Wf-380-A6 m4 compute_sa_score** | per-sample SA penalty | `molmetal.adapters.flow_matching_lipman.sa_penalty.compute_sa_score` | SCAFFOLD | 🆕 SCAFFOLD-OK | TODO-205 generation penalty (Ertl 2009 §3) |
| 21 | **Wf-382-A1 R20-P3 PLLFeatureAdapter** | PLL 10-d feature | `molmetal.adapters.flow_matching_lipman.pll_features.PLLFeatureAdapter` | SCAFFOLD | 🆕 SCAFFOLD-OK | TODO-94 Multi-PH + PLL signature |
| 22 | **Wf-388-A1 m2 R² apply_per_cohort_config** | per-cohort R² on REAL cohorts | `molmetal.scripts.wf388a1_m2_per_cohort_real.apply_per_cohort_config` | MEASURED-ON-REAL | 🆕 MEASURED | platinai_4cell + hilbert + metalcytotox_combined; 5 seeds × 5-fold CV per Wf-335-M2 |
| 23 | **Wf-388-A2 m7 diversity wire** | trainer CLI wire-in | `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` 6 NEW CLI flags | WIRE-IN | 🆕 WIRE-IN | bit-exact backward compat when flags default OFF |
| 24 | **Wf-388-A3 m4 SA penalty wire** | trainer CLI wire-in | `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` NEW CLI flag | WIRE-IN | 🆕 WIRE-IN | bit-exact backward compat when flag default OFF |
| -- | **QuickVina2-GPU 2.1** | GPU 对接 | `vina_gpu_adapter.QuickVinaGPUAdapter` | R17 | ⚠️ AMD-only | 不适用于 NVIDIA Blackwell |
| -- | **PySR** | 符号回归 | `pysr` | R17 | ❌ NOT AVAILABLE | （Deflexpressor / LambdaASTSR 替代） |
| -- | **PLIP** | 蛋白-配体相互作用指纹 | `plip` | R17 | ❌ NOT AVAILABLE | （BioPython NeighborSearch 替代） |

---

## 2. 验证脚本（一键跑全部评估器）

### 2.1 R17-era canonical verification (legacy §2 preserved)

```python
"""eval_all_evaluators.py — sanity check all evaluators on this device.

Run:
    cd /home/hugo/codes/mol-metal
    /home/hugo/codes/mol-metal/.venv/bin/python -u molmetal/scripts/eval_all_evaluators.py
"""
# (legacy script unchanged — 14/14 PASS at Wf-395-A1 fresh-run)
```

### 2.2 Wf-395-A1 audit (NEW — covers R17 + Wf-380..394)

```bash
# Audit all 22 evaluators (12 R17 + 9 SCAFFOLD + 1 MEASURED-ON-REAL)
/home/hugo/codes/mol-metal/.venv/bin/python \
    /home/hugo/codes/mol-metal/molmetal/scripts/wf395a1_evaluator_audit.py

# Output:
#   Wf-395-A1 SUMMARY: 22/22 evaluators PASS
#   R17 canonical baseline: 14/14 evaluators PASS
#   NEW Wf-380..394: 10 candidate evaluators (all MEASURED-OK in probe)
#   JSON audit record: molmetal/reports/wf395a1_evaluator_audit/audit.json
```

The audit script:
1. Enumerates each evaluator module + function callable from the project tree.
2. Probes each on a small synthetic SMILES cohort (CCO + cisplatin + others).
3. Reports the MEASURED pass count vs the legacy R17 14/14 baseline.
4. Distinguishes SCAFFOLD-tier (CPU-only scaffold, NOT MEASURED-on-real)
   from MEASURED-on-real (Wf-388-A1 cohorts).

---

## 3. 各评估器的端到端用法

### 3.1 Vina 对接 (R17)
[same as legacy §3.1]

### 3.2 SA score (R17)
[same as legacy §3.2]

### 3.3 QED (R17)
[same as legacy §3.3]

### 3.4 PoseBusters (R17)
[same as legacy §3.4]

### 3.5 BioPython-based interaction fingerprint (R17, PLIP replacement)
[same as legacy §3.5]

### 3.6 OpenBabel (R17)
[same as legacy §3.6]

### 3.7 Wf-380-A2 m2 R² per-cohort (NEW — SCAFFOLD tier)

```python
from molmetal.scripts.wf380a2_m2_r2_per_cohort import select_features, compute_r2_for_cohort

# 3-tier feature selection per cohort size
config = select_features(n_samples=200)  # → {'tier': 'n_lt_300', 'features': 'rdkit_12d', 'model': 'ridge', 'alpha': 1.0}

# Per-cohort R² via 5 seeds × 5-fold CV (Wf-335-M2 protocol)
r2 = compute_r2_for_cohort(
    cohort="platinai_4cell",
    n_samples=200,
    config=config,
    n_seeds=5, n_folds=5,
)
# MEASURED-on-real per Wf-388-A1: platinai_4cell R²=-0.1440 ± 0.0325
# Source: TODO/completed/388a1_m2_per_cohort_real.md §0
```

### 3.8 Wf-380-A3 m7 Diversity sampling (NEW — SCAFFOLD tier)

```python
import torch
from molmetal.adapters.flow_matching_lipman.diversity_sampling import (
    nucleus_sample, mmd_loss, repulsion_loss, diversity_aware_sampling_step,
)
from molmetal.adapters.flow_matching_lipman.scaffold_conditioning import (
    murcko_scaffold_from_smiles, scaffold_condition, scaffold_buckets,
)

# Nucleus sampling (Holtzman 2020)
logits = torch.randn(2, 8)  # (batch, vocab)
samples = nucleus_sample(logits, p=0.9, k=50, temperature=1.0)

# MMD unbiased U-statistic (Gretton 2012)
embeddings = torch.randn(8, 16)
loss = mmd_loss(embeddings, sigma=1.0)

# Murcko scaffold (Bemis-Murcko 1996)
scaffold = murcko_scaffold_from_smiles("CC(=O)Oc1ccccc1C(=O)O")
```

### 3.9 Wf-380-A4 m3 Vina 3-level config (NEW — SCAFFOLD tier)

```python
from molmetal.scripts.wf380a4_vina_param_tuning import load_vina_config

# 3-level exhaustiveness grid (low/medium/high) per TODO-203
config = load_vina_config()
# → {'levels': {'low': {'exhaustiveness': 8}, 'medium': {'exhaustiveness': 32}, 'high': {'exhaustiveness': 64}}, ...}
```

### 3.10 Wf-380-A5 m15 RMSD multi-template (NEW — SCAFFOLD tier)

```python
from molmetal.adapters.ligand_vs_pocket_rmsd_adapter import (
    compute_pose_rmsd, LigandVsPocketRMSD,
)

# Single-template pose RMSD
rmsd = compute_pose_rmsd(
    mol_smiles="CCO",
    pocket_atoms=[(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)],
)

# Multi-template bench (TODO-204)
bench = LigandVsPocketRMSD(templates=["1h36", "3eml", "3ny8"])
result = bench.best_of_n_pose(mol_smiles="CCO")
```

### 3.11 Wf-380-A6 m4 SA penalty (NEW — SCAFFOLD tier)

```python
from molmetal.adapters.flow_matching_lipman.sa_penalty import (
    compute_sa_score, sa_aware_loss_step, aggregate_sa_penalty,
)

# Per-sample SA score (Ertl 2009 §3)
sa = compute_sa_score(mol)  # 1-10 scale

# SA-aware loss step (penalty term for trainer)
loss = sa_aware_loss_step(sa, target_sa=2.5, penalty_weight=0.1)
```

### 3.12 Wf-382-A1 R20-P3 PLLFeatureAdapter (NEW — SCAFFOLD tier)

```python
from molmetal.adapters.flow_matching_lipman.pll_features import (
    PLLFeatureAdapter, pll_to_feature_vector,
)

adapter = PLLFeatureAdapter()  # 10-d feature vector
features = pll_to_feature_vector(mol, adapter=adapter)
```

### 3.13 Wf-388-A1 m2 R² per-cohort REAL (NEW — MEASURED-ON-REAL tier)

```python
from molmetal.scripts.wf388a1_m2_per_cohort_real import (
    load_platinai_4cell, load_hilbert, load_metalcytotox_combined,
    apply_per_cohort_config, train_eval_per_cohort, aggregate_per_cohort_r2,
)

X, y, smiles, target = load_platinai_4cell(max_samples=200)
config = apply_per_cohort_config(n_samples=200)
result = train_eval_per_cohort(X, y, smiles, target, config=config, n_seeds=5, n_folds=5)
# MEASURED-on-real: platinai_4cell R²=-0.1440 ± 0.0325
# Source: TODO/completed/388a1_m2_per_cohort_real.md §0
```

### 3.14 Wf-388-A2 m7 diversity wire + Wf-388-A3 m4 SA penalty wire (NEW — WIRE-IN tier)

These are trainer CLI flag wire-ins (`molmetal/scripts/train_cfm_yuelbond_wasserstein.py`).
When flags default OFF, trainer loss is bit-exact equivalent to Wf-380-B1 ship surface.

```bash
# m7 diversity wire (6 CLI flags: --top-p --top-k --temperature --mmd-weight --repulsion-weight --scaffold-conditioning)
/home/hugo/codes/mol-metal/.venv/bin/python \
    molmetal/scripts/train_cfm_yuelbond_wasserstein.py \
    --top-p 0.7 --mmd-weight 0.1 --scaffold-conditioning

# m4 SA penalty wire
/home/hugo/codes/mol-metal/.venv/bin/python \
    molmetal/scripts/train_cfm_yuelbond_wasserstein.py \
    --sa-penalty-weight 0.05
```

---

## 4. 与 paper §4 Evaluation 的对应

[same as legacy §4 table + Wf-380..394 additions]

| Paper § | 评估器 | metric 名 | Tier | Source |
|---|---|---|---|---|
| §4.1 合成可及性 | SA score (Ertl) | `sa_score` / `sa_unit` | R17 | EVALUATORS_MIGRATED.md §3.2 |
| §4.2 类药性 | QED | `qed` | R17 | EVALUATORS_MIGRATED.md §3.3 |
| §4.3 对接亲和 | Vina / QuickVina | `vina_kcal_per_mol` | R17 | EVALUATORS_MIGRATED.md §3.1 |
| §4.4 高亲和率 | Vina + ref | `vina_high_affinity` | R17 | EVALUATORS_MIGRATED.md §3.1 |
| §4.5 几何合理性 | PoseBusters | `pb_pass` | R17 | EVALUATORS_MIGRATED.md §3.4 |
| §4.6 蛋白-配体相互作用指纹 | BioPython NeighborSearch | `interaction_fingerprint` | R17 | EVALUATORS_MIGRATED.md §3.5 |
| §4.7 3D 一致性 | OpenBabel + RMSD | `mol/atm stable` | R17 | EVALUATORS_MIGRATED.md §3.6 |
| §4.8 m2 R² per-cohort | Wf-388-A1 apply_per_cohort_config | `m2_r2` | MEASURED-ON-REAL | EVALUATORS_MIGRATED.md §3.13 |
| §4.9 m7 Diversity (scaffold) | Wf-380-A3 Murcko scaffold | `m7_diversity` | SCAFFOLD | EVALUATORS_MIGRATED.md §3.8 |
| §4.10 m4 SA generation penalty | Wf-380-A6 compute_sa_score | `m4_sa_penalty` | SCAFFOLD | EVALUATORS_MIGRATED.md §3.11 |
| §4.11 m15 RMSD multi-template | Wf-380-A5 compute_pose_rmsd | `m15_rmsd` | SCAFFOLD | EVALUATORS_MIGRATED.md §3.10 |
| §4.12 R20-P3 PLL features | Wf-382-A1 PLLFeatureAdapter | `pll_features` | SCAFFOLD | EVALUATORS_MIGRATED.md §3.12 |

---

## 5. 已知 regression 与 workaround

### 5.1 PLIP 不可用 (R17)
[same as legacy]

### 5.2 PySR 不可用 (R17)
[same as legacy]

### 5.3 QuickVina2-GPU 2.1 是 AMD-only (R17)
[same as legacy]

### 5.4 SCAFFOLD-tier evaluators (NEW — Wf-395-A1)

**Honest framing** (per party-twenty-spirit §3.5):
- 9 SCAFFOLD-tier evaluators (rows 13-21) pass synthetic-input probe but are
  **NOT yet MEASURED end-to-end on real cohorts**.
- Their real-cohort lift is PROJECTED per Wf-380..382 closeouts; full lift
  requires USER_OK-gated GPU retrain per AGENTS.md §5.1.
- Honest-negative preserved per `docs/HONEST_NEGATIVE_CATALOG.md`.

### 5.5 m6_kappa HONEST_NEGATIVE_PARTIAL_STRUCTURAL (preserved)

Per Wf-365-A3 verdict doc + Wf-377-A1 redesign + Wf-389-A1 drop-from-§3:
- **m6_kappa DROPPED** from Tier-2 §3 main-result
- top-3 alternatives (MolLogP + MorganDensity_r2 + Chi3v) PROMOTED to §3 supplementary
- m6_kappa remains §6 honest-limitations complementary heuristic only
- Source: `docs/M6_KAPPA_FINAL_VERDICT.md` + `docs/M6_KAPPA_REDESIGN_PROPOSAL.md`

### 5.6 m2 R² platinai_4cell + hilbert HONEST_NEGATIVE_R2_DEEP (preserved)

Per Wf-369-A2 verdict + Wf-377-A2 5-alternative-split diagnostic:
- **platinai_4cell (n=200) + hilbert (n=200) R² PRESERVED HONEST_NEGATIVE_R2_DEEP**
- metalcytotox_combined (n=2000) R²=+0.1981 ± 0.0056 MEASURED WEAK_RECOVERY
- Source: `docs/M2_R2_COHORT_DIAGNOSTIC.md`

---

## 6. 评估器与 4 个 skill 的对应

[same as legacy + Wf-380..394 additions]

| Skill | 评估器入口 | Tier |
|---|---|---|
| `mol-metal-cfm-training` | `molmetal.metrics.by_metric.cfm_decode_real.json` (generate 阶段) | R17 |
| `mol-metal-lambda-calculus` | `molmetal.molmetal_lam.reward.{sa_score,qed_scorer}.py` | R17 |
| `mol-metal-click-chemistry` | `molmetal.molmetal_lam.sbdd_env.synthesis_gate.SynthesisGate` (18-rule + 84-MLE) | R17 |
| `mol-metal-anticancer` | `molmetal.molmetal_lam.sbdd_env.vina_adapter.VinaDockingAdapter` + `posebusters_adapter.PoseBustersAdapter` + `metal_generator_adapter` | R17 |
| `mol-metal-cfm-training` (NEW) | `molmetal.adapters.flow_matching_lipman.diversity_sampling.nucleus_sample` (m7) | SCAFFOLD |
| `mol-metal-cfm-training` (NEW) | `molmetal.adapters.flow_matching_lipman.sa_penalty.compute_sa_score` (m4) | SCAFFOLD |
| `mol-metal-multiph-pll` (NEW) | `molmetal.adapters.flow_matching_lipman.pll_features.PLLFeatureAdapter` (R20-P3) | SCAFFOLD |
| `mol-metal-anticancer` (NEW) | `molmetal.scripts.wf388a1_m2_per_cohort_real.apply_per_cohort_config` (m2 R²) | MEASURED-ON-REAL |

---

## 7. 下一步

- [x] R17: 把 Vina/SA/QED 三个评估器接入 `molmetal/molmetal_lam/pipeline/closed_loop.py` 主循环 (legacy §7 row 1)
- [x] R17: 写 `molmetal/scripts/eval_pipeline.py`：input SMILES → output {sa, qed, vina, pb, interactions} 一行 JSON (legacy §7 row 2 — `molmetal/scripts/eval_all_evaluators.py` covers this)
- [ ] Wf-395-A1: end-to-end integrate Wf-380..394 SCAFFOLD evaluators into `closed_loop.py` — requires USER_OK-gated GPU retrain per AGENTS.md §5.1
- [ ] Wf-395-A1: convert SCAFFOLD evaluators to MEASURED-on-real via Wf-388-A1 protocol extension (m7 + m4 + m15 + R20-P3) — deferred to Wf-396+ ultracode wave
- [x] Wf-395-A1: write `molmetal/scripts/wf395a1_evaluator_audit.py` (this row — 22/22 MEASURED-OK)

---

## 8. Wf-395-A1 audit script reference (NEW)

**File**: `molmetal/scripts/wf395a1_evaluator_audit.py` (NEW per Wf-395-A1)
**Test file**: `molmetal/tests/test_wf395a1_evaluator_audit.py` (NEW per Wf-395-A1 — 4 tests)
**Audit JSON**: `molmetal/reports/wf395a1_evaluator_audit/audit.json` (per-evaluator record)
**Closeout**: `TODO/completed/395a1_evaluators_refresh.md`

This audit script:
1. Enumerates 22 evaluators (12 R17 + 9 SCAFFOLD + 1 MEASURED-ON-REAL) by tier.
2. Probes each on synthetic input (CCO + cisplatin + others).
3. Persists per-evaluator PASS/FAIL record to JSON.
4. Distinguishes SCAFFOLD-tier (NOT MEASURED-on-real) from MEASURED-on-real (Wf-388-A1).

---

## 9. Cross-references

- `TODO/completed/388a1_m2_per_cohort_real.md` — m2 R² REAL-cohort MEASUREMENT source
- `TODO/completed/380a2_per_cohort_features.md` — Wf-380-A2 m2 R² per-cohort SCAFFOLD
- `TODO/completed/380a3_diversity_sampling.md` — Wf-380-A3 m7 diversity sampling SCAFFOLD
- `TODO/completed/380a4_vina_param_tuning.md` — Wf-380-A4 m3 Vina tuning SCAFFOLD
- `TODO/completed/380a5_multi_template_mm_gbsa.md` — Wf-380-A5 m15 RMSD multi-template SCAFFOLD
- `TODO/completed/380a6_sa_generation_penalty.md` — Wf-380-A6 m4 SA penalty SCAFFOLD
- `TODO/completed/382a1_multiph_pll_scaffold.md` — Wf-382-A1 R20-P3 PLL features SCAFFOLD
- `TODO/completed/388a2_m7_diversity_wire.md` — Wf-388-A2 m7 diversity WIRE-IN
- `TODO/completed/388a3_m4_sa_wire.md` — Wf-388-A3 m4 SA penalty WIRE-IN
- `TODO/02_metrics/METRICS_TARGETS.md` §11 (Wf-393-A2 refresh) — per-metric Wf-380..390 status
- `docs/HONEST_NEGATIVE_CATALOG.md` — honest-negative preservation catalog
- `molmetal/scripts/eval_all_evaluators.py` — R17-era canonical 14/14 verification
- `molmetal/scripts/wf395a1_evaluator_audit.py` — NEW audit (this row)
- `molmetal/reports/wf395a1_evaluator_audit/audit.json` — per-evaluator audit record
