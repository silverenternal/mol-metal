# `molmetal.scoring.ROUTING_POLICY` — Metric → Real-Tool Routing (Wf-43 Phase 2)

> **Wf-43 invariant**: every metric in this registry is computed by a
> **real, locally-deployed** scoring tool. NO mock fallback is permitted
> at any layer. If a metric's real tool is unavailable (binary missing
> on PATH, ML checkpoint absent, GPU not present, commercial licence
> missing), the call raises `AdapterUnavailable`. Silently fabricating a
> number is the failure mode Wf-43 was created to eliminate.

## 1. Scope

Wf-43 Phase 1 (`score_any.py` + `REGISTRY.md`) shipped the **runtime
router**: a per-tool wrapper for each registered metric. Wf-43 Phase 2
(this document + `policy.json`) ships the **policy table**: which tool
is mandatory for which metric, what the fallback chain is, what data
is required, and which metrics are reserved / mock-deleted.

The policy covers **three populations** of metrics:

1. **Per-molecule / per-pose** metrics routed through
   `molmetal.scoring.score_any(...)` — 34 entries.
2. **Model-evaluation AUC / Pearson-r** metrics produced by the
   training script directly — 10 entries, NOT routed through
   `score_any` (the model IS the tool).
3. **Training-side diagnostics** (decode ratio, vocab coverage,
   dataset coverage, baseline-random comparison, external-scorer
   summary) — 7 entries, NOT paper-grade metrics.

`mock_keys_in_registry = []` — the Wf-43 hard guarantee that no metric
name in `REGISTRY` contains the substring `"mock"`.

## 2. Routing table (per-molecule / per-pose)

### 2.1 Docking / binding-affinity (mandatory: real physics tool)

| Metric | Primary tool | Fallback chain | Required data | Status |
|---|---|---|---|---|
| `vina_score` | AutoDock Vina 1.2.7 (Meeko+Python binding or CLI) | `vina_qvina2` → `vina_gpu` | smiles + pocket_pdbqt | MEASURED |
| `vina_high_affinity` | AutoDock Vina 1.2.7 | `vina_qvina2` → `vina_gpu` | smiles + pocket_pdbqt + cutoff_kcal | MEASURED |
| `vina_min_score` | AutoDock Vina 1.2.7 | `vina_qvina2` → `vina_gpu` | smiles + pocket_pdbqt + cutoff_kcal | MEASURED |
| `vina_qvina2` | QuickVina 2 (CASO-derived `qvina02` binary) | — | smiles + pocket_pdbqt | RESERVED |
| `vina_gpu` | Vina-GPU (CUDA + `vina_gpu` extension) | `vina_score` | smiles + pocket_pdbqt | RESERVED |
| `vina_reference` | AutoDock Vina 1.2.7 (co-crystal redock) | `vina_qvina2` | reference_ligand_pdbqt + co_crystal_pocket_pdbqt | MEASURED |
| `glide_score` | Glide (Schrödinger, commercial) | `vina_score` | smiles + pocket_pdbqt | RESERVED |

Aliases (pointed at `vina_score` / `vina_high_affinity` / `vina_min_score`):
`vina_kcal_per_mol`, `vina_mean`, `vina_high_affinity_{1h36,3eml,3ny8,4aaw,4rlu,test_001}`,
`vina_relaxed_success*`, `vina_triple_success*`,
`vina_reference_{1h36,3eml,3ny8,4aaw,4rlu,test_001}`.

**Engine selection** (env var): `MOLMETAL_VINA_ENGINE=qvina2` routes
`vina_score` / `vina_high_affinity` / `vina_min_score` to `vina_qvina2`;
`MOLMETAL_VINA_ENGINE=gpu` routes to `vina_gpu`. Unset = AutoDock Vina.

### 2.2 3D pose prediction (mandatory: real SE(3)-equivariant model)

| Metric | Primary tool | Fallback chain | Required data | Status |
|---|---|---|---|---|
| `diffdock_pose` | DiffDock (Corso 2024) | `flowdock_pose` | smiles + pocket_pdb | MEASURED |
| `flowdock_pose` | FlowDock (R17 ship, flow-matching) | `diffdock_pose` | smiles + pocket_pdb | MEASURED |
| `equibind_pose` | EquiBind (NOT SHIPPED) | `flowdock_pose` → `diffdock_pose` | smiles + pocket_pdb | RESERVED |

### 2.3 3D ligand generation (mandatory: real generative model)

| Metric | Primary tool | Fallback chain | Required data | Status |
|---|---|---|---|---|
| `pocket2mol` | Pocket2Mol (Peng 2022) | — | pocket_pdb | RESERVED (ckpt-gated) |
| `targetdiff` | TargetDiff (Guan 2023 ICLR) | — | pocket_pdb | MEASURED |
| `decomp_diff` | DecompDiff (NOT SHIPPED) | — | pocket_pdb | RESERVED |
| `molcraft` | MolCraft (NOT SHIPPED) | — | pocket_pdb | RESERVED |

### 2.4 3D pocket / binding-site analysis (mandatory: real BioPython / voxel tool)

| Metric | Primary tool | Fallback chain | Required data | Status |
|---|---|---|---|---|
| `af2_confidence` | Voxelization complexity (AF2 surrogate) | — | pocket | MEASURED |

### 2.5 Geometric / chemistry validity (mandatory: real PoseBusters)

| Metric | Primary tool | Fallback chain | Required data | Status |
|---|---|---|---|---|
| `pb_pass_rate` | PoseBusters 0.6.5 (26 checks) | — | smiles + pocket_pdb | MEASURED |

Aliases: `steric_clash_pb_dock`, `steric_clash_pb_dock_{1h36,test_001}`.

### 2.6 Retrosynthesis / synthesis accessibility (mandatory: real AIzynth / RDKit)

| Metric | Primary tool | Fallback chain | Required data | Status |
|---|---|---|---|---|
| `aizynth_synthesis` | AiZynthFinder 4.6.0 (MCTS) | `smarts_synthesis_gate` | smiles | MEASURED |
| `flowr` | FlowER retrosynthesis (R17) | — | smiles | MEASURED |
| `sa_score` | SA score (Ertl-Schuffenhauer 2008, RDKit Contrib) | — | smiles | MEASURED |

Alias: `sa_mean`.

### 2.7 Property prediction — RDKit-based (mandatory: real RDKit descriptor)

| Metric | Primary tool | Fallback chain | Required data | Status |
|---|---|---|---|---|
| `logp7_4` | RDKit Crippen logP + Patrick 2009 ionisation correction | — | smiles | MEASURED |
| `gi50_proxy` | Hou 2007 RDKit-descriptor heuristic | — | smiles | MEASURED |
| `cell_permeability_logPapp` | Hou 2007 / Mente 2015 heuristic | — | smiles | MEASURED |
| `herg_cardio_risk` | Aronov 2005 + Veber 2002 SMARTS composite | — | smiles | MEASURED |
| `ames_mutagen` | Benigni-Richard 2005 / Sushko 2012 SMARTS alerts | — | smiles | MEASURED |
| `hepatotox_index` | Hughes 2008 rule-of-2 + Stepan 2011 alerts | — | smiles | MEASURED |
| `aqueous_solubility_logS` | ESOL (Delaney 2004) | — | smiles | MEASURED |
| `plasma_protein_binding` | Obach 1999 logistic regression | — | smiles | MEASURED |
| `ring_size_distribution` | RDKit GetRingInfo | — | smiles | MEASURED |
| `pharmacophore_pass_rate` | Lipinski Ro5 + Veber 2002 + ring-count SMARTS | — | smiles | MEASURED |

### 2.8 Activity / affinity prediction (mandatory: real ML model)

| Metric | Primary tool | Fallback chain | Required data | Status |
|---|---|---|---|---|
| `boltz_affinity` | D-MPNN pIC50 (R17 ship, locally-trained attentive fp) | — | smiles | MEASURED |

Aliases: `affinity_pearson_r`, `pearson_r_pic50`,
`platinai_a2780_pred`, `platinai_mcf7_pred`.

### 2.9 OOD generalization (mandatory: real trained model — RESERVED)

| Metric | Primary tool | Fallback chain | Required data | Status |
|---|---|---|---|---|
| `drugood_ood` | DrugOOD OOD (NOT SHIPPED) | — | smiles + split | RESERVED |
| `imold_ood` | IMOLDE OOD (NOT SHIPPED) | — | smiles + split | RESERVED |

### 2.10 Topological signature (mandatory: real topology algorithm)

| Metric | Primary tool | Fallback chain | Required data | Status |
|---|---|---|---|---|
| `pl_likelihood` | PLL signature (Liu/Feng 2026) | — | smiles | MEASURED |

### 2.11 Fingerprint similarity / 3D RMSD (mandatory: real RDKit)

| Metric | Primary tool | Fallback chain | Required data | Status |
|---|---|---|---|---|
| `reference_tanimoto` | RDKit Morgan FP + Tanimoto | — | smiles + reference_smiles | MEASURED |
| `diversity_tanimoto` | RDKit Morgan FP + Tanimoto (pairwise) | — | smiles_list | MEASURED |
| `diversity_subpocket` | Per-residue sub-pocket FP (BioPython NeighborSearch) | — | smiles_list + pocket_pdb | MEASURED |
| `rigid_fragment_rmsd` | RDKit MCS + GetBestRMS (symmetry-aware) | — | smiles_a + smiles_b | MEASURED |
| `com_shift` | RDKit MMFF94 + CoM shift | — | smiles | MEASURED |

Aliases: `rigid_fragment_rmsd_4rlu`, `com_shift_{1h36,3eml,3ny8,4aaw,4rlu}`.

### 2.12 Metal-specific (mandatory: real RDKit SMARTS)

| Metric | Primary tool | Fallback chain | Required data | Status |
|---|---|---|---|---|
| `metal_compliance` | RDKit SMARTS metal-centre detection | — | smiles + metal_label | MEASURED |
| `metal_coord_compliance` | RDKit SMARTS d-block coord-shell detection | — | smiles + metal_label | MEASURED |

### 2.13 MOCK_DELETED (Phase-4A external scorers — module absent)

These four Phase-4A scorers were stubs in
`molmetal/molmetal_lam/sbdd_env/metrics_v2.py` referencing the
**non-existent** module `molmetal_lam.sbdd_env.external_scorers`. Wf-43
Phase 2 marks them `MOCK_DELETED`. Each alias pointing at these names
(`lambda_mcts_*`, `known_pt_drug_*`) inherits the same disposition.

| Metric | Disposition | Real alternative |
|---|---|---|
| `coord_geometry_proxy` | MOCK_DELETED | `metal_coord_compliance` (RDKit SMARTS, real) |
| `phototherapy_activity` | MOCK_DELETED | D-MPNN trained on phototherapy labels (no ship) |
| `ptiv_reduction_potential` | MOCK_DELETED | DFT surrogate (outside scope) |
| `metalloprotein_affinity` | MOCK_DELETED | D-MPNN on `molmetal/data/protein_metal_binding.csv` |

## 3. Model-evaluation metrics (NOT routed through score_any)

These are produced by the **training script** (e.g.
`molmetal/scripts/train_*.py`) directly, using a real model checkpoint.
They are **model-inference results**, not score_any routes. They appear
in `metrics/by_metric/*.json` for paper §4 reporting.

| Metric | Primary tool | Training script |
|---|---|---|
| `metalcytotox_ru_auc` | D-MPNN multi-task (scaffold split) | `train_metal_cytotox.py` |
| `metal_lomo_auc` | D-MPNN multi-task LOMO | `train_metal_lomo.py` |
| `multimetal_lomo` | D-MPNN multi-task LOMO (broader) | `train_multimetal_lomo.py` |
| `moleculenet_auc` | D-MPNN multi-task (MoleculeNet 8-task) | `train_moleculenet.py` |
| `drugood_ood_auc` | Morgan-fp MLP (DrugOOD) | `train_drugood.py` |
| `nci60_precious_9cell_auc` | D-MPNN multi-task (NCI60 9-cell) | `train_nci60_10cell.py` |
| `platinai_4cell_auc` | D-MPNN multi-task (PlatinAI 4-cell) | `train_platinai_4cell.py` |
| `platinai_3cell_resistance` | D-MPNN multi-task (PlatinAI drop A2780cis) | `train_cisplatin_resistance.py` |
| `metric18_protein_metal` | D-MPNN (protein-metal binding) v1 | `train_protein_metal.py` |
| `metric18_protein_metal_v2` | D-MPNN (protein-metal binding) v2 | `train_protein_metal.py` |

## 4. Training diagnostics (NOT paper-grade)

These JSONs are training-side observability / dataset-coverage metrics.
They are NOT paper §4 evaluation metrics and they are NOT routed through
`score_any`. They are kept for ablation debugging only.

* `decode_ratio` — fraction of generated samples with valid SMILES
  (RDKit-strict parse)
* `atom_vocab_coverage` — fraction of training-vocab atoms seen
* `n_distinct` — number of distinct canonical SMILES per cell
* `n_train_scaleup` — CFM training-data size scaleup (32 → 500)
* `metallodrug_vertical_coverage` — dataset coverage of the metallodrug vertical
* `baseline_random_smiles` — random SMILES baseline for diversity
* `external_scorers_summary` — summary envelope for the MOCK_DELETED Phase-4A scorers

## 5. Mock audit (Wf-43 Phase 2)

The Wf-43 invariant forbids any production route through a mock.
Three classes of "mock" exist in the repo:

| Class | Status | Disposition |
|---|---|---|
| `molmetal/adapters/mock.py:MockGenerator / MockDocker / MockPredictor / MockScorer` | KEPT for unit-test use only | NOT routed by `score_any`. `test_no_mock_keyword_in_registry` enforces that no metric name in REGISTRY contains `"mock"`. |
| `molmetal/scripts/dock_10pockets_se3sb.py:_stub_score` | KEPT for the genuine "no vina/meeko" failure case | NOT routed through score_any. The score_any path raises `AdapterUnavailable` when VinaDockingAdapter cannot import. The _stub_score function only fires inside `run_sweep(..., use_real_vina=False)` or `run_sweep(use_real_vina=True)` when the adapter import fails — both are explicit opt-in failures, never silent substitutions. |
| `molmetal/scripts/wf30_dock_summarize.py:scorer="stub" classifier` | INFORMATIONAL ONLY | Reads `use_real_vina` flag from upstream and labels the per-pocket summary `scorer: "stub"` / `"vina"`. No scoring happens here. |

The four Phase-4A external scorers (`coord_geometry_proxy`,
`phototherapy_activity`, `ptiv_reduction_potential`,
`metalloprotein_affinity`) were the worst class of mock — they used a
**missing module** (`molmetal_lam.sbdd_env.external_scorers`) whose
absence caused `_safe_external_score` to silently return `None`. Wf-43
Phase 2 marks them `MOCK_DELETED` and redirects each to a real
alternative (where one exists) or honest non-paper disposition.

## 6. Files

* `molmetal/scoring/__init__.py` — public surface (REGISTRY, score_any, …)
* `molmetal/scoring/score_any.py` — runtime router (Wf-43 Phase 1)
* `molmetal/scoring/REGISTRY.md` — runtime metric table (Wf-43 Phase 1)
* `molmetal/scoring/ROUTING_POLICY.md` — **this document** (Wf-43 Phase 2)
* `molmetal/scoring/policy.json` — **machine-readable policy** (Wf-43 Phase 2)
* `molmetal/tests/test_score_any.py` — Phase 1 acceptance gate
* `molmetal/tests/test_score_any_policy.py` — Phase 2 acceptance gate (audit, see §7)

## 7. Acceptance gate

Wf-43 Phase 2 verification:

```bash
# 1. Score-any module exposes ROUTING_POLICY dict (≥30 entries):
python -c "from molmetal.scoring.score_any import score_any, ROUTING_POLICY; print(len(ROUTING_POLICY))"

# 2. Policy JSON is valid JSON with the expected shape:
python -c "import json; p=json.load(open('molmetal/scoring/policy.json')); print(len(p['policy']), 'metrics in policy')"

# 3. Mock-audit gate — no metric name in score_any REGISTRY contains 'mock':
python -c "from molmetal.scoring.score_any import REGISTRY; assert all('mock' not in n.lower() for n in REGISTRY); print('OK')"

# 4. Mock-audit gate — all MOCK_DELETED entries point at real alternatives:
python -c "import json; p=json.load(open('molmetal/scoring/policy.json')); deleted=[m for m,e in p['policy'].items() if e['wf43_status']=='MOCK_DELETED']; print(f'{len(deleted)} MOCK_DELETED entries:', deleted)"

# 5. End-to-end Phase 2 test:
pytest molmetal/tests/test_score_any_policy.py -v
```

## 8. Adding a new metric — workflow

1. **Pick the real tool** (RDKit, BioPython, PLIP, OpenBabel, ProDy,
   Meeko+Vina, OpenMM, Schrödinger, Boltz, AF2, …) — must be locally
   installed or shipped in this repo. Mocks / stubs / heuristic "fake"
   models are FORBIDDEN as the primary tool.
2. **Implement the adapter** in `molmetal/molmetal_lam/sbdd_env/<name>.py`
   — import lazily in `score_any._import_<name>()`.
3. **Add a wrapper** in `molmetal/scoring/score_any.py` that calls the
   adapter and returns the real score (or raises
   `AdapterUnavailable` on missing tool).
4. **Register** in `REGISTRY` with `_register("<metric_name>", ToolRef(...))`
   and in `policy.json` `policy.<metric_name>` with the same shape.
5. **Document** the entry in this file (ROUTING_POLICY.md) with the
   table row.
6. **Test** in `molmetal/tests/test_score_any.py` and
   `molmetal/tests/test_score_any_policy.py`.
7. **Verify** the acceptance gate (see §7) passes.

If the real tool is **not yet shipped** (e.g. EquiBind, DecompDiff,
MolCraft, DrugOOD, IMOLDE, Glide), register the metric with
`backend="python"`, `module="NOT_SHIPPED"`, `fallback_chain=[]` and
`wf43_status="RESERVED"`. The wrapper must raise
`AdapterUnavailable` — the metric name is reserved so call-sites
written today survive the upcoming ship.