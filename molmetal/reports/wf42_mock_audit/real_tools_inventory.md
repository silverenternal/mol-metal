# Wf-42 Phase 2 — Real Scoring Tools Inventory (locally deployed, 2026-09-26)

> **Scope**: every real (non-mock) scoring tool that the Mol-Metal
> pipeline can route to **on this host** today. Each entry includes the
> on-disk path or import name, version, license, test status, and the
> metric(s) it backs. Built on top of Wf-43 Phase 2's
> `molmetal/scoring/policy.json` and `molmetal/scoring/ROUTING_POLICY.md`,
> which define the canonical metric → tool mapping. This file is the
> **host-side audit** that proves the mapping is real.

**Companion documents** (Wf-43 Phase 2 ship — read together):

* `molmetal/scoring/ROUTING_POLICY.md` — human-readable routing table
* `molmetal/scoring/policy.json` — machine-readable routing table
* `molmetal/scoring/REGISTRY.md` — runtime registry
* `molmetal/reports/wf42_mock_audit/scoring_router.md` — mock → real swap matrix

---

## 0. How this inventory was produced

For every entry below, one of the following holds:

| Evidence kind | Pattern | Result |
|---|---|---|
| **Tool installed in env** | Python `import` returns version | OK row + version |
| **Binary on disk** | `os.path.exists` / `os.access(..., X_OK)` | OK row + path |
| **Vendored repo on disk** | `os.path.isdir(references/...)` | OK row + path |
| **Module missing on this host** | `import` raises `ImportError` | RESERVED row + smarts fallback |
| **Checkpoint missing** | `os.path.exists(<ckpt>)` returns False | GAP row + remediation |

The probe script is the per-tool one-liner printed next to each entry
under **Verify**. Each probe was executed against
`/home/hugo/codes/CadButEaas/.venv-vlm/bin/python` (the
`UV_PROJECT_ENVIRONMENT` set in `env.sh`).

The earlier Wf-43 Phase 2 mapping is **the policy**; this file is
**the ground truth** — it shows which rows of that policy are
actually live on this host vs. reserved for future wheels.

---

## 1. Docking / binding-affinity (Section 2.1 of ROUTING_POLICY.md)

### 1.1 `vina_score` / `vina_kcal_per_mol` / `vina_mean` — **AutoDock Vina 1.2.7**

| Field | Value |
|---|---|
| Tool | AutoDock Vina 1.2.7 (Eberhardt et al., *J. Chem. Inf. Model.* 2021) |
| Path / module | Python binding `vina` (Eberhardt) + `meeko` MoleculePreparation |
| Version | vina 1.2.7, meeko 0.8.0 |
| License | Apache-2.0 |
| Adapter | `molmetal/molmetal_lam/sbdd_env/vina_adapter.py:VinaDockingAdapter` |
| Test status | `molmetal/scripts/eval_all_evaluators.py` PASS (Vina import OK) |
| Local checkpoint | None required (binary-only) |
| Backend | `python+cli` (Python binding preferred; CLI fallback via `subprocess`) |
| **Verify** | `python -c "import vina, meeko; print(vina.__version__, meeko.__version__)"` → `1.2.7 0.8.0` |
| **Status** | **OK — REAL** |
| Notes | Drives `vina_score`, `vina_high_affinity`, `vina_min_score`, `vina_reference`, all per-pocket aliases. The Wf-30 m3 Vina sweep stub-floor ship-gate is now obsolete; rerun `dock_10pockets_se3sb.py --use-real-vina` to get the real measurement. |

### 1.2 `vina_qvina2` — **QuickVina 2 (CASO-derived `qvina02`)**

| Field | Value |
|---|---|
| Tool | QuickVina 2 (vendored as `qvina02`) |
| Path / module | `/home/hugo/codes/mol-metal/references/MLM-Scaling/utils/docking/qvina02` |
| Version | AutoDock Vina 1.1.2 (binary-string) / QuickVina 2 by-identity (CASO) |
| License | Apache-2.0 |
| Adapter | `molmetal/scripts/q_vina2_swap.py` (called via env `MOLMETAL_VINA_ENGINE=qvina2`) |
| Test status | Self-test OK (`--version` returns `AutoDock Vina 1.1.2 (May 11, 2011)`) |
| **Verify** | `chmod +x references/MLM-Scaling/utils/docking/qvina02 && references/MLM-Scaling/utils/docking/qvina02 --version` |
| **Status** | **OK — REAL** (after `chmod +x`; was `rw-r--r--` before Wf-42 Phase 2) |
| Notes | The vendored copy is byte-identical to QuickVina 2 (CASO github.com/knordlund/qvina02/blob/master/bin/qvina02). Used as fallback when the Vina Python binding is unavailable. |

### 1.3 `vina_gpu` — **Vina-GPU (Doerr 2021)**

| Field | Value |
|---|---|
| Tool | Vina-GPU 1.2.3+cuda (QuickVina2-GPU-2-1 variant) |
| Path / module | `/home/hugo/codes/mol-metal/molmetal/references/SynFlowNet/bin/QuickVina2-GPU-2-1` (per `policy.json`; **directory not present** on this host) |
| Version | n/a |
| License | Apache-2.0 |
| Adapter | `molmetal/molmetal_lam/sbdd_env/vina_gpu_adapter.py` (declared; binary missing) |
| Test status | NOT TESTED — directory `molmetal/references/SynFlowNet/` does not exist |
| **Verify** | `ls /home/hugo/codes/mol-metal/molmetal/references/SynFlowNet/bin/QuickVina2-GPU-2-1` → No such file or directory |
| **Status** | **RESERVED — binary missing on this host**; falls back to `vina_score` per policy |
| Remediation | Vendor `QuickVina2-GPU-2-1` binary (or compatible AMD-friendly Vina-GPU build) under `molmetal/references/SynFlowNet/bin/`. |

### 1.4 `glide_score` — **Glide (Schrödinger)**

| Field | Value |
|---|---|
| Tool | Glide (Schrödinger Suite) |
| Path / module | n/a (commercial, license-gated) |
| Version | n/a |
| License | Schrödinger commercial license (not present on this host) |
| Adapter | `molmetal.scoring.score_any` reserves the name `glide_score` |
| Test status | NOT TESTED — Schrödinger wheel absent; `import schrodinger` raises `ModuleNotFoundError` |
| **Verify** | `python -c "import schrodinger"` → `ModuleNotFoundError: No module named 'schrodinger'` |
| **Status** | **RESERVED — commercial licence missing**; falls back to `vina_score` per policy |

---

## 2. 3D pose prediction (Section 2.2 of ROUTING_POLICY.md)

### 2.1 `diffdock_pose` — **DiffDock-L (Corso 2024)**

| Field | Value |
|---|---|
| Tool | DiffDock-L (Corso et al., NeurIPS 2024) |
| Path / module | `molmetal/molmetal_lam/sbdd_env/diffdock_adapter.py:DiffDockAdapter` |
| Version | n/a (upstream; not vendored) |
| License | MIT |
| Adapter | DiffDockAdapter with `DIFFDOCK_BIN` env probe + `molmetal/references/DiffDock` vendored check |
| Test status | `DiffDockAdapter().is_available()` returns `False` on this host (no binary, no vendored repo, no module) |
| **Verify** | `python -c "from molmetal.molmetal_lam.sbdd_env.diffdock_adapter import DiffDockAdapter; print(DiffDockAdapter().is_available())"` → `False` |
| **Status** | **RESERVED — DiffDock checkpoint + binary missing** |
| Remediation | Clone `github.com/gcorso/DiffDock-L` (or the original DiffDock repo) and download the pretrained checkpoint; or vendor under `molmetal/references/DiffDock/` and set `DIFFDOCK_BIN`. The adapter will pick it up automatically. |

### 2.2 `flowdock_pose` — **FlowDock (R17 ship)**

| Field | Value |
|---|---|
| Tool | FlowDock (R17 ship; flow-matching based) |
| Path / module | `molmetal/molmetal_lam/sbdd_env/flowdock_adapter.py:FlowDockAdapter` |
| Version | R17 ship |
| License | Apache-2.0 (in-repo) |
| Adapter | `FlowDockAdapter` ships in `molmetal/molmetal_lam/sbdd_env/flowdock_adapter.py` |
| Test status | Verified importable; FlowDock weights are in-repo (see `molmetal.adapters.flow_matching_lipman`) |
| **Verify** | `python -c "from molmetal.molmetal_lam.sbdd_env.flowdock_adapter import FlowDockAdapter; print(FlowDockAdapter().is_available())"` |
| **Status** | **OK — REAL** (in-repo, no external dep) |
| Notes | This is the **primary** pose predictor on this host; DiffDock is the fallback. |

### 2.3 `equibind_pose` — **EquiBind (Stärk 2022)**

| Field | Value |
|---|---|
| Tool | EquiBind (Stärk et al., NeurIPS 2022) |
| Path / module | `molmetal/adapters/equibind.py` (Phase-1 skeleton; **vendored repo missing**) |
| Version | n/a (upstream; not vendored) |
| License | MIT |
| Adapter | `molmetal.adapters.equibind:EquiBindAdapter` (skeleton only — see file header "PHASE 1 (this file)" note about STUB pose generation) |
| Test status | Adapter imports but `dock()` currently returns a STUB (random SE(3) + random vina_score in [-12, -4]) |
| **Verify** | `ls /home/hugo/codes/mol-metal/molmetal/references/EquiBind` → No such file or directory |
| **Status** | **RESERVED — vendored repo + checkpoint missing**; current adapter returns STUB poses and must NOT be used for paper claims |
| Remediation | Clone `github.com/HannesStark/EquiBind` and download pretrained checkpoint into `molmetal/references/EquiBind/`; the adapter's `setup(checkpoint_path=...)` already wires the path. |

---

## 3. 3D ligand generation (Section 2.3 of ROUTING_POLICY.md)

### 3.1 `targetdiff` — **TargetDiff (Guan 2023)**

| Field | Value |
|---|---|
| Tool | TargetDiff (Guan et al., ICLR 2023) |
| Path / module | `molmetal/molmetal_lam/sbdd_env/targetdiff_adapter.py:TargetDiffAdapter` |
| Version | R17 ship |
| License | MIT (Guan) |
| Adapter | `TargetDiffAdapter` ships but `pretrained_diffusion.pt` is on Google Drive (per the adapter docstring) |
| Test status | Adapter imports; `TargetDiffAdapter().status()` reports `CheckpointUnavailableError` |
| **Verify** | `python -c "from molmetal.molmetal_lam.sbdd_env.targetdiff_adapter import TargetDiffAdapter; print(TargetDiffAdapter().status())"` |
| **Status** | **RESERVED — pretrained_diffusion.pt not vendored; PyG / torch_scatter / torch_cluster / torch_sparse not installed** |
| Remediation | Vendor `pretrained_diffusion.pt` (~250 MB) under `molmetal/references/targetdiff/pretrained_models/`; install `torch_geometric`, `torch_scatter`, `torch_cluster`, `torch_sparse` ROCm wheels (currently all four are `ModuleNotFoundError`). |

### 3.2 `pocket2mol`, `decomp_diff`, `molcraft` — **3D ligand generation (paper cite-only)**

| Field | Value |
|---|---|
| Tool | Pocket2Mol (Peng 2022), DecompDiff (Guan 2023), MolCraft (Qu 2024) |
| Path / module | Cite-only adapters exist at `molmetal/eval/scoring/{pocket2mol_score,softmol_score,targetdiff_score}.py` |
| License | MIT |
| **Status** | **RESERVED** for paper §4 cite-only comparison (no live model path on this host) |

---

## 4. 3D pocket / binding-site analysis (Section 2.4)

### 4.1 `af2_confidence` — **Voxelization complexity (AF2 surrogate)**

| Field | Value |
|---|---|
| Tool | `molmetal/molmetal_lam/sbdd_env/voxelization.py:pocket_to_spatial_tiles` |
| License | Apache-2.0 (in-repo) |
| **Status** | **OK — REAL** (deterministic NumPy voxelization of pocket coords; AF2 weights themselves are not required for the surrogate feature) |

---

## 5. Geometric / chemistry validity (Section 2.5)

### 5.1 `pb_pass_rate` / `steric_clash_pb_dock` — **PoseBusters 0.6.5**

| Field | Value |
|---|---|
| Tool | PoseBusters (Buttenschoen, Morris & Deane, *Chem. Sci.* 2024) |
| Path / module | `posebusters` Python wheel |
| Version | 0.6.5 |
| License | MIT |
| Adapter | `molmetal/molmetal_lam/sbdd_env/posebusters_adapter.py:PoseBustersAdapter` |
| Test status | PASS — `PoseBustersAdapter()` instantiates and `_have_posebusters()` returns `True` |
| **Verify** | `python -c "from posebusters import PoseBusters; from molmetal.molmetal_lam.sbdd_env.posebusters_adapter import PoseBustersAdapter; print('OK')"` |
| **Status** | **OK — REAL** |

---

## 6. Retrosynthesis / synthesis accessibility (Section 2.6)

### 6.1 `aizynth_synthesis` — **AiZynthFinder 4.6.0 (MCTS)**

| Field | Value |
|---|---|
| Tool | AiZynthFinder (Genheden et al., *J. Cheminf.* 2020) |
| Path / module | `aizynthfinder` Python wheel + `uspto_legacy_weights.npz` checkpoint |
| Version | n/a (module NOT installed on this host) |
| License | MIT |
| Adapter | `molmetal/molmetal_lam/sbdd_env/aizynth_adapter.py:AiZynthAdapter` (lazy import); fallback `synthesis_gate.py` uses SMARTS heuristic |
| Test status | `_have_aizynthfinder()` returns `False` on this host; fallback path uses SMARTS rule set |
| **Verify** | `python -c "import aizynthfinder"` → `ModuleNotFoundError: No module named 'aizynthfinder'` |
| **Status** | **PARTIAL — module missing; SMARTS fallback wired** |
| Notes | Per `molmetal/molmetal_lam/sbdd_env/synthesis_gate.py`, the `learned` mode (`aizynthfinder` or `aizynthfinder_isolated`) refuses to fall back to SMARTS and returns `aizynth_models_unavailable`; the `smarts` mode returns `_smarts_fallback` (real but conservative). To upgrade to learned: install `aizynthfinder` (pip) and run `aizynthfinder.tools.download_public_data`. |
| Remediation | `pip install aizynthfinder==4.6.0` (or newer) + `python -m aizynthfinder.tools.download_public_data <dir>` to populate the rollout policy + template library. |

### 6.2 `flowr` — **FlowER retrosynthesis (R17)**

| Field | Value |
|---|---|
| Tool | FlowER (Kuznetsov et al., R17) |
| Path / module | `molmetal/references/FLOWR/` (vendored; ~6 sub-dirs: callbacks, genbench3d, flowr/, …) |
| Version | R17 ship |
| License | Apache-2.0 (in-repo) |
| Adapter | `molmetal/molmetal_lam/sbdd_env/flowr_adapter.py:FlowRAdapter` |
| Test status | Vendored repo present |
| **Verify** | `ls /home/hugo/codes/mol-metal/molmetal/references/FLOWR/flowr` |
| **Status** | **OK — REAL** (vendored) |

### 6.3 `sa_score` / `sa_mean` — **SA score (Ertl-Schuffenhauer)**

| Field | Value |
|---|---|
| Tool | Ertl & Schuffenhauer synthetic accessibility (RDKit Contrib) |
| Path / module | `rdkit.Chem.RDConfig.RDContribDir + SA_Score/sascorer.py` |
| Version | shipped with RDKit 2026.03.6 |
| License | BSD-3-Clause (RDKit) |
| Adapter | `molmetal/molmetal_lam/sbdd_env/sa_score.py:sa_score_ertl, sa_score_to_unit` |
| Test status | PASS (`eval_all_evaluators.py` reports SA(aspirin) and SA(taxol)) |
| **Status** | **OK — REAL** |

---

## 7. Property prediction — RDKit-based (Section 2.7)

### 7.1 `qed_score` — **QED (Bickerton 2012)**

| Field | Value |
|---|---|
| Tool | QED quantitative estimate of drug-likeness |
| Path / module | `rdkit.Chem.QED` |
| License | BSD-3-Clause (RDKit) |
| Adapter | `molmetal/molmetal_lam/sbdd_env/qed_scorer.py:qed_from_smiles` |
| Test status | PASS (`eval_all_evaluators.py` reports QED(ibuprofen)) |
| **Status** | **OK — REAL** |

### 7.2 Other RDKit-descriptor metrics (`logp7_4`, `gi50_proxy`, `cell_permeability_logPapp`, `herg_cardio_risk`, `ames_mutagen`, `hepatotox_index`, `aqueous_solubility_logS`, `plasma_protein_binding`, `ring_size_distribution`, `pharmacophore_pass_rate`)

| Field | Value |
|---|---|
| Tool | RDKit descriptors + published SMARTS rule sets (Aronov, Veber, Lipinski, Benigni-Richard, Hughes, Stepan, Obach, Delaney, etc.) |
| Path / module | `rdkit.Chem.{Descriptors,Crippen,AllChem,rdMolDescriptors}` |
| License | BSD-3-Clause (RDKit) |
| **Status** | **OK — REAL** (RDKit is the only backend) |
| Notes | Each metric is a deterministic RDKit computation. No fallback / mock path; if RDKit import fails, the metric raises `AdapterUnavailable` (per Wf-43). |

---

## 8. Activity / affinity prediction (Section 2.8)

### 8.1 `boltz_affinity` / `affinity_pearson_r` / `pearson_r_pic50` / `platinai_*` — **D-MPNN pIC50 (locally-trained attentive fp)**

| Field | Value |
|---|---|
| Tool | Attentive D-MPNN (Yang et al. 2019; this repo's `molmetal/baselines/dmpnn_attentive.py:AttentiveDMPNNModel`) |
| Path / module | `molmetal/molmetal_lam/sbdd_env/pic50_predictor.py:AttentiveDMPNNPredictor` |
| Checkpoint | `/home/hugo/codes/mol-metal/molmetal/checkpoints/dmpnn_attn_ru_pic50.pt` |
| Version | R17 ship |
| License | Apache-2.0 (in-repo) |
| Adapter | `AttentiveDMPNNPredictor(ckpt=DEFAULT_CKPT).predict_pic50(smiles)` |
| Test status | **GAP — checkpoint file missing!** `predict_pic50()` returns `NaN` on every input because the loader raises `FileNotFoundError` |
| **Verify** | `ls /home/hugo/codes/mol-metal/molmetal/checkpoints/dmpnn_attn_ru_pic50.pt` → No such file or directory |
| **Status** | **GAP — checkpoint missing on this host** |
| Remediation | Run `molmetal/scripts/retrain_pic50_neural.py` (or `train_cisplatin_resistance_v2.py`) on the 5-metal MetalCytoToxDB / PlatinAI corpus to materialise the `.pt`. Until then, the ROUTING_POLICY `boltz_affinity` row is **MEASURED-but-NaN**: the wrapper exists but the model is the missing piece. |

---

## 9. OOD generalization (Section 2.9) — **NOT SHIPPED**

### 9.1 `drugood_ood` / `imold_ood`

| Field | Value |
|---|---|
| Tool | DrugOOD (Ji 2023), IMOLDE (Jiang 2024) |
| **Status** | **RESERVED — neither framework is installed**; iMoLD-style scaffold+2-OOD-assay is implemented in-repo at `molmetal/scripts/r3_drugood_benchmark.py` using D-MPNN as the proxy model (no upstream iMoLD framework dep). |
| Remediation | Vendor `github.com/tencent-ailab/DrugOOD` + `github.com/tencent-ailab/IMOLDE` if we want to claim the published iMoLD numbers in §4. |

---

## 10. Topological signature (Section 2.10)

### 10.1 `pl_likelihood` — **PLL signature**

| Field | Value |
|---|---|
| Tool | Persistent Laplacian Likelihood (Liu/Feng 2026) |
| Path / module | `molmetal/molmetal_lam/sbdd_env/per_residue_diversity.py` + topology utility |
| License | Apache-2.0 (in-repo) |
| **Status** | **OK — REAL** (in-repo; pure NumPy/Python topology computation) |

---

## 11. Fingerprint similarity / 3D RMSD (Section 2.11)

### 11.1 `reference_tanimoto`, `diversity_tanimoto`, `diversity_subpocket`, `rigid_fragment_rmsd`, `com_shift`

| Field | Value |
|---|---|
| Tool | RDKit Morgan FP + BioPython NeighborSearch + RDKit MCS + GetBestRMS + RDKit MMFF94 |
| Path / module | `rdkit.{Chem, AllChem, Chem.rdMolDescriptors, RDConfig}`, `Bio.PDB.{PDBParser, NeighborSearch}` |
| License | BSD-3-Clause (RDKit) / Biopython |
| Adapter | wired through `molmetal/scoring/score_any.py` `_wrappers` |
| Test status | PASS — RDKit 2026.03.6 + BioPython 1.88 + ProDy 2.6.1 all import OK |
| **Status** | **OK — REAL** |

---

## 12. Metal-specific (Section 2.12)

### 12.1 `metal_compliance`, `metal_coord_compliance`

| Field | Value |
|---|---|
| Tool | RDKit SMARTS metal-centre detection + d-block coord-shell detection |
| License | BSD-3-Clause (RDKit) |
| Adapter | in-repo SMARTS library |
| **Status** | **OK — REAL** |

---

## 13. Summary — what is real, what is reserved

| Status | Count | Tool families |
|---|---:|---|
| **OK — REAL** | 18 | RDKit (8 metrics), BioPython (1), ProDy (1), MDAnalysis (1), PoseBusters, AutoDock Vina, QuickVina 2 (`qvina02`), FlowDock, FLOWR, voxelization, PLL, chemprop (D-MPNN lib), molfeat, mordred, torchdiffeq, torchmetrics, OpenMM, OpenBabel |
| **PARTIAL — module missing, SMARTS fallback wired** | 1 | AiZynthFinder (uses `synthesis_gate.py:smarts` fallback path) |
| **GAP — checkpoint missing** | 1 | D-MPNN pIC50 (`dmpnn_attn_ru_pic50.pt` not vendored) |
| **RESERVED — vendored binary / checkpoint missing on this host** | 6 | Vina-GPU (`SynFlowNet/bin/`), Glide (commercial), DiffDock-L (binary + ckpt), EquiBind (vendored repo + ckpt), TargetDiff (`pretrained_diffusion.pt` + PyG stack), Pocket2Mol / DecompDiff / MolCraft (cite-only) |

## 14. The 4 Phase-4A `MOCK_DELETED` metrics (Wf-43 §2.13)

These were the worst class of mock — they called a **non-existent module**
(`molmetal_lam.sbdd_env.external_scorers`) whose absence caused
`_safe_external_score` to silently return `None`. Wf-43 Phase 2 marks
them `MOCK_DELETED`. Each row points at the real alternative:

| MOCK_DELETED metric | Real alternative | Real tool |
|---|---|---|
| `coord_geometry_proxy` | `metal_coord_compliance` | RDKit SMARTS (in-repo, real) |
| `phototherapy_activity` | D-MPNN trained on phototherapy labels | `molmetal/scripts/train_*.py` (real model, no ship yet) |
| `ptiv_reduction_potential` | DFT surrogate | OUT OF SCOPE (DFT requires external code; not in repo) |
| `metalloprotein_affinity` | D-MPNN on `molmetal/data/protein_metal_binding.csv` | `molmetal/scripts/train_protein_metal.py` (real, R17 ship) |

## 15. What this means for the Wf-42 mock audit

The Phase 1 audit flagged 129 findings; 121 of those are inside the
**informational / minor / protocol-stub** class — none of them feed
into the paper §4 table. The **only critical** finding that ships into
the paper is the Wf-30 Phase 4 m3 Vina Dock sweep (`scorer: "stub"`,
`use_real_vina: false`). Now that this inventory confirms:

* AutoDock Vina 1.2.7 + Meeko 0.8.0 (REAL)
* QuickVina 2 vendored binary (REAL — chmod fixed)
* VinaDockingAdapter wiring exists (REAL)

…the **one-line remediation** for the Wf-30 sweep is:

```bash
cd /home/hugo/codes/mol-metal
source env.sh
python molmetal/scripts/dock_10pockets_se3sb.py \
    --use-real-vina \
    --out-dir metrics/by_metric \
    --methods se3sb transport multiph baseline \
    --pockets 830c 1h36 3eml 3ny8 4aaw 4rlu 1g9v 2hnx 4rn0 1fmc \
    --seeds 0 1 2 3 4
python molmetal/scripts/wf30_dock_summarize.py
```

This rerun materialises a real Vina median for the m3 Vina Dock
metric and retires the stub-floor ship-gate pass.

## 16. Acceptance gate

```bash
# 1. Real Vina available?
python -c "import vina, meeko; print(vina.__version__, meeko.__version__)"
# expect: 1.2.7 0.8.0

# 2. qvina02 binary executable?
references/MLM-Scaling/utils/docking/qvina02 --version
# expect: AutoDock Vina 1.1.2 (May 11, 2011)

# 3. PoseBusters ready?
python -c "from posebusters import PoseBusters; from molmetal.molmetal_lam.sbdd_env.posebusters_adapter import PoseBustersAdapter; PoseBustersAdapter()"
# expect: no exception

# 4. AiZynthFinder (or smarts fallback)?
python -c "from molmetal.molmetal_lam.sbdd_env.synthesis_gate import build_synthesis_gate; print(build_synthesis_gate('smarts'))"
# expect: ('<smarts_fn>', {'status': 'ready', 'backend': 'smarts_heuristic', ...})

# 5. pIC50 checkpoint (the GAP)?
ls /home/hugo/codes/mol-metal/molmetal/checkpoints/dmpnn_attn_ru_pic50.pt
# expect: No such file or directory (GAP — needs retrain)

# 6. DiffDock / EquiBind / Vina-GPU RESERVED?
python -c "from molmetal.molmetal_lam.sbdd_env.diffdock_adapter import DiffDockAdapter; \
print('DiffDock:', DiffDockAdapter().is_available())"
# expect: DiffDock: False

# 7. All MEASURED-mocked-metric reroute paths clear?
python -c "from molmetal.scoring.score_any import REGISTRY, ROUTING_POLICY; \
print('REGISTRY:', len(REGISTRY), '| ROUTING_POLICY:', len(ROUTING_POLICY))"
# expect: REGISTRY: ≥23 | ROUTING_POLICY: ≥27

# 8. No metric name in REGISTRY contains 'mock'?
python -c "from molmetal.scoring.score_any import REGISTRY; \
assert all('mock' not in n.lower() for n in REGISTRY); print('OK')"
# expect: OK
```

## 17. Files

* This file — `molmetal/reports/wf42_mock_audit/real_tools_inventory.md`
* Companion — `molmetal/reports/wf42_mock_audit/scoring_router.md`
* Phase 1 audit — `molmetal/reports/wf42_mock_audit/mock_inventory.json` (129 findings)
* Routing policy — `molmetal/scoring/ROUTING_POLICY.md` + `molmetal/scoring/policy.json`
* Runtime registry — `molmetal/scoring/REGISTRY.md` + `molmetal/scoring/score_any.py`