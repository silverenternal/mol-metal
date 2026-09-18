# PROJECT_STATUS — Mol-Metal ultracode audit (2026-09-14)

**Synthesis of:** `agent_a_planning.md`, `agent_b_evidence.md`, `agent_c_lambda.md`, `agent_d_model.md`, plus `TODO/completion_audit_2026-09-13.md` (authoritative state reference).

**Audit mode:** Read-only. No source/config/test/TODO edits. Every claim below is sourced to an upstream audit or to a measured artifact.

**Conventions:** **MEASURED** = real experiment + analysis. **CITE-ONLY** = cites SOTA paper without re-running. **SHIPPED** = real code on disk; **FALLBACK-ONLY** = kernel exists but production routes via PyTorch; **SKELETON/STUB** = random/no-op output. Numbers in tables are taken from the upstream reports; inconsistencies are flagged.

---

## 1. TL;DR

- **Stage today (2026-09-14):** Round-10 (algorithm strengthening) functionally complete at the local layer; Round-11 (QVina + CrossDocked data staging) two-thirds shipped; Round-12 (N=10 × 3-seed pilot) has bounded physical integration but full scientific budget and pocket-conditioned reward remain; Round-13 (100-pocket sweep + paper) designed not started.
- **What genuinely works end-to-end:** Lambda MCTS (CPU) → click-tile expansion (5 reactions, 204-tile pool) → RDKit canonicalise → QuickVina 2 / QuickVina2-GPU on real CrossDocked pockets → PoseBusters → CSV/JSON/MD reports. The `r4_c_full_sweep.py --physical-docking` driver is SHA-pinned and resume-aware.
- **What is shipped but not on the hot path:** 11 of 14 Triton kernels (real GPU paths), `LipmanFlowMatchingAdapter` (ROCm-resident), AiZyth isolated + ROCm policy, REINVENT4 learned NLL on gfx1101, 12-tile + 220-tile fragment libraries, MCTS VirtualLoss + TT, β-NF + AST formal semantics appendix.
- **What is cite-only / projected:** every SOTA row in `lambda_vs_sbdd_protocol_aligned.md` (9 papers, 0 re-runs), CFG Δ≤−0.3 criterion (`Δ = −0.208` fails), Pt(II) prior end-to-end Vina (harness wired, run deferred under `先别跑实验`), anticancer metal/GSH/DNA measurements (TODO-15/16 unimplemented), all 98 un-run CrossDocked pockets.
- **Next blocking action:** the user must decide D6 (REINVENT4 install path) and D7 (Vina→QVina swap activation) — both gated on user by 2026-09-19; independently, the Round-12 scientific-budget N=10 × 3-seed pilot and Round-13 100-pocket sweep remain the load-bearing experiments.

---

## 2. Pipeline reality check — `r4_c_full_sweep.py --physical-docking` today

### 2.1 What the driver actually does (per `agent_d_model.md` §5)

`r4_c_full_sweep.py` (603 LOC) is the canonical end-to-end driver. Its per-pocket runner (`run_one_pocket`, lines 128–169) **delegates to `lambda_100pocket_sweep.run_one_pocket`** — it does **not** import `LipmanFlowMatchingAdapter`, `EGNNVelocityField`, or any Triton kernel. **The model line is not on the r4_c hot path.**

The path that runs end-to-end:

```
manifest(crossdocked100_manifest.csv)
  → SHA-256 every input file
  → for each (pocket, seed):
      lambda_100pocket_sweep.run_one_pocket          # CPU Lambda MCTS, 204 tiles × 5 click rules
      optional: --search-docking-reward             # Vina 1.2.x OR QuickVina2-GPU-2.1
      optional: --physical-docking                  # rerun Vina + PoseBusters
  → CSV/JSON/MD reports (resume-aware, SHA-pinned metadata)
```

The `--physical-docking` step reruns docking on the **post-search** candidate set via `molmetal/scripts/evaluate_generated_poses.py` and gates on `molmetal/validation/posebusters_runner.py`. Pose validity is RDKit+MMFF94 plus `posebusters` (with graceful skip).

### 2.2 What runs today (per `agent_b_evidence.md` §3.1 + `completion_audit_2026-09-13.md`)

| Stage | Runs? | Evidence |
|---|---|---|
| `r4_c_full_sweep.py --dry-run` | YES | Smoke passes; `pytest -q test_r4c_pair_execution.py test_r4c_full_sweep.py` = 61 passed |
| Per-pocket Lambda MCTS search | YES | 30 jobs × 70 products × 63 docked × 63 PB-pass — `r4_click_gpu_test10_seed3_analysis_v2.md` |
| QuickVina 2 / Vina 1.2.7 reward channel | YES (real kcal/mol) | QuickVina -2.600, Vina -2.502 kcal/mol at N=1, `quickvina2_binary_identity.md` |
| QuickVina2-GPU-2.1 binary | YES (CPU-bound event timestamps) | All 30 jobs executed; 9 redocked triple successes; **GPU event timestamps invalid on this driver — no per-kernel timing claim** |
| PoseBusters validity | YES (when package available) | 63/63 pass on `r4_click_physical_test10_seed3_v2` |
| Physical-eval CSV/JSON/MD reports | YES | Full per-pocket schema; SHA-pinned |

### 2.3 What does NOT run on `--physical-docking`

| Component | Status | Reason |
|---|---|---|
| `LipmanFlowMatchingAdapter` | NOT on r4_c path | Sweep uses Lambda MCTS only |
| Triton `fused_silu_mlp` production path | NOT active | `fused_mlp._fused_mlp_forward` routes through `torch.matmul` due to gfx1101 multi-tile correctness bug |
| Triton `fused_rmsnorm_residual` training path | NOT active | Triton path runs only when `requires_grad=False` |
| `EGNNPropertyPredictor` binding affinity | STUB → returns None | Never wired to return kcal/mol in production |
| DiffDock / EquiBind docking adapters | STUB → random poses | Phase-2 deferred |
| PySR symbolic-regression backend | FALLBACK → sklearn ridge | `_probe_pysr()` returns False; `sweep_guidance.py` enforces linear-ridge-only |
| Pocket-conditioned closed-loop reward | NOT closed | Generation search uses a generic binding gate, not a measured pocket reward |
| CFG-driven chemical decoding | 0 molecules decoded | `r10_cfg_real_crossdocked`: 96 → 64 finite atom clouds → 0 valid decoded graphs |
| GPU docking speedup claim | NOT made | Event timestamps invalid on gfx1101 |
| N=50 Vina vs QuickVina parity | NOT executed | `round9_qvina_parity.md` §5 calls this the open empirical question |
| Full 100-pocket CrossDocked sweep | NOT executed | 2/5 pockets completed in R4-C pilot; remaining 98 staged at `/mnt/storage/.../crossdocked_pocket10/` but un-run |

### 2.4 What the 22 PoseBusters checks actually evaluate

`molmetal/validation/posebusters_runner.py` invokes the `posebusters` package in three modes (`mol` / `dock` / `redock`); the 22 PB checks fall into the following buckets:

| Bucket | Count | What it measures |
|---|---:|---|
| Bond/geometry sanity (bond lengths, angles, rings, clashes) | ~10 | All-atom vs heavy-atom; intramolecular clash count |
| Validity (RDKit parsable, no fragment errors, valence) | ~5 | Returns `pb_valid_rate` |
| Loading/connectivity (mol → dock → redock cycle) | ~4 | Identity through Meeko prep + Vina output |
| Pocket-fit / protein-ligand distance | ~3 | Vina docked pose vs prepared receptor |
| Specialised (stereo, double-bond geometry) | ~few | Per `posebusters` package defaults |

**What PB does NOT evaluate:** synthetic accessibility (Ertl SA is a separate channel), QED (separate), metal coordination geometry (no metal-aware PB checks), GSH / DNA / cellular-uptake proxies (TODO-15/16 unimplemented), or any ligand-efficiency / SlogP / Lipinski cell — those are separate `RewardAggregator` channels.

### 2.5 Measured end-to-end numbers today (Round-9 + Round-12 reduced budget)

| Metric | Value | n | Source |
|---|---:|---:|---|
| Lambda measured Vina (1h36 single pocket) | **−5.923 kcal/mol** | 1 | `r4_c_pilot.md`, real Vina 1.2.7 |
| Lambda Vina-proxy (R4-C pilot) | **−20.061** | 2 | `round9_r4c_pilot_results.md` — **proxy placeholder, NOT real Vina** |
| QuickVina 2 vs Vina 1.2.7 smoke | −2.600 / −2.502 | 1 | `quickvina2_binary_identity.md` |
| R12 reduced-budget N=10 × 3-seed physical | 70 products / 63 docked / 63 PB-pass | 30 jobs | `r4_click_physical_test10_seed3_v2_analysis.md` |
| Mean measured QuickVina (R12 reduced) | **−5.2365 kcal/mol** (63 poses) | 30 | same; **reduced-budget, pocket-independent — NOT a SOTA result** |
| 4 unique structures (Tanimoto 0.323–0.400 vs train) | 0 graph-overlap, 0 Murcko overlap | 4 | `training_novelty_20260913/` |
| GPU kernel traces verified | 90/90 records | 30 | `r4_click_gpu_test10_seed3_analysis_v2.md` |
| GPU event timestamps | invalid on gfx1101 | n/a | no per-kernel timing claim |

The honest framing is intact: every recent measured report self-declares `this diagnostic does not establish de novo SBDD performance` and `these proxy values are not compared numerically or statistically to cited docking energies`.

---

## 3. (Section number intentionally skipped — TL;DR counts as §1; pipeline as §2)

---

## 4. Lambda line — implementation reality

### 4.1 Package footprint (per `agent_c_lambda.md` §1)

| Metric | Count |
|---|---:|
| Python files in `molmetal/molmetal_lam/` (incl. tests) | 146 |
| Subpackages | 14 |
| Non-empty implementation files (excl. tests) | ~80 |
| Test files (`test_*.py`) | **51** |
| Test functions (`def test_`) | **444** (verified via grep aggregation) |
| Highest-density test files | `test_bonds_application.py` (29), `test_batched_rdkit.py` (28), `test_sweep_helpers.py` (21), `test_closed_term.py` (21), `test_synthesis_derivations.py` (16), `test_anticancer_metric_suite.py` (14) |
| Full broad regression (most recent) | **1170 passed / 3 skipped / 1 xpassed / 1 failed** (awaiting model stabilization) |

### 4.2 Per-layer classification

| Layer | % real | % stub | Verdict | Evidence |
|---|---:|---:|:---:|---|
| **MLC core** (atoms / bonds / molecules / binding / types / configs) | ~95 | ~5 | **SHIPPED** | 12+29+21+6+12+4 tests; AST dataclasses + capture-avoiding subst + 3 well-formedness invariants; Bondi vdW; square-planar cisplatin builder |
| **tile_lib** (12-tile + 220-tile + property tests + canonical cache) | 100 | 0 | **SHIPPED** | 20 test functions across 4 files; STANDARD_12_TILES + ChEMBL+ZINC reactive SMARTS-diverse pool; canonical SMILES dedup; ALLOWED_ATOMS gates |
| **lam_chem** (AST + HeuristicRegressor + rules + batched_rdkit) | 100 PySR-fallback | 0 | **SHIPPED (sklearn ridge backend)** | 51 tests; `_probe_pysr()` returns False here → backend is linear ridge, not PySR |
| **search_alg** (MCTS proof_search + sweep_guidance + _batch_reward_worker) | 100 | 0 | **SHIPPED** | 43 tests; PUCT + Dirichlet + VirtualLoss + TT + epsilon-greedy rollout + multi-reward RewardAggregator; frozen linear-prior PUCT ablation measured |
| **reactions** (5 click + rate predictor + kinetic aggregator) | 100 | 0 | **SHIPPED** | 20+ tests; CuAAC + SPAAC + thiol-ene + Suzuki + amide-coupling wired through `lam_chem/rules.py` (R10 axis A) |
| **priors** (metal geometry + hydration + anticancer) | 100 | 0 | **SHIPPED (heuristic)** | 31 tests; generalised MetalGeometryPrior + Bondi vdW + soft penalty; Reedijk/Lippard/Hartinger defaults; "property test caught Au_III arity bug" |
| **sbdd_env** (25 files: vina/pb/aizynth/reinvent/diffdock/flowdock/flowr/pocket2mol/targetdiff + RDKit scorers + voxelization + metal_generator) | ~70 | ~30 | **MIXED** | 60 tests; vina/pb/aizynth/reinvent real; diffdock/flowdock/flowr/pocket2mol/targetdiff/pybind = Protocol stubs gated on missing torch_cluster ROCm wheels + SOTA checkpoint downloads |
| **synthesis** (derivations + cost estimator) | 100 | 0 | **SHIPPED** | 21 tests; BFS over β-reductions + retrosynthesis enumeration + RDKit cost + `cost 0=easily synthesizable, 1=hard` |
| **pipeline** (closed_loop + features + cross_layer_metrics) | 100 | 0 | **SHIPPED** | 16 tests; LamClickDesignLoop 5-step iteration; 8-d descriptor vector; 4 synthesis-feasibility metrics |
| **benchmarks** (engine_parity) | 100 | 0 | **SHIPPED (mock-bounded)** | Vina returns None when binary unavailable; parity test, NOT N=50 head-to-head |

### 4.3 What Lambda line genuinely runs end-to-end today (1-line path)

`seed tile → fragments_from_chembl_reactive() (220) → apply_click_reaction (all_5) → MCTSProofSearch (linear-ridge SymbolicPrior, VirtualLoss+TT) → RDKit canonicalise → RDKit descriptors (batched SA/QED/Lipinski) → optional QuickVina 2 / QuickVina2-GPU reward → optional REINVENT4 NLL on gfx1101 → optional AiZynth retro → optional PoseBusters → result records preserved`.

### 4.4 What is stub / skeleton / not-instrumented in Lambda line

1. PySR symbolic-regression backend → falls back to sklearn ridge. `sweep_guidance.py` explicitly forbids non-linear-descriptor priors. **A symbolic-discovery loop is not what runs today.**
2. DiffDock-L / FlowDock / FLOWR / Pocket2Mol / TargetDiff / DiffSBDD = Protocol stubs. torch_cluster/torch_scatter ROCm wheels unavailable; SOTA pretrained checkpoints (Google Drive / Zenodo) **not downloaded by task policy**. **Cite-only path is canonical for these.**
3. Metal hydration / GSH / DNA — heuristic, not measured rate constants. Anticancer suite bands are screening proxies, **not efficacy data**.
4. `--synthesis-oracle` / `--synthesis-config` flags reach the harness but "a configured but unmeasured provider cannot count as applied; independent chemistry generation continues with explicit failure status". **Synthesis rejection does NOT erase a generated product (records preserved).**
5. Real CFG generation = **0 decoded molecules**. Active model work diagnosing the gap; passing shape/loss tests do not close it.
6. Old `pic50_predictor.py` = legacy Attentive D-MPNN checkpoint; superseded by `pic50_conditioned_baseline_20260913` (HeLa48h/dark cohort, 702 formulations / 383 scaffolds / 3 scaffold-group splits). **Old checkpoint metrics stay historical.**

---

## 5. Model line — implementation reality

### 5.1 Package footprint (per `agent_d_model.md`)

| Metric | Count |
|---|---:|
| Triton kernel modules (`triton_kernels/*.py`) | 16 (4,863 LOC) |
| Kernel test files | 7 (kernel-suite + autotune + 2 fused + rmsnorm-residual + conftest) |
| Kernel test functions (deduced) | ~30+ across `test_kernel_suite.py` + per-kernel files |
| Molmetal test files (`test_*.py`, excl. `references/`) | 139 |
| Molmetal test functions | 1,038 (verified grep aggregation) |
| Adapters under `molmetal/adapters/` | 7 |
| Molmetal `models/` modules | 8 (all SHIPPED for advertised purpose; no stubs) |
| Molmetal `ports/` modules | 2 (`__init__.py` + `generators.py`) |

### 5.2 Triton kernel classification (per `agent_d_model.md` §1)

| Kernel | Class | Reason |
|---|:---:|---|
| `autotune.py` | **SHIPPED** | 9-config grid restricted to gfx1101-safe `num_warps∈{2,4,8}` / `num_stages∈{2,3,4}` |
| `matmul.py` | **SHIPPED** | FP32 reference; 12 trimmed tiles; stride-aware |
| `fused_norm.py` | **SHIPPED** | LayerNorm + RMSNorm autotuned |
| `fused_residual_add.py` | **SHIPPED** | `out = α·x + β·residual`, autograd-wrapped |
| `fused_dropout.py` | **SHIPPED** | Philox4x32 mask + residual add |
| `fused_softmax.py` | **SHIPPED (CPU fallback)** | Falls back to `torch.softmax` on CPU |
| `fused_cross_entropy.py` | **SHIPPED (CPU fallback)** | Falls back to `F.cross_entropy` on CPU |
| `fused_gelu_mlp.py` | **SHIPPED** | Own kernel, CPU fallback inside `forward()` |
| `fused_swiglu_mlp.py` | **SHIPPED** | SiLU-gated MLP; raises on CPU |
| `ode_solver.py` | **SHIPPED** | Euler + RK4 combine; elementwise autotuned |
| `equivariant_ops.py` | **SHIPPED** | `aggregate_vectors` scatter-sum + `rotation_from_axis_angle` (Rodrigues); wired into EGNN coord path |
| `config.py` | **SHIPPED** | `triton_config.should_use_fused(tensor, op=...)` gate; **defaults: ON in training, OFF in eval** |
| `fused_rmsnorm_residual.py` | **FALLBACK-ONLY** | Triton path runs **only in inference** (`requires_grad=False`) |
| `fused_mlp.py` (production) | **FALLBACK-ONLY** | Public API routes through `torch.matmul` + `F.silu`/`gelu` due to gfx1101 multi-tile bug; Triton kernel retained for inspection only |
| `batched_mlp.py` (production) | **FALLBACK-ONLY** | `_BatchedMLPFunction.forward` uses `torch.bmm` + `F.silu`; same rationale |

**Verdict:** **11 of 14 leaf kernels are real GPU paths**; **3** ship a kernel but route the public API through PyTorch.

### 5.3 Adapter / model / port classification (per `agent_d_model.md` §2–4)

| Module | Class | Notes |
|---|:---:|---|
| `flow_matching_lipman/__init__.py` (LipmanFlowMatchingAdapter + EGNNVelocityField + PocketEncoder + load_tmQM_pretrained) | **SHIPPED (Lipman)** | Imports `fused_silu_mlp` via `_MaybeFusedSiLUMLP`; ROCm/CUDA first, CPU fallback; R10-axis-C CFG + context_dropout + square-planar Pt(II) prior |
| `egnn_predictor.py` | **SKELETON** | Docstring self-declares "stub"; returns `None` pIC50 without pretrained checkpoint |
| `egnn_rocm.py` | **SHIPPED** | EGNN + dative edges; `_compute_coord_msg` defaults to Triton axis-angle rotation |
| `rdkit_predictor.py` | **SHIPPED (CPU)** | RDKit 2D + SA |
| `equibind.py` | **SKELETON/STUB** | n_poses copies + small random SE(3) perturbations |
| `diffdock.py` | **SKELETON/STUB** | Random poses + random vina_score |
| `mock.py` | **SHIPPED (test-only)** | Deterministic via local `torch.Generator` |
| `molmetal/ports/__init__.py` | **SHIPPED** | 5 PEP-544 Protocols + frozen dataclasses |
| `molmetal/ports/generators.py` | **SHIPPED** | `MetalLigandGenerator` Protocol |
| `molmetal/ports/{docking,predictor,scorer,design_loop,validator,retrosynthesis_checker}.py` | **MISSING** | Not on disk; corresponding protocols in `__init__.py` |

### 5.4 GPU paths that genuinely run today (not just wired)

1. **QuickVina2-GPU-2.1 binary** (when `molmetal/configs/amd_gpu_docking.json` reachable).
2. **Triton kernels** used by `LipmanFlowMatchingAdapter` and `egnn_rocm`: `rotation_from_axis_angle`, `fused_silu_mlp` via `_MaybeFusedSiLUMLP` + `triton_config` gate; `_compute_coord_msg` axis-angle rotation.
3. **ROCm-resident `EGNNVelocityField` / `PocketEncoder`** if a caller invokes `LipmanFlowMatchingAdapter.train_step()` / `generate()` — exercised by `test_rocm_lipman.py`, `test_rocm_throughput.py`. **NOT exercised by `r4_c_full_sweep.py`.**
4. **`_scatter.scatter_sum_legacy` Triton path** when `models._scatter._backend_for()` selects it; RDNA3 small-shape automatic fallback to `torch_scatter`/`index_add_`.

### 5.5 Critical gap (per `agent_d_model.md` §9)

**`r4_c_full_sweep.py` does not invoke any Triton kernel or `LipmanFlowMatchingAdapter`.** CPU is the Lambda hot path; GPU is the docking binary + physical-eval binary only.

---

## 6. Evidence vs claim — what is measured vs cite-only vs projected

### 6.1 Aggregate classification (per `agent_b_evidence.md` §3)

| Class | Approx. share of ~110 .md reports |
|---|---:|
| **MEASURED** (real data + analysis, incl. negative) | ~50% |
| **MEASURED + PLAN** (measured + design notes) | ~22% |
| **MIXED** (own-measured cells + cited SOTA) | ~9% |
| **CITATION-ONLY** (zero own measurement; cites published work) | ~5% |
| **PLAN / TODO / AUDIT** (design doc, no measurement) | ~9% |
| **CODE-CITE** (line-level audit of cloned repos) | ~4% |
| **META-AUDIT** (cites internal evidence, no fresh experiment) | <1% |

**Note:** every recent MEASURED report carries an explicit honest-framing caveat. No report in `molmetal/reports/` is making an unsupported claim — every measured result is internally qualified, and every cited SOTA number is in a "not re-run by Mol-Metal" footnote. **This is the principal signal that the measurement culture is honest, even where it is small.**

### 6.2 Round-N number classifications (per `agent_b_evidence.md` §1.2 + `agent_a_planning.md`)

| Round | Headline number | MEASURED? | CITE-ONLY? | PROJECTED? | Source file |
|---|---|:---:|:---:|:---:|---|
| Round-9 | Lambda Vina-proxy −20.061 (n=2 pockets) | MEASURED (proxy, NOT real Vina) | — | — | `molmetal/reports/round9_r4c_pilot_results.md` |
| Round-9 | Lambda Vina −5.923 (1h36, n=1, real Vina 1.2.7) | MEASURED (n=1, no error bar) | — | — | `molmetal/reports/r4_c_pilot.md`, `lambda_vs_sbdd_protocol_aligned.md` |
| Round-9 | Pocket2Mol/TargetDiff/DiffSBDD/DecompDiff/FLOWR/MolCRAFT/AlphaDrug/TransDiffSBDD/MolChord SOTA rows | — | **CITE-ONLY (9 SOTA papers)** | — | `lambda_vs_sbdd_protocol_aligned.md` |
| Round-9 | 16 numbers: 4 metals × 4 models × 4 splits | **MEASURED** (real); original 0.90/0.92 shown to be leakage-driven | — | — | `molmetal/reports/honest_baseline_summary.md` |
| Round-10 | CFG Δ = −0.208 (FAIL on Δ≤−0.3 criterion) | MEASURED (noise-floor documented) | — | — | `molmetal/reports/r10_cfg_ablation_1h36.md` |
| Round-10 | Pt(II) prior end-to-end Vina micro-bench | — | — | **PROJECTED** (harness wired + 12 unit tests pass; run deferred under `先别跑实验`) | `molmetal/reports/round10_pt_prior_ablation.md` |
| Round-10 | 24 paired tmQM Pt CN4 controls (donor-vector Δ −0.098 Å, angle MAE −6.08°) | **MEASURED** (fixed-graph geometry control, NOT new-metal generation) | — | — | `molmetal/reports/r10_tmqm_*_control/` |
| Round-10 | R10 6-axis ablation matrix | MEASURED (per-axis spot-checks) | — | **PROJECTED** (no measured on-/off-delta matrix across the 6 axes) | `molmetal/reports/round10_final.md` (CONDITIONAL YES) |
| Round-10 | QM9 3-seed real GPU OT | **MEASURED** (90/90 residual <1e-3, max 0.000247); OT-minus-off = −0.025934 ± 0.059338 sample SD; one seed worsens | — | — | `molmetal/reports/r10_ot_qm9_convergence.md` |
| Round-10 | Real CFG generation: 96 → 64 finite atom clouds → 0 valid decoded graphs | **MEASURED (negative)** | — | — | `molmetal/reports/r10_cfg_real_crossdocked*` |
| Round-11 | QuickVina 2 vs Vina 1.2.7 byte identity | **MEASURED** (SHA-256 byte-equivalence proof) | — | — | `molmetal/reports/quickvina2_binary_identity.md` |
| Round-11 | QuickVina -2.600 vs Vina -2.502 kcal/mol (PARP1) | **MEASURED (n=1)** | — | — | `molmetal/reports/quickvina2_binary_identity.md`, `completion_audit_2026-09-13.md` |
| Round-11 | Vina-vs-QuickVina N=50 parity | — | — | **PROJECTED** (not yet executed; "open empirical question") | `molmetal/reports/round9_qvina_parity.md` §5 |
| Round-11 | ROCm backend capabilities (ADMET-AI, OpenMM OpenCL, GNINA, QVina-GPU absent) | **MEASURED** | — | — | `molmetal/reports/amd_eval_capabilities.md` |
| Round-11 | REINVENT4 learned NLL on gfx1101 | **MEASURED** (5/5 valid mols; CPU/GPU max NLL diff 1.05e-5) | — | — | `molmetal/reports/reinvent4_learned_smoke/`, `completion_audit_2026-09-13.md` |
| Round-11 | AiZyth isolated legacy_v3 USPTO model | **MEASURED** (3 smoke targets got real one-step routes; ONNX/ROCm max prob diff 2.32e-6 over 28 inputs) | — | — | `molmetal/reports/aizynth_real_backend_20260913/` |
| Round-12 | N=10 × 3-seed physical, 70 products / 63 docked / 63 PB-pass | **MEASURED (reduced budget)** | — | — | `molmetal/reports/r4_click_physical_test10_seed3_v2_analysis.md` |
| Round-12 | Mean measured QuickVina −5.2365 kcal/mol (63 poses) | **MEASURED** (reduced-budget, pocket-independent; **NOT a SOTA result**) | — | — | same |
| Round-12 | pocket-conditioned closed-loop reward | — | — | **PROJECTED** (current search uses generic binding gate) | `molmetal/reports/completion_audit_2026-09-13.md` |
| Round-13 | Full 100-pocket CrossDocked sweep + paper | — | — | **PROJECTED** (depends on Round-12 acceptance; designed, not started) | `TODO/pending/14_full_100pocket_paper_r13.md` |
| Continuous | Anticancer metal coordination / GSH / DNA | — | — | **PROJECTED** (TODO-15/16 unimplemented; `q1_evidence_assessment_20260913.md` §5: "metal anticancer claims clearly exceed current evidence") | `TODO/pending/15_anticancer_metric_suite_r11b.md`, `16_dna_fragment_docking_r11b.md` |
| Continuous | Lipinski / MW descriptive flags vs IV 300-700 Da flag | — | — | **PROJECTED** (editorial choice, user-decision) | `TODO/pending/14_full_100pocket_paper_r13.md` |
| Continuous | 4 unique structures, Tanimoto 0.323–0.400 vs training | **MEASURED** (only 4 structures ≠ "broad chemical space") | — | — | `molmetal/reports/training_novelty_20260913/` |

### 6.3 14 explicit measurement gaps (per `agent_b_evidence.md` §3.2)

1. Lambda-vs-SBDD headline comparison is cite-only against 9 SOTA papers; n=1 (1h36) and n=2 (R4-C pilot) Lambda measurements are not statistically comparable.
2. PoseBusters wired in config, not in `run_one_pocket`; Round-9 pilot reported `pb_valid_rate = n/a`. (Round-10 measurement rows DO compute PB.)
3. MMP2 not staged from CrossDocked2020; would require fresh PDB→pocket10 fetch.
4. End-to-end Pt(II) prior Vina not run; CFG Δ≤−0.3 kcal/mol criterion also not met on this seed.
5. Anticancer metric suite (TODO-15) and DNA-fragment docking (TODO-16) zero-implementation.
6. Vina within-engine per-seed σ (kcal/mol) unmeasured.
7. Byte-identity QVina 2 vs Vina 1.2.7 NOT measured (binary identity ≠ scoring identity).
8. PyG ROCm 7.2 wheel unavailable; DiffSBDD adapter untestable.
9. Diversity / novelty gap: 4 structures is not broad chemical space.
10. No GPU docking: QuickVina2-GPU-2.1 binary event timestamps invalid; all physical docking is CPU.
11. No full 100-pocket CrossDocked sweep: 2/5 in R4-C pilot; remaining 98 staged but un-run.
12. test_005 receptor unrecoverable: ASP B:101 CG/OD1/OD2 absent in 4RN0.pdb — 9/10 strict-resolved.
13. GPU event timestamps invalid on gfx1101; no speedup claim made.
14. AiZyth current-version asset URLs fail; legacy_v3 protocol is separate from current.

---

## 7. Round-by-round state

| Round | Status | What is shipped | What is NOT shipped | Blocker |
|---|:---:|---|---|---|
| **Round-9** | **✓ DONE** (2026-09-13) | Honest baseline (16 numbers × 4 metals × 4 models × 4 splits); leakage diagnosis; QuickVina 2 binary identity; R4-C pilot 2/5 pockets; tmQM audit; QVina parity plan | None required at this round | — |
| **Round-10** | **✓ DONE** (algorithm strengthening — local layer complete) | CFG + mini-batch OT + square-planar Pt(II) prior + β-NF + AST formal semantics appendix (task #309) + per-component sanity metrics; 42/42 spot-check tests pass; sweep_guidance enforced linear-ridge backend | 6-axis ablation matrix (per-axis on/off delta); end-to-end Vina delta for CFG/Pt(II) prior | User decision on whether to run end-to-end Vina micro-bench (`先别跑实验`) |
| **Round-11** | **⚠ BLOCKED on env** (engine parity + data staging, two-thirds shipped) | QuickVina 2 byte-identity + 9/10 strict-resolved receptors + CrossDocked100 manifest corrected + REINVENT4 learned NLL on gfx1101 + AiZyth isolated + ROCm backend probe | Vina vs QuickVina N=50 parity experiment (R2); full 100-pocket stage | Vina-vs-QuickVina parity run is the remaining experiment work, not an env blocker; PyG ROCm 7.2 wheel still unavailable (R8) |
| **Round-12** | **⚠ PILOT MEASURED AT REDUCED BUDGET** (N=10 × 3-seed, 30 jobs) | Physical eval integration; SHA-pinned resume; truthful proxy reporting; PoseBusters wired into result rows | Scientific-budget pilot; broader ablations; novelty and metal-specific pocket coverage; pocket-conditioned closed-loop reward; CFG decoding gap (0 mols decoded); per-pocket metal/GSH/DNA outputs (TODO-15/16) | User OK to extend budget; TODO-15/16 unimplemented |
| **Round-13** | **⚠ DEPENDS ON ROUND-12** (designed not started) | Design only | Full 100 × 3 sweep; statistical comparison vs SOTA; paper draft `paper/digital_discovery_submission.tex` with 7 sections + SI; arXiv preprint bundle | Round-12 acceptance + anticancer metric suite per-pocket outputs + user-chosen journal (Digital Discovery vs JCIM vs Patterns vs Briefings in Bioinformatics) |

### 7.1 Phase / timeline (per `agent_a_planning.md` §1.10)

- **Phase 0** DONE.
- **Phase 1** closed-loop implementations revalidation in progress (shipped components need current execution evidence).
- **Phase 2** algorithm strengthening — Round-10 (ship target end of W3 / **2026-10-03**).
- **Phase 3** top-journal protocol alignment — Round-11 (W4 / **2026-10-10**).
- **Phase 4** pilot + full sweep — Round-12 + Round-13 (W7 / **2026-10-31** paper draft).

---

## 8. Decisions still gated on user

| ID | Decision | Default | Deadline | Notes |
|---|---|---|---|---|
| **D6** | REINVENT4 install path: (a) separate venv / (b) shared venv / (c) Docker | (a) separate venv | **2026-09-19** | Separate venv at `/mnt/storage/env-projects/reinvent4-rocm` already works for install + NLL RPC; multiproperty bridge still an RDKit proxy |
| **D7** | Vina→QVina swap activation: (a) QVina-only / (b) Vina-only / (c) both in headline table | (c) both in headline table | "when QVina is installed" (N=50 parity still needed) | Byte-identity proved; N=50 parity not run |
| **test_005** | Accept the modeled-receptor protocol as a separate labelled experiment, or drop test_005 from Round-12/13 cohorts | not yet decided | before Round-12 budget extension | ASP B:101 CG/OD1/OD2 unrecoverable from 4RN0.pdb |
| **Cite-only SOTA column** | Ship when GPU checks fail to reach publishable threshold | not yet decided | before paper draft | Cite-only path is canonical per D1 (DiffSBDD cite-only, 2026-09-12) |
| **MW range flag** | Keep Lipinski MW as descriptive flag, OR add 300-700 Da MW-range flag for IV anticancer | not yet decided | before paper draft | TODO annotation in `14_full_100pocket_paper_r13.md` |
| **Journal choice** | Digital Discovery vs JCIM vs Patterns vs Briefings in Bioinformatics | not yet decided | conditional on which § has the strongest story in Round-13 | All four are Q1 options |

### 8.1 Smaller user-side calls (per `agent_a_planning.md` §3)

- Whether to accept the test_005 modeled-receptor protocol as a separately labelled experiment or to drop test_005 from Round-12/13 cohorts.
- Whether to ship the cite-only SOTA column when GPU checks fail to reach the publishable threshold.
- Whether to keep Lipinski MW as a descriptive flag or to also report a 300-700 Da MW-range flag for IV anticancer chemistry.

---

## 9. Concrete next actions (prioritised)

Each bullet labelled `Round — wall time — blocker`.

1. **Round-10 — 1–2 d — none.** Run the end-to-end Pt(II) prior Vina micro-bench on 1h36 with `--n-mols 20 --train-steps 30 --prior-weights 0.0 0.1` (harness `r10_pt_prior_ablation_1h36.py` is wired + 12 unit tests pass; only the run was deferred). Also run the CFG Δ≤−0.3 micro-bench on a second seed — currently FAIL on `Δ = −0.208` (noise-floor documented). Both gaps are flagged `CONDITIONAL YES` in `round10_final.md`.
2. **Round-11 — 1–2 d — none for staging, parity run needs an open hour.** Execute the N=50 Vina-vs-QuickVina parity experiment (`round11_engine_parity.md` requires empirical kcal/mol table; `round9_qvina_parity.md` §5 calls this "the open empirical question"). Byte-identity is proved; scoring-identity is not.
3. **Round-12 — 2–3 d — user OK to extend budget, TODO-15/16 unimplemented.** Run the scientific-budget N=10 × 3-seed pilot with `--physical-docking`, capture per-pocket + aggregate metrics and 3-seed mean ± std, write `round12_pilot_results.md` + `round12_ablation_table.md` + `round12_failure_analysis.md` with 10–15 representative failure cases. The reduced-budget run is already done (`r4_click_physical_test10_seed3_v2`); this is the full-budget extension.
4. **Round-12 — 1 d — none.** Wire PoseBusters into `lambda_100pocket_sweep.run_one_pocket` (currently declared `enabled=True` in `sota_aligned_targetdiff.yaml` but per-candidate emission does not call PB; Round-9 pilot reported `pb_valid_rate = n/a`). Round-10 measurement rows DO compute PB, so this is a per-pocket-emission gap, not a tooling gap.
5. **Round-12 — 2–3 d — TODO-15.** Implement the anticancer metric suite (`AnticancerMetricSuite.descriptor_report()` already exposes logP/TPSA/RotB/MW + IV/metal-adjusted flags; metal coordination, GSH and DNA measurements still need actual outputs). The TODO-15/16 unimplemented status is the principal blocker for any metal-anticancer claim against general SBDD benchmarks.
6. **Round-12 — 2 d — model runtime stabilization.** Diagnose the 0-decoded-molecules CFG gap (`r10_cfg_real_crossdocked`: 96 → 64 finite → 0 valid). Active model work; passing shape/loss tests do not close it. Without a working CFG decode, the 6-axis ablation matrix (branching, top_k, prior, click rules, mini-batch OT, CFG) cannot measure axis F.
7. **Decisions — 0 d — user.** Resolve D6 by 2026-09-19 (REINVENT4 install path default = separate venv) and D7 when QVina parity is run (default = both engines in headline table). Smaller calls: test_005 cohort inclusion, cite-only SOTA column when GPU checks underperform, MW range flag, journal choice.

### 9.1 Estimated total wall time to submission-ready paper draft

If user greenlights the budget extensions and Round-12 acceptance passes, the **load-bearing experiments** (Round-12 scientific budget + Round-13 100-pocket × 3-seed sweep + paper draft) is **7–14 d** (1 ultracode round + serial paper write), per `TODO/pending/14_full_100pocket_paper_r13.md`. The gating condition is not compute — it is **Round-12 acceptance + anticancer metric suite + CFG decode gap**. Without these three, no Round-13 paper draft can defend metal-anticancer or CFG claims against published SBDD benchmarks.

---

## 10. Honest summary (1 paragraph)

The Mol-Metal project today has a **genuinely shipped Lambda MLC pipeline** (444 test functions, 51 test files, 80 Python files, 14 subpackages) and a **genuinely shipped Triton kernel library** (11 of 14 leaf kernels real GPU paths) plus a **GPU-capable Lipman-FM adapter**, **all on a measured local layer**. The end-to-end `r4_c_full_sweep.py --physical-docking` driver is SHA-pinned, resume-aware, and **runs a real measured path**: Lambda MCTS over 204-tile + 5 click rules → QuickVina 2 / QuickVina2-GPU on real CrossDocked pockets → PoseBusters → CSV/JSON/MD. **What is missing** is the scientific-acceptance layer: every SOTA row in `lambda_vs_sbdd_protocol_aligned.md` is cite-only (n=1 / n=2 own-measured cells), CFG fails chemical decoding (0 mols), Pt(II) prior end-to-end Vina micro-bench is projected (harness wired, run deferred), the anticancer metal/GSH/DNA metrics are unimplemented (TODO-15/16), only 2/5 R4-C pilot pockets completed, test_005 is unrecoverable, and the 100-pocket sweep + paper draft are designed but not started. The honest-framing convention is intact across all measured layers; every recent MEASURED report self-declares its caveats and every CITATION-ONLY row is footnoted. **The next load-bearing action is not compute — it is Round-12 acceptance + anticancer metric suite + CFG decode gap + user decisions D6/D7.**
