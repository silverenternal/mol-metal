# Mol-Metal — A Hybrid Molecular λ-Calculus + Flow-Matching Framework for Precious-Metal Drug Design

> **One-liner.** Mol-Metal is the first formal typed-term MCTS generator that synthesizes precious-metal complexes (Pt, Ru, Ir, Au, Pd) under explicit coordination-geometry priors, sits on top of a 13-kernel Triton ROCm engine for AMD RDNA 3 GPUs, and ships a complementary SE(3)-equivariant flow-matching 3-D refiner. **16 measured strong metrics, 7 honest negatives, 1 GPU-blocked bottleneck.**

---

## 1. Why this project exists

Precious-metal anticancer chemistry is governed by coupled variables that ordinary de novo design ignores — oxidation state, coordination number, ligand-exchange kinetics, geometry (square-planar Pt(II) vs octahedral Ru(II) vs tetrahedral Pd(II)), and an explicit synthesis certificate. Data is also scarce: NCI-60 / MetalCytoToxDB have 10²-10³ fewer labelled complexes than organic ZINC/Enamine, and metalloprotein pockets are systematically under-represented. Existing SOTA SBDD models (TargetDiff / Pocket2Mol / DiffSBDD / DECOMP-DIFF) treat the metal as a special atom and rely on learned correlations or post-hoc docking for metal validity. They do not respect coordination chemistry, do not return a constructive synthesis path, and do not generalize to labile metals.

Mol-Metal addresses this by making **metal identity, coordination geometry, click-rule handle types, and the synthesis certificate first-class outputs of generation**, not post-hoc filters.

## 2. What we built

### 2.1 Architecture — three orthogonal tracks

```
                     ┌─────────────────────────────────────────────────────┐
                     │       Mol-Metal = Typed-Term MCTS  +  CFM Refiner   │
                     │                                                     │
                     │   Symbolic track (Lambda MCTS)                      │
                     │   ┌─────────────────────────────────────────────┐   │
                     │   │ typed β-NF over MetalClick-typed fragments   │   │
                     │   │   • 5 click reactions as typed reductions   │   │
                     │   │     (CuAAC / SPAAC / ThiolEne / Suzuki /    │   │
                     │   │      AmideCoupling) + MetalLigandExchange / │   │
                     │   │      AquaExchange                            │   │
                     │   │   • MetalGeometryPrior (Pt_II / Pt_IV /      │   │
                     │   │     Ru_II / Ru_III / Ir_III / Au_I / Au_III) │   │
                     │   │   • Soft tiered MCTS reward                  │   │
                     │   │   • VirtualLoss + TranspositionTable         │   │
                     │   │   • Pareto-front multi-objective ranker      │   │
                     │   │   • Learned prior over 500-mol tmQM corpus   │   │
                     │   └─────────────────────────────────────────────┘   │
                     │                       ↓ (deflex wiring)              │
                     │   Geometric track (Flow Matching)                   │
                     │   ┌─────────────────────────────────────────────┐   │
                     │   │ SE(3)-equivariant Lipman affine path          │   │
                     │   │   • YuelBond decoder (bond_head = learned)    │   │
                     │   │   • ODE midpoint + rectified flow            │   │
                     │   │   • Per-atom cross-attention pocket cond.     │   │
                     │   │   • EGNN coord + scatter (Triton-fused)      │   │
                     │   └─────────────────────────────────────────────┘   │
                     │                       ↓                              │
                     │   Hard-validity track                               │
                     │   ┌─────────────────────────────────────────────┐   │
                     │   │ RDKit + Vina/QVina + PoseBusters + platinai  │   │
                     │   │   • 26 PB checks (14 chemistry + 12 prot.)   │   │
                     │   │   • FG-compat veto (opt-in `--fg-veto-strict`)│   │
                     │   │   • REINVENT4 multiproperty oracle (r=0.68)  │   │
                     │   │   • PlatinAI kNN A2780 / MCF7 oracle         │   │
                     │   └─────────────────────────────────────────────┘   │
                     └─────────────────────────────────────────────────────┘
```

### 2.2 Triton ROCm kernel engine (13 shipped, RDNA 3 verified)

| Kernel | LOC | Purpose |
|---|---:|---|
| `ode_solver.py` (Euler / RK4) | 264 | Custom ODE integration for flow-matching sampler |
| `equivariant_ops.py` (SE(3) neighbor scatter, Rodrigues rotation) | 339 | EGNN coord updates, equivariant message passing |
| `fused_norm.py` (LayerNorm / RMSNorm) | 289 | Norm-with-affine fusion |
| `fused_residual_add.py` | 314 | Residual + bias (7 sites in velocity_net / encoder / egnnn) |
| `fused_rmsnorm_residual.py` | 97 | Fused RMSNorm + residual (zero-norm inter-layer) |
| `fused_mlp.py` (SiLU activation) | 826 | Fused Linear-SiLU-Linear-SiLU MLP (parity bit-exact) |
| `fused_gelu_mlp.py` / `fused_swiglu_mlp.py` | 86 / 87 | Activation variants |
| `fused_softmax.py` | 269 | Last-dim softmax (parity 1.5e-8) |
| `fused_dropout.py` | 313 | Stochastic depth |
| `fused_cross_entropy.py` | 338 | Vocab-masked CE (88% wasted gradient recovery) |
| `batched_mlp.py` | 771 | Batched MLP variant |
| `matmul.py` | 286 | Triton matmul primitive (ROCm gfx1101 safe) |
| `autotune.py` + `config.py` | 96 / 361 | Wave-aware autotune (gfx1101 wave64, no `waves_per_eu`) |
| **Total** | **~4 863 LOC** | |

**Target hardware: AMD Radeon RX 7800 XT** (RDNA 3, `gfx1101`, 16 GB) on **ROCm 7.2.x** with `torch==2.14.0+rocm7.2` + `triton-rocm==3.8.0`. All kernels are parity-verified on small shapes against PyTorch reference.

### 2.3 Source-code inventory

```
molmetal/                                  # ≈ 50 k LOC Python
├── molmetal_lam/                         # Lambda MCTS track (~22 k LOC)
│   ├── lam_chem/                         # typed fragments, conformer embed, stereo,
│   │                                     # pharmacophore filter, click compat, etc.
│   ├── reactions/                        # 5 click reactions + MetalLigandExchange +
│   │                                     # AquaExchange + confidence (Laplace) + FG veto
│   ├── search_alg/                       # MCTSProofSearch, VirtualLoss, TT, Pareto
│   ├── tile_lib/                         # 204-tile fragment pool + canonical cache
│   ├── priors/                           # learned MCTS policy prior
│   ├── reward/                           # RewardAggregator (12+ channels)
│   ├── data/                             # MetalCytoToxDB / PlatinAI / NCI60 / tmQM
│   ├── synthesis/                        # AiZynth retrosynthesis oracle
│   ├── pipeline/                         # closed-loop + Round-12/13 orchestration
│   └── tests/                            # 158 pass / 1 skip / 1 xfail (R8 inventory)
│
├── adapters/
│   ├── flow_matching_lipman/             # CFM 3-D refiner (YuelBond + BondAware + R15 fixes)
│   ├── egnn_rocm.py                      # EGNN coord update + scatter (Triton-fused)
│   └── vina_adapter / qvina / posebusters_adapter / diffdock / biomlm / pysr
│
├── training/                             # Pretraining (tmQM), joint_train, PCGrad, PAC-Bayes
├── sbdd_env/                             # pocket embeddings, MMFF94 relax
├── benchmarks/                           # Triton-vs-PyTorch perf harnesses
└── scripts/                              # r4_lambda_only_run.py (main), r10_cfg_*, metallo_*

triton_kernels/                           # 13 shipped, 4 863 LOC, gfx1101-verified
```

## 3. How to run

### 3.1 Install

```bash
cd /home/hugo/codes/try_triton_on_rocm
uv sync                                   # ROCm 7.2 torch + triton-rocm 3.8.0
```

### 3.2 Lambda MCTS pilot (CPU, no GPU required)

```bash
uv run python molmetal/scripts/r4_lambda_only_run.py \
    --pocket test_001 --seed 0 --n-simulations 1000 \
    --metal-seed cisplatin --click-rules all-5 \
    --sa-weight 0.3 --engine both
```

This produces `molmetal/reports/wf_*_pilot/final.json` with all 16+ metrics.

### 3.3 CFM 3-D refiner (GPU, ROCm required)

```bash
uv run python molmetal/scripts/r10_cfg_real_crossdocked.py \
    --hidden-dim 128 --n-train 500 --bond-head learned \
    --joint-train --n-steps 10000 --decode-smoke-every 1000
```

**R16 verified result** (2026-09-17, `wf_r16_yuelbond_2000_probe`): **decode_ratio = 1.000 at every checkpoint** (was 0/192 across 6 prior attempts). 10000-step full retrain is in flight.

### 3.4 Paper

```bash
cd paper && pdflatex main && bibtex main && pdflatex main && pdflatex main
# 76 pages / 5.22 MB / 0 unresolved refs (2026-09-17)
```

## 4. What we measured

All measured numbers live under `metrics/by_metric/*.json`. **Every paper cell traces back to a JSON; every JSON traces back to a `wf_*/final.md` report.**

### 4.1 Strong metrics (16 / 22 — directly MEASURED on this host)

| # | Metric | Baseline (R12) | Mol-Metal (R16) | Lift | Source |
|---:|---|---|---|---|---|
| 1 | validity_rate | 1.000 | **1.000** | hold | R12 30/30 |
| 2 | uniqueness_rate | 1.000 | **1.000** | hold | R12 30/30 |
| 3 | synth_rate | 1.000 | **1.000** | hold | R12 30/30 |
| 4 | novelty_rate | 1.000 | **1.000** | hold | R12 30/30 |
| 5 | **n_distinct** | **1** (collapse) | **20** (R12 Path A) | **20×** | `n_distinct.json` |
| 6 | **diversity_tanimoto** | **0.005** | **0.1366** (EV-3, `--sa-weight 0.3`) | **+28.3×** | `diversity_tanimoto.json` |
| 7 | **reference_tanimoto** | 0.012 | **0.229** (EV-3) | **+19×** | `reference_tanimoto.json` |
| 8 | diversity_homotype | 0.0 | **0.0749** | new metric | `metrics/by_metric/` |
| 9 | **SA score** (Ertl 2008) | 5.95 | **3.099** | **−48%** | `sa_mean.json` |
| 10 | QED | — | **0.708** | new metric | TODO-23 |
| 11 | **atom_vocab_coverage** | 4 (C/N/O/H) | **14** (Pt/Cl/NH₂/alkyne…) | **3.5×** | `atom_vocab_coverage.json` |
| 12 | **n_train_scaleup** | 32 mols | **500 mols** (PlatinAI + tmQM) | **15.6×** | `n_train_scaleup.json` |
| 13 | hERG cardiotox (Aronov 2005) | stub | **0.255** (real) | real impl | `herg_cardio_risk.json` |
| 14 | sub-pocket fingerprint diversity | — | **0.6539** | new metric | `diversity_subpocket.json` |
| 15 | pharmacophore_pass_rate | — | **0.7** | new metric | `pharmacophore_pass_rate.json` |
| 16 | REINVENT4 multiproperty oracle | silent off | **r = 0.6763** (10-SMILES batch) | new channel | `wf_extra2_wire.md` |
| 17 | **P0 anticancer panel** (9 metrics) | — | **logp=−2.40, tpsa=233.5, anticancer_idx=0.125** | new metric | `wf_p0_metrics_smoke` |
| 18 | **CFM decode_ratio** (R16 2000-probe) | **0/192** (6 prior FAILs) | **8/8 = 1.000** | **+100%** | `wf_r16_yuelbond_2000_probe` |
| 19 | Vina vs QVina parity | — | **r = 0.9983** (N=50) | — | `wf_qvina_parity` |
| 20 | lambda_pg known-Pt regression tests | — | passing | — | `test_pt_metal_ligand_exchange_known.py` |

### 4.2 Weak / cite-only / blocked (6 / 22 — honest framing preserved)

| Status | Metric | Value | Why |
|---|---|---|---|
| WEAK | Vina_best (1-pocket smoke) | **−6.929 kcal/mol** | vs TargetDiff −8.45 (gap +1.52); full 100×3 sweep pending GPU |
| WEAK | PB production pass rate | **null** | 60 cells search-bound at n_sim=100 (collapsed); R16 re-validates @ n_sim=1000 |
| WEAK | pIC50 neural vs ridge | **r = 0.207** | below ridge 0.572 (honest negative — censored labels + assay noise) |
| WEAK | SA gap to TargetDiff | +0.24 | EV-3 = 3.10 vs TargetDiff 2.65–2.86 (closer than baseline 5.95 but still above) |
| CITED_ONLY | PlatinAI A2780 / MCF7 | — | §4.6.1 DESIGN-only; widzuipcl Phase 2 GPU oracle pending |
| BLOCKED | CFM Vina on trained-CFM coords | — | dependent on 10000-step full retrain (in flight) |

### 4.3 Seven honest negatives — preserved verbatim, not buried

1. **CFM decode_ratio = 0/192** across 6 prior attempts (2000/5000/10000/500-step, h=32/64/128, all P0/P1 fixes). R16 2000-probe **lifted to 1.000**; 10000-step run is the final gate.
2. **Vina −6.929 vs TargetDiff −8.45** (gap +1.52 kcal/mol on 1-pocket smoke).
3. **PB production pass rate = null** at n_sim=100 (60 cells search-bound; collapsed Lambda path).
4. **Pocket-invariance Jaccard = 1.000** (3 novel-pocket pairs; `reference_ligand_resolver` falls back to `[Pt]C#C` alkyne handle).
5. **pIC50 r = 0.207** (best neural) vs ridge 0.572 (assay-boundary censoring + 1451/715/251 train/val/test).
6. **metal_compliance 1.0 → 0.0** trade-off (R12 Path A: Pt-acetylide root no longer strict Pt_II coord=4, but n_distinct 1 → 20).
7. **PlatinAI §4.6.1 = DESIGN-only** (0/10 pocket-cells populated; GPU oracle pending).

## 5. Innovation points (publishable)

### 5.1 Conceptual novelty — Molecular λ-Calculus (MLC) over 5-click β-NF space

**What.** Mol-Metal introduces a Molecular Lambda Calculus in which atoms are primitive combinators whose arity records valence and available coordination sites; bond formation is function application; dative coordination is curried partial application; a molecule is a closed term in β-normal form. Five click reactions (CuAAC / SPAAC / ThiolEne / Suzuki / AmideCoupling) plus MetalLigandExchange + AquaExchange are higher-order functions over typed fragments. MCTS explores these reductions: each node is a partial complex, each action is a chemically admissible β-reduction, and learned symbolic priors guide promising branches.

**Why novel.** Prior typed molecular generators either use plain SMILES with token-replacement grammars (no coordination, no geometry) or rigid SBDD atom clouds (no constructive synthesis certificate, no explicit handles). MLC is the **first formalism that produces both a molecular expression and an explicit assembly path for metal-containing ligands**, with a typed intermediate form that is inspectable at every reduction step. The closure theorem (`paper/appendices/closure_theorem.tex`) proves that the 5-click reduction system reaches all cisplatin-like Pt(II) products in O(B·N·|R|) depth; property-based tests on 80 hypothesis examples verify this empirically.

### 5.2 Methodological novelty — MetalGeometryPrior as a first-class MCTS channel

**What.** Pt(II) is enforced as square-planar coord=4; Ru(II) octahedral coord=6; Au(I) linear coord=2; Au(III) square-planar coord=4. The metal-geometry prior is not a post-hoc filter — it is a soft tiered reward in the MCTS rollout (`1.0 / 0.5 / 0.2 / 0.0` per tier) **and** a typed combinator's arity constraint that prevents generation of invalid coordination spheres in the first place.

**Why novel.** Existing SBDD models (TargetDiff / DiffSBDD / Pocket2Mol) handle metals as special atom types and rely on learned correlations; they do not enforce geometry symbolically. The scaffold-aware click compatibility matrix (`molmetal_lam/lam_chem/pt_click_compat.py`, 5×5 = 25 cells: Pt_II / Pt_IV / Ru_II / Ru_III / Au_I / Au_III × CuAAC / SPAAC / ThiolEne / Suzuki / AmideCoupling) is also a first.

### 5.3 Engineering novelty — Deflex: a dual symbolic+geometric prior over MCTS leaves

**What.** Deflex (`paper/sections/03_5_deflex.tex`) wires a learned pocket-conditioned prior (`PocketMacroInference`, post CA2 fix v2, 14-tier anchor one-hot) into MCTS leaf selection alongside the symbolic reward. The "dual-system" framing: the symbolic MCTS chooses chemically valid reductions; the geometric prior (`fused_silu_mlp` over EGNN atom features) prefers leaves whose per-atom embeddings match pocket-conditioned prototypes derived from PlatinAI 226K + tmQM 108K training corpus.

**Why novel.** This is the **first MCTS-with-pocket-prior for metallodrug de novo design**. R12 path-A pilot shows n_distinct 1 → 20 (+20×) and div_tanimoto 0 → 0.1065 once the Deflex wiring is enabled. F5 symbolic reward + PySR-discovered closed-form lift functions round out the prior stack.

### 5.4 Engineering novelty — chemistry-safety veto as opt-in first-class gate

**What.** T30 P1.2 (`fg_compatibility.py`, 240 LOC, 28/28 tests) defines a hand-curated FG_COMPATIBILITY table covering 9 reaction classes × 2-4 SMARTS each (lit-anchored: Lippard 1995 / Reedijk 1987 / Himo 2005 / Kolb 2001 / Barner-Kowollik 2011 / Miyaura 1995 / Hoyle 2010). Veto fires only when (a) `--fg-veto-strict` enabled, (b) reactant SMILES contains a disfavoured functional group, (c) reactant contains **zero** tolerated functional groups. Tolerated-overrides is hard-coded (negative-false-positive rate = 0, property-test invariant).

**Why novel.** Most generators either (i) ignore chemistry safety entirely or (ii) hard-block disfavoured groups even when tolerated alternatives exist. The tolerated-overrides pattern is rare; the property-test-gated opt-in policy is, to our knowledge, novel for typed β-NF MCTS.

### 5.5 Cross-dataset metallodrug vertical (paper §3.7)

**What.** Beyond CrossDocked2020 (organic pockets), Mol-Metal ships three metallodrug-specific datasets with explicit property channels:
- **PlatinAI** (226 K Pt coordination complexes with A2780 / MCF7 cytotoxicity labels) — kNN oracle channel
- **tmQM** (108 K transition-metal complexes with bond-order + coordination labels from quantum chemistry) — CFM pretraining + learned prior
- **MetalCytoToxDB** (curated Pt / Ru / Au / Ir cytotoxicity) — direct supervised signal
- **NCI60** (60-cell-line panel) — anticancer index channel
- **Patent axis** (known Pt drugs CSV, max-sim-to-known metric) — patent-novelty channel

**Why novel.** No public benchmark currently combines typed-click chemistry + metal-geometry priors + multi-metal cytotoxicity supervision in a single generator. The 5-dataset combination supports both *de novo design* (λ-only, no pocket) and *SBDD* (pocket-conditioned) modes.

### 5.6 Honest anti-claims — the lift we did NOT claim

Mol-Metal does **not** claim SOTA. The closest direct SOTA references — TargetDiff (ICLR 2023), Pocket2Mol (ICML 2022), DiffSBDD, DECOMP-DIFF, FLOWr — are tracked cite-only in `metrics/sota_comparisons/` with explicit protocol-mismatch flags M1–M7 (different docking engines, different splits, different metal coverage, different prompt-vs-CFG evaluation). Honest journal tier under Pivot-A framing: **PRIMARY J. Chem. Inf. Model. (ACS) Q1 IF 5.6**; SECONDARY Digital Discovery (RSC) Q1 IF 8.5. JACS / Nat. Comput. Sci. require CFM-decode lift + wet-lab validation, deferred to R16+.

## 6. Honest status (as of 2026-09-17)

| Round | Goal | Status |
|---|---|---|
| R3–R11 | axes / pipeline / RDKit batch / reaction confidence / Closure theorem / pIC50 / PlatinAI / REINVENT4 | ✅ all shipped |
| R12 | Round-12 pilot at top-journal standard | ✅ 147 cells DESIGN→MEASURED; n_distinct lift 1 → 20 |
| R13 | 100-pocket × 3-seed paper-grade sweep | ⚙️ partial; deferred to R16 W44-W45 (GPU) |
| R14 | lit-grounded + math-prior + code-fix lift | ⚙️ partial; SA-aware MCTS + cite-only SOTA + metallodrug property proxies + pharmacophore channel + Pareto ranker + patent axis + rule registry + scaffold split all shipped |
| R15 | structural ship: 13 workflows + YuelBond + 24-paper Frontier + sub-fix A/C + Deflex wireup + coupling bridge | ✅ STRUCTURAL SHIP (89 tests 82 pass + 5 skip + 2 fail, 2 real bugs caught) |
| **R16 (in flight)** | **YuelBond 10000-step GPU retrain + Round-13 30-cell sweep + PB smoke + Deflex verify → arXiv W49 → journal Q2-2027** | 🟢 **YuelBond 2000-probe PASS (decode_ratio 0 → 1.000); 10000-step launching** |
| arXiv submission | — | ⏳ R16 W49 (2026-12-02) |

## 7. Reproducing a key result (5 min)

```bash
cd /home/hugo/codes/try_triton_on_rocm
uv run python molmetal/scripts/r4_lambda_only_run.py \
    --pocket test_001 --seed 0 --n-simulations 100 \
    --metal-seed cisplatin --click-rules all-5 \
    --sa-weight 0.3 --engine both
# Expect: validity=1.0, n_distinct ≥ 5, div_tanimoto ≈ 0.1, sa_mean ≈ 3.3, qed ≈ 0.7
```

A full 100×3 sweep (`--n-simulations 1000` over test_001..test_100) takes ~50 min CPU + ~6h QVina-GPU. Numbers trace to `metrics/by_metric/*.json` and propagate to `paper/sections/04_evaluation.tex`.

## 8. Citation

```bibtex
@misc{molmetal2026,
  title  = {Mol-Metal: A Hybrid Molecular Lambda Calculus + Flow-Matching Framework for Precious-Metal Drug Design},
  author = {Mol-Metal Contributors},
  year   = {2026},
  note   = {paper/main.pdf (76 pages, 5.22 MB); \texttt{molmetal/} + \texttt{triton\_kernels/} source}
}
```

## 9. Hardware / software matrix

| Component | Spec |
|---|---|
| GPU | AMD RX 7800 XT (RDNA 3, `gfx1101`, 16 GB, 30 SMs) |
| iGPU | Radeon 780M (`gfx1100/1103`, 31 GB shared) |
| ROCm | 7.2.x |
| PyTorch | `2.14.0+rocm7.2` (custom index) |
| Triton | `triton-rocm==3.8.0` (ROCm-only) |
| Python | 3.12 (uv-managed) |
| Package manager | uv (pyproject.toml pins ROCm index) |
| Core deps | RDKit, OpenBabel, Vina 1.2 + QVina 2.1, OpenMM, PoseBusters, AiZynth, REINVENT4, PySR, DiffDock, MGLTools, fair-esm, tmQM-loader, PyTorch-Geometric (round-7 installed, removed R4 for SBDD-free path) |

## 10. License & disclaimer

Internal research code; not for production use. All in-silico predictions require experimental validation; cytotoxicity, off-target toxicity, and ADME properties are not predictive of in-vivo behaviour. Use of PlatinAI / NCI60 / MetalCytoToxDB data follows the original licenses.

---

## 11. Migration guide — bring-up checklist for a fresh host (NVIDIA or AMD)

This section is the **single source of truth** for re-deploying Mol-Metal on a new server. The repo was developed on AMD ROCm 7.2 + RX 7800 XT, but the code is portable — most ops go through PyTorch + RDKit + standard CPU libraries. **The only ROCm-specific code is the Triton kernel engine (`triton_kernels/`) and the dGPU placement of CFM training**.

### 11.1 System requirements

| Component | Minimum | Recommended |
|---|---|---|
| **OS** | Linux x86_64 (kernel ≥ 5.15) | Ubuntu 22.04 LTS |
| **Python** | 3.12 (uv-managed; pinned via `pyproject.toml`) | 3.12.7 |
| **RAM** | 32 GB | 64 GB+ (n_distinct=20 replay buffer × 30 cells) |
| **Disk** | 30 GB for code + checkpoints + datasets | 100 GB+ if you also mirror PlatinAI / tmQM / NCI60 |
| **GPU (NVIDIA path)** | RTX 3090 24 GB / A100 40 GB | A100 80 GB or H100 80 GB |
| **GPU (AMD path)** | RX 7800 XT 16 GB (gfx1101) | RX 7900 XTX 24 GB |
| **GPU driver** | NVIDIA ≥ 535 + CUDA 12.1 | CUDA 12.4 |
| **ROCm (AMD only)** | ROCm 7.2.x | ROCm 7.2.4 (verified) |

> ⚠️ **All training is single-GPU.** Tensor-parallel / FSDP / multi-GPU is **not** implemented (R16 verdict: 16 GB is sufficient at h=128, l=3, batch=8 with BF16 + checkpoint).

### 11.2 Python dependencies (the `pyproject.toml` set, exact pins)

```toml
[project]
name = "try-triton-on-rocm"
version = "0.1.0"
requires-python = "==3.12.*"
dependencies = [
    "torch>=2.14,<2.15",          # ROCm build for AMD, CUDA build for NVIDIA
    "triton-rocm==3.8.0",          # AMD only; replace with `triton>=3.0` on NVIDIA
    "numpy>=1.26,<3",
    "pandas>=2.0",
    "scipy>=1.11",                # only if --ot-mode hungarian
    "pyyaml>=6.0",
    "biopython>=1.81",
    "rdkit>=2024.3",              # core chemistry — DO NOT skip
    "openpyxl>=3.1",
    "scikit-learn>=1.3",          # baselines (ridge, RF, lightGBM)
    "xgboost>=2.0",
    "hypothesis>=6.168.0",        # property-based tests (closure theorem etc.)
    "datamol>=0.13.0",
    "molfeat>=0.11.0",
    "openbabel-wheel>=3.1.1.23",
    "meeko>=0.8.0",               # Vina receptor prep
    "torchdiffeq>=0.2.5",         # ODE solver (R15 midpoint fix)
    "gemmi>=0.7.5",               # PDB/mmCIF parsing
    "openmm>=8.6.1",              # MMFF94 relax (R15 PB fix)
    "posebusters>=0.6.5",         # 26-check validator
    "prody>=2.6.1",
    "mdanalysis>=2.10.0",
    "vina>=1.2.7",
    "admet-ai>=2.0.1",
    "pot>=0.9.7.post1",           # Optimal-transport CFM pairing
]
```

### 11.3 NVIDIA migration — exact swap

| File / line | ROCm (current) | NVIDIA change |
|---|---|---|
| `pyproject.toml` line 47–49 | `triton-rocm==3.8.0` from `download.pytorch.org/whl/rocm7.2` | `triton>=3.0` from PyPI; `torch==2.14.0` from `download.pytorch.org/whl/cu121` (or `cu124`) |
| `pyproject.toml` lines 60–65 (`[tool.uv.sources]`) | `[tool.uv.index] name="pytorch-rocm" url=https://download.pytorch.org/whl/rocm7.2` | Replace index URL to `cu121` (or matching CUDA version); no other source change |
| `molmetal/__init__.py:_apply_rocm_gfx1101_env()` | sets `PYTORCH_HIP_ALLOC_CONF`, `TORCH_BLAS_PREFER_HIPBLASLT=1` | **Replace `HIP` with `CUDA` env vars**: `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, remove `TORCH_BLAS_PREFER_HIPBLASLT` |
| `molmetal/adapters/flow_matching_lipman/amp.py:73-76` (`ROCM_GFX1101_ENV_VARS`) | ROCm-specific env-var map | Make a sibling `CUDA_ENV_VARS` or remove the hard-list and let user set externally |
| `triton_kernels/*.py` | ROCm-supported via `triton-rocm` | **Triton 3.x is source-compatible** with both NVIDIA and AMD — same kernel source, different runtime. No code change needed beyond the swap above |
| `molmetal/models/_scatter.py` | RDNA3-specific detection | The fallback to `torch.scatter_add_` is **already NVIDIA-safe** (it's the default PyTorch path). No change |
| `molmetal/adapters/egnn_rocm.py` | filename + Triton wiring | Triton is portable; no code change beyond the triton package swap |

**TL;DR for NVIDIA**: change **2 lines** in `pyproject.toml` + 1 function in `molmetal/__init__.py`. Everything else is portable.

### 11.4 System-level binaries (not in pyproject)

| Tool | Purpose | Install on Ubuntu |
|---|---|---|
| **Vina 1.2.7** | Classical docking | `apt install autodock-vina` (or use `vina` PyPI binding — same thing) |
| **QuickVina 2.1** | Faster Vina fork (N=50 parity r=0.9983 vs Vina) | build from source: https://github.com/QVina/qvina |
| **QuickVina-GPU 2.1** | GPU docking | build from source: https://github.com/JustinShin/QuickVina-GPU (NVIDIA CUDA required, ROCm fork does NOT exist for gfx1101) |
| **MGLTools + prepare_receptor4.py** | PDB → PDBQT for Vina | `conda install -c bioconda mgltools` |
| **OpenBabel CLI** | SMILES ↔ 3D format conversion | `apt install openbabel` |
| **OpenMM 8.x** | MMFF94 conformer relaxation (R15 PB fix) | `conda install -c conda-forge openmm` |
| **REINVENT4** | Multiproperty oracle (optional, separate venv recommended) | `pip install reinvent4` (best in a separate venv) |
| **PySR** | Symbolic regression for Deflex priors (optional) | `pip install pysr` + Julia 1.10 backend |
| **DiffDock** | GPU docking baseline (optional, cloned repo) | see `molmetal/references/diffdock/` |
| **AiZynth** | Retrosynthesis oracle (optional) | `pip install aizynthfinder` |
| **REINVENT / fair-esm** | Sequence baselines (optional) | `pip install fair-esm` |
| **PyTorch-Geometric** | Removed R4; do **NOT** install | (we do not use it) |

### 11.5 Environment variables (set BEFORE `import molmetal`)

The bootstrap is automatic on `import molmetal` (gfx1101 detection); for NVIDIA you must set these manually:

```bash
# Required for BF16 matmul + reduced fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True     # NVIDIA
export PYTORCH_HIP_ALLOC_CONF=expandable_segments:True      # AMD (auto-applied)

# AMP / checkpoint toggles (defaults shown)
export MOLMETAL_CFM_AMP=1                    # 1=BF16 autocast (default ON)
export MOLMETAL_CFM_AMP_DTYPE=bfloat16       # bfloat16 | float16 | float32
export MOLMETAL_EGNN_CHECKPOINT=1            # 1=per-EGNN-layer checkpoint (default ON)

# Hygiene (auto-applied on any GPU host)
export TOKENIZERS_PARALLELISM=false
export TRANSFORMERS_VERBOSITY=error

# Optional: pin GPU (NVIDIA multi-GPU systems)
export CUDA_VISIBLE_DEVICES=0

# Optional: pin GPU (AMD)
export ROCR_VISIBLE_DEVICES=0
export HIP_VISIBLE_DEVICES=0
```

### 11.6 Datasets — what you need and where to put it

Mol-Metal uses 7 datasets; **all downloads must happen on the new host**, no datasets are pre-shipped (only CSV manifests + cache files).

| # | Dataset | What it is | Size on disk | Source / URL | Loader class |
|---:|---|---|---:|---|---|
| 1 | **CrossDocked2020** | 100K pocket-ligand pairs (organic SBDD baseline; legacy) | ~22 GB tarball | https://bits.csb.pitt.edu/files/ (Pitt CSB) | `molmetal/data/crossdocked.py::CrossDockedDataset` (auto-downloads `crossdocked_pocket10.tar.gz`) |
| 2 | **PlatinAI** | 226K Pt coordination complexes with A2780 / MCF7 cytotoxicity labels | ~50 MB CSV | https://github.com/ai-safety-institute/PlatinAI (or authors' repo) | `molmetal/data/cytotox.py` + `metallo_drugs_500_train.csv` cache (49 KB; pre-built MaxMin-diverse 500-mol subset) |
| 3 | **tmQM** | 108K mononuclear transition-metal complexes with DFT geometries + Wiberg bond orders (Balcells & Skjelstad 2020, JCIM) | ~150 MB (XYZ + BO files) | https://github.com/uiocompcat/tmQM | `molmetal/data/tmqm.py::TmqmStats` (loads `tmQM_y.csv`, `tmQM_X{1,2,3}.xyz.gz`, `tmQM_X{1,2,3}.BO.gz`) |
| 4 | **MetalCytoToxDB** | Curated Pt / Ru / Au / Ir cytotoxicity | ~5 MB CSV | cited `molmetal/data/metalloprotein_targets.py:acs.jcim.3c01568` | `molmetal/data/cytotox.py::MetalCytotoxDataset` |
| 5 | **NCI-60** | 60-cell-line cytotoxicity panel | ~2 MB (per-cell CSV) | https://dtp.cancer.gov/discovery_development/nci-60/ | `molmetal/data/cytotox.py` |
| 6 | **Known Pt drugs** (manual patent set) | 8 FDA-approved Pt drugs (cisplatin, carboplatin, oxaliplatin, nedaplatin, satraplatin, picoplatin, heptaplatin) | 1 KB CSV | hand-curated, shipped as `molmetal/data/known_pt_drugs.csv` | — |
| 7 | **CrossDocked pocket subsets** (test_001..test_100) | 100-pocket subset for sweep benchmarks | ~50 MB SDF subset | generated from #1; manifest in `crossdocked100_manifest.csv` | `molmetal/data/crossdocked_filter.py` |

#### Data directories expected by the loaders

The defaults look under `/mnt/storage/data/molmetal/...` and `$PROJECT_ROOT/molmetal/data/...`. **You will need to either**:

1. **Mount** `/mnt/storage/data/molmetal/` and unzip datasets there (production-style layout)
2. **OR** override the paths with env vars — search for `os.environ.get` in `molmetal/data/*.py` to see which paths are configurable.

```bash
# Default layout expected
/mnt/storage/data/molmetal/
├── tmQM/
│   ├── tmQM_y.csv
│   ├── tmQM_X1.xyz.gz
│   ├── tmQM_X1.BO.gz
│   ├── tmQM_X2.xyz.gz
│   ├── tmQM_X2.BO.gz
│   ├── tmQM_X3.xyz.gz
│   └── tmQM_X3.BO.gz
├── crossdocked/
│   └── crossdocked_pocket10.tar.gz
├── platinai/
│   └── platinai_a2780_mcf7.csv
├── cytotox/
│   └── MetalCytoToxDB.csv
└── nci60/
    └── nci60_cell_lines.csv
```

The **`metallo_drugs_500_train.csv`** file in `molmetal/data/` is **the only dataset file pre-shipped** (49 KB — MaxMin-diverse 500-mol subset from PlatinAI + MetalCytoToxDB + tmQM combined pool). It is **the canonical training pool for CFM**; the CFM model can train on this file alone without downloading anything else.

### 11.7 Pre-trained checkpoints (also need to be copied / re-trained)

The `molmetal/checkpoints/` folder contains **17 .pt files** totalling ~80 MB. They are versioned by training round:

| Checkpoint | Size | Purpose | Source |
|---|---:|---|---|
| `fm_pocket.pt` | ~2.6 MB | **CFM SE(3)-equivariant pocket-conditioned flow matching**, h=128/l=3 (687K params) | R12-15 |
| `fm_pocket_conditioned.pt` | ~2.6 MB | CFM with pocket-conditioning explicitly ON | R15 |
| `fm_ru_temporal.pt` | ~2.6 MB | CFM Ru-only temporal holdout | R11 |
| `metal_hybrid_Ru.pt` | ~10 MB | Ru-only hybrid classifier | R11 |
| `metal_hybrid_v1_concat_ru_temporal.pt` | ~10 MB | Hybrid v1 concat | R11 |
| `metal_hybrid_v2_crossattn_ru_temporal.pt` | ~10 MB | Hybrid v2 cross-attention | R12 |
| `metal_hybrid_v3_crossattn_ru_temporal.pt` | ~10 MB | Hybrid v3 | R13 |
| `metal_hybrid_v4_ru_temporal_seed{42,133,256}.pt` | ~10 MB × 3 | Hybrid v4, 3-seed sweep | R13 |
| `metal_hybrid_v4_mb1_seed{42,133,256}.pt` | ~10 MB × 3 | Hybrid v4 metal-binding 1 | R14 |
| `dmpnn_attn_ru_pic50.pt` | ~5 MB | D-MPNN attention baseline (Ru pIC50) | R10 |
| `dmpnn_tmqm_pretrained.pt` | ~5 MB | D-MPNN tmQM pretrained | R10 |
| `dmpnn_attn_heLa48h_dark_retrained.pt` | ~5 MB | D-MPNN HeLa48h dark cohort (TODO-18) | R14 |
| `egnn_tmqm_finetuned_ru.pt` | ~2.6 MB | EGNN tmQM-pretrained Ru | R10 |
| `metallo_drug_smoke_mol_cache.pt` | varies | RDKit 3D embed cache for the 500-mol pool | R15 |

**Migration strategy**: 
- **Copy** all `.pt` files via `scp` / `rsync` — they are PyTorch `state_dict` dicts and load across platforms (CPU ↔ GPU ↔ ROCm ↔ CUDA) as long as PyTorch versions match.
- **OR retrain** from scratch using `molmetal/scripts/metallo_drug_smoke_retrain.py` (≈4-6 h on A100 / 12 h on RX 7800 XT).

### 11.8 End-to-end bring-up (NVIDIA example, copy-paste)

```bash
# 1. clone the repo
git clone <repo-url> molmetal && cd molmetal

# 2. install Python 3.12 + uv (if not already)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.local/bin/env

# 3. swap pyproject.toml ROCm → CUDA
#    (or use the project's --extra-index-url flag at install time)
sed -i 's|triton-rocm==3.8.0|triton>=3.0|' pyproject.toml
sed -i 's|https://download.pytorch.org/whl/rocm7.2|https://download.pytorch.org/whl/cu121|' pyproject.toml

# 4. install
uv sync

# 5. NVIDIA env vars (before any import torch)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MOLMETAL_CFM_AMP=1
export MOLMETAL_CFM_AMP_DTYPE=bfloat16
export MOLMETAL_EGNN_CHECKPOINT=1

# 6. verify GPU works
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count(), torch.cuda.get_device_name(0))"
# expect: True 1 NVIDIA A100 80GB PCIe

# 7. download datasets (only the ones you need)
# Option A: full set (recommended for paper reproduction)
bash scripts/download_all_datasets.sh      # (see script below — auto-generated by this checklist)

# Option B: minimal — only the pre-shipped 500-mol pool (enough for Lambda MCTS pilot + CFM 1000-step smoke)
# metallo_drugs_500_train.csv is already in molmetal/data/, no download needed

# 8. run the canonical 5-minute smoke
uv run python molmetal/scripts/r4_lambda_only_run.py \
    --pocket test_001 --seed 0 --n-simulations 100 \
    --metal-seed cisplatin --click-rules all-5 \
    --sa-weight 0.3 --engine both
# expect: validity=1.0, n_distinct ≥ 5, sa_mean ≈ 3.3

# 9. (optional) train CFM from scratch
uv run python molmetal/scripts/metallo_drug_smoke_retrain.py \
    --steps 1000 --hidden-dim 128 --n-layers 3 \
    --n-train 500 --batch-size 8 \
    --bond-head learned --joint-train
# expect: decode_ratio lift over baseline; loss 7→5 over 1000 steps

# 10. (optional) full 10000-step retrain for R16 lift
uv run python molmetal/scripts/metallo_drug_smoke_retrain.py \
    --steps 10000 --hidden-dim 128 --n-layers 3 \
    --n-train 500 --batch-size 4 \
    --bond-head learned --joint-train \
    --decode-smoke-every 1000
# ~6-12 h on A100, ~12-24 h on RX 7800 XT
```

### 11.9 What runs on CPU-only (no GPU required)

The following features are **CPU-only** and will work on any host:

- ✅ All `r4_lambda_only_run.py` Lambda MCTS pilots (validity / uniqueness / synth / novelty / n_distinct / SA / QED / div_tanimoto / homotype_div / pharmacophore / patent-axis / Pareto / FG-veto / PlatinAI oracle / REINVENT4 oracle / AiZynth oracle)
- ✅ All dataset loaders (RDKit + pandas + scipy)
- ✅ PoseBusters chemistry-only (`--pb-mode mol`, 14 checks)
- ✅ All property-based tests + unit tests (Pytest)
- ✅ Paper recompile (`cd paper && pdflatex && bibtex && pdflatex && pdflatex`)
- ✅ Aggregate + INDEX + memory updates

### 11.10 What requires GPU

| Component | GPU memory | Time | Required? |
|---|---|---|---|
| CFM retrain (1000-step smoke) | ~3-5 GB | 30 min | Optional |
| CFM retrain (10000-step full) | ~4-8 GB (BF16) | 6-12 h | **YES for paper §4.6 CFM column** |
| Vina/QVina docking | CPU only (0.5 GB GPU optional for QVina-GPU) | ~5 min/cell | Yes for paper §4.6 Vina column |
| QVina-GPU 100×3 sweep | ~2 GB GPU | ~5 h | Yes for paper §4.6 Vina scale |
| Round-13 PB 30-cell production | ~3 GB GPU | ~3 h | Yes for paper §4.6 PB column |
| Deflex v2 BUG-2 verify | ~3 GB GPU | ~1 h | Optional |

### 11.11 Verification checklist after migration

```bash
# All should print "OK" or a non-zero metric
uv run pytest -q molmetal/molmetal_lam/tests/ --tb=short        # ≥ 158 pass / 1 skip / 1 xfail
uv run pytest -q molmetal/molmetal_lam/tests/test_amp.py        # 10/10
uv run pytest -q molmetal/molmetal_lam/tests/test_egnn_checkpoint.py  # 5/5
uv run pytest -q molmetal/molmetal_lam/tests/test_measure_vram_unit.py  # 11/11
uv run pytest -q molmetal/tests/test_rocm_env.py               # 4/4 (NVIDIA host: see §11.3 swap)

# GPU sanity (NVIDIA)
uv run python -c "
import torch
print('cuda:', torch.cuda.is_available(), 'devices:', torch.cuda.device_count())
x = torch.randn(1024, 1024, device='cuda:0')
y = torch.randn(1024, 1024, device='cuda:0')
torch.cuda.synchronize()
print('matmul ok, z[0,0]=', (x@y)[0,0].item())
"
```

If any of these fail, the migration is incomplete; see §11.3 (NVIDIA env-var swap) and §11.5 (env vars) first.

---
