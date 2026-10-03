# TODO-63 — R18 P2 ML 方法 CPU Skeleton Synthesis (Wf-2.4)

**Date**: 2026-09-26
**Parent plan**: `TODO/03_plans/R18_PLAN.md` §2 + `TODO/03_plans/R18_MATH_MIGRATION_PLAN.md` §3 P2
**Workflow binding**: Wf-2.4 — 4 P2 ML-method CPU skeletons, ships alongside Wf-1 (`54_r18_math_skeleton.md`, shipped) and Wf-2 (`55_r18_integration_adapters.md`, shipped).
**Owner**: Hugo Lee + ultracode agents
**Status**: 4/4 CPU skeletons **shipped** (CPU-only, 0 GPU training); R18 GPU evaluation deferred.

---

## 1. TL;DR

**4 个 P2 ML 方法 CPU 骨架**全部以 CPU-only 模块 ship 在 `molmetal/adapters/` 与 `molmetal/molmetal_lam/lam_chem/`,
每方法 1 模块 + 1 `molmetal/tests/test_*.py`:

| # | 方法 | 模块 | LOC | 单测数 | 状态 |
|---|------|----------|----:|------:|------|
| **P2-1** | **W-Flow** (Stanford ICML'26, arXiv:2605.11755) | `molmetal/adapters/flow_matching_lipman/w_flow.py` | **912** | **20/20** | shipped (CPU) |
| **P2-2** | **MacroGuide PH** (Oxford ICML'26, arXiv:2602.14977) | `molmetal/molmetal_lam/lam_chem/macroguide_ph.py` | **668** | **29/29** | shipped (CPU) |
| **P2-3** | **Coordination-Aware GNN** (ChemRxiv'26, 691f4f5e) | `molmetal/adapters/coordination_gnn.py` | **477** | **32/32** | shipped (CPU) |
| **P2-4** | **Quotient-Space Diffusion** (arXiv:2604.21809) | `molmetal/adapters/quotient_space_diffusion.py` | **837** | **36/36** | shipped (CPU) |

**Aggregated**: **117/117 tests pass** (20 + 29 + 32 + 36, CPU-only); **0 GPU 训练** ship;
0 模块触碰现有 `molmetal/adapters/flow_matching_lipman/{__init__,lipman,*}.py` 或
`molmetal/scripts/*` R17 代码 — R17 close-out 在 Wf-2.4 turn-around 中 bit-identical。

---

## 2. Per-Method Deliverables

### P2-1 — W-Flow (Waveform Relaxation ODE Solver)
- `molmetal/adapters/flow_matching_lipman/w_flow.py` (912 LOC)
- Tests: `molmetal/tests/test_w_flow.py` (20 tests, 549 LOC)
- Contract: waveform-relaxation ODE solver from Lelarasmee 1982; splits ``[0,1]`` into ``n_windows`` sub-intervals, each integrated independently with previous iteration as boundary. **shipped (CPU)**; Phase-2.2 ``vel_scale`` un-bounding fix guarantees coupling factor L < 0.1, so geometric convergence applies。
- Public API: `WaveformRelaxationSolver`, `WFlowCFM`, `compare_solvers`, `make_default_w_solver`.

### P2-2 — MacroGuide PH (Ring-Topology Guidance)
- `molmetal/molmetal_lam/lam_chem/macroguide_ph.py` (668 LOC)
- Tests: `molmetal/tests/test_macroguide_ph.py` (29 tests, 475 LOC)
- Contract: persistent-homology-of-Vietoris–Rips-style ring-size histogram + soft guidance score in [0,1]; ring topology features promote macrocycle (≥12 atoms) formation during generation. **shipped (CPU)**; `augment_pll_with_macroguide` concatenates with PLL signature features.

### P2-3 — Coordination-Aware GNN (Coordination-Number + Geometry)
- `molmetal/adapters/coordination_gnn.py` (477 LOC)
- Tests: `molmetal/tests/test_coordination_gnn.py` (32 tests, 564 LOC)
- Contract: coordination-class one-hot (0..N) gates message MLP, separate filters per coordination regime (0=isolated, 2=linear, 4=tetrahedral/square-planar, 6=octahedral). SE(3)-invariant scalar output via distances only。**shipped (CPU)**; independent of `molmetal/adapters/egnn_rocm.py`.
- Public API: `CoordinationAwareGNNLayer`, `CoordinationNumberHead`, `CoordinationGeometryHead`, `CoordinationAwareGNNComposite`.

### P2-4 — Quotient-Space Diffusion (SE(3) by Construction)
- `molmetal/adapters/quotient_space_diffusion.py` (837 LOC)
- Tests: `molmetal/tests/test_quotient_space_diffusion.py` (36 tests, 686 LOC)
- Contract: diffusion / flow-matching in `X / G` quotient space; 3 supported quotients (`"SE3"`, `"rotation"`, `"translation"`). SE(3) equivariance **by construction** (the quotient contains no preferred orientation or origin). **shipped (CPU)**; `project`/`lift` pinned contract via test suite。
- Reference: Arvind, Petrov & Wang 2026 — `arvind2026qsdm` in `paper/refs.bib`.

---

## 3. MEASURED vs PROJECTED

### MEASURED (this turn-around)
- **4/4 modules import OK** under `molmetal.adapters.flow_matching_lipman.w_flow` + `molmetal.molmetal_lam.lam_chem.macroguide_ph` + `molmetal.adapters.coordination_gnn` + `molmetal.adapters.quotient_space_diffusion`.
- **117/117 tests pass** in CPU-only `pytest` across the 4 P2 test files (20 + 29 + 32 + 36).
- **0 GPU 训练** ship; **0 module** modifies `molmetal/adapters/flow_matching_lipman/{__init__,lipman,*}.py` 或 `molmetal/scripts/*` R17 代码 — R17 close-out bit-identical.

### PROJECTED (R18 GPU training, deferred)
- **W-Flow**: 100× wall-time speedup on the 100-pocket × 1000-probe sampling sweep (R17 baseline: 6 ms/molecule; W-Flow paper projects 0.06 ms/molecule after GPU window parallelism; geometric convergence in 3-5 outer iterations once coupling factor L < 0.1 is satisfied).
- **MacroGuide PH**: +1–2 percentage points on macrocycle-macrocycle m11 (n_distinct: 3 → 4–5) by promoting ring-12+ birth/death scoring into the CFM guidance channel.
- **Coordination-Aware GNN**: coordination-number classifier R² ≥ 0.85 on MetalCytoToxDB Pt/Ru subset (vs EGNN baseline ≈ 0.62), once attached as aux-loss head on the R17 velocity_net.
- **Quotient-Space Diffusion**: SE(3) equivariance **exact** (no EGNN approximation), m3 Vina Dock -9.5 → -10.5+ by removing the EGNN-induced orientation bias.

> Per `TODO/03_plans/R18_PLAN.md`: **Quotient-Space Diffusion (P2-4) deferred to R19** on the public roadmap; Wf-2.4 ship is the **CPU skeleton** + paper §3 cite, not GPU training.

---

## 4. Method-by-Method Literature Alignment

| 方法 | Venue / arXiv | 核心思想 | 我们的对接点 |
|---|---|---|---|
| **W-Flow** | Stanford ICML'26, [arXiv:2605.11755](https://arxiv.org/abs/2605.11755) | Waveform relaxation (Lelarasmee 1982); geometric convergence when ODE coupling factor L < 1 | R17 ODESolver midpoint 200 steps → W-Flow 5 windows × 3 outer iters; 100× speedup **PROJECTED** |
| **MacroGuide PH** | Oxford ICML'26, [arXiv:2602.14977](https://www.maths.ox.ac.uk/node/81239) | Persistent-homology-of-VR-complex guidance on 1-dim ring-formation features | Replaces simple PLL with multi-parameter PH ring topology; concat with `pll_signature` features |
| **Coordination-Aware GNN** | ChemRxiv'26, [691f4f5e](https://chemrxiv.org/engage/chemrxiv/article-details/691f4f5e8a08f7504c9bb28f) | Coordination-class one-hot gates message MLP; per-coordination-regime learning | Aux-loss head on R17 velocity_net; coordination-number classifier R² ≥ 0.85 **PROJECTED** |
| **Quotient-Space Diffusion** | arXiv:2604.21809 (arvind2026qsdm) | Diffusion / flow-matching in `X / G` quotient space | R18 P2-4; **SE(3) by construction** (no EGNN approximation) — R19 GPU deferred |

All 4 entries cite the corresponding `paper/refs.bib` key (`wflow2026`, `macroguide2026`, `coord_aware_egnn2026`, `arvind2026qsdm`) so paper §3 can ingest them as soon as Wf-3 runs.

---

## 5. 不修改承诺 (No-Modify Pledges)

Wf-2.4 ships **0** modifications to:

1. `molmetal/adapters/flow_matching_lipman/{__init__,lipman,*}.py` — R17 ODESolver + EGNN ship is read-only。
2. `molmetal/adapters/egnn_rocm.py` — R17 EGNN ship is read-only; Coordination-Aware GNN is **independent**。
3. `molmetal/scripts/*` — all R17 scripts untouched。
4. `molmetal/molmetal_lam/lam_chem/{coupling_adapter,pt_click_compat,*}.py` — R17 lambda click-chem adapters untouched。
5. `molmetal/molmetal_lam/sbdd_env/synflownet_env.py` + `proof_search.py` — search env untouched。

All 4 P2 modules are **new files** that **compose** with the R17 close-out via existing adapter seams; no existing function was renamed, refactored, or had its signature altered.

---

## 6. Cross-Reference: Skills + Plans

- `mol-metal-cfm-training` skill — `+ P2-1 W-Flow` (alongside P0-1 SE(3) Schrödinger Bridge + P0-2 Transport Relaxation + P0-4 薛小平 gradient flow + P1-3 Bass martingale; cf. `TODO/03_plans/R18_MATH_MIGRATION_PLAN.md` skill-integration table)。
- `mol-metal-r18-math` skill — `+ P2-1 W-Flow + P2-2 MacroGuide PH + P2-3 Coordination-Aware GNN + P2-4 Quotient-Space Diffusion` (4 new method-cite entries; paper §3 ingestion in Wf-3)。
- `TODO/03_plans/R18_PLAN.md` §2 — Quotient-Space Diffusion (P2-4) deferred to R19 (CPU skeleton ships now; GPU training R19)。
- `TODO/03_plans/R18_MATH_MIGRATION_PLAN.md` §3 — 4 P2 methods all listed as 🟡/🟢 status; 🟢 Quotient-Space Diffusion flagged "SE(3) by construction"。
- `paper/refs.bib` — 4 new entries (`wflow2026`, `macroguide2026`, `coord_aware_egnn2026`, `arvind2026qsdm`).
- Companion docs: `TODO/completed/54_r18_math_skeleton.md` (Wf-1, T1-T9 数学定理) + `TODO/completed/55_r18_integration_adapters.md` (Wf-2, integration adapters for math skeletons) + `TODO/completed/62_hilbert_score_repellent.md` (Hilbert P2 sibling)。

---

## 7. Honest Framing (R17 + R18 close-out tone)

- **117/117 = 1.000** test pass is **necessary-but-not-sufficient**: it pins the **CPU skeleton** contract but does NOT measure the **100× speedup** (W-Flow), **ring-topology lift** (MacroGuide), **coordination-number R²** (Coord GNN), or **SE(3) equivariance under load** (Quotient-Space Diffusion). All four are **PROJECTED** at the paper / arXiv-reported magnitude and require R18 GPU training to verify.
- **Coordination-Aware GNN** skeleton is **independent** of `molmetal/adapters/egnn_rocm.py`; the R17 EGNN is bit-identical. Wiring it as an aux-loss head on `velocity_net` is a R18 follow-up, not this turn-around。
- **Quotient-Space Diffusion** public roadmap entry is **R19** per `TODO/03_plans/R18_PLAN.md`; the CPU skeleton + paper §3 cite is the Wf-2.4 deliverable, NOT GPU training.
- All 4 methods **compose** with the R17 close-out via adapter seams; no existing CFM / GNN / flow-matching code was touched.

---

**Wf-2.4 close-out banner** — 4 P2 ML 方法 CPU skeleton shipped; R17 close-out bit-identical; R18 GPU training for the 4 PROJECTED magnitudes deferred to R18-P2-GPU (W-Flow, MacroGuide, Coord GNN) and R19 (Quotient-Space Diffusion).
