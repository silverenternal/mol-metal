# TODO-65 — Wf-12 P2.4 + P2.5 + P1.5 + P1.2 真活 Synthesis

**Date**: 2026-09-26
**Parent plan**: `TODO/pending/30_pitfall_reinforce_plan.md` (P1.2 Rank 10, P2.4 Rank 12, P2.5 Rank 13) + `TODO/pending/25_round14_lit_grounded_plan.md` §P1-5 分数阶反应扩散
**Workflow binding**: Wf-12 — 4 真活 CPU 骨架 (genuine, not stub-only), ships alongside Wf-2.4 (`63_p2_ml_methods.md`) + Wf-1 (`54_r18_math_skeleton.md`)
**Owner**: Hugo Lee + ultracode agents
**Status**: 4/4 真活 CPU skeleton shipped; R18 GPU 训练 deferred

---

## 1. TL;DR

4 真活 (genuine implementation, not stub) CPU skeleton ship, 每个 1 module + 1 pytest + 1 verify script:

| # | Pitfall / Method | Module | Tests | Status |
|---|----|------|----:|------|
| **P2.4** | skeleton jump / fragment-swap | `molmetal/molmetal_lam/lam_chem/skeleton_jump.py` (915 LOC) | **25/25** | shipped |
| **P2.5** | MD relax stub | `molmetal/molmetal_lam/lam_chem/md_relaxation_stub.py` (819 LOC) | **24/24** | shipped |
| **P1.5** | 分数阶 ODE (Caputo fractional) | `molmetal/adapters/flow_matching_lipman/fractional_ode.py` (607 LOC) | **25/25** | shipped |
| **P1.2** | FG-compat / leave-group veto | `molmetal/molmetal_lam/lam_chem/fg_compat_veto.py` (843 LOC) | **25/25** | shipped |
| | | **Aggregated** | **99/99** | 4/4 CPU |

4 verify scripts: `molmetal/scripts/verify_{skeleton_jump,md_relaxation,fractional_ode,fg_compat_veto}.py` — all `<5 s` wall-time per `--smoke` run.

---

## 2. Per-Deliverable

### 2.1 P2.4 — Skeleton Jump / Fragment-Swap (TODO-30 Rank 12)
- **Module**: `/home/hugo/codes/mol-metal/molmetal/molmetal_lam/lam_chem/skeleton_jump.py` (915 LOC)
- **Test**: `/home/hugo/codes/mol-metal/molmetal/tests/test_skeleton_jump.py` (492 LOC, 25 tests)
- **Verify**: `/home/hugo/codes/mol-metal/molmetal/scripts/verify_skeleton_jump.py` (267 LOC)
- **Public API**: `FragmentLibrary`, `SkeletonJumpRule`, `DiversityAwareSkeletonJump`, `register_skeleton_jump`
- **Status**: **shipped** (CPU, pure-RDKit, NO torch/GPU)

### 2.2 P2.5 — MD Relaxation Stub (TODO-30 Rank 13)
- **Module**: `/home/hugo/codes/mol-metal/molmetal/molmetal_lam/lam_chem/md_relaxation_stub.py` (819 LOC)
- **Test**: `/home/hugo/codes/mol-metal/molmetal/tests/test_md_relaxation_stub.py` (374 LOC, 24 tests)
- **Verify**: `/home/hugo/codes/mol-metal/molmetal/scripts/verify_md_relaxation.py` (165 LOC)
- **Public API**: `SimpleForceField`, `MDRelaxation`, `GeometryValidator`, `register_md_relaxation`
- **Force field**: bond (k=300) + LJ-12-6 truncated (ε=0.1) + angle (k=100, θ₀=π/2), numerical gradient via central diff (h=1e-3)
- **Status**: **shipped** (CPU stub, INDEPENDENT of R17 `conformer_embed`/`batched_rdkit`)

### 2.3 P1.5 — 分数阶 ODE (Caputo Fractional Derivative)
- **Module**: `/home/hugo/codes/mol-metal/molmetal/adapters/flow_matching_lipman/fractional_ode.py` (607 LOC)
- **Test**: `/home/hugo/codes/mol-metal/molmetal/tests/test_fractional_ode.py` (355 LOC, 25 tests)
- **Verify**: `/home/hugo/codes/mol-metal/molmetal/scripts/verify_fractional_ode.py` (146 LOC)
- **Public API**: `CaputoDerivative`, `FractionalODESolver`, `FractionalCFM`, `fractional_convergence_rate`
- **Numerical method**: Grünwald-Letnikov weights + FAB predictor-corrector (Diethelm 2002, Nonlinear Dyn. 29: 3-22); convergence rate O(h^α), α ∈ (0, 1]
- **Status**: **shipped** (CPU, pure numpy, NO torch/CUDA)

### 2.4 P1.2 — FG-compat / Leave-Group Veto (TODO-30 Rank 10)
- **Module**: `/home/hugo/codes/mol-metal/molmetal/molmetal_lam/lam_chem/fg_compat_veto.py` (843 LOC)
- **Test**: `/home/hugo/codes/mol-metal/molmetal/tests/test_fg_compat_veto.py` (451 LOC, 25 tests)
- **Verify**: `/home/hugo/codes/mol-metal/molmetal/scripts/verify_fg_compat_veto.py` (357 LOC)
- **Public API**: `FGCompatMatrix` (12-metal × 7-ligand = 84-cell), `LeaveGroupVeto`, `VetoAugmentedReducer`, `register_fg_compat_veto`
- **Veto policy**: `strict=True` opt-in; fires iff disfavored-matched AND tolerated-empty; tolerates ambiguity as compatible
- **Status**: **shipped** (CPU, opt-in, property-test gated, negative-false-positive = 0 invariant)

---

## 3. MEASURED vs PROJECTED

### MEASURED (this turn-around)
- **4/4 modules import OK** under `molmetal.molmetal_lam.lam_chem.{skeleton_jump,md_relaxation_stub,fg_compat_veto}` + `molmetal.adapters.flow_matching_lipman.fractional_ode`.
- **99/99 tests pass** in CPU-only pytest across 4 test files (25 + 24 + 25 + 25).
- **4/4 smoke scripts return exit 0** under `--smoke`: `verify_skeleton_jump`, `verify_md_relaxation`, `verify_fractional_ode`, `verify_fg_compat_veto`.
- **0 GPU 训练 ship**; **0 module** modifies the 8-file no-modify pledge (see §7).

### PROJECTED (R18 GPU 训练, deferred)
- **P2.4**: scaffold diversity lift on the 100-pocket × 1000-probe sweep (target: n_distinct: 1 → 4-6 via fragment-swap diversification).
- **P2.5**: post-relax energy reduction (target: mean pre/post ΔE < -1.0 kcal/mol on the TMQM 50-conf subset; docking score lift +0.5 to +1.5 kcal/mol on Platin-AI 4-cell).
- **P1.5**: long-memory stability on stiff pockets of v_θ (target: O(h^α) error convergence verified at α=0.5 / 0.9; wall-time budget <2× Euler per the verify script n_runs benchmark).
- **P1.2**: precision uplift on Round-14 metallodrug runs (target: zero chemically-impossible β-reductions like aldehyde+aldehyde, COOH+COOH; --fg-veto-strict default OFF preserves R12/13 measurements).

---

## 4. P2.4 + P2.5 Alignment with TODO-30 Pitfall Rank 12-13

`TODO/pending/30_pitfall_reinforce_plan.md` Rank 12 (P2.4) read: *"DEFER (sequence after P2.3; high invalid-structure risk)"*. Wf-12 ships P2.4 as a **CPU-only, RDKit-sanitisable** swap path: the `FragmentLibrary` requires sanitizable SMILES, the `SkeletonJumpRule.compute_compatibility` returns 0 for invalid outputs, the `DiversityAwareSkeletonJump` enforces history capacity to bound the search blow-up.

Rank 13 (P2.5) read: *"DEFER (needs ROCm OpenMM validation)"*. Wf-12 ships P2.5 as a **CPU stub that does NOT depend on OpenMM** — `SimpleForceField` uses only covalent radii + numerical gradients, so the stub is independent of the ROCm toolchain. The geometry-validator gates implausible bond lengths + close contacts before downstream docking, addressing the P2.5 high-risk invalid-structure symptom without the OpenMM dependency.

Both modules comply with the TODO-30 "no modification of click-rule registry" guard (see §7).

---

## 5. P1.5 Alignment with 数学学报 2026 分数阶反应扩散

P1-5 is R18 P1-5 in `TODO/pending/25_round14_lit_grounded_plan.md` §P1-5, citing 数学学报 2026 分数阶反应扩散: CFM ODE `D^α y = f(t, y)` discretised via Grünwald-Letnikov + FAB multi-step (Diethelm 2002, *Nonlinear Dynamics* 29: 3-22). The skeleton ships three classes — `CaputoDerivative` (analytical sanity on constant / linear inputs), `FractionalODESolver` (FAB predictor-corrector with configurable `base_method`), `FractionalCFM` (CFM wrapper exposing `compare_with_base` for end-to-end α-sweep). At `α=1` it reduces exactly to the integer-order Euler step in the R17 sampler; at `α∈(0,1)` the convergence rate is `O(h^α)` per Lubich 1986 *SIAM J. Math. Anal.* 17(3): 704-719. `fractional_convergence_rate` makes the rate observable in CI without torch.

---

## 6. P1.2 Alignment with FG-Compat Veto (TODO-30 Rank 10)

TODO-30 Rank 10 P1.2 read: *"FG-compat / leave-group veto in `_reduce` — DEFER (over-narrows veto; needs property tests first)"*. Wf-12 ships the veto as an **opt-in sidecar** (`VetoAugmentedReducer` wraps the existing reducer without modifying it), with a hand-curated **84-cell FG-compat matrix** mirroring the R17 84-rule MLE matrix (12 metals × 7 ligands), plus a **leave-group veto** layered on March's "good/mediocre/bad LG" ordering. The test suite enforces **negative-false-positive rate = 0** on canonical educt pairs (every canonical pair still passes under `strict=True`).

Wf-12 also complements the earlier `wf_t30_p12_fg_veto` shim (`molmetal/reports/wf_t30_p12_fg_veto/final.md`) — that workflow shipped a 9-entry `FG_COMPATIBILITY` SMARTS table keyed on reaction-class; this Wf-12 module ships a procedural 84-cell matrix keyed on FG identity, so the two coexist and can be cross-referenced in the audit report.

---

## 7. 不修改承诺 (No-Modify Pledge)

Wf-12 ships **0** modifications to:

1. `molmetal/molmetal_lam/lam_chem/rules.py`
2. `molmetal/molmetal_lam/lam_chem/lambda_combinators.py`
3. `molmetal/molmetal_lam/lam_chem/closure.py`
4. `molmetal/molmetal_lam/lam_chem/conformer_embed.py`
5. `molmetal/molmetal_lam/lam_chem/batched_rdkit.py`
6. `molmetal/molmetal_lam/lam_chem/ode_solver.py`
7. `molmetal/molmetal_lam/lam_chem/loss.py`
8. `molmetal/molmetal_lam/lam_chem/reducer.py`
9. `molmetal/molmetal_lam/lam_chem/stereo_aware_reduction.py`

All 4 真活 modules are **new files** that compose with R17 close-out via adapter seams; no existing function was renamed, refactored, or had its signature altered.

---

## 8. Cross-Reference: Skills + Plans

- **mol-metal-click-chemistry** skill — `+ P1.2 FG-compat veto` (alongside the existing `wf_t30_p12_fg_veto` 9-entry SMARTS table; the 84-cell matrix here is keyed on FG identity, NOT reaction-class, so the two are complementary).
- **mol-metal-lambda-calculus** skill — `+ P2.4 skeleton jump` (fragment-swap rule that the MLC `_reduce` macro can call as a single expansion step; the MCTS `_safe_reduce` is untouched).
- **mol-metal-cfm-training** skill — `+ P1.5 分数阶 ODE` (alongside P0-1 SE(3) Schrödinger Bridge + P0-2 Transport Relaxation + P0-4 薛小平 gradient flow + P2-1 W-Flow; cf. `TODO/pending/25_round14_lit_grounded_plan.md`).
- **mol-metal-r18-math** skill — `+ P1.5 分数阶 ODE` (数学学报 2026 cite, alongside the 4 P2 ML methods in `63_p2_ml_methods.md`).
- `TODO/pending/30_pitfall_reinforce_plan.md` — P1.2 Rank 10 / P2.4 Rank 12 / P2.5 Rank 13 all now have CPU skeletons + property tests (R18 GPU training deferred).
- `TODO/pending/25_round14_lit_grounded_plan.md` §P1-5 — 数学学报 2026 分数阶反应扩散 cite.
- `TODO/completed/63_p2_ml_methods.md` (Wf-2.4, 4 P2 ML-method CPU skeletons, 117/117 tests).
- `TODO/completed/54_r18_math_skeleton.md` (Wf-1, T1-T9 数学定理 CPU skeletons).
- `molmetal/reports/wf_t30_p12_fg_veto/final.md` (P1.2 shim from 2026-09-17, 9-entry reaction-class SMARTS table).
- `molmetal/reports/wf_pitfall_reinforce/final.md` (P1-P7 CI guard rails, 18 tests, 7/7 pitfalls).

---

## 9. Honest Framing

- **99/99 = 1.000 test pass is necessary-but-not-sufficient**: it pins the CPU skeleton contract but does NOT measure the diversity lift (P2.4), MD energy reduction (P2.5), long-memory stability (P1.5), or precision uplift on novel inputs (P1.2). All four are PROJECTED at the paper / TODO-30 estimated magnitude and require R18 GPU training to verify.
- **P1.2 veto is conservative by design** — ambiguous (fg, fg) pairs are treated as compatible (the MCTS can still fire); only blatantly-bad combinations are vetoed. The negative-false-positive invariant guarantees zero false rejections on the canonical educt set, but does NOT bound the false-acceptance rate on novel chemistry — that is a TODO-31 follow-up.
- **P2.5 stub is INDEPENDENT of R17 `conformer_embed`/`batched_rdkit`** — it uses covalent radii + numerical gradients so the stub runs on any CPU without OpenMM. The full OpenMM-based relaxation (with bonded terms + implicit solvent) is the R19 follow-up per the TODO-30 P2.5 caveat.
- **P1.5 fractional ODE** at α=1 reduces to integer-order Euler exactly; at α∈(0,1) the convergence rate `O(h^α)` is verified by `fractional_convergence_rate` in CI. The P1.5 wrapper integrates with PyTorch tensors via `numpy.asarray` at the call site; in-tensor GPU integration is the R18 follow-up.
- **P2.4 skeleton jump** is RDKit-sanitisable only — the swap-cut algorithm produces well-defined prefixes for chains but rejects aromatic-ring prefixes that leave broken aromaticity (noted in `verify_skeleton_jump.py` smoke fixtures). TODO-31 learned sampler is the fix for the ring-prefix case.

---

**Wf-12 close-out banner** — 4 真活 CPU skeleton shipped (P2.4 + P2.5 + P1.5 + P1.2); 99/99 tests pass; 4/4 verify scripts return 0 under `--smoke`; R18 GPU 训练 for the 4 PROJECTED magnitudes deferred to R18-P2-GPU (P2.4 + P2.5) and R19 (P1.5 + P1.2).
