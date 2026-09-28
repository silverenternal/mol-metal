# wf_lambda_cfm_coupling_audit — TODO-21 Lambda × CFM coupling 现状 audit

**Workflow:** Lambda × CFM coupling status audit (TODO-21 follow-up)
**Phase:** READ-ONLY audit
**Date:** 2026-09-26
**Scope:** Strategy 1 (Lambda-as-reward) + Strategy 3 (Cascade) **CPU joint inference skeleton** design + honest framing
**Owner:** R17 ultracode dispatch (audit-only, no code changes)

---

## TL;DR — coupling 状态一句话

The Lambda × CFM coupling **infrastructure ships today** (730 LOC coupling_adapter + 1003 LOC tmqm CPU pretraining + 1059 LOC LambdaASTSR reward shaper), with a **CPU-only smoke that exercises the full Lambda-MCTS → bridge adapter → 64-d embedding contract** (RMSE 0.0046 on PlatinAI 5K, 100 epochs, 5.9 s wall, 11/11 new tests pass, 30/30 decode probes). What is **deferred to post-GPU-retrain** is end-to-end joint inference that consumes real CFM ODE refine output — that gate is still pinned by `decode_ratio = 0/192` on the h=128 5000-step checkpoint per `wf_cfm_frontier_research/final.md` (2026-09-15). **Strategy 1 (Lambda-as-reward)** is CPU-only shippable today; **Strategy 3 (Cascade)** needs a CPU joint inference skeleton + smoke (designed below, **NOT YET implemented**); **Strategy 2 (Joint training)** remains deferred.

---

## §1 — Audit current coupling status

### 1.1 Shipped (CPU-verifiable, R17 close-out)

| Component | LOC | Status | Refs |
|---|---:|---|---|
| `molmetal/molmetal_lam/lam_chem/coupling_adapter.py` | 730 | SHIP — `CouplingAdapter` dataclass + `embed_pocket`/`embed_features` + `load_coupling_adapter` factory + `is_coupling_enabled` env gate + `_stub_adapter` fallback + **`lambda_samples_to_reward_targets` Strategy 1 entry point** | TODO-21 §3 |
| `molmetal/molmetal_lam/reward/lambda_astsr.py` | 1059 | SHIP — `LambdaAST` AST + `LambdaASTSR` Deflexpressor + `make_lambda_astsr_reward_channel` + `swap_f5_for_astsr` integration helper | R17 L-R17-B3 |
| `molmetal/scripts/tmqm_coupling_pretraining_cpu.py` | 1003 | SHIP — CPU-only PlatinAI 5K pretraining of 32-d λ → 64-d bridge adapter (Adam, 100 epochs, batch=64, cosine+MSE loss) | TODO-21 §C-R17-C1 |
| `models/coupling_adapter_platinai5k.npz` | 44 KB | SHIP — pretrain checkpoint (5.9 s wall, final train loss 0.00018, holdout RMSE 0.0046, well under 0.05 gate) | wf_coupling_pretrain_5k |
| `models/coupling_adapter_platinai5k.json` | 6.6 KB | SHIP — metadata sidecar (n_train=4000, n_holdout=1000, 100-epoch loss/rmse history) | wf_coupling_pretrain_5k |
| `molmetal/docs/architecture/lambda_cfm_cascaded.md` | 353 | DESIGN — Strategy 3 reference architecture (C-R17-C2) | TODO-21 §3 Strategy 3 |

### 1.2 Wired into orchestrator (env-gated, opt-in)

- `molmetal/molmetal_lam/search_alg/warm_start.py` — accepts `coupling_adapter` kwarg; `COUPLING_ENABLED=1` → use bridge embedding; default `COUPLING_ENABLED=False` → no-op stub.
- `molmetal/molmetal_lam/search_alg/learned_prior.py` — same env-gated drop-in.
- `molmetal/molmetal_lam/reward/learned_shaping.py` — `swap_f5_for_astsr()` swaps the F5 formula for the ASTSR; default is ASTSR (`USE_LAMBDA_ASTSR=1`).

### 1.3 Verified (CPU test suite, 2026-09-25)

- `molmetal/tests/test_coupling_adapter.py` — 5 tests (load, embed, stub, env-gate, shape contract) — pass.
- `molmetal/tests/test_coupling_adapter_platinai5k.py` — 11 tests (PlatinAI pretrain load, shape, RMSE ≤ 0.05, deterministic, finite-embedding) — pass.
- `molmetal/tests/test_lambda_reward_coupling.py` — smoke for the `lambda_samples_to_reward_targets` Strategy 1 entry point (Lambda MCTS → (SMILES, weight) tuples for CFM regression mix) — pass.
- `molmetal/tests/test_wf_coupling_ablation.py` — ASTSR R²=1.0000 vs F5 R²=0.9324 vs no-coupling R²=-0.0038; gate Δ=+0.0676 ≥ +0.05 — pass.
- Decode probes 30/30 = 1.000 across seed/step/h combinations.
- Bridge smoke (`wf_lambda_cfm_bridge_smoke.md`) — Lambda-as-reward end-to-end smoke PASS, 20 samples → dataset len 9 → sample shape (5,) (5,3).

### 1.4 Deferred (gated on GPU retrain)

- **Real CFM decode lift** — Path B chem-aware soft bond prior lifts decode to 192/192 bond-bearing on synthetic clouds (`wf_cfm_path_b_decoder_rework/final.md`), but **real-CFM lift unmeasured** because `decode_ratio = 0/192` on h=128 5000-step P1-fixed checkpoint.
- **GPU joint training** (Strategy 2, TODO-21 §3 line 56) — 5-10 d effort, 12-24 h GPU per cycle, no CPU-only fallback. **BLOCKED** on `decode_ratio > 0` gate.
- **End-to-end coupling on real CFM output** — not exercised in Round-13 100×3 sweep because CFM side has nothing to couple to.
- **True λ → CFM translator** — current bridge target is `Y = [lambda_32 || lambda_32]` (identity in lambda space); when GPU CFM training lands, target should be replaced by learned CFM pocket embedding.

---

## §2 — What coupling needs to be added for CPU joint inference (3 strategies by EV)

Per TODO-21 §3, three coupling strategies ranked by **effort ÷ expected lift** (lower is better). All three are documented; the ordering below matches the TODO-21 memo (Strategy 1 < 3 < 2 by EV).

### 2.1 Strategy 1 — Lambda-as-reward (RECOMMENDED, CPU-only shippable)

- **What:** Lambda MCTS generates β-NF candidates per pocket; QED / SA / PB-pass computed per candidate; product `r_qed * r_sa * r_pb ∈ [0, 1]` becomes a per-mol reward weight in the CFM regression mix.
- **Already shippable:** `lambda_samples_to_reward_targets(n_samples=200, pocket_id="…", reward_channels=("qed", "sa", "pb"))` returns `[(smiles, weight)]` tuples ordered by descending weight — 3-tier cascade (full MCTSProofSearch over 204-tile pool → CuAAC-only surrogate → STANDARD_12 seed fallback), CPU-only, NaN-safe, deterministic.
- **What's missing:** integration into the CFM training loop's velocity loss. TODO-21 §4 plan: 1-line adapter change in `flow_matching_lipman/__init__.py:_generate_impl` line 2017 area, pass `lambda_samples_reward` as auxiliary loss term with weight 0.1. 8 new tests + wiring = 1 d.
- **Expected lift (per TODO-21 §3 line 41):** PROBABLE +0.05 to +0.15 decode_ratio, +1 to +3 kcal/mol Vina lift — **NOT MEASURED** (reward weighting is a known technique; bounded by what Lambda's reward surface can express).
- **CPU-only payoff:** The `lambda_samples_to_reward_targets` module + 3-tier cascade + tests ship today. Paper §3.5 can describe the coupling architecture even without measured lift.

### 2.2 Strategy 3 — Cascaded Lambda → CFM-refine (PRAGMATIC MIDDLE, needs CPU joint inference skeleton)

- **What:** Two-stage pipeline — Stage A Lambda MCTS emits top-K β-NF SMILES; Bridge RDKit `AllChem.EmbedMolecule` → 3D init coords; Stage B CFM ODE refine 100-200 steps conditioned on pocket + typed-variable bond prior. Architecture doc: `molmetal/docs/architecture/lambda_cfm_cascaded.md`.
- **CPU joint inference skeleton** (designed in §3 below): 1 module `molmetal/molmetal_lam/lam_chem/lambda_cfm_joint_inference.py` (~150 LOC) that orchestrates Stage A → Bridge → Stage B with `cfm_inference_mode = "stub"` for CPU smoke (returns deterministic placeholder for refined coords + inherits Lambda's bond topology). 3 test paths (e2e smoke / stub determinism / bond-prior preservation).
- **What's missing:** the skeleton module + 3 tests. CPU-only, ~2 d effort, no GPU dependency. **GATED** on real CFM for the non-stub refine path.
- **Expected lift (per TODO-21 §3 line 92):** PROBABLE +1 to +2 kcal/mol Vina lift, +0.10 to +0.30 decode_ratio — **NOT MEASURED** (refine is well-known SBDD technique; novelty is typed-variable path constraint).
- **Why it's Strategy 3, not 1:** requires CFM forward pass to produce sane outputs (decode_ratio > 0 gate applies, even for stub); doubles inference cost (Lambda + CFM forward per candidate); Lambda doesn't learn from CFM refine failures (vs Strategy 2).

### 2.3 Strategy 2 — Joint Lambda × CFM training (BIG LEVER, BIG RISK, DEFERRED)

- **What:** Shared encoder + cross-attention between CFM hidden state and Lambda's β-NF state + joint optimizer with PCGrad multi-task loss. Per TODO-21 §3 line 56.
- **CPU joint inference?** **No** — every iteration is GPU-gated. CPU cannot back-prop through both branches.
- **Why deferred:** 5-10 d effort vs 1 week for Strategy 1; BLOCKED on `decode_ratio > 0`; architectural rework risk on 6217-line `proof_search.py`; no CPU-only fallback; multi-week debugging trap if joint loss doesn't converge.
- **What ships regardless of GPU:** the **architectural spec** in `molmetal/docs/architecture/lambda_cfm_cascaded.md` §6 pseudo-code (DESIGN ONLY) — useful for paper §7 future-work framing even without implementation.

### 2.4 Ranking summary (matches TODO-21 §3)

| Strategy | Effort | GPU? | Expected lift (PROJECTED, NOT MEASURED) | Status |
|---|---|---|---|---|
| 1. Lambda-as-reward | 1 week (3 d adapter + 2 d verify) | optional for retrain; adapter CPU-only | +0.05-0.15 decode_ratio, +1-3 kcal/mol Vina | **CPU-only entry point SHIP** (`lambda_samples_to_reward_targets`); integration gated |
| 3. Cascade | 2-3 d (skeleton + smoke) | required for non-stub refine | +0.10-0.30 decode_ratio, +1-2 kcal/mol Vina | **CPU skeleton design SHIP** (§3); implementation NOT YET STARTED |
| 2. Joint training | 5-10 d | required (every iteration) | +0.20-0.40 decode_ratio, +3-5 kcal/mol Vina | **DEFERRED** (no CPU-only path) |

---

## §3 — CPU joint inference skeleton design (Strategy 3, NOT YET IMPLEMENTED)

**Design only — no code shipped. Read-only audit output.**

### 3.1 Module interface

**Target:** `molmetal/molmetal_lam/lam_chem/lambda_cfm_joint_inference.py` (~150 LOC)

```python
@dataclass
class LambdaCFMJointConfig:
    """CPU joint inference config — stub CFM by default."""
    n_lambda_candidates: int = 5            # top-K from Stage A
    n_ode_steps: int = 200                  # refine steps (used iff cfm_inference_mode == "stub" returns init)
    cfm_inference_mode: Literal["stub", "real"] = "stub"   # CPU-safe default
    pocket_id: str = "test_001"
    seed: int = 0
    use_bond_prior_mask: bool = True        # Lambda typed-variable → CFM soft prior

@dataclass
class LambdaCFMJointOutput:
    """Per-pocket joint inference output."""
    pocket_id: str
    candidates: List[JointCandidate]        # (smiles, atom_types, init_coords, refined_coords, bond_prior_mask)

def lambda_cfm_joint_inference(
    pocket_id: str,
    coupling_adapter: CouplingAdapter,
    config: LambdaCFMJointConfig = LambdaCFMJointConfig(),
) -> LambdaCFMJointOutput:
    """Two-stage Lambda → CFM-refine CPU joint inference.

    Stage A: lambda_samples_to_reward_targets(n_samples=K) → top-K β-NF SMILES.
    Bridge:  RDKit AllChem.EmbedMolecule → (atom_types, init_coords).
    Stage B: cfm.refine_from_init(z, x_init, pocket_embed, bond_prior_mask, n_steps)
             when cfm_inference_mode="real"; else return (init_coords, identity bond_mask)
             so the orchestrator runs end-to-end on CPU.
    """
    ...
```

### 3.2 3 test paths (CPU-verifiable)

**Path 1 — e2e smoke (deterministic, CPU-only, < 5 s wall):**

```python
def test_lambda_cfm_joint_inference_smoke() -> None:
    """Smoke: top-3 Lambda candidates → embed → stub-refine → output shape contract."""
    adapter = load_coupling_adapter()  # real platinai5k.npz
    cfg = LambdaCFMJointConfig(n_lambda_candidates=3, pocket_id="test_001", seed=0)
    out = lambda_cfm_joint_inference("test_001", adapter, cfg)
    assert len(out.candidates) == 3
    for c in out.candidates:
        assert isinstance(c.smiles, str) and len(c.smiles) > 0
        assert c.atom_types.shape[0] == c.init_coords.shape[0]
        assert c.init_coords.shape[1] == 3
        assert c.refined_coords.shape == c.init_coords.shape
        assert np.isfinite(c.refined_coords).all()
        assert c.bond_prior_mask.shape[0] == c.atom_types.shape[0]
```

**Path 2 — stub determinism (bit-for-bit reproducibility):**

```python
def test_lambda_cfm_joint_inference_stub_determinism() -> None:
    """Two calls with same seed produce byte-identical refined_coords."""
    adapter = load_coupling_adapter()
    cfg = LambdaCFMJointConfig(n_lambda_candidates=2, pocket_id="test_002", seed=42)
    out1 = lambda_cfm_joint_inference("test_002", adapter, cfg)
    out2 = lambda_cfm_joint_inference("test_002", adapter, cfg)
    for c1, c2 in zip(out1.candidates, out2.candidates):
        np.testing.assert_array_equal(c1.refined_coords, c2.refined_coords)
        assert c1.smiles == c2.smiles
```

**Path 3 — bond-prior mask preservation (typed-variable → CFM soft prior):**

```python
def test_lambda_cfm_joint_inference_bond_prior_preserved() -> None:
    """The bond_prior_mask from Lambda's typed-variable annotations
    survives into the stub CFM output (CPU contract; real-CFM test gated)."""
    adapter = load_coupling_adapter()
    cfg = LambdaCFMJointConfig(n_lambda_candidates=2, pocket_id="test_003", seed=0,
                                use_bond_prior_mask=True)
    out = lambda_cfm_joint_inference("test_003", adapter, cfg)
    for c in out.candidates:
        # Bond prior must reference at least one typed-variable annotation.
        assert c.bond_prior_mask.sum() > 0
        # Mask must be shape-compatible with atom count.
        assert c.bond_prior_mask.shape == (c.atom_types.shape[0], c.atom_types.shape[0])
```

### 3.3 Implementation effort & dependencies

- **LOC:** ~150 (module) + ~120 (3 tests) = ~270 LOC total.
- **Effort:** 2 d (1 d module + 1 d tests + smoke verify).
- **Dependencies:** existing — `coupling_adapter.load_coupling_adapter` + `lambda_samples_to_reward_targets` + `conformer_embed.py` (RDKit `AllChem.EmbedMolecule`) + `decoder_rework.py` Path B bond-prior mask helper.
- **No new external deps.** Pure CPU. No torch / no numpy-incompatible calls.
- **GPU path:** when `cfm_inference_mode="real"`, dispatch to `flow_matching_lipman.refine_from_init(z, x_init, pocket_embed, bond_prior_mask, n_steps=200)`; CPU stub keeps the orchestrator runnable end-to-end on CI.

---

## §4 — Honest framing

### 4.1 CPU joint inference ≠ joint training

The CPU joint inference skeleton (§3) **is not** joint training. The distinction is critical:

| Aspect | CPU joint inference (skeleton) | Joint training (Strategy 2) |
|---|---|---|
| Lambda emits candidates? | YES | YES |
| CFM refines coords? | YES (stub on CPU; real CFM when GPU healthy) | YES |
| Gradient flow back to Lambda? | NO | YES (cross-attention + PCGrad) |
| Lambda learns from CFM failures? | NO | YES |
| GPU required? | NO (stub mode) | YES (every iteration) |
| Effort | 2-3 d (skeleton + smoke) | 5-10 d |
| Expected lift | PROJECTED +0.10-0.30 decode_ratio | PROJECTED +0.20-0.40 |

The skeleton is **a smoke harness for the cascade architecture** — it proves the orchestrator wires Stage A → Bridge → Stage B correctly, validates the 64-d embedding contract on the bridge side, and gives the test suite something to assert against. It does **NOT** validate that the CFM refine step produces better coords than Lambda's RDKit init.

### 4.2 What the smoke harness DOES claim

- Bit-for-bit determinism across runs (same seed → same refined_coords).
- Shape contract: `init_coords.shape == refined_coords.shape == (N, 3)`.
- Bond-prior mask survives the stub CFM step (the Lambda typed-variable annotation carries through).
- The 64-d bridge embedding is finite and shape-correct at every stage.
- End-to-end wall-clock under 5 s on CPU for top-3 candidates.

### 4.3 What the smoke harness DOES NOT claim

- **NO performance lift.** A skeleton that returns `init_coords` as `refined_coords` cannot improve Vina scores.
- **NO real CFM integration.** The stub mode is for CI / unit-test verification; the real CFM path requires `decode_ratio > 0` on a healthy GPU run.
- **NO feedback to Lambda.** The orchestrator runs one-way (Lambda → CFM), unlike Strategy 2 joint training.
- **NO bond-prior effectiveness measurement.** The mask is wired through but its downstream impact on real CFM decode is unmeasured.

### 4.4 Sequencing recommendation (matches TODO-21 §3 line 105)

1. **Ship Strategy 1 (Lambda-as-reward) adapter integration** — CPU-only, 1 d module + 1 d tests, reuses the shipped `lambda_samples_to_reward_targets` entry point. Lift measurement gated on GPU retrain.
2. **Ship Strategy 3 (Cascade) CPU joint inference skeleton** — 2 d module + 3 d tests, gives the cascade architecture a runnable harness on CI.
3. **When `decode_ratio > 0` gate lifts:** flip `cfm_inference_mode="real"` and measure Vina lift on 10×3 sweep.
4. **Strategy 2 (Joint training) — DEFERRED.** Only attempt after Round-13 λ-only + Round-14 cascade columns ship.

### 4.5 Why the audit's framing matters

Per `paper/sections/06_limitations.tex`, all coupling lift numbers in this report are `\PROJECTED{}` (per TODO-21 §3 + `lambda_cfm_cascaded.md` §5.2 honest framing), **not** `\MEASURED{}`. The shipped coupling infrastructure is real and verifiable; the lift projections are not. The Strategy 3 skeleton is designed but **NOT YET implemented** — this audit is the spec, not the ship.

---

## §5 — References

- TODO-21 strategy memo: `TODO/pending/21_lambda_model_coupling.md` (refreshed 2026-09-16).
- Cascade architecture design: `molmetal/docs/architecture/lambda_cfm_cascaded.md` (2026-09-17, DESIGN ONLY).
- Bridge pretraining report: `molmetal/reports/wf_coupling_pretrain_5k/final.md` (PlatinAI 5K, RMSE 0.0046, 5.9 s wall).
- Coupling shaper ablation: `molmetal/reports/wf_coupling_ablation/final.md` (ASTSR R²=1.0000 vs F5 R²=0.9324, Δ=+0.0676).
- Bridge smoke: `molmetal/reports/wf_lambda_cfm_bridge_smoke.md` (Lambda-as-reward e2e PASS).
- Phase 1 ship: `molmetal/reports/wf_lambda_cfm_coupling/final.md` (4 phases, 2026-09-16, ~25 min wall).
- Skill anchors: `mol-metal-lambda-calculus` (Lambda MCTS + Deflexpressor) + `mol-metal-cfm-training` (Lipman CFM + YuelBond bond head).

---

**End audit. Read-only. No code modified. Coupling infrastructure ships; CPU joint inference skeleton designed but NOT YET implemented; Strategy 2 joint training deferred.**
