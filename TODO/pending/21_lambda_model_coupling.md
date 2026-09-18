# TODO-21 — Lambda × 模型侧联合训练 (战略方向, refresh)

**Status:** 战略方向, deferred to post-Round-12-pilot, **MEMO refreshed 2026-09-16**
**Priority:** high (long-term strategic)
**Effort:** 1-week (Strategy 1) / 2-3 d (Strategy 3) / 5-10 d (Strategy 2)
**Owner:** (unset)
**Depends on:** GPU healthy + Round-13 paper shipped + decode_ratio > 0 gate lifted
**Blockers:** GPU time budget; `decode_ratio=0` on existing h=128 checkpoint per `wf_cfm_frontier_research/final.md` 2026-09-15

## 1. TL;DR — what coupling means

Lambda (算法侧, β-NF typed reductions + MCTS) 与 CFM (模型侧, EGNN + flow matching + bond head) 是 Mol-Metal 的两个**独立生成器**: Lambda 通过显式合成路径组合小分子; CFM 通过连续流匹配直接生成 3D 坐标 + 键. **"耦合"** 指让 Lambda 的合成路径信号进入 CFM 训练 / 推理, 让 CFM 的连续生成能力为 Lambda 的骨架提供几何完善 — 即"**算法 × 模型**"协同, 而非二者替换. 3 个候选策略按 EV 排序: **(1) Lambda-as-reward** < **(3) Cascaded Lambda→CFM-refine** < **(2) Joint training**.

## 2. Current state — what's shipped, what's blocked

**Shipped (A-line, 0 GPU):**
- `molmetal/scripts/r4_lambda_only_run.py` — Lambda-only orchestrator, 9 P0 metrics + SA weight + metal-seed + click-rule alias + n_sim=1000 hard cap
- `molmetal/molmetal_lam/lam_chem/decoder_rework.py` (926 LOC) — Path B chem-aware soft bond prior, lifts bond-bearing decode 0/192 → 192/192 on synthetic clouds per `wf_cfm_path_b_decoder_rework/final.md` (Δ=+1.0 absolute)
- `molmetal/molmetal_lam/lam_chem/pt_click_compat.py` — 5×5 Pt-click compat matrix + auto-detection
- `molmetal/adapters/coupling_adapter.py` — Lambda-CFM coupling adapter (drops into `warm_start.py` + `learned_prior.py`)
- 78/78 spot-check tests pass; Round-12 paper ships with λ-only column

**Blocked (model-side):**
- `decode_ratio = 0/192` on h=128 5000-step checkpoint per `wf_cfm_frontier_research/final.md` 2026-09-15 (Fix #1+#2+#3 active, Δ=0 vs baseline). Existing checkpoint's EGNN is upstream bottleneck.
- GPU healthy but flaky (one-shot probe 2026-09-15; SMU-hang root cause not permanently fixed per `wf_gpu_diag_fix`).
- Path B decoder lifts decode **off zero on synthetic clouds** but real-CFM decode lift unmeasured (gated on GPU retrain).

**Coupling adapter status:**
- `coupling_adapter.py` exists and is wired into `warm_start.py` + `learned_prior.py` (per `wf_phase3b` tasks #833-834)
- **Not yet exercised** in a real Round-13 100-pocket × 3-seed sweep because CFM side has nothing to couple to
- Decision: ship coupling infrastructure (CPU-verifiable), defer end-to-end coupling verification to post-GPU-retrain

## 3. 3 coupling strategies ranked by EV

### Strategy 1 — Lambda as reward channel for CFM retrain (RECOMMENDED)

**What it is:** Lambda MCTS generates 100-500 β-NF candidates per pocket (already does, 5×1 in 11s). Compute their QED / SA / PB-pass as **reward signal** for CFM retraining. CFM is told "**a sample like this** (Lambda output) is **good**, here's a sample like this (random atom cloud) that's bad". CFM learns the conditional distribution conditioned on being Lambda-similar.

**Implementation:** 1-line adapter change — pass `lambda_samples_reward` as auxiliary loss term in `flow_matching_lipman/__init__.py:_generate_impl`. Use **reward-weighted regression** on Lambda-generated (atom_type, coord) targets. Reuses existing `reward_aggregator.r_qed`, `r_sa`, `r_pb` channels.

**Effort:** **1 week** (3 d adapter + 2 d 5000-step CFM retrain at h=64 + 2 d verify)
**GPU:** ~6-12 h single-card (1× 5000-step run + 3-seed verify)
**Expected lift:** PROBABLE +0.05 to +0.15 decode_ratio (based on similar work in RxnFlow / SoftMol: reward-shaping narrows the search space, EGNN converges faster on task-relevant coords). +1 to +3 kcal/mol Vina lift via "Lambda priors propagate through the velocity field."

**Why it's Strategy 1:**
1. **Lowest architecture change** — doesn't require new modules, doesn't change CFM forward pass, doesn't touch bond head
2. **Reuses existing reward infrastructure** (Round-8 wire-clone-scoring gives us 24 REAL adapters; we already have `r_qed`, `r_sa`, `r_pb`, `r_reinvent4`)
3. **Independent of decode_ratio>0 gate** — reward-weighted training works even at 0/192 decode because Lambda provides its own valid samples as targets
4. **CPU-only verifiable** for the adapter part; only the GPU retrain is gated

**Why not higher EV than (2) or (3):**
- Reward shaping is a known technique; the +0.05-0.15 lift is bounded by what Lambda's reward surface can express
- Doesn't fundamentally restructure Lambda ↔ CFM information flow

### Strategy 2 — Joint Lambda × CFM trained jointly (BIG LEVER, BIG RISK)

**What it is:** Lambda MCTS and CFM share a **single training loop** where:
- CFM samples 3D atom clouds conditioned on the current β-NF state
- Lambda evaluates the CFM output for synthetic-feasibility + pocket-fit
- The combined loss back-propagates through both branches

**Implementation:** Per `wf_cfm_internal_review/diagnose.md` Phase 1 fixes, requires:
- Shared encoder (pocket embedding used by both Lambda's `proof_search.py` and CFM's `velocity_net.py`)
- Cross-attention between CFM hidden state and Lambda's β-NF state
- Joint optimizer with PCGrad multi-task loss balancing

**Effort:** **5-10 days** (2 d architecture + 3 d training + 2 d verify)
**GPU:** ~12-24 h single-card (3 retrain cycles × 10000-step)
**Expected lift:** PROJECTED +0.20 to +0.40 decode_ratio (joint optimization > sum-of-parts). +3 to +5 kcal/mol Vina lift (projected). **NOT MEASURED.**

**Why it's the BIG LEVER:**
- Architecturally the most aligned with how humans do metallodrug design (think combinatorially → refine geometry → re-evaluate combinatorial moves)
- Captures feedback loops that reward shaping alone can't (Lambda learns to emit candidates the CFM can refine)

**Why it's Strategy 2, not 1:**
1. **5-10 d effort** vs 1 week for Strategy 1
2. **Blocks on decode_ratio>0 gate** — joint training needs both branches to produce meaningful gradients
3. **Architectural rework risk** — current `proof_search.py` is 6217 lines; touching it without regressing Round-12 λ-only is dangerous
4. **No CPU-only fallback** — every iteration is GPU-gated

### Strategy 3 — Cascaded Lambda → CFM-refine (PRAGMATIC MIDDLE)

> **Architecture doc (2026-09-17):** `molmetal/docs/architecture/lambda_cfm_cascaded.md` — full design including ASCII flow diagram, two-stage implementation roadmap, blockers, honest framing, and reference pseudo-code. **DESIGN ONLY — no code, no measurements.**

**What it is:** Two-stage pipeline:
1. Lambda MCTS generates β-NF candidate (synth-valid, click-rule-compliant) — already does this, 5-50 candidates per pocket in 11-50s
2. CFM takes the Lambda candidate's atom types + rough 3D coords, **refines** the coordinates via 100-200 ODE steps conditioned on the pocket + the typed-variable constraints
3. Output: refined 3D mol that respects Lambda's synthetic pathway

**Implementation:** CFM forward takes `(atom_types, init_coords_estimate, pocket)` → refined `(coords, bonds)`. Init coords estimated via RDKit `AllChem.EmbedMolecule` on the SMILES (already exists in `decoder_rework.py`).

**Effort:** **2-3 days** (1 d adapter + 1 d 200-step ODE refine + 1 d verify)
**GPU:** ~2-4 h single-card (per-pocket refine is fast)
**Expected lift:** PROBABLE +1 to +2 kcal/mol Vina lift (geometries refine toward pocket, dock scores drop). +0.10 to +0.30 decode_ratio (typed-variable constraints remove ambiguity).

**Why it's Strategy 3, not 1:**
- Requires CFM side to work at all (decode_ratio > 0 gate applies, even if just for refine)
- Refine is a known SBDD technique (TargetDiff §4.3, FLOWr §3.4); no novel contribution
- Doubles the inference cost (Lambda + CFM forward per candidate)

**Why it's not Strategy 2:**
- Doesn't capture feedback (Lambda doesn't learn from CFM refine failures)
- CFM treats Lambda output as fixed input, not as a distribution

## 4. Recommended next action: Strategy 1 (Lambda-as-reward)

**Why:** Lowest EV-positive ratio (effort / lift ratio), **CPU-only-verifiable adapter change** (can ship without GPU), **independent of decode_ratio gate** (Lambda provides its own valid samples as reward targets), reuses existing reward infrastructure (Round-8 wire-clone-scoring → 24 REAL adapters).

**Concrete plan when GPU returns:**
1. **Day 1-3 (CPU):** Add `lambda_reward_shaping` module to `molmetal/adapters/flow_matching_lipman/__init__.py:_generate_impl` line 2017 area. Wire `r_qed + r_sa + r_pb` from `coupling_adapter.py` outputs into the velocity loss with weight 0.1. Tests: `test_lambda_reward_shapes_loss.py` (5 tests), `test_lambda_samples_are_valid_targets.py` (3 tests). 0 GPU.
2. **Day 4-5 (GPU, gated):** Re-run 5000-step diagnostic with `--lambda-reward-weight 0.1` + `--n-train 32` + `--n-samples 16` × 3 seeds. Compare `decode_ratio` + `bond_loss` vs the `wf_gpu_recovery_now` baseline (0/192).
3. **Day 6-7 (analysis):** If `decode_ratio > 0.05` → PROMOTE to Round-13 sweep column. If still 0 → BAIL to Strategy 3 (cascaded) or stay on λ-only.

**What gets shipped regardless of GPU:** The `lambda_reward_shaping` module + 8 tests + integration into `coupling_adapter.py`. This is the **CPU-only payoff**: paper §3.5 can describe the coupling architecture even without measured lift.

## 5. Blockers

| Blocker | Status | Resolution |
|---|---|---|
| GPU healthy | Flaky; one-shot recovery 2026-09-15 per `wf_gpu_recovery_now`; SMU-hang root cause not permanently fixed | Wait for stable 24-hour probe; alternative: iGPU via `detect_active_gpu()` selector (currently latent) |
| 5000-step retrain budget | 6-12 h GPU; Path B lift PROBABLE but unmeasured | After Strategy 1 ships, re-run `wf_gpu_recovery_now` style probe with `--lambda-reward-weight 0.1` |
| 5-click SMARTS integration | Already in `lam_chem/decoder_rework.py:419-465` and `pt_click_compat.py`; 5×5 matrix tested | DONE — no blocker |
| Lambda-CFM joint-test set | `coupling_adapter.py` exists; integration tests in `test_coupling_adapter.py` (5 tests) | DONE — no blocker |
| Round-13 100-pocket sweep budget | Estimated 100×3 = 300 evaluations × 30s = 2.5 h wall; needs GPU + docking licenses (Vina/QVina both available per D7) | Schedule after Round-13 λ-only column ships |

## 6. Risk assessment per strategy

**Strategy 1 risks:**
- **Reward hacking:** CFM learns to produce mols that look good to QED/SA but aren't chemically meaningful. Mitigation: use PB-pass as a hard filter, not as differentiable reward. PB is non-differentiable, so use it as post-filter only.
- **Mode collapse:** CFM concentrates on a few high-reward modes. Mitigation: include `r_diversity` (homotype + Tanimoto) as auxiliary reward. Already in Round-9 P0 metrics.
- **Lift below noise floor:** +0.05 decode_ratio might be within 3-seed std. Mitigation: require 5 seeds + paired t-test for significance.

**Strategy 2 risks:**
- **Joint training instability:** Combined loss may not converge. Mitigation: PCGrad multi-task loss per `wf_vina_lift_phase23/pac_bayes.md`.
- **Proof-search regression:** Touching `proof_search.py` (6217 lines) may regress Round-12 λ-only. Mitigation: ship Strategy 1 first; only attempt Strategy 2 after Round-13 ships.
- **Multi-week debugging trap:** 5-10 d effort may balloon to 3 weeks if architecture rework requires multiple iterations.

**Strategy 3 risks:**
- **Cascaded failure modes:** If Lambda emits invalid scaffold (rare but happens with strict Pt_II), CFM refine can't recover. Mitigation: Lambda MCTS hard-passes PB before cascade.
- **Inference cost:** 2-stage pipeline doubles wall time per candidate. Mitigation: cache Lambda output (reused across N refine attempts).
- **No feedback to Lambda:** Lambda doesn't learn from CFM refine failures (vs Strategy 2). Acceptable for paper-grade; suboptimal for long-term.

## 7. 5 user-decision points before any strategy starts

These are blocking questions for the user. None of the strategies can start until the user signs off:

1. **D8** — Which coupling strategy? (1=reward, 2=joint, 3=cascade, or "ship all 3 in order"). Recommendation: Strategy 1.
2. **D9** — Lambda-reward weight? (default 0.1, range 0.01-1.0). Higher = more aggressive shaping. Recommendation: 0.1 first, sweep if budget allows.
3. **D10** — Lambda sample count per pocket? (default 50, range 10-500). More = better reward signal but slower. Recommendation: 50 first.
4. **D11** — Joint-train vs Lambda-only-pretrain-then-finetune-CFM? (Two-phase alternative to Strategy 2). Recommendation: defer until Strategy 1 lift measured.
5. **D12** — Cascade inference mode for Strategy 3: Lambda+MCTS picks top-K, CFM refines all K, return best by Vina? Or: Lambda picks 1, CFM refines with N=10 seeds, return best? Recommendation: first (top-K) — reuses Round-12 pipeline.

## 8. Update protocol

Append-only. Future entries below this line preserve history.