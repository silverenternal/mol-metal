# TODO-150 — Dream-RSI Integration Plan for mol-metal

**Status:** PLAN drafted 2026-09-27 (post-Wf-124 P3, pre-Wf-125)
**Source paper:** arXiv:2609.14858 — Zheng, Wu, Zhang et al., *"Dream-RSI: Recursive Self-Improvement through Evolving Worlds"* (Sep 2026)
**Reference analyses:** three independent analyses consolidated below (training-acceleration, quality-methods, applicable-techniques)
**Author-attribution caveat:** paper is **NOT Google-affiliated** — author list is academic. The user's "Google's dream-rsi" phrasing is **UNVERIFIED**. Plan proceeds on the assumption the user means this paper (closest conceptual fit to recursive-self-improvement for training); alternative readings listed in §5.

---

## §0 Paper summary (3 bullets)

- **What it is.** A framework for *recursive self-improvement of exploration policies* over an accumulated discovery-tree replay buffer. The underlying agent (e.g. a coding LLM) is left untouched; a lightweight orchestration layer wraps it, runs "dreams" by deterministically replaying cached historical trajectories, and uses the resulting off-policy feedback to score and refine candidate policies without paying for fresh online rollouts. Improved policy is redeployed, the buffer grows, the loop tightens.
- **Headline result.** Competitive or improved solution quality across three domains (algorithm engineering, mathematical optimization, GPU kernel engineering) while *substantially* reducing discovery cost (50×–162× call reduction reported in some settings) by replacing repeated expensive online evaluations with cheap replay-based scoring.
- **What it is *not*.** It is not a synthetic-data generator — there is no learned world model fit, no dreamed molecules, no imagined trajectories. Replay is strictly deterministic traversal of recorded history. It is also not a search-algorithm rewrite: the exploration policy is *programmable* but the underlying domain agent is unchanged.

---

## §1 Applicable techniques (sorted by impact)

Ranked by **expected net lift to the 5-metal × 3-seed × 30-epoch training program** minus **risk + implementation cost**. All techniques borrow the *replay-simulator* pattern from Dream-RSI; none of them claim the 50×–162× call-reduction headline because mol-metal's bottleneck is GPU-bound D-MPNN training, not LLM agent calls.

| # | Technique (Dream-RSI primitive → mol-metal mapping) | Net rank | Why this rank |
|---|---|---|---|
| **A1** | **Replay-buffer off-policy evaluation for D-MPNN config search.** Cache every (config, per-epoch metric trajectory, final test score) from Wf-120/Wf-124 runs into a queryable pool; evaluate next round's config deltas (LR, hidden_dim, depth, loss weights) by deterministic traversal + cheap gradient steps on cached weights before committing to a full retrain. | **#1 IMPACT** | Largest per-cycle compute saving in absolute hours. Maps onto the 5-metal × 3-seed matrix now being scoped (commit `f06ce77`). Paper's "single costly online run enables thousands of rapid off-policy evals" maps directly onto "one 30-epoch Ru/Ir/Rh/Os/Re run enables fast pruning of obviously-bad configs before the next round." |
| **A2** | **Replay-pool reward model for cheap candidate ranking (replace repeated Vina calls).** Cache (smiles, D-MPNN embedding, sub-pocket features, Vina/pIC50 score) triples; train a lightweight surrogate (XGBoost on RDKit features, or small MLP head on frozen D-MPNN message-passing embeddings) for off-policy ranking; call the expensive Vina + scoring oracle only on top-K. | **#2 HIGH** | The single biggest cost lever. Most direct translation of Dream-RSI's replay-evaluator-replaces-online-call insight. Uses real recorded (molecule, score) pairs — no synthetic labels, no dreamed molecules (paper explicitly avoids generative world models). |
| **A3** | **Per-metal discovery-tree bootstrap for D-MPNN hyperparameter sweeps.** Treat each per-metal (epoch → val_mae → metal) trajectory as a discovery-tree node; pre-screen LR schedules, dropout, head-share, loss-balancing weights before kicking off the next full D-MPNN run. | **#3 HIGH** | Operates on existing `dmpnn_ckpt_status.json` (25 ckpts, 5 metals × 5 seeds, per Wf-120 P2) plus Wf-124 P3's fresh Ru seed=0 result (test_auc=0.7613, commit `1a8eebd`). Reuses recorded history — no new oracle cost. |
| **A4** | **Adaptive compute allocation with plateau detection.** When validation AUC plateaus, automatically expand augmentation pool or activate harder pockets; when improvement is steady, prune expensive branches. Mirrors Figure 6's adaptive-effort curve (110 → 50 attempts when improving, back to 110 when plateaued). | **#4 MED** | Cheap to implement; saves compute rather than directly raising quality. Maps onto the paper's 1.79×–2.43× fewer-generation result on VGG16/LayerNorm. |
| **A5** | **Off-policy fine-tune policy eval for DrugOOD 1.58M → MetalCytoToxDB 5K transfer.** Cache (pre-train checkpoint, fine-tune subset, fine-tune HP, post-fine-tune pIC50 MAE) triples; rank candidate fine-tune configs (freeze depth, LR schedule, replay ratio) by replay-dream before committing GPU. | **#5 MED** | Cuts trial-and-error on the 5K fine-tune corpus where each cycle is expensive relative to data size. Supports TODO-23 weak-to-strong plan without running every strategy to convergence. |
| **A6** | **Replay-disagreement as epistemic-uncertainty estimator.** Diversity of replay outcomes across alternative sampling policies = proxy for uncertainty. Flag disagreement-high molecules/pockets for prioritized oracle evaluation next round. | **#6 LOW-MED** | Replaces heuristic uncertainty without a separate Bayesian D-MPNN. Active-learning signal without extra oracle calls. |
| **A7** | **Domain priors extracted from historical discovery trees.** Walk the cached tree to extract (a) high-pIC50 metal-ligand substructures, (b) best-discriminating pockets, (c) scaffold mutations correlated with score gains. Encode as soft priors. | **#7 LOW-MED** | Paper §5.1 ablation shows prompt-level semantic guidance *underperforms* replay-simulator learning. Therefore priors must be derived from replay data structure (counts, branches, leaf scores), not injected as textual rules. Use with caution. |
| **A8** | **Replay-simulator over historical docking trajectories for Lambda × CFM × MCTS framework.** Wrap MCTS selection policy in an orchestration layer that "dreams" by replaying cached docking + pIC50 records instead of invoking Vina for every rollout. | **#8 MED (gated)** | 10–100× Vina-call reduction in principle. **GATED** on (a) Lambda pipeline currently in deployment state and (b) CFM `decode_ratio>0` gate per TODO-21. Not actionable until CFM unblocks. |
| **A9** | **Cost-aware multi-objective replay schedule (learned early-stopper + adaptive LR).** Treat each D-MPNN epoch as a decision round; predict plateau epoch from historical replay pool; use as learned threshold. | **#9 LOW** | 1.2–1.5× wall-clock saving per run; modest. |
| **A10** | **OOD-split-aware validation-set selection via replay dreaming.** Use historical (smiles, scaffold, assay, size-bucket, OOD-fold, model_score) records to pick which DrugOOD scaffold/assay/size hold-out is most informative for early stopping, without paying a full OOD evaluation pass. | **#10 LOW** | OOD splits are deterministic functions of the dataset, so the "dream" is closer to statistical analysis than a learned policy; modest upside unless paired with search-loop integration. |
| **B1** | **LLM-driven policy-code rewrite as meta-controller.** Use an LLM/RL meta-controller to propose new training scripts or sweep configs each round. | **NOT RECOMMENDED** | Paper's 162× call reduction comes from agent-call economics that don't apply to GPU-bound D-MPNN. Marginal benefit on the project's tightly-typed Chemprop + D-MPNN stack; effort 120h, expected 1.1–1.3× low-confidence. Defer. |
| **B2** | **Synthetic-trajectory data augmentation from a learned world model.** Generate fake training trajectories. | **NOT APPLICABLE** | Paper explicitly does NOT do this; no generative world model. Closest analog is A2 (replay-pool reward model) which uses *real* recorded pairs. If synthetic molecule generation is desired, look to graph-generative-model literature, not this paper. |

---

## §2 Integration plan with concrete steps

### Phase P0 — Replay-pool infrastructure (BLOCKER for A1–A10, 12-18 h)

P0 is unconditional; every downstream technique depends on it.

- **P0.1** (4 h) Build `ReplayPool` class in new `molmetal/replay/pool.py` with:
  - `add(trajectory: TrajectoryRecord)` — append (config_hash, hp_dict, per_epoch_metrics, final_test_score, metal, seed, timestamp)
  - `query(filter: dict) -> DataFrame` — pandas-backed for cheap groupby
  - `coverage_report(metal: str) -> CoverageStats` — per-metal cell count, scaffold diversity, score variance
  - `replay_predict(hp_delta: dict, metal: str) -> ScoreEstimate` — deterministic lookup of nearest neighbor + bias-corrected extrapolation
- **P0.2** (3 h) Backfill from existing checkpoints: walk `dmpnn_ckpt_status.json` (25 ckpts per Wf-120 P2), `metal_lomo_auc_wasserstein.json`, `metalcytotox_combined_r2.json`, per-metal `metalcytotox_<metal>_auc.json`. Should yield **>200 trajectories** with config + final score.
- **P0.3** (3 h) Wire `ReplayPool.add()` into the Wf-124 P3 trainer (`molmetal_lam/training/dmpnn_multitask.py`) so every future run appends automatically. Add `--replay-pool-output path` CLI flag, default ON.
- **P0.4** (2-4 h) Per-metal coverage diagnostics: flag metals with <10 trajectories as `LOW_COVERAGE` and apply pessimistic shrinkage to replay predictions on those metals (see §4).
- **P0.5** (2 h) Tests: `test_replay_pool_add_query.py`, `test_replay_pool_coverage.py`, `test_replay_pool_replay_predict.py`.

**Exit gate:** `ReplayPool` populated, `coverage_report` printed per metal, all A1–A10 techniques can now read from a single source of truth.

### Phase P1 — A1 + A3: D-MPNN config search via replay (HIGH PRIORITY, 18-30 h)

P1 ships the techniques with the highest expected compute-saving per effort-hour.

- **P1.1** (6 h) **A1 wrapper.** New `molmetal/scripts/r4_dmpnn_replay_sweep.py` — takes a list of config deltas, scores each by `ReplayPool.replay_predict()`, ranks, optionally promotes top-N to a real Wf-124-style run. `--replay-only` (no GPU) and `--replay-then-verify` (top-3 to GPU) modes.
- **P1.2** (8 h) **A3 per-metal bootstrap.** Extend `ReplayPool` with `per_metal_bootstrap(metal, hp_delta) -> Estimate` that uses leave-one-out within metal: for each metal, predict fold-out performance by averaging neighbors, then compute mean-improvement-proposal for the candidate HP delta.
- **P1.3** (4-8 h) **Verify on Ru seed=0 baseline.** Take Wf-124 P3's Ru seed=0 result (commit `1a8eebd`, test_auc=0.7613) as held-out; replay-predict it from seeds 1-4 alone; confirm prediction falls within ±0.02 AUC. **If error > 0.05, halt and audit replay-pool coverage** before proceeding.
- **P1.4** (4 h) Tests + integration into next Wf-125 sweep: `replay_sweep_dryrun` should rank 10 candidate HPs in <60s with no GPU.
- **P1.5** (2-4 h) Write `molmetal/reports/wf_dream_rsi_p1_dmpnn_replay/` with `MASTER.md` + per-step measurement JSON.

**Exit gate:** on Wf-125's 5-metal × 3-seed × 30-epoch matrix, A1+ A3 must correctly shortlist the top-2 LR schedules and reject ≥1 of the bottom-2 from the candidate pool *before* the full GPU run. Compare wall-clock saving vs always-on 5×3=15 runs.

### Phase P2 — A2: Vina-call reduction via replay-pool reward model (HIGH PRIORITY, gated on P0, 35-50 h)

P2 is the single biggest cost lever in absolute oracle-hours saved. Gated on P0; can start as soon as P0.5 closes.

- **P2.1** (8 h) Build `RewardSurrogate` class in `molmetal/replay/surrogate.py`:
  - Input: frozen D-MPNN message-passing embedding + RDKit features + pocket one-hot + metal one-hot
  - Architecture choice (pilot both): XGBoost on RDKit features (~200h training); small MLP head on frozen D-MPNN embeddings (~2h training)
  - Loss: MSE on cached (smiles → Vina/pIC50) pairs from `vina_high_affinity*.json` + `metalcytotox_*_auc.json`
- **P2.2** (8 h) Backfill training data: extract all cached (smiles, score) pairs from existing metrics files. Target ≥5000 labelled pairs across the 5 metals.
- **P2.3** (6 h) Wire surrogate into MCTS rollout in `proof_search.py` (`r_vina` channel pattern, parallel to `r_qed` / `r_sa` / `r_pb` / `r_reinvent4`):
  - At rollout selection time, score candidate by surrogate first; only invoke real Vina for top-K (default K=20)
  - `--surrogate-topk` CLI flag, default K=20 (validate vs K=5, K=50 in P2.5)
- **P2.4** (8 h) Surrogate retraining loop: every N rollouts (default N=100), refit surrogate on the new (smiles, Vina score) pairs added to the replay pool. Self-improving loop mirrors Dream-RSI §3.
- **P2.5** (5-8 h) A/B test on 2-3 known-Pt-drug regression pockets from `TODO/completed/30_pitfall_reinforce_plan.md` Rank 3 (cisplatin, carboplatin, oxaliplatin). Compare MCTS+surrogate vs MCTS-only on: (a) wall-clock, (b) number of Vina calls per 100 candidates, (c) pocket-mean Vina kcal/mol. Acceptance: surrogate route uses ≤50% of Vina calls for ≤0.3 kcal/mol pocket-mean degradation.
- **P2.6** (4 h) Tests + write `molmetal/reports/wf_dream_rsi_p2_surrogate/MASTER.md`.

**Exit gate:** MCTS rollout loops use surrogate-ranked candidates with verifiable Vina-call reduction ≥2× on 3 known-Pt-drug pockets, no pocket-mean Vina regression beyond 0.3 kcal/mol.

### Phase P3 — A4 + A9: Adaptive compute (LOW-EFFORT, 12-20 h)

Can run in parallel with P2 once P0 closes.

- **P3.1** (8 h) **A4 plateau detector.** Track val_mae per-epoch trajectory; fit a piecewise-linear model on `ReplayPool` neighbors with same config-signature; trigger early-stop when projected next-epoch improvement < `--plateau-eps` (default 0.005) AND current epoch > `--min-epoch` (default 10). Wire into trainer as `--adaptive-epoch-budget`.
- **P3.2** (6 h) **A9 learned early-stopper.** Same machinery as P3.1 but the trigger threshold is itself replay-predicted per config.
- **P3.3** (4-6 h) Verify on a 5-seed re-run: expect 15-30% epochs saved on average; expect ≥1 config to plateau 5+ epochs earlier than the fixed-30 baseline.

**Exit gate:** epoch budget adaptive by config, no AUC degradation on Wf-120/Wf-124 reference runs.

### Phase P4 — A5 + A6 + A7 (MEDIUM, 24-40 h, all gated on P0)

- **P4.1** (10 h) **A5 off-policy fine-tune eval.** Wire into `TODO/completed/05_tmqm_pretraining.md` weak-to-strong pipeline (TODO-23). Replay-predict fine-tune gain from (pre-train ckpt, fine-tune subset, HP) → post-fine-tune MAE triples; only commit GPU on positive predicted gain.
- **P4.2** (8 h) **A6 replay-disagreement uncertainty.** Use A4 + A5's replay pool to compute, for each candidate molecule/pocket, the variance across `K=5` replay-predicted scores from randomly-sampled historical configs that observed similar (smiles, pocket) pairs. Flag disagreement-high for prioritized oracle evaluation.
- **P4.3** (6-8 h) **A7 domain priors.** Extract high-pIC50 substructure frequencies from replay pool; encode as soft prior in `molmetal_lam/training/dmpnn_multitask.py` message-passing initialization (NOT as prompt-level semantic guidance per paper §5.1 warning). Validate prior-vs-no-prior on Ru seed=0.

### Phase P5 — A8 (DEFERRED, gated on Lambda × CFM unblock)

- **P5.1** (size TBD, gated on TODO-21 Strategy 1 closing + CFM `decode_ratio > 0` gate) **A8 Lambda × CFM × MCTS replay.** Re-implement as soon as CFM unblocks; reference design pre-staged in `molmetal/docs/architecture/lambda_cfm_cascaded.md`.

### Phase P6 — A10 (OPTIONAL, 8-12 h)

- **P6.1** OOD-split-aware validation-set selection via replay dreaming. Low priority; defer to Round-13 unless P0-P5 close early.

---

## §3 Expected improvements (quantified)

Honest-framing: all numbers below are **projections, not measured**. The paper's 50×–162× call-reduction figure is *not directly transferable* to mol-metal because (a) mol-metal's bottleneck is GPU minutes, not LLM API calls; (b) mol-metal's replay pool at P0-start has ~200 trajectories vs the paper's >10000 per task.

| Technique | Projected speedup | Projected quality lift | Confidence | Cite-only vs measured |
|---|---|---|---|---|
| **A1 replay-config-search** | 2-5× fewer 30-epoch runs needed per round to find a competitive HP | +0.01-0.03 AUC on Ru seed=0 vs Wf-124 baseline 0.7613 | Medium | PROJECTED (paper claim 10-50×, but mol-metal's regime smaller) |
| **A2 Vina-call reduction** | 5-20× fewer Vina calls per MCTS rollout (paper claims 50× in some settings) | ≤0.3 kcal/mol pocket-mean degradation (acceptance criterion P2.5) | Medium-High | PROJECTED (depends on surrogate quality; will be measured in P2.5) |
| **A3 per-metal bootstrap** | 2-5× fewer compute-hours per hyperparameter sweep | None directly; quality-preserving | High | PROJECTED |
| **A4 plateau detection** | 1.2-1.5× wall-clock per training run (15-35% epochs saved) | None directly | High | PROJECTED (mechanism is well-understood) |
| **A5 fine-tune replay eval** | 2-4× fewer fine-tune cycles to pick best weak-to-strong strategy | TBD; expect +0.01-0.03 Pearson on 5K fine-tune | Medium | PROJECTED |
| **A6 replay-disagreement uncertainty** | Cheap active-learning signal | +1-3% recall on hard examples per round | Low-Medium | PROJECTED |
| **A7 domain priors** | None | +0.005-0.02 AUC on Ru | Low | PROJECTED with paper §5.1 caveat |
| **A8 Lambda × CFM replay** | 10-100× fewer Vina + oracle calls per MCTS generation step (paper headline regime) | TBD; depends on Lambda + CFM status | Low (gated) | NOT MEASURABLE until P5 unblocks |
| **A9 learned early-stopper** | 1.2-1.5× wall-clock per training run | None directly | High | PROJECTED |
| **A10 OOD-split selection** | Modest | Modest | Low | PROJECTED |
| **B1 LLM meta-controller** | 1.1-1.3× | n/a | Low | NOT RECOMMENDED |

**Net expected project impact (P0 + P1 + P2 + P3 closed):**
- Wf-125 (5-metal × 3-seed × 30-epoch matrix): expected **30-50% wall-clock reduction** vs always-on 15 full runs
- MCTS candidate generation round: expected **5-10× Vina-call reduction** for ≤0.3 kcal/mol pocket-mean Vina degradation
- Per-round hyperparameter search: expected **2-5× fewer compute-hours** to reach a competitive HP

**Cite-only claims from the paper** (NOT measured in mol-metal):
- "50×–162× discovery-cost reduction" — paper claim, not transferable headline; mol-metal's regime has different bottleneck
- "Competitive or improved quality across 3 domains" — paper claim, mol-metal regime is 1 domain
- "1.79×–2.43× fewer generations for comparable quality" — paper claim; closest analog in mol-metal is A9

**Honest summary:** Dream-RSI's *mechanism* (replay-buffer as cheap off-policy evaluator; programmable orchestration layer; self-improving loop that grows the buffer) is **highly applicable** to mol-metal's training loop. The paper's *headline numbers* are not directly transferable; mol-metal's projected speedups are 1.2×–5× in the techniques, not 50×–162×. The compounding self-improvement claim is **plausible but not validated** in mol-metal's GPU-bound regime.

---

## §4 Risks + mitigations

| # | Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|---|
| **R1** | **Replay-pool coverage collapse.** If cached records under-represent certain metal/pocket cells, replay predictions will be optimistic for those cells and degrade real generalization. | High (Ru has 2509 train rows, Re has 121 — 20× gap) | High | (a) P0.4 per-metal coverage diagnostics flag `LOW_COVERAGE`. (b) Apply pessimistic shrinkage: replay prediction for metal M is `coverage(M)/max_coverage * replay_prediction + (1 - ...) * baseline_prior`. (c) Never promote a replay-only decision on a `LOW_COVERAGE` cell without a verify-run confirmation. (d) Document per-metal replay-coverage in every `MASTER.md` report. |
| **R2** | **Spurious historical correlations in replay pool.** D-MPNN training dynamics are highly non-stationary across metals; replay can pick up patterns like "Ru always likes lr=1e-4" that don't generalize. | Medium | High | (a) Treat replay predictions as guidance only; require validate-on-a-held-out-run for every promotion. (b) Use leave-one-out within metal in A3 to detect overfitting. (c) Maintain a "replay-vs-actual" delta log; if median delta > 0.05 AUC across 5+ promotions, halt and audit. |
| **R3** | **Cargo-cult adoption of paper architectures.** Copying verbatim the paper's value-function architecture or RL algorithm choice without an implementation-level read of the full PDF. | Medium | Medium | (a) Each integration technique is mapped to mol-metal's existing stack (XGBoost on RDKit, MLP head on D-MPNN, etc.), not paper-specific architectures. (b) Defer paper-specific architecture grafts to after full PDF read. |
| **R4** | **Diminishing returns after 2-3 cycles.** Unlike paper's open-ended algorithm search, mol-metal's oracle (Vina + CytoTox) is bounded by physical chemistry. Successive Dream-RSI-style self-improvement rounds may plateau once per-metal replay coverage saturates. | Medium | Low | (a) Track per-cycle gain; if cycle-3 gain < 50% of cycle-1 gain, halt RSI expansion. (b) Budget: 3 cycles max per metal before requiring human-in-the-loop audit. |
| **R5** | **Surrogate reward model overfit (A2).** XGBoost/MLP surrogate trained on cached pairs may overfit to training-pocket-specific patterns and underperform on novel pocket geometries. | Medium-High | High | (a) Cross-pocket validation in P2.5 (3 known-Pt-drug pockets are not in the surrogate's training set). (b) Refit surrogate every N rollouts (default 100) to track distribution shift. (c) Fallback to real Vina when surrogate confidence (e.g. distance to nearest training neighbor) is below threshold. |
| **R6** | **Stale replay pool on data drift.** New pockets / new metal-coordination contexts added between rounds will not be in the replay pool; replay predictions for them will be unreliable. | Medium | Medium | (a) Recompute `coverage_report` at start of each round; flag new contexts. (b) Require N≥5 historical trajectories per (metal, pocket-class) before using replay for that cell. |
| **R7** | **Pretending Google's dream-rsi when the paper isn't Google-affiliated.** User phrasing suggests Google involvement; the paper's author list is academic. If the user meant a Google paper specifically, we have built the plan on the wrong foundation. | Low-Medium | Low | (a) Section §5 explicit honest-framing note. (b) Alternative-readings list. (c) This plan should be reviewed against the user's intent before Phase P1 commit. |
| **R8** | **arXiv-only preprint with no independent reproduction.** arXiv:2609.14858 is brand new (Sep 2026) with no independently reproduced numbers yet. Cost-reduction claims are hypothesis, not guarantee. | Medium | Low-Medium | (a) Frame all projections as PROJECTED, not MEASURED. (b) Treat 5×–20× Vina-call-reduction as the *upper bound* of what A2 might achieve; do not promise 50×. (c) If paper gets retracted or substantially corrected within 6 months, re-audit. |
| **R9** | **A8 (Lambda × CFM replay) is doubly gated.** Requires both TODO-21 unblock AND CFM `decode_ratio > 0` gate lifted per `wf_cfm_frontier_research/final.md` 2026-09-15. | High | Low (deferred) | Defer P5 entirely until gates clear; do not invest engineering time on doubly-blocked work. |
| **R10** | **Phase-2 P2 surrogate silently degrades pocket-mean Vina.** A2's acceptance criterion (≤0.3 kcal/mol) is loose; a real regression could slip through. | Medium | High | (a) Hard-coded CI gate: P2.5 must show non-degradation on 3 holdout pockets before P2 is marked DONE. (b) `--surrogate-topk` defaults to K=20 (conservative); operators must opt into K=5. (c) Rollback path: bypass surrogate entirely (`--no-surrogate` flag). |

---

## §5 Honest framing per party-twenty-spirit §3.5

This section is the cite-only / honest-framing discipline applied to the integration plan itself, per `wf_pitfall_audit_done_2026-09-17` and `TODO/completed/30_pitfall_reinforce_plan.md` §"Reference evidence".

### §5.1 What this plan claims honestly

- **Mechanism is borrowed directly from the paper.** Replay-buffer-as-cheap-off-policy-evaluator, programmable orchestration layer wrapping the underlying agent (here: the trainer / MCTS), self-improving loop where each round grows the simulator pool — these are paper primitives applied 1-to-1 to mol-metal's training + MCTS loops. The mapping is documented in §1 + §2 and is the only thing this plan claims directly.
- **Implementation is grounded in existing mol-metal infrastructure.** `ReplayPool` consumes existing checkpoint status JSON, existing metric files, existing MCTS reward aggregator. No new external services, no new GPU requirements beyond the planned Wf-125 run, no new dependencies.
- **Per-technique projected speedups are conservative and cite the paper's regime.** Where the paper claims 50×–162× call reduction, mol-metal's projection is 2×–20× because the bottleneck differs (GPU-bound vs LLM-bound) and the replay pool starts smaller (~200 trajectories vs paper's >10000 per task).

### §5.2 What this plan does NOT claim

- **No measurement of the projected speedups.** All numbers in §3 are PROJECTED, not MEASURED. The plan's phases (P0 → P6) include measurement steps; the projections are hypotheses until the corresponding exit gate passes.
- **No claim of headline 50×–162× speedup.** The paper's headline is in a different bottleneck regime; transferring it to mol-metal would be dishonest.
- **No claim of generality beyond the 5-metal × 3-seed program.** All projections are scoped to the current Wf-124/Wf-125 plan. Extrapolation to Round-13 100-pocket sweep is not claimed.
- **No claim that this replaces GPU training.** Replay is a *shortlist* mechanism, not a substitute for the actual run. Every promoted replay decision must be confirmed by at least one held-out verify run.
- **No claim that paper is Google-affiliated.** Per the source analyses' own suggestions, the user said "Google's dream-rsi" but the paper has no Google affiliation. This plan proceeds on the assumption the user meant this paper; alternative readings are listed in §5.4.

### §5.3 Risks per §3.5 of party-twenty-spirit

- **The plan's biggest risk is R1 (replay-pool coverage collapse)** — mol-metal's per-metal sample sizes differ by 20× (Ru 2509 vs Re 121). Without per-cell coverage diagnostics and pessimistic shrinkage, replay predictions for the smaller metals will be optimistic and misleading. P0.4 + §4 R1 mitigation is the load-bearing piece of the plan; if it's not implemented rigorously, the plan fails.
- **The second-biggest risk is R5 (surrogate overfit)** — A2 is the single biggest cost lever, and a silently-overfit surrogate that calls real Vina on the wrong top-K would *increase* oracle cost rather than decrease it. P2.5's cross-pocket validation is the load-bearing mitigation.
- **R8 (arXiv-only preprint, no independent reproduction)** means the plan's projected numbers are hypothesis-grade. Operators should expect ±50% deviation from the projections.

### §5.4 Alternative readings of the user's "Google's dream-rsi" phrasing

The user's request mentions "Google's dream-rsi paper" but the arXiv:2609.14858 author list is **academic, not Google-affiliated**. Possible alternative intentions:

- **(a) This paper.** Most likely if user meant the conceptual framework. Plan applies.
- **(b) Google DeepMind AlphaEvolve / AlphaProof.** LLM-driven algorithm discovery with evolutionary search; conceptually adjacent (evolutionary loop over algorithm space) but architecturally different (LLM-driven mutations, not replay-buffer). Complementary techniques available (island-model migration, LLM-driven crossover, lineage-tree diversity).
- **(c) DeepMind FunSearch** (Romera-Paredes et al., Nature Dec 2023). Functional-search framework using LLM-written code scored by automated evaluation; conceptually adjacent.
- **(d) DeepMind AlphaTensor** (Nature 2022). RL-discovered matrix-multiplication algorithms; less applicable to mol-metal's continuous + categorical search space.
- **(e) DeepMind AI Co-Scientist** (2024). Multi-agent scientific-hypothesis generation; conceptually adjacent but not directly applicable to a training-acceleration role.

If the user meant (b)–(e), the **applicable-techniques ranking shifts** but the §1 mechanism table remains mostly intact (replay-buffer + programmable orchestration + self-improving loop are universal patterns in RSI literature, not Dream-RSI's invention). **Recommended next step:** confirm the user's intent before committing to P1; if the user meant a different paper, swap the cite-only references and re-rank §1.

### §5.5 What success looks like

The plan is successful if, after P0–P3 close (estimated 80–130 h), mol-metal has:
- A populated `ReplayPool` covering all 5 metals × ≥3 seeds with diagnostic per-metal coverage report
- A working A1 config-search that prunes ≥2 of the bottom configs from each sweep before GPU
- A working A2 surrogate that cuts Vina calls by ≥2× on the 3 known-Pt-drug pockets with ≤0.3 kcal/mol degradation
- A working A4 plateau detector that saves ≥15% epochs on average without AUC regression

If even 2 of these 4 hold, the plan has paid for itself. If none hold, the replay pool was empty / coverage-collapsed, and the next iteration must diagnose why before re-attempting.

---

## §6 Cross-references

- Source paper: https://arxiv.org/abs/2609.14858 (preprint, Sep 2026)
- Code (paper): https://github.com/zhengkid/Dream-RSI
- Project page: https://dream-rsi.com
- TODO-19 user decisions: `/home/hugo/codes/mol-metal/TODO/completed/19_user_decisions.md`
- TODO-21 Lambda × CFM coupling: `/home/hugo/codes/mol-metal/TODO/completed/21_lambda_model_coupling.md`
- TODO-22 data-gap alignment: `/home/hugo/codes/mol-metal/TODO/completed/22_data_gap_alignment_plan.md`
- TODO-23 weak-to-strong: `/home/hugo/codes/mol-metal/TODO/completed/23_weak_to_strong_plan.md`
- TODO-26 round 13/14 complete: `/home/hugo/codes/mol-metal/TODO/completed/26_round13_round14_complete_plan.md`
- TODO-30 pitfall reinforce: `/home/hugo/codes/mol-metal/TODO/completed/30_pitfall_reinforce_plan.md`
- TODO-05 tmQM pretraining: `/home/hugo/codes/mol-metal/TODO/completed/05_tmqm_pretraining.md`
- TODO-101/102 OOD splits: `/home/hugo/codes/mol-metal/TODO/completed/101_wf73_p2_ood_permetal.md`, `102_wf73_p3_ood_perelement.md`
- Wf-120 P2 D-MPNN 5-seed: `metrics/by_metric/dmpnn_ckpt_status.json`
- Wf-124 P3 multi-task D-MPNN + Wasserstein: commits `1a8eebd`, `f06ce77`, `e8f68f2` (5-metal × 3-seed × 30-epoch)
- Honest-framing doctrine: `wf_pitfall_audit_done_2026-09-17` memory; `TODO/completed/30_pitfall_reinforce_plan.md` §"Reference evidence"
- Party-twenty-spirit §3.5: cite-only / honest-framing discipline (preserved verbatim across the 6 pitfall audits)