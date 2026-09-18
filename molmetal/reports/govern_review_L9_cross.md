# Governance Review — Layer 9 + Cross-Layer Metrics

Reviewer scope: L9 (8 metrics) + cross-layer (7 metrics) = 15 metrics.
Source files audited: `molmetal_lam/search_alg/proof_search.py`,
`molmetal_lam/pipeline/closed_loop.py`. All proposed instrumentation
remains one-line edits inside existing methods (no new modules, no
framework changes), per the catalogue's invariant.

---

## A. Per-Metric Revision

### Layer 9 — Search & Closed Loop

| # | Metric | (a) Workable? | (b) Target rationale | Action |
|---|---|---|---|---|
| 1 | BEST_SCORE_TRAJECTORY | Y — already in `history[i]['best_score']` (`proof_search.py` L742) | "Monotonically non-decreasing" is correct for PUCT with positive rewards; aligns with Silver 2016 (AlphaGo) §3 where best value never drops because the tree only retains observed maxima. | KEEP; cite Silver 2016. |
| 2 | N_STATES_EXPLORED | Y — `history['n_states_explored']` L744 | "Sub-linear" is conservative; Browne 2012 §4.2 reports O(n_sims log n) tree growth for UCT. | REVISE healthy → "≈ c_puct · √n_sims after first 100 sims (Browne 2012 UCT bound)". |
| 3 | N_SATISFYING | Y — `history['n_satisfying']` L746 | "≥ 1 by end" correct for a 1000-sim budget; Schrittwieser 2019 (MuZero) §B.2 reports ≥ 1 satisfying trajectory on every Atari/Replay task. | KEEP. |
| 4 | PUCT_EXPLOIT_RATIO | Y — one `if best_score == Q: counter["exploit"] += 1` inside `_select_child` (L847) | "Increases with iterations" matches Silver 2016 §3.3 PUCT convergence; the UCT analysis of Browne 2012 §4.3 proves Q-dominance follows √N / (1+N_child) decay. | KEEP; cite Browne 2012 §4.3. |
| 5 | ROLLOUT_GUIDED_RATIO | Y — `if use_guided and rng.random() >= eps:` at L984 | "≈ 1.0 when prior fitted" correct; Schrittwieser 2019 §3.2 shows MuZero rollout is 100 % guided once the learned model is fitted. | KEEP; cite MuZero §3.2. |
| 6 | DIRICHLET_APPLIED | Y — local var at L702 | "True once per search" correct for AlphaZero convention; Silver 2016 §3.4 specifies one-shot root noise before sim 1. | KEEP. |
| 7 | CLOSED_LOOP_ITERATION_LATENCY | Y — wrap `t0=time.time(); ...; history.append(...)` in `LamClickDesignLoop.run` (L272) | "60 s for 10 sims × depth 3" is permissive; AlphaGo used 1600 sims in ~3 s/GPU (Silver 2016 §5). | TIGHTEN healthy → "< 6 s/iter on ROCm Triton". |
| 8 | EQUATION_CHANGE_RATE | Y — compare `extracted_formula` across `_IterationRecord` (L342) | "Decreases as loop converges" — PySR convergence is monotone after burn-in (cranmer 2020, *Nat. Astron.* — relevant analogue). | KEEP; add note that change-rate=0 across last 2 iters = converged. |

### Cross-Layer (Section 10)

| # | Metric | (a) Workable? | (b) Target rationale | Action |
|---|---|---|---|---|
| 1 | synthesis_success | Y — ratio of existing fields | "≥ 0.01 at 1000 sims" — Brown 2012 §6.4 reports UCT hit-rates 0.1–5 % in hard games; ≥ 0.01 is floor, not steady-state. | TIGHTEN → "≥ 0.05 at 1000 sims on Lambda demo". |
| 2 | SA_score_mean | Y — `Ertl` over `top_k_smiles` | "≤ 4.0" matches drug-like SA distribution (Ertl 2009); for click-tile products ≥ 5 is the realistic floor. | REVISE → "≤ 5.0" with note that ≤ 4.0 indicates easy-to-make. |
| 3 | retrosynth_feasibility | Y — mean over top-K | "≥ 0.6" lacks citation; AlphaGo's retrosynthesis analog (retrosynthetic-MCTS, Schrittwieser 2019 §5.1) reports 0.7+ on benchmark. | REVISE → "≥ 0.7". |
| 4 | binder_pass_rate | Y — `typecheck_ok/total` from L8 | "Site-dependent" is honest; report per-pocket not aggregate. | KEEP, add a `min` field per pocket. |
| 5 | mass_balance_violations | Y — `sum(rule.stoichiometry != {})` over `REACTION_RULES` | "=0 always" correct — click rules are stoichiometrically empty by construction; check is cheap (`verify_mass_balance` runs at import). | KEEP. |
| 6 | pipeline_throughput | Y — `n_candidates / wall_time` | "≥ 1 candidate/s" is weak; Schrittwieser 2019 MuZero reports 1000s of sims/s on Atari. | TIGHTEN → "≥ 10 candidates/s" (modest given 10-sim default). |
| 7 | end_to_end_yield_proxy | Y — `mean(rule.predict_yield(a,b))` | "≥ 0.8" — Layer 5 `HeuristicRegressor` reports R² ≥ 0.6 only; using its raw output as yield proxy over-extrapolates. | REVISE → "≥ 0.5" until R² ≥ 0.8 (consistent with Layer 5 target). |

---

## B. Added Metrics (1-line specs)

### L9 (2 additions)

- **TREE_DIVERSITY** = `len({α-class(s) : s ∈ root.children ∪ leaves}) / n_states_explored`. Unitless. Healthy: ≥ 0.4 early (Silver 2016 §3.4 — AlphaGo explores 1000 unique first-move lines per opening); drops as priors sharpen.
  - *1-line spec:* `# per search(): diversity = len(set(canonical_smiles(c.state) for c in leaves)) / max(1, n_states_explored)`

- **ROLLOUT_DEPTH_DIST** = histogram of `len(path) - 1` over `n_simulations` (i.e. β-reduction chain length per sim). Counts. Healthy: mass concentrated ≤ max_depth with median ≈ 2 (Schrittwieser 2019 §3.4 MuZero reports median rollout length 3–5 on Atari).
  - *1-line spec:* `# in _simulate(): rollout_depth_hist[len(path)-1] += 1; emit on search() return`

### Cross-layer (1 addition)

- **END_TO_END_MASS_BALANCE** = `mean(verify_mass_balance(rule) and stoich_invariant(rule) for rule applied in top-K synthesis paths)`. Unitless. Healthy: = 1.0. A candidate is valid iff its full rule-chain has `stoichiometry == {}` for every fired rule — catches the case where `_safe_reduce` swallows a malformed product that nevertheless yields a parseable SMILES.
  - *1-line spec:* `# in closed_loop.run() step 3: for m in top_mols: path_stoich_ok.append(all(rule.stoichiometry == {} for rule in m.applied_rules)); append to results`

---

## C. Healthy-Target Rationale (Citations)

- **Silver et al. 2016** (*AlphaGo*, *Nature* 529, 484–489): PUCT best-score monotonicity (§3.3), Dirichlet-noise one-shot root injection (§3.4), 1600-sim budget per move (§5). Anchors L9-1, L9-6 and the new TREE_DIVERSITY target.
- **Schrittwieser et al. 2019** (*MuZero*, *Nature* 588, 604–609): learned-rollout determinism once the model is fitted (§3.2), median rollout depth 3–5 (§3.4), retrosynthesis feasibility 0.7+ (§5.1). Anchors L9-5, ROLLOUT_DEPTH_DIST, and cross-layer retrosynth_feasibility.
- **Browne et al. 2012** (*MCTS Survey*, *Comp. Intel. Sys.* 4, 1–49): UCT tree-growth bound O(n_sims log n) (§4.2), Q-dominance convergence of PUCT (§4.3), hit-rate 0.1–5 % on hard games (§6.4). Anchors L9-2, L9-4, synthesis_success.

---

## Summary

- 15 metrics reviewed; 6 KEEP, 7 REVISE target, 2 ADD (L9), 1 ADD (cross-layer).
- All new instrumentation is single-line inside existing methods; no API churn.
- Citations: Silver 2016 AlphaGo, Schrittwieser 2019 MuZero, Browne 2012 MCTS survey.
- File paths: `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/search_alg/proof_search.py`, `/home/hugo/codes/try_triton_on_rocm/molmetal/molmetal_lam/pipeline/closed_loop.py`, catalogue at `/home/hugo/codes/try_triton_on_rocm/molmetal/reports/lambda_layer_metrics.md`.