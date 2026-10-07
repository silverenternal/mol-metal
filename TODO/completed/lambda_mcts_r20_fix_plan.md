# Wf-R20-S — Lambda MCTS 纯算法强化: fix plan + 量化指标体系

> **SUPERSEDED 2026-10-07** by Plan A (see `TODO/pending/plan_a_overview.md`). This file moved to `TODO/completed/` for archival reference.
>
> **RE-AUDIT 2026-10-07 (Wf-R26) — variance-cPUct + FPU original claims REVISED:**
> - FIX 1 (bench harness) SHIPPED in commit `89cdbbd017` ✅
> - FIX 2 (variance-cPUct) **was FLAG_ONLY** per §5.7: `_MCTSNode` lacked `leaf_value_var` field; `getattr(node, ...)` always returned 0.0 → variance mode = silent no-op. **Repair SHIPPED in Wf-R26** (`proof_search.py:2924 value_samples` + `:4947 node.value_var` direct read; no getattr fallback). REPAIR commit pending.
> - FIX 3 (FPU rescale) **claim REFUTED** by Wf-R26 audit: FPU is Q-space constant, not N-dependent (KataGo canonical). **Adjacent real defect FOUND** (`proof_search.py:5018 parent_q_max = 0.0` seed → FPU silent disable for negative-Q region). **Repair SHIPPED in Wf-R26** (`parent_q_max = -inf` + isfinite guard).
> - FIX 4 (cross-search TT) **in flight** in wf-R23 (`wf_51df2826-762`).
> - FIX 5 (subtree bias) deferred behind Plan A Step 1.
>
> **Bonus Wf-R26 audit findings (NOT in original R20 plan):**
> - **F0 PUCT Q-damping** CONFIRMED — `proof_search.py:5072` `Q / (1+N)` collapses exploit toward 0. **Repair SHIPPED in Wf-R26** (`:5073 + float(Q)` undamped, virtual_loss stays damped).
> - **F3/F5 PLL size-blind truncation** CONFIRMED — `pll_heuristic.py:202/588` `n = min(ev_a.size, ev_b.size)` discards longer spectrum. **Repair SHIPPED in Wf-R26** (`:144 _pad_spectra` zero-pads shorter to `max(size)`; `:220/:621` use new helper).

**Date:** 2026-10-07 · **Agent:** Wf-R20-S synthesis · **Inputs:** R20-A (code audit) + R20-B (paper research)
**Scope:** 纯算法线 (no trained network). 模型集成部分按用户指示推迟。

## §0 已核验前提 (MEASURED, 2026-10-07 against disk)

| Claim | Verified at |
|---|---|
| `c_puct: float = 1.4` fixed, never variance-scaled | `molmetal/molmetal_lam/search_alg/proof_search.py:3205` (docstring `:3153`) |
| PUCT score site | `proof_search.py:4655-4660` — `w = float(self.c_puct)`; `w` scales ONLY the exploration term |
| `_TranspositionTable` **rebuilt per search** | `proof_search.py:4029` (constructor call inside `search()`) |
| FPU **never called** in search | `grep -c fpu proof_search.py` = **0**; impl exists at `molmetal/molmetal_lam/search_alg/pll_heuristic.py:258 apply_fpu_bonus` |
| Variance signal already recorded | `proof_search.py:3317` (cumulative leaf-reward variance), emitted `:4095` `PUCT_EXPLOIT_RATIO_VAR`, `:4012` `LEAF_VALUE_VAR` |
| File size | 6370 LOC |
| `atomic_write_file` available | `molmetal/scripts/wf386a3_atomic_write.py:147` |

**Honest-negative (carried forward):** R20-A shows both existing bench JSONs are degenerate
(`wf_mcts_real_bench/results.json`: n_samples=3, n_distinct=0.0, n_success=0;
`wf40_mcts_tree/tree_stats.json`: mean_max_depth=3.0, std=0.0, leaf_value_var=0.0, 10 trees).
**Every "5–20× / order-of-magnitude" claim in `mcts_strengthening.md` is PROJECTED, not MEASURED.**
KataGo Elo numbers below are *borrowed* projections, not our measurements.

---

## §1 Fix 排序 (priority order)

### FIX 1 — 量化指标 harness `bench_mcts_search_quality.py` ⭐ PREREQUISITE
- **What:** NEW CPU-only script `molmetal/scripts/bench_mcts_search_quality.py`. Fixed 50-target known-answer benchmark; emits JSON `{success_rate, sims_to_first_solution, leaf_SA/QED/validity/diversity, hit_rate, nfe_per_solution, atom_count_hist, tanimoto_novelty}`. No change to `proof_search.py` in this fix.
- **Why:** R20-A GAP#1 — 没有 quantitative search-quality benchmark; R20-B priority#2 — 所有 fix 在有 baseline 之前不可测。
- **Wall:** 2 day (0.5 day schema + 1 day target set + 0.5 day baseline run). CPU-only. **GPU: 0**.
- **Expected lift:** Δ = 0 (it is the measuring instrument). Unblocks PROJECTED→MEASURED for FIX 2-5.
- **Risk:** low. Risk is *degenerate* benchmark repeating the Wf-40 failure — mitigate with an explicit degeneracy guard: if `n_success==0` and `mean_depth<4` the harness itself FAILs rather than emitting a JSON (refuse-to-emit rule, mirrors §5.5 filesystem-truth).

### FIX 2 — Variance-scaled dynamic cPUct
- **What:** `proof_search.py:3205` `c_puct: float = 1.4` → `c_puct: float | str = 1.4` (accept `"variance"`), + new SCAFFOLD module `molmetal/molmetal_lam/search_alg/variance_cpuct.py` (~80 LOC) exposing `effective_cpuct(base, variance, alpha)`. Read the per-node variance already buffered at `:3317`; smooth over parent subtree. Single line change at `:4655` (`w = float(self.c_puct)` → `w = self._cpuct_for(node)`). Default stays `1.4` → **zero behaviour change unless flag set** (A/B safety).
- **Why:** KataGoMethods 2021 — variance-scaled cPUct ≈ **MEASURED ~75 Elo** vs prior release (~50 Elo from the dynamic term alone; ~25 Elo is pure constant re-tuning). https://github.com/lightvector/KataGo/blob/master/docs/KataGoMethods.md — largest documented lift, and **pure-algorithm** (the uncertainty-*weighted playout* half needs an aux head → deferred).
- **Wall:** 0.5 day code + 0.5 day bench (baseline vs variance arm, ≥10 seeds). CPU. **GPU: 0**.
- **Expected lift (PROJECTED until FIX 1 measures it):** ΔER@N5000 ≥ +3 pts; ΔS2T ≥ −15%; root-visit entropy +.
- **Risk:** **med** — our reward is a Pareto aggregate across metal pockets, so σ conflates pocket heterogeneity with genuine uncertainty. Mitigation: clamp variance per-node with running-mean shrinkage (α=0.8, KataGo-style) and keep the constant arm as control.

### FIX 3 — FPU-reduction + root temperature schedule
- **What:** two flags in `_select_child` (`:4582`): (a) unvisited child (`N_child==0`) → `pll_heuristic.apply_fpu_bonus(...)` instead of falling through to `P=0.5` default; (b) `--root-tau-schedule 1.0:0.25` interpolating root temperature over N sims. ~150 LOC.
- **Why:** AlphaZero FPU-reduction (unvisited Q init = parent's Q) fixes first-visit selection bias while `SymbolicPrior` is untrained; root τ breaks the all-children-tied pathology that Dirichlet noise alone does not fix at low `n_sim`. `apply_fpu_bonus` is **already implemented at `pll_heuristic.py:258` with zero call sites** — pure win, no new algorithm to write.
- **Wall:** 0.5 day + 0.5 day bench. CPU. **GPU: 0**.
- **Expected lift (PROJECTED):** ΔS2T at `n_sim≤100` ≥ −20% (the all-tied case is exactly where S2T is worst); Δb_eff ↑ at low sim; leaf diversity ↑ (FPU stops the first-visit collapse).
- **Risk:** **low-med** — FPU bias can make the search over-commit to a bad early child; guard with the τ schedule decaying to 0.25 so the effect vanishes in the high-sim regime.

### FIX 4 — Cross-search global tree reuse (MCGS-lite)
- **What:** hoist `_TranspositionTable` out of `search()` (currently constructed at `:4029`) to an instance-level cache keyed on β-NF normal form + metal pocket; existing `save_tree`/`load_tree` (`:6086`/`:6170`) as the eviction path. ~120 LOC.
- **Why:** KataGo MCGS (128-bit position hash) — described as "one of the largest single improvements" (qualitative, no Elo published); the 2-5× search-speedup figure is commonly cited but is **PROJECTED for our case**. Our repeated-seed reruns (seeds 0/1/2 per metal) rebuild near-identical subtrees — reuse is nearly free compute.
- **Wall:** 1 day + 0.5 day bench. CPU. **GPU: 0**.
- **Expected lift (PROJECTED):** Δ `TRANSPOSITION_HIT_RATE` ↑ (already instrumented, so directly observable); wall per solution ↓; **must not** regress solution quality — that is the acceptance gate.
- **Risk:** **med** — stale cache across differing reward contexts could silently degrade quality. Mitigation: namespace the cache by the full reward-config fingerprint; `TRANSPOSITION_HIT_RATE` must not be the only guard — pair it with ΔS2T.

### FIX 5 — Subtree value bias correction
- **What:** NEW module `subtree_bias.py` (~100 LOC): bucket nodes by (last β-reduction form, metal pocket, ring pattern), track running `E[heuristic − subtree_MCTS]`, subtract `λ·bucket_bias` (α=0.8 decay). Hooked in the backup step.
- **Why:** KataGoMethods 2021 — **MEASURED +30–60 Elo** (λ=0.35, α=0.8). Second-largest documented lift. β-redex states in MLC are *exactly* template-repetitive like Go local patterns, which is why the bucket key is natural here rather than contrived.
- **Wall:** 1.5 day + 1 day bench. CPU. **GPU: 0**.
- **Expected lift (PROJECTED):** ΔER@N ↑ (search stops re-exploring systematically-mispriced regions); should attack the "L-A2 exploit-ratio plateau" already logged at `:4095`.
- **Risk:** **high** — the bucket key is a design choice with no ground truth; a wrong key makes the correction noise. Deprioritise behind FIX 1-4 and only attempt once FIX 1 can prove or disprove a lift.

### 明确不做 (this phase)
Optimistic policy (aux head), uncertainty-weighted playouts (learned variance head), Fixup / nested bottleneck /
variance-preserving init — all require a trained network, i.e. **exactly the part the user deferred**.
Shaped Dirichlet noise stays as-is (already shipped `:4069`, α=0.3, ε=0.25) but KataGo reports it as
**"NOT validated as a measurable improvement"** — honest-negative, do not claim lift.

---

## §2 量化指标体系 (7 metrics)

| # | Name | Definition | Unit | MEASURED protocol |
|---|---|---|---|---|
| 1 | **ER@N** (Exploit Ratio) | mean root-action visit share of the top-1 action after exactly N=5000 playouts | unitless [0,1] | fixed 50-target set, `--simulations 5000`, ≥10 seeds, mean ± std; control = current `c_puct=1.4` |
| 2 | **S2T** (Simulations-To-threshold) | playouts needed to first reach HV ≥ 90% of the arm's own final HV | playouts (int) | same 50 targets, ≥10 seeds, report median + IQR; **lower is better**; the metric that actually rewards algorithm over compute |
| 3 | **HV** (Pareto Hypervolume) @7 objectives — QED, SA, Vina-proxy, novelty, uniqueness, diversity, anticancer-proxy | 7-D hypervolume, single fixed reference point shared across all arms | unitless (ref-point-dependent) | 10k molecules/run per PMMG (Adv. Sci. 2025) protocol; reference point **frozen in the bench config**, never recomputed per arm |
| 4 | **SR** (Success Rate) | fraction of targets with ≥1 molecule clearing all hard thresholds | [0,1] | same 50 targets × ≥10 seeds |
| 5 | **b_eff + root-visit entropy** | `b_eff = N_visited^(1/depth_reached)`; plus normalised Shannon entropy of root visit distribution | dimensionless; entropy in bits | read from existing counters at `:4130-4151`; no new instrumentation needed — this is the direct read-out of whether FIX 2 actually widens exploration |
| 6 | **Validity / SA / QED / Tanimoto-diversity** of the MCTS leaf set | standard RDKit descriptors over the leaf set | [0,1] / [1,10] / [0,1] / [0,1] | computed per search, aggregated over the 50-target set; this is the *molecular quality* axis that ER alone misses |
| 7 | **NFE per solution** | `CUMULATIVE_NFE / n_solutions` (already emitted) | NFE (int) | straight from existing `:4151`-region instrumentation |

### ⚠️ Anti-gaming guard (mandatory companion to metrics 3/4/6)
MolGenBench (bioRxiv 2025.11.03.686215) shows QED / SA / Vina-proxy **correlate strongly with atom
count and are gameable** — an AddCarbon-style degenerate solution scores near-perfect.
**Every HV / SR / QED / SA number MUST be reported alongside the atom-count histogram and Tanimoto
novelty vs. the training set.** Without those two side-by-side, HV/SR is not defensible to a Tier-2
reviewer. The bench harness must emit them as *required* fields, not optional extras.

### Headline pair
Report **ER@N / S2T** as the headline (the 2018 Artif. Intell. quality-based-evaluation point:
normalise by difficulty so a run cannot score well merely by being easy). Difficulty normalisation
inputs: depth reached + transposition hit rate.

---

## §3 Benchmark protocol

- **Fold:** the fixed **50-target known-answer set** defined *inside* FIX 1's bench config
  (frozen file under `configs/`, hash-pinned). NOT the existing `wf_mcts_real_bench` targets —
  those produced a degenerate 3-sample run and are retained only as historical record.
- **Seeds:** ≥10 per arm (ER@N and S2T are stochastic; <10 seeds cannot support a mean ± std).
  Seeds fixed and recorded in the output JSON so a rerun is bit-comparable.
- **Arms:** A = control (`c_puct=1.4`, no FPU, no τ, per-search TT) → B = FIX 2 only →
  C = FIX 2+3 → D = FIX 2+3+4. One variable at a time, so each Δ is attributable.
- **Gate:** a fix is SHIPPED only if Δ is measured **and** the degeneracy guard passes
  (`n_success > 0`, `mean_depth ≥ 4`). A fix that produces a degenerate bench run FAILS regardless
  of its nominal metric — this directly guards against repeating the Wf-40 / Wf-R20-A failure.
- **Smoke (per §5.7):** each new flag gets a 50-step CPU smoke before any bench run
  (`run_smoke_50step` + `verify_all_invariants`). A flag with unit tests but no end-to-end smoke is
  **FLAG_ONLY**, not SHIPPED.
- **Wall per arm:** ~2-4 hr CPU. 5 arms (baseline + 4) ≈ **1 day wall** once FIX 1 lands.

---

## §4 Wall / GPU budget

| Fix | Code | Bench | GPU |
|---|---|---|---|
| FIX 1 bench harness | 2 d | included | **0** |
| FIX 2 variance-cPUct | 0.5 d | 0.5 d | **0** |
| FIX 3 FPU + τ | 0.5 d | 0.5 d | **0** |
| FIX 4 cross-search TT | 1 d | 0.5 d | **0** |
| FIX 5 subtree bias | 1.5 d | 1 d | **0** |
| **Total** | **5.5 d** | **3 d** | **0** |

**GPU_BUDGET_TOTAL: 0 GPU-hours.** The entire pure-algorithm line is CPU-only by construction —
this is the point of scoping it separately from the model-integration line. No USER_OK gate
(§5.1) is triggered; no pre-flight (§4) is required for CPU-only bench work.

## §5 Sequencing rule

**FIX 1 lands and produces a non-degenerate baseline JSON on disk BEFORE any of FIX 2-5 is
reported as SHIPPED.** A KataGo Elo number is not a measurement of our system; borrowing it is
projection laundering unless our own before/after is on disk. Per §5.5, the baseline JSON must be
`ls -la`-verified before downstream agents may cite it.

## §6 Honest-negative inventory for any writeup

1. All R20-A degenerate bench numbers (`n_samples=3`, `mean_depth=3.0`, `leaf_value_var=0.0`).
2. All KataGo Elo figures are **borrowed projections**, not our measurements.
3. Shaped Dirichlet noise: KataGo reports **no validated measurable improvement**.
4. `mcts_strengthening.md` file paths point at a different clone (`try_triton_on_rocm`) — stale, not this repo.
5. QED/SA/Vina are atom-count-gameable (MolGenBench) → never report without the atom-count + novelty guard.
6. GPU-batched MCTS, optimistic policy, uncertainty-weighted playouts, network init schemes:
   **out of scope this phase** (require a trained model).

## §7 Cross-ref

`molmetal/molmetal_lam/search_alg/proof_search.py` (6370 LOC; `:3205` c_puct, `:4029` per-search TT,
`:4582` `_select_child`, `:4655` `w = float(self.c_puct)`, `:3317` variance buffer, `:6090`+ save/load tree) ·
`molmetal/molmetal_lam/search_alg/pll_heuristic.py:258` `apply_fpu_bonus` (implemented, 0 call sites) ·
`molmetal/scripts/wf386a3_atomic_write.py:147` · R20-A audit table · R20-B method table ·
KataGoMethods.md · PMMG Adv. Sci. 2025 · MolGenBench bioRxiv 2025 ·
IJAR 2015 metric suite · Artif. Intell. 2018 quality-based evaluation.
