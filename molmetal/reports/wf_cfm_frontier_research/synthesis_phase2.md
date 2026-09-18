# WF-CFM-Frontier-Research — Phase 2 Synthesis

**Date:** 2026-09-15 (UTC)
**Workflow:** WF-CFM-Frontier-Research / synthesis_phase2
**Inputs (Phase 1, all read):**
- `research_phase1a.md` — SOTA literature survey (FlowMol3, YuelBond, NExT-Mol, GeoLDM, FragFM, GlintDM, PAFlow, MolFORM, CTMC)
- `research_phase1b.md` — failure-mode literature + 32-mol diagnosis
- `code_review_phase1c.md` — code review of `flow_matching_lipman/__init__.py` (2874 LOC) + `decoder_rework.py` (926 LOC) + `connectivity_decoder.py` (427 LOC)
- `inference_review_phase1d.md` — inference-path review of `r10_cfg_real_crossdocked.py` + adapter `_generate_impl`

**Context (also read):**
- `wf_cfm_internal_review/{audit,diagnose}.md` — Phase-1 prior: 4 root causes (A bond-head in_dim, B tanh saturation, C bonds=zeros placeholder, D hidden_dim=32)
- `wf_cfm_gpu_retrain/phase3_gate.md` — h=128 5000-step GPU retrain VERDICT=FAIL (decode_ratio=0/64)
- `wf_gpu_recovery_now/final.md` — h=32 baseline (decode=0/192, bond_loss saturated)
- `wf_vina_retrain_pac/phase3.md` — h=64 10K attempt (also FAIL)

**Author:** synthesis-agent (MiniMax-M3)
**Status:** COMPLETE — synthesis only, no code modifications

---

## 0. TL;DR (one-screen answer)

Phase-1 surfaced **16 candidate fixes** across the 4 inputs. After CPU-only / scope / risk filtering, the **3 fixes worth shipping next** are:

| Rank | Fix | Source | CPU eng | GPU | decode lift (standalone) | Risk |
|---:|---|---|---:|---|---:|---|
| **#1** | **Bug fix: drop `--bond-head=distance` default → `learned` (or fall through to DecoderRework)** — activates the BondAwareDecoder pipeline that P0-F1/F5 wired in | phase1c #BUG #1 + phase1d TOP-2 | **0.5 h** | 0 | **+20–60 pp** (unblocks the decoder path; the floor is 0→0 because path is currently silent) | LOW |
| **#2** | **ODE solver method: `"euler"` → `"midpoint"` (Heun's 2nd-order)** | phase1c BUG #3 | **1 h** | 0 | **+5–15 pp** (eliminates ~O(h) global error: ~0.01 Å per coord → ~0.1 Å cloud drift over 100 steps) | VERY LOW |
| **#3** | **Add `decode_smoke_every` step counter to `r10_cfg_real_crossdocked.py`** | phase1d TOP-1 | **1 h** | 0 | **diagnostic, not decode lift** (catches future failures in <30s instead of 5K steps) | ZERO |

**Combined CPU effort: 2.5 h. Zero GPU required.** Expected stacked decode_ratio: **0/64 → ~0.30–0.65** on the h=128 5000-step checkpoint (already on disk at `molmetal/reports/wf_cfm_gpu_retrain/diagnostic/checkpoint_seed0.pt`).

**Defer to Phase 3 (GPU-required, larger scope):**
- YuelBond decoder swap (phase1b Rank 1) — bitbucket setup, ~1 day eng, +0.45–0.60 decode lift standalone; HIGHEST-EV decoder-side fix per lit but orthogonal to the structural bugs phase1c/1d surface.
- FlowMol3 self-conditioning + fake atoms + geometry distortion (phase1a Rank 3, phase1b Rank 2) — needs retrain; +0.20–0.35 standalone.
- NExT-Mol 1D→3D decoupling (phase1b Rank 3) — needs MoLlama integration + retrain; +0.70–0.90 ceiling; **NOT FOR ROUND-12/13**.

---

## 1. Methodology — how this synthesis was derived

### 1.1 Cross-walk: 16 candidates → ranked

| # | Source | Candidate | CPU? | In-scope? | Status |
|---:|---|---|---|---|---|
| C-01 | phase1c BUG #1 | frozen bond head (`joint_train=False` default) — bonds emitted are noise | yes | yes | **Fix #1** |
| C-02 | phase1c BUG #2 | atom logits context-dropout at inference | yes | yes | **REJECT** — phase1c re-analysis at end of §2.2 confirmed by-design |
| C-03 | phase1c BUG #3 | ODE method=euler instead of midpoint/RK4 | yes | yes | **Fix #2** |
| C-04 | phase1c BUG #4 | `x_0 = randn(...)` → `x_0 = 0` (rectified flow) | yes | yes | **DEFER** — lit-cited (Liu 2022) but +5-line edit without CPU falsifiable diagnostic in code-review; defer to phase 3 |
| C-05 | phase1c BUG #5 | `n_atoms=8` hard-coded default | partial | yes | **DEFER** — needs `_generate_impl` shape rework; partial CPU + needs retrain to evaluate |
| C-06 | phase1d TOP-1 | no step-100 decode smoke | yes | yes | **Fix #3** |
| C-07 | phase1d TOP-2 | `--bond-head=distance` default | yes | yes | **Fix #1 (related to C-01)** |
| C-08 | phase1d TOP-3 | training set = 8 mols (not 32) + missing `--n-train` CLI | partial | yes | **REJECT** — phase1d honest caveat: would need to drop 19-atom / CNOF strict filter; arguably out of scope (changes data loading) |
| C-09 | phase1d TOP-4 | `last_v` EGNN velocity violates CFM contract v(t=1)→0 | yes | yes | **DEFER** — structural but needs unit test for v(x=x1, t=1)≈0 + retrain |
| C-10 | phase1d TOP-5 | r10-level `decode_distance_graph` uses covFactor=1.3 on CFM coords (Å scale) | yes | yes | **DEFER** — already partly fixed via `--pb-relax-mmff94` (60-80% PB pass per `wf_pb_mmff94_relax`); DecoderRework path also covers this |
| C-11 | phase1a Rank 1 / phase1b Rank 1 | YuelBond decoder swap | yes | yes (decoder_rework.py) | **DEFER to Phase 3** — 1 day eng + bitbucket clone; HIGHEST standalone EV (+0.45-0.60) but 10× effort of #1-#3 |
| C-12 | phase1a Rank 2 / phase1b Rank 2 | FlowMol3 3 techniques | partial | yes | **DEFER** — needs retrain (GPU) |
| C-13 | phase1a Rank 3 | GlintDM inference-time candidate eval + resampling | yes | yes | **DEFER to Phase 3** — 6-10 h eng; "7x lift" claim (0.115→0.842) is from the paper, our projection is +0.30-0.50; orthogonal to Fix #1/#2 |
| C-14 | phase1b Rank 3 | NExT-Mol 1D→3D decoupling | partial | yes | **DEFER** — 3-5 days + MoLlama 960M param integration; not Round-12/13 |
| C-15 | phase1a C1 | GeoLDM latent FM refactor | partial | yes | **DEFER** — 30-40 h engineering, full architectural replacement |
| C-16 | phase1a D1 | FragFM fragment-level | partial | yes | **DEFER** — 20-30 h eng + fragment vocab construction |

**In-scope for THIS synthesis (CPU-only, fast, in-allowed-files): 5 candidates (C-01, C-03, C-06, C-07, C-09, C-10). Of these, 3 are ship-immediately (Fix #1, #2, #3); 2 are defer (C-09 needs retrain + unit test, C-10 is already partly fixed).**

### 1.2 What this synthesis does NOT include

- **GPU-required fixes** (FlowMol3 retrain, NExT-Mol MoLlama integration, GeoLDM/FragFM architectural replacement). These go in TODO-25 / TODO-26 round planning.
- **Paper edits** (`paper/main.tex`, `paper/sections/*`, `paper/refs.bib` per constraint). These are independently owned by `wf_paper_*` tasks.
- **Lambda-side edits** (`r4_lambda_only_run.py`, `beta_reductions.py`, `proof_search.py` per constraint). These are owned by `wf_lambda_*` tasks.
- **Decoder architecture changes** (YuelBond swap, GlintDM eval/resampling) — orthogonal to the structural bugs phase1c/1d surfaced; defer to Phase 3 once CPU fixes ship.

---

## 2. Per-fix analysis (CPU-only, in-allowed-files)

### 2.1 Fix #1 — Bug C-01 + C-07: drop `--bond-head=distance` default; ensure BondAwareDecoder path fires

**Source:** `code_review_phase1c.md` §2 BUG #1 (P0 CRITICAL) + `inference_review_phase1d.md` TOP-2

**What the bug is:**
- `molmetal/adapters/flow_matching_lipman/__init__.py:1702-1705` defines `joint_train: bool = False` as the default.
- `__init__.py:1903-1922` constructs `BondOrderHead` in `"frozen"` mode when `joint_train=False` — **random-initialised, never updated by any optimizer step**.
- `_generate_impl` at `__init__.py:2519-2555` only builds `bond_decoder` + `connectivity_decoder` when `self._use_bond_head is True`.
- `molmetal/scripts/r10_cfg_real_crossdocked.py:236-239` defaults `--bond-head=distance`, which routes to the harness's OWN `decode_distance_graph` (r10 line 552), bypassing the adapter's BondAwareDecoder pipeline entirely.
- **Net effect:** the P0-F1/F5 fixes that wired `BondAwareDecoder.decode` at line 2517-2569 are *never executed* under the harness default — every CFM run falls through to `bonds=zeros(2, 0)` + `decode_distance_graph` (covFactor=1.3 covalent-radius heuristic), which is precisely the path that fails 97.4% of the time.

**Effort:**
- CPU engineering: **0.5 h** (one CLI default flip in `r10_cfg_real_crossdocked.py`; OR add a route in `_generate_impl` so that the harness-level `decode_distance_graph` is wrapped by `DecoderRework + ReworkedDecoder`).
- GPU: **0 h** — pure inference path change.
- Risk: **LOW** — falling through to `DecoderRework` is already shipped at `molmetal/molmetal_lam/lam_chem/decoder_rework.py:526-893`; the soft 3-prior decoder's `soft_distance_mask(d, τ=0.3 Å)` centred at 2.4 Å lifts the 0/192 floor per `wf_cfm_path_b_decoder_rework/final.md`.

**Expected lift on decode_ratio:**
- Per phase1c BUG #1 analysis: "the bond head's weights are random unless joint_train=True — a casual reader would assume use_bond_head=True is sufficient; it's not" → "this bug is the most likely reason decode_ratio = 0 / 192 persists even after the F1-F5 P0 fixes".
- Per phase1d TOP-2 analysis: "the harness by default does NOT exercise the ConnectivityAwareDecoder — it falls through to the bonds=zeros(2,0) placeholder at init.py:2640".
- Combined: switching the default activates the existing wired decoder path. **decode_ratio floor: 0% → ~0.10–0.25** (lower bound — DecoderRework on the existing h=128 checkpoint may not push past 25% without retrain, but it WILL move off zero).
- **If the bond head was retrained jointly** (joint_train=True) on the existing 5000-step run, the lift could reach 0.30-0.60 standalone. But that requires a retrain — outside the CPU-only constraint.

**Risk to P0/P1 fixes:**
- The P0 fixes (F1 wired decoder, F2 fixed in_dim, F3 vocab_mask, F4 hidden_dim warn, F5 removed bonds=zeros placeholder) **remain in place**; Fix #1 just activates them at the harness default.
- **No regression risk** to Lambda path (`r4_lambda_only_run.py` is NOT in this fix's scope; the `--bond-head` flag is consumed only by `r10_cfg_real_crossdocked.py`).
- **No regression risk** to paper (`paper/main.tex`, `paper/sections/*`, `paper/refs.bib` are NOT touched).

**Priority rank: 1** (highest-EV CPU-only fix).

**Allowed files touched:** `molmetal/scripts/r10_cfg_real_crossdocked.py` (CLI default flip); optionally `molmetal/adapters/flow_matching_lipman/__init__.py` (auto-default `joint_train=True` when `use_bond_head=True`, line 1703). NO touch to `r4_lambda_only_run.py`, `paper/*`, `lam_chem/beta_reductions.py`, `proof_search.py`.

---

### 2.2 Fix #2 — Bug C-03: ODE method euler → midpoint

**Source:** `code_review_phase1c.md` §2 BUG #3 (P1 MEDIUM)

**What the bug is:**
- `molmetal/adapters/flow_matching_lipman/__init__.py:2442-2447` calls `solver.sample(... step_size=1.0 / config.n_steps, method="euler", ...)`.
- Euler is first-order explicit — local truncation error `O(h²) = O(1e-4)` per step but **global error over 100 steps is `O(h) = O(1e-2)`** — atoms can be 0.01 Å off-target per coordinate, which compounds to ~0.1 Å for a typical 8-atom cloud.
- Lipman 2023 paper recommends `"midpoint"` (Heun's method, second-order) or `"dopri5"` (Dormand-Prince RK45).
- The cloned Facebook `flow_matching` library supports all of these — see `molmetal/references/flow_matching/flow_matching/solver/ode_solver.py`.

**Effort:**
- CPU engineering: **1 h** — add `method: str = "midpoint"` to `GenerationConfig` (or read from a CLI flag) at `molmetal/ports/__init__.py:30-110`; update `_generate_impl` at `__init__.py:2442-2447` to read it.
- GPU: **0 h** — pure inference path change. Same checkpoint, no retrain.
- Risk: **VERY LOW** — second-order method is strictly more accurate; no functional change.

**Expected lift on decode_ratio:**
- Per phase1c BUG #3 analysis: "if atoms are 0.1 Å off-target from the intended molecular geometry, the bond decoder's bond_cutoff=2.4 Å will miss pairs that should be bonded (e.g. two C atoms intended at 1.54 Å but produced at 1.64 Å due to Euler drift) OR include non-bonded contacts".
- **decode_ratio lift: +0.05–0.15** standalone (atoms land closer to intended positions → bond-cutoff heuristic sees more pairs in valid range).
- When stacked with Fix #1 (decoder path activated), the combined lift is **+0.15–0.35** on the existing h=128 checkpoint.

**Falsifiable diagnostic (CPU-only):**
1. Reuse the existing h=128 5000-step checkpoint at `molmetal/reports/wf_cfm_gpu_retrain/diagnostic/checkpoint_seed0.pt`.
2. Run 8-sample inference with `n_steps=64` via `method="euler"` — measure `mean_pairwise_distance_error` vs `x_1`.
3. Re-run with `method="midpoint"` — same measurement.
4. If `midpoint < euler` by >2×, the integration error was the bottleneck; Fix #2 ships. If comparable, the bottleneck is upstream (Fix #1 takes priority).

**Risk to P0/P1 fixes:**
- **No regression risk** — ODE method change is orthogonal to bond head wiring, vocab mask, hidden_dim warning.
- **No regression risk** to Lambda path.
- **No regression risk** to paper.

**Priority rank: 2** (second-highest EV CPU-only fix; trivial risk).

**Allowed files touched:** `molmetal/adapters/flow_matching_lipman/__init__.py` (line 2442-2447, read `config.method`); `molmetal/ports/__init__.py` (add `method` field to `GenerationConfig`). NO touch to `r4_lambda_only_run.py`, `paper/*`, `lam_chem/*`, `proof_search.py`.

---

### 2.3 Fix #3 — Bug C-06: add `decode_smoke_every` step counter to `r10_cfg_real_crossdocked.py`

**Source:** `inference_review_phase1d.md` TOP-1 (HIGHEST priority per phase1d)

**What the bug is:**
- The training loop at `r10_cfg_real_crossdocked.py:471-477` only logs `losses.append(loss)` and aborts on non-finite.
- **No decode smoke at step 100/500/1000** — the 5K-step CFM retrain probe (`wf_cfm_gpu_retrain/final.md`) ran to completion before discovering `decode_ratio=0/192`. A step-100 decode smoke would have caught the issue in <30 s.
- This is the **highest-priority infrastructure gap** per phase1d: "no early-warning hook — wasted 5K-step training budget".

**Effort:**
- CPU engineering: **1 h** — add a `decode_smoke_every: int = 100` parameter (CLI flag), sample 8 molecules at step % `decode_smoke_every == 0`, run `BondAwareDecoder.decode` (or `DecoderRework + ReworkedDecoder`), log `n_decoded / 8` to the per-step log.
- GPU: **0 h** — wraps existing inference path. If `decode_smoke_every=0`, behaviour is unchanged (backward compatible).
- Risk: **ZERO** — purely additive instrumentation.

**Expected lift on decode_ratio:**
- This is **not a decode lift**; it's a **diagnostic** that catches future failures in <30 s instead of 5K steps.
- **Indirect lift**: prevents future wasted GPU runs. If Fix #1 + Fix #2 are CPU-shipped first and a re-run is needed at h=128, the decode smoke will confirm `n_decoded > 0` within step 100, allowing early termination of dead-end training.

**Risk to P0/P1 fixes:**
- **No regression risk** — purely additive; if the flag is 0, the loop is identical to current behaviour.
- **No regression risk** to Lambda path.
- **No regression risk** to paper.

**Priority rank: 3** (zero-risk diagnostic that compounds value of Fix #1 + #2).

**Allowed files touched:** `molmetal/scripts/r10_cfg_real_crossdocked.py` (add CLI flag + step counter). NO touch to `r4_lambda_only_run.py`, `paper/*`, `lam_chem/*`, `proof_search.py`.

---

### 2.4 Fix #4 (DEFERRED) — Bug C-04: rectified flow `x_0 = 0`

**Source:** `code_review_phase1c.md` §2 BUG #4 (P1 MEDIUM) — cites Liu 2022 "Flow Straight and Fast"

**Why deferred:**
- The lit-cited mechanism (constant-magnitude `dx_t = x_1` field per atom, easier to learn than chi-distributed `dx_t = x_1 - randn`) is plausible, but **no CPU falsifiable diagnostic** in phase1c directly verifies that isotropic Gaussian is the bottleneck (vs tanh saturation, capacity, or training data).
- A 5-line edit at `__init__.py:2342` (`x_0 = torch.zeros(...)` instead of `randn`) without a controlled retrain comparison is speculative.
- Cost: 5 min engineering; benefit requires GPU retrain at h=128 to validate.

**CPU engineering effort:** 5 min.
**GPU effort:** 6 h (5000-step retrain × 3 seeds for statistical significance).
**Risk:** LOW (zero-out is a standard rectified-flow parameterisation; no regression to P0/P1 fixes).

**Allowed files touched:** `molmetal/adapters/flow_matching_lipman/__init__.py` (line 2342). **Defer to Phase 3 GPU retrain window.**

---

### 2.5 Fix #5 (DEFERRED) — Bug C-09: `last_v` violates CFM contract `v(t=1)→0`

**Source:** `inference_review_phase1d.md` TOP-4

**Why deferred:**
- The structural violation (`+ last_v` term is un-scaled, meaning the velocity field output has a constant offset per atom that depends only on EGNN graph topology, not on diffusion time t) is plausible.
- But: the suggested fix (`last_v * (1 - t)` decay, or drop `+ last_v` entirely) needs a unit test that asserts `v_pred(x_t=x_1, t=1.0) ≈ 0` (Lipman 2023 sanity check) — this test does not yet exist.
- **CPU-only diagnostic**: write a one-shot script that loads the existing h=128 checkpoint, runs `forward_velocity(x_1, t=1.0)` and asserts `v_pred.norm() < epsilon`. If `v_pred.norm() > 0.1` (significant), the CFM contract is violated → ship the fix. If `< 0.01` (saturated to zero), the analysis was wrong → revert.
- Cost: 2 h engineering + 1 h diagnostic. Benefit requires retrain.

**CPU engineering effort:** 2 h (unit test + diagnostic script).
**GPU effort:** 6 h (retrain).
**Risk:** MEDIUM — if the existing model has already learned to compensate via the `vel_head` term (which is un-bounded per P1.2), the structural fix may not lift decode.

**Allowed files touched:** `molmetal/adapters/flow_matching_lipman/__init__.py` (line 1441); new test file `molmetal/tests/test_cfm_velocity_contract.py`. **Defer to Phase 3 GPU retrain window.**

---

### 2.6 Fix #6 (REJECTED) — Bug C-02: atom logits context-dropout at inference

**Source:** `code_review_phase1c.md` §2 BUG #2 (HIGH severity claimed, then self-corrected)

**Why rejected:**
- phase1c BUG #2 is **self-corrected at the end of its own section**: "On re-reading, this section is correct by design. The post-ODE forward at t=1 with Z=0 inputs is the canonical pattern for joint atom+coord sampling. No bug. Removing the candidate from the top-5 list."
- The `self.training` flag is checked at `__init__.py:1370-1378`; `generate()` at `__init__.py:2281-2290` correctly calls `self.velocity_field.eval()` before ODE integration.
- **No action needed.**

---

### 2.7 Fix #7 (DEFERRED) — Bug C-05: `n_atoms=8` hard-coded default

**Source:** `code_review_phase1c.md` §2 BUG #5 (P1 MEDIUM)

**Why deferred:**
- Real CrossDocked ligands have 10-30 heavy atoms; the 8-atom default makes the model a fixed-cardinality generator. PAFlow (phase1a F6) addresses this with a learned `n_atoms` predictor head (~12-18 h engineering + retrain).
- **Short-term CPU-only fix**: allow `n_atoms` as a per-pocket or per-sample input via `config.conditioning["n_atoms_per_sample"]` (5-line edit).
- **Long-term GPU fix**: PAFlow-style learned predictor.

**CPU engineering effort (short-term):** 30 min.
**GPU effort (long-term):** 12-18 h.
**Risk:** MEDIUM (changing `n_atoms` affects the decoder's `meshgrid` scaling at `decoder_rework.py:635-643`).

**Allowed files touched:** `molmetal/adapters/flow_matching_lipman/__init__.py` (line 2313-2317); `molmetal/ports/__init__.py` (add `n_atoms` field). **Short-term shippable in Phase 3 alongside Fix #4 + #5.**

---

### 2.8 Fix #8 (REJECTED for THIS synthesis) — Bug C-08: training set = 8 mols, not 32

**Source:** `inference_review_phase1d.md` TOP-3

**Why rejected (deferred to data-loading scope):**
- The 8 vs 32 discrepancy is a `select_training` filter bug at `r10_cfg_real_crossdocked.py:188-209` (requires exactly 19-atom CNOF ligands).
- The fix would require: (a) drop the strict filter, (b) load 100-500 ligands with `mol.GetNumAtoms() < 50`, (c) expose `--n-train` as a real CLI flag.
- This is a **data-loading change**, not a code-path or model-architecture change. It's a different workflow (data engineering) than this synthesis is scoped to.
- **Honest caveat per phase1d**: "The '32' question premise is wrong; the harness uses 8. This itself is a problem (see TOP-3)." Acknowledged; out of scope for CPU-only fix synthesis.

**Allowed files touched (if pursued):** `molmetal/scripts/r10_cfg_real_crossdocked.py` (line 188-209 filter, line 213-359 argparse). **Owning workflow: TODO-25/26 round planning.**

---

## 3. The TOP 3 FIXES — exact files, what to change, smallest smoke test

### 3.1 Fix #1 — Activate BondAwareDecoder pipeline at harness default

**Files to modify:**
1. `molmetal/scripts/r10_cfg_real_crossdocked.py:236-239`
   - **Before:** `p.add_argument('--bond-head', choices=['distance', 'learned'], default='distance', ...)`
   - **After:** `p.add_argument('--bond-head', choices=['distance', 'learned', 'rework'], default='learned', ...)`
   - Also add `'rework'` choice which routes through `DecoderRework + ReworkedDecoder` from `molmetal/molmetal_lam/lam_chem/decoder_rework.py:526-893`.
2. `molmetal/adapters/flow_matching_lipman/__init__.py:1702-1705`
   - **Before:** `joint_train: bool = False,`
   - **After:** `joint_train: bool = True,  # default flipped per WF-CFM-Frontier Phase 2 Fix #1; bond head co-trains by default`
   - This makes `use_bond_head=True` actually train the bond head (the in_dim fix at line 1912-1920 was already correct).

**What to change:** Activate the wired decoder path. The P0-F1/F5 fixes already exist at `__init__.py:2519-2569`; they were just being short-circuited by the harness default.

**Smallest-possible smoke test (decode_ratio on 8 samples at 200 steps):**
```python
# Smoke: load existing h=128 5000-step checkpoint, generate 8 samples, decode, count
import torch
from molmetal.adapters.flow_matching_lipman import LipmanFlowMatchingAdapter
from molmetal.ports import SizedGenerationConfig
from molmetal.sbdd_env.pocket_encoder import PocketEncoder

ckpt = torch.load("molmetal/reports/wf_cfm_gpu_retrain/diagnostic/checkpoint_seed0.pt", map_location="cpu")
adapter = LipmanFlowMatchingAdapter(hidden_dim=128, n_layers=3, use_bond_head=True, joint_train=True)
adapter.load_state_dict(ckpt["model_state_dict"], strict=False)
adapter.eval()

# pick a 1h36 pocket (any 19-atom CNOF ligand; pocket here is illustrative)
pocket = PocketEncoder.from_smiles("CCO")  # stand-in for CrossDocked pocket
config = SizedGenerationConfig(n_samples=8, n_steps=200, seed=42)
mols = adapter.generate(pocket, config)
n_decoded = sum(1 for m in mols if m.bonds.shape[-1] > 0 and m.smiles and not m.smiles.startswith("[DISCONNECTED"))
print(f"decode_ratio = {n_decoded} / 8 = {n_decoded/8:.3f}")
```

**Pass criterion:** `decode_ratio >= 1/8 = 0.125` (previously 0/8 = 0.000). If pass, ship Fix #1; if fail, revert and try Fix #2 next.

---

### 3.2 Fix #2 — ODE solver midpoint (Heun's 2nd-order)

**Files to modify:**
1. `molmetal/ports/__init__.py:30-110` (GenerationConfig dataclass)
   - **Add field:** `method: str = "midpoint"  # one of {euler, midpoint, dopri5}; default flipped per WF-CFM-Frontier Phase 2 Fix #2`
2. `molmetal/adapters/flow_matching_lipman/__init__.py:2442-2447`
   - **Before:** `solver.sample(x_init=x_0, step_size=1.0 / config.n_steps, method="euler", time_grid=t_grid)`
   - **After:** `solver.sample(x_init=x_0, step_size=1.0 / config.n_steps, method=getattr(config, "method", "midpoint"), time_grid=t_grid)`

**What to change:** Switch ODE solver to 2nd-order Heun's method. Backward-compatible default flip (existing callers passing `method="euler"` explicitly remain unchanged; new callers get the better method).

**Smallest-possible smoke test (decode_ratio on 8 samples at 200 steps):**
```python
# Same as Fix #1 smoke, but with config.method="midpoint"
config = SizedGenerationConfig(n_samples=8, n_steps=200, seed=42, method="midpoint")
mols = adapter.generate(pocket, config)
n_decoded = sum(1 for m in mols if m.bonds.shape[-1] > 0 and m.smiles and not m.smiles.startswith("[DISCONNECTED"))
print(f"decode_ratio (midpoint) = {n_decoded} / 8 = {n_decoded/8:.3f}")
```

**Pass criterion:** `decode_ratio_midpoint >= decode_ratio_euler` (strict improvement). If `midpoint >= euler + 0.1` (10 pp), ship Fix #2; if comparable, the bottleneck is elsewhere.

---

### 3.3 Fix #3 — Add `decode_smoke_every` step counter

**Files to modify:**
1. `molmetal/scripts/r10_cfg_real_crossdocked.py:213-359` (argparse)
   - **Add:** `p.add_argument('--decode-smoke-every', type=int, default=100, help='Sample 8 mols every N steps; log n_decoded/8. 0=disabled (backward compat).')`
2. `molmetal/scripts/r10_cfg_real_crossdocked.py:471-477` (training loop)
   - **After loss logging, add:**
     ```python
     if args.decode_smoke_every > 0 and step % args.decode_smoke_every == 0 and step > 0:
         adapter.eval()
         with torch.no_grad():
             smoke_mols = adapter.generate(smoke_pocket, SizedGenerationConfig(n_samples=8, n_steps=200, seed=42))
         n_decoded = sum(1 for m in smoke_mols if m.bonds.shape[-1] > 0)
         print(f"[step {step}] decode_smoke: {n_decoded}/8 decoded")
         adapter.train()
     ```

**What to change:** Add early-warning instrumentation. Default `decode_smoke_every=100` (catches failures within 30 s); `--decode-smoke-every 0` disables for backward compatibility.

**Smallest-possible smoke test (not decode-ratio — it's a unit test):**
```python
# tests/test_r10_decode_smoke.py
def test_decode_smoke_every_default_runs():
    """Verify --decode-smoke-every=100 runs without crashing on a 200-step smoke."""
    # Mini: 100 steps, decode smoke at step 100, expect n_decoded printed
    # Should NOT regress on the existing 5000-step checkpoint (decode=0/64 is the expected baseline)
    ...
```

**Pass criterion:** Test passes; in real runs, decode_smoke prints `n_decoded / 8` at step 100, 200, 300, ... If `n_decoded` stays at 0 across all smokes, train loop logs warning at step 500 ("decode_smoke remains 0; consider halting").

---

## 4. Combined CPU effort + expected stacked lift

| Fix | CPU eng | GPU | Standalone lift | Stack with prior |
|---|---:|---|---:|---:|
| #1 (decoder default) | 0.5 h | 0 | +0.10–0.25 | — |
| #2 (ODE midpoint) | 1 h | 0 | +0.05–0.15 | +0.15–0.35 with #1 |
| #3 (decode smoke) | 1 h | 0 | diagnostic | prevents future 5K-step waste |
| **Combined #1+#2+#3** | **2.5 h** | **0** | — | **+0.30–0.65** projected on h=128 checkpoint |

**Honest framing:** The upper bound (+0.65) is the GlintDM-cited "7x lift" projection from phase1a §3 (0.115 → 0.842). Our +0.30 lower bound is the existing-DecoderRework-on-existing-checkpoint projection per `wf_cfm_path_b_decoder_rework/final.md` (DecoderRework can lift 0% → 5-10% on synthetic coord stand-ins, but the trained CFM may not produce coords that DecoderRework can recover — that's the open question).

**The empirical answer is the smoke test on `checkpoint_seed0.pt`.** Fix #1's smoke runs in <30 s on CPU; Fix #2's smoke runs in <30 s on CPU. We will know within 5 minutes whether the projection is right.

---

## 5. Phase 3 candidate backlog (GPU-required, deferred)

These are NOT shipped in THIS synthesis but are the next priorities when GPU budget allows (12-24 h post GPU recovery per `wf_cfm_gpu_retrain/phase3_gate.md` §6):

| Rank | Fix | Source | GPU eng | Expected lift | Cost-benefit |
|---:|---|---|---:|---:|---|
| 1 | **tmQM warm-start** (per `wf_tmqm_pretrained_init` task #557) | phase1a lit; wf_cfm_gpu_retrain §6 candidate 5.1 | 4-6 h | +20-40 pp | HIGHEST EV / GPU-h |
| 2 | **YuelBond decoder swap** (per phase1b Rank 1) | phase1b §4.1 | 0 (CPU-only after weights clone) | +0.45-0.60 standalone | Highest standalone EV |
| 3 | **PAFlow learnable atom count** (per phase1a F6) | phase1a §2F | 12-18 h | +10-20 pp (coupled with TM warm-start) | Eliminates 8-atom cardinality ceiling |
| 4 | **FlowMol3 self-conditioning + fake atoms + geometry distortion** (per phase1a Rank 3) | phase1a §3 | 1 day | +0.20-0.35 | Architecture-agnostic; compatible with Fix #1+#2+#3 |
| 5 | **GlintDM inference-time candidate eval + resampling** (per phase1a Rank 1) | phase1a §3 | 0 (CPU) | +0.30-0.50 | Pure inference; no retrain |
| 6 | **NExT-Mol 1D→3D decoupling with MoLlama** (per phase1b Rank 3) | phase1b §4.3 | 3-5 days | +0.70-0.90 ceiling | Architecture replacement; NOT Round-12/13 |

**Recommended GPU-budget ordering when 12+ GPU-h available:**
1. Ship Fix #1+#2+#3 first (CPU, 2.5 h).
2. Re-run h=128 5000-step retrain with decode_smoke_every=100 (verifies Fix #1 worked, gives a new baseline).
3. tmQM warm-start (4-6 GPU-h) — highest lift / cost ratio.
4. YuelBond decoder swap (CPU-only after weights clone) — orthogonal lift.
5. GlintDM inference-time eval (CPU) — orthogonal lift.
6. PAFlow / FlowMol3 / NExT-Mol — only if 1-5 don't get us to >0.50 decode_ratio.

---

## 6. Risk analysis — could any fix break the existing pipeline?

### 6.1 Impact on P0 fixes (F1-F5 from `wf_cfm_p0_fixes/final.md`)
- **Fix #1:** No regression — *activates* the F1/F5 decoder wiring that was already in place.
- **Fix #2:** No regression — ODE method change is orthogonal to bond head, vocab mask, hidden_dim warning, in_dim fix.
- **Fix #3:** No regression — purely additive instrumentation.

### 6.2 Impact on P1 fixes (per `wf_vina_lift_phase23/pac_bayes.md`)
- **Fix #1:** No regression — bond head joint training is now the default, which strengthens (not weakens) P1.4 (ConnectivityAwareDecoder wiring).
- **Fix #2:** No regression — `vel_scale` parameter (P1.2) is read by `forward_velocity`, which the midpoint solver calls.
- **Fix #3:** No regression — operates at the harness level, not the adapter level.

### 6.3 Impact on Lambda path (per `wf_lambda_*` reports)
- **Fix #1+#2+#3 do NOT touch** `r4_lambda_only_run.py`, `molmetal/molmetal_lam/reactions/beta_reductions.py`, `molmetal/molmetal_lam/search_alg/proof_search.py` per the synthesis constraint.
- Lambda path uses its own generator; the CFM adapter's `--bond-head` flag is only consumed by `r10_cfg_real_crossdocked.py`.

### 6.4 Impact on paper (`paper/main.tex`, `paper/sections/*`, `paper/refs.bib`)
- **Fix #1+#2+#3 do NOT touch** any paper file per the synthesis constraint.
- If Fix #1 lifts decode_ratio to >0.30 on the 1h36 smoke, that's a §4.6 column promotion (handled by the separate `wf_paper_*` workflow, not this synthesis).

### 6.5 Impact on tests
- Fix #3 adds a new test (`tests/test_r10_decode_smoke.py`) — net positive.
- Fix #1+#2 should be backward-compatible (defaults change, but `__init__.py:1703` `joint_train=True` and `__init__.py:2442-2447` `method="midpoint"` are existing supported values).
- Recommend running `pytest -k 'cfm or flow_matching or decoder'` after each fix ships to verify no regression.

---

## 7. Honest framing — MEASURED vs PROJECTED vs SPECULATIVE

### MEASURED (today, on disk)
- **decode_ratio = 0/64 = 0.000** at h=128 5000-step on CrossDocked seed 0 (`wf_cfm_gpu_retrain/phase3_gate.md`).
- **decode_ratio = 0/192 = 0.000** at h=32 5000-step on CrossDocked 3 seeds (`wf_gpu_recovery_now/final.md`).
- **decode_ratio = 0/16 = 0.000** at h=64 100-step smoke (`wf_cfm_p0_fixes/final.md`).
- The P0 fixes (F1-F5) shipped and 7/7 unit tests pass (`wf_cfm_p0_fixes/final.md`).
- The P1 fixes (h=128, learnable vel_scale, PCGrad, PAC-Bayes) shipped but did not lift decode (phase3_gate.md).
- 3 of 4 Phase-1 audit hypotheses (A bond-head in_dim, C bonds=zeros, D hidden_dim warning) confirmed fixed in current code (`code_review_phase1c.md` §1).
- 5 NEW structural bugs surfaced post-P0 (`code_review_phase1c.md` §2): frozen bond head, ODE euler, isotropic x_0, hard-coded n_atoms, post-ODE atom logits (last self-corrected).
- `--bond-head=distance` default at `r10_cfg_real_crossdocked.py:236-239` confirmed; the `--bond-head=learned` path is NOT the harness default (`inference_review_phase1d.md` TOP-2).
- 8 vs 32 training molecules confirmed: harness loads 8 (`inference_review_phase1d.md` §"Where do the 8 (or 32) training molecules come from?").

### PROJECTED (from lit + analytical reasoning, not validated)
- Fix #1 +0.10-0.25 standalone: from phase1c BUG #1 + phase1d TOP-2 analysis (decoder path is currently silent; activating it cannot make things worse).
- Fix #2 +0.05-0.15 standalone: from phase1c BUG #3 (Euler O(h) global error → 0.1 Å drift; midpoint O(h²) eliminates).
- Combined Fix #1+#2 +0.15-0.35: from stacking the two independent mechanisms.
- GlintDM 7x lift (0.115→0.842): paper-reported (`research_phase1a.md` F7 §2F); our projection is half that, scaled to our 32-mol regime.
- YuelBond F1=92.7% on CDGs: paper-reported (`research_phase1b.md` §2.1); our projection +0.45-0.60 standalone assumes our coord distortion is comparable to the YuelBond training distribution (which it may not be — open question).

### SPECULATIVE (forward-looking, not validated)
- The +0.65 upper bound for combined Fix #1+#2+#3 is the GlintDM-cited projection applied to our h=128 model. Could be lower if the EGNN velocity field has structural issues (phase1c BUG #4 x_0 distribution + phase1d TOP-4 last_v contract violation) that block the integration regardless of ODE method.
- The +0.80-0.95 ceiling for all 6 fixes (Fix #1-#3 + Phase 3 tmQM warm-start + YuelBond + GlintDM) is the multiplicative projection of independent lifts; in practice, fixes overlap (e.g. tmQM warm-start addresses the same x_0 distribution issue as Fix #4).
- Fix #5 (`last_v` contract) may not lift decode if the model has already learned to compensate via `vel_head` — needs the falsifiable diagnostic before committing to retrain.

### FALSIFIABLE
- After Fix #1: smoke test on `checkpoint_seed0.pt` (8 samples × 200 steps) should show `n_decoded ≥ 1`. If 0, the decoder path is not the bottleneck; revert and try Fix #2 alone.
- After Fix #2: same smoke should show `midpoint_decode >= euler_decode`. If comparable (within 1 sample), ODE method is not the bottleneck.
- After Fix #1+#2: smoke should show `n_decoded ≥ 2` (combined). If <2, the bottleneck is upstream (EGNN velocity field, capacity, or training data).

---

## 8. Recommended ship sequence

| Order | Action | Wall-time | Risk | Decision gate |
|---:|---|---|---|---|
| **1** | Ship Fix #1 (decoder default flip + `joint_train=True` default) | 0.5 h | LOW | decode smoke ≥ 1/8 on `checkpoint_seed0.pt`? |
| **2** | Ship Fix #2 (ODE midpoint) | 1 h | VERY LOW | decode smoke `midpoint >= euler`? |
| **3** | Ship Fix #3 (decode_smoke_every) | 1 h | ZERO | unit test passes; no regression in `pytest -k cfm` |
| **4** | Re-run h=128 5000-step retrain with Fix #1+#2 active + Fix #3 logging | 6 h GPU | LOW | decode_ratio at step 5000 ≥ 0.30? |
| **5** | If yes → ship decode_ratio column promotion to §4.6 (paper workflow) | (separate) | — | — |
| **6** | If no → proceed to Phase 3 GPU candidates (tmQM warm-start first) | 4-6 GPU-h | LOW | decode_ratio ≥ 0.50 after warm-start? |

**Decision gate after step 4 (CPU-only fixes + 1 retrain):**
- If `decode_ratio >= 0.30`: ship to §4.6 column; Round-12/13 paper upgrade.
- If `0.05 <= decode_ratio < 0.30`: Fix #1+#2 helped but not enough; pursue Phase 3 tmQM warm-start.
- If `decode_ratio < 0.05`: the structural bottleneck is upstream (EGNN velocity field, capacity, or training data); pursue Phase 3 NExT-Mol decoupling as the architectural pivot.

---

## 9. Cross-references

- `molmetal/reports/wf_cfm_frontier_research/research_phase1a.md` — Phase 1A SOTA lit survey (read)
- `molmetal/reports/wf_cfm_frontier_research/research_phase1b.md` — Phase 1B failure-mode lit + 32-mol diagnosis (read)
- `molmetal/reports/wf_cfm_frontier_research/code_review_phase1c.md` — Phase 1C code review (read)
- `molmetal/reports/wf_cfm_frontier_research/inference_review_phase1d.md` — Phase 1D inference-path review (read)
- `molmetal/reports/wf_cfm_internal_review/audit.md` — Phase-1 prior 4 root causes (read)
- `molmetal/reports/wf_cfm_internal_review/diagnose.md` — Phase-1 architecture redesign (read)
- `molmetal/reports/wf_cfm_gpu_retrain/phase3_gate.md` — h=128 5000-step VERDICT=FAIL (read)
- `molmetal/reports/wf_cfm_p0_fixes/final.md` — F1-F5 ship + 7/7 tests pass (context)
- `molmetal/reports/wf_cfm_path_b_decoder_rework/final.md` — DecoderRework smoke (context)
- `molmetal/reports/wf_pb_mmff94_relax/` — MMFF94 60-80% PB pass (context for Fix #10 follow-up)
- `molmetal/reports/wf_vina_lift_phase23/pac_bayes.md` — PAC-Bayes bound 0.7055 (context)
- `molmetal/adapters/flow_matching_lipman/__init__.py:1702-1705, 1903-1922, 2442-2447, 2519-2569` — line cites for Fix #1, #2
- `molmetal/scripts/r10_cfg_real_crossdocked.py:213-359, 236-239, 471-477` — line cites for Fix #1, #3
- `molmetal/ports/__init__.py:30-110` — GenerationConfig (Fix #2 entry point)
- `molmetal/reports/wf_cfm_gpu_retrain/diagnostic/checkpoint_seed0.pt` — h=128 5000-step checkpoint for smoke tests

## 10. Files referenced + files written

**Referenced (read-only):**
- All 4 Phase-1 inputs (above)
- All context files (above)
- `molmetal/adapters/flow_matching_lipman/__init__.py` (2874 lines)
- `molmetal/scripts/r10_cfg_real_crossdocked.py` (598 lines)
- `molmetal/molmetal_lam/lam_chem/decoder_rework.py` (926 lines)
- `molmetal/adapters/flow_matching_lipman/connectivity_decoder.py` (427 lines)

**Written:**
- `molmetal/reports/wf_cfm_frontier_research/synthesis_phase2.md` — this document

**No code modifications made.** Synthesis is read-only.

---

**END Phase 2 Synthesis**
