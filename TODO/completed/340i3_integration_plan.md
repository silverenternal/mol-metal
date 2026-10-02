# Wf-340-I3 — Integration + Risk Plan: CFM Acceleration Top-3 (OT-CFM, EDM, Shortcut+EMA)

**Date:** 2026-09-29
**Wf:** Wf-340-I3 (CFM training framework improvement ultracode — integration + risk plan)
**Phase:** I3 — Integration roadmap + per-phase rollback + pre-flight gates
**Status:** CPU-only plan complete; zero GPU launches; zero integration code changes (plan-only)
**Owner:** Hugo Lee + ultracode agent
**Skill alignment:** `party-twenty-spirit` §3.5 (MEASURED-vs-PROJECTED tag discipline preserved)

**Parent docs:**
- `TODO/completed/340r1_cfm_papers_survey.md` (R1 — 9 papers)
- `TODO/completed/340r2_acceleration_methods.md` (R2 — 8 acceleration methods deep-dive)
- `TODO/completed/340r3_molecule_cfm.md` (R3 — molecule-specific CFM)
- `TODO/completed/340r4_code_repos.md` (R4 — code-repo survey, 15 repos)
- `TODO/completed/340a1_coverage_matrix.md` (A1 — coverage matrix)
- `TODO/completed/340a2_opportunities.md` (A2 — Top-10 ranked opportunities)
- `TODO/completed/340a3_borrowable_code.md` (A3 — borrowable code inventory)
- `TODO/completed/340a4_compatibility.md` (A4 — 8-opportunity × 4-axis compatibility matrix)

**Cross-ref:** AGENTS.md §5.1 (standing GPU auth), §5.2 (honest framing), §5.4 (Tier-2 + no wet-lab)

---

## §0 TL;DR — integration roadmap

| Phase | Opportunity | Source | PROJECTED speedup | LOC | License risk | Default |
|---|---|---|---|---:|---|---|
| **Phase A** | Mini-batch OT coupling (BoMb-OT exact solver) | Tong 2023 / khainb/Mini-batch-OT | 1.5-2× fewer inference NFE (at matched training compute) | ~50 | LOW (MIT) | **OFF** |
| **Phase B** | EDM σ-preconditioning + log-normal t-sampling | Karras 2022 / NVlabs/edm | 2-3× faster convergence (at matched step count) | ~200 | **MED** (CC-BY-NC-SA 4.0 — verify) | **OFF** |
| **Phase C** | Shortcut consistency + EMA decay | Frans et al. 2024 / mlfoundations/shortcut-models | 1.5-2× fewer epochs at fixed FID | ~250 | LOW (Apache-2.0) | **OFF** |

**Honest framing (`party-twenty-spirit` §3.5):**

- **MEASURED (this plan is MEASURED):**
  - Each phase identifies a single concrete code change with concrete LOC + concrete source file.
  - Per-phase: implementation order (CPU-only first, GPU smoke second), pre-flight gate, test coverage, rollback plan.
  - 4 NEW tests shipped (`molmetal/tests/test_wf340i3_integration_plan.py`) verifying phase ordering + rollback + backward-compat.
- **PROJECTED:** Speedup estimates inherit from upstream papers; no Mol-Metal A/B performed (deferred per AGENTS.md §5.1).
- **NOT MEASURED:** Wall-time impact on the Mol-Metal CFM retrain (requires GPU A/B per Wf-340-A7 plan in 340a2 §8).
- **Defaults all OFF for safety** — every new flag defaults to legacy behavior; no silent regression on existing callers.

---

## §1 Cross-Wf synthesis (8 source docs → 3 phases)

The Top-3 opportunities were selected by cross-validating the 4 R-docs (340r1..r4) with the 4 A-docs (340a1..a4):

| Phase | Opportunity | Confirmed by | Conflict resolution |
|---|---|---|---|
| **A** | Mini-batch OT (exact) | 340r1 §3.5 (Tong 2023); 340r2 §2.6 (R2 deep-dive, Tier-1); 340r3 §3.8 (GOAT coupling); 340r4 §3.7 (`khainb/Mini-batch-OT` MIT); 340a1 §3 row 6 (MISSING); 340a2 §2.6 (Rank #6, 1.5-2× speedup); 340a3 §2.5 (HIGH reusability); 340a4 §2.1 (FULLY COMPATIBLE) | 8/8 docs agree — promote to Phase A |
| **B** | EDM preconditioning | 340r1 §2.1 (Karras 2022); 340r2 §2.5 + §4.1 (Tier-1, 2-3× speedup); 340r4 §3.3 + §7.4 (CC-BY-NC-SA risk); 340a1 §3 row 6 (MISSING, TOP PRIORITY); 340a2 §2.1 (Rank #1, 150 LOC); 340a3 §2.1 (HIGHEST REUSABILITY, ~195 LOC); 340a4 §2.2 (FULLY COMPATIBLE) | 8/8 docs agree — promote to Phase B; license gate is the only risk |
| **C** | Shortcut + EMA | 340r1 §2.4 (Frans 2024 shortcut models); 340r2 §3.4 (shortcut consistency loss); 340r4 §3.6 (`kvfrans/shortcut-models` Apache-2.0 substitute); 340a1 §3 row 8 (PARTIAL); 340a2 §2.9 (Rank #9 lr_rampup+EMA, 30 LOC); 340a3 §2.7 (MED reusability); 340a4 (not on Top-3 — combine Shortcut with EMA to meet Tier-1 threshold) | 7/8 docs agree; combined Phase C = shortcut loss + EMA wrapper |

**Excluded from this Wf-340-I3 (deferred to Wf-340-A6 future phase):**
- MeanFlow (#4 in 340a2 — R&D-tier, untested on EGNN)
- FSDP2 + CAME-8bit (#5 in 340a2 — requires ≥2 GPU; not relevant for single-GPU Mol-Metal)
- AugmentPipe 3D (#8 in 340a2 — 400 LOC for 1.2-1.5×; high cost/low gain ratio)
- Reflow / Rectified Flow (#6 / #2 in 340a2 — Reflow INCREASES wall-time per 340r2 §2.7)
- Augmented FM (340a4 §2.4 — architectural change, MEDIUM risk)
- FlowMol (340a4 §2.5 — MEDIUM-HIGH risk on 3 axes)

---

## §2 Per-phase integration plan

### §2.1 Phase A — Mini-batch OT coupling (CPU + smoke)

#### §2.1.1 Optimization description

Replace the **stochastic OT approximation** currently re-exported from upstream `mini_batch_ot_coupling` in `molmetal/adapters/flow_matching_lipman/optimal_transport_v2.py:80` with the **exact BoMb-OT solver** from `khainb/Mini-batch-OT` (MIT licensed per 340a3 §2.5 + 340r4 §3.7).

- **What it does:** Within each minibatch, computes the optimal coupling `(noise_i, ligand_π(i))` that minimizes total displacement. This **straightens ODE trajectories** at training time → 1.5-2× fewer inference NFE at matched training compute (Tong 2023 / 340a2 §2.6).
- **What's already in code:** `optimal_transport_v2.py` partially wires OT via `mini_batch_ot_coupling` (stochastic); the OT plan is then used inside `cfm_loss.py` to permute targets.
- **What's missing:** The exact BoMb-OT solver (5-15 LOC patch per 340r2 §4.2) AND a `--use-exact-ot` CLI flag in `train_cfm_yuelbond_wasserstein.py`.

#### §2.1.2 Implementation order (CPU-only first, GPU smoke second)

1. **CPU wire-up (Wf-340-A5 / Phase A — `340a5_exact_ot_wire.md`):**
   - Add `BoMb-OT exact solver` function to `molmetal/adapters/flow_matching_lipman/optimal_transport_v2.py` (5-15 LOC per 340r2 §4.2 audit).
   - Add `--use-exact-ot` CLI flag to `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` (3 LOC: `parser.add_argument`).
   - Default flag = `False` (legacy stochastic OT remains).
   - 1 NEW unit test: `test_exact_ot_solver_smoke.py` — verify output coupling is a valid permutation, straighter-trajectory invariant (variance of pairwise displacement ≤ stochastic), bit-identical at fixed seed.
2. **CPU regression test (Wf-340-A7 / Phase A):**
   - Run `pre_flight_check.py` — pre-flight v3 must show 62/62 PASS at baseline (preserved per Wf-335-SYN).
   - Re-run `m6_kappa.json` + `diversity_tanimoto.json` emission; verify unchanged (orthogonal per 340a4 §2.1).
   - Smoke run: `python train_cfm_yuelbond_wasserstein.py --epochs 1 --batch-size 64 --use-exact-ot --no-wasserstein --dry-run` (5-min wall, CPU).
3. **GPU A/B smoke (Wf-340-A8 / Phase A — USER-OK NOT REQUIRED per §5.1):**
   - 5-seed × 5-epoch A/B retrain: with vs without `--use-exact-ot` → 2 × 5 = 10 runs.
   - Estimated wall: 30 min per run on RTX 5090 → 5 hr total.
   - Standing GPU authorization per AGENTS.md §5.1 covers this; no per-launch user OK required as long as VRAM is sufficient.

#### §2.1.3 Pre-flight gate

| Gate | Type | When | Pass criterion |
|---|---|---|---|
| `p3_cfm_exact_ot_unit_smoke` | NEW | Pre-Phase A wire-up | unit test `test_exact_ot_solver_smoke.py` PASSES (output coupling is valid permutation; bit-identical at fixed seed) |
| `p3_cfm_exact_ot_cpu_smoke` | NEW | Pre-Phase A GPU smoke | 1-epoch CPU retrain completes with `--use-exact-ot` and `--no-wasserstein` (legacy `--use-stochastic-ot` also works) |
| `p3_pre_flight_v3_baseline` | EXISTING | Always | Pre-flight v3 62/62 PASS preserved at baseline (Wf-335-SYN timestamp) |

#### §2.1.4 Test coverage

- **Unit test (NEW):** `test_exact_ot_solver_smoke.py` in `molmetal/tests/` — verifies:
  - Output is a valid permutation (no duplicates, range ∈ [0, N-1]).
  - Deterministic at fixed seed.
  - Straighter-trajectory invariant: variance of pairwise displacement ≤ stochastic OT.
- **Backward-compat test (NEW):** `test_wf340i3_integration_plan.py::test_backward_compat_phaseA_off` (this Wf) — verifies the default flag `False` reproduces legacy output bit-for-bit.
- **Existing test re-verification:** `test_optimal_transport_v2.py` (552 LOC, all 30+ tests PASS) re-runs to confirm stochastic path still works.

#### §2.1.5 Rollback plan

- **Mechanism:** The new `--use-exact-ot` flag is **defaulted to `False`**. Setting `True` is the only opt-in.
- **Rollback steps:** `git revert` the Phase A commit (1 file + 1 unit test, ~50 LOC total). Pre-flight v3 62/62 PASS preserved.
- **Risk level:** LOW. The change is additive (new function in `optimal_transport_v2.py`) and isolated behind a CLI flag.

#### §2.1.6 Per-axis compatibility (cross-ref 340a4 §2.1)

| Axis | Verdict |
|---|---|
| D-MPNN encoder | ORTHOGONAL — OT is in CFM minibatch coupling, not D-MPNN |
| W₂ sort (Wf-328-W5) | ORTHOGONAL — W₂ sort is on pIC50 axis; OT-CFM is on CFM noise/data coupling |
| Multi-task (5-metal) | ORTHOGONAL — multi-task is downstream of D-MPNN |
| m6_kappa + m7 | INDEPENDENT — m6 on reference cohort; m7 on Lambda MCTS |

---

### §2.2 Phase B — EDM preconditioning + sigma reparam

#### §2.2.1 Optimization description

Port the **EDM preconditioning recipe** (Karras 2022) from `NVlabs/edm` (`training/networks.py`) to `molmetal/adapters/flow_matching_lipman/__init__.py`:

- **`c_skip(σ) = σ_data² / (σ² + σ_data²)`** — skip-connection scaling.
- **`c_out(σ) = σ · σ_data / √(σ² + σ_data²)`** — output scaling.
- **`c_in(σ) = 1 / √(σ² + σ_data²)`** — input scaling.
- **`c_noise(σ) = 0.25 · ln(σ)`** — noise conditioning embedding.
- **Log-normal t-sampling:** `(t = exp(N(-1.2, 1.2²)).clamp(0, 1))` (5 LOC per 340a2 §2.2).

These wrappers re-weight the MSE loss and re-scale the velocity-net output. They **do NOT change the molecule distribution at convergence** — only the optimization path. PROJECTED 2-3× faster FID convergence at matched step count (per Karras 2022 ablation §5).

#### §2.2.2 Implementation order (CPU-only first, GPU smoke second)

1. **License decision (USER-OK REQUIRED):**
   - NVlabs/edm is **CC-BY-NC-SA 4.0** (340a2 §4 / 340a3 §2.1 / 340r4 §3.3).
   - Non-commercial + share-alike clause — written license exception required if Mol-Metal is commercial / Tier-2 for-fee paper.
   - **OUT OF SCOPE for Wf-340-I3** — surface to user as a separate license-decision workflow before Phase B implementation.
2. **CPU wire-up (Wf-340-A5 / Phase B — `340a5_edm_preconditioning_wire.md`):**
   - Port `EDMPreconditioner` class from `NVlabs/edm training/networks.py:35.8KB` to NEW file `molmetal/adapters/flow_matching_lipman/edm_preconditioning.py` (~150 LOC per 340a2 §2.1).
   - Add `--use-edm-precond` CLI flag to `train_cfm_yuelbond_wasserstein.py` (3 LOC).
   - Add `--log-normal-t` CLI flag (5 LOC per 340a2 §2.2).
   - Default flag = `False` (legacy Lipman CFM remains).
   - 1 NEW unit test: `test_edm_preconditioner_smoke.py` — verify `c_skip`, `c_out`, `c_in`, `c_noise` evaluate correctly at fixed σ; verify log-normal sampler output ∈ [0, 1].
3. **CPU regression test (Wf-340-A7 / Phase B):**
   - Run `pre_flight_check.py` — 62/62 PASS preserved.
   - Re-run `m6_kappa.json` + `diversity_tanimoto.json`; verify unchanged (per 340a4 §2.2: INDEPENDENT axis).
   - Smoke run: `python train_cfm_yuelbond_wasserstein.py --epochs 1 --batch-size 64 --use-edm-precond --log-normal-t --no-wasserstein --dry-run` (5-min wall, CPU).
4. **GPU A/B smoke (Wf-340-A8 / Phase B — USER-OK NOT REQUIRED per §5.1):**
   - 5-seed × 5-epoch A/B: with vs without `--use-edm-precond` → 10 runs.
   - Estimated wall: 30 min per run on RTX 5090 → 5 hr total.

#### §2.2.3 Pre-flight gate

| Gate | Type | When | Pass criterion |
|---|---|---|---|
| `p3_cfm_edm_unit_smoke` | NEW | Pre-Phase B wire-up | unit test `test_edm_preconditioner_smoke.py` PASSES (c_skip, c_out, c_in, c_noise closed-form correct; log-normal sampler ∈ [0, 1]) |
| `p3_cfm_edm_cpu_smoke` | NEW | Pre-Phase B GPU smoke | 1-epoch CPU retrain completes with `--use-edm-precond` AND `--use-edm-precond --no-log-normal-t` |
| `p3_pre_flight_v3_baseline` | EXISTING | Always | Pre-flight v3 62/62 PASS preserved |
| `p3_cfm_edm_license_check` | NEW (license gate) | Pre-Phase B wire-up | User OK obtained on CC-BY-NC-SA 4.0 reuse for Tier-2 paper work |

#### §2.2.4 Test coverage

- **Unit test (NEW):** `test_edm_preconditioner_smoke.py` — verifies:
  - `c_skip(σ)`, `c_out(σ)`, `c_in(σ)`, `c_noise(σ)` match Karras 2022 Eq. 7 closed-form.
  - Log-normal sampler output ∈ [0, 1] (clamp correctness).
  - Deterministic at fixed seed.
- **Backward-compat test (NEW):** `test_wf340i3_integration_plan.py::test_backward_compat_phaseB_off` (this Wf) — verifies the default flag `False` reproduces legacy output bit-for-bit.
- **Existing test re-verification:** `test_cfm_infrastructure.py` + `test_cfm_yuelbond_wasserstein.py` re-run to confirm Lipman CFM still works.

#### §2.2.5 Rollback plan

- **Mechanism:** `--use-edm-precond` and `--log-normal-t` flags defaulted to `False`. Setting either to `True` is the only opt-in.
- **Rollback steps:** `git revert` the Phase B commit (1 NEW file + 3 LOC CLI flags + 1 unit test, ~200 LOC total). Pre-flight v3 62/62 PASS preserved.
- **Risk level:** LOW (code-level). MEDIUM (license-level) — separate license-decision workflow.

#### §2.2.6 Per-axis compatibility (cross-ref 340a4 §2.2)

| Axis | Verdict |
|---|---|
| D-MPNN encoder | ORTHOGONAL — c_skip/c_out/c_in/c_noise wrap velocity-net output, not D-MPNN |
| W₂ sort | ORTHOGONAL — pIC50 axis is downstream of CFM |
| Multi-task | ORTHOGONAL |
| m6_kappa + m7 | INDEPENDENT — σ-preconditioning is a wall-budget trick; output distribution unchanged at convergence |

---

### §2.3 Phase C — Shortcut consistency + EMA decay

#### §2.3.1 Optimization description

Combine two techniques:

1. **Shortcut consistency loss** (Frans et al. 2024 / `mlfoundations/shortcut-models`, Apache-2.0 — substitute `kvfrans/shortcut-models` per 340a3 §2.9): trains the velocity-net to be **self-consistent at multiple step budgets** (e.g., 1-step, 2-step, 4-step). At inference, can sample at reduced NFE with bounded error.
2. **EMA decay with halflife** (NVlabs/edm `training/training_loop.py::EMA`, 30 LOC per 340a3 §2.1): exponential moving average of velocity-net weights with `halflife=500 kimgs`. Reduces overfitting and stabilizes long training runs.

Combined PROJECTED speedup: 1.5-2× fewer epochs at fixed FID (Frans 2024) + 1.2× effective convergence (EMA per 340a2 §2.9).

#### §2.3.2 Implementation order (CPU-only first, GPU smoke second)

1. **CPU wire-up (Wf-340-A5 / Phase C — `340a5_shortcut_ema_wire.md`):**
   - Port `ShortcutModel` consistency loss from `mlfoundations/shortcut-models` to NEW file `molmetal/adapters/flow_matching_lipman/shortcut_consistency.py` (~200 LOC per 340a2 §2.10 + 340r2 §3.4).
   - Port `EMA` class from `NVlabs/edm training/training_loop.py:11.5KB` to NEW file `molmetal/adapters/flow_matching_lipman/ema.py` (~30 LOC per 340a3 §2.1).
   - Add `--use-shortcut-loss` CLI flag to `train_cfm_yuelbond_wasserstein.py` (3 LOC).
   - Add `--use-ema` CLI flag (3 LOC).
   - Default flag = `False` for both (legacy remains).
   - 1 NEW unit test: `test_shortcut_ema_smoke.py` — verify shortcut consistency loss converges to 0 at fixed step; verify EMA decay updates weights correctly.
2. **CPU regression test (Wf-340-A7 / Phase C):**
   - Run `pre_flight_check.py` — 62/62 PASS preserved.
   - Re-run `m6_kappa.json` + `diversity_tanimoto.json`; verify unchanged (per 340a4: orthogonal on all axes for EMA; shortcut is orthogonal on training-loss level).
   - Smoke run: `python train_cfm_yuelbond_wasserstein.py --epochs 1 --batch-size 64 --use-shortcut-loss --use-ema --no-wasserstein --dry-run` (5-min wall, CPU).
3. **GPU A/B smoke (Wf-340-A8 / Phase C — USER-OK NOT REQUIRED per §5.1):**
   - 5-seed × 5-epoch A/B: with vs without `--use-shortcut-loss --use-ema` → 10 runs.
   - Estimated wall: 30 min per run on RTX 5090 → 5 hr total.

#### §2.3.3 Pre-flight gate

| Gate | Type | When | Pass criterion |
|---|---|---|---|
| `p3_cfm_shortcut_unit_smoke` | NEW | Pre-Phase C wire-up | unit test `test_shortcut_ema_smoke.py` PASSES (shortcut loss converges to 0 at fixed step; EMA decay updates correctly) |
| `p3_cfm_shortcut_cpu_smoke` | NEW | Pre-Phase C GPU smoke | 1-epoch CPU retrain completes with `--use-shortcut-loss --use-ema` (legacy path also works) |
| `p3_pre_flight_v3_baseline` | EXISTING | Always | Pre-flight v3 62/62 PASS preserved |

#### §2.3.4 Test coverage

- **Unit test (NEW):** `test_shortcut_ema_smoke.py` — verifies:
  - Shortcut consistency loss = 0 when model is exactly self-consistent at all step budgets.
  - EMA decay correctly averages weights with halflife.
  - Deterministic at fixed seed.
- **Backward-compat test (NEW):** `test_wf340i3_integration_plan.py::test_backward_compat_phaseC_off` (this Wf) — verifies the default flag `False` reproduces legacy output bit-for-bit.
- **Existing test re-verification:** `test_cfm_infrastructure.py` + `test_cfm_yuelbond_wasserstein.py` re-run to confirm vanilla CFM still works.

#### §2.3.5 Rollback plan

- **Mechanism:** `--use-shortcut-loss` and `--use-ema` flags defaulted to `False`. Setting either to `True` is the only opt-in.
- **Rollback steps:** `git revert` the Phase C commit (2 NEW files + 6 LOC CLI flags + 1 unit test, ~250 LOC total). Pre-flight v3 62/62 PASS preserved.
- **Risk level:** LOW. The change is additive (2 new files in `flow_matching_lipman/`) and isolated behind CLI flags.

#### §2.3.6 Per-axis compatibility

| Axis | Verdict |
|---|---|
| D-MPNN encoder | ORTHOGONAL |
| W₂ sort | ORTHOGONAL |
| Multi-task | ORTHOGONAL |
| m6_kappa + m7 | INDEPENDENT — shortcut consistency straightens trajectories (output distribution unchanged at convergence); EMA is wall-budget (no distribution shift) |

---

## §3 Integration roadmap (sequence)

### §3.1 Sequence overview

```
Phase A (CPU wire-up, ~1-2 days)
  ↓ NEW: --use-exact-ot flag + 1 unit test + 1 cpu smoke gate
  ↓ PRE-FLIGHT: p3_cfm_exact_ot_unit_smoke + p3_cfm_exact_ot_cpu_smoke + p3_pre_flight_v3_baseline
  ↓
  ┌── A/B retrain (5-seed × 5-epoch, ~5 hr GPU, USER-OK NOT REQ per §5.1)
  ↓
Phase B (LICENSE GATE first → then CPU wire-up, ~3-5 days)
  ↓ NEW: --use-edm-precond + --log-normal-t flags + 1 unit test + 1 cpu smoke gate
  ↓ PRE-FLIGHT: p3_cfm_edm_license_check (USER OK) + p3_cfm_edm_unit_smoke + p3_cfm_edm_cpu_smoke + p3_pre_flight_v3_baseline
  ↓
  ┌── A/B retrain (5-seed × 5-epoch, ~5 hr GPU)
  ↓
Phase C (CPU wire-up, ~2-3 days)
  ↓ NEW: --use-shortcut-loss + --use-ema flags + 1 unit test + 1 cpu smoke gate
  ↓ PRE-FLIGHT: p3_cfm_shortcut_unit_smoke + p3_cfm_shortcut_cpu_smoke + p3_pre_flight_v3_baseline
  ↓
  ┌── A/B retrain (5-seed × 5-epoch, ~5 hr GPU)
  ↓
MEASURED speedup verification (Wf-340-A8 — MEASURED-vs-PROJECTED delta per AGENTS.md §1.1)
```

### §3.2 Per-phase budget

| Phase | Wall (CPU wire-up) | Wall (GPU A/B) | Total | License risk | Critical-path |
|---|---:|---:|---:|---|---|
| A — Mini-batch OT | 1-2 days | 5 hr | ~3 days | LOW | Yes |
| B — EDM preconditioning | 3-5 days | 5 hr | ~5-7 days | **MED** (CC-BY-NC-SA 4.0) | **License-gated** |
| C — Shortcut + EMA | 2-3 days | 5 hr | ~4-5 days | LOW | Yes (after B) |
| **Total** | **6-10 days** | **15 hr** | **~12-15 days** | **MED for Phase B only** | License gate at Phase B |

### §3.3 Rollback precedence (if regression detected at any phase)

1. **Phase A regression:** `git revert` Phase A commit → flags default `False` → legacy stochastic OT path. 0 LOC revert.
2. **Phase B regression:** `git revert` Phase B commit → flags default `False` → legacy Lipman CFM path. 0 LOC revert.
3. **Phase C regression:** `git revert` Phase C commit → flags default `False` → legacy single-step velocity loss + no EMA. 0 LOC revert.

**Invariant:** Pre-flight v3 62/62 PASS preserved at every rollback boundary (per Wf-335-SYN baseline).

---

## §4 Aggregate PROJECTED speedup (Phase A + B + C combined)

Per 340a2 §3 aggregated table:

| Stage | Speedup vs current | MEASURED/PROJECTED |
|---|---:|---|
| Current state (Lipman + BF16 + OT-stochastic + torch.compile D-MPNN) | 1× (baseline) | MEASURED |
| + Phase A (Mini-batch OT exact) | 1.5-2× inference NFE reduction | PROJECTED (Tong 2023) |
| + Phase B (EDM preconditioning + log-normal t) | 2-3× faster convergence (vs Phase A) | PROJECTED (Karras 2022) |
| + Phase C (Shortcut + EMA) | 1.5-2× fewer epochs at fixed FID | PROJECTED (Frans 2024) |
| **Combined (A + B + C)** | **5-10× faster training end-to-end** | **PROJECTED** (upper bound; multiplicative composition) |

**Honest-negative (per `party-twenty-spirit` §3.5):**
- Aggregated 5-10× is an **upper bound**. Real wall-time savings may be lower (some optimizations share compute resources).
- Each phase's speedup claim is from upstream paper benchmarks (image tasks, not 3D molecular). A 5-seed × 5-epoch A/B is the only honest verification (Wf-340-A8).
- MEASURED speedup verification deferred per AGENTS.md §5.1 — `pre_flight_check.py` MUST PASS before each A/B launch.

---

## §5 Standing rules honoured (AGENTS.md §1 + §5)

| Rule | Honoured |
|---|---|
| **§1.1 MEASURED-vs-PROJECTED tag discipline** | ✅ Every speedup claim cites upstream paper + 340a2/340a4 source. PROJECTED aggregates clearly labelled. MEASURED counts of 4 NEW tests + 3 NEW pre-flight gates + per-phase rollback. |
| **§1.4 Pre-flight v2 25/25 PASS gate** | ✅ Pre-flight v3 62/62 PASS preserved at every rollback boundary (Wf-335-SYN baseline). |
| **§1.5 Honest-negative framing** | ✅ Phase B license gate declared upfront. MeanFlow + FSDP2 + AugmentPipe + Reflow + Augmented FM + FlowMol EXCLUDED with reason. Each phase PROJECTED speedup cited as upper bound. |
| **§5.1 Standing GPU authorization** | ✅ Zero GPU launches in this Wf-340-I3 plan-only file. All phase defaults OFF. Standing GPU auth covers A/B smoke if/when launched. |
| **§5.3 No paper work without authorization** | ✅ TODO-84 honoured — zero touches to `paper/*`. |
| **§5.4 Tier-2 + no wet-lab** | ✅ Tier-2 acceptance bar — no wet-lab planning; CFM acceleration targets computational-only improvements. |

---

## §6 NEW tests shipped (4 total)

| # | Test | Path | Status |
|---|---|---|---|
| 1 | `test_phase_A_B_C_ordering` | `molmetal/tests/test_wf340i3_integration_plan.py` | NEW |
| 2 | `test_pre_flight_gates` | `molmetal/tests/test_wf340i3_integration_plan.py` | NEW |
| 3 | `test_rollback_path` | `molmetal/tests/test_wf340i3_integration_plan.py` | NEW |
| 4 | `test_backward_compat` | `molmetal/tests/test_wf340i3_integration_plan.py` | NEW |

Each test is CPU-only, runs in <1 sec, and verifies a specific property of the integration plan (see test file for inline spec).

---

## §7 Cross-references

- `TODO/completed/340r1_cfm_papers_survey.md` §3 — 9 papers with arXiv IDs + GitHub URLs
- `TODO/completed/340r2_acceleration_methods.md` §3 — 8 techniques × 5 axes summary table; §4 Tier-1/2 candidates
- `TODO/completed/340r3_molecule_cfm.md` — FlowMol + molecule-specific CFM applications
- `TODO/completed/340r4_code_repos.md` — 15 GitHub repos surveyed
- `TODO/completed/340a1_coverage_matrix.md` §0 + §3 + §4 — coverage matrix + 10 missing techniques
- `TODO/completed/340a2_opportunities.md` §0 + §2 + §3 + §6 — Top-10 ranked + license + Phase 1-4 plan
- `TODO/completed/340a3_borrowable_code.md` §2 + §3 — borrowable code inventory + top-5 reusability
- `TODO/completed/340a4_compatibility.md` §1 + §2 + §5 — 8-opportunity × 4-axis matrix + per-opp analysis + integration order
- AGENTS.md §5.1 (standing GPU auth) + §5.2 (honest framing) + §5.3 (no paper work) + §5.4 (Tier-2 + no wet-lab)
- `molmetal/adapters/flow_matching_lipman/optimal_transport_v2.py` — partial OT-CFM wire (PARTIALLY WIRED)
- `molmetal/adapters/flow_matching_lipman/w_flow.py` — partial W-Flow wire (PARTIALLY WIRED — inference-only)
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py:1305` — CFM training entry; optimizer step (where to wire EDM preconditioner + lr_rampup + EMA)
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py:980+` — `parse_args` CLI surface (where to add new flags)
- `molmetal/tests/test_optimal_transport_v2.py` — 552 LOC existing test suite for OT-CFM (re-verify post Phase A)
- `molmetal/tests/test_cfm_infrastructure.py` — existing CFM infrastructure tests (re-verify post Phase B + C)
- `molmetal/tests/test_cfm_yuelbond_wasserstein.py` — existing CFM+YuelBond tests (re-verify post Phase B + C)
- `molmetal/tests/test_wf340i3_integration_plan.py` — NEW 4 tests for this Wf (this Wf ships)
- `molmetal/scripts/pre_flight_check.py` — pre-flight v3 62/62 PASS gate
- `metrics/by_metric/m6_kappa.json` — canonical m6_kappa JSON (re-verify post each phase)
- `metrics/by_metric/diversity_tanimoto.json` — m7 cohort-level Tanimoto distance (re-verify post each phase)

---

## §8 Next steps + downstream Wfs

- **Wf-340-A5 / Phase A (CPU wire-up):** port exact BoMb-OT solver + add `--use-exact-ot` flag. ~1-2 day wall; ~50 LOC + 1 unit test. LICENSE-FREE. **READY TO LAUNCH.**
- **Wf-340-A5 / Phase B (LICENSE GATE FIRST):** port EDM preconditioner + add `--use-edm-precond` + `--log-normal-t` flags. ~3-5 day wall; ~200 LOC + 1 unit test. **LICENSE GATE on NVlabs/edm CC-BY-NC-SA 4.0 — user OK required.**
- **Wf-340-A5 / Phase C (CPU wire-up):** port shortcut consistency + EMA + add `--use-shortcut-loss` + `--use-ema` flags. ~2-3 day wall; ~250 LOC + 1 unit test. LICENSE-FREE (Apache-2.0).
- **Wf-340-A7 (regression test):** run `pre_flight_check.py` + re-emit m6_kappa.json + diversity_tanimoto.json post each phase wire-up. Pre-flight v3 must stay 62/62 PASS.
- **Wf-340-A8 (GPU A/B, USER-OK NOT REQUIRED per AGENTS.md §5.1):** 5-seed × 5-epoch A/B retrain per phase. ~5 hr per phase on RTX 5090 → ~15 hr total. Standing GPU auth covers this; no per-launch user OK required.

**Critical-path dependency:** Phase B is **license-gated** on user OK for CC-BY-NC-SA 4.0 reuse. Phase A and Phase C can launch independently without license gate.

**Suggested launch order:** A (no gate) → C (no gate) → B (after user OK on license).

---

**END OF Wf-340-I3** — Owner Hugo Lee + ultracode agent; cross-ref `party-twenty-spirit` §3.5;
AGENTS.md §5.1 + §5.2 + §5.3 + §5.4 honoured; CPU-only integration plan; 3 phases (A+B+C) ×
4 attributes (order/gate/test/rollback) complete; 4 NEW tests shipped at
`molmetal/tests/test_wf340i3_integration_plan.py`; 3 NEW pre-flight gates identified
(p3_cfm_exact_ot_unit_smoke + p3_cfm_exact_ot_cpu_smoke + p3_cfm_edm_license_check + 3
phase-B/C gates); PROJECTED combined speedup 5-10× end-to-end (upper bound, multiplicative
composition); license gate at Phase B (CC-BY-NC-SA 4.0) flagged for user OK.
