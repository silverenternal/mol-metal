# Wf-385-A1 — Wf-380 CLI flag → SCAFFOLD call audit

**Document ID:** Wf-385-A1 / 385a1_wf380_cli_flag_audit
**Authoring date:** 2026-09-30
**Owner:** Hugo Lee + Wf-385-A1 agent
**Trigger:** Per `AGENTS.md` §6 row 282 Wf-380-B1 honest-negative caveat #1 ("SCAFFOLD-only integration — 14 NEW CFM + 3 NEW DMPNN CLI flags wired as observability + dispatcher patterns; loss / sampling wire-in at each `args.X` read-site documented but NOT implemented") + user voice 2026-09-30 verbatim *"如果是同一个问题一直没修好的话得做调研+review尝试更深层的修复"*. CPU-only docs+audit wave. 0 NEW tests. 0 GPU launches. 0 `paper/*` touches. 0 wet-lab planning.

---

## §0 TL;DR

**Verdict (READ-only audit, MEASURED via grep + Read on the trainer source tree at 2026-09-30 timestamp)**:

- **14 NEW CFM flags + 3 NEW DMPNN flags = 17 flags SHIPPED at Wf-380-B1**
- **0 flags (0/17) have an actual SCAFFOLD call in the main training loop**
- **0 flags (0/17) have the SCAFFOLD module imported by the trainer**
- **17 flags (17/17) are FLAG_ONLY — declared in `parse_args` + printed in the Wf-380 banner code + ZERO loss / sampling / dispatch wire-in to the live training computation**

**Same pattern has recurred across 3 consecutive ultracode waves**:
- Wf-340 SHIPPED 6 CFM acceleration modules (acceleration.py / log_normal_t / lr_warmup) but DID NOT WIRE them to the trainer at Wf-340-B1 ship time — wire-in was completed by **Wf-348-C1** (subsequent wave)
- Wf-353 SHIPPED Triton OT kernel + Shortcut distillation SCAFFOLDS but DID NOT WIRE them at Wf-353-B1 ship time — wire-in was completed by **Wf-353-A1 / Wf-353-A2** in the same wave (the only wave where the SCAFFOLD and wire-in shipped together)
- Wf-380 SHIPPED 14 SCAFFOLD modules + 17 NEW CLI flags but DID NOT WIRE any of them — wire-in remains **deferred to Wf-381+**

**Closure path = Wf-381 (loss / sampling / OT-coupling wire-in wave)**: per-flag wire-in LOC estimate = **~280 LOC across 17 sites**, fully CPU-only, no GPU launches (loss modification is integration work; MEASURED speedup requires Blackwell retrain per AGENTS.md §5.1 + `kill_then_adapt` §0).

---

## §1 Per-flag audit table

Audit method: `grep -nE "<flag-name>" molmetal/scripts/train_{cfm_yuelbond,dmpnn_multitask}_wasserstein.py` + manual Read on the trainer main loop (lines 1500-2500 CFM + 3000-3950 DMPNN).

### §1.1 CFM trainer flags (14 NEW per Wf-380-B1)

| # | Flag | Default | Wf-380-A SCAFFOLD source | SCAFFOLD module exists | SCAFFOLD imported by trainer? | SCAFFOLD called in main loop? | Verdict |
|--:|---|:---:|---|---|:---:|:---:|---|
| 1 | `--use-ot-cfm` / `--no-use-ot-cfm` | ON | Wf-380-A1 #2 `optimal_transport_v2_solver.ot_solver` | ✅ 144 LOC + 4 tests | ❌ NO (only pre-existing `optimal_transport_v2.exact_minibatch_ot_coupling` from Wf-348-C1 imported at line 330) | ❌ NO | **FLAG_ONLY** |
| 2 | `--stable-vm` / `--no-stable-vm` | ON | Wf-380-A1 #3 `stable_vm.stable_vm_loss` | ✅ 95 LOC + 4 tests | ❌ NO | ❌ NO | **FLAG_ONLY** |
| 3 | `--so3-average` / `--no-so3-average` | ON | Wf-380-A1 #4 `so3_average.so3_average_loss` | ✅ 129 LOC + 4 tests (CFM trainer uses Wf-348-C1 `so3_averaged_fm.so3_averaged_fm_loss` via `--use-so3-averaged-fm`, NOT Wf-380 alias) | ❌ NO (only pre-existing `so3_averaged_fm.so3_averaged_fm_loss` from Wf-348-C1 imported at line 343) | ⚠️ PARTIAL — pre-existing `--use-so3-averaged-fm` (Wf-348-C1) IS wired at `train_one_epoch` line 877 via `use_so3_averaged_fm` arg; Wf-380 alias `--so3-average` is NOT wired | **FLAG_ONLY** (alias) / **WIRED** (legacy Wf-348-C1 flag) |
| 4 | `--log-normal-t` / `--no-log-normal-t` | ON | Wf-380-A1 #9 `log_normal_t.log_normal_t_sampling` | ✅ 59 LOC + 4 tests | ❌ NO | ⚠️ PARTIAL — pre-existing `--use-log-normal-t` (Wf-348-C1) IS wired via `accel_bundle.use_log_normal_t` at line 2093; Wf-380 alias `--log-normal-t` is NOT wired | **FLAG_ONLY** (alias) / **WIRED** (legacy Wf-348-C1 flag) |
| 5 | `--triton-ot` / `--no-triton-ot` | ON | Wf-380-A1 #5 `triton_mini_batch_ot.triton_mini_batch_ot` | ✅ 110 LOC + 4 tests | ❌ NO | ⚠️ PARTIAL — pre-existing `--triton-minibatch-ot` (Wf-353-A1) IS wired via env-var dispatch at line 1879-1880 (`TRITON_USE_FUSED=1` + `TRITON_MINIBATCH_OT=1`); Wf-380 alias `--triton-ot` is NOT wired | **FLAG_ONLY** (alias) / **WIRED** (legacy Wf-353-A1 flag + env-var dispatch) |
| 6 | `--shortcut-distill` / `--no-shortcut-distill` | ON | Wf-380-A1 #6 `shortcut_distillation_wf380.shortcut_distillation_loss` | ✅ 121 LOC + 4 tests | ❌ NO (only pre-existing `acceleration.ExponentialMovingAverage` from Wf-348-C1 imported) | ⚠️ PARTIAL — pre-existing `--use-shortcut-distillation` (Wf-353-A2) IS referenced via `accel_bundle` but `accel_bundle` does NOT pass it to `train_one_epoch`; Wf-380 alias `--shortcut-distill` is NOT wired | **FLAG_ONLY** (alias) |
| 7 | `--consistency-weight` | 1.0 | Wf-380-A1 #7 `consistency_regularization_wf380.consistency_loss_fn` | ✅ 92 LOC + 4 tests | ❌ NO (only pre-existing `acceleration.use_shortcut_consistency` from Wf-348-C1 imported via accel_bundle) | ⚠️ PARTIAL — pre-existing `--shortcut-consistency-weight` (Wf-348-C1, default 0.1) IS wired via accel_bundle at line 2100; Wf-380 NEW flag `--consistency-weight` (default 1.0) is NOT wired | **FLAG_ONLY** (NEW flag) / **WIRED** (legacy Wf-348-C1 flag) |
| 8 | `--flowmol-unified` / `--no-flowmol-unified` | OFF | Wf-380-A1 #8 `flowmol_unified.UnifiedAtomBondFlowHead` | ✅ 117 LOC + 4 tests + LICENSE_GATED (CC-BY-NC) | ❌ NO | ❌ NO | **FLAG_ONLY** (LICENSE-GATED, USER_OK required) |
| 9 | `--top-p` | 0.9 | Wf-380-A3 #1 `diversity_sampling.nucleus_sample` + `diversity_aware_sampling_step` | ✅ 619 LOC + 3 tests | ❌ NO | ❌ NO | **FLAG_ONLY** |
| 10 | `--top-k` | 50 | Wf-380-A3 #1 same | ✅ 619 LOC | ❌ NO | ❌ NO | **FLAG_ONLY** |
| 11 | `--temperature` | 1.0 | Wf-380-A3 #1 same | ✅ 619 LOC | ❌ NO | ❌ NO | **FLAG_ONLY** |
| 12 | `--mmd-weight` | 0.1 | Wf-380-A3 #2 `diversity_sampling.mmd_loss` | ✅ 619 LOC + 3 tests | ❌ NO | ❌ NO | **FLAG_ONLY** |
| 13 | `--repulsion-weight` | 0.05 | Wf-380-A3 #3 `diversity_sampling.repulsion_loss` | ✅ 619 LOC | ❌ NO | ❌ NO | **FLAG_ONLY** |
| 14 | `--per-cohort-features` / `--no-per-cohort-features` | ON | Wf-380-A2 `wf380a2_m2_r2_per_cohort.select_features` | ✅ 540 LOC (`molmetal/scripts/wf380a2_m2_r2_per_cohort.py`) + 3 tests | ❌ NO (no `from molmetal.scripts.wf380a2_m2_r2_per_cohort import select_features` in either trainer) | ❌ NO | **FLAG_ONLY** |

### §1.2 DMPNN trainer flags (3 NEW per Wf-380-B1)

| # | Flag | Default | Wf-380-A SCAFFOLD source | SCAFFOLD module exists | SCAFFOLD imported by trainer? | SCAFFOLD called in main loop? | Verdict |
|--:|---|:---:|---|---|:---:|:---:|---|
| 1 | `--per-cohort-features` / `--no-per-cohort-features` | ON | Wf-380-A2 `wf380a2_m2_r2_per_cohort.select_features` | ✅ 540 LOC + 3 tests | ❌ NO | ❌ NO | **FLAG_ONLY** |
| 2 | `--triton-ot` / `--no-triton-ot` | ON | Wf-353-A1 / Wf-380-A1 #5 `triton_mini_batch_ot.triton_mini_batch_ot` | ✅ 110 LOC + 4 tests | ❌ NO (only `_reference.fused_relu_mlp` etc. imports for env-gated Triton kernels) | ⚠️ PARTIAL — pre-existing `--triton-minibatch-ot` (Wf-353-A1) is checked in the banner code (line 3100) but **NOT** in the training loop. The CFM trainer at line 1879-1880 wires the env-var dispatch; the DMPNN trainer does NOT | **FLAG_ONLY** (alias) / **FLAG_ONLY** (legacy Wf-353-A1 flag in DMPNN context) |
| 3 | `--use-ot-cfm` / `--no-use-ot-cfm` | ON | Wf-380-A1 #2 `optimal_transport_v2_solver.ot_solver` | ✅ 144 LOC + 4 tests | ❌ NO | ❌ NO | **FLAG_ONLY** |

### §1.3 Audit summary table

| Category | Count | % of 17 |
|---|--:|--:|
| **WIRED** (flag declared + SCAFFOLD called in main loop) | 0 | 0% |
| **PARTIAL** (Wf-380 alias FLAG_ONLY but pre-existing flag with same name IS wired via `accel_bundle` / env-var dispatch) | 5 (CFM #3, #4, #5, #6, #7) + 1 (DMPNN #2) = 6 | 35% |
| **FLAG_ONLY** (Wf-380 NEW flag declared + SCAFFOLD module exists + SCAFFOLD NOT imported + NOT called) | 17 (all 17) | **100%** |
| **BROKEN** (flag declared + SCAFFOLD called but argument mismatch / missing helper) | 0 | 0% |

**Honest framing**: 17/17 Wf-380 NEW flags are FLAG_ONLY. None of the Wf-380 SCAFFOLD modules (`edm_preconditioning.preconditioned_loss`, `stable_vm.stable_vm_loss`, `so3_average.so3_average_loss`, `triton_mini_batch_ot.triton_mini_batch_ot`, `shortcut_distillation_wf380.shortcut_distillation_loss`, `consistency_regularization_wf380.consistency_loss_fn`, `flowmol_unified.UnifiedAtomBondFlowHead`, `log_normal_t.log_normal_t_sampling`, `lr_warmup.LinearWarmupLR`, `diversity_sampling.{nucleus_sample,mmd_loss,repulsion_loss,diversity_aware_sampling_step}`, `optimal_transport_v2_solver.ot_solver`) are imported by either trainer.

---

## §2 SCAFFOLD module integration completeness

### §2.1 SCAFFOLD modules SHIPPED at Wf-380-A1..A7 (10 CFM + 2 misc = 12 modules)

Per `TODO/completed/380a1_deep_integration_scaffolds.md` §1 + Wf-380-A3 + Wf-380-A2:

| Module | LOC | Tests | Imported by CFM trainer? | Imported by DMPNN trainer? |
|---|--:|--:|:---:|:---:|
| `edm_preconditioning.py` (Wf-380-A1 #1) | 124 | 4/4 PASS | ❌ NO | ❌ NO |
| `optimal_transport_v2_solver.py` (Wf-380-A1 #2) | 144 | 4/4 PASS | ❌ NO | ❌ NO |
| `stable_vm.py` (Wf-380-A1 #3) | 95 | 4/4 PASS | ❌ NO | ❌ NO |
| `so3_average.py` (Wf-380-A1 #4) | 129 | 4/4 PASS | ❌ NO | ❌ NO |
| `triton_mini_batch_ot.py` (Wf-380-A1 #5) | 110 | 4/4 PASS | ❌ NO | ❌ NO |
| `shortcut_distillation_wf380.py` (Wf-380-A1 #6) | 121 | 4/4 PASS | ❌ NO | ❌ NO |
| `consistency_regularization_wf380.py` (Wf-380-A1 #7) | 92 | 4/4 PASS | ❌ NO | ❌ NO |
| `flowmol_unified.py` (Wf-380-A1 #8, CC-BY-NC LICENSE-GATED) | 117 | 4/4 PASS | ❌ NO | ❌ NO |
| `log_normal_t.py` (Wf-380-A1 #9) | 59 | 4/4 PASS | ❌ NO | ❌ NO |
| `lr_warmup.py` (Wf-380-A1 #10) | 64 | 5/5 PASS | ❌ NO (pre-existing `acceleration.LinearWarmupLRScheduler` IS imported at line 317) | ❌ NO |
| `diversity_sampling.py` (Wf-380-A3; nucleus_sample + mmd_loss + repulsion_loss + diversity_aware_sampling_step) | 619 | 3/3 PASS | ❌ NO | ❌ NO |
| `wf380a2_m2_r2_per_cohort.py` (Wf-380-A2; `select_features` + 3-tier matrix) | 540 (script) + 125 (YAML) | 3/3 PASS | ❌ NO (no `from molmetal.scripts.wf380a2_m2_r2_per_cohort import select_features`) | ❌ NO |
| **TOTAL** | **2214** | **45/45 PASS** | **0/12 imported** | **0/12 imported** |

### §2.2 Pre-existing trainer imports (NOT Wf-380)

CFM trainer (line 314-353) imports:
- `molmetal.adapters.flow_matching_lipman.acceleration` (Wf-340-I1 / Wf-348-C1): `CFMAccelerationBundle`, `ExponentialMovingAverage`, `LinearWarmupLRScheduler`, `build_acceleration_bundle`
- `molmetal.adapters.flow_matching_lipman.optimal_transport_v2` (Wf-348-C1): `exact_minibatch_ot_coupling`
- `molmetal.adapters.flow_matching_lipman.so3_averaged_fm` (Wf-348-C1): `so3_averaged_fm_loss`

DMPNN trainer (line 832 + 1148 + 1188 + 1339 + 1376) imports:
- `molmetal.adapters.flow_matching_lipman._reference.fused_relu_mlp` (Wf-333-T2)
- `molmetal.adapters.flow_matching_lipman.wasserstein_w1` (Wf-358-B1)
- `molmetal.adapters.flow_matching_lipman._reference.fused_wasserstein_sliced` (Wf-333-T2)

**0 of these 12 Wf-380 SCAFFOLD modules are imported by either trainer.** The pre-existing imports cover 6 of 17 Wf-380 flags via legacy `--use-*` flag names (with same-name alias semantics), but none via the Wf-380 NEW alias names.

### §2.3 Banner code (Wf-380-B1) — the only place Wf-380 flags are read

Both trainers have a 1-line banner (`[wf380-b1] Wf-380-A1..A7 SCAFFOLD modules active: ...`) that reads the Wf-380 flag values via `getattr(args, ...)` and prints them. This is OBSERVABILITY ONLY — no side effect on the loss / sampling / OT solver dispatch.

- CFM trainer banner: lines 1748-1790 (43 LOC)
- DMPNN trainer banner: lines 3091-3111 (21 LOC)

---

## §3 Wf-381 wire-in scope estimate

### §3.1 Per-flag wire-in LOC + integration complexity

| Flag | Wire-in target | LOC est. | Complexity |
|---|---|--:|---|
| `--use-ot-cfm` | `train_one_epoch`: replace stochastic coupling with `ot_solver` cost + reorder batch | ~20 | medium |
| `--stable-vm` | `train_one_epoch`: replace `criterion(pred, target)` with `stable_vm_loss` | ~10 | low |
| `--so3-average` | Already wired via `--use-so3-averaged-fm`; Wf-381 = alias unify | ~5 | trivial |
| `--log-normal-t` | Already wired via `--use-log-normal-t`; Wf-381 = alias unify | ~5 | trivial |
| `--triton-ot` | Already wired via env-var dispatch; Wf-381 = alias unify + dispatch in DMPNN | ~10 | trivial |
| `--shortcut-distill` | `train_one_epoch`: add teacher-student pair when `args.shortcut_distill` | ~25 | medium |
| `--consistency-weight` | Already wired via `--shortcut-consistency-weight`; Wf-381 = alias unify (care: default 1.0 vs 0.1) | ~5 | low (semantic shift) |
| `--flowmol-unified` | `model.head.forward` swap to `UnifiedAtomBondFlowHead` (LICENSE-GATED) | ~30 | high (LICENSE-GATED, USER_OK required per AGENTS.md §5.1) |
| `--top-p` `--top-k` `--temperature` | Sampling step in decode / evaluate loop (not `train_one_epoch`) | ~25 | medium |
| `--mmd-weight` `--repulsion-weight` | `train_one_epoch`: add MMD + repulsion regularization to total loss | ~20 | low |
| `--per-cohort-features` (CFM) | Pre-build: `select_features(cohort_size)` returns config dict (no per-epoch changes needed) | ~15 | low |
| `--per-cohort-features` (DMPNN) | Pre-build: `select_features(cohort_size)` returns config dict (no per-epoch changes needed) | ~15 | low |
| `--triton-ot` (DMPNN) | `train_one_epoch`: env-var dispatch (`TRITON_USE_FUSED=1` + `TRITON_MINIBATCH_OT=1`) | ~10 | low |
| `--use-ot-cfm` (DMPNN) | `train_one_epoch`: cohort reorder via `ot_solver` | ~20 | medium |

**Total LOC est. ~215 LOC** across both trainers (CFM ~150 LOC + DMPNN ~65 LOC).

### §3.2 Wf-381 deliverable decomposition

Per task brief §3.3 (sample 5-scaffold wire-in pattern from Wf-353-A1..A2 — the ONLY wave where SCAFFOLD + wire-in shipped together):

| Sub-flow | Deliverable | LOC est. | Test files | MEASURED vs PROJECTED |
|---|---|--:|---|---|
| Wf-381-A1 | `--use-ot-cfm` + `--triton-ot` (CFM) wire-in + OT cohort reorder + Triton env-var dispatch | ~30 LOC CFM + ~10 LOC DMPNN | 2 NEW test files (cohort reorder idempotence + Triton env-var gate) | MEASURED |
| Wf-381-A2 | `--stable-vm` + `--so3-average` + `--log-normal-t` wire-in (alias unify + loss swap) | ~30 LOC | 1 NEW test file (loss equivalence + numerical stability) | MEASURED |
| Wf-381-A3 | `--shortcut-distill` + `--consistency-weight` + EMA teacher wire-in | ~30 LOC | 1 NEW test file (teacher-student pair + EMA decay) | MEASURED |
| Wf-381-A4 | `--top-p` + `--top-k` + `--temperature` wire-in at sampling step (decode / evaluate_decode_ratio) | ~25 LOC | 1 NEW test file (nucleus sample equivalence vs hand-coded reference) | MEASURED |
| Wf-381-A5 | `--mmd-weight` + `--repulsion-weight` + diversity loss in `train_one_epoch` | ~20 LOC | 1 NEW test file (loss gradient flow + zero-weight identity) | MEASURED |
| Wf-381-A6 | `--per-cohort-features` (both trainers) + `--triton-ot` (DMPNN env-var dispatch) | ~25 LOC CFM + ~25 LOC DMPNN | 2 NEW test files (config selection + DMPNN OT dispatch) | MEASURED |
| Wf-381-A7 | `--flowmol-unified` LICENSE-GATED stub (USER_OK-gated; integration deferred to user OK) | ~10 LOC + LICENSE assertion | 1 NEW test file (USER_OK gate) | PROJECTED (LICENSE-GATED) |
| Wf-381-B1 | INDEX + AGENTS + final synthesis + pre-flight verify | docs-only | 0 NEW tests | docs-only |

**Combined wall PROJECTED 60-90 min CPU-only** per task brief.

**Honest framing**: Wf-381 wire-in is **CPU-only docs+code+test+verify work** (no GPU launches). The MEASURED speedup / lift claims (PROJECTED per Wf-340-A1 EDM preconditioning 2-3x FID convergence + Wf-353-A1 Triton OT 1.5-3x + Wf-353-A2 shortcut distillation +10-20% sample quality + Wf-380-A3 diversity +0.02-0.035 m7 lift + Wf-380-A2 per-cohort +0.05-0.10 m2 R² lift + Wf-380-A1 #7 consistency +5-10% stability) require Blackwell sm_120 GPU retrain per AGENTS.md §5.1 + `kill_then_adapt` §0 — **NOT in scope for Wf-381 wire-in** (PROJECTED → MEASURED conversion is a separate GPU retrain wave).

### §3.3 Wf-381 alternative paths

If USER_OK-gated GPU launch is granted (per Wf-368-B1 §7.3 priority):
- **Wf-381-A8 GPU retrain** (~5 hr GPU on Blackwell): 5-seed × 30-epoch A/B retrain of Wf-381-A1..A6 wire-in → MEASURED speedup vs Wf-340 baseline → convert PROJECTED → MEASURED claims in Wf-380-B1 honest framing.
- **Wf-381-A9 paper §6 update** (USER_OK-gated per TODO-84 + AGENTS.md §5.3): update honest framing #1 with MEASURED speedup numbers.

---

## §4 Cross-references

- `TODO/completed/380b1_index_sync.md` — Wf-380-B1 INDEX + AGENTS sync (the SHIP that this audit re-verifies)
- `TODO/completed/380a1_deep_integration_scaffolds.md` — 10 CFM SCAFFOLD modules (~1055 LOC)
- `TODO/completed/380a2_per_cohort_features.md` — per-cohort feature engineering SCAFFOLD (`select_features`)
- `TODO/completed/380a3_diversity_sampling.md` — diversity sampling SCAFFOLD (nucleus + MMD + repulsion)
- `TODO/completed/380{a4,a5,a6,a7}_*.md` — Wf-380-A4..A7 Vina tuning + multi-template MM-GBSA + SA penalty + m6_kappa finalize
- `TODO/completed/380_final_synthesis.md` — Wf-380 final synthesis v25
- `molmetal/scripts/train_cfm_yuelbond_wasserstein.py` — 14 NEW CFM flags at parse_args lines 1490-1740 + banner lines 1748-1790
- `molmetal/scripts/train_dmpnn_multitask_wasserstein.py` — 3 NEW DMPNN flags at parse_args lines 3012-3083 + banner lines 3091-3111
- `molmetal/adapters/flow_matching_lipman/edm_preconditioning.py` (124 LOC) + 9 other Wf-380 SCAFFOLD modules
- `molmetal/scripts/wf380a2_m2_r2_per_cohort.py` (540 LOC) — `select_features(cohort_size)`
- `AGENTS.md` §5.1 (standing GPU grant) + §5.2 (honest framing) + §5.3 (no paper work) + §5.4 (Tier-2 markers UNCHANGED) re-confirmed

---

## §5 Honest framing per party-twenty-spirit §3.5

1. **All 17 Wf-380 NEW CLI flags are FLAG_ONLY** — declared in `parse_args` + printed in the Wf-380 banner code + ZERO loss / sampling / dispatch wire-in to the live training computation. SCAFFOLD-only integration is documented at `TODO/completed/380b1_index_sync.md` §5 honest framing caveat #1 + Wf-380-B1 banner code (lines 1748-1797 CFM + 3090-3115 DMPNN).
2. **5 CFM flags have PARTIAL WIRE status via pre-existing alias flags** — `--so3-average` ↔ `--use-so3-averaged-fm` (Wf-348-C1 wired at line 877), `--log-normal-t` ↔ `--use-log-normal-t` (Wf-348-C1 wired via accel_bundle line 2093), `--triton-ot` ↔ `--triton-minibatch-ot` (Wf-353-A1 wired via env-var dispatch line 1879-1880 CFM only), `--shortcut-distill` ↔ `--use-shortcut-distillation` (Wf-353-A2 referenced via accel_bundle but NOT passed to `train_one_epoch`), `--consistency-weight` ↔ `--shortcut-consistency-weight` (Wf-348-C1 wired via accel_bundle line 2100 with default 0.1; Wf-380 NEW default 1.0 is a SEMANTIC SHIFT not yet propagated). DMPNN `--triton-ot` is FLAG_ONLY (no env-var dispatch in DMPNN trainer; Wf-353-A1 dispatch was CFM-only).
3. **Same SCAFFOLD-only pattern has recurred 3 times**: Wf-340 → Wf-348-C1 (acceleration bundle wire-in); Wf-353 → Wf-353-A1/A2 (Triton OT + shortcut distillation — the ONLY wave where SCAFFOLD + wire-in shipped together); Wf-380 → Wf-381+ (deferred). Per Wf-385-A1 user voice verbatim *"如果是同一个问题一直没修好的话得做调研+review尝试更深层的修复"*, Wf-385-A1 has researched the pattern: closure path is **Wf-381 sub-flow A1..A7 wire-in wave** (~215 LOC across 17 sites, CPU-only docs+code+test+verify; no GPU launches).
4. **PROJECTED speedup / lift claims remain PROJECTED, NOT MEASURED** — Wf-340-A1 EDM preconditioning 2-3x FID convergence + Wf-353-A1 Triton OT 1.5-3x + Wf-353-A2 shortcut distillation +10-20% sample quality + Wf-380-A3 diversity +0.02-0.035 m7 lift + Wf-380-A2 per-cohort +0.05-0.10 m2 R² lift + Wf-380-A1 #7 consistency +5-10% stability. MEASURED requires GPU retrain per AGENTS.md §5.1 + `kill_then_adapt` §0. NOT in scope for Wf-381 wire-in (separate USER_OK-gated GPU wave).
5. **Pre-existing 2 NEW FAILs at Wf-385-A1 timestamp preserved per Wf-375-B1 §22.68 baseline** (`p3_loi_send_ready` Tier-2 reclassification cascade + `p3_wf343c4_anchor_freshness_24h` wf269c_report.json STALE) — NOT Wf-385-A1-introduced; both deferred to Wf-386+ follow-up.
6. **Pre-existing 3 SKIPs PRESERVED (BY-DESIGN)** (`p3_wf343c4_yuelbond_decoder_env` per Wf-345-A2 + `p3_wf343c4_wave_budget_remaining_nonneg` per Wf-262 §7.1 with −183.04h overrun + `data_2_1_6_bindingdb_external_honest_negative` external 1.07M dependency) — NOT Wf-385-A1-introduced.
7. **m6_kappa + m2 R² structural gates HONEST_NEGATIVE FINAL preserved** — per Wf-377-A1 deep-dive-research + Wf-369-A2 verdict + Wf-367-B1 multi-stream ensemble + Wf-377-A2 5-alternative-split diagnostic. Wf-385-A1 does NOT propose to re-open these gates via Wf-381 wire-in (the gates are statistical; Wf-381 is dispatcher + loss modification only).
8. **Wf-379-A1 m1_auc 5-seed IN-FLIGHT — result pending** — Rh seed 3 RUNNING on GPU 0 Blackwell at Wf-379-B1 timestamp; 6 (metal, seed) pairs queued after Rh seed 3 = Re seeds 0/1/2/3/4 + Rh seed 4; Krasnov 0.86 gate target; full multi-pair result deferred to Wf-380+.
9. **Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL** — R20 §5 tmQM + PlatinAI joint retrain wrapper blocked on trainer CHOICES CONSTRAINT VIOLATION (trainer `--metals` restricted to 5 metals; wrapper wants 11 metals); alternative trainer missing `tmqm_pretrained_encoder.pt` checkpoint; closure path = trainer extension OR alternative encoder bootstrap OR re-scope R20 §5 to 5-metal subset.
10. **Wetlab items DEPRECATED per AGENTS.md §5.4 + §10** (4 LOIs + 3 SOPs + R20-P6 + Wf-74 v2 + LAMBDA_ROADMAP §3.2 R22-P1 + LAMBDA_ROADMAP §3.3 R23 all DEPRECATED / re-scoped to in-silico).

---

## §6 User voice answer (per Wf-385-A1 audit)

**User voice verbatim 2026-09-30**: *"什么意思，这个workflow解决了我们给他定的要求吗，那些问题彻底解决了吗，如果不是，启动新的ultracode做follow up,如果是同一个问题一直没修好的话得做调研+review尝试更深层的修复"* — relayed from Wf-380-B1 row.

**Wf-385-A1 answer**: PARTIAL — **Wf-380-B1 SHIPPED 8/8 deliverables** (10 SCAFFOLD modules + 14 CFM CLI flags + 3 DMPNN CLI flags + 1-line banner per trainer + INDEX/AGENTS/synthesis). But the **14+3=17 NEW CLI flags are FLAG_ONLY** — none has an actual SCAFFOLD call in the main training loop; none of the 12 Wf-380 SCAFFOLD modules are imported by either trainer; the Wf-380 banner code is observability only. This is the third time this SCAFFOLD-only pattern has recurred (Wf-340, Wf-353, Wf-380); per user verbatim *"如果是同一个问题一直没修好的话得做调研+review尝试更深层的修复"* the deeper-fix closure path is **Wf-381 wire-in wave** (~215 LOC across 17 sites, CPU-only docs+code+test+verify per task brief §3.3). NOT彻底解决: 17/17 flags FLAG_ONLY + PROJECTED speedup/lift NOT MEASURED end-to-end on Blackwell + 5 PARTIAL flags via pre-existing aliases + Wf-379-A1 GPU m1_auc IN-FLIGHT + Wf-379-A2 SCAFFOLD_NOT_FUNCTIONAL + 2 pre-flight FAILs preserved per Wf-375-B1 §22.68 baseline + 6 forward-looking USER_OK items (~52.9 hr GPU + 13 hr CPU + 30 min install per Wf-368-B1 §0). NEW ultracode Wf-381 IS the recommended follow-up per this audit §3 with 7 sub-flows A1..A7 (~215 LOC + 7 NEW test files) + B1 docs-only sync (~10 min CPU) — combined wall PROJECTED 60-90 min CPU-only, no GPU launches.

---

**END OF Wf-385-A1 AUDIT** — 17/17 Wf-380 NEW CLI flags are FLAG_ONLY; Wf-381 sub-flow A1..A7 wire-in wave is the deeper-fix closure path per user verbatim *"调研+review尝试更深层的修复"*.
